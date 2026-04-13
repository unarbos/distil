#!/usr/bin/env python3
"""
King chat server bootstrapper.

Runs on the chat pod and turns a miner checkpoint into a vLLM-loadable model by:
1. downloading the model into /root/king-model
2. patching config/tokenizer for vLLM-compatible Qwen3.5 wrapper loading
3. grafting base-model visual weights into a small extra shard
4. exec'ing vLLM's OpenAI-compatible API server on port 8100

This keeps the existing /root/chat_server.py entrypoint stable for the validator/API,
while swapping the backend from HF transformers to vLLM.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "aceini/q-dist"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100
MODEL_DIR = Path("/root/king-model")
BASE_MODEL = "Qwen/Qwen3.5-4B"


def log(msg: str):
    print(f"[chat-vllm] {msg}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kwargs)


def download_model():
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Uses HF cache, so repeated downloads mostly relink/copy from cache.
    run(["huggingface-cli", "download", MODEL_NAME, "--local-dir", str(MODEL_DIR)])


def patch_config_and_tokenizer():
    from huggingface_hub import hf_hub_download

    config_path = MODEL_DIR / "config.json"
    tokenizer_path = MODEL_DIR / "tokenizer_config.json"

    with open(config_path) as f:
        text_config = json.load(f)

    # Patch text-only miner config into the wrapper architecture vLLM registers.
    if text_config.get("model_type") == "qwen3_5_text" and text_config.get("architectures") == ["Qwen3_5ForCausalLM"]:
        ref_path = hf_hub_download(BASE_MODEL, "config.json")
        with open(ref_path) as f:
            ref = json.load(f)

        wrapper = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": text_config,
            "torch_dtype": text_config.get("torch_dtype", "bfloat16"),
            "transformers_version": text_config.get("transformers_version", ref.get("transformers_version", "5.0.0")),
        }
        for key in ("vision_config", "image_token_id", "video_token_id"):
            if key in ref:
                wrapper[key] = ref[key]
        wrapper["text_config"]["model_type"] = "qwen3_5_text"
        wrapper["text_config"].pop("architectures", None)

        with open(config_path, "w") as f:
            json.dump(wrapper, f, indent=2)
        log("patched config → Qwen3_5ForConditionalGeneration wrapper")

    # Patch bogus miner tokenizer config to the base tokenizer class.
    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tok = json.load(f)
        tok["tokenizer_class"] = "Qwen2Tokenizer"
        tok.pop("auto_map", None)
        with open(tokenizer_path, "w") as f:
            json.dump(tok, f, indent=2)
        log("patched tokenizer_config.json")

    # Copy base tokenizer files (miner tokenizer.json can be corrupt/incompatible)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
            log(f"copied {fn} from base model")
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")

    # Copy base preprocessors expected by the wrapper.
    for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")


def inject_visual_weights():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_shard = hf_hub_download(BASE_MODEL, "model.safetensors-00002-of-00002.safetensors")
    visual_out = MODEL_DIR / "visual.safetensors"
    model_file = MODEL_DIR / "model.safetensors"
    index_file = MODEL_DIR / "model.safetensors.index.json"

    if not model_file.exists():
        raise RuntimeError(f"missing model weights: {model_file}")

    visual_tensors = {}
    with safe_open(base_shard, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.visual."):
                new_key = "visual." + key[len("model.visual."):]
                visual_tensors[new_key] = f.get_tensor(key)

    save_file(visual_tensors, str(visual_out))
    log(f"wrote visual shard with {len(visual_tensors)} tensors")

    index = {"metadata": {}, "weight_map": {}}
    with safe_open(str(model_file), framework="pt") as f:
        for key in f.keys():
            index["weight_map"][key] = "model.safetensors"
    for key in visual_tensors.keys():
        index["weight_map"][key] = "visual.safetensors"
    index["metadata"]["total_size"] = model_file.stat().st_size + visual_out.stat().st_size
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)
    log(f"wrote sharded index with {len(index['weight_map'])} keys")


def write_health():
    payload = {
        "status": "starting",
        "backend": "vllm",
        "model": MODEL_NAME,
        "port": PORT,
        "ts": time.time(),
    }
    with open("/root/chat_health.json", "w") as f:
        json.dump(payload, f)
    with open("/root/model_name.txt", "w") as f:
        f.write(MODEL_NAME)


def exec_vllm():
    write_health()
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_DIR),
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--trust-remote-code",
        "--served-model-name", "sn97-king",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
    ]
    log("exec vLLM")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    log(f"bootstrapping model={MODEL_NAME} port={PORT}")
    download_model()
    patch_config_and_tokenizer()
    inject_visual_weights()
    exec_vllm()
