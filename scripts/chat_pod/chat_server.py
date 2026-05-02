#!/usr/bin/env python3
"""
King chat server bootstrapper — runs on the chat-bench pod.

Starts an OpenAI-compatible vLLM server for the current king model so
chat.arbos.life can talk to it via the SSH tunnel on port 8100.

The subnet's miners publish models in a few different shapes:
  1. A flat text-only config.json with architectures=["Qwen3_5ForCausalLM"].
  2. A Qwen3_5ForConditionalGeneration wrapper with nested text_config but no
     top-level vision_config (e.g. text-only distillation students that reuse
     the base Qwen3.5-4B VL architecture without actually training visual
     layers — tom9491/distil-32 is this shape).
  3. A full VL model with both text_config and vision_config.

vLLM only registers Qwen3_5ForConditionalGeneration (the VL wrapper), so the
bootstrap *always* produces a wrapper on disk. When the miner didn't ship
visual weights, we graft them in from Qwen/Qwen3.5-4B (the undistilled base) so
vLLM's weight loader sees a complete checkpoint; the vision branch is unused
for text chat but the shapes have to match what the config declares.

Usage:
    python3 chat_server.py <hf_repo>[:revision] [port]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAW_MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
if ":" in RAW_MODEL:
    MODEL_NAME, MODEL_REVISION = RAW_MODEL.split(":", 1)
else:
    MODEL_NAME, MODEL_REVISION = RAW_MODEL, None
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

MODEL_DIR = Path("/root/king-model")
BASE_MODEL = "Qwen/Qwen3.5-4B"
SERVED_NAME = "sn97-king"


def log(msg: str):
    print(f"[chat-vllm] {msg}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kwargs)


def _hf_cli() -> str:
    """Return the HF CLI binary — `hf` replaced `huggingface-cli` in v1.10."""
    for candidate in ("hf", "huggingface-cli"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("no huggingface CLI found (need `hf` or `huggingface-cli`)")


def download_model():
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [_hf_cli(), "download", MODEL_NAME, "--local-dir", str(MODEL_DIR)]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    run(cmd)


def patch_config_and_tokenizer():
    """Normalize config.json to a Qwen3_5ForConditionalGeneration wrapper with
    text_config + vision_config + image/video token ids, so vLLM can resolve
    the architecture and so the visual weights we graft in line up.
    """
    from huggingface_hub import hf_hub_download

    config_path = MODEL_DIR / "config.json"
    tokenizer_path = MODEL_DIR / "tokenizer_config.json"

    with open(config_path) as f:
        cfg = json.load(f)

    ref_path = hf_hub_download(BASE_MODEL, "config.json")
    with open(ref_path) as f:
        ref = json.load(f)

    arch = cfg.get("architectures") or []

    if arch == ["Qwen3_5ForCausalLM"]:
        text_inner = dict(cfg)
        text_inner["architectures"] = ["Qwen3_5ForCausalLM"]
        text_inner["model_type"] = "qwen3_5_text"
        for k in (
            "vision_config", "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id",
            "tie_word_embeddings",
        ):
            text_inner.pop(k, None)
        cfg = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": text_inner,
            "torch_dtype": text_inner.get("torch_dtype", "bfloat16"),
            "transformers_version": ref.get("transformers_version"),
        }
        log("wrapped flat Qwen3_5ForCausalLM config → Qwen3_5ForConditionalGeneration")
    elif arch == ["Qwen3_5ForConditionalGeneration"]:
        if "text_config" not in cfg:
            raise RuntimeError(
                "Qwen3_5ForConditionalGeneration config is missing text_config"
            )
        text_inner = cfg["text_config"]
        text_inner.setdefault("architectures", ["Qwen3_5ForCausalLM"])
        text_inner.setdefault("model_type", "qwen3_5_text")
        log("using existing Qwen3_5ForConditionalGeneration wrapper")
    else:
        log(f"warning: unexpected architectures {arch!r}; leaving as-is")

    for key in (
        "vision_config", "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id",
    ):
        if key in ref:
            cfg[key] = ref[key]
    cfg.setdefault("tie_word_embeddings", ref.get("tie_word_embeddings", False))

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log("wrote patched config.json")

    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tok = json.load(f)
        tok["tokenizer_class"] = "Qwen2Tokenizer"
        tok.pop("auto_map", None)
        with open(tokenizer_path, "w") as f:
            json.dump(tok, f, indent=2)
        log("patched tokenizer_config.json")

    for fn in ("tokenizer.json", "tokenizer_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
            log(f"copied {fn} from base model")
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")

    for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")

    for fn in ("chat_template.jinja",):
        dst = MODEL_DIR / fn
        if dst.exists():
            continue
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, dst)
            log(f"copied {fn} from base model")
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")


def inject_visual_weights():
    """Graft base visual weights into the miner's shard set as visual.safetensors,
    and rebuild model.safetensors.index.json so vLLM finds every key.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_shard = hf_hub_download(
        BASE_MODEL, "model.safetensors-00002-of-00002.safetensors"
    )
    visual_out = MODEL_DIR / "visual.safetensors"
    index_file = MODEL_DIR / "model.safetensors.index.json"

    model_shards = sorted(MODEL_DIR.glob("model*.safetensors"))
    if not model_shards:
        raise RuntimeError(f"missing model weights under {MODEL_DIR}")

    visual_tensors = {}
    with safe_open(base_shard, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.visual."):
                new_key = "visual." + key[len("model.visual."):]
                visual_tensors[new_key] = f.get_tensor(key)

    save_file(visual_tensors, str(visual_out))
    log(f"wrote visual shard with {len(visual_tensors)} tensors")

    weight_map = {}
    total_size = visual_out.stat().st_size
    for shard in model_shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name
        total_size += shard.stat().st_size
    for key in visual_tensors.keys():
        weight_map[key] = "visual.safetensors"

    with open(index_file, "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f, indent=2,
        )
    log(f"wrote sharded index with {len(weight_map)} keys across {len(model_shards)+1} shards")


def write_health(status: str = "starting"):
    payload = {
        "status": status,
        "backend": "vllm",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "port": PORT,
        "ts": time.time(),
    }
    with open("/root/chat_health.json", "w") as f:
        json.dump(payload, f)
    with open("/root/model_name.txt", "w") as f:
        f.write(MODEL_NAME)


def exec_vllm():
    write_health()
    # Chat-king coexists with the validator's eval workload on the same GPU
    # most of the time. ``pod_eval.py`` claims ~0.90 of the H200 during
    # rounds, so a 0.90 chat slice would OOM the second vLLM to come up
    # (whichever loses the race). Default to a slim slice that comfortably
    # fits a 4B model + KV cache and tune via env if a future king is
    # bigger or the eval pod gets a smaller card.
    #
    # 2026-05-02 (v30.5 patch): max_model_len 8192 → 32768. User-reported
    # truncation: long math questions (multi-step word problems,
    # Fermi-style "how many jelly beans fill the ocean") were producing
    # answers that hit ``finish_reason=length`` mid-final-paragraph
    # because (prompt ~600 tokens) + (long answer ~7500 tokens) clipped
    # the 8192 cap. The Qwen3.5 king config exposes
    # ``max_position_embeddings=262144`` and 24 of 32 layers are
    # Mamba-style linear-attention (constant KV-cache cost regardless of
    # sequence length); the dominant memory cost comes from the 8 full-
    # attention layers. 32K context fits comfortably under
    # ``gpu_memory_utilization=0.30``. If a future bigger king OOMs at
    # 32K, drop CHAT_VLLM_MAX_MODEL_LEN via env and bump util.
    gpu_util = os.environ.get("CHAT_VLLM_GPU_UTIL", "0.30")
    max_model_len = os.environ.get("CHAT_VLLM_MAX_MODEL_LEN", "32768")
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_DIR),
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--served-model-name", SERVED_NAME,
        "--gpu-memory-utilization", str(gpu_util),
        "--enforce-eager",
        "--enable-auto-tool-choice",
        # 2026-05-02 (v30.5): switched ``hermes`` → ``qwen3_xml``. The
        # Qwen 3.5 / 3.6 family doesn't emit Hermes JSON tool calls;
        # it emits ``<tool_call><function=name><parameter=k>v
        # </parameter></function></tool_call>`` XML, which the
        # Hermes parser can't extract. Result: a Flue / OpenAI Agents
        # SDK / Vercel AI SDK client sees the XML as plaintext content
        # and ``message.tool_calls`` comes back null. The
        # ``qwen3_xml`` parser registered in vllm.tool_parsers.__init__
        # ships with vLLM 0.19.x and matches the Qwen3-family format
        # natively.
        "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--skip-mm-profiling",
    ]
    log(f"exec vLLM (gpu_util={gpu_util}, max_model_len={max_model_len})")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    rev_suffix = f"@{MODEL_REVISION}" if MODEL_REVISION else ""
    log(f"bootstrapping model={MODEL_NAME}{rev_suffix} port={PORT}")
    download_model()
    patch_config_and_tokenizer()
    inject_visual_weights()
    exec_vllm()
