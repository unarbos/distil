#!/usr/bin/env python3
"""
King chat server bootstrapper — runs on the chat-bench pod.

Starts an OpenAI-compatible vLLM server for the current king model so
chat.arbos.life can talk to it via the SSH tunnel on port 8100.

Two supported student families, selected automatically from config.json:

  (A) Qwen3.5 / Qwen3.6 family (``Qwen3_5ForCausalLM`` or
      ``Qwen3_5ForConditionalGeneration``). Legacy path — was the only
      supported family before the Kimi K2.6 teacher swap. Requires the
      Qwen3_5ForConditionalGeneration VL wrapper and graft-in of base
      Qwen3.5-4B visual weights so vLLM's weight loader sees a complete
      checkpoint. Kept for backward compatibility so the current Qwen
      king keeps serving while miners migrate to Kimi-family arch.

  (B) Kimi K2.6 family — text-only ``DeepseekV3ForCausalLM`` (inner text
      model of the Kimi K2.6 wrapper) or the full
      ``KimiK25ForConditionalGeneration`` wrapper. vLLM 0.19+ supports
      both natively with ``--trust-remote-code``; no config rewrite or
      weight grafting needed. We just download-and-serve.

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
# Legacy Qwen wrapper base — only used if the downloaded model is a
# Qwen3.5/3.6 family student that needs visual-weight grafting.
BASE_MODEL = "Qwen/Qwen3.5-4B"
SERVED_NAME = "sn97-king"


def _detect_arch_family() -> str:
    """Inspect the downloaded config.json and return the arch family.

    Returns one of ``"qwen35"``, ``"kimi_k2"`` (text-only), ``"kimi_k25"``
    (vision wrapper), or ``"unknown"``.
    """
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return "unknown"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return "unknown"
    archs = cfg.get("architectures") or []
    mt = cfg.get("model_type", "")
    if "KimiK25ForConditionalGeneration" in archs or mt == "kimi_k25":
        return "kimi_k25"
    if "DeepseekV3ForCausalLM" in archs and mt in ("kimi_k2", "deepseek_v3"):
        return "kimi_k2"
    if mt == "kimi_k2" or mt == "deepseek_v3":
        return "kimi_k2"
    if any(a.startswith("Qwen3_5") for a in archs) or mt.startswith("qwen3_5"):
        return "qwen35"
    return "unknown"


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
    # fits a 4B-class model + KV cache and tune via env if a future king is
    # bigger or the eval pod gets a smaller card.
    gpu_util = os.environ.get("CHAT_VLLM_GPU_UTIL", "0.30")
    max_model_len = os.environ.get("CHAT_VLLM_MAX_MODEL_LEN", "32768")
    family = _detect_arch_family()
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
    ]
    if family == "qwen35":
        # Qwen 3.5 / 3.6 emit ``<tool_call><function=name><parameter=k>v
        # </parameter></function></tool_call>`` XML — the ``qwen3_xml``
        # parser ships with vLLM 0.19+ and matches the family natively.
        cmd += [
            "--tool-call-parser", "qwen3_xml",
            "--reasoning-parser", "qwen3",
            "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
            "--skip-mm-profiling",
        ]
    elif family == "kimi_k25":
        # Kimi K2.5/K2.6 vision wrapper — disable vision path for text chat,
        # use the Kimi-native tool-call tokens (``<|tool_calls_section_begin|>``
        # ... ``<|tool_call_begin|>``). vLLM doesn't ship a dedicated
        # kimi tool parser yet, so leave ``--tool-call-parser`` off and let
        # the model emit raw tokens; clients that parse structured tool
        # calls on the Kimi chat template will still work.
        cmd += [
            "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
            "--skip-mm-profiling",
        ]
    elif family == "kimi_k2":
        # Text-only DeepSeek V3 inner of Kimi K2 — vanilla causal LM path.
        # No tool parser flag so clients parse Kimi tool tokens directly.
        pass
    else:
        # Unknown architecture — pass no family-specific flags; vLLM may
        # succeed on simple architectures (Llama-family) without them.
        log(f"warning: unknown architecture family, falling back to minimal vLLM args")
    log(f"exec vLLM (family={family}, gpu_util={gpu_util}, max_model_len={max_model_len})")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    rev_suffix = f"@{MODEL_REVISION}" if MODEL_REVISION else ""
    log(f"bootstrapping model={MODEL_NAME}{rev_suffix} port={PORT}")
    download_model()
    family = _detect_arch_family()
    log(f"detected arch family: {family}")
    if family == "qwen35":
        # Legacy Qwen students: run the VL-wrapper dance so vLLM sees the
        # full config + visual weights.
        patch_config_and_tokenizer()
        inject_visual_weights()
    else:
        # Kimi-family and unknown: download-and-serve. vLLM loads the
        # model's own config.json / tokenizer directly.
        log(f"skipping Qwen-specific patch + visual-weight grafting for family={family}")
    exec_vllm()
