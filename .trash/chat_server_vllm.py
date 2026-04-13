#!/usr/bin/env python3
"""
vLLM-based chat server for the king model. Runs on GPU pod, port 8100.

Uses vLLM's AsyncLLMEngine for 500-1000+ tok/s inference.
Auto-patches model config from Qwen3_5ForCausalLM → Qwen3_5ForConditionalGeneration
so vLLM can load it.

Usage:
    python3 chat_server_vllm.py <model_name> [port]
"""
import json
import sys
import os
import time
import asyncio
import uuid
import tempfile
import shutil
from pathlib import Path

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "aceini/q-dist"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

print(f"[chat-vllm] Starting with model: {MODEL_NAME}, port: {PORT}", flush=True)


def patch_model_config(model_name: str) -> str:
    """
    Download model config and check if it needs patching for vLLM compatibility.
    If model uses Qwen3_5ForCausalLM (text-only), convert to Qwen3_5ForConditionalGeneration
    (multimodal wrapper) which vLLM supports.
    
    Returns the model name/path to use (original if no patch needed, local path if patched).
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    
    try:
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"[chat-vllm] Cannot download config for {model_name}: {e}", flush=True)
        return model_name
    
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")
    
    if "Qwen3_5ForConditionalGeneration" in architectures:
        print(f"[chat-vllm] Config already vLLM-compatible", flush=True)
        return model_name
    
    if model_type != "qwen3_5_text" or "Qwen3_5ForCausalLM" not in architectures:
        print(f"[chat-vllm] Unknown architecture: {architectures}, trying as-is", flush=True)
        return model_name
    
    print(f"[chat-vllm] Patching config: {model_type}/Qwen3_5ForCausalLM → qwen3_5/Qwen3_5ForConditionalGeneration", flush=True)
    
    # Download the full model
    local_path = snapshot_download(model_name)
    
    # Read the text config (the flat config IS the text config)
    text_config = config.copy()
    
    # Build the wrapper config
    # Get reference config structure from base model
    try:
        ref_config_path = hf_hub_download("Qwen/Qwen3.5-4B", "config.json")
        with open(ref_config_path) as f:
            ref_config = json.load(f)
    except:
        ref_config = {}
    
    # Create wrapper config
    wrapper_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text_config,
        # Copy top-level fields that vLLM needs
        "torch_dtype": config.get("torch_dtype", "bfloat16"),
        "transformers_version": config.get("transformers_version", "5.0.0"),
    }
    
    # Copy any vision/processing config from reference (may not be needed for text-only)
    for key in ("vision_config", "processing_config", "vocab_size"):
        if key in ref_config and key not in wrapper_config:
            wrapper_config[key] = ref_config[key]
    
    # Also ensure text_config has the right model_type
    wrapper_config["text_config"]["model_type"] = "qwen3_5_text"
    
    # Write patched config
    patched_config_path = os.path.join(local_path, "config.json")
    with open(patched_config_path, "w") as f:
        json.dump(wrapper_config, f, indent=2)
    
    print(f"[chat-vllm] Config patched at {patched_config_path}", flush=True)
    return local_path


def start_vllm_server(model_path: str, port: int):
    """Start vLLM's OpenAI-compatible server."""
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server
    import uvloop
    
    # Use vLLM's built-in OpenAI-compatible server
    # This handles /v1/chat/completions, /v1/models, streaming, etc.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    
    args = [
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--trust-remote-code",
        "--served-model-name", MODEL_NAME,
        # Performance tuning for 4090
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",  # 4090 doesn't benefit from CUDA graphs as much
    ]
    
    print(f"[chat-vllm] Starting vLLM server: {' '.join(args)}", flush=True)
    
    # Use subprocess to run vllm serve
    import subprocess
    proc = subprocess.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server"] + args,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


def write_health_file(model_name: str, port: int):
    """Write health info for the API to check."""
    health = {
        "model": model_name,
        "port": port,
        "started": time.time(),
        "backend": "vllm",
    }
    with open("/root/chat_health.json", "w") as f:
        json.dump(health, f)
    with open("/root/model_name.txt", "w") as f:
        f.write(model_name)


if __name__ == "__main__":
    # Patch config if needed
    model_path = patch_model_config(MODEL_NAME)
    
    # Write health file
    write_health_file(MODEL_NAME, PORT)
    
    # Start vLLM server
    proc = start_vllm_server(model_path, PORT)
    
    print(f"[chat-vllm] Server PID: {proc.pid}", flush=True)
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
