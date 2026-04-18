"""
Chat-king pod management: SSH, vLLM chat server, benchmarks.
"""
import logging
import time

from scripts.validator.config import CHAT_POD_HOST, CHAT_POD_SSH_PORT, CHAT_POD_APP_PORT

logger = logging.getLogger("distillation.remote_validator")


def chat_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the chat-king pod via SSH."""
    import subprocess
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-p", str(CHAT_POD_SSH_PORT), f"root@{CHAT_POD_HOST}", cmd,
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception as e:
        logger.warning(f"Chat pod SSH failed: {e}")
        return ""


def restart_chat_server(model_name: str):
    """Kill old vLLM chat server and start with new king model."""
    logger.info(f"Restarting chat server with new king: {model_name}")
    try:
        chat_ssh("pkill -9 -f 'vllm.entrypoints.openai.api_server' || true", timeout=10)
        time.sleep(3)
        # Download model first so vLLM doesn't timeout during startup
        chat_ssh(
            f"python3 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_name}')\" 2>/dev/null || true",
            timeout=300,
        )
        # Start vLLM with same flags as current production setup
        chat_ssh(
            f"nohup python3 -m vllm.entrypoints.openai.api_server "
            f"--model '{model_name}' --served-model-name '{model_name}' "
            f"--port {CHAT_POD_APP_PORT} --max-model-len 8192 --dtype bfloat16 "
            f"--trust-remote-code --enforce-eager --reasoning-parser qwen3 "
            f"--enable-auto-tool-choice --tool-call-parser hermes "
            f'--skip-mm-profiling --limit-mm-per-prompt \'{{"image": 0}}\' '
            f"> /root/chat_vllm.log 2>&1 &",
            timeout=10,
        )
        logger.info("Chat vLLM server restart initiated")
    except Exception as e:
        logger.warning(f"Failed to restart chat server: {e}")


def trigger_benchmarks(model_name: str, king_uid: int):
    """Trigger auto-benchmarks on the eval pod for a new king model (background)."""
    logger.info(f"Triggering auto-benchmarks for UID {king_uid} ({model_name})")
    try:
        # Run in background — benchmarks take hours, don't block the validator
        chat_ssh(
            f"KING_UID={king_uid} MODEL='{model_name}' RESULTS_DIR=/root/benchmark_results "
            f"nohup /root/auto_benchmark.sh '{model_name}' {king_uid} > /root/benchmark.log 2>&1 &",
            timeout=10,
        )
        logger.info(f"Benchmark trigger sent for UID {king_uid}")
    except Exception as e:
        logger.warning(f"Failed to trigger benchmarks: {e}")


def ensure_chat_server_running(model_name: str):
    """Check if chat server is running with the right model; start/restart if needed."""
    try:
        stdout = chat_ssh(f"curl -fsS http://localhost:{CHAT_POD_APP_PORT}/v1/models || echo not_running", timeout=10)
        if "not_running" in stdout:
            logger.info(f"Chat server not running, starting with {model_name}")
            restart_chat_server(model_name)
        elif model_name not in stdout:
            logger.info(f"Chat server running wrong model, restarting with {model_name}")
            restart_chat_server(model_name)
    except Exception as e:
        logger.debug(f"Chat server check failed: {e}")
