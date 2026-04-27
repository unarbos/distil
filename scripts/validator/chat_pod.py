"""
Chat-king pod management: SSH, vLLM chat server, benchmarks.
"""
import logging
import os
import subprocess
import time
from pathlib import Path

from scripts.validator.config import CHAT_POD_HOST, CHAT_POD_SSH_PORT, CHAT_POD_APP_PORT
from scripts.validator.chat_pod_admin import (
    probe as _probe_chat_pod,
    write_chat_pod_state as _write_chat_pod_state,
)

logger = logging.getLogger("distillation.remote_validator")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_SERVER_SRC = REPO_ROOT / "scripts" / "chat_pod" / "chat_server.py"

# The benchmark runner lives on a separate eval pod: it opens its own SSH
# tunnel to the chat pod's vLLM and shells the affine-benchmark runner. The
# validator just fires-and-forgets at trigger time, and sync_benchmarks.sh
# pulls uid_*_summary.json back from this pod on a timer.
EVAL_POD_HOST = os.environ.get("DISTIL_EVAL_POD_HOST", "213.13.7.110")
EVAL_POD_SSH_PORT = int(os.environ.get("DISTIL_EVAL_POD_SSH_PORT", "6039"))


def _ssh(host: str, port: int, cmd: str, timeout: int = 30, label: str = "pod") -> str:
    if not host:
        logger.warning(f"{label} SSH skipped: host is not configured")
        return ""
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-p", str(port), f"root@{host}", cmd,
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception as e:
        logger.warning(f"{label} SSH failed: {e}")
        return ""


def chat_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the chat-king pod via SSH."""
    return _ssh(CHAT_POD_HOST, CHAT_POD_SSH_PORT, cmd, timeout=timeout, label="Chat pod")


def eval_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the eval pod (separate from the chat pod)."""
    return _ssh(EVAL_POD_HOST, EVAL_POD_SSH_PORT, cmd, timeout=timeout, label="Eval pod")


def _sync_chat_server():
    """Copy the canonical chat_server.py to the pod.

    The miner checkpoints come in multiple config shapes (flat text-only vs
    Qwen3_5ForConditionalGeneration wrapper), so the bootstrapper has to
    normalize config.json + graft base visual weights before vLLM will boot.
    Keeping the canonical script in the repo and scp'ing it every restart
    means pod state never drifts from what's version-controlled here.
    """
    if not CHAT_POD_HOST:
        logger.warning("Chat pod host is not configured; cannot sync chat_server.py")
        return False
    if not CHAT_SERVER_SRC.exists():
        logger.warning(f"chat_server.py source missing: {CHAT_SERVER_SRC}")
        return False
    scp_cmd = [
        "scp", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-P", str(CHAT_POD_SSH_PORT), str(CHAT_SERVER_SRC),
        f"root@{CHAT_POD_HOST}:/root/chat_server.py",
    ]
    try:
        subprocess.run(scp_cmd, check=True, capture_output=True, text=True, timeout=30)
        return True
    except Exception as e:
        logger.warning(f"Failed to scp chat_server.py to pod: {e}")
        return False


def restart_chat_server(model_name: str):
    """Kill old vLLM chat server and start with new king model.

    Uses the canonical chat_server.py bootstrap which handles config
    patching + visual weight grafting so vLLM can actually load the miner's
    checkpoint. Without that patch, miners that ship a Qwen3_5 VL-wrapper
    config (like tom9491/distil-32) fail with a weight-shape AssertionError
    during load because the visual branch is declared but empty.

    Persists ``model`` in ``state/chat_pod.json`` so the chat-tunnel watcher
    + healthcheck + admin CLI all agree on who's currently serving without
    re-reading h2h_latest.json (which the API path also touches and we
    don't want to make chat depend on validator pickup ordering).
    """
    logger.info(f"Restarting chat server with new king: {model_name}")
    if not CHAT_POD_HOST:
        logger.warning("Chat pod host is not configured; skipping chat server restart")
        return
    try:
        # VLLM v1 spawns a child that renames itself to "VLLM::EngineCore" —
        # killing only the entrypoints wrapper leaves the engine holding the
        # GPU. Kill both so the next server start can allocate cleanly.
        chat_ssh(
            "pkill -9 -f 'chat_server.py' 2>/dev/null; "
            "pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; "
            "pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; "
            "pkill -9 -x 'VLLM::EngineCor' 2>/dev/null; "
            "true",
            timeout=10,
        )
        time.sleep(3)
        _sync_chat_server()
        # setsid + < /dev/null so the bootstrap detaches cleanly from the SSH
        # session — otherwise the remote shell exits before nohup takes hold
        # and the python child dies with it.
        chat_ssh(
            f"setsid bash -c 'nohup python3 /root/chat_server.py "
            f"{model_name!r} {CHAT_POD_APP_PORT} "
            f"> /root/chat_server.log 2>&1 < /dev/null &'",
            timeout=10,
        )
        try:
            _write_chat_pod_state({"model": model_name}, source="restart_chat_server")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to persist chat pod model: {e}")
        logger.info("Chat vLLM server restart initiated")
    except Exception as e:
        logger.warning(f"Failed to restart chat server: {e}")


def trigger_benchmarks(model_name: str, king_uid: int):
    """Trigger auto-benchmarks on the eval pod for a new king model (background).

    Fires /root/auto_benchmark.sh on the eval pod (not the chat pod — the eval
    pod is where affine-benchmark is installed and where the DynamoDB writer
    lives). The script opens its own tunnel back to the chat pod's vLLM.
    """
    logger.info(f"Triggering auto-benchmarks for UID {king_uid} ({model_name})")
    if not CHAT_POD_HOST:
        logger.warning("Chat pod host is not configured; skipping auto-benchmark trigger")
        return
    try:
        # setsid + < /dev/null so the script detaches cleanly from SSH.
        # Quoting the model name via shlex equivalent (single-quote wrap) —
        # HF repo ids never contain single quotes so this is safe here.
        cmd = (
            f"setsid bash -c 'nohup /root/auto_benchmark.sh "
            f"\"{model_name}\" {int(king_uid)} "
            f"> /root/benchmark.log 2>&1 < /dev/null &'"
        )
        eval_ssh(cmd, timeout=10)
        logger.info(f"Benchmark trigger sent to eval pod for UID {king_uid}")
    except Exception as e:
        logger.warning(f"Failed to trigger benchmarks: {e}")


def ensure_chat_server_running(model_name: str):
    """Check if chat server is running with the right model; start/restart if needed.

    Calls into the dedicated probe helper so the validator + admin CLI +
    healthcheck all agree on what 'healthy' looks like (vLLM /v1/models
    plus model_name.txt match). Three failure modes get distinct handling:

    1. ``no chat pod configured`` — log + skip; don't pretend to fix it.
    2. ``ssh / curl unreachable`` — log warning; let the next round retry.
       Banging a relaunch on a dead pod just spams the GPU host.
    3. ``vLLM healthy but wrong model`` — relaunch with the right king.

    Relaunches also drive ``state/chat_pod.json`` so the chat-tunnel
    watcher and healthchecks don't lag behind.
    """
    if not CHAT_POD_HOST:
        logger.debug("Chat pod host is not configured; skipping ensure_chat_server_running")
        return
    try:
        result = _probe_chat_pod(timeout=12)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Chat server probe raised: {e}")
        return
    if not result.get("ok"):
        err = (result.get("error") or "").lower()
        if "no chat pod" in err:
            logger.debug("Chat pod not configured; skipping ensure_chat_server_running")
            return
        # Either the SSH connection failed (pod down/key rotated) or vLLM
        # is unreachable. Try a relaunch — chat_server.py is idempotent, it
        # kills any stale workers before binding the port.
        logger.info(f"Chat server unreachable ({err[:120]}); relaunching with {model_name}")
        restart_chat_server(model_name)
        return
    served = (result.get("model") or "").strip()
    if served != model_name:
        logger.info(
            f"Chat server running wrong model ({served!r}), restarting with {model_name}"
        )
        restart_chat_server(model_name)
