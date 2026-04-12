"""
Pod lifecycle management for the SN97 validator.

Handles: connection, dependency installation, disk cleanup,
file upload/download, and command execution with retries.
"""
import json
import os
import re
import logging
import time
from pathlib import Path

logger = logging.getLogger("distillation.pod")

# Patterns for sanitizing GPU logs before public exposure
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SECRET_PATTERNS = re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}')
_SENSITIVE_KEYWORDS = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret")


def sanitize_gpu_log(raw: str) -> str:
    """Strip ANSI codes, secrets, and SSH noise from GPU pod logs before writing to disk."""
    lines = []
    for line in raw.splitlines():
        cleaned = _ANSI_RE.sub('', line).strip()
        if not cleaned:
            continue
        if any(kw in cleaned for kw in _SENSITIVE_KEYWORDS):
            continue
        if any(noise in cleaned for noise in (
            "sftp", "Authentication", "Connected (version", "chan ",
            "Opened sftp", "sftp session closed",
        )):
            continue
        cleaned = _SECRET_PATTERNS.sub('[REDACTED]', cleaned)
        lines.append(cleaned)
    return '\n'.join(lines)


def _retry(fn, max_attempts: int = 3, delay: float = 5.0, label: str = "operation"):
    """Generic retry wrapper. Returns the result of fn() or raises on final failure."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{label} failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


class PodManager:
    """Manages a Lium GPU pod for remote evaluation.

    Handles pod discovery, dependency installation, file transfers,
    command execution, and disk cleanup.

    Usage:
        pm = PodManager(lium, pod_name="distil-validator")
        pm.connect()
        pm.upload("scripts/pod_eval_vllm.py", "/home/pod_eval.py")
        result = pm.exec("python3 /home/pod_eval.py ...")
        pm.download("/home/eval_results.json", "state/last_eval.json")
    """

    def __init__(self, lium, pod_name: str = "distil-validator"):
        """Initialize with a Lium client and pod name to search for."""
        self.lium = lium
        self.pod_name = pod_name
        self.pod = None

    def connect(self):
        """Find and connect to the named Lium pod.

        Raises RuntimeError if the pod is not found.
        """
        pods = self.lium.ps()
        for p in pods:
            if self.pod_name in p.name:
                self.pod = p
                logger.info(f"Connected to pod: {p.name} ({p.id[:12]})")
                return
        available = [p.name for p in pods]
        raise RuntimeError(f"Pod '{self.pod_name}' not found. Available: {available}")

    def reconnect(self):
        """Re-discover the pod. Use after network failures or pod restarts."""
        logger.info("Reconnecting to pod...")
        self.pod = None
        self.connect()

    def upload(self, local: str, remote: str, max_attempts: int = 5):
        """Upload a file to the pod with retries."""
        def _do():
            self.lium.upload(self.pod, local=local, remote=remote)
        _retry(_do, max_attempts=max_attempts, delay=10, label=f"Upload {local}")
        logger.info(f"Uploaded {local} → {remote}")

    def download(self, remote: str, local: str, max_attempts: int = 3):
        """Download a file from the pod with retries."""
        def _do():
            self.lium.download(self.pod, remote=remote, local=local)
        _retry(_do, max_attempts=max_attempts, delay=5, label=f"Download {remote}")

    def exec(self, command: str, env: dict = None, timeout: int = None):
        """Execute a command on the pod. Returns the result dict.

        If timeout is specified, raises TimeoutError if the command
        doesn't complete within that many seconds.
        """
        import concurrent.futures
        kwargs = {"command": command}
        if env:
            kwargs["env"] = env
        if timeout is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.lium.exec, self.pod, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Pod exec timed out after {timeout}s: {command[:80]}")
                    raise TimeoutError(f"Pod exec timed out after {timeout}s")
        return self.lium.exec(self.pod, **kwargs)

    def is_alive(self, timeout: int = 15) -> bool:
        """Quick liveness check — returns True if the pod responds."""
        try:
            result = self.exec("echo alive", timeout=timeout)
            return "alive" in result.get("stdout", "")
        except Exception:
            return False

    def ensure_dependencies(self, teacher_model: str = "Qwen/Qwen3.5-35B-A3B"):
        """Install required packages on the pod and apply B200 patches.

        Installs vllm, accelerate, transformers, and patches grouped_mm
        for B200 (sm_100) GPUs where torch._grouped_mm crashes.
        """
        try:
            logger.info("Ensuring pod dependencies...")
            dep_result = self.exec(
                "pip install --break-system-packages 'vllm>=0.19' accelerate -q 2>&1 | tail -1 && "
                "pip install --break-system-packages 'transformers>=5.0' -q 2>&1 | tail -1 && "
                "python3 -c 'import torch; import transformers; import vllm; "
                "print(f\"torch={torch.__version__} transformers={transformers.__version__} "
                "vllm={vllm.__version__} cuda={torch.cuda.is_available()}\")'"
            )
            logger.info(f"Pod deps: {dep_result.get('stdout', '').strip()}")

            # Patch grouped_mm for B200 sm_100
            self.exec(
                'python3 -c "import torch; cap=torch.cuda.get_device_capability(0); '
                'print(f\\"GPU compute capability: {cap}\\")" && '
                'grep -q PATCHED /usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py 2>/dev/null || '
                'sed -i \'s/return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/'
                '# PATCHED: force fallback on sm_100 (B200)\n'
                '    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)\n'
                '    if cap[0] >= 10:\n'
                '        return hasattr(torch.nn.functional, "grouped_mm")\n'
                '    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/\' '
                '/usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py'
            )
            logger.info("Applied grouped_mm B200 patch")
        except Exception as e:
            logger.warning(f"Pod dep check failed (non-fatal): {e}")

    def disk_cleanup(self, teacher_name: str, threshold: int = 85):
        """Clean non-teacher model caches and stale files from the pod.

        Keeps the teacher model cached. Cleans student caches,
        stale /tmp files, and old eval artifacts.
        """
        try:
            disk_check = self.exec("df --output=pcent / | tail -1 | tr -d ' %'")
            disk_pct_str = disk_check.get('stdout', disk_check) if isinstance(disk_check, dict) else disk_check
            disk_pct = int(str(disk_pct_str).strip())
            logger.info(f"Pod disk: {disk_pct}% used")

            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_teacher_only.json 2>/dev/null; "
                "df -h / | tail -1"
            )
            clean_result = self.exec(clean_cmd)
            clean_info = clean_result.get('stdout', clean_result) if isinstance(clean_result, dict) else clean_result
            logger.info(f"Cleanup done: {str(clean_info).strip()}")
            return disk_pct
        except Exception as e:
            logger.warning(f"Disk cleanup failed (non-fatal): {e}")
            return 0

    def clear_gpu(self):
        """Kill background GPU processes to free VRAM for eval."""
        try:
            self.exec("for s in distil train; do tmux kill-session -t $s 2>/dev/null; done; sleep 2; echo 'GPU cleared'")
            logger.info("Cleared GPU for eval")
        except Exception:
            pass

    def resume_background_tasks(self):
        """Restart any background tasks that were cleared for eval."""
        try:
            self.exec("test -f /home/autostart.sh && bash /home/autostart.sh; echo 'Background tasks resumed'")
            logger.info("Resumed background tasks on pod")
        except Exception:
            pass

    def post_eval_cleanup(self, teacher_name: str):
        """Clean up after eval: remove student caches, stale teacher cache, /tmp."""
        try:
            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "rm -f /home/teacher_cache.pt 2>/dev/null; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "df -h / | tail -1"
            )
            result = self.exec(clean_cmd)
            disk_info = result.get('stdout', result) if isinstance(result, dict) else result
            logger.info(f"Post-eval cleanup: {str(disk_info).strip()}")
        except Exception as e:
            logger.warning(f"Post-eval cleanup failed (non-fatal): {e}")
