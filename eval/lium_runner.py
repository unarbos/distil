"""
Run vLLM inference on a Lium GPU pod.

Handles pod lifecycle (create → setup → eval → download → cleanup)
and provides a high-level API for running model evaluation on remote GPUs.
"""
import json, time, logging, tempfile, os
from pathlib import Path

logger = logging.getLogger("distillation.lium")

LIUM_API_KEY = "***REMOVED***"
DEFAULT_EXECUTORS = [
    "b0dbc862-3056-4326-b1f6-1fea92cbb009",
    "d34ac3d3-a68a-4ddb-825d-65926d64ef46",
]


class LiumRunner:
    """Manages a Lium GPU pod for model evaluation."""

    def __init__(self, api_key=LIUM_API_KEY, executor_id=None):
        from lium import Lium, Config
        from pathlib import Path
        # Find SSH key
        ssh_key = None
        for key_name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
            key_path = Path.home() / ".ssh" / key_name
            if key_path.exists():
                ssh_key = key_path
                break
        self.cfg = Config(api_key=api_key, ssh_key_path=ssh_key)
        self.client = Lium(config=self.cfg)
        self.executor_id = executor_id
        self.pod = None

    def create_pod(self, name="distillation-eval"):
        """Spin up an H200 pod."""
        if not self.executor_id:
            executors = self.client.ls(gpu_type="H200")
            if executors:
                self.executor_id = executors[0].id
            else:
                self.executor_id = DEFAULT_EXECUTORS[0]

        logger.info(f"Creating pod on executor {self.executor_id}...")
        pod_data = self.client.up(
            executor_id=self.executor_id,
            name=name,
        )
        pod_id = pod_data["id"] if isinstance(pod_data, dict) else pod_data.id
        logger.info(f"Waiting for pod {pod_id} to be ready...")
        self.pod = self.client.wait_ready(pod_id, timeout=600)
        logger.info(f"Pod ready!")
        return self.pod

    def setup_vllm(self):
        """Install vLLM on the pod."""
        logger.info("Installing vLLM (this may take a few minutes)...")
        # Use --break-system-packages for externally-managed environments
        output = self.stream_command(
            "pip install vllm --break-system-packages 2>&1 | tail -30"
        )
        # Verify installation
        result = self.client.exec(self.pod, command="python3 -c 'import vllm; print(vllm.__version__)'")
        logger.info(f"vLLM version: {result}")
        return result

    def upload_file(self, local_path, remote_path):
        """Upload a file to the pod."""
        logger.info(f"Uploading {local_path} → {remote_path}")
        self.client.upload(self.pod, local=local_path, remote=remote_path)

    def download_file(self, remote_path, local_path):
        """Download a file from the pod."""
        logger.info(f"Downloading {remote_path} → {local_path}")
        self.client.download(self.pod, remote=remote_path, local=local_path)

    def run_eval_script(self, model_name, prompts_remote, output_remote,
                        max_tokens=256, top_k=20):
        """Run the pod_eval.py script for a single model."""
        cmd = (
            f"python3 /home/pod_eval.py "
            f"--model {model_name} "
            f"--prompts {prompts_remote} "
            f"--output {output_remote} "
            f"--max-tokens {max_tokens} "
            f"--top-k {top_k}"
        )
        logger.info(f"Running: {cmd}")
        output_lines = []
        for chunk in self.client.stream_exec(self.pod, command=cmd):
            text = chunk.get("output", chunk.get("stdout", str(chunk)))
            if text:
                logger.info(f"  [pod] {text.rstrip()}")
                output_lines.append(text)
        # Verify output file was created
        check = self.client.exec(self.pod, command=f"test -f {output_remote} && echo OK || echo MISSING")
        stdout = check.get("stdout", "") if isinstance(check, dict) else str(check)
        if "MISSING" in stdout:
            raise RuntimeError(f"Eval script failed for {model_name} — output file not created")
        return "\n".join(output_lines)

    def run_command(self, command):
        """Run an arbitrary command on the pod."""
        logger.info(f"Exec: {command}")
        return self.client.exec(self.pod, command=command)

    def stream_command(self, command):
        """Stream output of a command on the pod."""
        logger.info(f"Stream exec: {command}")
        output_lines = []
        for chunk in self.client.stream_exec(self.pod, command=command):
            text = chunk.get("output", chunk.get("stdout", str(chunk)))
            if text:
                logger.info(f"  [pod] {text.rstrip()}")
                output_lines.append(text)
        return "\n".join(output_lines)

    def cleanup(self):
        """Destroy the pod."""
        if self.pod:
            logger.info("Cleaning up pod...")
            try:
                self.client.rm(self.pod)
                logger.info("Pod destroyed")
            except Exception as e:
                logger.warning(f"Pod cleanup failed: {e}")
            self.pod = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
