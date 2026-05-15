"""Lium GPU-pod lifecycle wrapper.

The validator never executes Python on its own GPU — it rents a B200
pod via Lium per round, uploads :mod:`distil.pod` + the round spec,
runs ``python -m distil.pod /home/round.json``, and rsyncs back the
results JSON. This module is a thin convenience wrapper around the
``lium`` Python client + ``ssh`` for the actual command exec.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.eval.pod")

# Files we always upload from the validator host.
POD_UPLOAD_PATHS = (
    "distil",
    "pyproject.toml",
)


@dataclass
class Pod:
    """A live Lium pod handle (id + ssh target)."""

    pod_id: str
    ssh: str
    size: str = ""

    def run(self, cmd: str, *, timeout: int | None = None) -> tuple[int, str]:
        full = ["ssh", "-o", "StrictHostKeyChecking=no", self.ssh, cmd]
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def rsync_up(self, src: str, dst: str) -> None:
        subprocess.check_call(
            [
                "rsync",
                "-az",
                "--delete",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                src,
                f"{self.ssh}:{dst}",
            ]
        )

    def rsync_down(self, src: str, dst: str) -> None:
        subprocess.check_call(["rsync", "-az", f"{self.ssh}:{src}", dst])


def _lium_client():
    try:
        from lium import LiumClient
    except ImportError as exc:
        raise RuntimeError("lium client not installed") from exc
    return LiumClient(api_key=settings.lium_api_key or None)


@contextmanager
def acquire_pod(*, size: str | None = None, label: str = "round"):
    """Provision -> yield -> always tear down a Lium pod."""
    sz = size or settings.lium_default_pod_size
    client = _lium_client()
    pod = client.create_pod(size=sz, name=f"distil-{label}-{int(time.time())}")
    try:
        ssh = pod.ssh_target  # e.g. "ubuntu@1.2.3.4"
        yield Pod(pod_id=pod.id, ssh=ssh, size=sz)
    finally:
        try:
            client.terminate_pod(pod.id)
        except Exception as exc:
            logger.warning(f"pod {pod.id} terminate failed: {exc}")


def upload_runtime(pod: Pod) -> None:
    """Upload the distil package + pyproject for a round."""
    for src in POD_UPLOAD_PATHS:
        local = Path(settings.repo_root) / src
        if local.exists():
            pod.rsync_up(str(local) + ("/" if local.is_dir() else ""), "/home/")


def install_runtime(pod: Pod) -> None:
    code, out = pod.run(
        "cd /home && python3 -m pip install --break-system-packages -q -e .[gpu] 2>&1 | tail -20",
        timeout=900,
    )
    if code != 0:
        logger.warning(f"pod install rc={code}\n{out[-2000:]}")


def run_eval_on_pod(
    *,
    pod: Pod,
    round_spec: dict[str, Any],
    out_dir: Path,
    timeout_s: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Upload spec, run ``distil.pod``, fetch ``results.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_local = out_dir / "round_spec.json"
    spec_local.write_text(json.dumps(round_spec, indent=2))
    pod.rsync_up(str(spec_local), "/home/round_spec.json")

    cmd = (
        "cd /home && python3 -m distil.pod /home/round_spec.json "
        "--out /home/results.json --progress /home/eval_progress.json "
        "2>&1 | tee /home/eval.log"
    )
    code, out = pod.run(shlex.quote(cmd).strip("'"), timeout=timeout_s)
    if code != 0:
        logger.warning(f"pod run rc={code} (last 2k chars):\n{out[-2000:]}")
    pod.rsync_down("/home/results.json", str(out_dir))
    pod.rsync_down("/home/eval_progress.json", str(out_dir))
    pod.rsync_down("/home/eval.log", str(out_dir))
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise RuntimeError("pod did not produce results.json")
    return json.loads(results_path.read_text())
