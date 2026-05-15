"""Lium GPU-pod lifecycle wrapper.

Two operating modes:

* **Ephemeral** (``acquire_pod()``) — provision a fresh pod via the
  Lium client, run one round, terminate. Useful for tests and one-shot
  reproductions; not what runs in production.
* **Persistent** (``attach_pod(name)``) — look up an existing pod by
  name through the Lium client and re-use it for every round. This is
  the production path: validators run on the same 8xB200 pod for days,
  so teacher + student model weights stay cached and per-round latency
  drops from ~6 hours of cold downloads to ~47 minutes of evals.

The eval itself runs on the pod via
``python -m distil.pod.orchestrator``, which fans out N parallel
student shards across GPUs (see ``distil/pod/orchestrator.py``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.eval.pod")

# Files we upload to the pod for a round.
POD_UPLOAD_PATHS: tuple[str, ...] = ("distil", "pyproject.toml")

REMOTE_WORKDIR = "/home/distil_eval"  # Where round_spec.json + shard dirs live.


@dataclass
class Pod:
    """A live Lium pod handle (id + ssh target + lifecycle metadata)."""

    pod_id: str
    ssh: str  # e.g. "ubuntu@1.2.3.4" or "root@1.2.3.4 -p 10109"
    persistent: bool = False  # True = don't terminate on context exit.
    size: str = ""
    name: str = ""

    def _ssh_argv(self, cmd: str) -> list[str]:
        # Honour any "-p PORT" tail in the ssh target; the user@host[:port]
        # form is interpreted via shell-split here.
        parts = self.ssh.split()
        target = parts[0]
        extra = parts[1:]
        return ["ssh", "-o", "StrictHostKeyChecking=no", *extra, target, cmd]

    def run(self, cmd: str, *, timeout: int | None = None) -> tuple[int, str]:
        proc = subprocess.run(
            self._ssh_argv(cmd), capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def rsync_up(self, src: str, dst: str) -> None:
        parts = self.ssh.split()
        target = parts[0]
        extra = parts[1:]
        rsh = "ssh -o StrictHostKeyChecking=no"
        if extra:
            rsh += " " + " ".join(extra)
        subprocess.check_call(
            [
                "rsync",
                "-az",
                "-e",
                rsh,
                "--delete",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                src,
                f"{target}:{dst}",
            ]
        )

    def rsync_down(self, src: str, dst: str) -> None:
        parts = self.ssh.split()
        target = parts[0]
        extra = parts[1:]
        rsh = "ssh -o StrictHostKeyChecking=no"
        if extra:
            rsh += " " + " ".join(extra)
        subprocess.check_call(
            ["rsync", "-az", "-e", rsh, f"{target}:{src}", dst]
        )


def _lium_client():
    try:
        from lium import LiumClient
    except ImportError as exc:
        raise RuntimeError("lium client not installed") from exc
    return LiumClient(api_key=settings.lium_api_key or None)


@contextmanager
def acquire_pod(*, size: str | None = None, label: str = "round"):
    """Provision a fresh pod, yield :class:`Pod`, always terminate."""
    sz = size or settings.lium_default_pod_size
    client = _lium_client()
    pod = client.create_pod(size=sz, name=f"distil-{label}-{int(time.time())}")
    try:
        yield Pod(pod_id=pod.id, ssh=pod.ssh_target, size=sz, persistent=False)
    finally:
        try:
            client.terminate_pod(pod.id)
        except Exception as exc:
            logger.warning(f"pod {pod.id} terminate failed: {exc}")


@contextmanager
def attach_pod(name: str | None = None):
    """Re-use an existing Lium pod by name (production path).

    ``name`` defaults to ``settings.lium_pod_name`` (env: ``DISTIL_LIUM_POD_NAME``).
    Yields a :class:`Pod` with ``persistent=True``; the context exit does
    NOT terminate the pod.
    """
    pod_name = name or settings.lium_pod_name
    if not pod_name:
        raise ValueError("attach_pod: no pod name (set DISTIL_LIUM_POD_NAME)")
    client = _lium_client()
    pods = client.list_pods() or []
    match = next((p for p in pods if getattr(p, "name", None) == pod_name), None)
    if match is None:
        raise RuntimeError(f"attach_pod: no Lium pod named {pod_name!r}")
    yield Pod(
        pod_id=match.id,
        ssh=match.ssh_target,
        size=getattr(match, "size", "") or "",
        name=pod_name,
        persistent=True,
    )
    # Persistent: do NOT terminate.


def upload_runtime(pod: Pod) -> None:
    """Rsync the distil package + pyproject onto the pod."""
    for src in POD_UPLOAD_PATHS:
        local = Path(settings.repo_root) / src
        if local.exists():
            pod.rsync_up(str(local) + ("/" if local.is_dir() else ""), "/home/")


def install_runtime(pod: Pod) -> None:
    code, out = pod.run(
        "cd /home && python3 -m pip install --break-system-packages -q -e .[gpu] "
        "2>&1 | tail -20",
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
    n_gpus: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Upload spec, invoke the parallel orchestrator on the pod, return merged results.

    On the pod side this becomes:

        python -m distil.pod.orchestrator round_spec.json \\
            --workdir /home/distil_eval/<round_id> \\
            --out /home/distil_eval/<round_id>/results.json \\
            --progress /home/distil_eval/<round_id>/eval_progress.json \\
            --n-gpus 8
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_local = out_dir / "round_spec.json"
    spec_local.write_text(json.dumps(round_spec, indent=2))

    round_id = round_spec.get("round_id", int(time.time()))
    remote_run = f"{REMOTE_WORKDIR}/round_{round_id}"
    n = int(n_gpus or settings.eval_n_gpus or 8)

    pod.run(f"mkdir -p {remote_run}", timeout=30)
    pod.rsync_up(str(spec_local), f"{remote_run}/round_spec.json")

    cmd = (
        f"cd /home && {os.environ.get('DISTIL_PYTHON_REMOTE', 'python3')} "
        f"-u -m distil.pod.orchestrator {remote_run}/round_spec.json "
        f"--workdir {remote_run} "
        f"--out {remote_run}/results.json "
        f"--progress {remote_run}/eval_progress.json "
        f"--n-gpus {n} 2>&1 | tee {remote_run}/orchestrator.log"
    )
    code, out = pod.run(cmd, timeout=timeout_s)
    if code != 0:
        logger.warning(f"orchestrator rc={code} (last 2k):\n{out[-2000:]}")

    # Pull artefacts back to the validator host.
    pod.rsync_down(f"{remote_run}/results.json", str(out_dir))
    pod.rsync_down(f"{remote_run}/eval_progress.json", str(out_dir))
    pod.rsync_down(f"{remote_run}/orchestrator.log", str(out_dir))
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise RuntimeError("pod did not produce results.json")
    return json.loads(results_path.read_text())


__all__ = [
    "Pod",
    "acquire_pod",
    "attach_pod",
    "install_runtime",
    "run_eval_on_pod",
    "upload_runtime",
]
