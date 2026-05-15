"""Lium GPU-pod lifecycle wrapper.

Thin wrapper around the proven ``eval.pod.PodManager`` from the legacy
``scripts/`` stack. We *do not* reinvent SSH/SFTP plumbing here — the
legacy module has 364 lines of battle-tested code (paramiko keepalive,
SFTP serialization lock, secrets-via-stdin, retry/backoff, B200
grouped_mm patches, Kimi K2.6 transformers-5.x compat patches, disk
cleanup) that we want to keep single-sourced through the cutover.

Two operating modes:

* **Persistent** (``attach_pod(name)``) — look up an existing Lium pod
  by name and re-use it for every round. This is the production path:
  the validator runs on the same 8xB200 pod for days, so teacher +
  student weights stay cached and per-round latency drops from ~6 h
  cold to ~45 min hot.
* **Ephemeral** (``acquire_pod()``) — provision a fresh pod, run one
  round, terminate. Used for tests / one-shot reproductions.

The eval itself runs on the pod via ``python -m distil.pod.orchestrator``
which fans out N parallel student shards across GPUs (see
``distil/pod/orchestrator.py``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.eval.pod")

# ── compat shim: reuse legacy eval.pod.PodManager + lium SDK ──────────
#
# eval/ is a sibling of distil/ in the repo. Add the repo root to
# sys.path so ``from eval.pod import PodManager`` resolves. This mirrors
# the pattern in ``distil/api/compat.py`` for the prod API routers.

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "eval" / "pod.py").is_file() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Files we upload to the pod for each round.
POD_UPLOAD_PATHS: tuple[str, ...] = ("distil", "pyproject.toml")

REMOTE_WORKDIR = "/home/distil_eval"


def _make_lium():
    """Construct a Lium SDK client using the same Config layout the legacy
    validator uses (api_key + ssh_key_path).
    """
    from lium import Config, Lium

    api_key = settings.lium_api_key or os.environ.get("LIUM_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "LIUM_API_KEY is missing — set it in /home/distil/.secrets/distil.env "
            "or as an environment variable for the validator unit."
        )
    ssh_key_path = Path.home() / ".ssh" / "id_ed25519"
    cfg = Config(api_key=api_key, ssh_key_path=ssh_key_path)
    return Lium(config=cfg)


def _pod_manager(lium, pod_name: str):
    """Return ``eval.pod.PodManager`` instance wrapping ``lium``."""
    from eval.pod import PodManager  # type: ignore[import]

    return PodManager(lium, pod_name=pod_name)


# ── pod context managers ──────────────────────────────────────────────


@contextmanager
def attach_pod(name: str | None = None):
    """Re-use an existing Lium pod by name (production path).

    Yields a connected :class:`PodManager`; does NOT terminate the pod
    on context exit.
    """
    pod_name = name or settings.lium_pod_name
    if not pod_name:
        raise ValueError(
            "attach_pod: no pod name (set DISTIL_LIUM_POD_NAME or pass name=)"
        )
    lium = _make_lium()
    pm = _pod_manager(lium, pod_name)
    pm.connect()  # raises RuntimeError if pod is missing
    logger.info(f"attached to persistent pod {pod_name!r}")
    yield pm
    # Persistent: do NOT terminate.


@contextmanager
def acquire_pod(*, size: str | None = None, label: str = "round"):
    """Provision a fresh ephemeral pod, run, then terminate."""
    sz = size or settings.lium_default_pod_size
    lium = _make_lium()
    # Pick the first available executor of the requested GPU class.
    # Size convention: "B200x1" → gpu_type="B200", gpu_count=1.
    gpu_type, _, gpu_count_s = sz.partition("x")
    gpu_count = int(gpu_count_s or "1")
    executors = lium.ls(gpu_type=gpu_type, gpu_count=gpu_count)
    if not executors:
        raise RuntimeError(f"no executors available for size {sz!r}")
    pod_name = f"distil-{label}-{int(time.time())}"
    up_res = lium.up(executor_id=executors[0].id, name=pod_name)
    pod_id = up_res.get("id") if isinstance(up_res, dict) else getattr(up_res, "id", None)
    if not pod_id:
        raise RuntimeError(f"lium.up returned no pod id: {up_res!r}")
    try:
        pod_info = lium.wait_ready(pod_id, timeout=600)
        if pod_info is None:
            raise RuntimeError(f"pod {pod_id} did not become ready")
        pm = _pod_manager(lium, pod_name)
        pm.connect()
        logger.info(f"acquired ephemeral pod {pod_name!r} ({pod_id[:12]})")
        yield pm
    finally:
        try:
            # Need a PodInfo for rm(); ps() returns the live list.
            match = next(
                (p for p in lium.ps() if getattr(p, "id", None) == pod_id),
                None,
            )
            if match is not None:
                lium.rm(match)
        except Exception as exc:
            logger.warning(f"pod {pod_id} terminate failed: {exc}")


# ── runtime install / upload ──────────────────────────────────────────


def upload_runtime(pod) -> None:
    """rsync the distil package + pyproject onto the pod under /home/."""
    for src in POD_UPLOAD_PATHS:
        local = Path(settings.repo_root) / src
        if not local.exists():
            logger.warning(f"upload_runtime: {local} missing, skipping")
            continue
        # lium.rsync handles both files and directories. The target
        # /home/ has the package extracted into /home/distil/ etc.
        pod.lium.rsync(pod.pod, local=str(local), remote="/home/")
    logger.info(f"uploaded {POD_UPLOAD_PATHS} to pod")


def install_runtime(pod) -> None:
    """``pip install -e .[gpu]`` on the pod. Idempotent."""
    res = pod.exec(
        "cd /home && python3 -m pip install --break-system-packages -q -e .[gpu] "
        "2>&1 | tail -20",
        timeout=900,
    )
    if not res.get("success"):
        logger.warning(
            f"pod install rc={res.get('exit_code')}\n"
            f"stdout={res.get('stdout', '')[-2000:]}\n"
            f"stderr={res.get('stderr', '')[-2000:]}"
        )


# ── eval execution ────────────────────────────────────────────────────


def _pod_run_state(pod, remote_run: str) -> str:
    """Returns 'absent' | 'in_progress' | 'complete' by probing the pod.

    A round counts as ``in_progress`` ONLY when ``round_spec.json`` is
    present AND a ``distil.pod.orchestrator`` process is still alive
    on the pod. Without the process check we'd wait forever on a stale
    spec left behind by a crashed orchestrator (Phase 1 import error,
    OOM, GPU init failure, etc.) — the polling loop would otherwise
    spin until the eval_round_max_minutes deadline before failing.
    """
    res = pod.exec(
        f"if [ -f {remote_run}/results.json ]; then echo complete; exit 0; fi; "
        f"if [ -f {remote_run}/round_spec.json ]; then "
        f"  if pgrep -f 'distil.pod.orchestrator' >/dev/null 2>&1; then "
        f"    echo in_progress; "
        f"  else "
        f"    echo stale; "
        f"  fi; "
        f"  exit 0; "
        f"fi; "
        f"echo absent",
        timeout=20,
    )
    if not res.get("success"):
        return "absent"
    out = (res.get("stdout") or "absent").strip()
    state = out.splitlines()[-1].strip() if out else "absent"
    # ``stale`` is treated as ``absent`` by callers (re-launch fresh),
    # but the caller logs the case so we don't lose forensic context.
    if state == "stale":
        logger.warning(
            f"pod has round_spec at {remote_run} but no live orchestrator "
            f"— previous attempt crashed; restarting"
        )
        return "absent"
    return state


def run_eval_on_pod(
    *,
    pod,
    round_spec: dict[str, Any],
    out_dir: Path,
    timeout_s: int | None = None,
    n_gpus: int | None = None,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    """Upload spec, invoke the parallel orchestrator on the pod, return merged results.

    Resume semantics when ``resume=True``:

    * If ``remote_run/results.json`` already exists → round complete,
      pull the artefacts and return immediately. Handles a validator
      restart mid-round on a persistent pod.
    * If ``remote_run/round_spec.json`` exists with no results → round
      is in progress (the orchestrator subprocess survived the SSH
      disconnect under ``setsid -f``). Wait for results to appear.
    * Otherwise: upload spec, start fresh.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_local = out_dir / "round_spec.json"
    spec_local.write_text(json.dumps(round_spec, indent=2))

    round_id = round_spec.get("round_id", int(time.time()))
    remote_run = f"{REMOTE_WORKDIR}/round_{round_id}"
    n = int(n_gpus or settings.eval_n_gpus or 8)
    deadline_s = timeout_s or settings.eval_round_max_minutes * 60

    pre = _pod_run_state(pod, remote_run) if resume else "absent"
    if pre == "complete":
        logger.info(f"resume: pod has results for round {round_id}; downloading")
    elif pre == "in_progress":
        logger.info(
            f"resume: pod has in-progress round {round_id}; waiting "
            f"(deadline {deadline_s / 60:.0f} min)"
        )
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            time.sleep(20)
            if _pod_run_state(pod, remote_run) == "complete":
                break
        else:
            raise RuntimeError(
                f"resume: round {round_id} did not complete within {deadline_s / 60:.0f} min"
            )
    else:
        pod.exec(f"mkdir -p {remote_run}", timeout=30)
        pod.upload(str(spec_local), f"{remote_run}/round_spec.json")
        # Launch orchestrator under ``setsid -f`` so it survives an SSH
        # disconnect — resume can find it on next attach.
        cmd = (
            f"cd /home && setsid -f python3 "
            f"-u -m distil.pod.orchestrator {remote_run}/round_spec.json "
            f"--workdir {remote_run} "
            f"--out {remote_run}/results.json "
            f"--progress {remote_run}/eval_progress.json "
            f"--n-gpus {n} "
            f"> {remote_run}/orchestrator.log 2>&1 < /dev/null && "
            f"echo launched"
        )
        res = pod.exec(cmd, timeout=60)
        if not res.get("success"):
            raise RuntimeError(
                f"orchestrator launch failed (rc={res.get('exit_code')}): "
                f"{res.get('stdout', '')}{res.get('stderr', '')}"
            )
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            time.sleep(20)
            if _pod_run_state(pod, remote_run) == "complete":
                break
        else:
            raise RuntimeError(
                f"round {round_id} did not complete within {deadline_s / 60:.0f} min"
            )

    # Pull artefacts back to the validator host.
    pod.download(f"{remote_run}/results.json", str(out_dir / "results.json"))
    for opt in ("eval_progress.json", "orchestrator.log"):
        try:
            pod.download(f"{remote_run}/{opt}", str(out_dir / opt))
        except Exception as exc:
            logger.warning(f"non-fatal download {opt}: {exc}")

    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise RuntimeError("pod did not produce results.json")
    return json.loads(results_path.read_text())


__all__ = [
    "acquire_pod",
    "attach_pod",
    "install_runtime",
    "run_eval_on_pod",
    "upload_runtime",
]
