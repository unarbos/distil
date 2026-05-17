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


# Env vars forwarded from the validator host to the pod orchestrator
# command line. This is the same whitelist the legacy
# ``scripts/validator/pod_session.py`` uses (minus the legacy-only
# DISTIL_USE_PARALLEL_ORCH knob which has no analogue here — distil
# always parallelizes). The most critical entries are the cloud-teacher
# API key (``DISTIL_TEACHER_API_KEY`` / ``OPENROUTER_API_KEY``) and the
# vLLM tunables for student scoring; without those forwarded the pod
# Phase 1 fails fast and Phase 2 silently regresses to slow defaults.
_REMOTE_ENV_WHITELIST: tuple[str, ...] = (
    # Cloud-teacher path (Kimi-K2.6 cutover, 2026-05-03)
    "DISTIL_TEACHER_MODE",
    "DISTIL_TEACHER_REPO",
    "DISTIL_TEACHER_API_BASE",
    "DISTIL_TEACHER_API_KEY",
    "OPENROUTER_API_KEY",
    "DISTIL_TEACHER_API_MODEL",
    "DISTIL_TEACHER_API_ENDPOINT",
    "DISTIL_TEACHER_API_PROVIDERS",
    "DISTIL_TEACHER_API_CONCURRENCY",
    "DISTIL_TEACHER_API_TOP_LOGPROBS",
    "DISTIL_TEACHER_API_TIMEOUT_S",
    "DISTIL_TEACHER_API_DISABLE_REASONING",
    "DISTIL_OPENROUTER_REFERER",
    "DISTIL_OPENROUTER_TITLE",
    # Student vLLM (2026-05-14 8xB200 pivot)
    "DISTIL_STUDENT_BATCH_SIZE",
    "DISTIL_STUDENT_USE_VLLM",
    "DISTIL_STUDENT_VLLM_TOKENIZER",
    "DISTIL_STUDENT_VLLM_MAX_LEN",
    "DISTIL_STUDENT_VLLM_GPU_UTIL",
    "DISTIL_STUDENT_VLLM_TP",
    "DISTIL_STUDENT_VLLM_TRC",
    "DISTIL_STUDENT_VLLM_EAGER",
    # vLLM 0.20.2 needs deep_gemm disabled for Kimi-K2.6 MoE on B200
    "VLLM_USE_DEEP_GEMM",
    "VLLM_USE_DEEP_GEMM_E8M0",
    # Vocab override for K2.6 (163840) — pod precheck refuses without it.
    "ACTIVATION_FP_VOCAB_SIZE",
    "TEACHER_CONFIG_VOCAB_SIZE",
    # Watchdog tuning (2026-05-15 fan-out)
    "DISTIL_ORCH_WATCHDOG_S",
    # NOTE: orchestrator.py reads ``DISTIL_ORCH_WATCHDOG_REPEAT`` (no
    # trailing ``_N``). They MUST stay aligned or operator tuning is
    # silently ineffective — the orchestrator just keeps the hardcoded
    # default of 16 even when the host sets a different value.
    "DISTIL_ORCH_WATCHDOG_REPEAT",
    "DISTIL_CACHE_KEEP_MODELS",
    # HF auth for tokenizer/model pulls
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _shell_quote(value: str) -> str:
    """Single-quote for POSIX shell (escape inner single quotes)."""
    return "'" + value.replace("'", "'\\''") + "'"


def _build_remote_env_prefix(extra: dict[str, str] | None = None) -> str:
    """Return ``KEY=val KEY=val `` (trailing space) to prepend to a remote cmd.

    Values are POSIX-shell quoted. Only forwards env vars whose name is
    in ``_REMOTE_ENV_WHITELIST`` AND is non-empty in the validator
    process. An empty whitelisted var is treated as "operator wants
    the pod default" (consistent with legacy behavior).
    """
    parts: list[str] = []
    for key in _REMOTE_ENV_WHITELIST:
        val = os.environ.get(key)
        if val is None or val == "":
            continue
        parts.append(f"{key}={_shell_quote(val)}")
    if extra:
        for k, v in extra.items():
            if v is None or v == "":
                continue
            parts.append(f"{k}={_shell_quote(str(v))}")
    return (" ".join(parts) + " ") if parts else ""


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


def _stream_pod_log(
    pod,
    remote_run: str,
    round_id: int,
    *,
    round_spec: dict | None = None,
) -> None:
    """Mirror pod-side eval log + progress into ``state/`` so the
    healthcheck sees the round as alive.

    Production's ``sn97_healthcheck.py`` watches TWO files:

    * ``state/gpu_eval.log`` — if ``eval_progress.active`` is True
      and the log mtime is older than ``GPU_LOG_STALE_SEC``, it
      force-restarts the validator with reason ``gpu_log_stale``.
    * ``state/eval_progress.json`` — read for ``.active``,
      ``.phase``, ``.students_*``, ``.teacher_prompts_done`` etc.
      If the file is stale while ``.active=true``, it triggers
      ``validator:stale_eval_progress`` → restart.

    The legacy validator streamed paramiko stdout into gpu_eval.log
    continuously and rewrote eval_progress.json after every shard
    progress event. The distil polling loop wrote neither — which is
    why the cutover validator was SIGKILLed by the healthcheck every
    minute even though the pod-side orchestrator was healthy.

    Best-effort: fetch the last 200 lines of orchestrator.log + each
    active phase log and copy the pod-side ``eval_progress.json``
    onto the host. Wraps the pod payload so it matches the legacy
    eval_progress schema the healthcheck expects (``active``,
    ``phase``, ``students_total``, etc.). Never raises — a stale SSH
    channel is non-fatal; the next iteration retries.
    """
    try:
        state_dir = Path(settings.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        local_log = state_dir / "gpu_eval.log"
        local_progress = state_dir / "eval_progress.json"

        # ``orchestrator.log`` is the canonical pod-side eval log. We
        # also tail phase[1-3]_*.log so per-phase progress (api teacher
        # 50/256, student vllm warmup, etc.) shows up in the host log.
        cmd = (
            f"tail -n 200 {remote_run}/orchestrator.log 2>/dev/null; "
            f"echo '--- phase logs ---'; "
            f"for f in {remote_run}/phase*_*.log; do "
            f"[ -f \"$f\" ] && echo \"=== $(basename $f) ===\" && "
            f"tail -n 40 \"$f\"; "
            f"done 2>/dev/null; "
            f"echo '--- progress ---'; "
            f"cat {remote_run}/eval_progress.json 2>/dev/null"
        )
        res = pod.exec(cmd, timeout=15)
        if not res.get("success"):
            return
        content = res.get("stdout") or ""
        # Split off the progress JSON tail to write eval_progress.json
        # separately. The marker is unique enough that we don't need a
        # robust parser — last occurrence wins.
        pod_progress: dict[str, Any] = {}
        if "--- progress ---" in content:
            log_part, _, prog_part = content.rpartition("--- progress ---")
            try:
                pod_progress = json.loads(prog_part.strip())
            except Exception:
                pod_progress = {}
            content = log_part
        local_log.write_text(
            f"# round={round_id} updated_at={time.time():.0f}\n{content}"
        )

        # Build host-side eval_progress.json in the legacy schema the
        # healthcheck reads. ``active=true`` while polling; phase +
        # teacher_prompts_done come from the pod payload; models and
        # eval_order come from the round_spec we built on the host.
        # ``students_total`` should be the count of student models running
        # this round (1 king + N challengers, typically 11), NOT the
        # ``n_prompts`` (256) the teacher API call sends. The legacy
        # fallback ``pod_progress.get("n_prompts") or students_total``
        # had the wrong precedence — during the ``teacher_generating``
        # phase the pod payload carries ``n_prompts: 256`` but no
        # ``students_total``, so the dashboard rendered ``0/256
        # students`` for the entire teacher window, and miners read it
        # as "256 students to score". Fix: prefer pod-provided
        # ``students_total`` (set by the orchestrator during the
        # scoring phase) and otherwise fall back to the
        # round_spec's student list length, which IS the correct count.
        spec_students_total = len(round_spec.get("students") or []) if round_spec else None
        host_progress: dict[str, Any] = {
            "active": True,
            "updated_at": time.time(),
            "round_id": round_id,
            "phase": pod_progress.get("phase") or "polling",
            "students_total": (
                pod_progress.get("students_total")
                or spec_students_total
                or None
            ),
            "students_done": (
                pod_progress.get("students_done")
                if pod_progress.get("students_done") is not None
                else (
                    len(pod_progress.get("completed", []))
                    if isinstance(pod_progress.get("completed"), list)
                    else None
                )
            ),
            "teacher_prompts_done": pod_progress.get("teacher_prompts_done"),
            "n_prompts": pod_progress.get("n_prompts"),
            "pod": pod_progress,
        }
        if round_spec:
            host_progress["models"] = {
                str(s["uid"]): s.get("repo", "")
                for s in (round_spec.get("students") or [])
            }
            host_progress["eval_order"] = [
                {
                    "uid": s["uid"],
                    "model": s.get("repo", ""),
                    "role": "king" if s.get("is_king") else "challenger",
                }
                for s in (round_spec.get("students") or [])
            ]
            host_progress["king_uid"] = next(
                (s["uid"] for s in (round_spec.get("students") or []) if s.get("is_king")),
                None,
            )
            host_progress["challenger_uids"] = [
                s["uid"]
                for s in (round_spec.get("students") or [])
                if not s.get("is_king")
            ]
        tmp = str(local_progress) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(host_progress, f)
        os.replace(tmp, local_progress)
    except Exception as exc:
        logger.debug(f"_stream_pod_log non-fatal: {exc}")


def _mark_progress_inactive(round_id: int) -> None:
    """Flip ``state/eval_progress.json`` to ``active=false`` so the
    healthcheck stops watching the gpu log freshness for this round."""
    try:
        path = Path(settings.state_dir) / "eval_progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": False,
            "updated_at": time.time(),
            "round_id": round_id,
            "phase": "finished",
        }
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug(f"_mark_progress_inactive non-fatal: {exc}")


def _pod_run_state(pod, remote_run: str) -> str:
    """Returns 'absent' | 'in_progress' | 'complete' by probing the pod.

    Completion is gated on the ``results.done`` sentinel that the
    orchestrator writes ONLY after Phase 3 (judge grading) finishes
    merging its scores into ``results.json``. Probing for
    ``results.json`` directly is racy: Phase 2 writes the file first
    (no judge axes), then Phase 3 rewrites it 30-60 s later with
    judge_probe/long_form_judge_probe/chat_turns_probe merged in. The
    host's 20s polling loop would otherwise see ``results.json``
    existing between the two writes, mark the round complete, and
    download a judge-axis-less file — silently losing every Phase 3
    score for the round.

    A round counts as ``in_progress`` ONLY when ``round_spec.json`` is
    present AND a ``distil.pod.orchestrator`` process is still alive
    on the pod. Without the process check we'd wait forever on a stale
    spec left behind by a crashed orchestrator (Phase 1 import error,
    OOM, GPU init failure, etc.) — the polling loop would otherwise
    spin until the eval_round_max_minutes deadline before failing.
    """
    res = pod.exec(
        f"if [ -f {remote_run}/results.done ]; then echo complete; exit 0; fi; "
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
            _stream_pod_log(pod, remote_run, round_id, round_spec=round_spec)
            if _pod_run_state(pod, remote_run) == "complete":
                break
        else:
            _mark_progress_inactive(round_id)
            raise RuntimeError(
                f"resume: round {round_id} did not complete within {deadline_s / 60:.0f} min"
            )
    else:
        pod.exec(f"mkdir -p {remote_run}", timeout=30)
        pod.upload(str(spec_local), f"{remote_run}/round_spec.json")
        # Sweep orphaned vLLM EngineCore + distil.pod processes from
        # prior rounds. When a Phase-2 shard dies ungracefully (OOM, GPU
        # init crash, watchdog SIGKILL) its child ``VLLM::EngineCore``
        # gets re-parented to init and survives with the model weights
        # still resident in GPU memory. The next round's shard then
        # cannot allocate its own engine because the device is full
        # (e.g. round 1778905724 / 2026-05-16 lost the king because
        # GPU 0 was holding 162 GB from a 3-hour-old leak). Killing any
        # EngineCore whose parent is no longer a live ``distil.pod`` is
        # safe — the only legitimate parent is our orchestrator's own
        # shard subprocess, which won't exist yet when we sweep.
        cleanup_cmd = (
            "for pid in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do "
            "  ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' '); "
            "  parent=$(ps -o cmd= -p $ppid 2>/dev/null || echo init); "
            "  case \"$parent\" in "
            "    *distil.pod*) ;; "
            "    *) echo \"sweep: kill stale vLLM pid=$pid (parent='$parent')\"; "
            "       kill -9 $pid 2>/dev/null ;; "
            "  esac; "
            "done; "
            "pkill -9 -f 'distil.pod.orchestrator' 2>/dev/null; "
            "pkill -9 -f 'distil.pod ' 2>/dev/null; "
            "sleep 1; "
            "echo 'cleanup_done'"
        )
        sweep = pod.exec(cleanup_cmd, timeout=30)
        if sweep.get("stdout") and "sweep:" in sweep["stdout"]:
            logger.warning(
                f"pre-launch cleanup swept stale procs:\n{sweep['stdout'].strip()}"
            )
        # Forward configuration/secrets from the validator env to the pod.
        # The orchestrator inherits ``cd /home`` env only, so anything our
        # phase subprocesses need (API key for the cloud-teacher path,
        # vLLM/HF knobs, bench budgets, …) MUST be injected explicitly.
        # We re-use the legacy ``scripts/validator/pod_session.py`` env
        # whitelist verbatim — anything not in there is a deliberate
        # "pod-side default" knob that operators tune in the pod's own
        # systemd unit, not the host's.
        env_prefix = _build_remote_env_prefix()
        # Launch orchestrator under ``setsid -f`` so it survives an SSH
        # disconnect — resume can find it on next attach.
        cmd = (
            f"cd /home && {env_prefix}setsid -f python3 "
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
        # Touch the host-side heartbeat files immediately so the
        # sn97_healthcheck timer (which fires every minute) doesn't
        # observe a 15-minute-stale gpu_eval.log during the 20s
        # before the first poll iteration.
        _stream_pod_log(pod, remote_run, round_id, round_spec=round_spec)
        deadline = time.time() + deadline_s
        # Tolerate a brief startup window where the orchestrator process
        # hasn't yet been picked up by pgrep (setsid + forking takes a
        # few seconds). After that window, if ``_pod_run_state`` returns
        # ``absent`` (which is what crashed/exited orchestrators map to),
        # we fail fast instead of polling the full ``deadline_s`` — the
        # round can't recover without a re-launch from the host side.
        startup_grace_until = time.time() + 90
        while time.time() < deadline:
            time.sleep(20)
            _stream_pod_log(pod, remote_run, round_id, round_spec=round_spec)
            state = _pod_run_state(pod, remote_run)
            if state == "complete":
                break
            if state == "absent" and time.time() > startup_grace_until:
                _mark_progress_inactive(round_id)
                raise RuntimeError(
                    f"round {round_id} orchestrator died before producing "
                    f"results.done (check {remote_run}/phase*_*.log on the "
                    f"pod for the crash trace)"
                )
        else:
            _mark_progress_inactive(round_id)
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
    _mark_progress_inactive(round_id)

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
