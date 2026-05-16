import json
import logging
import os
import shlex
import tempfile
import time

from eval.pod import PodManager, sanitize_gpu_log
from eval.state import ValidatorState, log_event
from eval.runtime import TEACHER_CONFIG_VOCAB_SIZE
from scripts.eval_policy import as_env as eval_policy_env
from scripts.eval_policy import policy_env, policy_path
from scripts.validator.config import MAX_NEW_TOKENS, TEACHER_MODEL, VLLM_CONCURRENCY
from scripts.validator.pod_runtime import (
    CURRENT_FIELD_MAP,
    POD_PROGRESS_METADATA_KEYS,
    upload_aux_modules,
)

# Opt-in teacher tensor-parallel size. 0 = let pod autodetect from torch.cuda.device_count().
TP_SIZE = int(policy_env("DISTIL_TP_SIZE", "0") or "0")
# Same-point early-stop floor; 0 disables (matches legacy behaviour).
EARLY_STOP_MIN = int(policy_env("DISTIL_EARLY_STOP_MIN", "0") or "0")

logger = logging.getLogger("distillation.remote_validator")


def _is_api_teacher_mode() -> bool:
    """True iff DISTIL_TEACHER_MODE=api in the validator env."""
    return (policy_env("DISTIL_TEACHER_MODE", "") or "").lower() == "api"


def _pod_exec_silent(pod, cmd: str, *, timeout: int = 30, label: str | None = None) -> None:
    """Best-effort ``pod.exec(cmd)`` for cleanup/kill commands; swallows
    any exception (debug-logs ``label: <exc>`` when label is provided)."""
    try:
        pod.exec(cmd, timeout=timeout)
    except Exception as exc:
        if label:
            logger.debug("%s: %s", label, exc)


# Per-run remote artifact filenames. var_name matches the local that
# consumes it in run_eval_on_pod; basename is appended to run_dir.
# resume_pod_eval overrides per-key so re-attached runs keep writing
# to existing files even if basenames evolve.
_POD_REMOTE_FILES: tuple[tuple[str, str], ...] = (
    ("prompts_remote", "prompts.json"),
    ("remote_eval_script", "pod_eval.py"),
    ("progress_remote", "eval_progress.json"),
    ("results_remote", "eval_results.json"),
    ("done_marker_remote", "eval_done.marker"),
    ("log_remote", "eval_output.log"),
    ("eval_data_remote", "eval_data.json"),
    ("teacher_cache_remote", "teacher_cache.pt"),
    ("pid_remote", "pod_eval.pid"),
)


def _build_remote_paths(run_dir: str, resume: dict | None = None) -> dict[str, str]:
    """{var_name: remote_path} for one pod-eval run; resume overrides
    per-key for re-attached runs."""
    overrides = resume or {}
    return {
        var_name: overrides.get(var_name) or f"{run_dir}/{basename}"
        for var_name, basename in _POD_REMOTE_FILES
    }


_LOAD_STAGE_PREFIXES: tuple[str, ...] = (
    "loading_weights",
    "loading_student",
    "loading_teacher",
    "loading_model",
)


class StageStallWatchdog:
    """Kill a pod_eval pinned in the same stage past its limit.

    Fingerprint: (student_idx, student_name, stage, prompts_done,
    bench_axis_idx, teacher_prompts_done, phase). Any change resets the
    timer. Past half-limit -> warn_action fires once; past full limit
    -> kill_action fires once and the watchdog arms (no double-kill).
    """

    def __init__(
        self,
        *,
        load_timeout_s: int = 2700,
        default_timeout_s: int = 1500,
        kill_enabled: bool = True,
        warn_action=None,
        kill_action=None,
        time_fn=time.time,
    ) -> None:
        self.load_timeout_s = max(60, int(load_timeout_s))
        self.default_timeout_s = max(60, int(default_timeout_s))
        self.kill_enabled = bool(kill_enabled)
        self._warn_action = warn_action
        self._kill_action = kill_action
        self._time_fn = time_fn
        self.fingerprint: tuple | None = None
        self.since: float = float(time_fn())
        self.warned: bool = False
        self.killed: bool = False

    @staticmethod
    def _fingerprint_from(pod_progress: dict) -> tuple | None:
        if not isinstance(pod_progress, dict):
            return None
        cur = pod_progress.get("current")
        if not isinstance(cur, dict):
            return None
        return (
            cur.get("student_idx"),
            cur.get("student_name"),
            cur.get("stage"),
            cur.get("prompts_done"),
            cur.get("bench_axis_idx"),
            pod_progress.get("teacher_prompts_done"),
            pod_progress.get("phase"),
        )

    def _limit_for(self, stage: str | None) -> int:
        s = str(stage or "")
        for prefix in _LOAD_STAGE_PREFIXES:
            if s.startswith(prefix):
                return self.load_timeout_s
        return self.default_timeout_s

    def reset(self) -> None:
        self.fingerprint = None
        self.since = float(self._time_fn())
        self.warned = False

    def check(self, pod_progress: dict) -> str:
        """One of "ok" / "warn" / "killed". Idempotent once "killed"."""
        if self.killed:
            return "killed"
        fp = self._fingerprint_from(pod_progress)
        if fp is None:
            self.reset()
            return "ok"
        now = float(self._time_fn())
        if fp != self.fingerprint:
            self.fingerprint = fp
            self.since = now
            self.warned = False
            return "ok"
        cur = pod_progress.get("current") or {}
        stage = cur.get("stage")
        limit = self._limit_for(stage)
        elapsed = now - self.since
        # Hard limit always wins; the kill is the safety net.
        if elapsed >= limit:
            if self.kill_enabled and self._kill_action is not None:
                try:
                    self._kill_action(elapsed=elapsed, limit=limit, stage=stage, current=cur, pod_progress=pod_progress)
                except Exception:
                    pass
            self.killed = True
            return "killed"
        if not self.warned and elapsed >= limit / 2:
            self.warned = True
            if self._warn_action is not None:
                try:
                    self._warn_action(elapsed=elapsed, limit=limit, stage=stage, current=cur, pod_progress=pod_progress)
                except Exception:
                    pass
            return "warn"
        return "ok"


# Env vars propagated from validator to the pod inner_eval command.
# Pod-side semantics live in scripts/pod_eval_vllm.py.
_POD_EVAL_ENV_ALLOWLIST: tuple[str, ...] = (
    # Bench battery toggles + composite gates
    "BENCH_BATTERY_ENABLED",
    "BENCH_BATTERY_SHADOW_AXES",
    "BENCH_BATTERY_LITE",
    "POD_PER_MODEL_TIMEOUT",
    "ARENA_V3_AXES_IN_COMPOSITE",
    "REASONING_DENSITY_IN_COMPOSITE",
    "JUDGE_AXIS_IN_COMPOSITE",
    "CHAT_TURNS_AXIS_IN_COMPOSITE",
    "PARETO_DOMINANCE_GATE",
    "KING_REGRESSION_GATE",
    # Bench sample counts
    "BENCH_MATH_PER_ROUND",
    "BENCH_CODE_PER_ROUND",
    "BENCH_REASONING_PER_ROUND",
    "BENCH_KNOWLEDGE_PER_ROUND",
    "BENCH_IFEVAL_PER_ROUND",
    "BENCH_AIME_PER_ROUND",
    "BENCH_MBPP_PER_ROUND",
    "BENCH_TOOL_USE_PER_ROUND",
    "BENCH_SELF_CONSISTENCY_PER_ROUND",
    "BENCH_SELF_CONSISTENCY_SAMPLES",
    "BENCH_ARC_PER_ROUND",
    "BENCH_TRUTHFUL_PER_ROUND",
    "BENCH_LC_PER_ROUND",
    "BENCH_PROCEDURAL_PER_ROUND",
    "BENCH_ROBUSTNESS_PER_ROUND",
    "BENCH_ROBUSTNESS_PERTURB_K",
    "BENCH_NOISE_PER_ROUND",
    "BENCH_NOISE_PERTURB_K",
    # v31 procedural axes (policy-overridable for ablations/rollback)
    "BENCH_V31_GSM_SYMBOLIC_PER_ROUND",
    "BENCH_V31_GSM_SYMBOLIC_MAX_TOKENS",
    "BENCH_V31_MATH_COMPETITION_PER_ROUND",
    "BENCH_V31_MATH_COMPETITION_MAX_TOKENS",
    "BENCH_V31_MATH_ROBUSTNESS_PER_ROUND",
    "BENCH_V31_MATH_ROBUSTNESS_MAX_TOKENS",
    "BENCH_V31_CODE_PLUS_PER_ROUND",
    "BENCH_V31_CODE_PLUS_MAX_TOKENS",
    "BENCH_V31_LOGIC_GRID_PER_ROUND",
    "BENCH_V31_LOGIC_GRID_MAX_TOKENS",
    "BENCH_V31_DYVAL_PER_ROUND",
    "BENCH_V31_DYVAL_MAX_TOKENS",
    "BENCH_V31_RULER_PER_ROUND",
    "BENCH_V31_RULER_MAX_TOKENS",
    "BENCH_V31_KG_PER_ROUND",
    "BENCH_V31_KG_MAX_TOKENS",
    "BENCH_V31_IFEVAL_PER_ROUND",
    "BENCH_V31_IFEVAL_MAX_TOKENS",
    "BENCH_V31_TRUTHFULNESS_PER_ROUND",
    "BENCH_V31_TRUTHFULNESS_MAX_TOKENS",
    "BENCH_V31_CONSISTENCY_PER_ROUND",
    "BENCH_V31_CONSISTENCY_MAX_TOKENS",
    # Bench max-token budgets
    "BENCH_MATH_MAX_TOKENS",
    "BENCH_CODE_MAX_TOKENS",
    "BENCH_REASONING_MAX_TOKENS",
    "BENCH_KNOWLEDGE_MAX_TOKENS",
    "BENCH_IFEVAL_MAX_TOKENS",
    "BENCH_AIME_MAX_TOKENS",
    "BENCH_MBPP_MAX_TOKENS",
    "BENCH_TOOL_USE_MAX_TOKENS",
    "BENCH_SELF_CONSISTENCY_MAX_TOKENS",
    "BENCH_ARC_MAX_TOKENS",
    "BENCH_TRUTHFUL_MAX_TOKENS",
    "BENCH_LC_MAX_TOKENS",
    "BENCH_PROCEDURAL_MAX_TOKENS",
    "BENCH_ROBUSTNESS_MAX_TOKENS",
    "BENCH_NOISE_MAX_TOKENS",
    # Thinking-mode toggle (defaults ON in the pod; override via env
    # for emergency rollback)
    # without a code push. The pod-side default is also "1" so the
    # absence of this env var has the intended effect (thinking on)
    # without explicit propagation — listing it here is for the
    # disable-it-fast scenario only.
    "BENCH_ENABLE_THINKING",
    # ── Probe knobs ──
    "JUDGE_PROBE_PER_ROUND",
    "JUDGE_PROBE_MAX_TOKENS",
    "CHAT_TURNS_PROBE_PER_ROUND",
    "CHAT_TURNS_PROBE_MAX_TOKENS",
    "CHAT_TURNS_PROBE",
    "THINK_COLLAPSE_PROBE",
    # ── Cloud-API teacher path (2026-05-03 Kimi K2.6 cutover) ──
    # When DISTIL_TEACHER_MODE=api the pod skips local vLLM/HF teacher entirely
    # and fetches generation + top-K logprobs from an external OpenAI-compatible
    # provider. See scripts/api_teacher.py for design.
    "DISTIL_TEACHER_MODE",
    "DISTIL_TEACHER_API_BASE",
    "DISTIL_TEACHER_API_KEY",
    "OPENROUTER_API_KEY",  # accepted as alias inside api_teacher.from_env
    "DISTIL_TEACHER_API_MODEL",
    "DISTIL_TEACHER_API_ENDPOINT",
    "DISTIL_TEACHER_API_PROVIDERS",
    "DISTIL_TEACHER_API_CONCURRENCY",
    "DISTIL_TEACHER_API_TOP_LOGPROBS",
    "DISTIL_TEACHER_API_TIMEOUT_S",
    "DISTIL_TEACHER_API_DISABLE_REASONING",
    "DISTIL_OPENROUTER_REFERER",
    "DISTIL_OPENROUTER_TITLE",
    # ── Student-side knobs (2026-05-04) ──
    # ``DISTIL_STUDENT_BATCH_SIZE`` activates the v30.4 batched-forward path
    # (currently dormant unless set) which gives a 2-3x wall-time win on the
    # per-prompt KL loop. Without propagation the pod env never sees it and
    # the pod falls back to single-prompt forwards.
    "DISTIL_STUDENT_BATCH_SIZE",
    # ── Student vLLM (2026-05-14 8xB200 pivot) ──
    # Without these the pod silently falls back to the HF transformers
    # student path (~3 h/round). With ``DISTIL_STUDENT_USE_VLLM=1`` and
    # the tokenizer path the pod uses ``scripts/student_vllm.py``
    # (~30 min/round single GPU, much more under fan-out). Requires
    # the corresponding aux module entry in ``pod_runtime.py``.
    "DISTIL_STUDENT_USE_VLLM",
    "DISTIL_STUDENT_VLLM_TOKENIZER",
    "DISTIL_STUDENT_VLLM_MAX_LEN",
    "DISTIL_STUDENT_VLLM_GPU_UTIL",
    "DISTIL_STUDENT_VLLM_TP",
    "DISTIL_STUDENT_VLLM_TRC",
    "DISTIL_STUDENT_VLLM_EAGER",
    # vLLM 0.20.2 needs deep_gemm disabled for Kimi-K2.6 MoE on B200
    # (kernel selection bug). Re-enable when 0.21+ ships the fixed
    # selector.
    "VLLM_USE_DEEP_GEMM",
    "VLLM_USE_DEEP_GEMM_E8M0",
    # Vocab override for students compiled against Kimi K2.6 vocab=163840
    # rather than the older 160K. Pod precheck refuses without this.
    "ACTIVATION_FP_VOCAB_SIZE",
    "TEACHER_CONFIG_VOCAB_SIZE",
    # ── Parallel orchestrator (2026-05-15 8xB200 fan-out) ──
    # ``DISTIL_USE_PARALLEL_ORCH=1`` flips ``run_eval_on_pod`` over to
    # launching ``parallel_orchestrator.py`` instead of ``pod_eval.py``.
    # That orchestrator runs Phase 1 + king on GPU 0, then fans the
    # remaining challengers out across GPUs 1..N-1 using the same
    # pod_eval.py per-shard. Wall-clock improvement: 8 students on
    # 8xB200 drops from ~3 h sequential to ~25 min parallel.
    # ``DISTIL_PARALLEL_ORCH_GPUS`` overrides the default 8 — useful
    # for testing against a 2-GPU pod without code changes.
    # ``DISTIL_ORCH_WATCHDOG_S`` / ``..._REPEAT_N`` tune the
    # per-shard hang detector (silent + repeated-line patterns).
    # ``DISTIL_CACHE_KEEP_MODELS`` is set by the orchestrator itself
    # before each fan-out so the per-worker HF cache sweep preserves
    # sibling-shard models; we propagate it for chained orchestrator
    # invocations (e.g. retry workers) that share a workdir.
    "DISTIL_USE_PARALLEL_ORCH",
    "DISTIL_PARALLEL_ORCH_GPUS",
    "DISTIL_ORCH_WATCHDOG_S",
    "DISTIL_ORCH_WATCHDOG_REPEAT_N",
    "DISTIL_CACHE_KEEP_MODELS",
)


def run_eval_on_pod(pod: PodManager, models_to_eval: dict, king_uid, n_prompts: int, prompt_texts: list, state: ValidatorState, is_full_eval: bool, use_vllm: bool, eval_script: str, block_seed: int | None = None, resume_pod_eval: dict | None = None):
    """Drive a pod-side eval to completion, downloading results when done.

    Normally this clears any prior eval state, uploads the prompts + eval
    script, and starts a fresh detached run. Pass ``resume_pod_eval`` (a dict
    with ``run_dir``/``pid_remote``/``done_marker_remote``/``results_remote``/
    ``log_remote``/``eval_data_remote``/``progress_remote``/``started_at``)
    to skip cleanup + upload + start and instead poll the existing pod
    process to completion. The validator restart-and-resume path uses this
    so a mid-eval ``systemctl restart`` does not destroy the running pod
    eval (regression observed 2026-04-25 when the resume code re-entered
    the cleanup branch and lost ~75 min of student scoring).
    """
    import shutil
    import threading

    ordered_uids = []
    if king_uid is not None and king_uid in models_to_eval:
        ordered_uids.append(king_uid)
    # Order challengers by (failure_count asc, commit_block asc): proven
    # CUDA-poisoners go last so their cascade only torpedoes already-failed
    # models. Honest models still follow FIFO-by-commit.
    failures_state = getattr(state, "failures", {}) if state is not None else {}
    def _challenger_sort_key(uid):
        fc = int((failures_state or {}).get(str(uid), 0))
        cb = int((models_to_eval[uid] or {}).get("commit_block") or 0)
        return (fc, cb, int(uid))
    challenger_uids_sorted = sorted(
        [uid for uid in models_to_eval if uid != king_uid],
        key=_challenger_sort_key,
    )
    ordered_uids.extend(challenger_uids_sorted)
    now = time.time()
    is_resuming = isinstance(resume_pod_eval, dict) and bool(resume_pod_eval.get("run_dir"))
    if is_resuming:
        # Prefer the persisted started_at so the progress UI keeps the
        # original elapsed wall time.
        try:
            now = float(resume_pod_eval.get("started_at") or now)
        except (TypeError, ValueError):
            pass
    # Per-round wall-time estimate (calibrated on H200-NVL + Kimi K2.6
    # API teacher + bench battery + KL). Underestimating spikes
    # "eval is stuck" pings; padding is intentional.
    _api_mode = _is_api_teacher_mode()
    if _api_mode:
        try:
            api_concurrency = max(1, int(policy_env("DISTIL_TEACHER_API_CONCURRENCY", "8") or "8"))
        except (TypeError, ValueError):
            api_concurrency = 8
        try:
            teacher_max_new = int(policy_env("TEACHER_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS)) or MAX_NEW_TOKENS)
        except (TypeError, ValueError):
            teacher_max_new = MAX_NEW_TOKENS
        # API teacher wall time scales with prompt count / provider concurrency.
        per_prompt_s = 20.0 if teacher_max_new <= 768 else 26.0
        est_teacher_s = int((n_prompts * per_prompt_s) / api_concurrency + 300)
    else:
        est_teacher_s = 180
    # Probes + bench battery dominate -- ~12-16 min/student under the
    # LFJ derail cap with batched KL.
    est_per_student_s = 900
    est_total_s = est_teacher_s + est_per_student_s * len(models_to_eval)
    eval_order = []
    if king_uid is not None and king_uid in models_to_eval:
        eval_order.append({"uid": king_uid, "model": models_to_eval[king_uid]["model"], "role": "king"})
    for uid in challenger_uids_sorted:
        eval_order.append({"uid": uid, "model": models_to_eval[uid]["model"], "role": "challenger"})
    progress = {
        "active": True,
        "phase": "pod_bootstrap",
        "models": {str(uid): info["model"] for uid, info in models_to_eval.items()},
        "eval_order": eval_order,
        "students_total": len(models_to_eval),
        "students_done": 0,
        "prompts_total": n_prompts,
        "prompts_done": 0,
        "king_uid": king_uid,
        "challenger_uids": [uid for uid in models_to_eval if uid != king_uid],
        "started_at": now,
        "estimated_duration_s": est_total_s,
        "estimated_completion": now + est_total_s,
    }
    state.save_progress(progress)
    if is_resuming:
        run_dir = str(resume_pod_eval.get("run_dir"))
        _paths = _build_remote_paths(run_dir, resume=resume_pod_eval)
        logger.info(
            "run_eval_on_pod RESUME: attaching to existing pod eval "
            "run_dir=%s pid_remote=%s — skipping cleanup/upload/start.",
            run_dir, _paths["pid_remote"],
        )
    else:
        run_dir = f"/home/distil_eval_{int(now)}_{os.getpid()}"
        _paths = _build_remote_paths(run_dir)
    prompts_remote = _paths["prompts_remote"]
    remote_eval_script = _paths["remote_eval_script"]
    progress_remote = _paths["progress_remote"]
    results_remote = _paths["results_remote"]
    done_marker_remote = _paths["done_marker_remote"]
    log_remote = _paths["log_remote"]
    eval_data_remote = _paths["eval_data_remote"]
    teacher_cache_remote = _paths["teacher_cache_remote"]
    pid_remote = _paths["pid_remote"]
    if not is_resuming:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(prompt_texts, handle)
            handle.flush()
            os.fsync(handle.fileno())
            prompts_file = handle.name
        try:
            pod.exec(f"mkdir -p {shlex.quote(run_dir)}", timeout=30)
            pod.upload(prompts_file, prompts_remote, max_attempts=3)
        finally:
            os.unlink(prompts_file)
        pod.upload(eval_script, remote_eval_script, max_attempts=5)
        active_policy = policy_path()
        if active_policy and os.path.isfile(active_policy):
            try:
                pod.upload(active_policy, f"{run_dir}/eval_policy.json", max_attempts=3)
            except Exception as exc:
                logger.warning("Failed to upload eval policy (using env/defaults only): %s", exc)
    if not is_resuming:
        upload_aux_modules(pod, run_dir, logger)
    if is_resuming:
        # On resume, double-check the in-flight pod process actually still
        # exists (didn't crash while the validator was down) and that the
        # done marker hasn't already been written. If either guard fails,
        # bail out so the caller can decide to start fresh.
        def _retry_fresh(reason: str):
            logger.warning("Resume aborted (%s) — falling back to fresh run", reason)
            return run_eval_on_pod(
                pod, models_to_eval, king_uid, n_prompts, prompt_texts,
                state, is_full_eval, use_vllm, eval_script,
                block_seed=block_seed, resume_pod_eval=None,
            )

        try:
            pid_probe = pod.exec(
                f"if [ -f {shlex.quote(done_marker_remote)} ]; then echo DONE; "
                f"elif [ -f {shlex.quote(pid_remote)} ] && kill -0 \"$(cat {shlex.quote(pid_remote)})\" 2>/dev/null; then echo ALIVE; "
                "else echo MISSING; fi",
                timeout=30,
            )
            pid_status = (pid_probe.get("stdout") or "").strip()
        except Exception as exc:
            return _retry_fresh(f"resume probe failed: {exc}")
        if pid_status not in ("ALIVE", "DONE"):
            return _retry_fresh(
                f"resume target pid_remote={pid_remote} reports status={pid_status!r} (not ALIVE/DONE)"
            )
        logger.info("Resume probe: pod eval is %s — skipping cleanup, attaching to existing process.", pid_status)
    if not is_resuming:
        try:
            # vLLM v1 renames its EngineCore child via PR_SET_NAME, so a
            # plain pkill -f 'vllm.entrypoints' misses it. Kill by comm
            # ("VLLM::EngineCor", 15-char cap), cmdline, then nvidia-smi
            # as a last resort.
            pod.exec(
                # Preserve chat-king PIDs (sn97-king on :8100 + supervisor)
                # so chat stays up across the cleanup. Strategy: among any
                # ``--served-model-name sn97-king`` vLLM workers, keep the
                # YOUNGEST (oldest are SO_REUSEPORT-bound zombies from
                # prior chat-server restarts each squatting ~22 GB VRAM)
                # plus its descendants and the chat_server.py supervisor.
                # Fall back to broad preserve only if no chat-king is
                # alive (chat is starting up or genuinely dark).
                "preserve=''; "
                "all_chat_pids=$(ps auxww 2>/dev/null | grep 'served-model-name sn97-king' | grep -v grep | awk '{print $2}' | sort -u); "
                "if [ -n \"$all_chat_pids\" ]; then "
                # Find the youngest by smallest etimes (elapsed seconds).
                "  youngest=''; youngest_etimes=999999999; "
                "  for pid in $all_chat_pids; do "
                "    et=$(ps -p $pid -o etimes= 2>/dev/null | tr -d ' '); "
                "    if [ -n \"$et\" ] && [ \"$et\" -lt \"$youngest_etimes\" ]; then "
                "      youngest_etimes=$et; youngest=$pid; "
                "    fi; "
                "  done; "
                "  if [ -n \"$youngest\" ]; then "
                "    preserve=\"$youngest\"; "
                "    for desc in $(pgrep -P $youngest 2>/dev/null); do "
                "      preserve=\"$preserve $desc\"; "
                "      for gdesc in $(pgrep -P $desc 2>/dev/null); do preserve=\"$preserve $gdesc\"; done; "
                "    done; "
                "    echo \"[cleanup] preserving youngest chat-king PID=$youngest (etimes=$youngest_etimes); will kill any older sn97-king vllms\" >&2; "
                "  fi; "
                # also keep the chat_server.py supervisor if any (it's
                # a thin wrapper that spawns the vLLM)
                "  for pid in $(pgrep -f 'chat_server.py' 2>/dev/null); do "
                "    preserve=\"$preserve $pid\"; "
                "    for desc in $(pgrep -P $pid 2>/dev/null); do preserve=\"$preserve $desc\"; done; "
                "  done; "
                "fi; "
                # If no chat-king found (chat is genuinely dark), fall
                # back to broad include-all so we don't kill a
                # starting/restarting one mid-bootstrap.
                "if [ -z \"$preserve\" ]; then "
                "  for pid in $(pgrep -f 'chat_server.py' 2>/dev/null) "
                "               $(pgrep -f 'served-model-name sn97-king' 2>/dev/null) "
                "               $(pgrep -f 'port 8100' 2>/dev/null); do "
                "    preserve=\"$preserve $pid\"; "
                "    for desc in $(pgrep -P $pid 2>/dev/null); do "
                "      preserve=\"$preserve $desc\"; "
                "      for gdesc in $(pgrep -P $desc 2>/dev/null); do preserve=\"$preserve $gdesc\"; done; "
                "    done; "
                "  done; "
                "fi; "
                "kill_unless_chat() { "
                "  for pid in \"$@\"; do "
                "    case \" $preserve \" in (*\" $pid \"*) ;; (*) kill -9 $pid 2>/dev/null ;; esac; "
                "  done; "
                "}; "
                "kill_unless_chat $(pgrep -f pod_eval 2>/dev/null); "
                "kill_unless_chat $(pgrep -f 'vllm.entrypoints' 2>/dev/null); "
                "kill_unless_chat $(pgrep -f 'VllmWorker' 2>/dev/null); "
                "kill_unless_chat $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); "
                "kill_unless_chat $(pgrep -x 'VLLM::EngineCor' 2>/dev/null); "  # match comm (15-char trunc)
                "sleep 2; "
                "for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
                "  case \" $preserve \" in (*\" $pid \"*) continue ;; esac; "
                "  comm=$(cat /proc/$pid/comm 2>/dev/null); "
                "  cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null); "
                "  case \"$cmd$comm\" in "
                "    *vllm.entrypoints*|*VllmWorker*|*pod_eval*|*VLLM::EngineCor*) "
                "      kill -9 $pid 2>/dev/null ;; "
                "  esac; "
                "done; "
                "sleep 2; "
                # Last-resort sweep: if GPU is still non-empty, something slipped
                # through. Log the survivors so we can improve the patterns above.
                # Survivors that match the preserve list are NOT killed — that
                # would dark chat.arbos.life again.
                "survivors=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); "
                "if [ -n \"$survivors\" ]; then "
                "  for pid in $survivors; do "
                "    case \" $preserve \" in (*\" $pid \"*) continue ;; esac; "
                "    echo \"[cleanup] gpu still held by pid=$pid comm=$(cat /proc/$pid/comm 2>/dev/null) cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | head -c 200)\" >&2; "
                "    kill -9 $pid 2>/dev/null; "
                "  done; "
                "  sleep 2; "
                "fi; "
                # /dev/shm/vllm* is per-process; only safe to wipe when no
                # vllm process survives. Skip if chat-king still holds a slot.
                "if [ -z \"$preserve\" ]; then rm -rf /dev/shm/vllm* 2>/dev/null; fi; "
                "rm -f /home/pod_eval.py /home/prompts.json /home/eval_output.log /home/eval_results.json /home/eval_progress.json /home/teacher_cache.pt 2>/dev/null; "
                "sleep 1",
                timeout=40,
            )
            logger.info("Killed existing eval/vllm processes (chat-king preserved) and removed stale /home files")
        except Exception as exc:
            logger.debug(f"Pre-eval cleanup: {exc}")
        _pod_exec_silent(
            pod,
            f"rm -f {shlex.quote(progress_remote)} {shlex.quote(results_remote)} {shlex.quote(log_remote)} "
            f"{shlex.quote(eval_data_remote)} {shlex.quote(teacher_cache_remote)} {shlex.quote(pid_remote)} "
            f"{shlex.quote(done_marker_remote)}",
            label="clear pod artifacts",
        )
        logger.info("Cleared all pod artifacts (eval_results, teacher_cache, progress)")
        try:
            disk_pct = pod.disk_cleanup(TEACHER_MODEL)
            if disk_pct is not None:
                log_event(f"Pod disk: {disk_pct}% used after cleanup", state_dir=str(state.state_dir))
        except Exception as exc:
            log_event(f"Pod disk cleanup failed: {str(exc)[:100]}", level="warn", state_dir=str(state.state_dir))
        pod.clear_gpu()
        try:
            verify = pod.exec("pgrep -af pod_eval 2>/dev/null; ls /home/pod_eval.py 2>/dev/null; echo VERIFY_DONE", timeout=15)
            vout = verify.get("stdout", "")
            if "/home/pod_eval.py" in vout or ("pod_eval" in vout and "VERIFY_DONE" in vout and vout.strip() != "VERIFY_DONE"):
                logger.warning("Stale competing eval detected after cleanup, killing again")
                pod.exec("pkill -9 -f pod_eval 2>/dev/null; rm -f /home/pod_eval.py 2>/dev/null; sleep 2", timeout=20)
        except Exception:
            pass
    student_list = ",".join(models_to_eval[uid]["model"] for uid in ordered_uids)
    revision_list = ",".join(models_to_eval[uid].get("revision", "main") for uid in ordered_uids)
    king_flag = ""
    vllm_flag = " --no-vllm"
    # API-teacher mode takes precedence over local vLLM.
    _api_mode = _is_api_teacher_mode()
    if use_vllm and not _api_mode:
        # GPU is co-tenant with chat-king vLLM; default 0.65 leaves ~4.6 GiB
        # headroom for the ~44 GiB chat occupancy. Bump to 0.92 once chat
        # moves to a dedicated pod.
        eval_gpu_util = policy_env("VLLM_EVAL_GPU_UTIL", "0.65")
        vllm_flag = f" --vllm-gpu-util {eval_gpu_util}"
        if not is_full_eval and king_uid is not None and king_uid in models_to_eval:
            king_flag = f" --king {models_to_eval[king_uid]['model']}"
    tp_flag = f" --tensor-parallel-size {TP_SIZE}" if TP_SIZE > 0 else ""
    early_stop_flag = f" --early-stop-min {EARLY_STOP_MIN}" if EARLY_STOP_MIN > 0 else ""
    # 2026-05-15: parallel orchestrator wiring. When DISTIL_USE_PARALLEL_ORCH=1
    # (and the pod actually has ≥2 GPUs) we launch parallel_orchestrator.py
    # instead of pod_eval.py directly. The orchestrator runs Phase 1
    # (teacher API gen + king scoring) on GPU 0, then fans the remaining
    # challengers across GPUs 1..N-1. Each shard runs the same pod_eval.py
    # in single-GPU mode under CUDA_VISIBLE_DEVICES=k, so all the
    # vLLM/probe/disk-sweep code paths are reused unchanged.
    #
    # The orchestrator writes an aggregated eval_progress.json to the
    # run_dir (same path the validator already polls) with a ``shards``
    # array; the dashboard fans that out into a per-GPU view. King-mode
    # (--king) and resume aren't needed here because the orchestrator
    # always scores the king first as the canonical "shard 0".
    _use_parallel = (
        (policy_env("DISTIL_USE_PARALLEL_ORCH", "0") or "0").strip().lower()
        not in ("0", "false", "no", "off", "")
    )
    try:
        _orch_gpus = int(policy_env("DISTIL_PARALLEL_ORCH_GPUS", "0") or "0")
    except (TypeError, ValueError):
        _orch_gpus = 0
    if _use_parallel and king_uid is not None and king_uid in models_to_eval:
        king_model = models_to_eval[king_uid]["model"]
        king_rev = models_to_eval[king_uid].get("revision", "main")
        challenger_models: list[str] = []
        challenger_revs: list[str] = []
        for uid in ordered_uids:
            if uid == king_uid:
                continue
            m = models_to_eval[uid]
            challenger_models.append(m["model"])
            challenger_revs.append(m.get("revision", "main"))
        gpus_flag = f" --gpus {_orch_gpus}" if _orch_gpus > 0 else ""
        inner_eval = (
            f"cd {shlex.quote(run_dir)} && python3 -u parallel_orchestrator.py "
            f"--workdir {shlex.quote(run_dir)} "
            f"--prompts {shlex.quote(prompts_remote)} "
            f"--teacher-cache {shlex.quote(teacher_cache_remote)} "
            f"--out {shlex.quote(results_remote)} "
            f"--unified-progress {shlex.quote(progress_remote)} "
            f"--king-model {shlex.quote(king_model)} "
            f"--king-revision {shlex.quote(king_rev)} "
            f"--students {shlex.quote(','.join(challenger_models))} "
            f"--revisions {shlex.quote(','.join(challenger_revs))}"
            f"{gpus_flag}"
        )
    else:
        inner_eval = (
            f"cd {shlex.quote(run_dir)} && python3 -u {shlex.quote(remote_eval_script)} "
            f"--teacher {TEACHER_MODEL} "
            f"--students {student_list} "
            f"--revisions {revision_list} "
            f"--prompts {shlex.quote(prompts_remote)} "
            f"--output {shlex.quote(results_remote)} "
            f"--max-new-tokens {MAX_NEW_TOKENS} "
            f"--concurrency {VLLM_CONCURRENCY} "
            f"--teacher-logits {shlex.quote(teacher_cache_remote)}"
            f"{tp_flag}"
            f"{early_stop_flag}"
            f"{king_flag}"
            f"{vllm_flag}"
            f"{f' --block-seed {block_seed}' if block_seed is not None else ''}"
        )
    inner_q = shlex.quote(inner_eval)
    wrapped = (
        f"{{ echo $$ > {shlex.quote(pid_remote)}; "
        f"exec bash -c {inner_q}; }}"
    )
    start_cmd = (
        f"rm -f {shlex.quote(pid_remote)} && : > {shlex.quote(log_remote)} && "
        f"nohup bash -c {shlex.quote(wrapped)} >> {shlex.quote(log_remote)} 2>&1 & "
        f"disown; "
        f"for _ in 1 2 3 4 5 6 7 8 9 10; do "
        f"  [ -s {shlex.quote(pid_remote)} ] && break; sleep 0.2; "
        f"done; "
        f"echo DISTIL_PID:$(cat {shlex.quote(pid_remote)} 2>/dev/null)"
    )
    # Zombie-aware status: ``kill -0`` returns 0 for zombies, so use
    # ``ps -o stat=`` to flag 'Z'/'X' as DISTIL_STATUS:dead. Falls back
    # to ``kill -0`` when /proc isn't mounted.
    status_inner = (
        f"if [ -f {shlex.quote(done_marker_remote)} ]; then echo DISTIL_STATUS:done; "
        f"elif [ ! -f {shlex.quote(pid_remote)} ]; then echo DISTIL_STATUS:starting; "
        f"elif ! kill -0 \"$(cat {shlex.quote(pid_remote)})\" 2>/dev/null; then echo DISTIL_STATUS:dead; "
        f"else _stat=$(ps -p \"$(cat {shlex.quote(pid_remote)})\" -o stat= 2>/dev/null | tr -d ' '); "
        f"  case \"$_stat\" in "
        f"    Z*|X*) echo DISTIL_STATUS:dead ;; "
        f"    *) echo DISTIL_STATUS:running ;; "
        f"  esac; "
        f"fi"
    )
    status_cmd = f"bash -lc {shlex.quote(status_inner)}"
    poll_stop = threading.Event()
    gpu_log_path = state.state_dir / "gpu_eval.log"
    gpu_log_path.write_text("")

    # Progress-poll diagnostics. A zero-byte or stale eval_progress.json was
    # swallowed silently here for months, and the only user-visible symptom
    # was the dashboard stuck on "Loading teacher model…" for an entire
    # round. Log the first failure (per state) so operators can tell the
    # difference between "pod hasn't written yet" (expected early) vs
    # "pod writer is broken" (which is what the 2026-04-20 round looked
    # like — see scripts/pod_eval_vllm.py::_atomic_json_write for the
    # matching writer-side fix).
    _poll_last_err: dict = {"kind": None, "at": 0.0}

    def _poll_log_err(kind: str, msg: str):
        now = time.time()
        if _poll_last_err["kind"] == kind and (now - _poll_last_err["at"]) < 300:
            return
        _poll_last_err["kind"] = kind
        _poll_last_err["at"] = now
        logger.info("pod progress poll: %s — %s", kind, msg[:200])

    # 2026-05-08 — stage-stall watchdog. The May 7 incident sat in
    # ``loading_weights`` for ~4h22m on UID 213 ``const0312/wtbmts09``
    # before the outer loop noticed and replanned. The watchdog catches
    # the same shape inside ``DISTIL_STAGE_STALL_LOAD_S`` (default 45 min).
    # Detection logic is in ``StageStallWatchdog`` at module scope so it
    # can be unit-tested without spinning up a real pod loop.
    try:
        _STAGE_STALL_LOAD_S = int(policy_env("DISTIL_STAGE_STALL_LOAD_S", "2700") or "2700")
    except (TypeError, ValueError):
        _STAGE_STALL_LOAD_S = 2700
    try:
        _STAGE_STALL_DEFAULT_S = int(policy_env("DISTIL_STAGE_STALL_DEFAULT_S", "1500") or "1500")
    except (TypeError, ValueError):
        _STAGE_STALL_DEFAULT_S = 1500
    _STAGE_STALL_KILL = (policy_env("DISTIL_STAGE_STALL_KILL", "1") or "1").strip().lower() not in ("0", "false", "no", "off")

    def _stall_warn(*, elapsed, limit, stage, current, **_):
        is_loading = any(str(stage or "").startswith(p) for p in _LOAD_STAGE_PREFIXES)
        logger.warning(
            "pod progress stalled: student=%r stage=%r prompts_done=%r idx=%r "
            "for %.0fs (limit %ds, %s) — will force recovery if it doesn't progress",
            current.get("student_name"), stage, current.get("prompts_done"),
            current.get("student_idx"), elapsed, limit,
            "loading" if is_loading else "scoring",
        )

    def _stall_kill(*, elapsed, limit, stage, current, **_):
        logger.error(
            "pod progress stalled past hard limit: student=%r stage=%r elapsed=%.0fs "
            "limit=%ds — killing pod_eval to trigger DISTIL_STATUS:dead and let "
            "the outer loop replan",
            current.get("student_name"), stage, elapsed, limit,
        )
        # Be surgical: kill ONLY pod_eval.py inside this run dir. The
        # chat-king vLLM lives in a different path and must survive.
        kill_cmd = (
            "set -e; "
            f"PID=$(cat {shlex.quote(pid_remote)} 2>/dev/null || true); "
            "if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then "
            "  kill -9 \"$PID\" 2>/dev/null && echo killed:$PID || echo kill_failed:$PID; "
            "else "
            f"  pkill -9 -f {shlex.quote(run_dir)}/pod_eval.py 2>/dev/null && echo killed_by_pattern || echo no_match; "
            "fi"
        )
        _pod_exec_silent(pod, f"bash -lc {shlex.quote(kill_cmd)}", timeout=20, label="stage_stall_kill")
        try:
            log_event(
                f"pod progress stall watchdog killed pod_eval after {int(elapsed)}s "
                f"in stage={stage!r} student={current.get('student_name')!r}",
                state_dir=str(state.state_dir),
            )
        except Exception:
            pass

    _stall_watchdog = StageStallWatchdog(
        load_timeout_s=_STAGE_STALL_LOAD_S,
        default_timeout_s=_STAGE_STALL_DEFAULT_S,
        kill_enabled=_STAGE_STALL_KILL,
        warn_action=_stall_warn,
        kill_action=_stall_kill,
    )

    def poll_pod_progress():
        while not poll_stop.is_set():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                    tmp_path = handle.name
                try:
                    pod.download(progress_remote, tmp_path)
                except Exception as exc:
                    _poll_log_err("download_failed", f"{type(exc).__name__}: {exc}")
                    raise
                try:
                    size = os.path.getsize(tmp_path)
                except OSError:
                    size = -1
                if size <= 0:
                    _poll_log_err(
                        "empty_progress_file",
                        f"remote {progress_remote} downloaded as {size}-byte file — "
                        f"pod writer may have truncated. Holding last known progress.",
                    )
                    raise ValueError(f"zero-byte progress file (size={size})")
                with open(tmp_path) as handle:
                    try:
                        pod_progress = json.load(handle)
                    except json.JSONDecodeError as exc:
                        with open(tmp_path) as _h:
                            head = _h.read(200)
                        _poll_log_err(
                            "bad_progress_json",
                            f"parse error: {exc}; file head: {head!r}",
                        )
                        raise
                pod_phase = pod_progress.get("phase", "scoring")
                progress["phase"] = pod_phase
                progress["pod"] = pod_progress
                if pod_progress.get("current"):
                    current = pod_progress["current"]
                    progress.update({
                        out_key: current.get(in_key, default)
                        for out_key, in_key, default in CURRENT_FIELD_MAP
                    })
                else:
                    for out_key, _, _ in CURRENT_FIELD_MAP:
                        progress.pop(out_key, None)
                # Always propagate teacher_prompts_done so the dashboard's
                # Phase A progress bar fills correctly even after we transition
                # into student loading (the previous gate dropped the value
                # the moment loading_student began, leaving the bar at 0).
                progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)
                for key in POD_PROGRESS_METADATA_KEYS:
                    if key in pod_progress:
                        progress[key] = pod_progress[key]
                pod_completed = pod_progress.get("completed", [])
                progress["completed"] = pod_completed
                progress["students_done"] = len(pod_completed)
                state.save_progress(progress)
                if _poll_last_err["kind"] is not None:
                    logger.info("pod progress poll: recovered (last error: %s)", _poll_last_err["kind"])
                    _poll_last_err["kind"] = None
                    _poll_last_err["at"] = 0.0
                _stall_watchdog.check(pod_progress)
            except Exception as exc:
                if _poll_last_err["kind"] is None:
                    _poll_log_err("unclassified", f"{type(exc).__name__}: {exc}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            try:
                log_result = pod.exec(f"tail -100 {shlex.quote(log_remote)} 2>/dev/null || echo ''", timeout=30)
                log_text = log_result.get("stdout", "")
                if log_text.strip():
                    gpu_log_path.write_text(sanitize_gpu_log(log_text))
            except Exception:
                pass
            _pod_exec_silent(
                pod,
                "for p in $(pgrep -f 'pod_eval' 2>/dev/null); do "
                "  cmdline=$(cat /proc/$p/cmdline 2>/dev/null | tr '\\0' ' '); "
                "  case \"$cmdline\" in *distil_eval_*) ;; *) kill -9 $p 2>/dev/null;; esac; "
                "done; "
                "rm -f /home/pod_eval.py /home/prompts.json /home/pod_eval_vllm.py 2>/dev/null",
                timeout=15,
            )
            poll_stop.wait(15)

    poll_thread = threading.Thread(target=poll_pod_progress, daemon=True)
    poll_thread.start()
    n_eval_models = len(models_to_eval)
    try:
        eval_timeout = int(policy_env("DISTIL_POD_EVAL_TIMEOUT_S", str(8 * 3600)) or str(8 * 3600))
    except (TypeError, ValueError):
        eval_timeout = 8 * 3600
    logger.info(f"Running eval ({n_eval_models} models, {n_prompts} prompts, timeout={eval_timeout // 60}m)")
    log_event(f"Running eval on pod: king vs {n_eval_models - 1} challengers, {n_prompts} prompts", state_dir=str(state.state_dir))
    eval_env = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "TOKENIZERS_PARALLELISM": "false",
        "ACTIVATION_FP_VOCAB_SIZE": str(TEACHER_CONFIG_VOCAB_SIZE),
        "TEACHER_CONFIG_VOCAB_SIZE": str(TEACHER_CONFIG_VOCAB_SIZE),
    }
    # 2026-04-30: enable the Rust-based hf_transfer downloader on the pod.
    # Saturates the network link (>500MB/s typical) instead of CPython's
    # 50-100MB/s default. The package was just installed on the pod
    # (see /home/distil/.secrets/distil.env note); enabling via env is
    # all that's needed. Saves ~30s/round on student model downloads
    # (10 students × ~8GB × 80MB/s = 1000s; with hf_transfer 10 × 8GB ×
    # 500MB/s = 160s). Tunable via DISTIL_HF_TRANSFER=0 to disable.
    if policy_env("DISTIL_HF_TRANSFER", "1") == "1":
        eval_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # 2026-05-01 (v30.4): enable hf_xet HIGH_PERFORMANCE mode. The
    # Rust-based XET (Content-Addressable Storage) downloader is the
    # default for huggingface_hub 1.12+ on XET-enabled repos (all
    # current Qwen models). HIGH_PERFORMANCE=1 uses more RAM and CPU
    # cores per download but pushes throughput closer to the link
    # ceiling. The eval pod (256GB RAM, 24+ cores) has the slack —
    # tunable via DISTIL_HF_XET_HIPERF=0 to disable.
    if policy_env("DISTIL_HF_XET_HIPERF", "1") == "1":
        eval_env["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Bump default 10s timeout — fine for tiny config.json fetches
    # but flaky on multi-GB safetensors when the link is saturated
    # by parallel requests. 300s gives slow miners a chance without
    # hanging forever (the per-round timeout is the outer cap).
    eval_env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    eval_env.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # 2026-04-24 (distil-97, leeroyjkin): the heavy bench battery (Session 3
    # shadow axes: aime/mbpp/tool_use/self_consistency/arc/truthful/long_context)
    # adds ~6 min/student (~84 min/round for 14 students). Propagate the
    # tunables from validator env so we can flip them via systemd override
    # without redeploying code. The full list lives at module scope as
    # ``_POD_EVAL_ENV_ALLOWLIST``; see ``scripts/pod_eval_vllm.py`` for the
    # semantics of each variable.
    eval_env.update(eval_policy_env())
    if not is_resuming:
        eval_env["DISTIL_EVAL_POLICY"] = f"{run_dir}/eval_policy.json"
    for _propagate in _POD_EVAL_ENV_ALLOWLIST:
        _v = policy_env(_propagate)
        if _v is not None:
            eval_env[_propagate] = _v
    try:
        if is_resuming:
            logger.info(
                "RESUME: skipping start_cmd; attaching to existing pod eval at %s "
                "(pid_remote=%s)", run_dir, pid_remote,
            )
        else:
            start_res = pod.exec(start_cmd, env=eval_env, timeout=120)
            if not start_res.get("success"):
                logger.error("Failed to start detached eval: %s", start_res)
                return None
            logger.info("Detached GPU eval started: %s", (start_res.get("stdout") or "").strip()[:200])
        # 2026-04-24 (distil-97): persist pod paths so `_detect_resumable_round`
        # can reattach after a validator restart mid-eval. Previously the
        # `current_round.json.pod_eval` block was never populated, so a
        # restart during eval tossed the in-flight round and forced
        # re-planning (discarding partial results, corrupting state).
        try:
            if isinstance(state.current_round, dict):
                state.current_round["pod_eval"] = {
                    "run_dir": run_dir,
                    "pid_remote": pid_remote,
                    "done_marker_remote": done_marker_remote,
                    "log_remote": log_remote,
                    "progress_remote": progress_remote,
                    "results_remote": results_remote,
                    "eval_data_remote": eval_data_remote,
                    "started_at": now,
                }
                state.save_round()
        except Exception as _save_exc:
            logger.warning("Failed to persist pod_eval meta (non-fatal): %s", _save_exc)
        result = {"stdout": "", "stderr": "", "exit_code": -1, "success": False}
        dead_streak = 0
        starting_streak = 0
        deadline = time.time() + eval_timeout
        eval_started_at = time.time()
        while time.time() < deadline:
            status = pod.exec(status_cmd, timeout=90)
            out = status.get("stdout", "") or ""
            if "DISTIL_STATUS:done" in out:
                result = {"stdout": "", "stderr": "", "exit_code": 0, "success": True}
                break
            if "DISTIL_STATUS:dead" in out:
                dead_streak += 1
                starting_streak = 0
                if dead_streak >= 3:
                    logger.error("Eval worker on pod exited before writing eval_results.json")
                    result = {"stdout": "", "stderr": "worker_dead", "exit_code": -1, "success": False}
                    break
            elif "DISTIL_STATUS:starting" in out:
                dead_streak = 0
                starting_streak += 1
                if starting_streak >= 15 and (time.time() - eval_started_at) > 180:
                    probe = pod.exec(
                        f"tail -40 {shlex.quote(log_remote)} 2>/dev/null | grep -i -E 'traceback|error' | head -5",
                        timeout=30,
                    )
                    probe_out = (probe.get("stdout", "") or "").strip()
                    if probe_out:
                        logger.error(
                            "Eval stuck in 'starting' for >%ds; log shows: %s",
                            int(time.time() - eval_started_at), probe_out[:300],
                        )
                        result = {
                            "stdout": "", "stderr": f"worker_stuck_starting: {probe_out[:200]}",
                            "exit_code": -1, "success": False,
                        }
                        break
            else:
                dead_streak = 0
                starting_streak = 0
            time.sleep(20)
        else:
            logger.error(f"Eval timed out after {eval_timeout}s — killing")
            _pod_exec_silent(pod, "pkill -9 -f pod_eval.py; echo killed", timeout=30)
            try:
                pod.reconnect()
            except Exception as exc:
                logger.error(f"Reconnect failed after timeout: {exc}")
            result = {"stdout": "", "stderr": "timeout", "exit_code": -1, "success": False}
    except Exception as exc:
        logger.error(f"lium.eval EXCEPTION: {exc}")
        import traceback

        traceback.print_exc()
        try:
            pod.reconnect()
        except Exception:
            pass
        return None
    finally:
        poll_stop.set()
        poll_thread.join(timeout=5)

    def capture_failure_context(reason: str):
        try:
            snapshot = pod.exec(
                "echo '== files ==' && "
                f"ls -l {shlex.quote(results_remote)} {shlex.quote(progress_remote)} {shlex.quote(log_remote)} 2>/dev/null || true && "
                "echo '== ps ==' && "
                "pgrep -af \"pod_eval.py|vllm.entrypoints.openai.api_server\" || true && "
                "echo '== tail ==' && "
                f"tail -120 {shlex.quote(log_remote)} 2>/dev/null || true",
                timeout=60,
            )
            snapshot_text = sanitize_gpu_log((snapshot.get("stdout", "") or "") + "\n" + (snapshot.get("stderr", "") or "")).strip()
            if snapshot_text:
                gpu_log_path.write_text(snapshot_text)
            log_event(f"{reason}: exit_code={result.get('exit_code')} success={result.get('success')}", level="error", state_dir=str(state.state_dir))
        except Exception as exc:
            logger.warning(f"Failed to capture pod failure context: {exc}")

    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    logger.info("Pod exec finished: exit_code=%s success=%s", result.get("exit_code"), result.get("success"))
    if stdout.strip():
        for line in stdout.strip().split("\n")[-30:]:
            logger.info(f"  GPU: {line[:200]}")
    if stderr.strip():
        for line in stderr.strip().split("\n")[-10:]:
            logger.warning(f"  GPU ERR: {line[:200]}")
    results_local = str(state.state_dir / "last_eval.json")
    try:
        pod.download(results_remote, results_local)
    except Exception as exc:
        logger.error("Failed to download results: %s", exc)
        capture_failure_context("Failed to download eval results from pod")
        log_event("Failed to download eval results from pod", level="error", state_dir=str(state.state_dir))
        state.save_progress({"active": False, "failed": True, "stage": "results_download"})
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    try:
        logs_dir = state.state_dir / "pod_logs"
        logs_dir.mkdir(exist_ok=True)
        log_dest = str(logs_dir / f"eval_{ts}.log")
        pod.download(log_remote, log_dest)
        for old in sorted(logs_dir.glob("eval_*.log"))[:-20]:
            old.unlink(missing_ok=True)
        logger.info(f"Pod eval log saved: {log_dest}")
    except Exception as exc:
        logger.warning(f"Pod log retrieval failed (non-fatal): {exc}")
    try:
        eval_data_dir = state.state_dir / "eval_data"
        eval_data_dir.mkdir(exist_ok=True)
        # state/eval_data is API-served; state/eval_data_private is validator-only
        # raw copy used for offline audit.
        eval_data_private_dir = state.state_dir / "eval_data_private"
        eval_data_private_dir.mkdir(exist_ok=True)
        raw_dest = str(eval_data_private_dir / f"eval_data_{ts}.json")
        pod.download(eval_data_remote, raw_dest)
        public_dest = str(eval_data_dir / f"eval_data_{ts}.json")
        try:
            cur = state.current_round if isinstance(state.current_round, dict) else {}
            pp = cur.get("private_pool") or {}
            n_private = int(pp.get("n", 0) or 0)
            with open(raw_dest) as fh:
                raw = json.load(fh)
            if n_private > 0 and isinstance(raw, dict) and isinstance(raw.get("data"), list):
                rows = raw["data"]
                cutoff = max(0, len(rows) - n_private)
                redacted_rows = []
                for i, r in enumerate(rows):
                    if i >= cutoff and isinstance(r, dict):
                        redacted_rows.append({**r, "prompt": "[PRIVATE]",
                                              "continuation": "[PRIVATE]",
                                              "is_private": True})
                    else:
                        redacted_rows.append(r)
                raw_pub = {**raw, "data": redacted_rows, "private_redacted": n_private}
            else:
                raw_pub = raw
            with open(public_dest, "w") as fh:
                json.dump(raw_pub, fh)
            shutil.copy2(public_dest, str(state.state_dir / "eval_data_latest.json"))
        except Exception as exc:
            logger.warning(f"Private-pool redaction failed, copying raw (non-fatal): {exc}")
            shutil.copy2(raw_dest, public_dest)
            shutil.copy2(public_dest, str(state.state_dir / "eval_data_latest.json"))
        for old in sorted(eval_data_dir.glob("eval_data_*.json"))[:-10]:
            old.unlink(missing_ok=True)
        for old in sorted(eval_data_private_dir.glob("eval_data_*.json"))[:-30]:
            old.unlink(missing_ok=True)
        logger.info(f"Eval data saved: public={public_dest} raw={raw_dest}")
    except Exception as exc:
        logger.warning(f"Eval data retrieval failed (non-fatal): {exc}")
    try:
        with open(results_local) as handle:
            results = json.load(handle)
        n_students = len(results.get("students", {}))
        if n_students == 0 and not result.get("success", False):
            logger.error("Eval failed, no usable results")
            log_event("Eval failed: no usable results were recovered", level="error", state_dir=str(state.state_dir))
            state.save_progress({"active": False, "failed": True, "stage": "results_empty"})
            return None
        if not result.get("success", False):
            logger.warning(f"Eval failed but recovered {n_students} partial results")
    except Exception:
        logger.error("Results file corrupt")
        capture_failure_context("Eval results file corrupt or empty")
        log_event("Eval results file corrupt or empty", level="error", state_dir=str(state.state_dir))
        try:
            os.unlink(results_local)
        except Exception:
            pass
        state.save_progress({"active": False, "failed": True, "stage": "results_corrupt"})
        return None
    return results
