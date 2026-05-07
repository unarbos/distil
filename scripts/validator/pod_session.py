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

# Opt-in teacher tensor-parallel size. 0 = let pod autodetect from torch.cuda.device_count().
TP_SIZE = int(policy_env("DISTIL_TP_SIZE", "0") or "0")
# Same-point early-stop floor; 0 disables (matches legacy behaviour).
EARLY_STOP_MIN = int(policy_env("DISTIL_EARLY_STOP_MIN", "0") or "0")

logger = logging.getLogger("distillation.remote_validator")


def _is_api_teacher_mode() -> bool:
    """Return True iff DISTIL_TEACHER_MODE=api in the validator env.

    Routed through one helper so the (lower-cased, env-driven) check stays
    consistent across ETA computation, vLLM startup gating, and any future
    caller. The pod-side equivalent lives in ``scripts/api_teacher.py``.
    """
    return (policy_env("DISTIL_TEACHER_MODE", "") or "").lower() == "api"


def _pod_exec_silent(pod, cmd: str, *, timeout: int = 30, label: str | None = None) -> None:
    """Run ``pod.exec(cmd)`` swallowing any exception.

    Used for best-effort cleanup / kill commands where a failure must not
    abort the surrounding flow (network blip, pod just rebooted, target
    process already gone, etc.). Optionally logs ``label: <exc>`` at debug
    so we keep visibility without spamming WARN.
    """
    try:
        pod.exec(cmd, timeout=timeout)
    except Exception as exc:
        if label:
            logger.debug("%s: %s", label, exc)


# Single source of truth for the per-run remote artifact filenames.
# ``var_name`` matches the local that consumes it later in ``run_eval_on_pod``;
# ``basename`` is appended to the run_dir. The resume branch honours
# ``resume_pod_eval`` overrides per-key so a re-attached run keeps writing to
# the existing files even if the basenames evolve.
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
    """Return ``{var_name: remote_path}`` for one pod-eval run.

    ``resume`` (when truthy) provides explicit overrides — used by the
    validator-restart resume path so a re-attached run keeps writing to the
    same files even if the basename map evolves between releases.
    """
    overrides = resume or {}
    return {
        var_name: overrides.get(var_name) or f"{run_dir}/{basename}"
        for var_name, basename in _POD_REMOTE_FILES
    }


# Validator-side env vars propagated to the pod inner_eval command. Listed
# in one place so we don't have to dig through 90 lines of inline bash to
# add a new tunable. The pod-side script (``scripts/pod_eval_vllm.py``)
# documents each variable's semantics.
_POD_EVAL_ENV_ALLOWLIST: tuple[str, ...] = (
    # ── Bench battery toggles + composite gates ──
    "BENCH_BATTERY_ENABLED",
    "BENCH_BATTERY_SHADOW_AXES",
    "BENCH_BATTERY_LITE",
    "POD_PER_MODEL_TIMEOUT",
    "ARENA_V3_AXES_IN_COMPOSITE",
    "REASONING_DENSITY_IN_COMPOSITE",
    # 2026-04-26 — propagate validator-authoritative composite gates so the
    # pod-side ``pod_eval_vllm.py`` reports ``in_composite``/log labels that
    # match what the validator will actually do downstream. Previously the
    # pod read separate ``*_PROBE_IN_COMPOSITE`` variables that defaulted to
    # "0" while these axes defaulted to "1" in composite.py, so the eval log
    # lied about whether axes were in production.
    "JUDGE_AXIS_IN_COMPOSITE",
    "CHAT_TURNS_AXIS_IN_COMPOSITE",
    "PARETO_DOMINANCE_GATE",
    "KING_REGRESSION_GATE",
    # ── Bench sample counts ──
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
    # ── Bench max-token budgets ──
    # Added 2026-04-25 17:00 UTC after live round wall-time observation pegged
    # the bench battery at ~11 min/student. The default 1024-token AIME budget
    # alone burns ~120s/student even when the model can't get a single problem
    # right; tightening it (and the rest) is the single biggest lever.
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
    # ── Thinking-mode toggle (2026-05-05) ──
    # Defaults ON in the pod (BENCH_ENABLE_THINKING=1). Propagate so an
    # emergency rollback ("BENCH_ENABLE_THINKING=0") set on the
    # validator host immediately silences thinking on the next round
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
    # 2026-05-04 — order challengers by (failure_count asc, commit_block asc).
    # Previously the only sort key was commit_block; that's reasonable as a
    # tie-breaker but it puts proven CUDA-poisoners (UID 73 gutenmorgan/guten4
    # for 3 consecutive rounds) at the FRONT of the queue every round simply
    # because their commitment is older. When such a model triggers a
    # device-side assert it poisons the GPU context for every sibling student
    # — observed cost ~9 deferred students × 30+ min per round. Putting models
    # with prior failures last means the cascade only torpedoes already-failed
    # models, while the rest of the round still produces real composite scores.
    # Honest models with failure_count == 0 are still ordered by commit_block
    # so the FIFO-by-commit policy still holds for them.
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
        # If the persisted started_at is recent enough, prefer it so the
        # progress UI keeps showing the original elapsed wall time.
        try:
            now = float(resume_pod_eval.get("started_at") or now)
        except (TypeError, ValueError):
            pass
    # 2026-05-04: Per-round-wall-time recalibration. Old estimates
    # (teacher=90s, student=5*n_prompts=300s) predicted ~50 min for a
    # 10-student round; live timing on H200 NVL with the Kimi K2.6 API
    # teacher path shows ~370s teacher generation + ~15 min probe + KL
    # per student = ~3 hours. Underestimating drives miner panic in
    # Discord ("eval is stuck") because the dashboard ETA expires while
    # the round is barely past student #1.
    #
    # Components (median, single 1xH200 NVL pod):
    #   • teacher_api_phase:  370s (60 prompts × 6s avg via OpenRouter
    #                              concurrency=4)
    #   • teacher_probe_refs: 145s (32 think + 36 cap + 4 chat probes
    #                              via API)
    #   • per-student load:   140s (33B Kimi-arch download + load)
    #   • per-student probes: 380s (chat 65s + capability 36s + judge
    #                              16s + LFJ 720s on derail-prone
    #                              students... call it 380s for the
    #                              healthy case, gets revised below if
    #                              we detect derail in chat probe)
    #   • per-student KL:     180s (60 prompts × 3s on H200 eager mode)
    #
    # If DISTIL_TEACHER_MODE=api we skip local vLLM bootstrap, so the
    # teacher phase is dominated by API throughput, not GPU load.
    #
    # 2026-05-04 follow-up: round 04:15 telemetry showed the ETA was
    # *still* too low — actual student #1 runtime was ~50 min not the
    # ~12 min implied by 700s. The 700s figure assumed the LFJ +
    # bench-battery wall time we're paying with eager-attn Kimi 33B
    # on the derail-prone student set we currently have. Real cost
    # for a derail-prone student is:
    #   load 140s + chat 65s + cap 36s + judge 16s + LFJ 1027s +
    #   chat-turns 40s + bench 1700s + KL 180s = 3204s = 53 min.
    # Healthy students still pay the bench (~28 min) but skip the
    # derailed-LFJ tail, so they land around 45 min.
    #
    # The adaptive caps in pod_eval_vllm.py (cd483d0, 2026-05-04)
    # cut derailed students to ~19 min and DISTIL_STUDENT_BATCH_SIZE=4
    # shaves ~90s off KL for everyone, so the post-deploy ratio is
    # roughly 7 derailed × 19 min + 3 healthy × 45 min = ~270 min on
    # a 10-student round. Bake that into the ETA so miners' "stuck"
    # complaints stop spiking on every round restart.
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
        # API teacher wall time scales with prompt count and provider
        # concurrency. With reasoning disabled by default, live Kimi-K2.6
        # rounds are substantially faster than the old reasoning-mode
        # budget that produced 5h+ dashboard ETAs for 10-student rounds.
        per_prompt_s = 20.0 if teacher_max_new <= 768 else 26.0
        est_teacher_s = int((n_prompts * per_prompt_s) / api_concurrency + 300)
    else:
        est_teacher_s = 180
    # Current post-cap student rounds are dominated by probes + bench
    # battery and have been landing around 12-16 min/student after the
    # LFJ derail cap and batched KL path. Keep a modest buffer without
    # advertising a stale 5h full-round ETA.
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
    # 2026-04-24 — Pareto holistic eval v2 ships two small helper modules
    # alongside pod_eval.py: a vendored IFEval verifier set and a HumanEval
    # subprocess sandbox. They live next to pod_eval.py so the bench
    # probes can import them without touching sys.path.
    if not is_resuming:
        _aux_modules = [
            ("scripts/ifeval_vendor.py", "ifeval_vendor.py"),
            ("scripts/humaneval_sandbox.py", "humaneval_sandbox.py"),
            # 2026-05-03: cloud-API teacher module. Imported by
            # ``pod_eval.py`` (= scripts/pod_eval_vllm.py uploaded under a
            # neutral name) when DISTIL_TEACHER_MODE=api. Without this the
            # API path falls back to local vLLM with an ImportError noise.
            ("scripts/api_teacher.py", "api_teacher.py"),
            ("scripts/eval_progress_io.py", "eval_progress_io.py"),
            ("scripts/eval_policy.py", "eval_policy.py"),
            ("scripts/eval_benchmarks.py", "eval_benchmarks.py"),
            ("scripts/eval_items.py", "eval_items.py"),
        ]
        for local_aux, remote_name in _aux_modules:
            if os.path.isfile(local_aux):
                try:
                    pod.upload(local_aux, f"{run_dir}/{remote_name}", max_attempts=3)
                except Exception as exc:
                    logger.warning(f"Failed to upload {local_aux} (bench probes will skip): {exc}")
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
            # VLLM v1 spawns a child process that renames itself to "VLLM::EngineCore"
            # via prctl(PR_SET_NAME). That process holds the GPU allocation but will
            # NOT match `pkill -f 'vllm.entrypoints'` because its argv is literally
            # just "VLLM::EngineCore" — same story for "VllmWorker". If the parent
            # pod_eval dies without reaping the engine (as happens when the validator
            # is stopped hard), the EngineCore lives on forever holding ~130 GB of
            # GPU memory until the next reboot, causing OOMs in future rounds.
            #
            # To prevent that, we:
            #   1. pkill by comm (matches "VLLM::EngineCor", 15-char kernel limit)
            #   2. pkill by cmdline (belt and braces for any future vllm rename)
            #   3. fall back on nvidia-smi: if anything is still holding the GPU
            #      and it looks like a vllm/python worker, nuke it.
            pod.exec(
                # Build a deny-list of PIDs we MUST NOT kill: the chat-king
                # vLLM (chat_server.py + vllm.entrypoints --served-model-name
                # sn97-king on port 8100) and its descendants. Pre-2026-04-26
                # this cleanup blanket-killed every vllm.entrypoints process,
                # which is why chat.arbos.life went dark for ~30 minutes
                # every round. The chat-king and the eval-teacher coexist on
                # the same H200; eval-teacher is on port 9100 and uses
                # served-model-name "teacher", so the deny-list is precise.
                "preserve=''; "
                # Only one chat-king vLLM should be alive at any time. We
                # kept seeing 2-3 EngineCore processes simultaneously
                # because the previous detection used `ss -tlnp`, which
                # isn't installed on the eval pod (iproute2 missing) and
                # silently returned empty — fallback then preserved ALL
                # matching processes, leaving zombies untouched.
                #
                # New strategy (2026-04-28): pick the youngest
                # `vllm.entrypoints` chat-king API server (newest start
                # time per /proc/PID/stat field 22 or `ps -o etimes`)
                # and preserve only that process tree. Older duplicates
                # are leaks from prior chat-server restarts that
                # weren't reaped — they bind to port 8100 via
                # SO_REUSEPORT and squat on ~22 GB of VRAM each.
                #
                # Test history: 3 zombies observed on 2026-04-28
                # (PIDs 681678 [8h], 758224 [4h41m], 842769 [38min] —
                # all chat-king API servers, all alive, all
                # SO_REUSEPORT-bound to 8100). Fix kills the two
                # oldest, keeps the youngest (which is what
                # chat-tunnel.path most recently rebound to).
                #
                # Falls back to the broad include-all behaviour only if
                # NO chat-king is found at all (chat is genuinely
                # dark) so we don't kill a starting/restarting one.
                # Use `ps auxww` (not pgrep) so the pgrep command itself
                # doesn't appear in the match. pgrep -f matches its own
                # cmdline because the regex appears in argv. ps + grep -v
                # grep is the bullet-proof Unix idiom.
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
    # 2026-05-03: cloud-API teacher path takes precedence over the local
    # vLLM startup. ``pod_eval_vllm.py``'s API branch runs before the
    # vLLM branch and sets ``teacher_cache_loaded = True``, so the local
    # vLLM server never spins up — but pinning ``--no-vllm`` here keeps
    # the inner_eval command self-documenting and prevents any future
    # regression where the vLLM startup races with the API generation.
    _api_mode = _is_api_teacher_mode()
    if use_vllm and not _api_mode:
        # Eval shares the GPU with the chat-king vLLM. Two regimes:
        #
        #   1. Co-located chat-king (default today): chat-king *targets*
        #      0.15 of GPU memory but in practice we've seen leaked
        #      duplicate workers holding ~46 GB on a single H200
        #      (2026-04-27 — two engine-core processes from earlier
        #      chat-king restarts that weren't reaped). When the eval
        #      teacher then asks for 0.78 of 139.8 GB (109 GB) it fails
        #      with `Free memory ... is less than desired GPU memory
        #      utilization`, falls back to HF, and the round takes 80
        #      minutes of sequential generation instead of 5 minutes.
        #      0.65 leaves headroom for ~50 GB of chat-king occupancy
        #      (whether legit or leaked) without falling back.
        #
        #   2. Dedicated chat pod (recommended next step): use
        #      ``scripts/provision_chat_pod.py`` to rent a small GPU
        #      (RTX 4090 or H100) just for the chat king. Once chat is
        #      on its own pod the eval pod has the whole GPU; bump
        #      VLLM_EVAL_GPU_UTIL=0.92 in /home/distil/.secrets/distil.env
        #      to unlock it. Eval rounds finish in ~30-45 min instead of
        #      90+ min.
        #
        # Tune via VLLM_EVAL_GPU_UTIL.
        # 2026-05-01 (v30.3.5): lowered default 0.85 → 0.78. The pod is
        # co-tenant with the chat-king vLLM (~21 GiB at 0.15 util) AND
        # a dead vLLM EngineCore leaks ~3 GiB for 10-30s on crash. At
        # 0.85 the restart path hit "Free memory is less than desired
        # GPU memory utilization (0.85, 118.83 GiB)" and bailed to HF
        # for the rest of the round (root cause of the 4700s
        # teacher_generation timings on 5/1 morning). 0.78 leaves ~10
        # GiB of headroom; KV cache usage is <5% in practice so we
        # aren't losing throughput.
        #
        # 2026-05-03 (v30.3.6): lowered default 0.78 → 0.65. Chat-king
        # grew to ~44 GiB (was ~21 GiB at 0.15 util — likely KV cache
        # growth or leaked EngineCore). 0.78 * 139.8 = 109 GiB but only
        # 95.5 GiB free → teacher crash on init. 0.65 * 139.8 = 90.9 GiB
        # fits with ~4.6 GiB headroom. Eval throughput is barely affected
        # since KV cache is <5% in practice.
        eval_gpu_util = policy_env("VLLM_EVAL_GPU_UTIL", "0.65")
        vllm_flag = f" --vllm-gpu-util {eval_gpu_util}"
        if not is_full_eval and king_uid is not None and king_uid in models_to_eval:
            king_flag = f" --king {models_to_eval[king_uid]['model']}"
    tp_flag = f" --tensor-parallel-size {TP_SIZE}" if TP_SIZE > 0 else ""
    early_stop_flag = f" --early-stop-min {EARLY_STOP_MIN}" if EARLY_STOP_MIN > 0 else ""
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
    # 2026-05-03: zombie-aware status check.
    # ``kill -0 PID`` returns 0 for zombies (defunct processes whose entry is
    # still in the process table because no parent has reaped them), so the
    # previous check reported DISTIL_STATUS:running on a dead worker and the
    # validator polled forever. Workaround: also inspect ``ps -o stat=`` and
    # treat 'Z'/'X' (zombie / dying) as DISTIL_STATUS:dead. Falls through to
    # the cheap ``kill -0`` path when ``ps`` isn't available, which keeps
    # the legacy behaviour for hosts where /proc isn't mounted.
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
                # Single source of truth for the dashboard "current student"
                # field set. Both the populate and the clear paths iterate
                # over this so they cannot drift (regression prior to this
                # bind: current_se/current_ci/current_best were written into
                # ``progress`` but never popped, leaving stale values on the
                # dashboard between students).
                _CURRENT_FIELD_MAP = (
                    ("current_student", "student_name", None),
                    ("current_prompt", "prompts_done", 0),
                    ("current_kl", "kl_running_mean", None),
                    ("current_se", "kl_running_se", None),
                    ("current_ci", "ci_95", None),
                    ("current_best", "best_kl_so_far", None),
                    # 2026-05-04: per-stage progress so the dashboard shows
                    # "running bench: aime_bench (6/17)" instead of a
                    # static "0/60 prompts" while probes/bench run for
                    # ~25 min. The pod writes these via _set_stage().
                    ("current_stage", "stage", None),
                    ("bench_axis_idx", "bench_axis_idx", None),
                    ("bench_axis_total", "bench_axis_total", None),
                    ("current_student_started_at", "student_started_at", None),
                )
                if pod_progress.get("current"):
                    current = pod_progress["current"]
                    progress.update({
                        out_key: current.get(in_key, default)
                        for out_key, in_key, default in _CURRENT_FIELD_MAP
                    })
                else:
                    for out_key, _, _ in _CURRENT_FIELD_MAP:
                        progress.pop(out_key, None)
                # Always propagate teacher_prompts_done so the dashboard's
                # Phase A progress bar fills correctly even after we transition
                # into student loading (the previous gate dropped the value
                # the moment loading_student began, leaving the bar at 0).
                progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)
                for key in (
                    "run_started_at",
                    "teacher_started_at",
                    "teacher_finished_at",
                    "original_prompts_total",
                    "effective_prompts_total",
                    "effective_prompts_hash",
                    "teacher_mode",
                    "teacher_api",
                    "policy",
                    "script_revision",
                    "n_teacher_prompts_total",
                    "n_teacher_prompts_with_logprobs",
                    "n_teacher_prompts_dropped_missing_logprobs",
                ):
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
    eval_timeout = 5 * 60 * 60
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
