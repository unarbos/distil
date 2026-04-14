"""
Eval orchestration: prechecks, challenger selection, GPU eval execution,
result processing, and scoring logic.
"""
import json
import logging
import math
import os
import tempfile
import time

from eval.state import ValidatorState, log_event
from eval.pod import PodManager, sanitize_gpu_log
from eval.scoring import (
    record_failure, reset_failures, is_stale,
    is_disqualified, get_dq_reason, disqualify, is_flagged,
)
from eval.model_checker import (
    check_model_architecture, verify_model_integrity,
    compute_model_hash, check_duplicate_hash, register_model_hash,
    compute_tensor_metadata_hash,
)
from scripts.validator.config import (
    TEACHER_MODEL, MAX_KL_THRESHOLD, MAX_PROMPT_TOKENS, MAX_NEW_TOKENS,
    VLLM_CONCURRENCY, EPSILON, PAIRED_TEST_ALPHA,
    STALE_H2H_EPOCHS, TOP_N_ALWAYS_INCLUDE,
    ACTIVATION_COPY_THRESHOLD,
)

logger = logging.getLogger("distillation.remote_validator")


# ── Utility ───────────────────────────────────────────────────────────────

def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def check_activation_fingerprint(
    model_name: str, uid: int, fingerprint: dict, state_dir
) -> tuple[bool, int | None, str | None, float]:
    """
    Compare a model's activation fingerprint against all stored fingerprints.
    Returns (is_copy, original_uid, original_model, max_similarity).
    """
    from pathlib import Path

    fp_file = Path(state_dir) / "activation_fingerprints.json"
    stored = {}
    if fp_file.exists():
        try:
            stored = json.loads(fp_file.read_text())
        except Exception:
            stored = {}

    layer_fps = fingerprint.get("layer_fingerprints", {})
    if not layer_fps:
        return False, None, None, 0.0

    max_sim = 0.0
    max_sim_uid = None
    max_sim_model = None

    for other_uid_str, other_data in stored.items():
        other_uid = int(other_uid_str)
        if other_uid == uid:
            continue
        other_fps = other_data.get("layer_fingerprints", {})
        if not other_fps:
            continue

        # Compare matching layers
        sims = []
        for layer_key in layer_fps:
            if layer_key in other_fps:
                a = layer_fps[layer_key]
                b = other_fps[layer_key]
                if len(a) == len(b) and len(a) > 0:
                    sims.append(_cosine_sim(a, b))

        if sims:
            avg_sim = sum(sims) / len(sims)
            if avg_sim > max_sim:
                max_sim = avg_sim
                max_sim_uid = other_uid
                max_sim_model = other_data.get("model", "unknown")

    # Store this model's fingerprint
    stored[str(uid)] = {
        "model": model_name,
        "layer_fingerprints": layer_fps,
        "n_layers": fingerprint.get("n_layers"),
        "hidden_size": fingerprint.get("hidden_size"),
        "updated": time.time(),
    }
    try:
        fp_file.write_text(json.dumps(stored, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save fingerprints: {e}")

    is_copy = max_sim >= ACTIVATION_COPY_THRESHOLD
    return is_copy, max_sim_uid, max_sim_model, max_sim


# ── Pre-check models ─────────────────────────────────────────────────────

def precheck_all_models(commitments, uid_to_hotkey, uid_to_coldkey,
                        state: ValidatorState, max_params_b: float) -> tuple[dict, set]:
    """Run architecture/hash/integrity checks on all committed models.

    Returns (valid_models, disqualified_set) where valid_models is
    {uid: {model, revision, params_b, hotkey, commit_block}}.
    """
    valid_models = {}
    disqualified = set()

    for uid, commit in commitments.items():
        model_repo = commit["model"]
        revision = commit.get("revision", "main")
        hotkey = commit.get("hotkey", uid_to_hotkey.get(uid, ""))
        this_commit_block = commit.get("block")

        # Check DQ (per-hotkey per-submission only)
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=this_commit_block):
            reason = get_dq_reason(uid, hotkey, state.dq_reasons, commit_block=this_commit_block)
            logger.info(f"UID {uid} ({model_repo}): DISQUALIFIED — {reason}")
            disqualified.add(uid)
            continue

        # Already permanently DQ'd
        if state.scores.get(str(uid), 0) > MAX_KL_THRESHOLD:
            disqualified.add(uid)
            continue

        if is_stale(uid, state.failures):
            logger.debug(f"UID {uid}: stale (too many failures), skipping")
            disqualified.add(uid)
            continue

        # For already-evaluated UIDs, still verify architecture (lightweight config.json check)
        uid_str = str(uid)
        if (uid_str in state.evaluated_uids and uid_str in state.scores
                and state.scores[uid_str] <= MAX_KL_THRESHOLD):
            # Re-verify architecture (lightweight config.json check)
            try:
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(model_repo, "config.json", revision=revision)
                import json as _json
                with open(cfg_path) as _f:
                    cfg = _json.load(_f)
                archs = cfg.get("architectures", [])
                mtype = cfg.get("model_type", "")
                if mtype != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in archs:
                    logger.info(f"UID {uid} ({model_repo}): FAIL — wrong architecture ({mtype}/{','.join(archs)})")
                    record_failure(uid, state.failures)
                    disqualify(hotkey, f"arch: Must use Qwen3_5ForConditionalGeneration (found {','.join(archs)}, model_type={mtype}). Fix: edit config.json on HuggingFace.",
                               state.dq_reasons, commit_block=this_commit_block)
                    disqualified.add(uid)
                    state.scores.pop(uid_str, None)
                    state.evaluated_uids.discard(uid_str)
                    continue
            except Exception:
                pass  # Transient HF error — allow through, catch next epoch
            # Re-verify integrity: still public, weights unchanged
            expected_hash = state.model_hashes.get(str(uid))
            integrity = verify_model_integrity(model_repo, revision, expected_hash)
            if integrity.get("transient"):
                pass  # Transient error — allow through
            elif not integrity["pass"]:
                logger.info(f"UID {uid} ({model_repo}): INTEGRITY FAIL — {integrity['reason']}")
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons,
                           commit_block=this_commit_block)
                disqualified.add(uid)
                state.evaluated_uids.discard(uid_str)
                continue
            valid_models[uid] = {"model": model_repo, "revision": revision, "params_b": None, "hotkey": hotkey}
            continue

        logger.info(f"Checking {model_repo}...")

        # Flag check
        hf_user = model_repo.split("/")[0] if "/" in model_repo else None
        coldkey = uid_to_coldkey.get(uid)
        flag_reason = is_flagged(coldkey=coldkey, hf_username=hf_user, dq=state.dq_reasons)
        if flag_reason:
            logger.warning(f"UID {uid} FLAGGED: {flag_reason}")

        # Architecture check
        check = check_model_architecture(model_repo, revision, max_params_b)
        if check.get("transient"):
            logger.info(f"UID {uid} ({model_repo}): TRANSIENT ERROR — {check['reason']}, will retry next epoch")
            continue
        if not check["pass"]:
            logger.info(f"UID {uid} ({model_repo}): FAIL — {check['reason']}")
            record_failure(uid, state.failures)
            disqualify(hotkey, f"arch: {check['reason']}", state.dq_reasons,
                       coldkey=coldkey, hf_username=hf_user, commit_block=this_commit_block)
            disqualified.add(uid)
            continue

        # Duplicate hash check — earlier commitment wins
        model_hash = compute_model_hash(model_repo, revision)
        if model_hash:
            original_uid = check_duplicate_hash(model_hash, uid, state.state_dir)
            if original_uid is not None:
                orig_block = commitments.get(original_uid, {}).get("block", float("inf"))
                this_block = commit.get("block", float("inf"))
                if this_block >= orig_block:
                    orig_model = commitments.get(original_uid, {}).get("model", "?")
                    logger.info(f"UID {uid} ({model_repo}): DUPLICATE of UID {original_uid}")
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(hotkey, f"copy: identical weights to UID {original_uid} ({orig_model}), committed later at block {this_block} vs {orig_block}",
                               state.dq_reasons, commit_block=this_commit_block)
                    disqualified.add(uid)
                    continue
                else:
                    logger.info(f"UID {original_uid} is duplicate of UID {uid} (committed earlier)")
                    state.scores[str(original_uid)] = MAX_KL_THRESHOLD + 1
                    orig_hotkey = uid_to_hotkey.get(original_uid, str(original_uid))
                    orig_commit_block = commitments.get(original_uid, {}).get("block")
                    disqualify(orig_hotkey, f"copy: identical weights to UID {uid} ({model_repo}), committed later",
                               state.dq_reasons, commit_block=orig_commit_block)
                    valid_models.pop(original_uid, None)
                    disqualified.add(original_uid)
                    register_model_hash(model_hash, uid, state.state_dir)
            else:
                register_model_hash(model_hash, uid, state.state_dir)

        # Integrity check — reset expected hash if miner re-committed or UID recycled
        expected_hash = state.model_hashes.get(str(uid))
        stored_commit_block = state.model_hashes.get(f"{uid}_block")
        stored_hotkey = state.model_hashes.get(f"{uid}_hotkey")
        # Detect UID recycling (new hotkey) or re-commitment (new block)
        hotkey_changed = stored_hotkey is not None and stored_hotkey != hotkey
        block_changed = this_commit_block and stored_commit_block and this_commit_block != stored_commit_block
        # Also reset if we have a hash but no stored block (legacy data)
        legacy_no_block = expected_hash is not None and stored_commit_block is None and this_commit_block
        if hotkey_changed or block_changed or legacy_no_block:
            # Miner made a new commitment or UID recycled — accept new weights
            reason = "hotkey changed (UID recycled)" if hotkey_changed else "new commitment" if block_changed else "legacy hash (no block stored)"
            logger.info(f"UID {uid}: {reason} at block {this_commit_block} (was {stored_commit_block}), resetting hash")
            expected_hash = None
            state.model_hashes.pop(str(uid), None)
            state.model_hashes.pop(f"{uid}_block", None)
            state.model_hashes.pop(f"{uid}_hotkey", None)
            # Clear old DQ for this commitment (try both old and new hotkey keys)
            for dq_hk in [hotkey, stored_hotkey] if stored_hotkey else [hotkey]:
                for dq_key in [f"{dq_hk}:{stored_commit_block}", dq_hk]:
                    if dq_key and dq_key in state.dq_reasons:
                        logger.info(f"UID {uid}: Clearing stale DQ: {dq_key}")
                        del state.dq_reasons[dq_key]
            # Clear from evaluated_uids so they get re-evaluated
            state.evaluated_uids.discard(str(uid))
            state.scores.pop(str(uid), None)
            # Reset failure counter so stale UIDs get a fresh chance
            reset_failures(uid, state.failures)
        integrity = verify_model_integrity(model_repo, revision, expected_hash)
        if integrity.get("transient"):
            logger.info(f"UID {uid} integrity: TRANSIENT ERROR — {integrity['reason']}, will retry")
            continue
        if not integrity["pass"]:
            logger.info(f"UID {uid} DISQUALIFIED: {integrity['reason']}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons,
                       commit_block=this_commit_block)
            disqualified.add(uid)
            continue
        if integrity["current_hash"]:
            state.model_hashes[str(uid)] = integrity["current_hash"]
            if this_commit_block:
                state.model_hashes[f"{uid}_block"] = this_commit_block
            state.model_hashes[f"{uid}_hotkey"] = hotkey
            state.save_model_hashes()

        valid_models[uid] = {
            "model": model_repo, "revision": revision,
            "params_b": check.get("params_b", 0),
            "commit_block": commit.get("block", float("inf")),
            "hotkey": hotkey,
            "vllm_compatible": check.get("vllm_compatible"),
            "vllm_reason": check.get("vllm_reason"),
        }
        logger.info(f"UID {uid}: {model_repo} ({check.get('params_b', 0):.2f}B) ✓")

    return valid_models, disqualified


# ── Challenger Selection ──────────────────────────────────────────────────

def select_challengers(valid_models, state: ValidatorState, king_uid, king_kl,
                       epoch_count: int) -> dict:
    """Select challengers for this round using priority-based selection.

    Priority levels:
      P1: Brand-new models (never scored)
      P1b: Scored models untested vs new king (initial eval phase only)
      P3: Stale re-tests (>STALE_H2H_EPOCHS since last H2H)

    Returns dict of {uid: model_info} for challengers.
    """
    challengers = {}

    # Base challengers: unevaluated valid models
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.evaluated_uids and uid_str in state.scores:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        # Check model history — skip known-bad models
        best_ever = state.model_score_history.get(model_name, {}).get("best_kl")
        if best_ever is not None and king_kl < float("inf"):
            skip_threshold = max(king_kl * 2.0, king_kl + 0.05)
            if best_ever > skip_threshold:
                state.evaluated_uids.add(uid_str)
                continue
        challengers[uid] = info

    if king_uid is None:
        return challengers

    king_model_name = valid_models.get(king_uid, {}).get("model", "")

    # P1: New models (never scored at all)
    p1_new = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info["model"] in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        if state.scores.get(uid_str) is not None:
            continue
        if uid_str in state.evaluated_uids:
            continue
        p1_new.append(uid)

    for uid in p1_new:
        challengers[uid] = valid_models[uid]
    if p1_new:
        logger.info(f"🎯 SMART CHALLENGER: {len(p1_new)} new submission(s) — Priority 1: never evaluated")

    # P1b: Initial eval phase — scored models untested vs new king
    in_initial_eval = state.top4_leaderboard.get("phase") == "initial_eval"
    if in_initial_eval:
        FULL_EVAL_KL_CUTOFF = 0.12
        p1b = []
        for uid, info in valid_models.items():
            if uid == king_uid or uid in challengers:
                continue
            if info["model"] in state.permanently_bad_models:
                continue
            uid_str = str(uid)
            global_kl = state.scores.get(uid_str)
            if global_kl is None or global_kl <= 0 or global_kl > FULL_EVAL_KL_CUTOFF:
                continue
            h2h_record = state.h2h_tested_against_king.get(uid_str, {})
            if h2h_record.get("king_uid") == king_uid:
                continue
            p1b.append((uid, global_kl))
        if p1b:
            p1b.sort(key=lambda x: x[1])
            for uid, _ in p1b:
                challengers[uid] = valid_models[uid]
            logger.info(f"🏆 FULL EVAL: {len(p1b)} scored models added (untested vs new king, KL<=0.12)")

    # P3: Stale re-tests — DISABLED
    # Models are evaluated once. No re-evals against new kings.
    # The king must be beaten by NEW challengers, not by re-testing old ones.

    return challengers


def add_top5_contenders(challengers, valid_models, state: ValidatorState, king_uid):
    """Always include top contenders (by KL score) in every eval round."""
    if king_uid is None:
        return
    contenders_added = 0
    # Get all scored models sorted by KL, pick top 4 that aren't king
    scored = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is not None and 0 < kl < float('inf'):
            scored.append((uid, kl))
    scored.sort(key=lambda x: x[1])  # best KL first
    for uid, kl in scored[:TOP_N_ALWAYS_INCLUDE - 1]:  # top 4 (king is the 5th)
        challengers[uid] = valid_models[uid]
        contenders_added += 1
    if contenders_added:
        logger.info(f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) to eval")


def cap_challengers(challengers, state: ValidatorState, king_uid):
    """Hard cap challengers if too many (sanity check)."""
    phase = state.top4_leaderboard.get("phase", "maintenance")
    max_cap = 80 if phase == "initial_eval" else 15
    if len(challengers) <= max_cap:
        return
    logger.warning(f"{len(challengers)} challengers exceeds cap of {max_cap} (phase={phase}). Truncating.")
    king_entry = challengers.pop(king_uid, None)
    sorted_chall = sorted(challengers.items(), key=lambda x: state.scores.get(str(x[0]), 999))
    challengers.clear()
    challengers.update(dict(sorted_chall[:max_cap - (1 if king_entry else 0)]))
    if king_entry:
        challengers[king_uid] = king_entry


# ── Eval Execution ────────────────────────────────────────────────────────

def check_models_exist(models_to_eval, uid_to_hotkey, state: ValidatorState, commitments: dict) -> list:
    """Pre-scoring HF HEAD check — remove deleted models."""
    removed = []
    for uid in list(models_to_eval.keys()):
        mr = models_to_eval[uid]["model"]
        try:
            import urllib.request
            req = urllib.request.Request(f"https://huggingface.co/api/models/{mr}", method="HEAD")
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                logger.warning(f"UID {uid} ({mr}): deleted from HF — DQ")
                hk = models_to_eval[uid].get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                cb = models_to_eval[uid].get("commit_block")
                disqualify(hk, f"Model {mr} no longer exists on HuggingFace (404)",
                           state.dq_reasons, commit_block=cb)
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                state.evaluated_uids.add(str(uid))
                removed.append(uid)
    for uid in removed:
        models_to_eval.pop(uid, None)
    return removed


def run_eval_on_pod(pod: PodManager, models_to_eval: dict, king_uid, n_prompts: int,
                    prompt_texts: list, state: ValidatorState, max_params_b: float,
                    is_full_eval: bool, use_vllm: bool, eval_script: str,
                    eval_script_remote: str):
    """Execute the GPU eval on the remote pod and return results.

    Handles: prompt upload, eval script upload, progress polling,
    result download, and timeout management.

    Returns the parsed results dict or None on failure.
    """
    import threading
    import concurrent.futures
    import shutil

    # Sort challengers by commit block (earliest first)
    ordered_uids = []
    if king_uid is not None and king_uid in models_to_eval:
        ordered_uids.append(king_uid)
    challenger_uids_sorted = sorted(
        [uid for uid in models_to_eval if uid != king_uid],
        key=lambda uid: models_to_eval[uid].get("commit_block", float("inf")),
    )
    ordered_uids.extend(challenger_uids_sorted)

    # Write eval progress for dashboard
    now = time.time()
    est_teacher_s = 90
    est_per_student_s = 5 * n_prompts
    est_total_s = est_teacher_s + est_per_student_s * len(models_to_eval)
    eval_order = []
    if king_uid is not None and king_uid in models_to_eval:
        eval_order.append({"uid": king_uid, "model": models_to_eval[king_uid]["model"], "role": "king"})
    for uid in challenger_uids_sorted:
        eval_order.append({"uid": uid, "model": models_to_eval[uid]["model"], "role": "challenger"})
    progress = {
        "active": True, "phase": "teacher_loading",
        "models": {str(uid): info["model"] for uid, info in models_to_eval.items()},
        "eval_order": eval_order,
        "students_total": len(models_to_eval), "students_done": 0,
        "prompts_total": n_prompts, "prompts_done": 0,
        "king_uid": king_uid,
        "challenger_uids": [uid for uid in models_to_eval if uid != king_uid],
        "started_at": now,
        "estimated_duration_s": est_total_s,
        "estimated_completion": now + est_total_s,
    }
    state.save_progress(progress)

    # Upload prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(prompt_texts, f)
        f.flush()
        os.fsync(f.fileno())
        prompts_file = f.name
    try:
        pod.upload(prompts_file, "/home/prompts.json", max_attempts=3)
    finally:
        os.unlink(prompts_file)

    # Re-upload eval script
    pod.upload(eval_script, eval_script_remote, max_attempts=5)

    # Kill any existing eval process before starting a new one
    try:
        existing = pod.exec("pgrep -c -f pod_eval 2>/dev/null || echo 0", timeout=10)
        if isinstance(existing, dict):
            existing = existing.get('stdout', '') or existing.get('output', '') or '0'
        if isinstance(existing, dict):
            existing = '0'  # nested dict fallback
        count = int(str(existing).strip().split('\n')[-1])
        if count > 0:
            logger.warning(f"Found {count} existing eval process(es) on pod — killing before new round")
            pod.exec("pkill -9 -f pod_eval; sleep 2", timeout=15)
    except Exception as e:
        logger.debug(f"Pre-check for existing eval: {e}")

    # Clean ALL stale artifacts — every round starts fresh, no resume
    try:
        pod.exec("rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_progress.json /home/eval_results.json /home/teacher_cache.pt")
        logger.info("Cleared all pod artifacts (eval_results, teacher_cache, progress)")
    except Exception:
        pass

    # Disk cleanup + clear GPU
    try:
        disk_pct = pod.disk_cleanup(TEACHER_MODEL)
        if disk_pct is not None:
            log_event(f"Pod disk: {disk_pct}% used after cleanup", state_dir=str(state.state_dir))
    except Exception as e:
        log_event(f"Pod disk cleanup failed: {str(e)[:100]}", level="warn", state_dir=str(state.state_dir))
    pod.clear_gpu()

    # Build eval command — pin revisions to prevent weight-swap attacks
    student_list = ",".join(models_to_eval[uid]["model"] for uid in ordered_uids)
    revision_list = ",".join(models_to_eval[uid].get("revision", "main") for uid in ordered_uids)
    king_flag = ""
    vllm_flag = " --no-vllm"
    if use_vllm:
        vllm_flag = " --vllm-gpu-util 0.90"
        if not is_full_eval and king_uid is not None and king_uid in models_to_eval:
            king_flag = f" --king {models_to_eval[king_uid]['model']}"

    eval_cmd = (
        f"cd /home && python3 -u pod_eval.py "
        f"--teacher {TEACHER_MODEL} "
        f"--students {student_list} "
        f"--revisions {revision_list} "
        f"--prompts prompts.json "
        f"--output eval_results.json "
        f"--max-prompt-len {MAX_PROMPT_TOKENS} "
        f"--max-new-tokens {MAX_NEW_TOKENS} "
        f"--max-params-b {max_params_b} "
        f"--concurrency {VLLM_CONCURRENCY} "
        f"--teacher-logits /home/teacher_cache.pt"
        f"{king_flag}"
        f"{vllm_flag}"
        f" 2>&1 | tee /home/eval_output.log"
    )

    # Background progress polling
    poll_stop = threading.Event()
    progress_path = state.state_dir / "eval_progress.json"
    gpu_log_path = state.state_dir / "gpu_eval.log"

    def _poll_pod_progress():
        """Poll live progress from pod every 15s for dashboard updates."""
        while not poll_stop.is_set():
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = tmp.name
                pod.download("/home/eval_progress.json", tmp_path)
                with open(tmp_path) as f:
                    pod_progress = json.load(f)
                os.unlink(tmp_path)

                pod_phase = pod_progress.get("phase", "scoring")
                progress["phase"] = pod_phase
                progress["pod"] = pod_progress

                if pod_progress.get("current"):
                    cur = pod_progress["current"]
                    progress.update({
                        "current_student": cur.get("student_name"),
                        "current_prompt": cur.get("prompts_done", 0),
                        "current_kl": cur.get("kl_running_mean"),
                        "current_se": cur.get("kl_running_se"),
                        "current_ci": cur.get("ci_95"),
                        "current_best": cur.get("best_kl_so_far"),
                    })
                else:
                    for k in ("current_student", "current_prompt", "current_kl"):
                        progress.pop(k, None)

                if pod_phase in ("teacher_generation", "teacher_logits", "teacher_loading",
                                 "vllm_starting", "vllm_generating", "gpu_precompute", "loading_student"):
                    progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)

                pod_completed = pod_progress.get("completed", [])
                progress["completed"] = pod_completed
                progress["students_done"] = len(pod_completed)
                state.save_progress(progress)
            except Exception:
                pass

            try:
                log_result = pod.exec("tail -100 /home/eval_output.log 2>/dev/null || echo ''", timeout=30)
                log_text = log_result.get("stdout", "")
                if log_text.strip():
                    gpu_log_path.write_text(sanitize_gpu_log(log_text))
            except Exception:
                pass

            poll_stop.wait(15)

    poll_thread = threading.Thread(target=_poll_pod_progress, daemon=True)
    poll_thread.start()

    # Execute with timeout
    n_eval_models = len(models_to_eval)
    EVAL_TIMEOUT = 2 * 60 * 60  # 2 hours
    logger.info(f"Running eval ({n_eval_models} models, {n_prompts} prompts, timeout={EVAL_TIMEOUT // 60}m)")
    log_event(f"Running eval on pod: king vs {n_eval_models - 1} challengers, {n_prompts} prompts", state_dir=str(state.state_dir))
    eval_env = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "TOKENIZERS_PARALLELISM": "false",
    }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pod.exec, eval_cmd, env=eval_env)
            try:
                result = future.result(timeout=EVAL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error(f"Eval timed out after {EVAL_TIMEOUT}s — killing")
                try:
                    pod.exec("pkill -9 -f pod_eval.py; echo killed", timeout=30)
                except Exception:
                    pass
                # Connection may be dead — reconnect for result retrieval
                try:
                    pod.reconnect()
                except Exception as re_err:
                    logger.error(f"Reconnect failed after timeout: {re_err}")
                result = {"stdout": "", "stderr": "timeout", "exit_code": -1, "success": False}
    except Exception as e:
        logger.error(f"lium.exec EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        # Try to reconnect for future operations
        try:
            pod.reconnect()
        except Exception:
            pass
        return None
    finally:
        poll_stop.set()
        poll_thread.join(timeout=5)

    # Print last lines of output
    stdout = result.get('stdout', '') or ''
    stderr = result.get('stderr', '') or ''
    if stdout.strip():
        for line in stdout.strip().split('\n')[-30:]:
            logger.info(f"  GPU: {line[:200]}")
    if stderr.strip():
        for line in stderr.strip().split('\n')[-10:]:
            logger.warning(f"  GPU ERR: {line[:200]}")

    # Download results
    results_local = str(state.state_dir / "last_eval.json")
    try:
        pod.download("/home/eval_results.json", results_local)
    except Exception:
        logger.error("Failed to download results")
        if not result.get('success', False):
            state.save_progress({"active": False})
            return None

    # Save pod eval log for debugging/transparency
    try:
        logs_dir = state.state_dir / "pod_logs"
        logs_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_dest = str(logs_dir / f"eval_{ts}.log")
        pod.download("/home/eval_output.log", log_dest)
        # Keep only last 20 log files
        log_files = sorted(logs_dir.glob("eval_*.log"))
        for old in log_files[:-20]:
            old.unlink(missing_ok=True)
        logger.info(f"Pod eval log saved: {log_dest}")
    except Exception as log_err:
        logger.warning(f"Pod log retrieval failed (non-fatal): {log_err}")

    # Save eval data (prompts + teacher completions) for reproducibility
    try:
        eval_data_dir = state.state_dir / "eval_data"
        eval_data_dir.mkdir(exist_ok=True)
        eval_data_dest = str(eval_data_dir / f"eval_data_{ts}.json")
        pod.download("/home/eval_data.json", eval_data_dest)
        # Also save as "latest" for easy API access
        shutil.copy2(eval_data_dest, str(state.state_dir / "eval_data_latest.json"))
        # Keep only last 10 eval data files
        data_files = sorted(eval_data_dir.glob("eval_data_*.json"))
        for old in data_files[:-10]:
            old.unlink(missing_ok=True)
        logger.info(f"Eval data saved: {eval_data_dest}")
    except Exception as data_err:
        logger.warning(f"Eval data retrieval failed (non-fatal): {data_err}")

    # Check if results are usable
    try:
        with open(results_local) as f:
            results = json.load(f)
        n_students = len(results.get("students", {}))
        if n_students == 0 and not result.get('success', False):
            logger.error("Eval failed, no usable results")
            state.save_progress({"active": False})
            return None
        if not result.get('success', False):
            logger.warning(f"Eval failed but recovered {n_students} partial results")
    except Exception:
        logger.error("Results file corrupt")
        state.save_progress({"active": False})
        return None

    return results


# ── Result Processing ─────────────────────────────────────────────────────

def process_results(results, models_to_eval, king_uid, state: ValidatorState,
                    uid_to_hotkey, commitments, n_prompts, current_block, king_kl,
                    epoch_count, is_full_eval, epoch_start_time=None):
    """Process eval results: update scores, run paired t-test, crown winner.

    Returns (winner_uid, winner_kl, h2h_results_list, king_h2h_kl, king_per_prompt, this_round_uids).
    """
    from scipy import stats as _scipy_stats

    uid_to_model = {uid: m["model"] for uid, m in models_to_eval.items()}
    model_to_uid = {m: uid for uid, m in uid_to_model.items()}

    king_h2h_kl = None
    this_round_uids = set()

    # ── Score each model ──
    for model_name, student_result in results.get("students", {}).items():
        uid = model_to_uid.get(model_name)
        if uid is None:
            continue

        # Reference model — log but don't affect scores/weights/state
        is_reference = models_to_eval.get(uid, {}).get("is_reference", False)
        if is_reference:
            ref_kl = student_result.get("kl_global_avg", "error")
            logger.info(f"REFERENCE ({model_name}): KL={ref_kl} (baseline — not scored)")
            continue

        if "error" in student_result:
            logger.warning(f"UID {uid} ({model_name}): eval error — {student_result['error']}")
            record_failure(uid, state.failures)
            continue

        # Functional copy detection
        if student_result.get("functional_copy"):
            copy_of = student_result.get("copy_of", "unknown")
            copy_uid = next((u for u, i in models_to_eval.items() if i["model"] == copy_of), None)
            reason = f"copy: functional copy of {copy_of}" + (f" (UID {copy_uid})" if copy_uid else "") + " — identical logit distribution"
            logger.info(f"UID {uid} ({model_name}): FUNCTIONAL COPY — {reason}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            cb = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hk, reason, state.dq_reasons, commit_block=cb)
            state.evaluated_uids.add(str(uid))
            continue

        # Activation fingerprint copy detection
        fp = student_result.get("activation_fingerprint")
        if fp and fp.get("layer_fingerprints"):
            is_copy, orig_uid, orig_model, sim = check_activation_fingerprint(
                model_name, uid, fp, state.state_dir
            )
            if is_copy:
                reason = (f"copy: activation-space duplicate of UID {orig_uid} ({orig_model}) "
                          f"— cosine similarity {sim:.6f} > {ACTIVATION_COPY_THRESHOLD}")
                logger.info(f"UID {uid} ({model_name}): ACTIVATION COPY — {reason}")
                log_event(f"Activation copy detected: UID {uid} is copy of UID {orig_uid} (sim={sim:.6f})",
                          level="warning", state_dir=str(state.state_dir))
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                cb = models_to_eval.get(uid, {}).get("commit_block")
                disqualify(hk, reason, state.dq_reasons, commit_block=cb)
                state.evaluated_uids.add(str(uid))
                continue
            elif sim > 0.99:
                logger.info(f"UID {uid}: high similarity to UID {orig_uid} (sim={sim:.6f}) — below threshold, monitoring")

        # VRAM fraud check
        if student_result.get("status") == "fraud_vram":
            reason = student_result.get("reason", "VRAM fraud detected")
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            cb = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hk, reason, state.dq_reasons, commit_block=cb)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue

        speed_flag = student_result.get("speed_flag")
        if speed_flag:
            logger.warning(f"UID {uid} ({model_name}): ⚠️ {speed_flag}")

        kl = student_result.get("kl_global_avg", float("inf"))

        # KL=0 means model IS the teacher
        if kl <= 1e-6:
            reason = f"FRAUD: KL={kl:.10f} — model produces identical outputs to teacher"
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            cb = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hk, reason, state.dq_reasons, commit_block=cb)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue

        if kl == float("inf") or kl < 0:
            logger.warning(f"UID {uid}: invalid KL={kl}")
            record_failure(uid, state.failures)
            continue

        this_round_uids.add(uid)

        if uid == king_uid:
            king_h2h_kl = kl
            state.scores[str(uid)] = kl
            state.evaluated_uids.add(str(uid))
            logger.info(f"UID {uid} ({model_name}): H2H KL={kl:.6f} (king — global score UPDATED)")
            log_event(f"UID {uid}: KL={kl:.6f} (king)", state_dir=str(state.state_dir))
        else:
            state.scores[str(uid)] = kl
            state.evaluated_uids.add(str(uid))
            reset_failures(uid, state.failures)
            logger.info(f"UID {uid} ({model_name}): KL={kl:.6f}")
            # Compute vs-king info for log
            _vs_info = ""
            if king_h2h_kl is not None and king_h2h_kl > 0:
                _pct = (king_h2h_kl - kl) / king_h2h_kl * 100
                _vs_info = f", {_pct:+.2f}% vs king"
            log_event(f"UID {uid}: KL={kl:.6f}{_vs_info}", state_dir=str(state.state_dir))

    # ── Paired t-test dethronement ──
    if king_uid is not None and king_h2h_kl is None:
        # King MUST produce a fresh score every round. If it fails (deleted repo,
        # download error, etc.), it loses the crown to the best challenger.
        logger.warning(f"King UID {king_uid} did not produce a score — will lose crown to best challenger")
        # Get UIDs that actually produced results THIS round
        this_round_scored = set()
        for model_name, student_data in results.get("students", {}).items():
            if "error" not in student_data and student_data.get("kl_global_avg") is not None:
                for uid, info in models_to_eval.items():
                    if info.get("model") == model_name:
                        this_round_scored.add(uid)
                        break
        best_challenger_uid = None
        best_challenger_kl = float("inf")
        for uid in (uid for uid in models_to_eval if uid != king_uid and uid in this_round_scored):
            uid_str = str(uid)
            if uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD:
                if state.scores[uid_str] < best_challenger_kl:
                    best_challenger_kl = state.scores[uid_str]
                    best_challenger_uid = uid
        if best_challenger_uid is not None:
            logger.info(f"King failed eval — promoting best challenger UID {best_challenger_uid} (KL={best_challenger_kl:.6f}) [fresh score this round]")
            log_event(f"King UID {king_uid} failed to produce score — promoting UID {best_challenger_uid}",
                      level="warning", state_dir=str(state.state_dir))
            # Build minimal h2h_results for state recording
            king_fail_results = []
            for uid in this_round_scored:
                uid_str = str(uid)
                model_name = uid_to_model.get(uid, "")
                kl = state.scores.get(uid_str)
                if kl and kl > 0:
                    king_fail_results.append({"uid": uid, "model": model_name, "kl": round(kl, 6),
                                              "is_king": False, "vs_king": "king_failed"})
            king_fail_results.sort(key=lambda x: x["kl"])
            return best_challenger_uid, best_challenger_kl, king_fail_results, None, None, set(models_to_eval.keys())
        else:
            logger.error(f"King failed eval and no valid challengers produced fresh scores — king retains crown by default")
            log_event(f"King UID {king_uid} failed and no valid challengers with fresh scores",
                      level="error", state_dir=str(state.state_dir))
            return king_uid, king_kl, [], king_h2h_kl, None, set(models_to_eval.keys())

    king_new_kl = king_h2h_kl if king_h2h_kl is not None else state.scores.get(str(king_uid), king_kl) if king_uid else float("inf")
    epsilon_threshold = king_new_kl * (1.0 - EPSILON) if king_uid else float("inf")
    epsilon_dethroned_by = None

    king_model_name = uid_to_model.get(king_uid)
    king_per_prompt = None
    if king_model_name and king_model_name in results.get("students", {}):
        king_per_prompt = results["students"][king_model_name].get("kl_per_prompt")

    challengers = {uid: info for uid, info in models_to_eval.items() if uid != king_uid}

    if king_uid is not None and challengers:
        for uid in challengers:
            uid_str = str(uid)
            if uid_str not in state.scores or state.scores[uid_str] <= 0 or state.scores[uid_str] > MAX_KL_THRESHOLD:
                continue
            challenger_kl = state.scores[uid_str]
            challenger_model = uid_to_model.get(uid)
            challenger_per_prompt = None
            if challenger_model and challenger_model in results.get("students", {}):
                challenger_per_prompt = results["students"][challenger_model].get("kl_per_prompt")

            # Use paired prompts (intersection) for t-test — handles early-stopped models
            # that have fewer prompts than king
            MIN_PROMPTS_DETHRONE = 100  # minimum prompts for dethronement consideration
            if king_per_prompt and challenger_per_prompt:
                # Use the shorter length (aligned prompts)
                n_paired = min(len(king_per_prompt), len(challenger_per_prompt))
                if n_paired >= MIN_PROMPTS_DETHRONE:
                    deltas = [king_per_prompt[i] - challenger_per_prompt[i] for i in range(n_paired)]
                    mean_delta = sum(deltas) / len(deltas)
                    t_stat, p_value = _scipy_stats.ttest_1samp(deltas, 0.0, alternative='greater')
                    n_test = len(deltas)
                    pct_better = (mean_delta / king_new_kl * 100) if king_new_kl > 0 else 0

                    if p_value < PAIRED_TEST_ALPHA and mean_delta > 0:
                        logger.info(f"UID {uid} DETHRONED king UID {king_uid}! "
                                    f"p={p_value:.6f}, delta={mean_delta:.6f} ({pct_better:.2f}%), t={t_stat:.3f}, n={n_test}")
                        if epsilon_dethroned_by is None or challenger_kl < state.scores.get(str(epsilon_dethroned_by), float("inf")):
                            epsilon_dethroned_by = uid
                    elif mean_delta > 0:
                        logger.info(f"UID {uid}: better but not significant (p={p_value:.4f}, delta={mean_delta:.6f}, n={n_test})")
                    else:
                        logger.info(f"UID {uid}: worse than king (delta={mean_delta:.6f}, p={p_value:.4f}, n={n_test})")
                else:
                    logger.info(f"UID {uid}: insufficient prompts for dethronement ({n_paired} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
            else:
                # Legacy epsilon fallback — only if challenger has enough data
                challenger_n = len(challenger_per_prompt) if challenger_per_prompt else 0
                if challenger_n < MIN_PROMPTS_DETHRONE:
                    logger.info(f"UID {uid}: insufficient prompts for legacy epsilon ({challenger_n} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
                elif challenger_kl < epsilon_threshold:
                    logger.info(f"UID {uid} DETHRONED king UID {king_uid}! KL={challenger_kl:.6f} < {epsilon_threshold:.6f} [legacy epsilon, n={challenger_n}]")
                    if epsilon_dethroned_by is None or challenger_kl < state.scores.get(str(epsilon_dethroned_by), float("inf")):
                        epsilon_dethroned_by = uid

    # ── Determine winner ──
    h2h_candidates = []
    all_round_uids = set([king_uid] + list(challengers.keys())) if king_uid is not None else set(challengers.keys())
    for uid in all_round_uids:
        uid_str = str(uid)
        hotkey = uid_to_hotkey.get(uid, "")
        cb = commitments.get(uid, {}).get("block")
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=cb):
            continue
        if uid in this_round_uids and uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD:
            h2h_candidates.append((uid, state.scores[uid_str]))

    winner_uid, winner_kl = None, float("inf")
    if h2h_candidates:
        h2h_candidates.sort(key=lambda x: x[1])
        best_uid, best_kl = h2h_candidates[0]
        if king_uid is not None and best_uid != king_uid and epsilon_dethroned_by is None:
            winner_uid = king_uid
            winner_kl = state.scores.get(str(king_uid), king_kl)
            logger.info(f"King UID {king_uid} retains crown (no challenger passed epsilon)")
        elif epsilon_dethroned_by is not None:
            # Pre-dethronement integrity check: verify challenger model is still public on HuggingFace
            challenger_model = uid_to_model.get(epsilon_dethroned_by, "")
            try:
                from huggingface_hub import HfApi
                _hf = HfApi()
                _info = _hf.model_info(challenger_model)
                if _info.private:
                    logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} is now private!")
                    winner_uid = king_uid
                    winner_kl = state.scores.get(str(king_uid), king_kl)
                    logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                    state.dq_reasons[str(epsilon_dethroned_by)] = f"Model went private after scoring"
                    epsilon_dethroned_by = None
                else:
                    # Shard-invariant hash check: block re-sharded copies of the king
                    shard_blocked = False
                    try:
                        king_model_name = uid_to_model.get(king_uid, "")
                        if king_model_name and challenger_model:
                            challenger_shard_hash = compute_tensor_metadata_hash(challenger_model)
                            king_shard_hash = compute_tensor_metadata_hash(king_model_name)
                            if challenger_shard_hash and king_shard_hash:
                                if challenger_shard_hash == king_shard_hash:
                                    logger.warning(
                                        f"BLOCKED dethronement: UID {epsilon_dethroned_by} ({challenger_model}) "
                                        f"has identical tensor metadata hash as king UID {king_uid} ({king_model_name}) "
                                        f"— re-sharded copy detected! hash={challenger_shard_hash[:16]}..."
                                    )
                                    winner_uid = king_uid
                                    winner_kl = state.scores.get(str(king_uid), king_kl)
                                    challenger_hotkey = uid_to_hotkey.get(epsilon_dethroned_by, str(epsilon_dethroned_by))
                                    challenger_cb = commitments.get(epsilon_dethroned_by, {}).get("block")
                                    disqualify(
                                        challenger_hotkey,
                                        f"copy: identical weights (shard-invariant hash) to king UID {king_uid} ({king_model_name})",
                                        state.dq_reasons, commit_block=challenger_cb,
                                    )
                                    shard_blocked = True
                                    epsilon_dethroned_by = None
                                else:
                                    logger.info(
                                        f"Shard hash OK: challenger={challenger_shard_hash[:16]}... "
                                        f"king={king_shard_hash[:16]}... (different weights confirmed)"
                                    )
                            else:
                                logger.warning(f"Could not compute shard hash for comparison — allowing dethronement")
                    except Exception as sh_err:
                        logger.warning(f"Shard hash check failed (non-blocking): {sh_err}")

                    # Store shard hash for future reference
                    if not shard_blocked:
                        try:
                            import datetime
                            shard_hashes_path = state.state_dir / "model_hashes_shards.json"
                            shard_data = {}
                            if shard_hashes_path.exists():
                                shard_data = json.loads(shard_hashes_path.read_text())
                            ch_hash = locals().get('challenger_shard_hash')
                            if ch_hash:
                                shard_data[str(epsilon_dethroned_by)] = {
                                    "model_id": challenger_model,
                                    "hash": ch_hash,
                                    "timestamp": datetime.datetime.utcnow().isoformat(),
                                }
                            shard_hashes_path.write_text(json.dumps(shard_data, indent=2))
                        except Exception as e:
                            logger.warning(f"Failed to store shard hash: {e}")

                    if not shard_blocked:
                        winner_uid = epsilon_dethroned_by
                        winner_kl = state.scores.get(str(epsilon_dethroned_by), best_kl)
                        logger.info(f"UID {winner_uid} is new king (paired t-test p<{PAIRED_TEST_ALPHA}), integrity + shard hash check passed")
            except Exception as e:
                logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} integrity check failed: {e}")
                winner_uid = king_uid
                winner_kl = state.scores.get(str(king_uid), king_kl)
                logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                state.dq_reasons[str(epsilon_dethroned_by)] = f"Model not accessible on HuggingFace"
                epsilon_dethroned_by = None
        else:
            winner_uid, winner_kl = best_uid, best_kl

    # ── Build H2H results for dashboard ──
    h2h_results = _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl,
                                     king_per_prompt, uid_to_model)

    # ── Print leaderboard ──
    logger.info(f"H2H ROUND RESULTS (block {current_block}):")
    for rank, (uid, kl) in enumerate(h2h_candidates, 1):
        marker = " ← WINNER" if uid == winner_uid else ""
        is_king = " (king)" if uid == king_uid else ""
        logger.info(f"  #{rank}  UID {uid}: KL={kl:.6f}{marker}{is_king}")

    logger.info("GLOBAL LEADERBOARD:")
    sorted_scores = sorted(state.scores.items(), key=lambda x: x[1])
    for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
        uid = int(uid_str)
        hotkey = uid_to_hotkey.get(uid, "")
        cb = commitments.get(uid, {}).get("block")
        dq = " ⛔ DQ" if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=cb) else ""
        marker = " ← H2H WINNER" if uid == winner_uid else ""
        in_round = " (in round)" if uid in all_round_uids else ""
        logger.info(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{in_round}{dq}")

    return winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, this_round_uids


def _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl,
                       king_per_prompt, uid_to_model):
    """Build H2H result entries for dashboard display."""
    from scipy import stats as _scipy_stats

    h2h_results = []
    for uid, info in models_to_eval.items():
        model_name = info["model"]
        student_data = results.get("students", {}).get(model_name, {})
        kl = student_data.get("kl_global_avg")
        if kl is None or "error" in student_data:
            continue
        is_king = (uid == king_uid)
        vs_king = ""
        t_test_info = None
        if king_h2h_kl is not None and not is_king and king_h2h_kl > 0:
            pct = (king_h2h_kl - kl) / king_h2h_kl * 100
            c_per_prompt = student_data.get("kl_per_prompt")
            if (king_per_prompt and c_per_prompt
                    and len(king_per_prompt) == len(c_per_prompt)
                    and len(king_per_prompt) >= 20):
                deltas = [k - c for k, c in zip(king_per_prompt, c_per_prompt)]
                mean_d = sum(deltas) / len(deltas)
                t_s, p2 = _scipy_stats.ttest_1samp(deltas, 0.0)
                p_val = p2 / 2 if t_s > 0 else 1.0 - p2 / 2
                t_test_info = {"p": round(p_val, 6), "t": round(t_s, 3), "n": len(deltas), "mean_delta": round(mean_d, 6)}
                if p_val < PAIRED_TEST_ALPHA and mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f} DETHRONED)"
                elif mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f}, not significant)"
                else:
                    vs_king = "worse"
            else:
                epsilon_threshold_h2h = king_h2h_kl * (1.0 - EPSILON)
                if kl < epsilon_threshold_h2h:
                    vs_king = f"-{pct:.3f}% (DETHRONED)"
                elif kl < king_h2h_kl:
                    vs_king = f"-{pct:.3f}% (not enough, need >{EPSILON * 100:.0f}%)"
                else:
                    vs_king = "worse"
        entry = {"uid": uid, "model": model_name, "kl": round(kl, 6), "is_king": is_king, "vs_king": vs_king}
        if t_test_info:
            entry["t_test"] = t_test_info
        if info.get("is_reference"):
            entry["is_reference"] = True
            entry["vs_king"] = "baseline (undistilled)"
        h2h_results.append(entry)
    h2h_results.sort(key=lambda x: x["kl"])
    return h2h_results
