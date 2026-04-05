#!/usr/bin/env python3
"""
Remote Validator — King-of-the-Hill Architecture

Design:
  - The "king" is the miner with the best KL score (lowest)
  - Each epoch, only NEW/UNEVALUATED challengers are scored head-to-head vs the king
  - Challengers get MORE prompts (higher confidence) than the broad sweep
  - If a challenger beats the king, it becomes the new king
  - Pre-checks (architecture, hash, integrity) filter out invalid models BEFORE GPU eval
  - Wallet keys never leave this machine; GPU pod has no chain access

Flow:
  1. Read commitments, pre-check all models (arch, hash, integrity)
  2. Identify king (lowest KL from state) and challengers (new/unevaluated)
  3. If challengers exist: evaluate king + challengers head-to-head on GPU
  4. If a challenger beats king: it becomes king
  5. Set weights: king gets 1.0, everyone else 0.0
"""
import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
# Silence noisy libraries
for _lib in ("paramiko", "paramiko.transport", "paramiko.sftp", "urllib3", "httpx"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger("distillation.remote_validator")
logger.setLevel(logging.DEBUG)

from eval.state import ValidatorState, atomic_json_write, log_event
from eval.pod import PodManager, sanitize_gpu_log
from eval.chain import fetch_metagraph, parse_commitments, set_weights
from eval.scoring import (
    load_scores, save_scores,
    load_failures, save_failures, record_failure, reset_failures, is_stale,
    load_disqualified, save_disqualified, disqualify, is_disqualified,
    is_flagged, get_dq_reason, append_score_history,
)
from eval.model_checker import (
    check_model_architecture, verify_model_integrity,
    compute_model_hash, check_duplicate_hash, register_model_hash,
)
from eval.dataset import sample_prompts_from_dataset, format_prompt

# ── Constants ─────────────────────────────────────────────────────────────

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
NETUID = 97
MAX_KL_THRESHOLD = 2.0
MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 1024

EVAL_PROMPTS_FULL = 60    # Full eval: many models, need speed
EVAL_PROMPTS_H2H = 120    # Head-to-head: fewer models, need precision
EPSILON = 0.01             # Legacy fallback if per-prompt data unavailable
PAIRED_TEST_ALPHA = 0.05   # Significance level for paired t-test dethronement
STALE_H2H_EPOCHS = 50      # Re-test if last H2H was >N epochs ago
TOP_N_ALWAYS_INCLUDE = 5   # king + 4 contenders always in eval

# Discord announcement role
DISTIL_ROLE_ID = "1482026585358991571"


# ── Announcement ──────────────────────────────────────────────────────────

def _announce_new_king(new_uid, new_model, new_kl, old_uid, old_model, old_kl, state: ValidatorState):
    """Write a pending announcement to state for async Discord posting."""
    kl_diff_pct = ((old_kl - new_kl) / old_kl * 100) if old_kl > 0 else 0

    # Fetch earnings data
    earnings_line = ""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:3710/api/price", timeout=5)
        price_data = json.loads(resp.read())
        tao_per_day = price_data.get("miners_tao_per_day", 0)
        tao_usd = price_data.get("tao_usd", 0)
        usd_per_day = tao_per_day * tao_usd
        earnings_line = (
            f"\n💰 **Winner earns ~{tao_per_day:.1f} τ/day (${usd_per_day:,.0f}/day)** — "
            f"winner takes all!\n"
        )
    except Exception:
        pass

    role_ping = f"<@&{DISTIL_ROLE_ID}>"
    announcement = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": False,
        "message": (
            f"{role_ping}\n"
            f"## 🏆 New King of Distil SN97!\n\n"
            f"**UID {new_uid}** has dethroned **UID {old_uid}**\n\n"
            f"📊 **KL: {new_kl:.6f}** (previous king scored {old_kl:.6f} last eval)\n"
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
            f"👑 Previous king: [{old_model}](<https://huggingface.co/{old_model}>)\n"
            f"{earnings_line}\n"
            f"Dethronement uses paired t-test (p<0.05) on 120 prompts. "
            f"Check the [mining guide](<https://github.com/unarbos/distil#mining-guide>) to get started.\n\n"
            f"📈 [Live Dashboard](<https://distil.arbos.life>)"
        ),
        "data": {
            "new_uid": new_uid, "new_model": new_model, "new_kl": new_kl,
            "old_uid": old_uid, "old_model": old_model, "old_kl": old_kl,
        },
    }
    state.save_announcement(announcement)
    logger.info(f"Announcement written: UID {new_uid} dethroned UID {old_uid}")


# ── DQ Migration ──────────────────────────────────────────────────────────

def _migrate_dq_entries(state: ValidatorState, commitments: dict):
    """Migrate bare-hotkey and stale bare-UID DQ entries to per-commit format."""
    hotkey_to_block = {
        com["hotkey"]: com["block"]
        for com in commitments.values()
        if "hotkey" in com and "block" in com
    }

    # Migrate bare hotkey → hotkey:block
    migrated = 0
    for key in list(state.dq_reasons.keys()):
        if key.startswith("flag:") or key.isdigit() or ":" in key:
            continue
        if key in hotkey_to_block:
            new_key = f"{key}:{hotkey_to_block[key]}"
            state.dq_reasons[new_key] = state.dq_reasons.pop(key)
            migrated += 1
    if migrated:
        logger.info(f"Migrated {migrated} DQ entries to per-commit format")

    # Scrub stale bare-UID entries
    scrubbed = 0
    for key in list(state.dq_reasons.keys()):
        if not key.isdigit():
            continue
        uid = int(key)
        if uid not in commitments:
            continue
        com = commitments[uid]
        hk = com.get("hotkey", "")
        blk = com.get("block")
        if blk and f"{hk}:{blk}" in state.dq_reasons:
            del state.dq_reasons[key]
            scrubbed += 1
            continue
        current_model = com.get("model", "")
        dq_reason = state.dq_reasons[key]
        if current_model and current_model not in dq_reason:
            logger.info(f"Removing stale bare-UID DQ: UID {uid}")
            del state.dq_reasons[key]
            scrubbed += 1
    if scrubbed:
        logger.info(f"Scrubbed {scrubbed} stale bare-UID DQ entries")


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

        # Check DQ
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

        # Skip expensive HF checks for already-evaluated UIDs with valid scores
        uid_str = str(uid)
        if (uid_str in state.evaluated_uids and uid_str in state.scores
                and state.scores[uid_str] <= MAX_KL_THRESHOLD):
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

        # Integrity check — reset expected hash if miner re-committed
        expected_hash = state.model_hashes.get(str(uid))
        stored_commit_block = state.model_hashes.get(f"{uid}_block")
        if this_commit_block and stored_commit_block and this_commit_block != stored_commit_block:
            # Miner made a new commitment — accept new weights
            logger.info(f"UID {uid}: New commitment at block {this_commit_block} (was {stored_commit_block}), resetting hash")
            expected_hash = None
            state.model_hashes.pop(str(uid), None)
            state.model_hashes.pop(f"{uid}_block", None)
            # Clear old DQ for this commitment
            old_dq_key = f"{hotkey}:{stored_commit_block}"
            if old_dq_key in state.dq_reasons:
                del state.dq_reasons[old_dq_key]
            # Clear from evaluated_uids so they get re-evaluated
            state.evaluated_uids.discard(str(uid))
            state.scores.pop(str(uid), None)
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
            state.save_model_hashes()

        valid_models[uid] = {
            "model": model_repo, "revision": revision,
            "params_b": check.get("params_b", 0),
            "commit_block": commit.get("block", float("inf")),
            "hotkey": hotkey,
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

    # P3: Stale re-tests
    if king_kl > 0 and king_kl < float("inf"):
        stale_threshold = king_kl * 2.0
        p3_candidates = []
        for uid, info in valid_models.items():
            if uid == king_uid or uid in challengers:
                continue
            if info["model"] in state.permanently_bad_models:
                continue
            uid_str = str(uid)
            global_kl = state.scores.get(uid_str)
            if global_kl is None or global_kl <= 0 or global_kl > stale_threshold:
                continue
            h2h_record = state.h2h_tested_against_king.get(uid_str, {})
            if h2h_record.get("king_uid") != king_uid:
                continue  # untested — handled by P1
            epochs_since = epoch_count - h2h_record.get("epoch", 0)
            if epochs_since > STALE_H2H_EPOCHS:
                p3_candidates.append((uid, global_kl, epochs_since))
        if p3_candidates:
            p3_candidates.sort(key=lambda x: x[1])
            uid, kl, age = p3_candidates[0]
            challengers[uid] = valid_models[uid]
            logger.info(f"🎯 SMART CHALLENGER: UID {uid} — P3: stale re-test ({age} epochs, KL={kl:.6f})")

    return challengers


def _add_top5_contenders(challengers, valid_models, state: ValidatorState, king_uid):
    """Always include top-5 contenders from leaderboard in maintenance mode."""
    if state.top4_leaderboard.get("phase") != "maintenance" or king_uid is None:
        return
    contenders_added = 0
    for contender in (state.top4_leaderboard.get("contenders") or [])[:TOP_N_ALWAYS_INCLUDE - 1]:
        c_uid = contender.get("uid")
        if c_uid is not None and c_uid != king_uid and c_uid in valid_models and c_uid not in challengers:
            challengers[c_uid] = valid_models[c_uid]
            contenders_added += 1
    if contenders_added:
        logger.info(f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) to eval")


def _cap_challengers(challengers, state: ValidatorState, king_uid):
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

def _check_models_exist(models_to_eval, uid_to_hotkey, state: ValidatorState, commitments: dict) -> list:
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
                    eval_script_remote: str, resuming_round: bool = False):
    """Execute the GPU eval on the remote pod and return results.

    Handles: prompt upload, eval script upload, progress polling,
    result download, and timeout management.

    Returns the parsed results dict or None on failure.
    """
    import threading
    import concurrent.futures

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

    # Clean stale artifacts; handle resume vs new round
    try:
        pod.exec("rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_progress.json")
        if resuming_round:
            logger.info("Resuming round (keeping eval_results.json + teacher_cache.pt on pod)")
        else:
            pod.exec("rm -f /home/eval_results.json /home/teacher_cache.pt")
            logger.info("New round (cleared eval_results.json + teacher_cache.pt)")
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

    # Build eval command
    student_list = ",".join(models_to_eval[uid]["model"] for uid in ordered_uids)
    king_flag = ""
    vllm_flag = " --no-vllm"
    if use_vllm:
        vllm_flag = " --vllm-gpu-util 0.45"
        if not is_full_eval and king_uid is not None and king_uid in models_to_eval:
            king_flag = f" --king {models_to_eval[king_uid]['model']}"

    eval_cmd = (
        f"cd /home && python3 -u pod_eval.py "
        f"--teacher {TEACHER_MODEL} "
        f"--students {student_list} "
        f"--prompts prompts.json "
        f"--output eval_results.json "
        f"--max-prompt-len {MAX_PROMPT_TOKENS} "
        f"--max-new-tokens {MAX_NEW_TOKENS} "
        f"--max-params-b {max_params_b} "
        f"--teacher-logits /home/teacher_cache.pt "
        f"--resume"
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
                log_result = pod.exec("tail -100 /home/eval_output.log 2>/dev/null || echo ''")
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
    EVAL_TIMEOUT = (n_eval_models * 10 + 30) * 60
    logger.info(f"Running eval ({n_eval_models} models, {n_prompts} prompts, timeout={EVAL_TIMEOUT // 60}m)")
    log_event(f"Running eval on pod: king vs {n_eval_models - 1} challengers, {n_prompts} prompts", state_dir=str(state.state_dir))
    eval_env = {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pod.exec, eval_cmd, env=eval_env)
            try:
                result = future.result(timeout=EVAL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error(f"Eval timed out after {EVAL_TIMEOUT}s — killing")
                try:
                    pod.exec("pkill -9 -f pod_eval.py; echo killed")
                except Exception:
                    pass
                result = {"stdout": "", "stderr": "timeout", "exit_code": -1, "success": False}
    except Exception as e:
        logger.error(f"lium.exec EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
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
                    epoch_count, is_full_eval):
    """Process eval results: update scores, run paired t-test, crown winner.

    Returns (winner_uid, winner_kl, h2h_results_list).
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
        logger.warning(f"King UID {king_uid} did not produce a score — retaining crown by default")

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

            if (king_per_prompt and challenger_per_prompt
                    and len(king_per_prompt) == len(challenger_per_prompt)
                    and len(king_per_prompt) >= 20):
                deltas = [k - c for k, c in zip(king_per_prompt, challenger_per_prompt)]
                mean_delta = sum(deltas) / len(deltas)
                t_stat, p_two = _scipy_stats.ttest_1samp(deltas, 0.0)
                p_value = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
                n_test = len(deltas)
                pct_better = (mean_delta / king_new_kl * 100) if king_new_kl > 0 else 0

                if p_value < PAIRED_TEST_ALPHA and mean_delta > 0:
                    logger.info(f"UID {uid} DETHRONED king UID {king_uid}! "
                                f"p={p_value:.6f}, delta={mean_delta:.6f} ({pct_better:.2f}%), t={t_stat:.3f}, n={n_test}")
                    if epsilon_dethroned_by is None or challenger_kl < state.scores.get(str(epsilon_dethroned_by), float("inf")):
                        epsilon_dethroned_by = uid
                elif mean_delta > 0:
                    logger.info(f"UID {uid}: better but not significant (p={p_value:.4f}, delta={mean_delta:.6f})")
                else:
                    logger.info(f"UID {uid}: worse than king (delta={mean_delta:.6f}, p={p_value:.4f})")
            else:
                # Legacy epsilon fallback
                if challenger_kl < epsilon_threshold:
                    logger.info(f"UID {uid} DETHRONED king UID {king_uid}! KL={challenger_kl:.6f} < {epsilon_threshold:.6f} [legacy epsilon]")
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
            winner_uid = epsilon_dethroned_by
            winner_kl = state.scores.get(str(epsilon_dethroned_by), best_kl)
            logger.info(f"UID {winner_uid} is new king (paired t-test p<{PAIRED_TEST_ALPHA})")
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
        h2h_results.append(entry)
    h2h_results.sort(key=lambda x: x["kl"])
    return h2h_results


# ── Post-processing ───────────────────────────────────────────────────────

# ── Chat server management ────────────────────────────────────────────────

# Chat-king pod config (from lium)
CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST", "91.224.44.81")
CHAT_POD_SSH_PORT = os.environ.get("CHAT_POD_SSH_PORT", "20300")
CHAT_POD_APP_PORT = 8100


def _chat_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the chat-king pod via SSH."""
    import subprocess
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-p", CHAT_POD_SSH_PORT, f"root@{CHAT_POD_HOST}", cmd,
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception as e:
        logger.warning(f"Chat pod SSH failed: {e}")
        return ""


def _restart_chat_server(model_name: str):
    """Kill old chat server and start with new king model."""
    logger.info(f"Restarting chat server with new king: {model_name}")
    try:
        _chat_ssh("pkill -f chat_server.py || true", timeout=10)
        time.sleep(2)
        _chat_ssh(
            f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_APP_PORT} > /root/chat.log 2>&1 &",
            timeout=10,
        )
        logger.info("Chat server restart initiated")
    except Exception as e:
        logger.warning(f"Failed to restart chat server: {e}")


def _ensure_chat_server_running(model_name: str):
    """Check if chat server is running; start it if not."""
    try:
        stdout = _chat_ssh("pgrep -f chat_server.py || echo not_running", timeout=10)
        if "not_running" in stdout:
            logger.info(f"Chat server not running, starting with {model_name}")
            _chat_ssh(
                f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_APP_PORT} > /root/chat.log 2>&1 &",
                timeout=10,
            )
    except Exception as e:
        logger.debug(f"Chat server check failed: {e}")


def update_h2h_state(state: ValidatorState, h2h_results, king_uid, winner_uid,
                     king_h2h_kl, king_kl, king_per_prompt, current_block,
                     n_prompts, is_full_eval, uid_to_model, valid_models,
                     challengers, epoch_count, disqualified):
    """Update H2H state files: latest, history, tested-against-king."""

    n_challenger_results = sum(1 for r in h2h_results if not r.get("is_king"))
    if n_challenger_results == 0:
        logger.info("All challengers failed — skipping H2H round save")
        return

    king_changed = winner_uid != king_uid if king_uid is not None else False
    effective_king_uid = winner_uid if winner_uid is not None else king_uid
    effective_king_kl = king_h2h_kl
    effective_king_model = uid_to_model.get(effective_king_uid, valid_models.get(effective_king_uid, {}).get("model", ""))
    if king_changed and winner_uid is not None:
        for r in h2h_results:
            if r["uid"] == winner_uid:
                effective_king_kl = r.get("kl", king_h2h_kl)
                break

    h2h_round = {
        "block": current_block, "timestamp": time.time(),
        "king_uid": effective_king_uid, "king_model": effective_king_model,
        "prev_king_uid": king_uid,
        "king_h2h_kl": round(effective_king_kl, 6) if effective_king_kl else None,
        "king_global_kl": round(king_kl, 6),
        "epsilon": EPSILON,
        "epsilon_threshold": round(king_h2h_kl * (1.0 - EPSILON), 6) if king_h2h_kl else None,
        "paired_test_alpha": PAIRED_TEST_ALPHA,
        "dethrone_method": "paired_t_test" if king_per_prompt else "legacy_epsilon",
        "n_prompts": n_prompts, "results": h2h_results,
        "king_changed": king_changed,
        "new_king_uid": winner_uid if king_changed else None,
        "type": "full_eval" if is_full_eval else "h2h",
    }

    state.h2h_latest = h2h_round
    # Replace preliminary entries
    state.h2h_history = [h for h in state.h2h_history if not (h.get("block") == current_block and h.get("_preliminary"))]
    state.h2h_history.append(h2h_round)
    state.h2h_history = state.h2h_history[-50:]
    state.save_h2h()

    # Auto-restart chat server with new king model
    if king_changed and effective_king_model:
        _restart_chat_server(effective_king_model)
    elif not king_changed:
        # Even if king didn't change, ensure chat server is running with correct model
        _ensure_chat_server_running(effective_king_model)

    # Update tested-against-king tracker
    if king_uid is not None:
        for uid in challengers:
            uid_str = str(uid)
            if uid_str in state.scores and state.scores[uid_str] > 0:
                state.h2h_tested_against_king[uid_str] = {
                    "king_uid": king_uid, "epoch": epoch_count,
                    "block": current_block, "kl": round(state.scores[uid_str], 6),
                    "model": challengers[uid].get("model", ""), "timestamp": time.time(),
                }
        atomic_json_write(state._path("h2h_tested_against_king.json"),
                          state.h2h_tested_against_king, indent=2)


def update_model_tracking(state: ValidatorState, models_to_eval, current_block,
                          king_kl, disqualified):
    """Update persistent model score history and permanently bad models."""
    for uid, info in models_to_eval.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.scores and state.scores[uid_str] > 0:
            kl = state.scores[uid_str]
            prev = state.model_score_history.get(model_name, {})
            if kl <= MAX_KL_THRESHOLD:
                prev_best = prev.get("best_kl", float("inf"))
                if kl < prev_best:
                    state.model_score_history[model_name] = {
                        **prev, "best_kl": round(kl, 6), "uid": uid,
                        "block": current_block, "timestamp": time.time(),
                    }
            else:
                prev_worst = prev.get("worst_kl", 0)
                if kl > prev_worst:
                    state.model_score_history[model_name] = {
                        **prev, "worst_kl": round(kl, 6), "uid": uid,
                        "block": current_block, "timestamp": time.time(),
                    }
                if "best_kl" not in state.model_score_history.get(model_name, {}):
                    state.model_score_history.setdefault(model_name, {})["best_kl"] = round(kl, 6)

    # Permanently bad models
    if king_kl > 0 and king_kl < float("inf"):
        perm_bad_threshold = king_kl * 10.0
        newly_banned = []
        for uid, info in models_to_eval.items():
            uid_str = str(uid)
            if uid_str in state.scores and state.scores[uid_str] > perm_bad_threshold:
                model_name = info["model"]
                if model_name not in state.permanently_bad_models:
                    state.permanently_bad_models.add(model_name)
                    newly_banned.append(f"{model_name} (UID {uid}, KL={state.scores[uid_str]:.4f})")
        if newly_banned:
            logger.info(f"🚫 Added {len(newly_banned)} models to permanently_bad_models")

    state.save_model_tracking()


def update_top4_leaderboard(state: ValidatorState, winner_uid, king_uid, king_kl,
                            h2h_results, uid_to_model, valid_models, current_block,
                            epoch_count, disqualified):
    """Update the top-4 leaderboard (initial eval → maintenance transition)."""
    try:
        if state.top4_leaderboard.get("phase") == "initial_eval":
            # Check if all models tested
            untested_count = 0
            tested_results = []
            for uid_str, score in state.scores.items():
                if score <= 0 or score > MAX_KL_THRESHOLD:
                    continue
                if int(uid_str) in disqualified:
                    continue
                record = state.h2h_tested_against_king.get(uid_str, {})
                if record.get("king_uid") == king_uid and record.get("kl"):
                    tested_results.append((uid_str, record["kl"], record.get("model", "")))
                else:
                    untested_count += 1

            if untested_count == 0 and len(tested_results) >= 4:
                tested_results.sort(key=lambda x: x[1])
                state.top4_leaderboard["king"] = {
                    "uid": int(tested_results[0][0]), "model": tested_results[0][2],
                    "h2h_kl": round(tested_results[0][1], 6), "block": current_block,
                }
                state.top4_leaderboard["contenders"] = [
                    {"uid": int(tested_results[i][0]), "model": tested_results[i][2],
                     "h2h_kl": round(tested_results[i][1], 6), "block": current_block}
                    for i in range(1, min(4, len(tested_results)))
                ]
                state.top4_leaderboard["phase"] = "maintenance"
                state.top4_leaderboard["initial_eval_complete"] = True
                state.top4_leaderboard["completed_at"] = time.time()
                state.top4_leaderboard["completed_block"] = current_block
                logger.info(f"👑 TOP-4 INITIAL EVAL COMPLETE")
            else:
                logger.info(f"📊 Initial eval: {len(tested_results)} tested, {untested_count} remaining")

        elif state.top4_leaderboard.get("phase") == "maintenance":
            actual_king = winner_uid if winner_uid is not None else king_uid
            king_model = uid_to_model.get(actual_king, valid_models.get(actual_king, {}).get("model", "unknown"))
            king_kl_lb = next((r["kl"] for r in h2h_results if r["uid"] == actual_king), state.scores.get(str(actual_king), 999))

            state.top4_leaderboard["king"] = {
                "uid": actual_king, "model": king_model,
                "h2h_kl": round(king_kl_lb, 6) if isinstance(king_kl_lb, float) else king_kl_lb,
                "block": current_block,
            }
            contenders = []
            for r in h2h_results:
                if r["uid"] == actual_king:
                    continue
                if int(r["uid"]) in disqualified:
                    continue
                contenders.append({
                    "uid": r["uid"], "model": r["model"],
                    "h2h_kl": round(r["kl"], 6), "block": current_block,
                })
                if len(contenders) >= 4:
                    break
            state.top4_leaderboard["contenders"] = contenders

        state.save_top4()
        top4_str = ", ".join(
            f"#{i+1} UID {e['uid']} (KL={e['h2h_kl']})"
            for i, e in enumerate([state.top4_leaderboard.get('king', {})] + (state.top4_leaderboard.get('contenders') or []))
            if e and e.get('uid') is not None
        )
        if top4_str:
            logger.info(f"📊 TOP-4: {top4_str}")
    except Exception as e:
        logger.warning(f"Top-4 leaderboard error (non-fatal): {e}")


# ── Main Loop ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--network", default="finney")
@click.option("--netuid", type=int, default=NETUID)
@click.option("--wallet-name", default="affine")
@click.option("--hotkey-name", default="validator")
@click.option("--wallet-path", default="~/.bittensor/wallets/")
@click.option("--lium-api-key", required=True, envvar="LIUM_API_KEY")
@click.option("--lium-pod-name", default="distil-validator")
@click.option("--state-dir", default="state")
@click.option("--max-params-b", type=float, default=5.25)
@click.option("--tempo", type=int, default=360, help="Seconds between epochs")
@click.option("--once", is_flag=True, help="Run one epoch and exit (for testing)")
@click.option("--use-vllm", is_flag=True, default=False, envvar="USE_VLLM",
              help="Use vLLM-accelerated evaluation")
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         lium_api_key, lium_pod_name, state_dir, max_params_b, tempo, once, use_vllm):
    """Run the distillation validator with king-of-the-hill evaluation."""
    import bittensor as bt
    from lium import Lium, Config

    # ── Init state ──
    state = ValidatorState(state_dir)
    state.load()

    # ── Init chain ──
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    subtensor = bt.Subtensor(network=network)

    # ── Init pod ──
    cfg = Config(api_key=lium_api_key, ssh_key_path=Path.home() / ".ssh" / "id_ed25519")
    lium = Lium(config=cfg)
    pod = PodManager(lium, pod_name=lium_pod_name)
    pod.connect()

    # ── Upload eval script ──
    eval_script = "scripts/pod_eval_vllm.py"
    eval_script_remote = "/home/pod_eval.py"
    pod.upload(eval_script, eval_script_remote, max_attempts=5)

    # ── Ensure pod deps ──
    pod.ensure_dependencies(TEACHER_MODEL)

    epoch_count = 0

    while True:
        try:
            epoch_start = time.time()
            epoch_count += 1
            logger.info(f"\n=== EPOCH {epoch_count} ===")
            log_event(f"Starting epoch {epoch_count}", state_dir=state_dir)

            # ── Clear stale eval progress ──
            if state.eval_progress.get("active"):
                age_min = (time.time() - state.eval_progress.get("started_at", 0)) / 60
                if age_min > 30:
                    logger.warning(f"STALE ROUND: active for {age_min:.0f}m — clearing")
                    state.save_progress({"active": False, "stale_cleared": True, "stale_age_min": round(age_min, 1)})
                    state.clear_round()

            # ── Fetch chain state ──
            try:
                metagraph, current_block, current_block_hash = fetch_metagraph(subtensor, netuid)
                n_uids = int(metagraph.n)
                revealed = subtensor.get_all_revealed_commitments(netuid)
                logger.info(f"Block {current_block}, n={n_uids}, {len(revealed)} revealed")
            except Exception as chain_err:
                logger.error(f"Chain unreachable: {chain_err}, sleeping 5min")
                log_event(f"Chain unreachable: {str(chain_err)[:150]}, retrying in 5min", level="error", state_dir=state_dir)
                time.sleep(300)
                continue

            commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
            logger.info(f"Found {len(commitments)} miner commitments")

            if not commitments:
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── DQ migration ──
            _migrate_dq_entries(state, commitments)

            # ── State validation ──
            issues = state.validate_consistency(uid_to_hotkey, commitments, MAX_KL_THRESHOLD)
            if issues:
                state.save()
                logger.info(f"State auto-repaired ({len(issues)} issues)")

            # Update hotkey map
            state.uid_hotkey_map = {str(k): v for k, v in uid_to_hotkey.items()}

            # ── Phase 1: Pre-check all models ──
            valid_models, disqualified = precheck_all_models(
                commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b
            )

            n_valid = len(valid_models)
            n_dq = len(disqualified)
            n_total = len(commitments)
            log_event(f"Prechecked {n_total} models: {n_valid} valid, {n_dq} DQ, {n_total - n_valid - n_dq} error", state_dir=state_dir)

            if not valid_models:
                logger.info("No valid models after pre-checks")
                state.save()
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── Phase 2: Identify king and challengers ──
            king_uid = None
            king_kl = float("inf")

            # King from h2h_latest (authoritative)
            if state.h2h_latest:
                h2h_king = state.h2h_latest.get("king_uid")
                if h2h_king is not None and h2h_king in valid_models:
                    king_uid = h2h_king
                    king_kl = state.scores.get(str(h2h_king), float("inf"))
                    logger.info(f"King from h2h_latest: UID {king_uid} (KL={king_kl:.6f})")

            # Fallback: lowest score
            if king_uid is None:
                for uid in valid_models:
                    uid_str = str(uid)
                    if uid_str in state.scores and state.scores[uid_str] <= MAX_KL_THRESHOLD:
                        if state.scores[uid_str] < king_kl:
                            king_kl = state.scores[uid_str]
                            king_uid = uid
                if king_uid is not None:
                    logger.info(f"King from scores fallback: UID {king_uid} (KL={king_kl:.6f})")

            challengers = select_challengers(valid_models, state, king_uid, king_kl, epoch_count)
            challengers_before_top5 = set(challengers.keys())
            _add_top5_contenders(challengers, valid_models, state, king_uid)
            _cap_challengers(challengers, state, king_uid)

            has_new_challengers = len(challengers_before_top5) > 0
            if not challengers or not has_new_challengers:
                logger.info(f"No new challengers, king UID {king_uid} (KL={king_kl:.6f}) holds")
                if king_uid is not None:
                    weights = [0.0] * max(n_uids, king_uid + 1)
                    weights[king_uid] = 1.0
                    set_weights(subtensor, wallet, netuid, n_uids, weights, king_uid)
                state.save()
                if once:
                    break
                time.sleep(60)
                continue

            # ── Phase 3: GPU evaluation ──
            models_to_eval = {}
            is_full_eval = state.top4_leaderboard.get("phase") == "initial_eval"
            if not is_full_eval and king_uid is not None and king_uid in valid_models:
                models_to_eval[king_uid] = valid_models[king_uid]
            for uid, info in challengers.items():
                models_to_eval[uid] = info

            n_challengers_in_eval = sum(1 for uid in models_to_eval if uid != king_uid)
            if n_challengers_in_eval == 0:
                logger.info(f"No challengers in eval batch — king UID {king_uid} holds")
                state.save()
                if once:
                    break
                time.sleep(60)
                continue

            n_prompts = EVAL_PROMPTS_FULL if is_full_eval else EVAL_PROMPTS_H2H
            logger.info(f"H2H: king=UID {king_uid} vs {n_challengers_in_eval} challengers ({n_prompts} prompts)")
            challenger_uids_list = [uid for uid in models_to_eval if uid != king_uid]
            log_event(f"Starting h2h round {epoch_count}, king=UID {king_uid}, challengers={challenger_uids_list}", state_dir=state_dir)

            # Model existence check
            removed = _check_models_exist(models_to_eval, uid_to_hotkey, state, commitments)
            if removed:
                logger.info(f"Removed {len(removed)} deleted models")
                if not models_to_eval:
                    state.save()
                    if once:
                        break
                    time.sleep(60)
                    continue

            # Round resumption
            resuming_round = False
            if state.current_round.get("prompts"):
                saved_models = set(state.current_round.get("model_names", []))
                current_models = set(info["model"] for info in models_to_eval.values())
                if saved_models & current_models:
                    prompt_texts = state.current_round["prompts"]
                    resuming_round = True
                    logger.info(f"RESUMING incomplete round ({len(prompt_texts)} prompts)")

            if not resuming_round:
                epoch_prompts = sample_prompts_from_dataset(n_prompts, current_block, block_hash=current_block_hash)
                prompt_texts = [format_prompt(p) for p in epoch_prompts]

            # Save round state for crash recovery
            state.current_round = {
                "started_at": time.time(), "block": current_block,
                "block_hash": current_block_hash, "king_uid": king_uid,
                "model_names": [info["model"] for info in models_to_eval.values()],
                "prompts": prompt_texts,
            }
            state.save_round()

            # Run eval on pod
            results = run_eval_on_pod(
                pod, models_to_eval, king_uid, n_prompts, prompt_texts,
                state, max_params_b, is_full_eval, use_vllm,
                eval_script, eval_script_remote,
                resuming_round=resuming_round,
            )
            if results is None:
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── Persist raw results immediately (crash resilience) ──
            uid_to_model = {uid: m["model"] for uid, m in models_to_eval.items()}
            model_to_uid = {m: uid for uid, m in uid_to_model.items()}
            try:
                imm_h2h = []
                imm_king_kl = None
                for mn, sr in results.get("students", {}).items():
                    mu = model_to_uid.get(mn)
                    if mu is None or "error" in sr:
                        continue
                    mkl = sr.get("kl_global_avg")
                    if mkl is None:
                        continue
                    ik = (mu == king_uid)
                    if ik:
                        imm_king_kl = mkl
                    imm_h2h.append({"uid": mu, "model": mn, "kl": round(mkl, 6), "is_king": ik, "vs_king": ""})
                imm_h2h.sort(key=lambda x: x["kl"])
                if imm_h2h:
                    imm_round = {
                        "block": current_block, "timestamp": time.time(),
                        "king_uid": king_uid, "prev_king_uid": king_uid,
                        "king_h2h_kl": round(imm_king_kl, 6) if imm_king_kl else None,
                        "king_global_kl": round(king_kl, 6),
                        "n_prompts": n_prompts, "results": imm_h2h,
                        "king_changed": False, "new_king_uid": None,
                        "type": "full_eval" if is_full_eval else "h2h",
                        "_preliminary": True,
                    }
                    state.h2h_history.append(imm_round)
                    state.h2h_history = state.h2h_history[-50:]
                    atomic_json_write(state._path("h2h_history.json"), state.h2h_history, indent=2)
                    logger.info(f"Preliminary H2H ({len(imm_h2h)} results) persisted")
            except Exception as e:
                logger.warning(f"Failed to persist immediate results: {e}")

            # ── Phase 4: Process results ──
            (winner_uid, winner_kl, h2h_results,
             king_h2h_kl, king_per_prompt, this_round_uids) = process_results(
                results, models_to_eval, king_uid, state,
                uid_to_hotkey, commitments, n_prompts, current_block, king_kl,
                epoch_count, is_full_eval,
            )

            # Set weights
            if winner_uid is not None:
                weights = [0.0] * max(n_uids, winner_uid + 1)
                weights[winner_uid] = 1.0
                set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid)
            else:
                logger.info("No valid miners — skipping weight setting")

            # ── Persist state ──
            state.save()

            # ── Update H2H state ──
            update_h2h_state(
                state, h2h_results, king_uid, winner_uid, king_h2h_kl, king_kl,
                king_per_prompt, current_block, n_prompts, is_full_eval,
                uid_to_model, valid_models, challengers, epoch_count, disqualified,
            )

            # ── Update model tracking ──
            update_model_tracking(state, models_to_eval, current_block, king_kl, disqualified)

            # ── Score history ──
            valid_scores = {
                uid_str: kl for uid_str, kl in state.scores.items()
                if uid_str not in state.dq_reasons and 0 < kl <= MAX_KL_THRESHOLD
            }
            if valid_scores:
                append_score_history(
                    block=current_block, timestamp=time.time(),
                    scores=valid_scores, king_uid=winner_uid, state_dir=state.state_dir,
                )

            # ── Update top-4 leaderboard ──
            update_top4_leaderboard(
                state, winner_uid, king_uid, king_kl, h2h_results,
                uid_to_model, valid_models, current_block, epoch_count, disqualified,
            )

            # ── Round complete ──
            state.clear_round()
            state.save_progress({"active": False})

            # ── Pod cleanup ──
            try:
                pod.post_eval_cleanup(TEACHER_MODEL)
                pod.resume_background_tasks()
            except Exception as cleanup_err:
                log_event(f"Pod cleanup error: {str(cleanup_err)[:100]}", level="warn", state_dir=state_dir)
                logger.warning(f"Pod cleanup error: {cleanup_err}")

            # ── Announcement ──
            if winner_uid is not None and winner_uid != king_uid and king_uid is not None:
                new_king_model = uid_to_model.get(winner_uid, valid_models.get(winner_uid, {}).get("model", "unknown"))
                old_king_model = uid_to_model.get(king_uid, valid_models.get(king_uid, {}).get("model", "unknown"))
                old_kl = king_h2h_kl if king_h2h_kl is not None else king_kl
                try:
                    _announce_new_king(winner_uid, new_king_model, winner_kl,
                                       king_uid, old_king_model, old_kl, state)
                except Exception as ann_err:
                    logger.warning(f"Announcement failed: {ann_err}")

            elapsed = time.time() - epoch_start
            logger.info(f"Epoch complete in {elapsed:.0f}s")

            # Log round completion
            winner_model = uid_to_model.get(winner_uid, "unknown") if winner_uid else "none"
            w_kl = state.scores.get(str(winner_uid), 0) if winner_uid else 0
            king_changed = winner_uid is not None and winner_uid != king_uid and king_uid is not None
            if king_changed:
                log_event(f"Round complete. New king: UID {winner_uid} ({winner_model}), KL={w_kl:.6f}. Dethroned UID {king_uid}.", state_dir=state_dir)
            else:
                log_event(f"Round complete. Winner: UID {winner_uid}, KL={w_kl:.6f}. Weights set.", state_dir=state_dir)

            if once:
                break
            logger.info("Checking for new challengers immediately...")

        except KeyboardInterrupt:
            logger.info("Shutting down")
            state.save()
            break
        except Exception as e:
            logger.error(f"EPOCH ERROR: {e}")
            log_event(f"Epoch error: {str(e)[:200]}", level="error", state_dir=state_dir)
            import traceback
            traceback.print_exc()
            state.save()
            if once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
