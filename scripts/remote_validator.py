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
from eval.chain import fetch_metagraph, parse_commitments, set_weights
from eval.scoring import (
    load_scores, save_scores,
    load_failures, save_failures, record_failure, reset_failures, is_stale,
    load_disqualified, save_disqualified, disqualify, is_disqualified,
    is_flagged, get_dq_reason, append_score_history,
)
from eval.dataset import sample_prompts_from_dataset, format_prompt

# ── Validator submodules ──────────────────────────────────────────────────
from scripts.validator.config import (
    TEACHER_MODEL, NETUID, MAX_KL_THRESHOLD,
    EVAL_PROMPTS_FULL, EVAL_PROMPTS_H2H,
    REFERENCE_MODEL, REFERENCE_UID,
)
from scripts.validator.eval_orchestrator import (
    precheck_all_models, select_challengers, add_top5_contenders,
    cap_challengers, check_models_exist, run_eval_on_pod, process_results,
)
from scripts.validator.pod_manager import init_pod
from scripts.validator.chain import write_api_commitments_cache
from scripts.validator.state_manager import (
    migrate_dq_entries, update_h2h_state, update_model_tracking,
    update_top4_leaderboard,
)
from scripts.validator.announcements import announce_new_king


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

    # ── Log git version ──
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_msg = subprocess.check_output(
            ["git", "log", "--oneline", "-1"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        print(f"[validator] Git: {git_msg}", flush=True)
        logger.info(f"Running commit: {git_hash}")
    except Exception:
        pass

    # ── Init state ──
    state = ValidatorState(state_dir)
    state.load()

    # ── Init chain ──
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    subtensor = bt.Subtensor(network=network)

    # ── Init pod ──
    eval_script = "scripts/pod_eval_vllm.py"
    eval_script_remote = "/home/pod_eval.py"
    cfg = Config(api_key=lium_api_key, ssh_key_path=Path.home() / ".ssh" / "id_ed25519")
    lium = Lium(config=cfg)
    pod = init_pod(lium, lium_pod_name, eval_script, eval_script_remote, TEACHER_MODEL)

    epoch_count = 0

    while True:
        try:
            epoch_start = time.time()
            epoch_count += 1
            # Re-force our logging level after bittensor clobbers it
            logging.getLogger().setLevel(logging.INFO)
            logger.setLevel(logging.DEBUG)
            print(f"\n[validator] === EPOCH {epoch_count} ===", flush=True)
            logger.info(f"=== EPOCH {epoch_count} ===")
            log_event(f"Starting epoch {epoch_count}", state_dir=state_dir)

            # ── Orphan cleanup: remove UIDs from evaluated_uids that have no score ──
            orphans = [uid for uid in list(state.evaluated_uids) if uid not in state.scores]
            if orphans:
                for uid in orphans:
                    state.evaluated_uids.discard(uid)
                state.save_model_tracking()
                logger.info(f"Cleaned {len(orphans)} orphaned UIDs from evaluated_uids")

            # ── Clear stale eval progress ──
            if state.eval_progress.get("active"):
                age_min = (time.time() - state.eval_progress.get("started_at", 0)) / 60
                if age_min > 30:
                    logger.warning(f"STALE ROUND: active for {age_min:.0f}m — clearing")
                    state.save_progress({"active": False, "stale_cleared": True, "stale_age_min": round(age_min, 1)})
                    state.clear_round()

            # ── Fetch chain state ──
            print("[validator] Fetching chain state...", flush=True)
            try:
                metagraph, current_block, current_block_hash = fetch_metagraph(subtensor, netuid)
                n_uids = int(metagraph.n)
                revealed = subtensor.get_all_revealed_commitments(netuid)
                print(f"[validator] Block {current_block}, n={n_uids}, {len(revealed)} revealed", flush=True)
                logger.info(f"Block {current_block}, n={n_uids}, {len(revealed)} revealed")
            except Exception as chain_err:
                logger.error(f"Chain unreachable: {chain_err}, sleeping 5min")
                log_event(f"Chain unreachable: {str(chain_err)[:150]}, retrying in 5min", level="error", state_dir=state_dir)
                time.sleep(300)
                continue

            commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
            write_api_commitments_cache(commitments, state_dir)
            logger.info(f"Found {len(commitments)} miner commitments")

            if not commitments:
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── DQ migration ──
            migrate_dq_entries(state, commitments)

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
            log_event(f"select_challengers returned {len(challengers)} (P1/P3), king={king_uid}", state_dir=state_dir)
            add_top5_contenders(challengers, valid_models, state, king_uid)
            cap_challengers(challengers, state, king_uid)

            has_new_challengers = len(challengers_before_top5) > 0
            top5_only = not has_new_challengers and len(challengers) > 0
            if not challengers:
                log_event(f"No challengers at all (before_top5={len(challengers_before_top5)}, after_all={len(challengers)})", state_dir=state_dir)
                logger.info(f"No challengers, king UID {king_uid} (KL={king_kl:.6f}) holds")
                if king_uid is not None:
                    weights = [0.0] * max(n_uids, king_uid + 1)
                    weights[king_uid] = 1.0
                    set_weights(subtensor, wallet, netuid, n_uids, weights, king_uid)
                state.save()
                if once:
                    break
                logger.info(f"No challengers — sleeping {tempo}s before next epoch")
                time.sleep(tempo)
                continue

            if top5_only:
                log_event(f"Top-5 only round: {len(challengers)} contender(s), no new P1/P3", state_dir=state_dir)
                logger.info(f"Running top-5-only round with {len(challengers)} contender(s)")

            # ── Phase 3: GPU evaluation ──
            models_to_eval = {}
            is_full_eval = state.top4_leaderboard.get("phase") == "initial_eval"
            if not is_full_eval and king_uid is not None and king_uid in valid_models:
                models_to_eval[king_uid] = valid_models[king_uid]
            for uid, info in challengers.items():
                models_to_eval[uid] = info

            # Inject reference (base) model for baseline comparison
            if REFERENCE_MODEL and REFERENCE_UID not in models_to_eval:
                models_to_eval[REFERENCE_UID] = {
                    "model": REFERENCE_MODEL,
                    "commit_block": 0,
                    "hotkey": "reference",
                    "is_reference": True,
                }

            n_challengers_in_eval = sum(1 for uid in models_to_eval if uid != king_uid and uid != REFERENCE_UID)
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
            removed = check_models_exist(models_to_eval, uid_to_hotkey, state, commitments)
            if removed:
                logger.info(f"Removed {len(removed)} deleted models")
                if not models_to_eval:
                    state.save()
                    if once:
                        break
                    time.sleep(60)
                    continue

            # Fresh prompts every round — no resume
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
                        "block": current_block, "block_hash": current_block_hash, "timestamp": time.time(),
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
                epoch_count, is_full_eval, epoch_start_time=epoch_start,
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
                block_hash=current_block_hash, epoch_start_time=epoch_start,
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
                    uid_to_hotkey=uid_to_hotkey,
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
                winner_entry = next((r for r in h2h_results if r.get("uid") == winner_uid), {})
                winner_tt = winner_entry.get("t_test") if isinstance(winner_entry.get("t_test"), dict) else {}
                try:
                    announce_new_king(winner_uid, new_king_model, winner_kl,
                                      king_uid, old_king_model, old_kl, state,
                                      paired_prompts=winner_entry.get("paired_prompts") or winner_entry.get("prompts_scored"),
                                      total_prompts=winner_entry.get("prompts_total") or n_prompts,
                                      p_value=winner_tt.get("p"))
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
