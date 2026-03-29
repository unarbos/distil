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
logger = logging.getLogger("distillation.remote_validator")
logger.setLevel(logging.DEBUG)

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
NETUID = 97
MAX_KL_THRESHOLD = 2.0
MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 1024

# Prompts per head-to-head evaluation (king + challenger on same prompts)
EVAL_PROMPTS = 40
# Epsilon: challenger must beat king by this relative margin to dethrone
# e.g., 0.01 = challenger KL must be < king_kl * 0.99 (1% better)
EPSILON = 0.01


@click.command()
@click.option("--network", default="finney")
@click.option("--netuid", type=int, default=NETUID)
@click.option("--wallet-name", default="affine")
@click.option("--hotkey-name", default="validator")
@click.option("--wallet-path", default="~/.bittensor/wallets/")
@click.option("--lium-api-key", required=True, envvar="LIUM_API_KEY")
@click.option("--lium-pod-name", default="distill-persistent")
@click.option("--state-dir", default="state")
@click.option("--max-params-b", type=float, default=5.25)
@click.option("--tempo", type=int, default=360, help="Seconds between epochs")
@click.option("--once", is_flag=True, help="Run one epoch and exit (for testing)")
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         lium_api_key, lium_pod_name, state_dir, max_params_b, tempo, once):
    """Run the distillation validator with king-of-the-hill evaluation."""
    import bittensor as bt
    from lium import Lium, Config
    from eval.scoring import (
        load_scores, save_scores,
        load_failures, save_failures, record_failure, reset_failures, is_stale,
        load_disqualified, save_disqualified, disqualify,
        compute_winner_weights,
        append_score_history,
    )
    from eval.model_checker import (
        check_model_architecture, verify_model_integrity,
        compute_model_hash, check_duplicate_hash, register_model_hash,
    )
    from eval.dataset import load_prompts_from_hf, sample_prompts_seeded, format_prompt

    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    # ── Init chain ──
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    subtensor = bt.Subtensor(network=network)

    # ── Init Lium ──
    cfg = Config(api_key=lium_api_key, ssh_key_path=Path.home() / ".ssh" / "id_ed25519")
    lium = Lium(config=cfg)

    # Find pod
    pods = lium.ps()
    pod = None
    for p in pods:
        if lium_pod_name in p.name:
            pod = p
            break
    if not pod:
        logger.error(f"Lium pod '{lium_pod_name}' not found. Available: {[p.name for p in pods]}")
        sys.exit(1)
    logger.info(f"Using Lium pod: {pod.name} ({pod.id[:12]})")

    # ── Load dataset ──
    all_prompts = load_prompts_from_hf(n=500)
    print(f"[VALIDATOR] Loaded {len(all_prompts)} prompts from HF", flush=True)

    # ── Load state ──
    scores = load_scores(state_path)
    failures = load_failures(state_path)
    dq_reasons = load_disqualified(state_path)
    epoch_count = 0

    # ── Track which UIDs have been evaluated ──
    evaluated_file = state_path / "evaluated_uids.json"
    evaluated_uids = set()
    if evaluated_file.exists():
        try:
            evaluated_uids = set(json.loads(evaluated_file.read_text()))
        except Exception:
            pass

    def save_evaluated():
        evaluated_file.write_text(json.dumps(list(evaluated_uids)))

    # ── Upload eval script ──
    logger.info("Uploading eval script to pod...")
    lium.upload(pod, local="scripts/pod_eval.py", remote="/home/pod_eval.py")

    while True:
        try:
            epoch_start = time.time()
            epoch_count += 1
            print(f"\n[VALIDATOR] === EPOCH {epoch_count} ===", flush=True)
            print(f"[VALIDATOR] Fetching metagraph...", flush=True)
            metagraph = subtensor.metagraph(netuid)
            current_block = subtensor.block
            n_uids = int(metagraph.n)
            print(f"[VALIDATOR] Block {current_block}, n={n_uids}", flush=True)

            # ── Read commitments ──
            print(f"[VALIDATOR] Reading commitments...", flush=True)
            revealed = subtensor.get_all_revealed_commitments(netuid)
            print(f"[VALIDATOR] Got {len(revealed)} revealed entries", flush=True)
            commitments = {}
            for uid in range(n_uids):
                hotkey = str(metagraph.hotkeys[uid])
                if hotkey in revealed and len(revealed[hotkey]) > 0:
                    block, data = revealed[hotkey][0]
                    try:
                        parsed = json.loads(data)
                        if "model" in parsed:
                            commitments[uid] = {"block": block, **parsed}
                    except Exception:
                        continue

            print(f"[VALIDATOR] Found {len(commitments)} miner commitments", flush=True)
            if not commitments:
                logger.info(f"No commitments, sleeping {tempo}s")
                if once:
                    break
                time.sleep(tempo)
                continue

            # ══════════════════════════════════════════════════════════════
            # PHASE 1: Pre-check ALL models (no GPU needed)
            # ══════════════════════════════════════════════════════════════
            valid_models = {}  # uid -> {model, revision, params_b}
            disqualified = set()

            for uid, commit in commitments.items():
                model_repo = commit["model"]
                revision = commit.get("revision", "main")

                # Already permanently disqualified (duplicate hash)
                if scores.get(str(uid), 0) > MAX_KL_THRESHOLD:
                    disqualified.add(uid)
                    continue

                if is_stale(uid, failures):
                    logger.debug(f"UID {uid}: stale (too many failures), skipping")
                    disqualified.add(uid)
                    continue

                print(f"[VALIDATOR] Checking {model_repo}...", flush=True)

                # Architecture check
                check = check_model_architecture(model_repo, revision, max_params_b)
                if not check["pass"]:
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): FAIL — {check['reason']}", flush=True)
                    record_failure(uid, failures)
                    disqualify(uid, f"arch: {check['reason']}", dq_reasons)
                    disqualified.add(uid)
                    continue

                # Duplicate hash check — earlier commitment wins
                model_hash = compute_model_hash(model_repo, revision)
                if model_hash:
                    original_uid = check_duplicate_hash(model_hash, uid, state_path)
                    if original_uid is not None:
                        orig_block = commitments.get(original_uid, {}).get("block", float("inf"))
                        this_block = commit.get("block", float("inf"))
                        if this_block >= orig_block:
                            orig_model = commitments.get(original_uid, {}).get("model", "?")
                            print(f"[VALIDATOR] UID {uid} ({model_repo}): DUPLICATE of UID {original_uid}", flush=True)
                            scores[str(uid)] = MAX_KL_THRESHOLD + 1
                            disqualify(uid, f"copy: identical weights to UID {original_uid} ({orig_model}), committed later at block {this_block} vs {orig_block}", dq_reasons)
                            disqualified.add(uid)
                            continue
                        else:
                            print(f"[VALIDATOR] UID {original_uid} is duplicate of UID {uid} (committed earlier)", flush=True)
                            scores[str(original_uid)] = MAX_KL_THRESHOLD + 1
                            disqualify(original_uid, f"copy: identical weights to UID {uid} ({model_repo}), committed later", dq_reasons)
                            valid_models.pop(original_uid, None)
                            disqualified.add(original_uid)
                            register_model_hash(model_hash, uid, state_path)
                    else:
                        register_model_hash(model_hash, uid, state_path)

                # Integrity check — model still public + unchanged
                hash_file = state_path / "model_hashes.json"
                known_hashes = {}
                if hash_file.exists():
                    try:
                        known_hashes = json.loads(hash_file.read_text())
                    except Exception:
                        pass
                expected_hash = known_hashes.get(str(uid))
                integrity = verify_model_integrity(model_repo, revision, expected_hash)
                if not integrity["pass"]:
                    print(f"[VALIDATOR] UID {uid} DISQUALIFIED: {integrity['reason']}", flush=True)
                    scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(uid, f"integrity: {integrity['reason']}", dq_reasons)
                    disqualified.add(uid)
                    continue
                if integrity["current_hash"]:
                    known_hashes[str(uid)] = integrity["current_hash"]
                    hash_file.write_text(json.dumps(known_hashes, indent=2))

                valid_models[uid] = {
                    "model": model_repo,
                    "revision": revision,
                    "params_b": check.get("params_b", 0),
                }
                print(f"[VALIDATOR] UID {uid}: {model_repo} ({check.get('params_b', 0):.2f}B) ✓", flush=True)

            if not valid_models:
                print("[VALIDATOR] No valid models after pre-checks", flush=True)
                save_scores(scores, state_path)
                save_failures(failures, state_path)
                save_disqualified(dq_reasons, state_path)
                if once:
                    break
                time.sleep(tempo)
                continue

            # ══════════════════════════════════════════════════════════════
            # PHASE 2: Identify king and challengers
            # ══════════════════════════════════════════════════════════════
            king_uid = None
            king_kl = float("inf")
            for uid in valid_models:
                uid_str = str(uid)
                if uid_str in scores and scores[uid_str] <= MAX_KL_THRESHOLD:
                    if scores[uid_str] < king_kl:
                        king_kl = scores[uid_str]
                        king_uid = uid

            # Challengers = valid models that haven't been evaluated yet
            challengers = {
                uid: info for uid, info in valid_models.items()
                if str(uid) not in evaluated_uids
            }

            if not challengers:
                print(f"[VALIDATOR] No new challengers, king UID {king_uid} (KL={king_kl:.6f}) holds", flush=True)
                # Still set weights to keep tempo
                weights, winner_uid, winner_kl = compute_winner_weights(
                    scores, failures, n_uids, max_kl=MAX_KL_THRESHOLD,
                )
                if winner_uid is not None:
                    _set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid)
                save_scores(scores, state_path)
                save_failures(failures, state_path)
                save_disqualified(dq_reasons, state_path)
                elapsed = time.time() - epoch_start
                print(f"[VALIDATOR] Epoch complete in {elapsed:.0f}s (no eval needed)", flush=True)
                if once:
                    break
                print(f"[VALIDATOR] Sleeping {tempo}s...", flush=True)
                time.sleep(tempo)
                continue

            # ══════════════════════════════════════════════════════════════
            # PHASE 3: GPU evaluation — king + challengers, same prompts
            # ══════════════════════════════════════════════════════════════
            # King is always included so both are scored on identical prompts.
            # King's weights are permanent so its score is stable — but we need
            # the head-to-head comparison on the SAME prompt set for a fair test.
            models_to_eval = {}
            if king_uid is not None and king_uid in valid_models:
                models_to_eval[king_uid] = valid_models[king_uid]
            for uid, info in challengers.items():
                models_to_eval[uid] = info

            n_prompts = EVAL_PROMPTS
            chall_str = ", ".join(f"UID {u}" for u in challengers)
            king_str = f"UID {king_uid}" if king_uid else "none"
            print(f"[VALIDATOR] Head-to-head: king={king_str} vs challengers=[{chall_str}] ({n_prompts} prompts)", flush=True)

            # Prepare prompts
            epoch_prompts = sample_prompts_seeded(all_prompts, n_prompts, current_block)
            prompt_texts = [format_prompt(p) for p in epoch_prompts]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(prompt_texts, f)
                prompts_file = f.name
            lium.upload(pod, local=prompts_file, remote="/home/prompts.json")
            os.unlink(prompts_file)

            # Re-upload eval script (in case it changed)
            lium.upload(pod, local="scripts/pod_eval.py", remote="/home/pod_eval.py")

            # Run eval
            student_list = ",".join(m["model"] for m in models_to_eval.values())
            cmd = (
                f"cd /home && python3 pod_eval.py "
                f"--teacher {TEACHER_MODEL} "
                f"--students {student_list} "
                f"--prompts prompts.json "
                f"--output eval_results.json "
                f"--max-prompt-len {MAX_PROMPT_TOKENS} "
                f"--max-new-tokens {MAX_NEW_TOKENS} "
                f"--max-params-b {max_params_b}"
            )
            print(f"[VALIDATOR] Running eval on Lium pod ({len(models_to_eval)} models, {n_prompts} prompts)...", flush=True)

            try:
                result = lium.exec(pod, command=cmd)
                print(f"[VALIDATOR] Pod exit code: {result['exit_code']}", flush=True)
            except Exception as exec_err:
                print(f"[VALIDATOR] lium.exec EXCEPTION: {exec_err}", flush=True)
                import traceback
                traceback.print_exc()
                if once:
                    break
                time.sleep(tempo)
                continue

            if result['stdout'].strip():
                for line in result['stdout'].strip().split('\n')[-30:]:
                    print(f"  GPU: {line[:200]}", flush=True)
            if result['stderr'].strip():
                for line in result['stderr'].strip().split('\n')[-10:]:
                    print(f"  GPU ERR: {line[:200]}", flush=True)
            if not result['success']:
                print(f"[VALIDATOR] Eval failed on pod, skipping", flush=True)
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── Download results ──
            results_local = str(state_path / "last_eval.json")
            try:
                lium.download(pod, remote="/home/eval_results.json", local=results_local)
            except Exception as e:
                logger.error(f"Failed to download results: {e}")
                if once:
                    break
                time.sleep(tempo)
                continue

            with open(results_local) as f:
                results = json.load(f)

            # ══════════════════════════════════════════════════════════════
            # PHASE 4: Process results — update scores, crown new king
            # ══════════════════════════════════════════════════════════════
            uid_to_model = {uid: m["model"] for uid, m in models_to_eval.items()}
            model_to_uid = {m: uid for uid, m in uid_to_model.items()}

            for model_name, student_result in results.get("students", {}).items():
                uid = model_to_uid.get(model_name)
                if uid is None:
                    continue

                if "error" in student_result:
                    logger.warning(f"UID {uid} ({model_name}): eval error — {student_result['error']}")
                    record_failure(uid, failures)
                    continue

                kl = student_result.get("kl_global_avg", float("inf"))
                if kl == float("inf") or kl <= 0:
                    logger.warning(f"UID {uid}: invalid KL={kl}")
                    record_failure(uid, failures)
                    continue

                scores[str(uid)] = kl
                evaluated_uids.add(str(uid))
                reset_failures(uid, failures)
                print(f"[VALIDATOR] UID {uid} ({model_name}): KL={kl:.6f}", flush=True)

            # ── Epsilon check: challenger must beat king by >1% to dethrone ──
            if king_uid is not None and challengers:
                king_new_kl = scores.get(str(king_uid), king_kl)
                threshold = king_new_kl * (1.0 - EPSILON)
                for uid in challengers:
                    uid_str = str(uid)
                    if uid_str in scores and scores[uid_str] <= MAX_KL_THRESHOLD:
                        challenger_kl = scores[uid_str]
                        if challenger_kl < threshold:
                            print(f"[VALIDATOR] UID {uid} DETHRONED king UID {king_uid}! "
                                  f"KL={challenger_kl:.6f} < {threshold:.6f} (king {king_new_kl:.6f} - {EPSILON*100:.0f}%)", flush=True)
                        else:
                            pct = ((king_new_kl - challenger_kl) / king_new_kl * 100) if king_new_kl > 0 else 0
                            print(f"[VALIDATOR] UID {uid} did NOT beat king (KL={challenger_kl:.6f}, "
                                  f"needed <{threshold:.6f}, only {pct:.1f}% better)", flush=True)

            # ── Compute winner & set weights ──
            weights, winner_uid, winner_kl = compute_winner_weights(
                scores, failures, n_uids, max_kl=MAX_KL_THRESHOLD,
                epsilon=EPSILON,
            )

            # Leaderboard
            print(f"\n[VALIDATOR] LEADERBOARD (block {current_block}):", flush=True)
            sorted_scores = sorted(
                [(uid_str, kl) for uid_str, kl in scores.items()],
                key=lambda x: x[1]
            )
            for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
                uid = int(uid_str)
                dq = " ⛔ DQ" if uid in disqualified else ""
                marker = " ← KING" if uid == winner_uid else ""
                new = " (NEW)" if uid_str in [str(u) for u in challengers] else ""
                print(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{new}{dq}", flush=True)

            if winner_uid is not None:
                _set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid)
            else:
                print("[VALIDATOR] No valid miners — skipping weight setting", flush=True)

            # ── Persist state ──
            save_scores(scores, state_path)
            save_failures(failures, state_path)
            save_disqualified(dq_reasons, state_path)
            save_evaluated()

            elapsed = time.time() - epoch_start
            print(f"\n[VALIDATOR] Epoch complete in {elapsed:.0f}s", flush=True)

            if once:
                break
            print(f"[VALIDATOR] Sleeping {tempo}s...", flush=True)
            time.sleep(tempo)

        except KeyboardInterrupt:
            logger.info("Shutting down")
            save_scores(scores, state_path)
            save_failures(failures, state_path)
            save_disqualified(dq_reasons, state_path)
            save_evaluated()
            break
        except Exception as e:
            print(f"[VALIDATOR ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()
            save_scores(scores, state_path)
            save_failures(failures, state_path)
            save_disqualified(dq_reasons, state_path)
            save_evaluated()
            if once:
                break
            time.sleep(60)


def _set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid):
    """Set weights on-chain with retry."""
    print(f"\n[VALIDATOR] Setting weights: UID {winner_uid} = 1.0", flush=True)
    uids = list(range(n_uids))
    for attempt in range(3):
        try:
            success = subtensor.set_weights(
                wallet=wallet, netuid=netuid,
                uids=uids, weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if success:
                print("[VALIDATOR] ✓ Weights set on-chain!", flush=True)
                return
            logger.warning(f"Attempt {attempt + 1}: rejected")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
