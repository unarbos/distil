#!/usr/bin/env python3
"""
Remote Validator — runs eval on Lium GPU, sets weights locally.

Architecture:
  1. This script runs on the secure server (has wallet keys)
  2. Uploads eval script to Lium GPU pod
  3. Pod runs teacher + student forward passes, returns KL scores
  4. This script reads KL scores and sets weights on-chain

No wallet keys leave this machine. No chain access needed on the GPU pod.
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
SAMPLES_PER_EPOCH = 20


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
    """Run the distillation validator with remote GPU eval."""
    import bittensor as bt
    from lium import Lium, Config
    from eval.scoring import (
        load_scores, save_scores,
        load_failures, save_failures, record_failure, reset_failures, is_stale,
        compute_winner_weights,
    )
    from eval.model_checker import (
        check_model_architecture, verify_tokenizer, verify_model_integrity,
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

    # ── Upload eval script ──
    logger.info("Uploading eval script to pod...")
    lium.upload(pod, local="scripts/pod_eval.py", remote="/home/pod_eval.py")

    while True:
        try:
            epoch_start = time.time()
            print(f"[VALIDATOR] Fetching metagraph...", flush=True)
            metagraph = subtensor.metagraph(netuid)
            current_block = subtensor.block
            print(f"[VALIDATOR] Block {current_block}, n={metagraph.n}", flush=True)
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH — Block {current_block}")
            logger.info(f"{'='*60}")

            # ── Read commitments ──
            print(f"[VALIDATOR] Reading commitments...", flush=True)
            revealed = subtensor.get_all_revealed_commitments(netuid)
            print(f"[VALIDATOR] Got {len(revealed)} revealed entries", flush=True)
            commitments = {}
            for uid in range(metagraph.n):
                hotkey = str(metagraph.hotkeys[uid])
                if hotkey in revealed and len(revealed[hotkey]) > 0:
                    block, data = revealed[hotkey][0]  # First commit only
                    try:
                        parsed = json.loads(data)
                        if "model" in parsed:
                            commitments[uid] = {"block": block, **parsed}
                    except:
                        continue

            print(f"[VALIDATOR] Found {len(commitments)} miner commitments", flush=True)
            if not commitments:
                logger.info(f"No commitments, sleeping {tempo}s")
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── Pre-check models locally (no GPU needed) ──
            valid_models = {}
            for uid, commit in commitments.items():
                model_repo = commit["model"]
                revision = commit.get("revision", "main")

                if is_stale(uid, failures):
                    logger.debug(f"UID {uid}: stale, skipping")
                    continue

                print(f"[VALIDATOR] Checking {model_repo}...", flush=True)
                check = check_model_architecture(model_repo, revision, max_params_b)
                if not check["pass"]:
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): FAIL — {check['reason']}", flush=True)
                    record_failure(uid, failures)
                    continue

                # Duplicate hash check — reject if same weights as another miner
                model_hash = compute_model_hash(model_repo, revision)
                if model_hash:
                    original_uid = check_duplicate_hash(model_hash, uid, state_path)
                    if original_uid is not None:
                        print(f"[VALIDATOR] UID {uid} ({model_repo}): DUPLICATE of UID {original_uid} — same model weights", flush=True)
                        scores[str(uid)] = MAX_KL_THRESHOLD + 1  # Permanent zero weight
                        continue
                    register_model_hash(model_hash, uid, state_path)

                valid_models[uid] = {"model": model_repo, "revision": revision, "params_b": check.get("params_b", 0)}
                print(f"[VALIDATOR] UID {uid}: {model_repo} ({check.get('params_b', 0):.2f}B) ✓", flush=True)

            if not valid_models:
                print("[VALIDATOR] No valid models to evaluate", flush=True)
                if once:
                    break
                time.sleep(tempo)
                continue

            # ── Prepare prompts ──
            epoch_prompts = sample_prompts_seeded(all_prompts, SAMPLES_PER_EPOCH, current_block)
            prompt_texts = [format_prompt(p) for p in epoch_prompts]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(prompt_texts, f)
                prompts_file = f.name
            lium.upload(pod, local=prompts_file, remote="/home/prompts.json")
            os.unlink(prompts_file)

            # ── Run eval on Lium GPU ──
            student_list = ",".join(m["model"] for m in valid_models.values())
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
            print(f"[VALIDATOR] Running eval on Lium pod ({len(valid_models)} models, {SAMPLES_PER_EPOCH} prompts)...", flush=True)

            print(f"[VALIDATOR] >>> Calling lium.exec now...", flush=True)
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

            # ── Process KL scores (raw, no EMA) ──
            uid_to_model = {uid: m["model"] for uid, m in valid_models.items()}
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
                reset_failures(uid, failures)
                print(f"[VALIDATOR] UID {uid} ({model_name}): KL={kl:.6f}", flush=True)

            # ── Integrity check: verify models are public + unchanged ──
            hash_file = state_path / "model_hashes.json"
            known_hashes = {}
            if hash_file.exists():
                with open(hash_file) as f:
                    known_hashes = json.load(f)

            disqualified = set()
            for uid_str in list(scores.keys()):
                uid = int(uid_str)
                model_info_entry = valid_models.get(uid) or commitments.get(uid)
                if not model_info_entry:
                    continue
                model_repo = model_info_entry["model"]
                revision = model_info_entry.get("revision", "main")
                expected_hash = known_hashes.get(uid_str)

                integrity = verify_model_integrity(model_repo, revision, expected_hash)
                if not integrity["pass"]:
                    print(f"[VALIDATOR] UID {uid} DISQUALIFIED: {integrity['reason']}", flush=True)
                    disqualified.add(uid)
                    scores[uid_str] = MAX_KL_THRESHOLD + 1  # Zero weight
                else:
                    if integrity["current_hash"]:
                        known_hashes[uid_str] = integrity["current_hash"]

            with open(hash_file, "w") as f:
                json.dump(known_hashes, f, indent=2)

            if disqualified:
                print(f"[VALIDATOR] {len(disqualified)} miners disqualified", flush=True)

            # ── Compute winner & set weights ──
            weights, winner_uid, winner_kl = compute_winner_weights(
                scores, failures, int(metagraph.n), max_kl=MAX_KL_THRESHOLD,
            )

            # Leaderboard
            print(f"\n[VALIDATOR] LEADERBOARD (block {current_block}):", flush=True)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
                uid = int(uid_str)
                dq = " ⛔ DISQUALIFIED" if uid in disqualified else ""
                marker = " ← WINNER" if uid == winner_uid else ""
                print(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{dq}", flush=True)

            if winner_uid is not None:
                print(f"\n[VALIDATOR] Setting weights: UID {winner_uid} = 1.0", flush=True)
                uids = list(range(int(metagraph.n)))
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
                            break
                        logger.warning(f"Attempt {attempt + 1}: rejected")
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1}: {e}")
                    time.sleep(30)
            else:
                print("[VALIDATOR] No valid miners — skipping weight setting", flush=True)

            # ── Persist state ──
            save_scores(scores, state_path)
            save_failures(failures, state_path)

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
            break
        except Exception as e:
            print(f"[VALIDATOR ERROR] Epoch error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            save_scores(scores, state_path)
            save_failures(failures, state_path)
            if once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
