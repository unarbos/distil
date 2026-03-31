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
EVAL_PROMPTS = 60
# Epsilon: challenger must beat king by this relative margin to dethrone
# e.g., 0.01 = challenger KL must be < king_kl * 0.99 (1% better)
EPSILON = 0.01


def _announce_new_king(new_uid, new_model, new_kl, old_uid, old_model, old_kl, state_dir):
    """Write a pending announcement to state/announcement.json for async Discord posting."""
    # Note: "old_kl" is the PREVIOUS king's score from the LAST eval round.
    # "new_kl" is the NEW king's score on THIS eval round's prompts.
    # These are on DIFFERENT prompt sets, so direct comparison shows prompt variance, not real improvement.
    # We still show both numbers for transparency but label them correctly.
    kl_diff_pct = ((old_kl - new_kl) / old_kl * 100) if old_kl > 0 else 0

    # Fetch earnings data for the announcement
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

    announcement = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": False,
        "message": (
            f"## 🏆 New King of Distil SN97!\n\n"
            f"**UID {new_uid}** has dethroned **UID {old_uid}**\n\n"
            f"📊 **KL: {new_kl:.6f}** (previous king scored {old_kl:.6f} last eval)\n"
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
            f"👑 Previous king: [{old_model}](<https://huggingface.co/{old_model}>)\n"
            f"{earnings_line}\n"
            f"Think you can beat **{new_kl * (1 - EPSILON):.6f} KL** (1% epsilon)? "
            f"Check the [mining guide](<https://github.com/unarbos/distil#mining-guide>) to get started.\n\n"
            f"📈 [Live Dashboard](<https://distil.arbos.life>)"
        ),
        "data": {
            "new_uid": new_uid, "new_model": new_model, "new_kl": new_kl,
            "old_uid": old_uid, "old_model": old_model, "old_kl": old_kl,
        },
    }
    ann_path = Path(state_dir) / "announcement.json"
    with open(ann_path, "w") as f:
        json.dump(announcement, f, indent=2)
    print(f"[VALIDATOR] Announcement written: UID {new_uid} dethroned UID {old_uid}", flush=True)


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
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         lium_api_key, lium_pod_name, state_dir, max_params_b, tempo, once):
    """Run the distillation validator with king-of-the-hill evaluation."""
    import bittensor as bt
    from lium import Lium, Config
    from eval.scoring import (
        load_scores, save_scores,
        load_failures, save_failures, record_failure, reset_failures, is_stale,
        load_disqualified, save_disqualified, disqualify, is_disqualified, get_dq_reason,
        compute_winner_weights,
        append_score_history,
    )
    from eval.model_checker import (
        check_model_architecture, verify_model_integrity,
        compute_model_hash, check_duplicate_hash, register_model_hash,
    )
    from eval.dataset import sample_prompts_from_dataset, format_prompt

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
    print(f"[VALIDATOR] Prompts sampled fresh from full dataset each epoch", flush=True)

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

    # ── Upload eval script (with retry — SFTP can be flaky on Lium pods) ──
    for _upload_attempt in range(5):
        try:
            logger.info(f"Uploading eval script to pod (attempt {_upload_attempt + 1}/5)...")
            lium.upload(pod, local="scripts/pod_eval.py", remote="/home/pod_eval.py")
            logger.info("Upload successful")
            break
        except Exception as e:
            logger.warning(f"Upload failed: {e}")
            if _upload_attempt < 4:
                time.sleep(10 * (_upload_attempt + 1))
            else:
                raise RuntimeError(f"Failed to upload eval script after 5 attempts: {e}")

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
            uid_to_hotkey = {}
            for uid in range(n_uids):
                hotkey = str(metagraph.hotkeys[uid])
                uid_to_hotkey[uid] = hotkey
                if hotkey in revealed and len(revealed[hotkey]) > 0:
                    block, data = revealed[hotkey][0]
                    try:
                        parsed = json.loads(data)
                        if "model" in parsed:
                            commitments[uid] = {"block": block, "hotkey": hotkey, **parsed}
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
            # STALE SCORE CLEANUP: Clear scores for recycled UIDs
            # When a UID gets a new hotkey, its old score is invalid.
            # ══════════════════════════════════════════════════════════════
            hotkey_map_file = state_path / "uid_hotkey_map.json"
            prev_hotkey_map = {}
            if hotkey_map_file.exists():
                try:
                    prev_hotkey_map = json.loads(hotkey_map_file.read_text())
                except Exception:
                    pass

            stale_cleared = 0
            for uid_str, hotkey in uid_to_hotkey.items():
                uid_s = str(uid_str)
                prev_hotkey = prev_hotkey_map.get(uid_s)
                if prev_hotkey and prev_hotkey != hotkey and uid_s in scores:
                    old_score = scores.pop(uid_s)
                    evaluated_uids.discard(uid_s)
                    stale_cleared += 1
                    print(f"[VALIDATOR] UID {uid_s}: hotkey changed ({prev_hotkey[:8]}→{hotkey[:8]}), "
                          f"cleared stale score {old_score:.6f}", flush=True)

            if stale_cleared:
                save_scores(scores, state_path)
                save_evaluated()
                print(f"[VALIDATOR] Cleared {stale_cleared} stale scores from recycled UIDs", flush=True)

            # Save current hotkey map for next epoch
            hotkey_map_file.write_text(json.dumps({str(k): v for k, v in uid_to_hotkey.items()}))

            # ══════════════════════════════════════════════════════════════
            # PHASE 1: Pre-check ALL models (no GPU needed)
            # ══════════════════════════════════════════════════════════════
            valid_models = {}  # uid -> {model, revision, params_b}
            disqualified = set()

            for uid, commit in commitments.items():
                model_repo = commit["model"]
                revision = commit.get("revision", "main")
                hotkey = commit.get("hotkey", uid_to_hotkey.get(uid, ""))

                # Check DQ by hotkey (stable identity, survives UID recycling)
                if is_disqualified(uid, hotkey, dq_reasons):
                    reason = get_dq_reason(uid, hotkey, dq_reasons)
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): DISQUALIFIED — {reason}", flush=True)
                    disqualified.add(uid)
                    continue

                # Already permanently disqualified (duplicate hash)
                if scores.get(str(uid), 0) > MAX_KL_THRESHOLD:
                    disqualified.add(uid)
                    continue

                if is_stale(uid, failures):
                    logger.debug(f"UID {uid}: stale (too many failures), skipping")
                    disqualified.add(uid)
                    continue

                # Skip expensive HF checks for already-evaluated UIDs with valid scores.
                # They'll be rechecked if their model/revision changes (new commitment).
                uid_str = str(uid)
                if uid_str in evaluated_uids and uid_str in scores and scores[uid_str] <= MAX_KL_THRESHOLD:
                    valid_models[uid] = {"model": model_repo, "revision": revision, "params_b": None, "hotkey": hotkey}
                    continue

                print(f"[VALIDATOR] Checking {model_repo}...", flush=True)

                # Architecture check
                check = check_model_architecture(model_repo, revision, max_params_b)
                if check.get("transient"):
                    # Transient error (rate limit, network) — skip this epoch, retry later
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): TRANSIENT ERROR — {check['reason']}, will retry next epoch", flush=True)
                    continue
                if not check["pass"]:
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): FAIL — {check['reason']}", flush=True)
                    record_failure(uid, failures)
                    disqualify(hotkey, f"arch: {check['reason']}", dq_reasons)
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
                            disqualify(hotkey, f"copy: identical weights to UID {original_uid} ({orig_model}), committed later at block {this_block} vs {orig_block}", dq_reasons)
                            disqualified.add(uid)
                            continue
                        else:
                            print(f"[VALIDATOR] UID {original_uid} is duplicate of UID {uid} (committed earlier)", flush=True)
                            scores[str(original_uid)] = MAX_KL_THRESHOLD + 1
                            orig_hotkey = uid_to_hotkey.get(original_uid, str(original_uid))
                            disqualify(orig_hotkey, f"copy: identical weights to UID {uid} ({model_repo}), committed later", dq_reasons)
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
                if integrity.get("transient"):
                    print(f"[VALIDATOR] UID {uid} integrity check: TRANSIENT ERROR — {integrity['reason']}, will retry next epoch", flush=True)
                    continue
                if not integrity["pass"]:
                    print(f"[VALIDATOR] UID {uid} DISQUALIFIED: {integrity['reason']}", flush=True)
                    scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(hotkey, f"integrity: {integrity['reason']}", dq_reasons)
                    disqualified.add(uid)
                    continue
                if integrity["current_hash"]:
                    known_hashes[str(uid)] = integrity["current_hash"]
                    hash_file.write_text(json.dumps(known_hashes, indent=2))

                valid_models[uid] = {
                    "model": model_repo,
                    "revision": revision,
                    "params_b": check.get("params_b", 0),
                    "commit_block": commit.get("block", float("inf")),
                    "hotkey": hotkey,
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

            # Challengers = valid models that haven't been successfully evaluated yet
            # A UID is only "evaluated" if it has a score — handles cases where
            # eval was tainted (e.g. exploit) and score was never written
            challengers = {
                uid: info for uid, info in valid_models.items()
                if str(uid) not in evaluated_uids or str(uid) not in scores
            }

            if not challengers:
                print(f"[VALIDATOR] No new challengers, king UID {king_uid} (KL={king_kl:.6f}) holds", flush=True)
                # Still set weights periodically to keep tempo — use king directly
                if king_uid is not None:
                    weights = [0.0] * max(n_uids, king_uid + 1)
                    weights[king_uid] = 1.0
                    _set_weights(subtensor, wallet, netuid, n_uids, weights, king_uid)
                save_scores(scores, state_path)
                save_failures(failures, state_path)
                save_disqualified(dq_reasons, state_path)
                elapsed = time.time() - epoch_start
                print(f"[VALIDATOR] Epoch complete in {elapsed:.0f}s (no eval needed)", flush=True)
                if once:
                    break
                # Poll for new challengers every 60s instead of sleeping full tempo
                poll_interval = 60
                print(f"[VALIDATOR] Polling for new challengers every {poll_interval}s...", flush=True)
                time.sleep(poll_interval)
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

            # Skip eval if only the king is in models_to_eval (no challengers survived filtering)
            # This wastes compute and produces useless king-only H2H rounds on the dashboard
            n_challengers_in_eval = sum(1 for uid in models_to_eval if uid != king_uid)
            if n_challengers_in_eval == 0:
                print(f"[VALIDATOR] No challengers in eval batch — skipping (king UID {king_uid} holds)", flush=True)
                save_scores(scores, state_path)
                save_failures(failures, state_path)
                save_disqualified(dq_reasons, state_path)
                if once:
                    break
                time.sleep(60)
                continue

            n_prompts = EVAL_PROMPTS
            chall_str = ", ".join(f"UID {u}" for u in challengers)
            king_str = f"UID {king_uid}" if king_uid else "none"
            print(f"[VALIDATOR] Head-to-head: king={king_str} vs challengers=[{chall_str}] ({n_prompts} prompts)", flush=True)

            # Sort challengers by commit block (earliest first) — used for both
            # progress display and eval ordering
            challenger_uids_sorted = sorted(
                [uid for uid in models_to_eval if uid != king_uid],
                key=lambda uid: models_to_eval[uid].get("commit_block", float("inf")),
            )

            # ── Write eval progress (for dashboard live display) ──
            # Realistic estimates: teacher gen ~90s, each student ~5s/prompt on Blackwell
            est_teacher_s = 90
            est_per_student_s = 5 * n_prompts  # ~5s per prompt per student (not 30s)
            est_total_s = est_teacher_s + est_per_student_s * len(models_to_eval)
            progress_path = state_path / "eval_progress.json"
            now = time.time()
            eval_order = []
            if king_uid is not None and king_uid in models_to_eval:
                eval_order.append({"uid": king_uid, "model": models_to_eval[king_uid]["model"], "role": "king"})
            for uid in challenger_uids_sorted:
                eval_order.append({"uid": uid, "model": models_to_eval[uid]["model"], "role": "challenger"})
            progress = {
                "active": True,
                "phase": "teacher_loading",
                "models": {str(uid): info["model"] for uid, info in models_to_eval.items()},
                "eval_order": eval_order,
                "students_total": len(models_to_eval),
                "students_done": 0,
                "prompts_total": n_prompts,
                "prompts_done": 0,
                "king_uid": king_uid,
                "challenger_uids": list(challengers.keys()),
                "started_at": now,
                "estimated_duration_s": est_total_s,
                "estimated_completion": now + est_total_s,
            }
            with open(progress_path, "w") as f:
                json.dump(progress, f)

            # ── Round resumption: reuse prompts from an incomplete round if available ──
            round_file = state_path / "current_round.json"
            resuming_round = False
            if round_file.exists():
                try:
                    saved_round = json.loads(round_file.read_text())
                    saved_models = set(saved_round.get("model_names", []))
                    current_models = set(info["model"] for info in models_to_eval.values())
                    saved_prompts = saved_round.get("prompts", [])
                    # Resume if we have prompts AND models overlap significantly.
                    # Exact match not required — new models can join an existing round.
                    # pod_eval --resume will score them; already-scored models are skipped.
                    if saved_prompts and (saved_models & current_models):
                        prompt_texts = saved_prompts
                        resuming_round = True
                        new_models = current_models - saved_models
                        dropped_models = saved_models - current_models
                        if new_models:
                            print(f"[VALIDATOR] RESUMING round + {len(new_models)} new models added", flush=True)
                        if dropped_models:
                            print(f"[VALIDATOR] RESUMING round, {len(dropped_models)} models dropped (DQ/stale)", flush=True)
                        print(f"[VALIDATOR] RESUMING incomplete round ({len(prompt_texts)} prompts, {len(current_models)} models)", flush=True)
                except Exception as e:
                    print(f"[VALIDATOR] Could not load saved round: {e}", flush=True)

            if not resuming_round:
                # New round — sample fresh prompts
                epoch_prompts = sample_prompts_from_dataset(n_prompts, current_block)
                prompt_texts = [format_prompt(p) for p in epoch_prompts]

            # Save round state so we can resume after crash
            round_state = {
                "started_at": time.time(),
                "block": current_block,
                "king_uid": king_uid,
                "model_names": [info["model"] for info in models_to_eval.values()],
                "prompts": prompt_texts,
            }
            round_file.write_text(json.dumps(round_state))

            # Upload prompts
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(prompt_texts, f)
                f.flush()
                os.fsync(f.fileno())
                prompts_file = f.name
            fsize = os.path.getsize(prompts_file)
            print(f"[VALIDATOR] Prompts file: {fsize} bytes, {len(prompt_texts)} prompts", flush=True)
            for _up_att in range(3):
                try:
                    lium.upload(pod, local=prompts_file, remote="/home/prompts.json")
                    break
                except Exception as e:
                    print(f"[VALIDATOR ERROR] Prompts upload failed (attempt {_up_att+1}/3): {e}", flush=True)
                    if _up_att < 2:
                        time.sleep(5)
                    else:
                        raise
            os.unlink(prompts_file)

            # Re-upload eval script (in case it changed)
            for _up_att in range(3):
                try:
                    lium.upload(pod, local="scripts/pod_eval.py", remote="/home/pod_eval.py")
                    break
                except Exception as e:
                    print(f"[VALIDATOR ERROR] Eval script upload failed (attempt {_up_att+1}/3): {e}", flush=True)
                    if _up_att < 2:
                        time.sleep(5)
                    else:
                        raise

            # NEVER delete teacher_cache.pt or eval_results.json blindly.
            # pod_eval.py checks the prompts hash inside teacher_cache.pt and
            # --resume skips already-scored students. Let pod_eval handle
            # cache validity — it's the only thing that knows if the hash matches.
            try:
                lium.exec(pod, command="rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_progress.json")
                if resuming_round:
                    print("[VALIDATOR] Resuming round (keeping eval_results.json + teacher_cache.pt on pod)", flush=True)
                else:
                    # New round: remove eval_results.json (scores from old prompts are invalid)
                    # but KEEP teacher_cache.pt — pod_eval.py will check the prompts hash
                    # and reuse it if prompts happen to match, or regenerate if they don't.
                    lium.exec(pod, command="rm -f /home/eval_results.json")
                    print("[VALIDATOR] New round (cleared eval_results.json, keeping teacher_cache.pt for hash check)", flush=True)
            except Exception:
                pass

            # Pre-eval disk check — clean student cache if disk is >80% full
            try:
                disk_check = lium.exec(pod, command="df --output=pcent / | tail -1 | tr -d ' %'")
                disk_pct_str = disk_check.get('stdout', disk_check) if isinstance(disk_check, dict) else disk_check
                disk_pct = int(str(disk_pct_str).strip())
                if disk_pct > 80:
                    print(f"[VALIDATOR] Disk {disk_pct}% full — cleaning student model cache", flush=True)
                    clean_cmd = (
                        "cd /root/.cache/huggingface/hub 2>/dev/null && "
                        "for d in models--*; do "
                        "  case \"$d\" in models--Qwen--Qwen3.5-35B-A3B) continue;; esac; "
                        "  rm -rf \"$d\"; "
                        "done; "
                        "df -h / | tail -1"
                    )
                    clean_result = lium.exec(pod, command=clean_cmd)
                    clean_info = clean_result.get('stdout', clean_result) if isinstance(clean_result, dict) else clean_result
                    print(f"[VALIDATOR] Pre-eval cleanup done: {str(clean_info).strip()}", flush=True)
                else:
                    print(f"[VALIDATOR] Disk {disk_pct}% — OK", flush=True)
            except Exception as e:
                print(f"[VALIDATOR] Disk check failed (non-fatal): {e}", flush=True)

            # Kill any background GPU processes to free VRAM for eval
            try:
                lium.exec(pod, command="for s in distil train vllm-teacher; do tmux kill-session -t $s 2>/dev/null; done; sleep 2; echo 'GPU cleared'")
                print("[VALIDATOR] Cleared GPU for eval", flush=True)
            except Exception:
                pass

            # ── Start vLLM teacher server for fast generation ──
            # vLLM with tensor parallelism is 3-5x faster than HF for 35B teacher.
            # Greedy decoding (temp=0) is deterministic and reproducible by miners.
            VLLM_PORT = 9100
            VLLM_URL = f"http://localhost:{VLLM_PORT}"
            use_vllm = True
            try:
                vllm_start_cmd = (
                    f"tmux new-session -d -s vllm-teacher '"
                    f"python3 -m vllm.entrypoints.openai.api_server "
                    f"--model {TEACHER_MODEL} "
                    f"--served-model-name teacher "
                    f"--dtype bfloat16 "
                    f"--trust-remote-code "
                    f"--port {VLLM_PORT} "
                    f"--disable-log-requests "
                    f"2>&1 | tee /home/vllm_teacher.log'"
                )
                lium.exec(pod, command=vllm_start_cmd)
                print(f"[VALIDATOR] Starting vLLM teacher server on port {VLLM_PORT}...", flush=True)

                # Wait for vLLM to be ready (check /health endpoint)
                for wait_i in range(90):  # up to 4.5 min
                    time.sleep(3)
                    try:
                        health = lium.exec(pod, command=f"curl -s -o /dev/null -w '%{{http_code}}' {VLLM_URL}/health")
                        code = health.get("stdout", "").strip()
                        if code == "200":
                            print(f"[VALIDATOR] vLLM teacher ready after {(wait_i+1)*3}s", flush=True)
                            break
                    except Exception:
                        pass
                else:
                    print("[VALIDATOR] vLLM teacher startup timed out, falling back to HF", flush=True)
                    lium.exec(pod, command="tmux kill-session -t vllm-teacher 2>/dev/null")
                    use_vllm = False
            except Exception as e:
                print(f"[VALIDATOR] vLLM startup failed ({e}), falling back to HF", flush=True)
                use_vllm = False

            # Run eval — king first, then challengers by commit block (earliest first).
            # Earlier commits are more established → likely lower KL → sets best_kl_so_far
            # early for better early-stopping on weaker newcomers.
            ordered_uids = []
            if king_uid is not None and king_uid in models_to_eval:
                ordered_uids.append(king_uid)
            ordered_uids.extend(challenger_uids_sorted)
            student_list = ",".join(models_to_eval[uid]["model"] for uid in ordered_uids)

            # Detect number of GPUs on pod for parallel eval
            n_gpus = 1
            try:
                gpu_check = lium.exec(pod, command="python3 -c 'import torch; print(torch.cuda.device_count())'")
                n_gpus = int(gpu_check.get("stdout", "1").strip())
            except Exception:
                pass

            if n_gpus >= 2 and len(ordered_uids) >= 2:
                # Parallel eval: teacher on GPU 0, then split students across GPUs
                print(f"[VALIDATOR] Parallel eval: {n_gpus} GPUs, {len(models_to_eval)} models, {n_prompts} prompts", flush=True)

                # Step 1: Teacher generates logits on GPU 0 and saves cache
                # Always pass --teacher-logits + --resume so pod_eval reuses
                # cached teacher logits (if prompts hash matches) and skips
                # already-scored students.
                vllm_flag = f" --teacher-vllm-url {VLLM_URL}" if use_vllm else ""
                teacher_cmd = (
                    f"cd /home && python3 pod_eval.py "
                    f"--teacher {TEACHER_MODEL} "
                    f"--students {models_to_eval[ordered_uids[0]]['model']} "
                    f"--prompts prompts.json "
                    f"--output /home/eval_teacher_only.json "
                    f"--max-prompt-len {MAX_PROMPT_TOKENS} "
                    f"--max-new-tokens {MAX_NEW_TOKENS} "
                    f"--max-params-b {max_params_b} "
                    f"--gpu 0 "
                    f"--teacher-logits /home/teacher_cache.pt "
                    f"--save-teacher-logits /home/teacher_cache.pt "
                    f"--resume"
                    f"{vllm_flag}"
                )
                print("[VALIDATOR] Step 1: Teacher inference + first student on GPU 0...", flush=True)
                try:
                    result_teacher = lium.exec(pod, command=teacher_cmd)
                    print(f"[VALIDATOR] Teacher step exit: {result_teacher.get('exit_code')}", flush=True)
                except Exception as e:
                    print(f"[VALIDATOR] Teacher step failed: {e}", flush=True)

                # Step 2: Remaining students split across GPUs using cached teacher logits
                remaining_uids = ordered_uids[1:]  # first student already done in step 1
                if remaining_uids:
                    mid = (len(remaining_uids) + 1) // 2
                    group_0 = remaining_uids[:mid]
                    group_1 = remaining_uids[mid:]

                    def _build_student_cmd(uids, gpu_id, output_file):
                        sl = ",".join(models_to_eval[u]["model"] for u in uids)
                        return (
                            f"cd /home && python3 pod_eval.py "
                            f"--teacher {TEACHER_MODEL} "
                            f"--students {sl} "
                            f"--prompts prompts.json "
                            f"--output {output_file} "
                            f"--max-prompt-len {MAX_PROMPT_TOKENS} "
                            f"--max-new-tokens {MAX_NEW_TOKENS} "
                            f"--max-params-b {max_params_b} "
                            f"--gpu {gpu_id} "
                            f"--teacher-logits /home/teacher_cache.pt"
                        )

                    cmd_gpu0 = _build_student_cmd(group_0, 0, "/home/eval_gpu0.json") if group_0 else None
                    cmd_gpu1 = _build_student_cmd(group_1, 1, "/home/eval_gpu1.json") if group_1 else None

                    # Run both in parallel using background processes
                    bg_cmds = []
                    if cmd_gpu0 and cmd_gpu1:
                        parallel_cmd = f"({cmd_gpu0}) & ({cmd_gpu1}) & wait"
                        print(f"[VALIDATOR] Step 2: {len(group_0)} students GPU0 + {len(group_1)} students GPU1 in parallel", flush=True)
                    elif cmd_gpu0:
                        parallel_cmd = cmd_gpu0
                        print(f"[VALIDATOR] Step 2: {len(group_0)} students on GPU0", flush=True)
                    elif cmd_gpu1:
                        parallel_cmd = cmd_gpu1
                        print(f"[VALIDATOR] Step 2: {len(group_1)} students on GPU1", flush=True)
                    else:
                        parallel_cmd = None

                    if parallel_cmd:
                        try:
                            result_parallel = lium.exec(pod, command=parallel_cmd)
                            print(f"[VALIDATOR] Parallel step exit: {result_parallel.get('exit_code')}", flush=True)
                        except Exception as e:
                            print(f"[VALIDATOR] Parallel step failed: {e}", flush=True)

                # Step 3: Merge all results into eval_results.json
                merge_cmd = """python3 -c "
import json, glob, os
merged = None
for f in ['/home/eval_teacher_only.json', '/home/eval_gpu0.json', '/home/eval_gpu1.json']:
    if not os.path.exists(f): continue
    with open(f) as fh:
        data = json.load(fh)
    if merged is None:
        merged = data
    else:
        merged['students'].update(data.get('students', {}))
if merged:
    with open('/home/eval_results.json', 'w') as fh:
        json.dump(merged, fh)
    print(f'Merged {len(merged[\"students\"])} students')
else:
    print('ERROR: No results to merge')
"
"""
                try:
                    merge_result = lium.exec(pod, command=merge_cmd)
                    print(f"[VALIDATOR] Merge: {merge_result.get('stdout', '').strip()}", flush=True)
                except Exception as e:
                    print(f"[VALIDATOR] Merge failed: {e}", flush=True)

                # Fake result for downstream code
                result = {"exit_code": 0}
            else:
                # Single GPU: original sequential eval
                # --resume + --teacher-logits: if a prior eval crashed mid-round,
                # reuse teacher logits and skip already-scored students.
                vllm_flag = f" --teacher-vllm-url {VLLM_URL}" if use_vllm else ""
                cmd = (
                    f"cd /home && python3 pod_eval.py "
                    f"--teacher {TEACHER_MODEL} "
                    f"--students {student_list} "
                    f"--prompts prompts.json "
                    f"--output eval_results.json "
                    f"--max-prompt-len {MAX_PROMPT_TOKENS} "
                    f"--max-new-tokens {MAX_NEW_TOKENS} "
                    f"--max-params-b {max_params_b} "
                    f"--teacher-logits /home/teacher_cache.pt "
                    f"--save-teacher-logits /home/teacher_cache.pt "
                    f"--resume"
                    f"{vllm_flag}"
                )
                print(f"[VALIDATOR] Running eval on Lium pod ({len(models_to_eval)} models, {n_prompts} prompts)...", flush=True)

                # Update progress: scoring phase (clear stale data from previous eval)
                progress["phase"] = "scoring"
                progress["completed"] = []
                progress.pop("pod", None)
                progress.pop("current_student", None)
                progress.pop("current_prompt", None)
                progress.pop("current_kl", None)
                with open(progress_path, "w") as f:
                    json.dump(progress, f)

                # Background thread: poll live progress from pod every 10s
                import threading
                poll_stop = threading.Event()

                def _poll_pod_progress():
                    while not poll_stop.is_set():
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                                tmp_path = tmp.name
                            lium.download(pod, remote="/home/eval_progress.json", local=tmp_path)
                            with open(tmp_path) as f:
                                pod_progress = json.load(f)
                            os.unlink(tmp_path)
                            progress["pod"] = pod_progress
                            pod_phase = pod_progress.get("phase", "scoring")
                            progress["phase"] = pod_phase
                            if pod_phase == "teacher_generation":
                                progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)
                                progress["prompts_total"] = pod_progress.get("prompts_total", n_prompts)
                            elif pod_progress.get("current"):
                                cur = pod_progress["current"]
                                progress["current_student"] = cur.get("student_name")
                                progress["current_prompt"] = cur.get("prompts_done", 0)
                                progress["current_kl"] = cur.get("kl_running_mean")
                                progress["current_se"] = cur.get("kl_running_se")
                                progress["current_ci"] = cur.get("ci_95")
                                progress["current_best"] = cur.get("best_kl_so_far")
                                progress["students_done"] = cur.get("student_idx", 0)
                            progress["completed"] = pod_progress.get("completed", [])
                            with open(progress_path, "w") as f:
                                json.dump(progress, f)
                        except Exception:
                            pass
                        poll_stop.wait(5)

                poll_thread = threading.Thread(target=_poll_pod_progress, daemon=True)
                poll_thread.start()

                # Dynamic timeout: 10 min per model + 30 min buffer for teacher generation
                # Per-model timeout (10 min) is enforced inside pod_eval.py; this is a safety net
                n_eval_models = len(models_to_eval)
                EVAL_TIMEOUT = (n_eval_models * 10 + 30) * 60
                print(f"[VALIDATOR] Eval timeout: {EVAL_TIMEOUT//60}m ({n_eval_models} models × 10m + 30m buffer)", flush=True)
                eval_env = {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(lium.exec, pod, command=cmd, env=eval_env)
                        try:
                            result = future.result(timeout=EVAL_TIMEOUT)
                        except concurrent.futures.TimeoutError:
                            print(f"[VALIDATOR] Eval timed out after {EVAL_TIMEOUT}s — killing pod process and recovering partial results", flush=True)
                            try:
                                lium.exec(pod, command="pkill -9 -f pod_eval.py; echo killed")
                            except Exception:
                                pass
                            result = {"stdout": "", "stderr": "timeout", "exit_code": -1, "success": False}
                    print(f"[VALIDATOR] Pod exit code: {result['exit_code']}", flush=True)
                except Exception as exec_err:
                    print(f"[VALIDATOR] lium.exec EXCEPTION: {exec_err}", flush=True)
                    import traceback
                    traceback.print_exc()
                    poll_stop.set()
                    poll_thread.join(timeout=5)
                    if once:
                        break
                    time.sleep(tempo)
                    continue
                finally:
                    poll_stop.set()
                    poll_thread.join(timeout=5)

            if result['stdout'].strip():
                for line in result['stdout'].strip().split('\n')[-30:]:
                    print(f"  GPU: {line[:200]}", flush=True)
            if result['stderr'].strip():
                for line in result['stderr'].strip().split('\n')[-10:]:
                    print(f"  GPU ERR: {line[:200]}", flush=True)
            # ── Kill vLLM teacher server to free VRAM ──
            if use_vllm:
                try:
                    lium.exec(pod, command="tmux kill-session -t vllm-teacher 2>/dev/null")
                    print("[VALIDATOR] Killed vLLM teacher server", flush=True)
                except Exception:
                    pass

            # ── Download results (try even on failure — partial results may exist) ──
            results_local = str(state_path / "last_eval.json")
            download_ok = False
            for _dl_attempt in range(3):
                try:
                    lium.download(pod, remote="/home/eval_results.json", local=results_local)
                    download_ok = True
                    break
                except Exception as e:
                    logger.warning(f"Download attempt {_dl_attempt+1}/3 failed: {e}")
                    if _dl_attempt < 2:
                        time.sleep(5)
            if not download_ok:
                logger.error("Failed to download results after 3 attempts")
                if not result['success']:
                    print(f"[VALIDATOR] Eval failed and no results to recover, skipping", flush=True)
                    with open(progress_path, "w") as f:
                        json.dump({"active": False}, f)
                    if once:
                        break
                    time.sleep(tempo)
                    continue

            if not result['success']:
                # Check if partial results are usable
                try:
                    with open(results_local) as f:
                        partial = json.load(f)
                    n_students = len(partial.get("students", {}))
                    if n_students > 0:
                        print(f"[VALIDATOR] Eval failed but recovered {n_students} partial results", flush=True)
                    else:
                        print(f"[VALIDATOR] Eval failed, no usable partial results", flush=True)
                        with open(progress_path, "w") as f:
                            json.dump({"active": False}, f)
                        if once:
                            break
                        time.sleep(tempo)
                        continue
                except Exception:
                    print(f"[VALIDATOR] Eval failed, results file corrupt", flush=True)
                    with open(progress_path, "w") as f:
                        json.dump({"active": False}, f)
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

            king_h2h_kl = None  # King's score on THIS eval's prompts

            for model_name, student_result in results.get("students", {}).items():
                uid = model_to_uid.get(model_name)
                if uid is None:
                    continue

                if "error" in student_result:
                    logger.warning(f"UID {uid} ({model_name}): eval error — {student_result['error']}")
                    record_failure(uid, failures)
                    continue

                # Check for functional copy detected by logit fingerprinting
                if student_result.get("functional_copy"):
                    copy_of_model = student_result.get("copy_of", "unknown")
                    # Find the UID of the model it's a copy of
                    copy_of_uid = None
                    for other_uid, other_info in models_to_eval.items():
                        if other_info["model"] == copy_of_model:
                            copy_of_uid = other_uid
                            break
                    reason = f"copy: functional copy of {copy_of_model}" + (f" (UID {copy_of_uid})" if copy_of_uid else "") + " — identical logit distribution"
                    print(f"[VALIDATOR] UID {uid} ({model_name}): FUNCTIONAL COPY — {reason}", flush=True)
                    scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    _hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    disqualify(_hk, reason, dq_reasons)
                    evaluated_uids.add(str(uid))
                    continue

                # ANTI-CHEAT: Check for fraud signals from pod_eval
                fraud_status = student_result.get("status", "")
                if fraud_status == "fraud_vram":
                    reason = student_result.get("reason", "VRAM fraud detected")
                    print(f"[VALIDATOR] UID {uid} ({model_name}): {reason}", flush=True)
                    _hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    disqualify(_hk, reason, dq_reasons)
                    scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    evaluated_uids.add(str(uid))
                    continue

                speed_flag = student_result.get("speed_flag")
                if speed_flag:
                    print(f"[VALIDATOR] UID {uid} ({model_name}): ⚠️ {speed_flag}", flush=True)

                kl = student_result.get("kl_global_avg", float("inf"))

                # ANTI-CHEAT: KL=0 or near-zero means the model IS the teacher
                if kl <= 1e-6:
                    reason = f"FRAUD: KL={kl:.10f} — model produces identical outputs to teacher (likely teacher weights)"
                    print(f"[VALIDATOR] UID {uid} ({model_name}): {reason}", flush=True)
                    _hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    disqualify(_hk, reason, dq_reasons)
                    scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    evaluated_uids.add(str(uid))
                    continue

                if kl == float("inf") or kl < 0:
                    logger.warning(f"UID {uid}: invalid KL={kl}")
                    record_failure(uid, failures)
                    continue

                # For challengers, update their global score.
                # For the king, only use H2H score for epsilon comparison — don't
                # overwrite their global score. Different prompt sets cause variance,
                # and overwriting would let old challengers (scored on different prompts)
                # appear to beat the king unfairly.
                if uid == king_uid:
                    king_h2h_kl = kl  # Store for epsilon comparison
                    # Update king's global score with H2H score so compute_winner_weights
                    # sees the real performance, not a stale score from an old prompt set
                    scores[str(uid)] = kl
                    evaluated_uids.add(str(uid))
                    print(f"[VALIDATOR] UID {uid} ({model_name}): H2H KL={kl:.6f} (king — global score UPDATED)", flush=True)
                else:
                    scores[str(uid)] = kl
                    evaluated_uids.add(str(uid))
                    reset_failures(uid, failures)
                    print(f"[VALIDATOR] UID {uid} ({model_name}): KL={kl:.6f}", flush=True)

            # ── Epsilon enforcement: challenger must beat king by >1% to dethrone ──
            # Use the king's H2H score (same prompts as challengers) for fair comparison.
            # The king's GLOBAL score is preserved — only challengers who beat the king
            # on the SAME prompt set can dethrone.
            if king_uid is not None and challengers:
                king_new_kl = king_h2h_kl if king_h2h_kl is not None else scores.get(str(king_uid), king_kl)
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
                            # Enforce epsilon: if challenger is close but not epsilon-better,
                            # don't let it dethrone. Set score to king + tiny margin so
                            # compute_winner_weights still picks king.
                            if challenger_kl < king_new_kl:
                                scores[uid_str] = king_new_kl + 1e-8
                                print(f"[VALIDATOR] UID {uid}: epsilon-pinned score to {scores[uid_str]:.8f} "
                                      f"(actual was {challenger_kl:.6f})", flush=True)

            # ── Determine winner from H2H round results ONLY ──
            # DO NOT use compute_winner_weights on global scores — scores from
            # different prompt sets are not comparable. The H2H winner is whoever
            # got the lowest KL in THIS round (same prompts for all models).
            h2h_candidates = []
            all_round_uids = set([king_uid] + challengers) if king_uid is not None else set(challengers)
            for uid in all_round_uids:
                uid_str = str(uid)
                if uid in disqualified:
                    continue
                if uid_str in scores and 0 < scores[uid_str] <= MAX_KL_THRESHOLD:
                    h2h_candidates.append((uid, scores[uid_str]))

            if h2h_candidates:
                h2h_candidates.sort(key=lambda x: x[1])
                winner_uid, winner_kl = h2h_candidates[0]
            else:
                winner_uid, winner_kl = None, float("inf")

            # Build weights array (winner-take-all)
            weights = [0.0] * max(n_uids, (winner_uid or 0) + 1)
            if winner_uid is not None:
                weights[winner_uid] = 1.0

            # Leaderboard (show H2H round results + full global scores)
            print(f"\n[VALIDATOR] H2H ROUND RESULTS (block {current_block}):", flush=True)
            for rank, (uid, kl) in enumerate(h2h_candidates, 1):
                marker = " ← WINNER" if uid == winner_uid else ""
                is_king = " (king)" if uid == king_uid else ""
                print(f"  #{rank}  UID {uid}: KL={kl:.6f}{marker}{is_king}", flush=True)

            print(f"\n[VALIDATOR] GLOBAL LEADERBOARD:", flush=True)
            sorted_scores = sorted(
                [(uid_str, kl) for uid_str, kl in scores.items()],
                key=lambda x: x[1]
            )
            for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
                uid = int(uid_str)
                dq = " ⛔ DQ" if uid in disqualified else ""
                marker = " ← H2H WINNER" if uid == winner_uid else ""
                in_round = " (in round)" if uid in all_round_uids else ""
                print(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{in_round}{dq}", flush=True)

            if winner_uid is not None:
                _set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid)
            else:
                print("[VALIDATOR] No valid miners — skipping weight setting", flush=True)

            # ── Persist state ──
            save_scores(scores, state_path)
            save_failures(failures, state_path)
            save_disqualified(dq_reasons, state_path)
            save_evaluated()

            # ── Append score history (non-DQ scores only) ──
            valid_scores = {
                uid_str: kl for uid_str, kl in scores.items()
                if uid_str not in dq_reasons and 0 < kl <= MAX_KL_THRESHOLD
            }
            if valid_scores:
                append_score_history(
                    block=current_block,
                    timestamp=time.time(),
                    scores=valid_scores,
                    king_uid=winner_uid,
                    state_dir=state_path,
                )

            # ── Save H2H round details for dashboard transparency ──
            h2h_results = []
            for uid, info in models_to_eval.items():
                model_name = info["model"]
                student_data = results.get("students", {}).get(model_name, {})
                kl = student_data.get("kl_global_avg")
                if kl is None or "error" in student_data:
                    continue
                is_king = (uid == king_uid)
                vs_king = ""
                if king_h2h_kl is not None and not is_king and king_h2h_kl > 0:
                    pct = (king_h2h_kl - kl) / king_h2h_kl * 100
                    epsilon_threshold = king_h2h_kl * (1.0 - EPSILON)
                    if kl < epsilon_threshold:
                        vs_king = f"-{pct:.3f}% (DETHRONED)"
                    elif kl < king_h2h_kl:
                        vs_king = f"-{pct:.3f}% (not enough, need >{EPSILON*100:.0f}%)"
                    else:
                        vs_king = "worse"
                h2h_results.append({
                    "uid": uid,
                    "model": model_name,
                    "kl": round(kl, 6),
                    "is_king": is_king,
                    "vs_king": vs_king,
                })
            h2h_results.sort(key=lambda x: x["kl"])

            # Don't save king-only rounds — they clutter the dashboard and mean
            # all challengers failed during eval (errors, timeouts, etc.)
            n_challenger_results = sum(1 for r in h2h_results if not r.get("is_king"))
            if n_challenger_results == 0:
                print(f"[VALIDATOR] All challengers failed — skipping H2H round save (king-only)", flush=True)

            if n_challenger_results > 0:
                h2h_round = {
                    "block": current_block,
                    "timestamp": time.time(),
                    "king_uid": king_uid,
                    "king_h2h_kl": round(king_h2h_kl, 6) if king_h2h_kl else None,
                    "king_global_kl": round(king_kl, 6),
                    "epsilon": EPSILON,
                    "epsilon_threshold": round(king_h2h_kl * (1.0 - EPSILON), 6) if king_h2h_kl else None,
                    "n_prompts": EVAL_PROMPTS,
                    "results": h2h_results,
                    "king_changed": winner_uid != king_uid if king_uid is not None else False,
                    "new_king_uid": winner_uid if winner_uid != king_uid else None,
                }
                # Save latest round + append to history
                h2h_path = state_path / "h2h_latest.json"
                with open(h2h_path, "w") as f:
                    json.dump(h2h_round, f, indent=2)
                h2h_history_path = state_path / "h2h_history.json"
                history = []
                if h2h_history_path.exists():
                    try:
                        with open(h2h_history_path) as f:
                            history = json.load(f)
                    except Exception:
                        history = []
                history.append(h2h_round)
                # Keep last 50 rounds
                history = history[-50:]
                with open(h2h_history_path, "w") as f:
                    json.dump(history, f, indent=2)

            # ── Round complete — clear round state so next epoch starts fresh ──
            round_file = state_path / "current_round.json"
            if round_file.exists():
                round_file.unlink()
                print("[VALIDATOR] Cleared current_round.json (round complete)", flush=True)

            # ── Clear eval progress ──
            progress_path = state_path / "eval_progress.json"
            with open(progress_path, "w") as f:
                json.dump({"active": False}, f)

            # ── Clean HF model cache to prevent disk full ──
            # Keep only the teacher model; students re-download each eval anyway
            try:
                clean_cmd = (
                    "cd /root/.cache/huggingface/hub 2>/dev/null && "
                    "for d in models--*; do "
                    "  case \"$d\" in models--Qwen--Qwen3.5-35B-A3B) continue;; esac; "
                    "  rm -rf \"$d\"; "
                    "done; "
                    "df -h / | tail -1"
                )
                result = lium.exec(pod, command=clean_cmd)
                disk_info = result.get('stdout', result) if isinstance(result, dict) else result
                print(f"[VALIDATOR] Cache cleanup: {str(disk_info).strip()}", flush=True)
            except Exception as e:
                print(f"[VALIDATOR] Cache cleanup failed (non-fatal): {e}", flush=True)

            # ── Restart any background tasks that were cleared for eval ──
            try:
                lium.exec(pod, command="test -f /home/autostart.sh && bash /home/autostart.sh; echo 'Background tasks resumed'")
                print("[VALIDATOR] Resumed background tasks on pod", flush=True)
            except Exception:
                pass

            # ── Discord announcement if king changed ──
            if winner_uid is not None and winner_uid != king_uid and king_uid is not None:
                new_king_model = (uid_to_model.get(winner_uid)
                                  or valid_models.get(winner_uid, {}).get("model", "unknown"))
                old_king_model = (uid_to_model.get(king_uid)
                                  or valid_models.get(king_uid, {}).get("model", "unknown"))
                try:
                    _announce_new_king(
                        new_uid=winner_uid, new_model=new_king_model, new_kl=winner_kl,
                        old_uid=king_uid, old_model=old_king_model, old_kl=king_kl,
                        state_dir=state_path,
                    )
                except Exception as ann_err:
                    print(f"[VALIDATOR] Discord announcement failed: {ann_err}", flush=True)

            elapsed = time.time() - epoch_start
            print(f"\n[VALIDATOR] Epoch complete in {elapsed:.0f}s", flush=True)

            if once:
                break
            # After eval, check immediately for new challengers (may have arrived during eval)
            print(f"[VALIDATOR] Checking for new challengers immediately...", flush=True)

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
