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

import re as _re

# Patterns for sanitizing GPU logs before public exposure
_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*m')
_SECRET_PATTERNS = _re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}')
_SENSITIVE_KEYWORDS = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret")


def _sanitize_gpu_log(raw: str) -> str:
    """Strip ANSI codes, secrets, and SSH noise from GPU pod logs before writing to disk."""
    lines = []
    for line in raw.splitlines():
        cleaned = _ANSI_RE.sub('', line).strip()
        if not cleaned:
            continue
        # Drop lines with sensitive keywords
        if any(kw in cleaned for kw in _SENSITIVE_KEYWORDS):
            continue
        # Drop SSH/SFTP noise
        if any(noise in cleaned for noise in (
            "sftp", "Authentication", "Connected (version", "chan ",
            "Opened sftp", "sftp session closed",
        )):
            continue
        # Redact any token/key patterns
        cleaned = _SECRET_PATTERNS.sub('[REDACTED]', cleaned)
        lines.append(cleaned)
    return '\n'.join(lines)


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
@click.option("--use-vllm", is_flag=True, default=False, envvar="USE_VLLM",
              help="Use vLLM-accelerated pod_eval_vllm.py instead of HF pod_eval")
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         lium_api_key, lium_pod_name, state_dir, max_params_b, tempo, once, use_vllm):
    """Run the distillation validator with king-of-the-hill evaluation."""
    import bittensor as bt
    from lium import Lium, Config
    from eval.scoring import (
        load_scores, save_scores,
        load_failures, save_failures, record_failure, reset_failures, is_stale,
        load_disqualified, save_disqualified, disqualify, is_disqualified, is_flagged, get_dq_reason,
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

    def validate_state_consistency(scores, evaluated_uids, uid_to_hotkey, commitments, dq_reasons):
        """
        Pre-flight state validation. Catches inconsistencies BEFORE they waste GPU time.
        Returns (fixed_scores, fixed_evaluated, issues_found).

        Checks:
        1. Every scored UID must be in evaluated_uids (and vice versa)
        2. Every scored UID must have a valid commitment on-chain
        3. No DQ'd UIDs in scores
        4. No recycled UIDs (hotkey changed since last scoring)
        5. King must exist in scores and not be DQ'd
        """
        issues = []
        fixed_scores = dict(scores)
        fixed_evaluated = set(evaluated_uids)

        # Load hotkey map for recycling detection
        hotkey_map_file = state_path / "uid_hotkey_map.json"
        prev_hotkey_map = {}
        if hotkey_map_file.exists():
            try:
                prev_hotkey_map = json.loads(hotkey_map_file.read_text())
            except Exception:
                pass

        # Check 1: Scored UIDs must be evaluated
        scored_uids = set(fixed_scores.keys())
        for uid_str in scored_uids - fixed_evaluated:
            issues.append(f"UID {uid_str} has score but NOT in evaluated_uids — adding")
            fixed_evaluated.add(uid_str)
        for uid_str in fixed_evaluated - scored_uids:
            # Evaluated but no score is OK (could have failed/DQ'd during eval)
            pass

        # Check 2: No DQ'd UIDs in scores
        for uid_str in list(fixed_scores.keys()):
            uid = int(uid_str)
            hotkey = uid_to_hotkey.get(uid, uid_to_hotkey.get(uid_str, ""))
            _cb = commitments.get(uid, {}).get("block")
            if is_disqualified(uid, hotkey, dq_reasons, commit_block=_cb):
                issues.append(f"UID {uid_str} is DQ'd but has score {fixed_scores[uid_str]:.6f} — removing")
                fixed_scores.pop(uid_str)

        # Check 3: Recycled UIDs (hotkey changed)
        for uid_str in list(fixed_scores.keys()):
            hotkey = str(uid_to_hotkey.get(int(uid_str), uid_to_hotkey.get(uid_str, "")))
            prev = prev_hotkey_map.get(uid_str, "")
            if prev and hotkey and prev != hotkey:
                issues.append(f"UID {uid_str} hotkey changed ({prev[:8]}→{hotkey[:8]}) — clearing stale score")
                fixed_scores.pop(uid_str)
                fixed_evaluated.discard(uid_str)

        # Check 4: Scored UIDs must have a commitment on-chain
        # commitments is a dict keyed by UID (int)
        commitment_uids = set()
        for uid in commitments:
            commitment_uids.add(str(uid))
        for uid_str in list(fixed_scores.keys()):
            if uid_str not in commitment_uids:
                issues.append(f"UID {uid_str} has score but no on-chain commitment — removing")
                fixed_scores.pop(uid_str)
                fixed_evaluated.discard(uid_str)

        # Check 5: Validate h2h_latest king
        h2h_file = state_path / "h2h_latest.json"
        if h2h_file.exists():
            try:
                h2h = json.loads(h2h_file.read_text())
                h2h_king = h2h.get("king_uid")
                new_king = h2h.get("new_king_uid")
                king_changed = h2h.get("king_changed", False)
                if new_king is not None and str(new_king) not in fixed_scores:
                    issues.append(f"h2h_latest.new_king_uid={new_king} has no valid score — stale")
                if h2h_king is not None and str(h2h_king) not in fixed_scores:
                    issues.append(f"h2h_latest.king_uid={h2h_king} has no valid score — stale")
                # If king_changed, king_uid should have been updated to new_king
                if king_changed and new_king is not None and h2h_king != new_king:
                    issues.append(f"h2h_latest: king_changed=true but king_uid={h2h_king} != new_king_uid={new_king} — fixing")
                    h2h["king_uid"] = new_king
                    h2h_file.write_text(json.dumps(h2h, indent=2))
            except Exception:
                pass

        # Check 6: Remove garbage/sentinel scores
        import math
        for uid_str in list(fixed_scores.keys()):
            kl = fixed_scores[uid_str]
            if not isinstance(kl, (int, float)) or math.isnan(kl) or math.isinf(kl) or kl < 0 or kl >= MAX_KL_THRESHOLD:
                issues.append(f"UID {uid_str} has garbage score {kl} — removing")
                fixed_scores.pop(uid_str)
                fixed_evaluated.discard(uid_str)

        if issues:
            print(f"[VALIDATOR] ⚠️ STATE VALIDATION found {len(issues)} issues:", flush=True)
            for issue in issues:
                print(f"  • {issue}", flush=True)
        else:
            print("[VALIDATOR] ✅ State validation passed", flush=True)

        return fixed_scores, fixed_evaluated, issues

    # ── Upload eval script (with retry — SFTP can be flaky on Lium pods) ──
    eval_script = "scripts/pod_eval_vllm.py"
    eval_script_remote = "/home/pod_eval.py"  # same remote name either way
    print(f"[VALIDATOR] Eval script: {eval_script} (vLLM w/ HF fallback)", flush=True)
    for _upload_attempt in range(5):
        try:
            logger.info(f"Uploading eval script to pod (attempt {_upload_attempt + 1}/5)...")
            lium.upload(pod, local=eval_script, remote=eval_script_remote)
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
            # ── Fetch chain state (with retry — Bittensor RPCs can be flaky) ──
            for _chain_attempt in range(3):
                try:
                    print(f"[VALIDATOR] Fetching metagraph...", flush=True)
                    metagraph = subtensor.metagraph(netuid)
                    current_block = subtensor.block
                    # Fetch the REAL on-chain block hash — unpredictable, derived from
                    # actual chain state. Prevents miners from gaming prompt selection.
                    try:
                        current_block_hash = subtensor.substrate.get_block_hash(current_block)
                        if current_block_hash:
                            print(f"[VALIDATOR] Block {current_block}, hash={current_block_hash[:18]}...", flush=True)
                        else:
                            current_block_hash = None
                            print(f"[VALIDATOR] Block {current_block}, hash=UNAVAILABLE (will fallback)", flush=True)
                    except Exception as bh_err:
                        current_block_hash = None
                        print(f"[VALIDATOR] Block {current_block}, hash fetch failed: {bh_err}", flush=True)
                    n_uids = int(metagraph.n)
                    print(f"[VALIDATOR] n={n_uids}", flush=True)

                    print(f"[VALIDATOR] Reading commitments...", flush=True)
                    revealed = subtensor.get_all_revealed_commitments(netuid)
                    print(f"[VALIDATOR] Got {len(revealed)} revealed entries", flush=True)
                    break
                except Exception as chain_err:
                    print(f"[VALIDATOR] Chain RPC error (attempt {_chain_attempt + 1}/3): {chain_err}", flush=True)
                    if _chain_attempt < 2:
                        time.sleep(30)
                    else:
                        print("[VALIDATOR] Chain unreachable after 3 attempts, sleeping 5min", flush=True)
                        time.sleep(300)
                        continue
            commitments = {}
            uid_to_hotkey = {}
            uid_to_coldkey = {}
            for uid in range(n_uids):
                hotkey = str(metagraph.hotkeys[uid])
                uid_to_hotkey[uid] = hotkey
                try:
                    uid_to_coldkey[uid] = str(metagraph.coldkeys[uid])
                except Exception:
                    pass
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

            # ── Migrate bare-hotkey DQ entries to hotkey:block format ──
            # Old DQ entries used bare hotkeys. New format is hotkey:block
            # so miners can re-register with a new commit and not be permanently banned.
            _migrated = 0
            _hotkey_to_block = {com["hotkey"]: com["block"] for com in commitments.values() if "hotkey" in com and "block" in com}
            for key in list(dq_reasons.keys()):
                if key.startswith("flag:") or key.isdigit() or ":" in key:
                    continue  # skip flags, UIDs, already-migrated
                if key in _hotkey_to_block:
                    new_key = f"{key}:{_hotkey_to_block[key]}"
                    dq_reasons[new_key] = dq_reasons.pop(key)
                    _migrated += 1
            if _migrated:
                save_disqualified(dq_reasons, state_path)
                print(f"[VALIDATOR] Migrated {_migrated} DQ entries to per-commit format", flush=True)

            # ══════════════════════════════════════════════════════════════
            # STATE VALIDATION: Catch inconsistencies before they waste GPU
            # ══════════════════════════════════════════════════════════════
            scores, evaluated_uids, state_issues = validate_state_consistency(
                scores, evaluated_uids, uid_to_hotkey, commitments, dq_reasons
            )
            if state_issues:
                save_scores(scores, state_path)
                save_evaluated()
                print(f"[VALIDATOR] State auto-repaired ({len(state_issues)} issues fixed)", flush=True)

            # Save current hotkey map for next epoch (stale cleanup handled by validate_state_consistency above)
            hotkey_map_file = state_path / "uid_hotkey_map.json"
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

                # Check DQ by hotkey:block (per-commitment DQ)
                this_commit_block = commit.get("block")
                if is_disqualified(uid, hotkey, dq_reasons, commit_block=this_commit_block):
                    reason = get_dq_reason(uid, hotkey, dq_reasons, commit_block=this_commit_block)
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

                # Check if this miner's coldkey or HF username is flagged
                hf_user = model_repo.split("/")[0] if "/" in model_repo else None
                coldkey = uid_to_coldkey.get(uid)
                flag_reason = is_flagged(coldkey=coldkey, hf_username=hf_user, dq=dq_reasons)
                if flag_reason:
                    print(f"[VALIDATOR] ⚠️ UID {uid} FLAGGED: {flag_reason}", flush=True)

                # Architecture check
                check = check_model_architecture(model_repo, revision, max_params_b)
                if check.get("transient"):
                    # Transient error (rate limit, network) — skip this epoch, retry later
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): TRANSIENT ERROR — {check['reason']}, will retry next epoch", flush=True)
                    continue
                if not check["pass"]:
                    print(f"[VALIDATOR] UID {uid} ({model_repo}): FAIL — {check['reason']}", flush=True)
                    record_failure(uid, failures)
                    hf_user = model_repo.split("/")[0] if "/" in model_repo else None
                    coldkey = uid_to_coldkey.get(uid)
                    disqualify(hotkey, f"arch: {check['reason']}", dq_reasons,
                               coldkey=coldkey, hf_username=hf_user,
                               commit_block=this_commit_block)
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
                            disqualify(hotkey, f"copy: identical weights to UID {original_uid} ({orig_model}), committed later at block {this_block} vs {orig_block}", dq_reasons,
                                       commit_block=this_commit_block)
                            disqualified.add(uid)
                            continue
                        else:
                            print(f"[VALIDATOR] UID {original_uid} is duplicate of UID {uid} (committed earlier)", flush=True)
                            scores[str(original_uid)] = MAX_KL_THRESHOLD + 1
                            orig_hotkey = uid_to_hotkey.get(original_uid, str(original_uid))
                            orig_commit_block = commitments.get(original_uid, {}).get("block")
                            disqualify(orig_hotkey, f"copy: identical weights to UID {uid} ({model_repo}), committed later", dq_reasons,
                                       commit_block=orig_commit_block)
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
                    disqualify(hotkey, f"integrity: {integrity['reason']}", dq_reasons,
                               commit_block=this_commit_block)
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
            # Determine king from h2h_latest (authoritative) — NOT from global
            # scores, because scores from different prompt sets aren't comparable.
            king_uid = None
            king_kl = float("inf")
            h2h_file = state_path / "h2h_latest.json"
            if h2h_file.exists():
                try:
                    h2h_data = json.loads(h2h_file.read_text())
                    h2h_king = h2h_data.get("king_uid")
                    if h2h_king is not None and h2h_king in valid_models:
                        king_uid = h2h_king
                        king_kl = scores.get(str(h2h_king), float("inf"))
                        print(f"[VALIDATOR] King from h2h_latest: UID {king_uid} (KL={king_kl:.6f})", flush=True)
                except Exception:
                    pass
            # Fallback: if h2h_latest doesn't exist or king isn't valid, use lowest score
            if king_uid is None:
                for uid in valid_models:
                    uid_str = str(uid)
                    if uid_str in scores and scores[uid_str] <= MAX_KL_THRESHOLD:
                        if scores[uid_str] < king_kl:
                            king_kl = scores[uid_str]
                            king_uid = uid
                if king_uid is not None:
                    print(f"[VALIDATOR] King from scores fallback: UID {king_uid} (KL={king_kl:.6f})", flush=True)

            # ── Load persistent model score history ──
            # Tracks best-ever KL by model repo name (not UID — UIDs recycle).
            # Models that scored terribly before don't need re-evaluation even
            # on new prompt sets — they're clearly not competitive.
            model_history_file = state_path / "model_score_history.json"
            model_score_history = {}
            if model_history_file.exists():
                try:
                    model_score_history = json.loads(model_history_file.read_text())
                except Exception:
                    pass

            # Challengers = valid models that haven't been successfully evaluated yet
            challengers = {}
            skipped_known_bad = 0
            for uid, info in valid_models.items():
                uid_str = str(uid)
                # Already scored in THIS round's scoring context — skip
                if uid_str in evaluated_uids and uid_str in scores:
                    continue
                # Check persistent model history — if this model scored > 2x king's KL
                # on ANY previous evaluation, don't waste GPU time re-evaluating it.
                # It's clearly not competitive regardless of prompt set variance.
                model_name = info["model"]
                best_ever = model_score_history.get(model_name, {}).get("best_kl")
                if best_ever is not None and king_kl < float("inf"):
                    skip_threshold = max(king_kl * 2.0, king_kl + 0.05)  # 2x king or king+0.05, whichever is larger
                    if best_ever > skip_threshold:
                        skipped_known_bad += 1
                        if not evaluated_uids.__contains__(uid_str):
                            evaluated_uids.add(uid_str)
                        continue
                challengers[uid] = info

            if skipped_known_bad:
                print(f"[VALIDATOR] Skipped {skipped_known_bad} models with historically bad scores (>2x king KL)", flush=True)

            # Sanity check: if too many challengers, something may be wrong with state
            MAX_REASONABLE_CHALLENGERS = 20
            if len(challengers) > MAX_REASONABLE_CHALLENGERS:
                print(f"[VALIDATOR] ⚠️ {len(challengers)} challengers detected — this seems high.", flush=True)
                print(f"  evaluated_uids: {len(evaluated_uids)}, scores: {len(scores)}, valid_models: {len(valid_models)}", flush=True)
                # Don't block — just log the warning. The eval will handle it.

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
                epoch_prompts = sample_prompts_from_dataset(
                    n_prompts, current_block, block_hash=current_block_hash
                )
                prompt_texts = [format_prompt(p) for p in epoch_prompts]

            # Save round state so we can resume after crash
            round_state = {
                "started_at": time.time(),
                "block": current_block,
                "block_hash": current_block_hash,
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
            for _up_att in range(5):
                try:
                    lium.upload(pod, local=eval_script, remote=eval_script_remote)
                    break
                except Exception as e:
                    print(f"[VALIDATOR ERROR] Eval script upload failed (attempt {_up_att+1}/5): {e}", flush=True)
                    if _up_att < 4:
                        time.sleep(5)
                    else:
                        raise RuntimeError(f"Failed to upload eval script after 5 attempts: {e}")

            # NEVER delete teacher_cache.pt or eval_results.json blindly.
            # pod_eval checks the prompts hash inside teacher_cache.pt and
            # --resume skips already-scored students. Let pod_eval handle
            # cache validity — it's the only thing that knows if the hash matches.
            try:
                lium.exec(pod, command="rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_progress.json")
                if resuming_round:
                    print("[VALIDATOR] Resuming round (keeping eval_results.json + teacher_cache.pt on pod)", flush=True)
                else:
                    # New round: remove eval_results.json (scores from old prompts are invalid)
                    # but KEEP teacher_cache.pt — pod_eval checks the prompts hash
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

                # Fake result for downstream code — must include all keys accessed later
                result = {"exit_code": 0, "stdout": "", "stderr": "", "success": True}
            else:
                # Single GPU: original sequential eval
                # --resume + --teacher-logits: if a prior eval crashed mid-round,
                # reuse teacher logits and skip already-scored students.
                # Build eval command — use vLLM script if enabled
                king_flag = ""
                if use_vllm and king_uid is not None and king_uid in models_to_eval:
                    king_model_name = models_to_eval[king_uid]["model"]
                    king_flag = f" --king {king_model_name}"
                eval_cmd_core = (
                    f"cd /home && python3 -u pod_eval.py "
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
                    f"{king_flag}"
                )
                # Tee output to log file for live streaming to dashboard
                cmd = f"{eval_cmd_core} 2>&1 | tee /home/eval_output.log"
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
                progress_lock = threading.Lock()

                # Log file path for pod output streaming
                gpu_log_path = state_path / "gpu_eval.log"

                def _poll_pod_progress():
                    while not poll_stop.is_set():
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                                tmp_path = tmp.name
                            lium.download(pod, remote="/home/eval_progress.json", local=tmp_path)
                            with open(tmp_path) as f:
                                pod_progress = json.load(f)
                            os.unlink(tmp_path)
                            with progress_lock:
                                progress["pod"] = pod_progress
                                pod_phase = pod_progress.get("phase", "scoring")
                                progress["phase"] = pod_phase

                                _student_keys = ("current_student", "current_prompt", "current_kl")

                                # Teacher phases (generation, logit extraction, cache load)
                                if pod_phase in ("teacher_generation", "teacher_logits",
                                                 "teacher_loading", "vllm_starting",
                                                 "vllm_generating"):
                                    progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)
                                    progress["prompts_total"] = pod_progress.get("prompts_total", n_prompts)
                                    for k in _student_keys:
                                        progress.pop(k, None)

                                # Student scoring phase
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
                                elif pod_phase == "scoring":
                                    for k in _student_keys:
                                        progress.pop(k, None)

                                # Always update completed count
                                pod_completed = pod_progress.get("completed", [])
                                progress["completed"] = pod_completed
                                progress["students_done"] = len(pod_completed)

                                with open(progress_path, "w") as f:
                                    json.dump(progress, f)
                        except Exception:
                            pass

                        # Fetch pod stdout log (last 100 lines) and sanitize before writing
                        try:
                            log_result = lium.exec(pod, command="tail -100 /home/eval_output.log 2>/dev/null || echo ''")
                            log_text = log_result.get("stdout", "")
                            if log_text.strip():
                                gpu_log_path.write_text(_sanitize_gpu_log(log_text))
                        except Exception:
                            pass

                        poll_stop.wait(5)

                poll_thread = threading.Thread(target=_poll_pod_progress, daemon=True)
                poll_thread.start()

                # Dynamic timeout: 10 min per model + 30 min buffer for teacher generation
                # Per-model timeout (10 min) is enforced inside pod_eval; this is a safety net
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

            stdout = result.get('stdout', '') or ''
            stderr = result.get('stderr', '') or ''
            if stdout.strip():
                for line in stdout.strip().split('\n')[-30:]:
                    print(f"  GPU: {line[:200]}", flush=True)
            if stderr.strip():
                for line in stderr.strip().split('\n')[-10:]:
                    print(f"  GPU ERR: {line[:200]}", flush=True)
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
                if not result.get('success', False):
                    print(f"[VALIDATOR] Eval failed and no results to recover, skipping", flush=True)
                    with open(progress_path, "w") as f:
                        json.dump({"active": False}, f)
                    if once:
                        break
                    time.sleep(tempo)
                    continue

            if not result.get('success', False):
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
                    _cb = models_to_eval.get(uid, {}).get("commit_block")
                    disqualify(_hk, reason, dq_reasons, commit_block=_cb)
                    evaluated_uids.add(str(uid))
                    continue

                # ANTI-CHEAT: Check for fraud signals from pod_eval
                fraud_status = student_result.get("status", "")
                if fraud_status == "fraud_vram":
                    reason = student_result.get("reason", "VRAM fraud detected")
                    print(f"[VALIDATOR] UID {uid} ({model_name}): {reason}", flush=True)
                    _hk = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    _cb = models_to_eval.get(uid, {}).get("commit_block")
                    disqualify(_hk, reason, dq_reasons, commit_block=_cb)
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
                    _cb = models_to_eval.get(uid, {}).get("commit_block")
                    disqualify(_hk, reason, dq_reasons, commit_block=_cb)
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

            # ── Epsilon enforcement + winner determination ──
            # Challenger must beat king by >EPSILON to dethrone.
            # Scores are NEVER mutated — epsilon is enforced in winner selection only.
            # This preserves real scores for transparency on the dashboard.
            #
            # If king failed to eval this round, we MUST use the fresh scores
            # from challengers only — the king retains crown by default.
            if king_uid is not None and king_h2h_kl is None:
                print(f"[VALIDATOR] ⚠️ King UID {king_uid} did not produce a score this round — retaining crown by default", flush=True)
            king_new_kl = king_h2h_kl if king_h2h_kl is not None else scores.get(str(king_uid), king_kl) if king_uid else float("inf")
            epsilon_threshold = king_new_kl * (1.0 - EPSILON) if king_uid else float("inf")
            epsilon_dethroned_by = None  # Track which challenger dethroned king

            if king_uid is not None and challengers:
                for uid in challengers:
                    uid_str = str(uid)
                    if uid_str in scores and 0 < scores[uid_str] <= MAX_KL_THRESHOLD:
                        challenger_kl = scores[uid_str]
                        if challenger_kl < epsilon_threshold:
                            print(f"[VALIDATOR] UID {uid} DETHRONED king UID {king_uid}! "
                                  f"KL={challenger_kl:.6f} < {epsilon_threshold:.6f} (king {king_new_kl:.6f} - {EPSILON*100:.0f}%)", flush=True)
                            if epsilon_dethroned_by is None or challenger_kl < scores.get(str(epsilon_dethroned_by), float("inf")):
                                epsilon_dethroned_by = uid
                        else:
                            pct = ((king_new_kl - challenger_kl) / king_new_kl * 100) if king_new_kl > 0 else 0
                            if challenger_kl < king_new_kl:
                                print(f"[VALIDATOR] UID {uid}: better than king but within epsilon "
                                      f"(KL={challenger_kl:.6f}, needed <{epsilon_threshold:.6f}, only {pct:.1f}% better)", flush=True)

            # ── Determine winner from H2H round results ONLY ──
            # DO NOT use compute_winner_weights on global scores — scores from
            # different prompt sets are not comparable. The H2H winner is whoever
            # got the lowest KL in THIS round (same prompts for all models).
            h2h_candidates = []
            all_round_uids = set([king_uid] + list(challengers.keys())) if king_uid is not None else set(challengers.keys())
            for uid in all_round_uids:
                uid_str = str(uid)
                hotkey = uid_to_hotkey.get(uid, "")
                _cb = commitments.get(uid, {}).get("block")
                if is_disqualified(uid, hotkey, dq_reasons, commit_block=_cb):
                    continue
                if uid_str in scores and 0 < scores[uid_str] <= MAX_KL_THRESHOLD:
                    h2h_candidates.append((uid, scores[uid_str]))

            if h2h_candidates:
                h2h_candidates.sort(key=lambda x: x[1])
                best_uid, best_kl = h2h_candidates[0]
                # Respect epsilon: if best is a challenger but didn't beat king by epsilon,
                # king retains the crown even if their KL is slightly worse
                if king_uid is not None and best_uid != king_uid and epsilon_dethroned_by is None:
                    # No challenger passed epsilon — king wins
                    winner_uid = king_uid
                    winner_kl = scores.get(str(king_uid), king_kl)
                    print(f"[VALIDATOR] King UID {king_uid} retains crown (no challenger passed epsilon)", flush=True)
                elif epsilon_dethroned_by is not None:
                    # A challenger passed epsilon — they're the winner
                    winner_uid = epsilon_dethroned_by
                    winner_kl = scores.get(str(epsilon_dethroned_by), best_kl)
                    print(f"[VALIDATOR] UID {winner_uid} is new king (passed epsilon)", flush=True)
                else:
                    winner_uid, winner_kl = best_uid, best_kl
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
                hotkey = uid_to_hotkey.get(uid, "")
                _cb = commitments.get(uid, {}).get("block")
                dq = " ⛔ DQ" if (uid in disqualified or is_disqualified(uid, hotkey, dq_reasons, commit_block=_cb)) else ""
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

            # ── Update persistent model score history ──
            # Track best-ever KL by model name so we never re-eval known-bad models.
            model_history_file = state_path / "model_score_history.json"
            model_score_history = {}
            if model_history_file.exists():
                try:
                    model_score_history = json.loads(model_history_file.read_text())
                except Exception:
                    pass
            for uid, info in models_to_eval.items():
                uid_str = str(uid)
                model_name = info["model"]
                if uid_str in scores and 0 < scores[uid_str] <= MAX_KL_THRESHOLD:
                    kl = scores[uid_str]
                    prev = model_score_history.get(model_name, {})
                    prev_best = prev.get("best_kl", float("inf"))
                    if kl < prev_best:
                        model_score_history[model_name] = {
                            "best_kl": round(kl, 6),
                            "uid": uid,
                            "block": current_block,
                            "timestamp": time.time(),
                        }
            model_history_file.write_text(json.dumps(model_score_history, indent=2))

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
                king_changed = winner_uid != king_uid if king_uid is not None else False
                h2h_round = {
                    "block": current_block,
                    "timestamp": time.time(),
                    # king_uid = the winner (for next round to pick up correctly)
                    "king_uid": winner_uid if winner_uid is not None else king_uid,
                    "prev_king_uid": king_uid,
                    "king_h2h_kl": round(king_h2h_kl, 6) if king_h2h_kl else None,
                    "king_global_kl": round(king_kl, 6),
                    "epsilon": EPSILON,
                    "epsilon_threshold": round(king_h2h_kl * (1.0 - EPSILON), 6) if king_h2h_kl else None,
                    "n_prompts": EVAL_PROMPTS,
                    "results": h2h_results,
                    "king_changed": king_changed,
                    "new_king_uid": winner_uid if king_changed else None,
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
                # Use H2H KL (same prompt set) for accurate comparison in announcement
                old_kl_for_announcement = king_h2h_kl if king_h2h_kl is not None else king_kl
                try:
                    _announce_new_king(
                        new_uid=winner_uid, new_model=new_king_model, new_kl=winner_kl,
                        old_uid=king_uid, old_model=old_king_model, old_kl=old_kl_for_announcement,
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
            result = subtensor.set_weights(
                wallet=wallet, netuid=netuid,
                uids=uids, weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            # set_weights returns (bool, str) tuple
            ok = result[0] if isinstance(result, (tuple, list)) else bool(result)
            if ok:
                print("[VALIDATOR] ✓ Weights set on-chain!", flush=True)
                return
            err_msg = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else str(result)
            logger.warning(f"Attempt {attempt + 1}: rejected — {err_msg}")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
