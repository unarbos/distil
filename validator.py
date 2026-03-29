#!/usr/bin/env python3
"""
Distillation Subnet Validator (v0.6.0) — Production-ready for mainnet.

Chi pattern: single long-running process, no synapse/axon/dendrite.
Miners commit HuggingFace model links on-chain via Commitments pallet.
Validator evaluates KL(teacher || student) using full-distribution GPU forward passes.

Key features:
  - ONE commitment per hotkey, PERMANENTLY (first commit wins, later ones ignored)
  - Full-distribution KL on 248K vocab (not top-k approximation)
  - Teacher continuation: generates 512 tokens, scores on continuation positions
  - Proportional inverse-KL weights (not winner-take-all)
  - EMA smoothing across epochs (alpha=0.3)
  - Block-seeded prompt selection (unpredictable, reproducible)
  - Model caching: only re-eval changed commitments
  - Copy detection via SHA256 of first safetensors shard
  - Same-tokenizer enforcement (exact encoding match)
  - Staleness: 3 failures → weight 0 until new commitment
  - MoE-aware param counting
"""
import os
import sys
import time
import json
import gc
import logging
import traceback
from pathlib import Path

import click
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.validator")

# ── Constants ──────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
DEFAULT_MAX_PARAM_RATIO = 0.1  # Students ≤ 10% of teacher = 3.5B total
MAX_KL_THRESHOLD = 2.0  # Quality floor — reject if KL above this (good distill ~0.1-0.5)
EMA_ALPHA = 0.3
MAX_EVAL_PER_EPOCH = 5  # Max new models to evaluate per epoch
MAX_NEW_TOKENS = 512  # Teacher continuation length
MAX_PROMPT_TOKENS = 1024
STATE_DIR = Path("state")


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--wallet-name", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey-name", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--teacher-model", default=TEACHER_MODEL)
@click.option("--max-param-ratio", type=float, default=DEFAULT_MAX_PARAM_RATIO)
@click.option("--dataset-path", default="./dataset")
@click.option("--samples-per-epoch", type=int, default=12)
@click.option("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
@click.option("--max-eval-per-epoch", type=int, default=MAX_EVAL_PER_EPOCH)
@click.option("--tempo", type=int, default=360, help="Seconds between evaluation epochs")
@click.option("--state-dir", type=click.Path(), default=str(STATE_DIR))
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
def main(
    network, netuid, wallet_name, hotkey_name, teacher_model, max_param_ratio,
    dataset_path, samples_per_epoch, max_new_tokens, max_eval_per_epoch,
    tempo, state_dir, log_level,
):
    """Run the distillation subnet validator."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    import bittensor as bt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.kl_divergence import evaluate_kl_with_continuation
    from eval.dataset import load_swe_infinite_prompts, sample_prompts_seeded, format_coding_prompt
    from eval.model_checker import (
        check_model_architecture, compute_model_hash,
        check_duplicate_hash, register_model_hash,
        verify_tokenizer,
    )
    from eval.scoring import (
        load_ema_scores, save_ema_scores, update_ema,
        load_failures, save_failures, record_failure, reset_failures, is_stale,
        load_commitment_cache, save_commitment_cache, commitment_changed,
        compute_proportional_weights,
    )

    max_student_params_b = TEACHER_TOTAL_PARAMS_B * max_param_ratio
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Init chain ─────────────────────────────────────────────────────
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)

    # ── Load dataset ───────────────────────────────────────────────────
    all_prompts = load_swe_infinite_prompts(dataset_path)
    logger.info(f"Loaded {len(all_prompts)} prompts from {dataset_path}")

    # ── Load teacher model (kept resident) ─────────────────────────────
    logger.info(f"Loading teacher model: {teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    teacher.eval()
    logger.info("Teacher model loaded and resident in GPU memory")

    # ── Load persistent state ──────────────────────────────────────────
    ema_scores = load_ema_scores(state_path)
    failures = load_failures(state_path)
    commit_cache = load_commitment_cache(state_path)

    # ── Main loop ──────────────────────────────────────────────────────
    while True:
        try:
            metagraph.sync(subtensor=subtensor)
            current_block = subtensor.block

            # ── Read commitments (FIRST per hotkey — permanent, no updates) ─
            commitments = {}
            revealed = subtensor.get_all_revealed_commitments(netuid)
            for uid in range(metagraph.n):
                hotkey = metagraph.hotkeys[uid]
                try:
                    if hotkey in revealed and len(revealed[hotkey]) > 0:
                        # Use FIRST commitment only — miners cannot update
                        block, commit_data = revealed[hotkey][0]
                        data = json.loads(commit_data)
                        if "model" in data:
                            commitments[uid] = {"block": block, **data}
                            # Log if miner tried to update (extra commits ignored)
                            if len(revealed[hotkey]) > 1:
                                logger.debug(
                                    f"UID {uid} has {len(revealed[hotkey])} commits — "
                                    f"only first (block {block}) is honored"
                                )
                except Exception:
                    continue

            logger.info(f"Found {len(commitments)} miner commitments (first-commit-only)")

            if not commitments:
                logger.info(f"No commitments, sleeping {tempo}s")
                time.sleep(tempo)
                continue

            # ── Determine which models need (re-)evaluation ────────────
            # Commitments are permanent — no "changed" check needed.
            # Priority: unevaluated miners first, then re-eval oldest for EMA freshness.
            needs_eval = []
            already_scored = []
            for uid, commit in commitments.items():
                # Skip stale miners entirely
                if is_stale(uid, failures):
                    continue

                if str(uid) not in ema_scores:
                    # Never evaluated — highest priority
                    needs_eval.append((uid, commit))
                else:
                    already_scored.append((uid, commit))

            # Re-evaluate oldest cached scores (freshness rotation for EMA)
            already_scored.sort(key=lambda x: commit_cache.get(str(x[0]), {}).get("block", 0))
            needs_eval.extend(already_scored[:2])  # Re-eval up to 2 oldest

            # Cap evaluations per epoch
            to_eval = needs_eval[:max_eval_per_epoch]
            logger.info(f"Evaluating {len(to_eval)} models this epoch "
                        f"(of {len(needs_eval)} pending)")

            # ── Block-seeded prompt selection ──────────────────────────
            epoch_prompts = sample_prompts_seeded(all_prompts, samples_per_epoch, current_block)
            prompt_texts = [format_coding_prompt(p) for p in epoch_prompts]
            logger.info(f"Selected {len(prompt_texts)} prompts (block seed: {current_block})")

            # ── Tokenize prompts ──────────────────────────────────────
            input_ids_list = []
            for text in prompt_texts:
                ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS,
                ).input_ids.to(device)
                input_ids_list.append(ids)

            # ── Evaluate each model ───────────────────────────────────
            for uid, commit in to_eval:
                model_repo = commit["model"]
                revision = commit.get("revision", "main")
                student = None

                try:
                    # 1. Architecture check
                    check = check_model_architecture(model_repo, revision, max_student_params_b)
                    if not check["pass"]:
                        logger.warning(f"UID {uid} model check failed: {check['reason']}")
                        record_failure(uid, failures)
                        continue

                    # 2. Tokenizer verification
                    tok_ok, tok_reason = verify_tokenizer(teacher_model, model_repo)
                    if not tok_ok:
                        logger.warning(f"UID {uid} tokenizer mismatch: {tok_reason}")
                        record_failure(uid, failures)
                        continue

                    # 3. Copy detection
                    model_hash = compute_model_hash(model_repo, revision)
                    if model_hash:
                        dup_uid = check_duplicate_hash(model_hash, uid, state_path)
                        if dup_uid is not None:
                            logger.warning(
                                f"UID {uid} model is duplicate of UID {dup_uid} — rejected"
                            )
                            record_failure(uid, failures)
                            continue
                        register_model_hash(model_hash, uid, state_path)

                    # 4. Load student
                    logger.info(f"Evaluating UID {uid}: {model_repo}@{revision[:12] if revision else 'main'} "
                                f"({check.get('params_b', '?'):.2f}B total)")
                    student = AutoModelForCausalLM.from_pretrained(
                        model_repo,
                        revision=revision,
                        torch_dtype=torch.bfloat16,
                        device_map=device,
                        trust_remote_code=True,
                    )
                    student.eval()

                    # 5. KL evaluation with teacher continuation
                    kl_results = []
                    for i, ids in enumerate(input_ids_list):
                        result = evaluate_kl_with_continuation(
                            teacher, student, ids,
                            max_new_tokens=max_new_tokens,
                            device=device,
                            block_seed=current_block,
                        )
                        kl_results.append(result)
                        logger.debug(
                            f"  Prompt {i}: KL={result['kl_mean']:.4f} "
                            f"(gen_len={result['gen_len']}, positions={result['n_positions']})"
                        )

                    # Weighted average by number of positions
                    total_positions = sum(r["n_positions"] for r in kl_results)
                    if total_positions == 0:
                        logger.warning(f"UID {uid}: no positions evaluated")
                        record_failure(uid, failures)
                        continue

                    avg_kl = sum(
                        r["kl_mean"] * r["n_positions"] for r in kl_results
                    ) / total_positions

                    logger.info(
                        f"  UID {uid}: avg KL={avg_kl:.6f} "
                        f"({total_positions} total positions across {len(kl_results)} prompts)"
                    )

                    # 6. Update EMA
                    new_ema = update_ema(uid, avg_kl, ema_scores, EMA_ALPHA)
                    logger.info(f"  UID {uid}: EMA KL={new_ema:.6f}")

                    # 7. Update cache
                    commit_cache[str(uid)] = {
                        "model": model_repo,
                        "revision": revision,
                        "kl": avg_kl,
                        "block": current_block,
                    }
                    reset_failures(uid, failures)

                except Exception as e:
                    logger.error(f"UID {uid} evaluation failed: {e}")
                    traceback.print_exc()
                    record_failure(uid, failures)

                finally:
                    if student is not None:
                        del student
                    free_gpu()

            # ── Compute and set proportional weights ──────────────────
            weights = compute_proportional_weights(
                ema_scores, failures, metagraph.n,
                max_kl=MAX_KL_THRESHOLD,
            )

            non_zero = [(i, w) for i, w in enumerate(weights) if w > 0]
            if non_zero:
                logger.info("Weight distribution:")
                for uid, w in sorted(non_zero, key=lambda x: -x[1])[:10]:
                    logger.info(f"  UID {uid}: weight={w:.4f} (EMA KL={ema_scores.get(str(uid), '?')})")

                uids = list(range(metagraph.n))
                for attempt in range(3):
                    try:
                        success = subtensor.set_weights(
                            wallet=wallet, netuid=netuid,
                            uids=uids, weights=weights,
                            wait_for_inclusion=True,
                            wait_for_finalization=True,
                        )
                        if success:
                            logger.info("Weights set successfully")
                            break
                        logger.warning(f"Weight setting attempt {attempt + 1} rejected")
                    except Exception as e:
                        logger.error(f"Weight setting attempt {attempt + 1} failed: {e}")
                    time.sleep(30)
            else:
                logger.info("No valid miners — skipping weight setting")

            # ── Persist state ─────────────────────────────────────────
            save_ema_scores(ema_scores, state_path)
            save_failures(failures, state_path)
            save_commitment_cache(commit_cache, state_path)

            logger.info(f"Epoch complete, sleeping {tempo}s")
            time.sleep(tempo)

        except KeyboardInterrupt:
            logger.info("Shutting down — persisting state")
            save_ema_scores(ema_scores, state_path)
            save_failures(failures, state_path)
            save_commitment_cache(commit_cache, state_path)
            break

        except Exception as e:
            logger.error(f"Epoch error: {e}")
            traceback.print_exc()
            # Persist state even on error
            try:
                save_ema_scores(ema_scores, state_path)
                save_failures(failures, state_path)
                save_commitment_cache(commit_cache, state_path)
            except Exception:
                pass
            time.sleep(60)


if __name__ == "__main__":
    main()
