#!/usr/bin/env python3
"""
Distillation Subnet Validator

Evaluates miners' distilled GLM-5 models by KL-divergence of logprobs.
Miners commit HuggingFace model links on-chain; validator downloads and evaluates locally.
Winner-take-all: lowest KL-divergence gets all weight.
"""
import os, sys, time, json, random, math, logging, traceback
import click
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.validator")

# ── Constants ──────────────────────────────────────────────────────────────
TEACHER_MODEL = "zai-org/GLM-5"
TEACHER_TOTAL_PARAMS_B = 744.0
DEFAULT_MAX_PARAM_RATIO = 0.1  # Miner model must be ≤ 10% of teacher


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--wallet-name", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey-name", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--teacher-model", default=TEACHER_MODEL)
@click.option("--max-param-ratio", type=float, default=DEFAULT_MAX_PARAM_RATIO)
@click.option("--dataset-path", default="./dataset")
@click.option("--samples-per-epoch", type=int, default=5)
@click.option("--max-tokens", type=int, default=128)
@click.option("--top-k-logprobs", type=int, default=50)
@click.option("--tensor-parallel-size", type=int, default=1)
@click.option("--tempo", type=int, default=360, help="Seconds between evaluation epochs")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
def main(
    network, netuid, wallet_name, hotkey_name, teacher_model, max_param_ratio,
    dataset_path, samples_per_epoch, max_tokens, top_k_logprobs,
    tensor_parallel_size, tempo, log_level,
):
    """Run the distillation subnet validator."""
    logging.getLogger().setLevel(getattr(logging, log_level))

    import bittensor as bt
    from eval.inference import load_model, generate_with_logprobs, unload_model
    from eval.kl_divergence import compute_kl_divergence
    from eval.dataset import load_swe_infinite_prompts, sample_prompts, format_coding_prompt
    from eval.model_checker import check_model_architecture

    max_student_params_b = TEACHER_TOTAL_PARAMS_B * max_param_ratio

    # Init
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)

    # Load dataset
    all_prompts = load_swe_infinite_prompts(dataset_path)
    logger.info(f"Loaded {len(all_prompts)} prompts from {dataset_path}")

    # Load teacher model (kept resident)
    logger.info(f"Loading teacher model: {teacher_model}")
    teacher_llm = load_model(teacher_model, tensor_parallel_size=tensor_parallel_size)
    logger.info("Teacher model loaded")

    # ── Main loop ──────────────────────────────────────────────────────
    while True:
        try:
            metagraph.sync(subtensor=subtensor)

            # Read all miner commitments
            commitments = {}  # uid -> {model, revision}
            for uid in range(metagraph.n):
                hotkey = metagraph.hotkeys[uid]
                try:
                    revealed = subtensor.get_all_revealed_commitments(netuid)
                    if hotkey in revealed:
                        _, commit_data = revealed[hotkey][-1]
                        data = json.loads(commit_data)
                        if "model" in data:
                            commitments[uid] = data
                except Exception:
                    continue

            logger.info(f"Found {len(commitments)} miner commitments")

            if not commitments:
                logger.info(f"No commitments found, sleeping {tempo}s")
                time.sleep(tempo)
                continue

            # Sample prompts for this epoch
            epoch_prompts = sample_prompts(all_prompts, samples_per_epoch)
            prompt_texts = [format_coding_prompt(p) for p in epoch_prompts]

            # Get teacher logprobs
            logger.info(f"Generating teacher logprobs for {len(prompt_texts)} prompts")
            teacher_results = generate_with_logprobs(
                teacher_llm, prompt_texts, max_tokens=max_tokens, top_k_logprobs=top_k_logprobs
            )

            # Evaluate each miner
            scores = {}  # uid -> kl_divergence (lower is better)

            for uid, commitment in commitments.items():
                model_repo = commitment["model"]
                revision = commitment.get("revision")

                try:
                    # 1. Check model architecture & size
                    check_result = check_model_architecture(model_repo, revision, max_student_params_b)
                    if not check_result["pass"]:
                        logger.warning(f"UID {uid} model check failed: {check_result['reason']}")
                        continue

                    # 2. Load student model
                    logger.info(f"Evaluating UID {uid}: {model_repo}")
                    student_llm = load_model(
                        model_repo, revision=revision, tensor_parallel_size=tensor_parallel_size
                    )

                    # 3. Generate student logprobs
                    student_results = generate_with_logprobs(
                        student_llm, prompt_texts,
                        max_tokens=max_tokens, top_k_logprobs=top_k_logprobs,
                    )

                    # 4. Compute KL-divergence averaged across prompts
                    kl_divs = []
                    for t_res, s_res in zip(teacher_results, student_results):
                        kl = compute_kl_divergence(t_res["logprobs"], s_res["logprobs"])
                        kl_divs.append(kl)

                    avg_kl = np.mean(kl_divs)
                    scores[uid] = avg_kl
                    logger.info(f"  UID {uid}: avg KL-div = {avg_kl:.6f}")

                except Exception as e:
                    logger.error(f"  UID {uid} evaluation failed: {e}")
                    traceback.print_exc()
                finally:
                    # Unload student to free GPU
                    if "student_llm" in locals():
                        unload_model(student_llm)
                        del student_llm

            # ── Winner-take-all weight setting ──
            if scores:
                winner_uid = min(scores, key=scores.get)
                logger.info(f"Winner: UID {winner_uid} with KL-div {scores[winner_uid]:.6f}")

                uids = list(range(metagraph.n))
                weights = [0.0] * metagraph.n
                weights[winner_uid] = 1.0

                # Set weights with retry
                for attempt in range(3):
                    try:
                        success = subtensor.set_weights(
                            wallet=wallet, netuid=netuid,
                            uids=uids, weights=weights,
                            wait_for_inclusion=True, wait_for_finalization=True,
                        )
                        if success:
                            logger.info("Weights set successfully")
                            break
                        logger.warning(f"Weight setting attempt {attempt+1} rejected")
                    except Exception as e:
                        logger.error(f"Weight setting attempt {attempt+1} failed: {e}")
                    time.sleep(30)

            logger.info(f"Epoch complete, sleeping {tempo}s")
            time.sleep(tempo)

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Epoch error: {e}")
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    main()
