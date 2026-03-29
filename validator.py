"""
Distillation subnet validator.

Evaluates miners' distilled GLM-5 models by comparing KL-divergence of
logprobs on coding prompts from the SweInfinite dataset.

Follows the Chi pattern: single-file validator with click CLI.
No synapse/axon/dendrite — miners commit HuggingFace model links on-chain
and validators pull models directly for local GPU evaluation.
"""

import json
import logging
import os
import sys
import threading
import time

import bittensor as bt
import click
import numpy as np
from bittensor_wallet import Wallet

from eval.dataset import format_coding_prompt, load_swe_infinite_prompts, sample_prompts
from eval.inference import (
    generate_with_logprobs,
    get_model_params_billions,
    load_model,
    unload_model,
)
from eval.kl_divergence import compute_kl_divergence
from eval.tokenizer import check_tokenizer_compatibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEARTBEAT_TIMEOUT = 600  # seconds
TEACHER_MODEL = "zai-org/GLM-5"
TEACHER_TOTAL_PARAMS_B = 744.0
DEFAULT_MAX_PARAM_RATIO = 0.1  # student ≤ 10% of teacher


# ---------------------------------------------------------------------------
# Heartbeat monitor (Chi pattern)
# ---------------------------------------------------------------------------


def heartbeat_monitor(
    last_heartbeat: list[float], stop_event: threading.Event
) -> None:
    """Kill the process if the main loop stalls for too long."""
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            logger.error(
                "No heartbeat for %d seconds — restarting.", HEARTBEAT_TIMEOUT
            )
            logging.shutdown()
            os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# Commitment helpers
# ---------------------------------------------------------------------------


def read_miner_commitment(
    subtensor: bt.Subtensor, netuid: int, hotkey_ss58: str
) -> dict | None:
    """
    Read a miner's commitment from the Commitments pallet.

    Expected commitment JSON:
        {"model_repo": "user/model", "tokenizer": "...", "params_b": 70.0}

    Returns parsed dict or None on failure.
    """
    try:
        result = subtensor.query_module(
            "Commitments", "CommitmentOf", [netuid, hotkey_ss58]
        )
        if result is None:
            return None

        # The commitment is stored as nested fields — extract UTF-8 bytes
        info = result.value if hasattr(result, "value") else result
        if isinstance(info, dict):
            fields = info.get("info", {}).get("fields", [])
            for field_group in fields:
                for field in field_group if isinstance(field_group, list) else [field_group]:
                    if isinstance(field, dict) and "Utf8" in field:
                        raw = field["Utf8"]
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        return json.loads(raw)

        # Fallback: try treating whole result as string
        if isinstance(info, (str, bytes)):
            raw = info.decode() if isinstance(info, bytes) else info
            return json.loads(raw)

    except Exception as exc:
        logger.debug("Could not read commitment for %s: %s", hotkey_ss58, exc)

    return None


# ---------------------------------------------------------------------------
# Cached tokenizer compatibility checks
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, bool] = {}


def cached_tokenizer_check(model_repo: str, teacher_model: str) -> bool:
    """Check tokenizer compat with caching so we don't re-download each epoch."""
    if model_repo not in _tokenizer_cache:
        _tokenizer_cache[model_repo] = check_tokenizer_compatibility(
            model_repo, teacher_model
        )
    return _tokenizer_cache[model_repo]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--network",
    default=lambda: os.getenv("NETWORK", "finney"),
    help="Network to connect to (finney, test, local)",
)
@click.option(
    "--netuid",
    type=int,
    default=lambda: int(os.getenv("NETUID", "1")),
    help="Subnet netuid",
)
@click.option(
    "--coldkey",
    default=lambda: os.getenv("WALLET_NAME", "default"),
    help="Wallet coldkey name",
)
@click.option(
    "--hotkey",
    default=lambda: os.getenv("HOTKEY_NAME", "default"),
    help="Wallet hotkey name",
)
@click.option(
    "--teacher-model",
    default=lambda: os.getenv("TEACHER_MODEL", TEACHER_MODEL),
    help="HuggingFace repo for the teacher model (GLM-5)",
)
@click.option(
    "--max-param-ratio",
    type=float,
    default=DEFAULT_MAX_PARAM_RATIO,
    help="Maximum student/teacher parameter ratio (default 0.1 = 10%%)",
)
@click.option(
    "--dataset-path",
    default=lambda: os.getenv("DATASET_PATH", "./dataset"),
    help="Path to SweInfinite dataset directory",
)
@click.option(
    "--samples-per-epoch",
    type=int,
    default=5,
    help="Number of coding prompts to evaluate per epoch",
)
@click.option(
    "--max-tokens",
    type=int,
    default=128,
    help="Maximum tokens to generate per prompt",
)
@click.option(
    "--top-k-logprobs",
    type=int,
    default=50,
    help="Number of top logprobs to collect per token",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=lambda: int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
    help="vLLM tensor parallel size (GPUs per model)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.90,
    help="Fraction of GPU memory vLLM may use",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=lambda: os.getenv("LOG_LEVEL", "INFO"),
    help="Logging level",
)
def main(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    teacher_model: str,
    max_param_ratio: float,
    dataset_path: str,
    samples_per_epoch: int,
    max_tokens: int,
    top_k_logprobs: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    log_level: str,
) -> None:
    """Run the Distillation subnet validator."""

    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    max_student_params_b = TEACHER_TOTAL_PARAMS_B * max_param_ratio

    logger.info(
        "Starting validator — network=%s  netuid=%d  teacher=%s  max_student=%.1fB",
        network,
        netuid,
        teacher_model,
        max_student_params_b,
    )

    # ── Heartbeat ──────────────────────────────────────────────────────
    last_heartbeat: list[float] = [time.time()]
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=heartbeat_monitor,
        args=(last_heartbeat, stop_event),
        daemon=True,
    )
    hb_thread.start()

    try:
        # ── Bittensor setup ────────────────────────────────────────────
        wallet = Wallet(name=coldkey, hotkey=hotkey)
        subtensor = bt.Subtensor(network=network)
        metagraph = bt.Metagraph(netuid=netuid, network=network)
        metagraph.sync(subtensor=subtensor)
        logger.info("Metagraph synced: %d neurons at block %d", metagraph.n, metagraph.block)

        my_hotkey = wallet.hotkey.ss58_address
        if my_hotkey not in metagraph.hotkeys:
            logger.error("Hotkey %s not registered on netuid %d", my_hotkey, netuid)
            stop_event.set()
            return
        my_uid = metagraph.hotkeys.index(my_hotkey)
        logger.info("Validator UID: %d", my_uid)

        tempo = subtensor.get_subnet_hyperparameters(netuid).tempo
        logger.info("Subnet tempo: %d blocks", tempo)

        # ── Load teacher model (kept resident) ─────────────────────────
        logger.info("Loading teacher model: %s", teacher_model)
        teacher_llm = load_model(
            teacher_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("Teacher model loaded successfully")

        # ── Load dataset ───────────────────────────────────────────────
        all_prompts = load_swe_infinite_prompts(dataset_path)
        if not all_prompts:
            logger.error("No prompts found in %s — cannot evaluate", dataset_path)
            stop_event.set()
            return
        logger.info("Loaded %d coding prompts from SweInfinite", len(all_prompts))

        last_weight_block = 0

        # ── Main loop ─────────────────────────────────────────────────
        while True:
            try:
                metagraph.sync(subtensor=subtensor)
                current_block = subtensor.get_current_block()
                last_heartbeat[0] = time.time()

                blocks_since_last = current_block - last_weight_block
                if blocks_since_last < tempo:
                    logger.debug(
                        "Block %d: waiting for tempo (%d/%d)",
                        current_block,
                        blocks_since_last,
                        tempo,
                    )
                    time.sleep(12)
                    continue

                logger.info(
                    "Block %d: starting evaluation epoch", current_block
                )

                # ── Sample prompts for this epoch ─────────────────────
                epoch_problems = sample_prompts(all_prompts, samples_per_epoch)
                epoch_texts = [format_coding_prompt(p) for p in epoch_problems]

                # ── Get teacher logprobs (once per epoch) ─────────────
                logger.info("Running teacher inference on %d prompts", len(epoch_texts))
                teacher_results = generate_with_logprobs(
                    teacher_llm,
                    epoch_texts,
                    max_tokens=max_tokens,
                    top_k_logprobs=top_k_logprobs,
                )

                # ── Evaluate each miner ───────────────────────────────
                scores: dict[int, float] = {}  # uid → kl-divergence

                for uid in range(metagraph.n):
                    if uid == my_uid:
                        continue

                    hotkey_ss58 = metagraph.hotkeys[uid]
                    commitment = read_miner_commitment(subtensor, netuid, hotkey_ss58)
                    if commitment is None:
                        logger.debug("UID %d: no commitment", uid)
                        continue

                    model_repo = commitment.get("model_repo")
                    if not model_repo:
                        logger.debug("UID %d: commitment missing model_repo", uid)
                        continue

                    logger.info("UID %d: evaluating model %s", uid, model_repo)

                    # ── Param-count gate ──────────────────────────────
                    try:
                        params_b = get_model_params_billions(model_repo)
                    except Exception as exc:
                        logger.warning("UID %d: cannot get param count for %s: %s", uid, model_repo, exc)
                        continue

                    if params_b <= 0:
                        logger.warning("UID %d: unknown param count, skipping", uid)
                        continue
                    if params_b > max_student_params_b:
                        logger.warning(
                            "UID %d: model too large (%.1fB > %.1fB), skipping",
                            uid,
                            params_b,
                            max_student_params_b,
                        )
                        continue

                    # ── Tokenizer gate ────────────────────────────────
                    if not cached_tokenizer_check(model_repo, teacher_model):
                        logger.warning(
                            "UID %d: tokenizer incompatible, skipping", uid
                        )
                        continue

                    # ── Student inference ─────────────────────────────
                    try:
                        student_llm = load_model(
                            model_repo,
                            tensor_parallel_size=tensor_parallel_size,
                            gpu_memory_utilization=gpu_memory_utilization,
                        )
                        student_results = generate_with_logprobs(
                            student_llm,
                            epoch_texts,
                            max_tokens=max_tokens,
                            top_k_logprobs=top_k_logprobs,
                        )
                        unload_model(student_llm)
                    except Exception as exc:
                        logger.error(
                            "UID %d: inference failed for %s: %s",
                            uid,
                            model_repo,
                            exc,
                        )
                        continue

                    # ── Compute average KL-divergence across prompts ──
                    kl_values: list[float] = []
                    for t_res, s_res in zip(teacher_results, student_results):
                        kl = compute_kl_divergence(
                            t_res["logprobs"], s_res["logprobs"]
                        )
                        if kl != float("inf"):
                            kl_values.append(kl)

                    if kl_values:
                        avg_kl = float(np.mean(kl_values))
                        scores[uid] = avg_kl
                        logger.info(
                            "UID %d: avg KL-divergence = %.6f", uid, avg_kl
                        )
                    else:
                        logger.warning("UID %d: all KL computations failed", uid)

                # ── Winner-take-all weight setting ────────────────────
                if not scores:
                    logger.warning("No valid miners this epoch — skipping weight set")
                    time.sleep(12)
                    continue

                winner_uid = min(scores, key=scores.get)  # type: ignore[arg-type]
                winner_kl = scores[winner_uid]
                logger.info(
                    "Winner: UID %d with KL-divergence %.6f",
                    winner_uid,
                    winner_kl,
                )

                # Build weight vector: winner gets 1.0, everyone else 0.0
                uids_list = list(range(metagraph.n))
                weights_list = [0.0] * metagraph.n
                weights_list[winner_uid] = 1.0

                success = subtensor.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids_list,
                    weights=weights_list,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )

                if success:
                    logger.info("Weights set successfully for %d neurons", len(uids_list))
                    last_weight_block = current_block
                else:
                    logger.warning("Failed to set weights on chain")

                time.sleep(12)

            except KeyboardInterrupt:
                logger.info("Validator stopped by user")
                break
            except Exception as exc:
                logger.error("Error in validator loop: %s", exc, exc_info=True)
                time.sleep(12)

    finally:
        stop_event.set()
        hb_thread.join(timeout=2)


if __name__ == "__main__":
    main()
