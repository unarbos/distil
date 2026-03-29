"""
Distillation subnet miner.

Commits a HuggingFace model URL to the chain via the Commitments pallet.
The model must:
  1. Be a distilled/quantized version of GLM-5
  2. Use the same tokenizer as GLM-5
  3. Have ≤ 74.4 B total parameters

This script verifies the model metadata, publishes the commitment, and
then keeps alive — periodically re-committing if the model repo changes.
"""

import json
import logging
import os
import sys
import time

import bittensor as bt
import click
from bittensor_wallet import Wallet

from eval.inference import get_model_params_billions
from eval.tokenizer import check_tokenizer_compatibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEACHER_MODEL = "zai-org/GLM-5"
TEACHER_TOTAL_PARAMS_B = 744.0
DEFAULT_MAX_PARAM_RATIO = 0.1
RECOMMIT_INTERVAL_BLOCKS = 300  # re-commit every ~1 hour


# ---------------------------------------------------------------------------
# Commitment helper
# ---------------------------------------------------------------------------


def commit_model(
    subtensor: bt.Subtensor,
    wallet: Wallet,
    netuid: int,
    commitment_data: str,
) -> bool:
    """Write a UTF-8 commitment string to the Commitments pallet."""
    try:
        call = subtensor.substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": netuid,
                "info": {
                    "fields": [[{"Utf8": commitment_data.encode()}]],
                },
            },
        )
        extrinsic = subtensor.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.hotkey
        )
        receipt = subtensor.substrate.submit_extrinsic(
            extrinsic, wait_for_inclusion=True
        )
        if hasattr(receipt, "is_success") and not receipt.is_success:
            logger.error("Commitment extrinsic failed: %s", receipt.error_message)
            return False
        return True
    except Exception as exc:
        logger.error("Failed to commit: %s", exc)
        return False


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
    "--model-repo",
    required=True,
    help="HuggingFace model repo (e.g. 'username/distilled-glm5')",
)
@click.option(
    "--teacher-model",
    default=lambda: os.getenv("TEACHER_MODEL", TEACHER_MODEL),
    help="Teacher model for tokenizer compatibility check",
)
@click.option(
    "--max-param-ratio",
    type=float,
    default=DEFAULT_MAX_PARAM_RATIO,
    help="Maximum student/teacher parameter ratio",
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
    model_repo: str,
    teacher_model: str,
    max_param_ratio: float,
    log_level: str,
) -> None:
    """Run the Distillation subnet miner."""

    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    max_student_params_b = TEACHER_TOTAL_PARAMS_B * max_param_ratio

    logger.info("Distillation miner starting — model_repo=%s", model_repo)

    # ── Pre-flight checks ─────────────────────────────────────────────

    # 1. Parameter count
    logger.info("Checking parameter count for %s …", model_repo)
    try:
        params_b = get_model_params_billions(model_repo)
    except Exception as exc:
        logger.error("Cannot query model info for %s: %s", model_repo, exc)
        sys.exit(1)

    if params_b <= 0:
        logger.warning("Could not determine param count — proceeding with caution")
    elif params_b > max_student_params_b:
        logger.error(
            "Model too large: %.1fB > %.1fB limit. Aborting.",
            params_b,
            max_student_params_b,
        )
        sys.exit(1)
    else:
        logger.info("Parameter count OK: %.1fB (limit %.1fB)", params_b, max_student_params_b)

    # 2. Tokenizer compatibility
    logger.info("Checking tokenizer compatibility …")
    if not check_tokenizer_compatibility(model_repo, teacher_model):
        logger.error(
            "Tokenizer of %s is NOT compatible with %s. Aborting.",
            model_repo,
            teacher_model,
        )
        sys.exit(1)
    logger.info("Tokenizer check passed ✓")

    # ── Bittensor setup ───────────────────────────────────────────────

    wallet = Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)
    metagraph = bt.Metagraph(netuid=netuid, network=network)
    metagraph.sync(subtensor=subtensor)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        logger.error("Hotkey %s not registered on netuid %d", my_hotkey, netuid)
        sys.exit(1)
    my_uid = metagraph.hotkeys.index(my_hotkey)
    logger.info("Miner UID: %d", my_uid)

    # ── Build commitment JSON ─────────────────────────────────────────

    commitment_json = json.dumps(
        {
            "model_repo": model_repo,
            "tokenizer": teacher_model,
            "params_b": round(params_b, 1) if params_b > 0 else None,
        },
        separators=(",", ":"),
    )
    logger.info("Commitment payload: %s", commitment_json)

    # ── Initial commit ────────────────────────────────────────────────

    if commit_model(subtensor, wallet, netuid, commitment_json):
        logger.info("Initial commitment published successfully ✓")
    else:
        logger.error("Failed to publish initial commitment")
        sys.exit(1)

    # ── Keep-alive loop: re-commit periodically ───────────────────────

    last_commit_block = subtensor.get_current_block()

    while True:
        try:
            time.sleep(12)  # ~1 block
            current_block = subtensor.get_current_block()

            if current_block - last_commit_block >= RECOMMIT_INTERVAL_BLOCKS:
                logger.info(
                    "Block %d: re-committing (every %d blocks)",
                    current_block,
                    RECOMMIT_INTERVAL_BLOCKS,
                )
                if commit_model(subtensor, wallet, netuid, commitment_json):
                    last_commit_block = current_block
                    logger.info("Re-commitment successful ✓")
                else:
                    logger.warning("Re-commitment failed — will retry next interval")

        except KeyboardInterrupt:
            logger.info("Miner stopped by user")
            break
        except Exception as exc:
            logger.error("Error in keep-alive loop: %s", exc)
            time.sleep(12)


if __name__ == "__main__":
    main()
