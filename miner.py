#!/usr/bin/env python3
"""
Distillation Subnet Miner (v0.5.0)

Commits a distilled model to the Bittensor chain.
Each hotkey can only have ONE active commitment — re-running replaces the old one.

Usage:
    python miner.py --model-repo user/my-distilled-qwen --network finney --netuid 1
"""
import os
import sys
import json
import time
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.miner")

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.1  # 3.5B max


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--wallet-name", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey-name", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--model-repo", required=True, help="HuggingFace repo e.g. 'user/distilled-qwen'")
@click.option("--revision", default=None, help="HF commit SHA (latest if omitted)")
def main(network, netuid, wallet_name, hotkey_name, model_repo, revision):
    """Commit a distilled model to the distillation subnet."""
    import bittensor as bt
    from huggingface_hub import repo_info
    from eval.model_checker import check_model_architecture

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO

    # Resolve revision
    if not revision:
        info = repo_info(model_repo, repo_type="model")
        revision = info.sha
        logger.info(f"Using latest revision: {revision[:12]}...")

    # Pre-flight architecture check
    logger.info(f"Checking model: {model_repo}@{revision[:12]}...")
    check = check_model_architecture(model_repo, revision, max_params_b)
    if not check["pass"]:
        logger.error(f"Model check failed: {check['reason']}")
        sys.exit(1)
    logger.info(f"Model check passed: {check.get('params_b', '?'):.2f}B params, "
                f"vocab_size={check.get('vocab_size', '?')}")

    # Commit on-chain
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    subtensor = bt.Subtensor(network=network)

    commit_data = json.dumps({"model": model_repo, "revision": revision})
    logger.info(f"Committing: {commit_data}")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )
    logger.info("Commitment submitted successfully")

    # Keep alive
    logger.info("Miner is live. Press Ctrl+C to exit.")
    logger.info("To update your model, re-run with a new --model-repo.")
    try:
        while True:
            time.sleep(600)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
