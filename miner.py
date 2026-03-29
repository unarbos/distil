#!/usr/bin/env python3
"""
Distillation Subnet Miner (v3)

Changes from v2:
  1. One submission per hotkey — submitting a new model replaces the old commitment.
     The miner enforces this by always committing (no "already committed" check).
"""
import os, sys, json, time, logging
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.miner")


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--wallet-name", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey-name", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--model-repo", required=True, help="HuggingFace repo e.g. 'user/distilled-glm5'")
@click.option("--revision", default=None, help="HF commit SHA (latest if omitted)")
def main(network, netuid, wallet_name, hotkey_name, model_repo, revision):
    """Commit a distilled model to the distillation subnet.

    Each hotkey can only have ONE active commitment. Submitting a new model
    replaces the previous commitment — the validator always uses the latest.
    """
    import bittensor as bt
    from huggingface_hub import repo_info
    from eval.model_checker import check_model_architecture

    max_params_b = 74.4  # 10% of GLM-5's 744B

    # Resolve revision to latest if not specified
    if not revision:
        info = repo_info(model_repo, repo_type="model")
        revision = info.sha
        logger.info(f"Using latest revision: {revision[:12]}...")

    # Pre-flight: check model architecture
    logger.info(f"Checking model: {model_repo}@{revision[:12]}...")
    check = check_model_architecture(model_repo, revision, max_params_b)
    if not check["pass"]:
        logger.error(f"Model check failed: {check['reason']}")
        sys.exit(1)
    logger.info(f"Model check passed: {check.get('params_b', '?')}B params")

    # Init wallet & subtensor
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    subtensor = bt.Subtensor(network=network)

    # ─── Change 1: Always commit — replaces any previous commitment ───
    # The validator only reads the LATEST commitment per hotkey, so
    # re-committing is the correct way to update your model.
    commit_data = json.dumps({"model": model_repo, "revision": revision})
    logger.info(f"Committing (replaces any previous): {commit_data}")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )
    logger.info("Commitment submitted — this is now your ONLY active submission")

    # Keep alive
    logger.info("Miner is live. Press Ctrl+C to exit.")
    logger.info("To update your model, re-run this script with a new --model-repo.")
    try:
        while True:
            time.sleep(600)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
