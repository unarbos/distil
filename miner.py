#!/usr/bin/env python3
"""
Distillation Subnet Miner

Commits a HuggingFace model URL + revision on-chain.
Model must use same tokenizer as GLM-5 and have ≤ 74.4B params.
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
    """Commit a distilled model to the distillation subnet."""
    import bittensor as bt
    from huggingface_hub import model_info, repo_info
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

    # Commit
    commit_data = json.dumps({"model": model_repo, "revision": revision})
    logger.info(f"Committing: {commit_data}")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )
    logger.info("Commitment submitted successfully")

    # Keep alive — re-commit periodically if desired
    logger.info("Miner is live. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(600)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
