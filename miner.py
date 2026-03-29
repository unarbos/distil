#!/usr/bin/env python3
"""
Distillation Subnet Miner — One-time model commitment.

This is a one-shot script. It commits your distilled model to the chain
and exits. Once committed, your model is PERMANENT — you cannot update,
replace, or re-commit. One model per hotkey, forever.

Usage:
    python miner.py \
        --model-repo user/my-distilled-qwen \
        --wallet-name my_wallet \
        --hotkey-name my_hotkey \
        --network finney \
        --netuid 1
"""
import os
import sys
import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.miner")

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15  # ~5.25B max


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--wallet-name", required=True, help="Name of existing Bittensor wallet")
@click.option("--wallet-path", default="~/.bittensor/wallets", help="Path to wallet directory")
@click.option("--hotkey-name", required=True, help="Name of existing hotkey")
@click.option("--model-repo", required=True, help="HuggingFace repo e.g. 'user/distilled-qwen'")
@click.option("--revision", default=None, help="HF commit SHA (pinned at latest if omitted)")
@click.option("--force", is_flag=True, help="Skip the existing-commitment check (DANGEROUS)")
def main(network, netuid, wallet_name, wallet_path, hotkey_name, model_repo, revision, force):
    """
    Commit a distilled model to the distillation subnet.

    This is PERMANENT. Once committed, you cannot change your model.
    One commitment per hotkey, forever.
    """
    import bittensor as bt
    from huggingface_hub import repo_info
    from eval.model_checker import check_model_architecture

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO

    # ── Resolve revision (pin to specific SHA) ─────────────────────────
    if not revision:
        info = repo_info(model_repo, repo_type="model")
        revision = info.sha
        logger.info(f"Pinning to latest revision: {revision[:12]}...")
    else:
        logger.info(f"Using specified revision: {revision[:12]}...")

    # ── Pre-flight architecture check ──────────────────────────────────
    logger.info(f"Checking model: {model_repo}@{revision[:12]}...")
    check = check_model_architecture(model_repo, revision, max_params_b)
    if not check["pass"]:
        logger.error(f"Model check FAILED: {check['reason']}")
        logger.error("Your model does not meet subnet requirements. Fix and retry.")
        sys.exit(1)

    logger.info(f"✓ Model check passed: {check.get('params_b', 0):.2f}B params, "
                f"vocab_size={check.get('vocab_size', '?')}")

    # ── Load existing wallet (do NOT create) ───────────────────────────
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)

    # Verify wallet exists
    try:
        _ = wallet.hotkey
    except Exception as e:
        logger.error(f"Could not load wallet '{wallet_name}' hotkey '{hotkey_name}' "
                     f"at {wallet_path}: {e}")
        logger.error("Create your wallet first with: btcli wallet create")
        sys.exit(1)

    subtensor = bt.Subtensor(network=network)

    # ── Check for existing commitment (ONE per hotkey, EVER) ───────────
    if not force:
        try:
            revealed = subtensor.get_all_revealed_commitments(netuid)
            hotkey_str = wallet.hotkey.ss58_address
            if hotkey_str in revealed and len(revealed[hotkey_str]) > 0:
                existing_block, existing_data = revealed[hotkey_str][-1]
                logger.error("=" * 60)
                logger.error("COMMITMENT ALREADY EXISTS — CANNOT UPDATE")
                logger.error("=" * 60)
                logger.error(f"  Hotkey: {hotkey_str}")
                logger.error(f"  Block:  {existing_block}")
                logger.error(f"  Data:   {existing_data}")
                logger.error("")
                logger.error("This subnet enforces ONE commitment per hotkey, permanently.")
                logger.error("You cannot update, replace, or re-commit.")
                logger.error("If you need to change models, register a new hotkey.")
                sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not check existing commitments: {e}")
            logger.warning("Proceeding — validator will enforce the one-commit rule anyway.")

    # ── Commit on-chain ────────────────────────────────────────────────
    commit_data = json.dumps({"model": model_repo, "revision": revision})

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMMITTING MODEL (this is PERMANENT)")
    logger.info("=" * 60)
    logger.info(f"  Model:    {model_repo}")
    logger.info(f"  Revision: {revision}")
    logger.info(f"  Hotkey:   {wallet.hotkey.ss58_address}")
    logger.info(f"  Network:  {network}")
    logger.info(f"  Netuid:   {netuid}")
    logger.info("")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )

    logger.info("✓ Commitment submitted successfully!")
    logger.info("")
    logger.info("Your model is now registered on the subnet.")
    logger.info("The validator will evaluate it in the next epoch.")
    logger.info("You CANNOT change this commitment — it is permanent.")


if __name__ == "__main__":
    main()
