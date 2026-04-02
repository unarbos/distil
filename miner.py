#!/usr/bin/env python3
"""
Distillation Subnet Miner — One-time model commitment.

This is a one-shot script. It commits your distilled model to the chain
and exits. Once committed, your model is PERMANENT — you cannot update,
replace, or re-commit. One model per hotkey, forever.

Usage:
    # Dry run (validate without committing):
    python miner.py \\
        --model-repo user/my-distilled-qwen \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --dry-run

    # Full commitment:
    python miner.py \\
        --model-repo user/my-distilled-qwen \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --network finney \\
        --netuid 1

    # Test-only (alias for --dry-run):
    python miner.py \\
        --model-repo user/my-distilled-qwen \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --test-only
"""
import os
import sys
import json
import logging
import time

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.miner")

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15  # ~5.25B max
MIN_BITTENSOR_VERSION = "9.5.0"


def _check_bittensor():
    """Check bittensor is installed and meets version requirement. Exit with clear message if not."""
    try:
        import bittensor as bt
        version = getattr(bt, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(version) < Version(MIN_BITTENSOR_VERSION):
            logger.error(f"bittensor {version} is too old. Required: >= {MIN_BITTENSOR_VERSION}")
            logger.error(f"Older versions are missing set_reveal_commitment().")
            logger.error(f"Upgrade with:")
            logger.error(f"    pip install bittensor>={MIN_BITTENSOR_VERSION}")
            sys.exit(1)
        return bt
    except ImportError:
        logger.error("bittensor is not installed.")
        logger.error(f"Install with:")
        logger.error(f"    pip install bittensor>={MIN_BITTENSOR_VERSION}")
        sys.exit(1)


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"),
              help="Bittensor network (default: finney)")
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")),
              help="Bittensor netuid (default: 1)")
@click.option("--wallet-name", required=True, help="Name of existing Bittensor wallet")
@click.option("--wallet-path", default="~/.bittensor/wallets", help="Path to wallet directory")
@click.option("--hotkey-name", required=True, help="Name of existing hotkey")
@click.option("--model-repo", required=True,
              help="HuggingFace repo e.g. 'user/distilled-qwen'")
@click.option("--revision", default=None, help="HF commit SHA (pinned at latest if omitted)")
@click.option("--force", is_flag=True, help="Skip the existing-commitment check (DANGEROUS)")
@click.option("--dry-run", is_flag=True,
              help="Run all validation checks without committing (safe to run)")
@click.option("--test-only", is_flag=True,
              help="Alias for --dry-run: run validation checks only")
def main(network, netuid, wallet_name, wallet_path, hotkey_name, model_repo, revision,
         force, dry_run, test_only):
    """
    Commit a distilled model to the distillation subnet.

    This is PERMANENT. Once committed, you cannot change your model.
    One commitment per hotkey, forever.

    \b
    Examples:
        # Validate only (no commitment):
        python miner.py --model-repo user/model --wallet-name w --hotkey-name h --dry-run

        # Commit for real:
        python miner.py --model-repo user/model --wallet-name w --hotkey-name h
    """
    is_dry_run = dry_run or test_only

    # ── Run validation checks ──────────────────────────────────────────
    if is_dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — running all validation checks, no commitment will be made")
        logger.info("=" * 60)

    # Import test_miner for validation (don't duplicate logic)
    from test_miner import run_all_checks, check_dependencies, BOLD, GREEN, RED, RESET

    check_dependencies()

    results = run_all_checks(
        model_repo=model_repo,
        revision=revision,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
        network=network,
        netuid=netuid,
    )

    if not results.all_passed:
        logger.error("")
        logger.error(f"Validation FAILED — {results.num_failed} check(s) did not pass.")
        logger.error("Fix the issues above before committing.")
        sys.exit(1)

    if is_dry_run:
        logger.info("")
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE — all checks passed ✓")
        logger.info("=" * 60)
        logger.info("Your model is ready. Remove --dry-run to commit for real.")
        sys.exit(0)

    # ── From here on: real commitment ──────────────────────────────────
    bt = _check_bittensor()
    from huggingface_hub import repo_info

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO

    # ── Resolve revision (pin to specific SHA) ─────────────────────────
    if not revision:
        info = repo_info(model_repo, repo_type="model")
        revision = info.sha
        logger.info(f"Pinning to latest revision: {revision}")
    else:
        logger.info(f"Using specified revision: {revision}")

    # ── Load existing wallet (do NOT create) ───────────────────────────
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)

    try:
        _ = wallet.hotkey
    except Exception as e:
        logger.error(f"Could not load wallet '{wallet_name}' hotkey '{hotkey_name}' "
                     f"at {wallet_path}: {e}")
        logger.error("Create your wallet first with:")
        logger.error(f"    btcli wallet create --name {wallet_name} --hotkey {hotkey_name}")
        sys.exit(1)

    subtensor = bt.Subtensor(network=network)

    # ── Check for existing commitment ──────────────────────────────────
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

    # ── Interactive confirmation ───────────────────────────────────────
    commit_data = json.dumps({"model": model_repo, "revision": revision})

    print("")
    print("=" * 60)
    print("  ⚠️  IRREVERSIBLE COMMITMENT — PLEASE REVIEW")
    print("=" * 60)
    print(f"  Model:    {model_repo}")
    print(f"  Revision: {revision}")
    print(f"  Hotkey:   {wallet.hotkey.ss58_address}")
    print(f"  Network:  {network}")
    print(f"  Netuid:   {netuid}")
    print(f"  Data:     {commit_data}")
    print("")
    print("  This commitment is PERMANENT. You cannot update, replace,")
    print("  or re-commit. One model per hotkey, forever.")
    print("=" * 60)
    print("")

    confirmation = input("  Type YES to confirm commitment: ").strip()
    if confirmation != "YES":
        logger.info("Commitment cancelled.")
        sys.exit(0)

    # ── Commit on-chain ────────────────────────────────────────────────
    logger.info("")
    logger.info("Submitting commitment...")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )

    logger.info("✓ Commitment submitted successfully!")

    # ── Post-commit verification ───────────────────────────────────────
    logger.info("Verifying commitment on-chain (waiting 12 seconds for block confirmation)...")
    time.sleep(12)

    try:
        revealed = subtensor.get_all_revealed_commitments(netuid)
        hotkey_str = wallet.hotkey.ss58_address
        if hotkey_str in revealed and len(revealed[hotkey_str]) > 0:
            block, data = revealed[hotkey_str][-1]
            logger.info("✓ Commitment verified on-chain!")
            logger.info(f"  Block: {block}")
            logger.info(f"  Data:  {data}")
        else:
            logger.warning("Commitment not yet visible on-chain — may take a few more blocks.")
            logger.warning("Check again in ~30 seconds with a block explorer or btcli.")
    except Exception as e:
        logger.warning(f"Could not verify commitment: {e}")
        logger.warning("The commitment was submitted. Check on-chain status manually.")

    logger.info("")
    logger.info("Your model is now registered on the subnet.")
    logger.info("The validator will evaluate it in the next epoch.")
    logger.info("You CANNOT change this commitment — it is permanent.")


if __name__ == "__main__":
    main()
