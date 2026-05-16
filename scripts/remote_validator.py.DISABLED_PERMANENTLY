#!/usr/bin/env python3
"""Remote Validator CLI entry point.

King-of-the-Hill orchestration lives in :mod:`scripts.validator.service`.
This file is intentionally a thin Click wrapper so the upload path used by
`systemd`, PM2 and ad-hoc shell invocations stays stable.
"""
import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
for _lib in ("paramiko", "paramiko.transport", "paramiko.sftp", "urllib3", "httpx"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logging.getLogger("distillation.remote_validator").setLevel(logging.DEBUG)

from eval.runtime import MAX_STUDENT_PARAMS
from scripts.validator.config import NETUID
from scripts.validator.service import run_validator

# 2026-05-02 (v30.5): default max-params-b now reads from
# subnet-config.json via eval.runtime.MAX_STUDENT_PARAMS so changes
# to the config (40B cap, future Kimi K2.6 swap) automatically apply
# without needing to update the systemd service args.
_DEFAULT_MAX_PARAMS_B = MAX_STUDENT_PARAMS / 1e9


@click.command()
@click.option("--network", default="finney")
@click.option("--netuid", type=int, default=NETUID)
@click.option("--wallet-name", default="affine")
@click.option("--hotkey-name", default="validator")
@click.option("--wallet-path", default="~/.bittensor/wallets/")
@click.option("--lium-api-key", required=True, envvar="LIUM_API_KEY")
@click.option("--lium-pod-name", default="distil-validator")
@click.option("--state-dir", default="state")
@click.option("--max-params-b", type=float, default=_DEFAULT_MAX_PARAMS_B)
@click.option("--tempo", type=int, default=360, help="Seconds between epochs")
@click.option("--once", is_flag=True, help="Run one epoch and exit (for testing)")
@click.option("--use-vllm", is_flag=True, default=False, envvar="USE_VLLM",
              help="Use vLLM-accelerated evaluation")
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         lium_api_key, lium_pod_name, state_dir, max_params_b, tempo, once, use_vllm):
    """Run the distillation validator with king-of-the-hill evaluation."""
    return run_validator(
        network,
        netuid,
        wallet_name,
        hotkey_name,
        wallet_path,
        lium_api_key,
        lium_pod_name,
        state_dir,
        max_params_b,
        tempo,
        once,
        use_vllm,
    )


if __name__ == "__main__":
    main()
