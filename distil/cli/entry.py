"""Single ``distil`` CLI entrypoint."""

from __future__ import annotations

import logging
import sys

import click


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Distil SN97 operator CLI (rewrite-v2)."""
    _setup_logging(log_level)


@main.command()
@click.option("--model-repo", required=True, help="HuggingFace repo id.")
@click.option("--revision", default="", help="Optional pinned revision.")
@click.option("--wallet-name", default=None)
@click.option("--hotkey-name", default=None)
@click.option("--dry-run", is_flag=True)
def miner(model_repo: str, revision: str, wallet_name, hotkey_name, dry_run: bool) -> None:
    """Submit a model commitment on-chain (after a precheck)."""
    from distil.cli.miner import run as miner_run

    sys.exit(miner_run(model_repo, revision, wallet_name, hotkey_name, dry_run=dry_run))


@main.command()
@click.option("--model-repo", required=True)
@click.option("--revision", default="")
def check(model_repo: str, revision: str) -> None:
    """Pre-submission validation: arch, vocab, size, integrity, hash."""
    from distil.cli.check import run as check_run

    sys.exit(check_run(model_repo, revision))


@main.command()
@click.option("--wallet-name", default=None)
@click.option("--hotkey-name", default=None)
@click.option("--once", is_flag=True, help="Run a single round, then exit.")
@click.option("--dry-run", is_flag=True, help="Skip set_weights() (shadow round).")
def validate(wallet_name, hotkey_name, once: bool, dry_run: bool) -> None:
    """Run the validator service loop."""
    from distil.eval.service import run as validate_run

    sys.exit(
        validate_run(
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            once=once,
            dry_run=dry_run,
        )
    )


@main.command(name="verify-kl")
@click.option("--model-repo", required=True)
@click.option("--n-prompts", default=100)
def verify_kl(model_repo: str, n_prompts: int) -> None:
    """One-shot KL parity check between HF logits and vLLM prompt_logprobs."""
    from distil.cli.verify_kl import run as verify_run

    sys.exit(verify_run(model_repo, n_prompts))


@main.command(name="migrate-state")
@click.option("--dry-run", is_flag=True)
def migrate_state(dry_run: bool) -> None:
    """Archive orphaned state shards to ``state/_legacy/`` (idempotent)."""
    from distil.state.migrate import run as migrate_run

    sys.exit(migrate_run(dry_run=dry_run))


@main.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def api(host, port) -> None:
    """Run the FastAPI dashboard backend (foreground)."""
    import uvicorn

    from distil.settings import settings

    uvicorn.run(
        "distil.api.server:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
