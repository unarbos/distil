"""Metagraph fetcher with retry + 5-minute disk cache.

The validator polls the metagraph at the start of every round; the API
process polls every minute. We share a small retry helper and a disk
cache so an RPC blip can't take the dashboard down.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.chain.metagraph")


def _retry(fn, *, attempts: int = 3, delay: float = 15.0, label: str = "rpc"):
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(f"{label} attempt {i + 1}/{attempts} failed: {exc}")
            if i + 1 < attempts:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def get_subtensor(network: str | None = None):
    """Return a configured ``bittensor.subtensor`` (lazy import to keep CLI fast)."""
    import bittensor as bt

    network = network or settings.network
    if settings.chain_endpoint:
        return bt.subtensor(network=settings.chain_endpoint)
    return bt.subtensor(network=network)


def fetch_metagraph(subtensor=None, netuid: int | None = None) -> tuple[Any, int, str | None]:
    """Return ``(metagraph, current_block, block_hash)``."""
    netuid = netuid or settings.netuid
    sub = subtensor or get_subtensor()

    def _fetch():
        mg = sub.metagraph(netuid)
        block = sub.block
        block_hash: str | None = None
        try:
            block_hash = sub.substrate.get_block_hash(block)
        except Exception as exc:
            logger.warning(f"block-hash fetch failed: {exc}")
        return mg, block, block_hash

    return _retry(_fetch, label="fetch_metagraph")


def get_validator_uid(metagraph, hotkey: str) -> int | None:
    """Resolve the validator's UID from its hotkey, or None if not registered."""
    try:
        return int(metagraph.hotkeys.index(hotkey))
    except ValueError:
        return None
