"""Miner commitments — fetch, parse, submit.

A commitment is a JSON blob ``{"model": "<repo_id>", "revision": "<sha?>"}``
written to chain via ``commit`` extrinsics. We keep only the latest
commitment per hotkey.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.chain.commitments")

_HF_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_REV_RE = re.compile(r"^[A-Za-z0-9._-]{0,64}$")


@dataclass(frozen=True)
class Commitment:
    """A miner's revealed model commitment."""

    uid: int
    hotkey: str
    block: int
    model: str
    revision: str = ""
    coldkey: str = ""

    @property
    def key(self) -> str:
        return f"{self.model}@{self.revision or 'latest'}"


def _is_valid_repo(model: str, revision: str) -> bool:
    if not isinstance(model, str) or not _HF_RE.match(model):
        return False
    return not (revision and not _REV_RE.match(revision))


def parse_commitments(
    metagraph,
    revealed: dict[str, list[tuple[int, str]]],
    n_uids: int | None = None,
) -> tuple[dict[int, Commitment], dict[int, str], dict[int, str]]:
    """Parse the chain ``get_all_revealed_commitments`` blob.

    Returns ``(by_uid, uid_to_hotkey, uid_to_coldkey)``.
    """
    by_uid: dict[int, Commitment] = {}
    uid_to_hotkey: dict[int, str] = {}
    uid_to_coldkey: dict[int, str] = {}
    n = n_uids or len(metagraph.hotkeys)

    for uid in range(n):
        hotkey = str(metagraph.hotkeys[uid])
        uid_to_hotkey[uid] = hotkey
        try:
            uid_to_coldkey[uid] = str(metagraph.coldkeys[uid])
        except Exception:
            pass
        rows = revealed.get(hotkey) or []
        if not rows:
            continue
        block, raw = max(rows, key=lambda x: x[0])
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        model = str(payload.get("model") or "").strip()
        revision = str(payload.get("revision") or "").strip()
        if not _is_valid_repo(model, revision):
            continue
        by_uid[uid] = Commitment(
            uid=uid,
            hotkey=hotkey,
            block=int(block),
            model=model,
            revision=revision,
            coldkey=uid_to_coldkey.get(uid, ""),
        )
    return by_uid, uid_to_hotkey, uid_to_coldkey


def fetch_revealed(subtensor, netuid: int | None = None) -> dict[str, list[tuple[int, str]]]:
    """Fetch every revealed commitment on the subnet."""
    netuid = netuid or settings.netuid
    return subtensor.get_all_revealed_commitments(netuid)


def commit_self(
    wallet,
    model: str,
    revision: str = "",
    *,
    netuid: int | None = None,
    subtensor=None,
    attempts: int = 3,
) -> tuple[bool, int, str]:
    """Submit ``{"model","revision"}`` as our commitment on chain.

    Returns ``(ok, block, message)``.
    """
    from distil.chain.metagraph import get_subtensor

    if not _is_valid_repo(model, revision):
        return False, 0, f"invalid model/revision: {model!r}@{revision!r}"

    netuid = netuid or settings.netuid
    sub = subtensor or get_subtensor()
    payload = json.dumps({"model": model, "revision": revision})

    last: str = ""
    for i in range(attempts):
        try:
            sub.commit(wallet=wallet, netuid=netuid, data=payload)
            block = sub.block
            return True, int(block), f"committed {model}@{revision or 'latest'} at block {block}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            logger.warning(f"commit attempt {i + 1}/{attempts} failed: {last}")
            if i + 1 < attempts:
                time.sleep(30)
    return False, 0, f"commit failed after {attempts} attempts: {last}"


def to_jsonable(c: Commitment) -> dict[str, Any]:
    return {
        "uid": c.uid,
        "hotkey": c.hotkey,
        "block": c.block,
        "model": c.model,
        "revision": c.revision,
        "coldkey": c.coldkey,
    }
