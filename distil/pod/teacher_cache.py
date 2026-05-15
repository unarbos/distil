"""Per-round teacher cache (improvement #2).

Phase 1 of the pod (teacher generation + sparse top-K logprobs) writes
``teacher_cache/<round_id>.json`` immediately after success. If the pod
restarts mid-round (Phase 2/3 crash, OOM, transient SSH drop) we skip
Phase 1 entirely as long as the cached ``block_hash`` matches the spec.

Format:
    {
        "round_id": int,
        "block_hash": str,
        "prompts": [str, ...],
        "teacher_continuations": [str, ...],
        "teacher_logprobs": [[{token_id: lp, ...}, ...], ...],
        "teacher_token_ids": [[int, ...], ...]
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("distil.pod.teacher_cache")

CACHE_DIR = Path("/home/teacher_cache")


def cache_path(round_id: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"round_{round_id}.json"


def load(round_id: int, *, expected_block_hash: str | None) -> dict | None:
    p = cache_path(round_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        logger.warning(f"teacher_cache load failed for round {round_id}: {exc}")
        return None
    if expected_block_hash and data.get("block_hash") != expected_block_hash:
        logger.info(
            f"teacher_cache block_hash mismatch (cached={data.get('block_hash')!r} "
            f"expected={expected_block_hash!r}); ignoring cache"
        )
        return None
    return data


def save(round_id: int, payload: dict) -> Path:
    p = cache_path(round_id)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, p)
    return p


def clear_old(keep: int = 4) -> None:
    """Drop all but the ``keep`` most-recent cache files."""
    if not CACHE_DIR.exists():
        return
    files = sorted(CACHE_DIR.glob("round_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
