"""Block-seeded prompt sampling.

Pulls deterministic prompts from ``karpathy/climbmix-400b-shuffle`` keyed
on ``(block_hash, n_prompts)`` so every validator on a round draws the
same items. Falls back to a local cache shard if HF is unreachable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from distil.settings import settings

logger = logging.getLogger("distil.eval.dataset")

_LOCAL_FALLBACK = Path(__file__).resolve().parents[2] / "state" / "_dataset_fallback.jsonl"


def _seed_for(block_hash: str | None) -> int:
    if not block_hash:
        return 0
    return int(hashlib.sha256(block_hash.encode()).hexdigest()[:12], 16)


def _read_local_fallback() -> list[str]:
    if not _LOCAL_FALLBACK.exists():
        return []
    try:
        out: list[str] = []
        for line in _LOCAL_FALLBACK.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") if isinstance(obj, dict) else line
            if isinstance(text, str) and text:
                out.append(text)
        return out
    except (OSError, ValueError) as exc:
        logger.warning(f"local-fallback dataset read failed: {exc}")
        return []


def sample_prompts(
    n: int = 256,
    *,
    block_hash: str | None = None,
    min_chars: int = 64,
    max_chars: int = 4096,
) -> list[str]:
    """Return ``n`` deterministic prompts seeded by ``block_hash``."""
    seed = _seed_for(block_hash)
    rng = random.Random(seed)

    try:
        from datasets import load_dataset

        ds = load_dataset(
            settings.eval_dataset,
            split=settings.eval_dataset_split,
            streaming=True,
            token=settings.hf_dl_token or None,
        )
        pool: list[str] = []
        for row in ds:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            if not (min_chars <= len(text) <= max_chars):
                continue
            pool.append(text)
            if len(pool) >= n * 4:
                break
        if pool:
            rng.shuffle(pool)
            return pool[:n]
    except Exception as exc:
        logger.warning(f"HF dataset load failed: {exc}; using local fallback")

    pool = _read_local_fallback()
    if not pool:
        return [f"sample prompt {i}" for i in range(n)]
    rng.shuffle(pool)
    return (pool * (n // len(pool) + 1))[:n]
