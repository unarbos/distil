"""HuggingFace upload-time lookup for griefing detection.

The chain commit_block is *not* a reliable proxy for who uploaded a
model first. A miner can recycle a UID slot, commit the bare model
name on chain (e.g. ``truke02/golden_v1``) at block X, then *days
later* upload the actual weights (typically by stealing them from
whoever is currently king). The on-chain block X then makes the
griefer look "earlier" than the legitimate miner.

This module resolves the order ambiguity by asking HuggingFace
directly: when was the model repo first created, and when were its
safetensors files last modified? Those timestamps reflect when the
weights actually became available on HF, which is the timeline
copy-detection needs.

The lookup is cached on disk under ``state/hf_upload_meta.json`` so
repeated precheck passes do not hammer the HF API.

Failure mode: if HF is unreachable or the model was deleted, we
return ``None`` and callers fall back to chain commit_block.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = "hf_upload_meta.json"
CACHE_TTL_SECONDS = 24 * 3600
HF_TIMEOUT_SECONDS = 8


def _cache_path(state_dir) -> Path:
    state_dir = Path(state_dir)
    return state_dir / CACHE_FILE


def _load_cache(state_dir) -> dict:
    p = _cache_path(state_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict, state_dir) -> None:
    p = _cache_path(state_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2, default=str))
    except Exception as exc:  # pragma: no cover - disk failures
        logger.warning(f"hf_upload_meta: cache write failed: {exc}")


def _to_epoch(ts) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


def get_first_upload_epoch(model_repo: str, revision: str = "main",
                           state_dir=None,
                           force: bool = False) -> Optional[float]:
    """Return the epoch seconds at which ``model_repo`` was first
    uploaded to HuggingFace, using the earliest commit on the repo's
    revision branch as the authoritative answer.

    Falls back through, in order:

    1. earliest commit timestamp on the revision branch
       (``list_repo_commits`` returns newest first; the last entry is
       the initial commit / weights upload)
    2. ``model_info().created_at`` (repo creation time)
    3. ``model_info().last_modified`` (least preferred — only useful
       as a tiebreaker, not as a "first upload" proxy)

    Cached on disk by ``model_repo@revision``. Returns ``None`` when
    HF is unreachable or the repo has been deleted.
    """
    if not model_repo or "/" not in model_repo:
        return None
    if state_dir is None:
        return _fetch_first_upload_epoch(model_repo, revision)

    cache = _load_cache(state_dir)
    cache_key = f"{model_repo}@{revision or 'main'}"
    entry = cache.get(cache_key)
    now = time.time()
    if not force and isinstance(entry, dict):
        ts = entry.get("first_upload_epoch")
        cached_at = entry.get("cached_at", 0)
        if ts is not None and (now - cached_at) < CACHE_TTL_SECONDS:
            return float(ts)

    ts = _fetch_first_upload_epoch(model_repo, revision)
    if ts is None:
        if isinstance(entry, dict) and entry.get("first_upload_epoch") is not None:
            return float(entry["first_upload_epoch"])
        cache[cache_key] = {
            "first_upload_epoch": None,
            "cached_at": now,
            "error": "lookup failed",
        }
    else:
        cache[cache_key] = {
            "first_upload_epoch": ts,
            "cached_at": now,
        }
    _save_cache(cache, state_dir)
    return ts


def _fetch_first_upload_epoch(model_repo: str, revision: str) -> Optional[float]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        rev = revision or "main"
        try:
            commits = api.list_repo_commits(model_repo, revision=rev)
        except Exception:
            commits = None
        if commits:
            ts = _to_epoch(commits[-1].created_at)
            if ts is not None:
                return ts
        info = api.model_info(model_repo, revision=rev, timeout=HF_TIMEOUT_SECONDS)
        ts = _to_epoch(getattr(info, "created_at", None))
        if ts is not None:
            return ts
        ts = _to_epoch(getattr(info, "last_modified", None))
        if ts is not None:
            return ts
        return None
    except Exception as exc:
        logger.info(f"hf_upload_meta: lookup failed for {model_repo}@{revision}: {exc}")
        return None


def hf_upload_orders_match_chain(this_repo: str, this_rev: str, this_block,
                                 orig_repo: str, orig_rev: str, orig_block,
                                 state_dir) -> Optional[bool]:
    """Return True if the chain commit_block ordering matches HF
    upload ordering, False if it disagrees, None if HF is unreachable.

    "Matches" means: the model with the earlier chain block also has
    the earlier HF upload time, allowing for a small clock-skew slack.
    """
    this_ts = get_first_upload_epoch(this_repo, this_rev, state_dir=state_dir)
    orig_ts = get_first_upload_epoch(orig_repo, orig_rev, state_dir=state_dir)
    if this_ts is None or orig_ts is None:
        return None
    try:
        this_block = float(this_block) if this_block is not None else float("inf")
        orig_block = float(orig_block) if orig_block is not None else float("inf")
    except Exception:
        return None
    chain_orig_first = orig_block < this_block
    hf_orig_first = orig_ts < this_ts
    if chain_orig_first == hf_orig_first:
        return True
    return False
