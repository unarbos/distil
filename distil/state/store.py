"""Read-side state store with mtime-invalidated caching.

The API process only reads state files (the validator owns writes), so we
can cache aggressively. Each file is reloaded only when its mtime changes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import RLock
from typing import Any

from distil.settings import settings
from distil.state.files import (
    ANNOUNCEMENT_FILE,
    CHAT_POD_FILE,
    COMPOSITE_SCORES_FILE,
    CURRENT_ROUND_FILE,
    DISQUALIFIED_FILE,
    EVAL_BACKLOG_FILE,
    EVAL_PROGRESS_FILE,
    H2H_HISTORY_FILE,
    H2H_LATEST_FILE,
    INCIDENTS_FILE,
    LAST_EVAL_FILE,
    MODEL_HASHES_FILE,
    SCORES_FILE,
    TOP4_LEADERBOARD_FILE,
    UID_HOTKEY_MAP_FILE,
    VALIDATOR_LOG_FILE,
    safe_json_load,
)


class StateStore:
    """Mtime-invalidated reader cache for ``state/*.json``."""

    def __init__(self, state_dir: Path | None = None):
        self.state_dir = Path(state_dir or settings.state_dir)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = RLock()

    def _read(self, name: str, default: Any) -> Any:
        path = self.state_dir / name
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return default
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached[0] == mtime:
                return cached[1]
            data = safe_json_load(path, default)
            self._cache[name] = (mtime, data)
            return data

    def read_jsonl(self, name: str, tail: int = 200) -> list[dict]:
        path = self.state_dir / name
        if not path.exists():
            return []
        try:
            lines = path.read_text().splitlines()[-tail:]
        except OSError:
            return []
        out: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ── Public read API (one method per consumer-facing shard) ─────────

    def scores(self) -> dict:
        return self._read(SCORES_FILE, {})

    def disqualified(self) -> dict:
        return self._read(DISQUALIFIED_FILE, {})

    def composite_scores(self) -> dict:
        return self._read(COMPOSITE_SCORES_FILE, {})

    def h2h_latest(self) -> dict:
        return self._read(H2H_LATEST_FILE, {})

    def h2h_history(self) -> list[dict]:
        return self._read(H2H_HISTORY_FILE, [])

    def uid_hotkey_map(self) -> dict:
        return self._read(UID_HOTKEY_MAP_FILE, {})

    def top4_leaderboard(self) -> dict:
        return self._read(TOP4_LEADERBOARD_FILE, {})

    def eval_progress(self) -> dict:
        return self._read(EVAL_PROGRESS_FILE, {})

    def current_round(self) -> dict:
        return self._read(CURRENT_ROUND_FILE, {})

    def model_hashes(self) -> dict:
        return self._read(MODEL_HASHES_FILE, {})

    def announcement(self) -> dict:
        return self._read(ANNOUNCEMENT_FILE, {})

    def eval_backlog(self) -> dict:
        return self._read(EVAL_BACKLOG_FILE, {})

    def chat_pod(self) -> dict:
        return self._read(CHAT_POD_FILE, {})

    def last_eval(self) -> dict:
        return self._read(LAST_EVAL_FILE, {})

    def validator_log(self, tail: int = 200) -> list[dict]:
        rows = self._read(VALIDATOR_LOG_FILE, [])
        return rows[-tail:] if isinstance(rows, list) else []

    def incidents(self, tail: int = 200) -> list[dict]:
        return self.read_jsonl(INCIDENTS_FILE, tail=tail)

    # ── Disk cache (cross-process for chain/api fetches) ──────────────

    def cache_path(self, name: str) -> Path:
        d = settings.cache_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.json"

    def read_cache(self, name: str, ttl: int = 60) -> Any:
        p = self.cache_path(name)
        if not p.exists():
            return None
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return None
        if time.time() - mtime > ttl:
            return None
        return safe_json_load(p, None)

    def write_cache(self, name: str, data: Any) -> None:
        from distil.state.files import atomic_json_write

        atomic_json_write(self.cache_path(name), data)

    def invalidate(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)


# Process-wide singleton; tests may instantiate their own.
store = StateStore()
