"""Pod-side progress + per-bench telemetry (improvement #5).

The pod writes ``eval_progress.json`` on every phase change; the validator
host rsyncs the file every 30s so the dashboard's live tab stays current.
After each axis runs, ``record_bench_timing`` appends a row with wall-time
and tokens/sec — the dashboard's bench-strip component plots these.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def write_progress(path: Path, **fields: Any) -> None:
    """Merge ``fields`` into ``eval_progress.json`` (atomic)."""
    state: dict[str, Any] = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
    state.update(fields)
    state["updated_at"] = time.time()
    _atomic_write(path, state)


def record_bench_timing(
    path: Path,
    *,
    name: str,
    wall_s: float,
    n_prompts: int,
    completion_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append ``{name, wall_s, prompts, tokens, tok_per_s}`` to ``phase_timings``."""
    row: dict[str, Any] = {
        "name": name,
        "wall_s": round(float(wall_s), 3),
        "prompts": int(n_prompts),
        "completion_tokens": int(completion_tokens),
        "tok_per_s": round(int(completion_tokens) / max(wall_s, 1e-6), 1),
        "ts": time.time(),
    }
    if extra:
        row.update(extra)
    state: dict[str, Any] = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
    timings = state.get("phase_timings") or []
    timings.append(row)
    state["phase_timings"] = timings[-200:]
    state["updated_at"] = time.time()
    _atomic_write(path, state)
