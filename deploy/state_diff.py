#!/usr/bin/env python3
"""Diff two state directories — used during cutover step 7.

Prints a concise per-shard delta summary so the operator can confirm
that the v2 validator is writing semantically equivalent state to the
legacy one before cutting traffic over.

Usage:
    python3 deploy/state_diff.py state state-shadow

Each diff line: ``shard | match | live_size | shadow_size | first_delta``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LIVE_SHARDS = (
    "scores.json",
    "composite_scores.json",
    "disqualified.json",
    "uid_hotkey_map.json",
    "h2h_latest.json",
    "top4_leaderboard.json",
    "recent_kings.json",
    "model_hashes.json",
    "evaluated_uids.json",
    "evaluated_hotkeys.json",
    "eval_progress.json",
)


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _first_diff(a, b, path: str = "") -> str | None:
    if type(a) != type(b):
        return f"{path} type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            extra = (ka - kb) or (kb - ka)
            return f"{path} keys differ ({sorted(extra)[:3]})"
        for k in sorted(a):
            d = _first_diff(a[k], b[k], path + "." + str(k))
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path} len {len(a)} != {len(b)}"
        for i in range(min(len(a), len(b))):
            d = _first_diff(a[i], b[i], path + f"[{i}]")
            if d:
                return d
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 5e-3:
            return f"{path} {a:.4f} != {b:.4f}"
    elif a != b:
        return f"{path} {a!r} != {b!r}"
    return None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("usage: state_diff.py LIVE_DIR SHADOW_DIR", file=sys.stderr)
        return 2
    live = Path(argv[0])
    shadow = Path(argv[1])
    print(f"diff {live} vs {shadow}\n")
    print(f"{'shard':32} {'match':5} {'live':>10} {'shadow':>10}  first_delta")
    print("-" * 110)
    n_match = n_total = 0
    for name in LIVE_SHARDS:
        a = _load(live / name)
        b = _load(shadow / name)
        if a is None and b is None:
            continue
        n_total += 1
        delta = _first_diff(a, b)
        match = "yes" if delta is None else "no"
        if delta is None:
            n_match += 1
        live_sz = (live / name).stat().st_size if (live / name).exists() else 0
        shadow_sz = (shadow / name).stat().st_size if (shadow / name).exists() else 0
        print(
            f"{name:32} {match:5} {live_sz:>10} {shadow_sz:>10}  {delta or ''}"
        )
    print(f"\n{n_match}/{n_total} shards match within tolerance.")
    return 0 if n_match == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
