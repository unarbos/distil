"""Per-axis correlation between validator score and held-out canary.

Why this exists. The validator is a faithful proxy for SOTA capability
ONLY if its per-axis scores correlate with the corresponding held-out
benchmark. The 2026-04-28 audit showed kings climbing the validator's
procedural axes while regressing on held-out evalscope canary — direct
evidence the proxy was diverging from the target.

This script computes the rolling correlation between each validator
axis and its held-out counterpart, joins by king UID, and writes a
summary the dashboard can surface in real time:

    state/h2h_history.json           composite axes per king per round
    state/composite_scores.json      latest composite per UID
    state/benchmarks/uid_*.json      evalscope held-out per UID
                                     (gsm8k, humaneval, bbh, ifeval, arc, mmlu_pro)
        ↓
    state/axis_correlation.json      {axis: pearson_r, n, last_updated}

Pairs we track (validator → held-out):

    math_bench       ↔ gsm8k
    code_bench       ↔ humaneval
    reasoning_bench  ↔ bbh
    ifeval_bench     ↔ ifeval
    mbpp_bench       ↔ humaneval     (closest analog; mbpp not in canary)
    aime_bench       ↔ gsm8k         (closest analog; aime not in canary)

A weak or NEGATIVE correlation means optimising the validator axis is
NOT translating to held-out skill — that's a Goodhart signal and the
axis needs design work (templates too narrow, items too easy / too
hard, distribution mismatch with the held-out test).

A strong positive correlation means the axis is doing its job: a model
that climbs it is genuinely getting better on the corresponding
real-world skill.

Run cadence
-----------
Designed to run quickly (~1 second on 50 rounds) so it's safe to call
from the dashboard refresh loop or from a post-round hook. Standalone
script so we can also run it ad-hoc:

    python scripts/audit/axis_correlation.py

Outputs both a human-readable table to stdout and the JSON payload at
``state/axis_correlation.json`` for the dashboard.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
H2H_HISTORY = REPO_ROOT / "state" / "h2h_history.json"
BENCH_DIR = REPO_ROOT / "state" / "benchmarks"
OUT_PATH = REPO_ROOT / "state" / "axis_correlation.json"

# Validator axis ↔ held-out canary axis mapping. Each value is (held_out_key,
# rationale). Add entries here when new bench axes have a clear evalscope
# counterpart.
AXIS_PAIRS: dict[str, tuple[str, str]] = {
    "math_bench":      ("gsm8k", "GSM8K-narrative-style multi-step word problems"),
    "code_bench":      ("humaneval", "HumanEval-difficulty algorithm completion"),
    "reasoning_bench": ("bbh", "BBH-style multi-step deduction"),
    "ifeval_bench":    ("ifeval", "instruction-following with structural constraints"),
    "mbpp_bench":      ("humaneval", "closest held-out analog (no MBPP in canary)"),
    "aime_bench":      ("gsm8k", "closest held-out analog (no AIME in canary)"),
    # debug_bench (v29.2) doesn't have a clean evalscope analog; tracked
    # manually via post-round king benchmark runs. Add when a held-out
    # debugging benchmark is wired into the canary set.
}


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Pearson r between two equal-length value lists. Returns (r, n).
    Returns (0.0, n) if stdev is zero on either side or n < 3.
    """
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0, n
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) / n
    sy = sum((y - my) ** 2 for y in ys) / n
    if sx <= 0 or sy <= 0:
        return 0.0, n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    r = cov / (math.sqrt(sx) * math.sqrt(sy))
    return float(r), n


def _load_h2h_history() -> list[dict]:
    if not H2H_HISTORY.exists():
        return []
    try:
        return json.loads(H2H_HISTORY.read_text())
    except Exception:
        return []


def _load_held_out_per_uid() -> dict[int, dict[str, Any]]:
    """Returns {uid_int: {"benchmarks": {...}, "counts": {...}, "ts": ...}}."""
    out: dict[int, dict[str, Any]] = {}
    if not BENCH_DIR.exists():
        return out
    for path in BENCH_DIR.glob("uid_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        uid = data.get("uid")
        if uid is None:
            continue
        try:
            uid = int(uid)
        except (TypeError, ValueError):
            continue
        out[uid] = {
            "benchmarks": data.get("benchmarks") or {},
            "counts": data.get("counts") or {},
            "ts": data.get("timestamp", ""),
            "model": data.get("model", ""),
            "kl": data.get("kl"),
        }
    return out


def _per_uid_validator_axes(history: list[dict]) -> dict[int, dict[str, float]]:
    """Take the most recent (King, axis) value for each UID across all
    rounds where that UID appeared as king. Falls back to its most
    recent challenger row if it never appeared as king (so a UID
    that's been a top-N challenger but never crowned still gets
    paired data when it has a canary file).
    """
    by_uid_king: dict[int, dict[str, float]] = {}
    by_uid_any: dict[int, dict[str, float]] = {}
    for round_data in history:
        results = round_data.get("results") or []
        for r in results:
            uid = r.get("uid")
            if uid is None or uid == -1:
                continue
            try:
                uid_i = int(uid)
            except (TypeError, ValueError):
                continue
            comp = r.get("composite") or {}
            axes = comp.get("axes") or {}
            if not axes:
                continue
            by_uid_any[uid_i] = {k: v for k, v in axes.items() if v is not None}
            if r.get("is_king"):
                by_uid_king[uid_i] = {k: v for k, v in axes.items() if v is not None}
    # Prefer king rows; fall back to any row.
    return {uid: by_uid_king.get(uid, by_uid_any.get(uid)) for uid in by_uid_any}


def _z_to_p(z: float) -> float:
    """Two-sided p value from a standard-normal z. Approximation good
    enough for telemetry (no scipy)."""
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def compute_correlations() -> dict[str, Any]:
    history = _load_h2h_history()
    held_out = _load_held_out_per_uid()
    val_axes = _per_uid_validator_axes(history)
    # Join by UID — only UIDs with BOTH a validator axis row AND a
    # held-out canary file count toward the correlation.
    pairs: dict[str, list[tuple[float, float, int]]] = {a: [] for a in AXIS_PAIRS}
    for uid, val_row in val_axes.items():
        if val_row is None:
            continue
        ho = held_out.get(uid)
        if not ho:
            continue
        ho_bench = ho["benchmarks"]
        ho_counts = ho.get("counts") or {}
        for val_axis, (ho_axis, _why) in AXIS_PAIRS.items():
            v = val_row.get(val_axis)
            h = ho_bench.get(ho_axis)
            if v is None or h is None:
                continue
            # Skip held-out values where count == 0 (axis didn't run
            # this round; values are noise).
            cnt = ho_counts.get(ho_axis)
            if cnt is not None and cnt == 0:
                continue
            pairs[val_axis].append((float(v), float(h), uid))
    summary: dict[str, Any] = {}
    for val_axis, items in pairs.items():
        if not items:
            summary[val_axis] = {
                "held_out_axis": AXIS_PAIRS[val_axis][0],
                "r": None,
                "n": 0,
                "interpretation": "no paired data yet",
            }
            continue
        xs = [it[0] for it in items]
        ys = [it[1] for it in items]
        r, n = _pearson(xs, ys)
        # 95 % CI on Fisher z (only when n >= 4).
        if n >= 4 and abs(r) < 0.999:
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1 / math.sqrt(n - 3)
            z_lo = z - 1.96 * se
            z_hi = z + 1.96 * se
            r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
            r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
        else:
            r_lo, r_hi = -1.0, 1.0
        # Interpretation flag.
        if n < 4:
            interp = f"insufficient data (n={n}, need >= 4)"
        elif r >= 0.6:
            interp = "strong positive — axis is a faithful proxy"
        elif r >= 0.3:
            interp = "moderate positive — axis tracks held-out somewhat"
        elif r >= 0.0:
            interp = "weak positive — needs work, not a Goodhart signal yet"
        elif r >= -0.3:
            interp = "weak negative — DRIFTING; revisit calibration"
        else:
            interp = "STRONG NEGATIVE — Goodhart, validator improvements ANTI-CORRELATE with held-out skill"
        summary[val_axis] = {
            "held_out_axis": AXIS_PAIRS[val_axis][0],
            "rationale": AXIS_PAIRS[val_axis][1],
            "r": round(r, 4),
            "ci_lo": round(r_lo, 4),
            "ci_hi": round(r_hi, 4),
            "n": n,
            "validator_mean": round(sum(xs) / n, 4) if n else None,
            "held_out_mean": round(sum(ys) / n, 4) if n else None,
            "validator_std": round(statistics.pstdev(xs), 4) if n >= 2 else None,
            "held_out_std": round(statistics.pstdev(ys), 4) if n >= 2 else None,
            "uids": [it[2] for it in items],
            "interpretation": interp,
        }
    return {
        "computed_at": time.time(),
        "computed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_kings_with_canary": sum(
            1 for uid in val_axes if uid in held_out
        ),
        "axis_pairs": summary,
    }


def main() -> int:
    payload = compute_correlations()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    # Human-readable summary.
    print(f"Per-axis validator↔held-out correlation (n_kings={payload['n_kings_with_canary']})")
    print(f"computed_at: {payload['computed_at_iso']}")
    print()
    print(f"  {'axis':<22} {'held-out':<12} {'r':>7} {'95% CI':>16} {'n':>4}  interpretation")
    print(f"  {'-'*22} {'-'*12} {'-'*7} {'-'*16} {'-'*4}  {'-'*30}")
    for val_axis, info in payload["axis_pairs"].items():
        r = info["r"]
        n = info["n"]
        if r is None:
            print(f"  {val_axis:<22} {info['held_out_axis']:<12} {'—':>7} {'—':>16} {n:>4}  {info['interpretation']}")
            continue
        ci = f"[{info['ci_lo']:+.2f}, {info['ci_hi']:+.2f}]"
        print(
            f"  {val_axis:<22} {info['held_out_axis']:<12} {r:>+7.3f} {ci:>16} {n:>4}  {info['interpretation']}"
        )
    print()
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
