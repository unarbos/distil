#!/usr/bin/env python3
"""Composite-score parity harness — distil vs prod.

Feeds a real production `eval_results_merged.json` from a previous round
into `distil.eval.composite.compute_composite` and compares the resulting
per-student composite scores against the canonical
`state/composite_scores.json` written by `scripts.validator.results`.

Surfaces:
  * per-student composite diff,
  * the set of axes only one stack scores,
  * the shared-axis baseline.

This is the cutover gate referenced in `REWRITE_PLAN.md` and
`docs/CUTOVER_PARITY_2026-05-15.md`.

Usage:
    python scripts/parity_check.py
    python scripts/parity_check.py --round round_20260514T053302Z
    python scripts/parity_check.py --sample 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distil.eval.composite import compute_composite  # noqa: E402


def latest_round(state_dir: Path) -> Path:
    incoming = state_dir / "incoming"
    candidates = sorted(p for p in incoming.glob("round_*") if (p / "eval_results_merged.json").exists())
    if not candidates:
        raise SystemExit(f"No round_*/eval_results_merged.json under {incoming}")
    return candidates[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", default=str(REPO / "state"))
    p.add_argument("--round", default=None,
                   help="Round dir name under state/incoming/; defaults to latest")
    p.add_argument("--sample", type=int, default=5,
                   help="How many non-teacher students to sample")
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    round_dir = (state_dir / "incoming" / args.round) if args.round else latest_round(state_dir)
    merged = round_dir / "eval_results_merged.json"
    prod_scores = state_dir / "composite_scores.json"
    if not merged.exists():
        raise SystemExit(f"missing {merged}")
    if not prod_scores.exists():
        raise SystemExit(f"missing {prod_scores}")

    students = json.loads(merged.read_text())["students"]
    sample = []
    for name, row in students.items():
        if row.get("is_teacher", False):
            continue
        sample.append((name, row))
        if len(sample) >= args.sample:
            break

    prod = json.loads(prod_scores.read_text())
    by_model = {v.get("model"): v for v in prod.values() if isinstance(v, dict) and "model" in v}

    print(f"Round:   {round_dir.name}")
    print(f"Sampled: {len(sample)} non-teacher students of {len(students)} total")
    print()
    print(f"{'student':35s} {'distil_final':>14s} {'prod_final':>14s} {'abs_diff':>10s}  shared")
    print("-" * 100)

    total_diff = 0.0
    n_compared = 0
    for name, row in sample:
        try:
            d_comp = compute_composite(row, king_kl=None, king_rkl=None, broken_axes=set())
        except Exception as e:  # pragma: no cover
            print(f"{name[:34]:35s} ERROR: {e}")
            continue
        prod_rec = by_model.get(name)
        d_final = d_comp.get("final") or 0.0
        if not prod_rec:
            print(f"{name[:34]:35s} {d_final:>14.4f}                (no prod record)")
            continue
        p_final = prod_rec.get("final") or 0.0
        diff = abs(d_final - p_final)
        total_diff += diff
        n_compared += 1
        d_axes = {k for k, v in d_comp.get("axes", {}).items() if v is not None}
        p_axes = {k for k, v in (prod_rec.get("axes") or {}).items() if v is not None}
        print(f"{name[:34]:35s} {d_final:>14.4f} {p_final:>14.4f} {diff:>10.4f}  "
              f"{len(d_axes & p_axes):3d}/{len(d_axes)}-{len(p_axes)}")

    if n_compared:
        print()
        print(f"Mean |diff| across {n_compared} compared students: {total_diff / n_compared:.4f}")

    # Axis overlap detail
    if sample:
        name, row = sample[0]
        d_comp = compute_composite(row, king_kl=None, king_rkl=None, broken_axes=set())
        d_axes = {k for k, v in d_comp.get("axes", {}).items() if v is not None}
        prod_rec = by_model.get(name)
        if prod_rec:
            p_axes = {k for k, v in (prod_rec.get("axes") or {}).items() if v is not None}
            print()
            print(f"=== axis overlap for first sampled student ({name}) ===")
            print(f"  distil only ({len(d_axes - p_axes)}): {sorted(d_axes - p_axes)}")
            print(f"  prod only   ({len(p_axes - d_axes)}): {sorted(p_axes - d_axes)}")
            print(f"  shared      ({len(d_axes & p_axes)}): {sorted(d_axes & p_axes)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
