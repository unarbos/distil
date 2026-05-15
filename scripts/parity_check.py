#!/usr/bin/env python3
"""Composite-score parity harness — distil vs prod.

Feeds the same student rows from a prod `eval_results_merged.json` into
BOTH `scripts.validator.composite.compute_composite` (prod) and
`distil.eval.composite.compute_composite` (rewrite), and reports the
absolute diff on the `final` composite for each student.

This is the cutover gate referenced in `REWRITE_PLAN.md` and
`docs/CUTOVER_PARITY_2026-05-15.md`. The two stacks share the same
23-axis weighted set; the remaining diff comes from the
`reasoning_density` axis which mixes legacy bench data only prod
produces (harmless in steady-state — distil end-to-end never produces
legacy bench data).

Usage:
    python scripts/parity_check.py
    python scripts/parity_check.py --round round_20260514T053302Z
    python scripts/parity_check.py --sample 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distil.eval.composite import compute_composite  # noqa: E402


def latest_round(state_dir: Path) -> Path:  # noqa: D401
    """Pick the most recent round_*/ dir under state/incoming/."""
    incoming = state_dir / "incoming"
    candidates = sorted(
        p for p in incoming.glob("round_*")
        if (p / "eval_results_merged.json").exists()
    )
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
    if not merged.exists():
        raise SystemExit(f"missing {merged}")

    from scripts.validator.composite import compute_composite as prod_compute
    from distil.settings import settings

    weighted = {k: v for k, v in settings.axis_weights().items() if v > 0}
    print(f"=== weighted axes (distil): {len(weighted)} total, sum={sum(weighted.values()):.3f} ===")
    for k, v in sorted(weighted.items(), key=lambda x: -x[1]):
        print(f"  {k:35s} {v:.3f}")
    print()

    students = json.loads(merged.read_text())["students"]
    sample = []
    for name, row in students.items():
        if row.get("is_teacher", False):
            continue
        sample.append((name, row))
        if len(sample) >= args.sample:
            break

    print(f"Round:   {round_dir.name}")
    print(f"Sampled: {len(sample)} non-erroring students of {len(students)} total")
    print()
    print(f"{'student':40s} {'prod_final':>11s} {'distil_final':>13s} {'abs_diff':>10s}")
    print("-" * 80)

    total_diff = 0.0
    max_diff = 0.0
    n_compared = 0
    largest = None
    for name, row in sample:
        if row.get("error") or row.get("status") == "error" or "kl_global_avg" not in row:
            continue
        try:
            p_comp = prod_compute(row, king_kl=None, king_rkl=None)
            d_comp = compute_composite(row, king_kl=None, king_rkl=None, broken_axes=set())
        except Exception as e:  # pragma: no cover
            print(f"{name[:39]:40s} ERROR: {e}")
            continue
        p_final = p_comp.get("final") or 0.0
        d_final = d_comp.get("final") or 0.0
        diff = abs(d_final - p_final)
        total_diff += diff
        max_diff = max(max_diff, diff)
        n_compared += 1
        if largest is None or diff > largest[0]:
            largest = (diff, name, p_comp, d_comp)
        print(f"{name[:39]:40s} {p_final:>11.4f} {d_final:>13.4f} {diff:>10.4f}")

    if n_compared:
        print()
        print(f"n: {n_compared}    max |diff|: {max_diff:.4f}    mean |diff|: {total_diff / n_compared:.4f}")
        if largest and largest[0] > 0.001:
            _, name, pc, dc = largest
            print()
            print(f"=== detail for largest diff: {name} ({largest[0]:.4f}) ===")
            print(f"  prod   worst_3_mean={pc.get('worst_3_mean')} weighted={pc.get('weighted')} final={pc.get('final')}")
            print(f"  distil worst_3_mean={dc.get('worst_3_mean')} weighted={dc.get('weighted')} final={dc.get('final')}")
            pa = pc.get("axes") or {}
            da = dc.get("axes") or {}
            shared = {k for k, v in pa.items() if v is not None} & {k for k, v in da.items() if v is not None}
            mismatches = [(k, pa[k], da[k], abs((pa[k] or 0) - (da[k] or 0))) for k in shared
                          if abs((pa[k] or 0) - (da[k] or 0)) > 1e-3]
            mismatches.sort(key=lambda t: -t[3])
            if mismatches:
                print("  shared axes with diff > 0.001:")
                for k, pv, dv, _ in mismatches[:5]:
                    print(f"    {k:35s} prod={pv}  distil={dv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
