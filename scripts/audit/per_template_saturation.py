"""Per-template saturation audit (v29.3, 2026-04-29).

Surfaces which procedural templates inside each bench axis have hit
ceiling (everyone passes — no signal at the top) or floor (everyone
fails — no signal at the bottom). The composite already gives us
per-axis pass-rate; this script drills one level deeper to attribute
pass-rate to specific template kinds (``shopping_budget`` vs
``recipe_scale`` inside ``math_bench``, ``coin_change_min`` vs
``transform_list`` inside ``code_bench``, etc.).

Why this matters
----------------
Each procedural axis is a mix of templates. If 4 of 12 templates
saturate at 1.0 across all models, those 4 templates contribute zero
discrimination (33 % of the axis's items waste budget). Conversely
if 2 templates always score 0, they're either too hard or buggy
(eval setup, not skill). This audit lets us prune saturated templates,
buff dead ones, or rebalance kinds-per-round to maximise per-axis
signal — operational equivalent to "tighten the eval" without
eyeballing thousands of items by hand.

Reads
-----
``state/h2h_history.json``                 v29.3+ rounds carry
                                           ``composite.per_src``.
                                           Older rounds without it
                                           are silently skipped.

Writes
------
``state/per_template_saturation.json``     summary keyed by
                                           ``axis.template`` with
                                           per-template pass_frac,
                                           variance, n_models seen,
                                           saturation flag.

Run cadence: ad-hoc (no cost overhead). Cheap (~1 second on 50 rounds).
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
H2H_HISTORY = REPO_ROOT / "state" / "h2h_history.json"
OUT_PATH = REPO_ROOT / "state" / "per_template_saturation.json"

# How many recent rounds to weigh (bias toward current eval calibration).
WINDOW_ROUNDS = 30

# Saturation thresholds.
SAT_HIGH = 0.95   # template's mean pass_frac across models hits this → ceiling
SAT_LOW  = 0.05   # template's mean pass_frac across models drops below → floor
MIN_OBS  = 6      # min n_observations before we trust the per-template stats


def _load_history() -> list[dict]:
    if not H2H_HISTORY.exists():
        return []
    try:
        return json.loads(H2H_HISTORY.read_text())
    except Exception:
        return []


def collect() -> dict[str, Any]:
    hist = _load_history()
    if not hist:
        return {"axes": {}, "n_rounds": 0, "computed_at": time.time()}
    recent = hist[-WINDOW_ROUNDS:]
    # axis -> template -> list[pass_frac]
    pass_fracs: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_rounds_with_data = 0
    for round_data in recent:
        results = round_data.get("results") or []
        round_had_data = False
        for entry in results:
            comp = entry.get("composite") or {}
            per_src = comp.get("per_src")
            if not isinstance(per_src, dict):
                continue
            round_had_data = True
            for axis, src_map in per_src.items():
                if not isinstance(src_map, dict):
                    continue
                for src, info in src_map.items():
                    if not isinstance(info, dict):
                        continue
                    n = info.get("n", 0)
                    if not n:
                        continue
                    correct = info.get("correct", 0)
                    pf = correct / n if n else 0.0
                    pass_fracs[axis][src].append(float(pf))
        if round_had_data:
            n_rounds_with_data += 1
    # Aggregate.
    axes_summary: dict[str, dict[str, Any]] = {}
    for axis, src_map in pass_fracs.items():
        templates: list[dict[str, Any]] = []
        for src, vals in sorted(src_map.items()):
            if len(vals) < MIN_OBS:
                templates.append({
                    "template": src, "n_obs": len(vals),
                    "mean_pf": round(statistics.mean(vals), 4) if vals else 0.0,
                    "stdev": None,
                    "saturation": "insufficient_data",
                })
                continue
            mean_pf = statistics.mean(vals)
            stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            if mean_pf >= SAT_HIGH:
                sat = "saturated_high"  # ceiling: no signal at the top
            elif mean_pf <= SAT_LOW:
                sat = "saturated_low"    # floor: no signal at the bottom
            elif stdev <= 0.05:
                sat = "low_variance"     # bunched up, weak signal
            else:
                sat = "discriminating"   # healthy
            templates.append({
                "template": src,
                "n_obs": len(vals),
                "mean_pf": round(mean_pf, 4),
                "stdev": round(stdev, 4),
                "saturation": sat,
            })
        # Per-axis summary.
        n_disc = sum(1 for t in templates if t["saturation"] == "discriminating")
        n_sat = sum(1 for t in templates if t["saturation"] == "saturated_high")
        n_floor = sum(1 for t in templates if t["saturation"] == "saturated_low")
        axes_summary[axis] = {
            "n_templates": len(templates),
            "n_discriminating": n_disc,
            "n_saturated_high": n_sat,
            "n_saturated_low": n_floor,
            "discrimination_ratio": round(
                n_disc / max(1, len(templates)), 4
            ),
            "templates": sorted(templates, key=lambda t: t["mean_pf"]),
        }
    return {
        "computed_at": time.time(),
        "computed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rounds_window": len(recent),
        "n_rounds_with_data": n_rounds_with_data,
        "saturation_high_threshold": SAT_HIGH,
        "saturation_low_threshold": SAT_LOW,
        "min_obs": MIN_OBS,
        "axes": axes_summary,
    }


def main() -> int:
    payload = collect()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    # Pretty print summary.
    print(f"Per-template saturation audit  (rounds={payload['n_rounds_window']}, with-data={payload['n_rounds_with_data']})")
    print(f"computed_at: {payload['computed_at_iso']}")
    print()
    if not payload["axes"]:
        print("No per-template data available yet. Persist a few rounds with v29.3 (composite.per_src) and re-run.")
        return 0
    print(f"  {'axis':<22} {'tmpl':>4} {'disc':>4} {'sat':>4} {'dead':>4} {'disc_ratio':>10}")
    print(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*10}")
    for axis, info in sorted(payload["axes"].items()):
        print(
            f"  {axis:<22} "
            f"{info['n_templates']:>4} "
            f"{info['n_discriminating']:>4} "
            f"{info['n_saturated_high']:>4} "
            f"{info['n_saturated_low']:>4} "
            f"{info['discrimination_ratio']:>10.2%}"
        )
    print()
    # Per-axis worst templates (saturated / dead) with their stats.
    for axis, info in sorted(payload["axes"].items()):
        bads = [t for t in info["templates"]
                if t["saturation"] in ("saturated_high", "saturated_low", "low_variance")
                and t["n_obs"] >= MIN_OBS]
        if not bads:
            continue
        print(f"== {axis}: {len(bads)} non-discriminating templates ==")
        for t in bads:
            print(
                f"   {t['template']:<40} mean={t['mean_pf']:>5.2f} "
                f"stdev={t['stdev']:>5.2f} n={t['n_obs']:>3}  ⇒ {t['saturation']}"
            )
        print()
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
