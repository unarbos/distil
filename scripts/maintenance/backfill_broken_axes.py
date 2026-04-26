#!/usr/bin/env python3
"""Backfill ``broken_axes`` + corrected ``worst`` on legacy composite records.

Why this exists
---------------
Before commit ed8e2f9 (2026-04-26), ``merge_composite_scores`` did not
persist ``broken_axes`` to ``state/composite_scores.json``. As a result,
~92% of current-schema records (n_axes >= 17) sit at ``worst == 0.0``
because a single eval-setup-broken axis (typically aime_bench or
mbpp_bench, which the reference base model itself can't pass under the
current token cap) floors ``min(axes)``. This makes king selection
collapse to a weighted-tiebreaker for almost every comparison and
neutralizes the dethrone gate for legacy records.

Going forward the merge_composite_scores fix populates broken_axes
correctly, but legacy records will *never* be re-evaluated under the
SINGLE_EVAL_MODE policy ("one commitment, one eval"). They need an
in-place backfill or they become permanent zombies on the leaderboard.

Backfill strategy (matches production filter)
---------------------------------------------
For each legacy record without broken_axes:
    1. Find the most recent round (in h2h_history.json) that contained
       this UID + the reference UID (-1).
    2. Compute the reference's broken bench axes for THAT round (axes
       where ref pass_frac == 0). This matches
       ``composite.resolve_reference_broken_axes`` exactly.
    3. Set the record's broken_axes to that set.
    4. Recompute ``worst`` as min over (axes - broken_axes), matching
       the asymmetric filter in ``compute_composite``. ``weighted`` is
       NOT recomputed: legacy weighted already includes broken axes (no
       change in behavior), and we don't want to change rank ordering
       beyond what the bug fix would have produced anyway.
    5. Write atomically with a timestamped backup.

Records skipped:
    - already have broken_axes set (no-op)
    - UID not in any historical round (orphan)
    - UID's most-recent round had no reference row
    - UID's most-recent round had zero broken bench axes
    - axes - broken_axes is empty after filter

Run as:
    sudo -u distil python3 /opt/distil/repo/scripts/maintenance/backfill_broken_axes.py
    sudo -u distil python3 /opt/distil/repo/scripts/maintenance/backfill_broken_axes.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

REPO_STATE = "/opt/distil/repo/state"
COMPOSITE_PATH = os.path.join(REPO_STATE, "composite_scores.json")
HISTORY_PATH = os.path.join(REPO_STATE, "h2h_history.json")
REFERENCE_UID = -1
REFERENCE_BROKEN_BENCH_FLOOR = 0.0
BENCH_AXES_FILTERED = {
    "aime_bench", "mbpp_bench", "code_bench", "math_bench",
    "knowledge_bench", "reasoning_bench", "tool_use_bench",
    "robustness_bench", "noise_resistance_bench", "ifeval_bench",
    "self_consistency_bench", "arc_bench", "truthful_bench",
    "long_context_bench", "procedural_bench",
}


def _broken_for_round_results(results: list[dict]) -> set[str]:
    """Replicate ``resolve_reference_broken_axes`` for a stored round."""
    ref_row = next((r for r in results if r.get("uid") == REFERENCE_UID), None)
    if not ref_row:
        return set()
    composite = ref_row.get("composite") or {}
    axes = composite.get("axes") or {}
    broken: set[str] = set()
    for axis in BENCH_AXES_FILTERED:
        v = axes.get(axis)
        if v is None:
            continue
        try:
            if float(v) <= REFERENCE_BROKEN_BENCH_FLOOR:
                broken.add(axis)
        except (TypeError, ValueError):
            continue
    return broken


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing.",
    )
    args = parser.parse_args()

    with open(COMPOSITE_PATH) as f:
        composite_scores = json.load(f)
    with open(HISTORY_PATH) as f:
        history = json.load(f)

    # Build UID -> (block, broken_axes). Use the most recent round each UID
    # appears in. Composite records aren't time-keyed precisely, but the
    # most-recent appearance is the closest match to the reference state at
    # eval time.
    uid_to_round: dict[int, tuple[int, set[str]]] = {}
    for entry in history:
        block = entry.get("block")
        if not isinstance(block, int):
            continue
        results = entry.get("results") or []
        broken = _broken_for_round_results(results)
        for r in results:
            uid = r.get("uid")
            if not isinstance(uid, int) or uid == REFERENCE_UID:
                continue
            prev = uid_to_round.get(uid)
            if prev is None or block > prev[0]:
                uid_to_round[uid] = (block, broken)

    # Walk composite records and produce updates.
    updates: list[dict] = []
    skipped_no_round = 0
    skipped_already_set = 0
    skipped_no_broken = 0
    skipped_empty_after_filter = 0
    for uid_str, rec in composite_scores.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if rec.get("broken_axes"):
            skipped_already_set += 1
            continue
        if uid not in uid_to_round:
            skipped_no_round += 1
            continue
        _, broken_for_uid = uid_to_round[uid]
        if not broken_for_uid:
            skipped_no_broken += 1
            continue
        axes = rec.get("axes") or {}
        remaining = {
            k: v for k, v in axes.items()
            if v is not None and k not in broken_for_uid
        }
        if not remaining:
            skipped_empty_after_filter += 1
            continue
        new_worst = float(min(remaining.values()))
        updates.append({
            "uid": uid,
            "old_worst": rec.get("worst"),
            "new_worst": new_worst,
            "broken_axes": sorted(broken_for_uid),
            "n_remaining": len(remaining),
        })

    print(f"Loaded {len(composite_scores)} composite records, "
          f"{len(history)} h2h history entries.")
    print(f"  - already have broken_axes: {skipped_already_set}")
    print(f"  - no historical round for uid: {skipped_no_round}")
    print(f"  - no broken axes in their last round: {skipped_no_broken}")
    print(f"  - empty after filter (won't backfill): "
          f"{skipped_empty_after_filter}")
    print(f"  -> would update {len(updates)} records")
    print()
    print("Sample updates (first 10):")
    for u in updates[:10]:
        print(f"  uid {u['uid']}: worst {u['old_worst']!r} -> "
              f"{u['new_worst']:.3f}, "
              f"dropping {u['broken_axes']} "
              f"(remaining {u['n_remaining']})")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return

    backup_path = COMPOSITE_PATH + f".bak.{int(time.time())}"
    shutil.copy(COMPOSITE_PATH, backup_path)
    print(f"\nBackup written to {backup_path}")

    by_uid = {u["uid"]: u for u in updates}
    for uid_str, rec in composite_scores.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if uid not in by_uid:
            continue
        u = by_uid[uid]
        rec["broken_axes"] = u["broken_axes"]
        rec["worst"] = u["new_worst"]

    tmp_path = COMPOSITE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(composite_scores, f, indent=2, sort_keys=True)
    os.replace(tmp_path, COMPOSITE_PATH)
    print(f"Wrote {COMPOSITE_PATH} ({len(updates)} records updated).")


if __name__ == "__main__":
    main()
