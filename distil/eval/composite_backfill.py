"""Backfill missing ``composite_scores`` entries from ``h2h_history``.

The cutover from prod's ``scripts/validator`` to ``distil/`` rebuilt
``composite_scores.json`` from scratch (correct schema → new file), but
``evaluated_uids.json`` was preserved (correct policy → keep slot
state). The mismatch left ~120 UIDs flagged as "evaluated" with no
composite on disk; the API surfaces this as
``eval_status: evaluated_no_composite`` and the emission share
calculator skips them (zero ``incentive``, zero ``emission``).

Flagged in #distil 2026-05-16: UID 214 (was king at block 8163228)
shows ``incentive: 0.0`` and ``emission: 0.0`` despite ``kl_score:
1.73`` — they had a valid composite in h2h_history that simply wasn't
copied into ``composite_scores.json`` during the cutover.

This module walks ``state.h2h_history`` newest-first for every
``evaluated_uids`` member without a current composite, finds the
most-recent round that DID produce a usable ``composite`` payload for
that UID, and copies it back into ``state.composite_scores`` so the
emission path sees it again. Idempotent: a UID that already has a
composite is skipped, and UIDs whose h2h rows have no composite (e.g.
prod-era rows without the new schema fields) are left alone — the
3-strikes ``record_failure`` path will handle their re-eval next round.

Runs at the top of ``service._round`` (alongside
``sweep_integrity_dq_recoveries``) so a redeploy that resets
``composite_scores.json`` heals itself within one round.
"""

from __future__ import annotations

import logging
from typing import Any

from distil.state.files import ValidatorState

logger = logging.getLogger("distil.eval.composite_backfill")


def backfill_missing_composites(state: ValidatorState) -> list[dict[str, Any]]:
    """Copy the most-recent h2h composite into ``state.composite_scores``
    for every ``evaluated_uids`` entry missing a current composite.

    Returns a list of backfill records (``[{"uid", "block", "model"}]``)
    suitable for logging. Mutates ``state`` in place; caller saves.
    """
    if not isinstance(state.composite_scores, dict):
        return []
    if not isinstance(state.h2h_history, list):
        return []
    evaluated = {str(u) for u in (state.evaluated_uids or [])}
    if not evaluated:
        return []
    missing = evaluated - set(state.composite_scores.keys())
    if not missing:
        return []
    # Newest-first walk, breaking on first composite-bearing row per uid.
    latest_for: dict[str, tuple[int, dict, str | None]] = {}
    for round_row in sorted(
        state.h2h_history,
        key=lambda r: int(r.get("block") or 0),
        reverse=True,
    ):
        block = int(round_row.get("block") or 0)
        for s in round_row.get("results") or round_row.get("students") or []:
            uid_raw = s.get("uid")
            if uid_raw is None:
                continue
            uid_str = str(int(uid_raw))
            if uid_str in latest_for:
                continue
            comp = s.get("composite")
            if not isinstance(comp, dict) or comp.get("final") is None:
                continue
            model = s.get("model") or s.get("name")
            latest_for[uid_str] = (block, comp, model)
    backfilled: list[dict[str, Any]] = []
    for uid_str in missing:
        rec = latest_for.get(uid_str)
        if rec is None:
            continue
        block, comp, model = rec
        # Preserve the legacy composite verbatim but enrich with the
        # model+revision metadata so downstream code
        # (``select_challengers`` re-commit eviction,
        # ``commitment_changed`` audit) can match the right
        # commitment hash without an extra h2h lookup.
        merged = dict(comp)
        if model and "model" not in merged:
            base = model.split("@", 1)[0] if "@" in model else model
            rev = model.split("@", 1)[1] if "@" in model else None
            merged["model"] = base
            if rev and "revision" not in merged:
                merged["revision"] = rev
        if "block" not in merged:
            merged["block"] = block
        state.composite_scores[uid_str] = merged
        backfilled.append({"uid": int(uid_str), "block": block, "model": model})
    if backfilled:
        logger.info(
            f"backfilled {len(backfilled)} composite_scores from h2h_history "
            f"(uids={[b['uid'] for b in backfilled[:12]]}"
            f"{'...' if len(backfilled) > 12 else ''})"
        )
    return backfilled
