"""Process the per-student JSON the pod returns into the validator's state.

* Compute composite scores for each student (with broken-axes + baseline penalty)
* Persist the head-to-head round record + history
* SHA256 + activation-fingerprint dedupe (DQ near-copies)
* Update ``model_hashes``, ``composite_scores``, ``h2h_*``, ``top4_leaderboard``
"""

from __future__ import annotations

import logging
import time
from typing import Any

from distil.eval.composite import (
    compute_axes,
    compute_composite,
    resolve_reference_broken_axes,
    resolve_teacher_broken_axes,
)
from distil.settings import settings
from distil.state.files import ValidatorState

logger = logging.getLogger("distil.eval.results")


def _student_row(rows: dict[str, dict], pred) -> dict | None:
    for row in rows.values():
        if pred(row):
            return row
    return None


def _resolve_anchor(rows: dict[str, dict], king_name: str | None, key: str) -> float | None:
    """Pick the anchor for relative axes (king if seated, else round-min)."""
    if king_name and (kr := rows.get(king_name)) is not None:
        v = kr.get(key)
        try:
            f = float(v)
            if f > 0 and f == f and f not in (float("inf"), float("-inf")):
                return f
        except (TypeError, ValueError):
            pass
    best: float | None = None
    for row in rows.values():
        if row.get("is_teacher"):
            continue
        try:
            v = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
    return best


def _opr_anchor(rows: dict[str, dict], king_name: str | None) -> float | None:
    """on_policy_rkl uses the round-wide best (lowest) RKL when a non-king reported."""
    best, non_king_best = None, None
    for name, row in rows.items():
        opr = (row or {}).get("on_policy_rkl") or {}
        rkl = opr.get("mean_rkl")
        if rkl is None:
            continue
        try:
            v = float(rkl)
        except (TypeError, ValueError):
            continue
        if v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
        if name != king_name and (non_king_best is None or v < non_king_best):
            non_king_best = v
    return best


def _is_near_copy(
    fp: list[float] | None,
    others: dict[str, list[float]],
    *,
    threshold: float,
) -> str | None:
    """Return the model_name of a near-copy match, else None."""
    if not fp:
        return None
    import math

    norm_a = math.sqrt(sum(x * x for x in fp)) or 1.0
    for name, other in others.items():
        if not other or len(other) != len(fp):
            continue
        norm_b = math.sqrt(sum(x * x for x in other)) or 1.0
        dot = sum(a * b for a, b in zip(fp, other, strict=False))
        cos = dot / (norm_a * norm_b)
        if cos >= threshold:
            return name
    return None


def process_round(
    *,
    state: ValidatorState,
    pod_results: dict[str, dict[str, Any]],
    king_name: str | None,
    reference_name: str | None,
    teacher_name: str | None,
    block: int,
    block_hash: str | None,
    timings: list[dict] | None = None,
) -> dict[str, Any]:
    """Mutate ``state`` in place; return the round record."""
    students: dict[str, dict] = dict(pod_results)
    teacher_row = students.get(teacher_name) if teacher_name else None
    reference_row = students.get(reference_name) if reference_name else None
    students.get(king_name) if king_name else None

    king_kl = _resolve_anchor(students, king_name, "kl_global_avg")
    king_rkl = _opr_anchor(students, king_name)
    broken = resolve_reference_broken_axes(reference_row) | resolve_teacher_broken_axes(
        teacher_row,
        king_kl=king_kl,
        king_rkl=king_rkl,
    )
    reference_axes: dict[str, float | None] | None = None
    if reference_row is not None:
        reference_axes = compute_axes(reference_row, king_kl=king_kl, king_rkl=king_rkl)

    composites: dict[str, dict] = {}
    for name, row in students.items():
        if row.get("is_teacher"):
            continue
        comp = compute_composite(
            row,
            king_kl=king_kl,
            king_rkl=king_rkl,
            broken_axes=broken,
            reference_axes=reference_axes if name != reference_name else None,
        )
        comp["evaluated_at"] = time.time()
        composites[name] = comp
        state.composite_scores[name] = comp

    # SHA256 dedupe + activation-fingerprint near-copy DQ.
    fps_by_other: dict[str, list[float]] = {}
    for name, row in students.items():
        sha = row.get("weights_sha256")
        if sha:
            prior = state.model_hashes.get(sha)
            if prior and prior != name:
                state.disqualify(row.get("hotkey", name), f"duplicate_weights_of:{prior}")
            else:
                state.model_hashes[sha] = name
        fp = row.get("activation_fingerprint")
        if isinstance(fp, list) and fp:
            state.activation_fingerprints[name] = fp
            match = _is_near_copy(fp, fps_by_other, threshold=settings.activation_fp_threshold)
            if match:
                state.disqualify(row.get("hotkey", name), f"activation_fp_near_copy_of:{match}")
            fps_by_other[name] = fp

    record = {
        "block": block,
        "block_hash": block_hash,
        "ts": time.time(),
        "king_name": king_name,
        "reference_name": reference_name,
        "teacher_name": teacher_name,
        "broken_axes": sorted(broken),
        "students": [
            {
                "name": name,
                "uid": row.get("uid"),
                "hotkey": row.get("hotkey"),
                "is_king": (name == king_name),
                "is_reference": (name == reference_name),
                "composite": composites.get(name),
                "axes_summary": composites.get(name, {}).get("axes"),
            }
            for name, row in students.items()
            if not row.get("is_teacher")
        ],
        "per_bench_timing": timings or [],
    }
    state.append_round(record)
    state.save()
    _refresh_top4(state, composites)
    return record


def _refresh_top4(state: ValidatorState, composites: dict[str, dict]) -> None:
    ranked = sorted(
        ((name, c) for name, c in composites.items() if c.get("final") is not None),
        key=lambda kv: kv[1]["final"],
        reverse=True,
    )[:4]
    state.top4_leaderboard = {
        "updated_at": time.time(),
        "rows": [
            {
                "rank": i + 1,
                "name": name,
                "final": c.get("final"),
                "worst_3_mean": c.get("worst_3_mean"),
                "weighted": c.get("weighted"),
                "present_count": c.get("present_count"),
            }
            for i, (name, c) in enumerate(ranked)
        ],
    }
