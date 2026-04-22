"""Multi-axis composite score computation — production ranking key.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible on the current SN97 king). The fix is to
score each student on several independent axes and combine them with a
worst-case rule, so that gaming any single axis penalizes overall rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: PRODUCTION (promoted from shadow 2026-04-19, commit 8eec9a2).
``composite.worst`` is the primary ranking key used by
``scripts/validator/results.py`` to order the leaderboard and select the
canonical challenger. Crown transitions still additionally require the
paired t-test on KL + 3% epsilon (``epsilon_dethroned_by``) so a single
bad round cannot dethrone the king on axis noise. Axes that are missing
for a given round (e.g. ``degeneracy`` while ``THINK_COLLAPSE_PROBE=0``)
drop out and the weighted mean renormalizes over the surviving axes.
"""
from __future__ import annotations

from typing import Any


# Five-axis composite (T1.3). On-policy RKL is the primary distillation
# signal under the new framework — it is the axis that miners cannot game
# by teacher-forced memorization, because the rollouts are the student's
# own policy. KL (off-policy forward-KL) is retained as a transparency
# axis but down-weighted; its existing saturation at the top of the board
# (Δ < 0.0005 nats across the top-5) is exactly what we’re moving away
# from. Capability and the two structural axes round out the signal so a
# model must be competitive on reasoning, length discipline, and
# non-degenerate generation — not just logit-matching.
AXIS_WEIGHTS = {
    "on_policy_rkl": 0.35,
    "kl": 0.15,
    "capability": 0.25,
    "length": 0.10,
    "degeneracy": 0.15,
}

COMPOSITE_SHADOW_VERSION = 2


def _axis_kl(student: dict, king_kl: float | None) -> float:
    """Normalize KL to [0, 1] higher-is-better.

    We normalize against the best (lowest) KL of the current king rather
    than an absolute anchor: anchoring on the king keeps the axis scaled
    to real, achievable values. A student with ``kl = king_kl`` scores
    1.0; KL at 2× king → ~0.5; KL at 10× king → ~0.1.
    """
    kl = student.get("kl_global_avg")
    if kl is None or kl <= 0 or king_kl is None or king_kl <= 0:
        return 0.0
    return max(0.0, min(1.0, king_kl / kl))


def _axis_capability(student: dict) -> float:
    """Verifiable-rewards pass fraction, normalized by teacher."""
    cap = student.get("capability") or {}
    frac = cap.get("pass_frac")
    if frac is None:
        return None
    teach = cap.get("teacher_pass_frac")
    if teach and teach > 0:
        return max(0.0, min(1.0, frac / teach))
    return max(0.0, min(1.0, frac))


def _axis_length(student: dict) -> float:
    """Length penalty as stored by the eval script. Already in [0, 1]."""
    la = student.get("length_axis") or {}
    pen = la.get("penalty")
    return None if pen is None else max(0.0, min(1.0, pen))


def _axis_degeneracy(student: dict) -> float:
    """Think-probe terminates+non-degenerate+self-bleu as a single score.

    Pass on all three components => 1.0. Partial pass linearly interpolated.
    The probe uses MAD-z against teacher statistics when available, so this
    axis is already threshold-free.
    """
    tp = student.get("think_probe") or {}
    if not tp:
        return None
    tested = tp.get("prompts_tested") or 0
    if tested == 0:
        return None
    term = tp.get("prompts_terminated") or 0
    degen = tp.get("prompts_degenerate") or 0
    sb = tp.get("self_bleu_across_prompts") or 0.0
    teach_sb = tp.get("teacher_self_bleu") or 0.0
    term_score = term / tested
    degen_score = max(0.0, 1.0 - degen / tested)
    sb_margin = max(0.0, 0.9 - max(sb - teach_sb, 0.0))
    sb_score = min(1.0, sb_margin / 0.9)
    return 0.4 * term_score + 0.4 * degen_score + 0.2 * sb_score


def _axis_on_policy_rkl(student: dict, king_rkl: float | None) -> float:
    """Normalize on-policy reverse KL to [0, 1] higher-is-better.

    On-policy RKL is the primary distillation signal under the new
    framework: it is computed on the student's *own* rollouts, so the
    student cannot hide behind teacher-forced memorization. Lower RKL
    means the student's policy is closer to the teacher in the mode-
    seeking direction, which is the actual objective of distillation.

    We normalize against the king's RKL: a student tied with the king
    scores 1.0; RKL at 2× king → ~0.5; at 10× king → ~0.1. If RKL
    numbers are missing (old snapshot, probe disabled, probe errored),
    return None so the axis drops out and the weighted mean
    renormalizes over the remaining axes.
    """
    opr = student.get("on_policy_rkl") or {}
    if not opr:
        return None
    rkl = opr.get("mean_rkl")
    if rkl is None or rkl != rkl:  # NaN
        return None
    if king_rkl is None or king_rkl <= 0:
        # Fall back to an absolute-ish anchor: RKL under ~0.1 nats is
        # typical for well-distilled students on-policy in our smoke tests.
        if rkl <= 0:
            return 1.0
        return max(0.0, min(1.0, 0.1 / max(rkl, 1e-6)))
    if rkl <= 0:
        return 1.0
    return max(0.0, min(1.0, king_rkl / rkl))


def compute_composite(student: dict, king_kl: float | None = None,
                      king_rkl: float | None = None) -> dict:
    """Return per-axis and composite (worst-case + weighted mean) scores.

    We emit *both* aggregations so the validator can A/B them offline
    before committing to one as the canonical score:

    - ``worst`` (Coste 2024, Pan 2025 min-form): the minimum of present
      axes. This is the anti-gaming rule — you win only if all axes are
      competitive. Robust to axis-specific overfitting.
    - ``weighted`` (standard convex combination with AXIS_WEIGHTS): a
      softer aggregation that still rewards high-KL students somewhat,
      useful during the grace period so we don't suddenly dethrone the
      current king while miners re-tool.
    """
    axes = {
        "on_policy_rkl": _axis_on_policy_rkl(student, king_rkl),
        "kl": _axis_kl(student, king_kl),
        "capability": _axis_capability(student),
        "length": _axis_length(student),
        "degeneracy": _axis_degeneracy(student),
    }
    present = {k: v for k, v in axes.items() if v is not None}
    if not present:
        return {"version": COMPOSITE_SHADOW_VERSION, "axes": axes,
                "worst": None, "weighted": None, "present_count": 0}
    worst = min(present.values())
    total_w = sum(AXIS_WEIGHTS[k] for k in present)
    weighted = sum(AXIS_WEIGHTS[k] * v for k, v in present.items()) / total_w if total_w else None
    return {
        "version": COMPOSITE_SHADOW_VERSION,
        "axes": {k: (round(v, 4) if v is not None else None) for k, v in axes.items()},
        "worst": round(worst, 4),
        "weighted": round(weighted, 4) if weighted is not None else None,
        "present_count": len(present),
    }


def _resolve_king_rkl(king_kl: float | None,
                      students_data: dict[Any, dict],
                      h2h_results: list[dict]) -> float | None:
    """Round-wide reference RKL for axis normalization.

    We anchor the RKL axis on the **best** (lowest) RKL observed in the
    round rather than the king's, because the king might be the model we
    are trying to dethrone for an on-policy pathology — if we anchored
    on the king's RKL the challenger could never score above 1.0 on
    this axis and the pathology would stay invisible in the composite.

    The resolution order is:
      1. Round-wide minimum mean_rkl across all students with a probe
         record, as long as at least one non-king student reported.
      2. Fall back to the king's own RKL if only the king reported
         (shouldn't really happen but keeps the axis defined).
      3. Return ``None`` if nobody reported, letting
         ``compute_composite`` use an absolute fallback anchor.
    """
    best = None
    non_king_best = None
    king_model = None
    king_entry = next((r for r in h2h_results if r.get("is_king")), None)
    if king_entry:
        king_model = king_entry.get("model")
    for model_name, data in students_data.items():
        opr = (data or {}).get("on_policy_rkl") or {}
        v = opr.get("mean_rkl")
        if v is None or v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
        if model_name != king_model and (non_king_best is None or v < non_king_best):
            non_king_best = v
    if non_king_best is not None and best is not None:
        return best
    if best is not None:
        return best
    return None


def annotate_h2h_with_composite(h2h_results: list[dict], king_kl: float | None,
                                students_data: dict[Any, dict]) -> None:
    """Mutates h2h_results in place to add ``composite`` per entry.

    ``students_data`` maps model_name -> the raw per-student dict from
    ``pod_eval_vllm.py`` output. We resolve each h2h entry's student by
    model name.
    """
    king_rkl = _resolve_king_rkl(king_kl, students_data, h2h_results)
    for entry in h2h_results:
        model = entry.get("model")
        if not model or model not in students_data:
            continue
        comp = compute_composite(students_data[model], king_kl, king_rkl)
        if entry.get("disqualified") and not entry.get("is_king"):
            comp = {**comp, "worst": 0.0, "weighted": 0.0,
                    "disqualified": True, "dq_reason": entry.get("dq_reason")}
        entry["composite"] = comp
