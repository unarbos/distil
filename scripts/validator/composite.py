"""Multi-axis composite score computation — production ranking + dethrone gate.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible on the 2026-04-22 SN97 king, which rambles
3–10x longer than the teacher on trivial prompts while passing KL). The
fix is to score each student on several independent axes and combine them
with a worst-case rule, so that gaming any single axis penalizes overall
rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: PRODUCTION — ranking + dethrone veto.
  * 2026-04-19 (commit 8eec9a2): promoted from shadow to production
    ranking key. ``composite.worst`` orders the leaderboard and selects
    the canonical challenger for display.
  * 2026-04-22 (this commit): ``composite.worst`` is now ALSO a dethrone
    gate. A challenger that passes the KL paired t-test + 3% epsilon is
    still blocked from taking the crown if its worst composite axis is
    below ``COMPOSITE_DETHRONE_FLOOR`` (currently 0.20). See
    ``scripts/validator/results.py::_composite_dethrone_veto``.
  * Same commit: the ``length`` axis is now always populated even when
    ``THINK_COLLAPSE_PROBE=0``. It falls back to the always-on
    ``chat_probe`` length vs a teacher anchor captured in
    ``prepare_teacher_probe_refs_*``. This closes the gap that let a
    KL-specialized-but-rambling model keep the crown unopposed.
  * 2026-04-23: judge-probe axis added in SHADOW mode. The teacher
    scores each student's greedy response to 16 rotated realistic
    prompts on a 1-5 rubric, normalized to [0, 1]. Computed + logged
    per round but excluded from ``worst`` / ``weighted`` aggregation
    until the ``JUDGE_AXIS_IN_COMPOSITE`` gate flips (Session 2). See
    ``reports/2026-04-23-goodhart-immune-eval.md``.

Axes that are missing for a given round (e.g. ``degeneracy`` while
``THINK_COLLAPSE_PROBE=0``) drop out and the weighted mean renormalizes
over the surviving axes. The veto fails open if fewer than
``COMPOSITE_DETHRONE_MIN_AXES`` axes are populated — we don't want a pod
probe outage to freeze the crown.
"""
from __future__ import annotations

import os
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

# 2026-04-23 — judge axis weight used only when promoted to production
# via ``JUDGE_AXIS_IN_COMPOSITE=1``. When promoted, the remaining axes
# retain their current relative weighting and the judge weight is added
# on top before normalization; callers see a per-axis breakdown and the
# aggregated worst/weighted, so the absolute number changes slightly
# but ordering intent is preserved.
JUDGE_AXIS_WEIGHT = float(os.environ.get("JUDGE_AXIS_WEIGHT", "0.20"))

# Shadow/promote gate. False by default: judge axis is computed, stored
# on each student's composite payload, and visible on the dashboard,
# but is excluded from ``worst`` + ``weighted`` and cannot dethrone.
# Flip to ``1`` during Session 2 rollout after 48h of telemetry.
JUDGE_AXIS_IN_COMPOSITE = os.environ.get("JUDGE_AXIS_IN_COMPOSITE", "0") != "0"

COMPOSITE_SHADOW_VERSION = 3

# ── Teacher sanity gate (2026-04-23) ──────────────────────────────────────
# For each ranking axis we can optionally compute the axis value for the
# teacher itself (scored as if it were a student). If the teacher's axis
# value falls below this floor, the axis is miscalibrated for the round
# (probe miscoded, prompt pool corrupted, etc.) and must be dropped before
# it can corrupt rankings. This is the structural defense against the
# 2026-04-19 outage class (Wilson-anchor think-probe DQ'd the teacher
# itself, so every student failed). See
# ``reports/2026-04-23-goodhart-immune-eval.md`` section on invariants.
#
# Threshold reasoning: a well-calibrated axis should show the teacher at
# >= 0.85 comfortably (the teacher IS what we distill to, any axis where
# the teacher scores poorly is definitionally measuring the wrong thing).
# We pick 0.70 as the "drop the axis" floor to give some slack for
# stochasticity in the teacher's own generations (temperature=0 helps but
# vLLM sampling can still jitter), while still catching outright bugs.
TEACHER_SANITY_FLOOR = 0.70


def _axis_kl(student: dict, king_kl: float | None) -> float | None:
    """Normalize KL to [0, 1] higher-is-better.

    We normalize against the best (lowest) KL of the current king rather
    than an absolute anchor: anchoring on the king keeps the axis scaled
    to real, achievable values. A student with ``kl = king_kl`` scores
    1.0; KL at 2× king → ~0.5; KL at 10× king → ~0.1.

    Returns ``None`` when KL data is missing (e.g. the teacher-as-student
    row, which has no KL vs itself). This lets the teacher sanity gate
    correctly skip the axis for the teacher rather than marking it
    "broken" because of absent-by-design data.
    """
    kl = student.get("kl_global_avg")
    if kl is None or kl <= 0 or king_kl is None or king_kl <= 0:
        return None
    return max(0.0, min(1.0, king_kl / kl))


def _axis_capability(student: dict) -> float:
    """Verifiable-rewards pass fraction with absolute-correctness floor.

    Before 2026-04-23 this axis was purely ``frac / teacher_frac``, which
    hit 1.0 whenever a student matched the teacher — *including on the
    teacher's wrong answers*. Empirically this was mild because the
    teacher got ~85-90% on the capability pool, but in principle it
    rewards teacher-hacking: a student that learns to echo teacher's
    mistakes scores identically to one that actually learned to answer.
    The 2026-04-23 Goodhart-immune eval design (see
    ``reports/2026-04-23-goodhart-immune-eval.md``) adds an absolute
    term so matching the teacher at a low absolute accuracy is not
    credited as full marks.

    Shape:
      score = (absolute_accuracy + min(frac / max(teacher, 0.5), 1.0)) / 2

    Properties:
      * Monotonic in both absolute (frac) and relative (frac / teacher)
        correctness.
      * Student at 100%, teacher at 100% → 1.0 (unchanged).
      * Student and teacher both at 30% → 0.65 (was 1.0).
      * Student at 50%, teacher at 100% → 0.50 (was 0.50; unchanged).
      * Student at 80%, teacher at 80% → 0.90 (was 1.0).
      * Student at 100%, teacher at 60% → (1.0 + 1.0) / 2 = 1.0.
      * Floor of 0.5 in the relative denominator prevents a flaky
        round (teacher errored on many items) from saturating the
        axis.

    Returns ``None`` if the probe didn't run.
    """
    cap = student.get("capability") or {}
    frac = cap.get("pass_frac")
    if frac is None:
        return None
    absolute = max(0.0, min(1.0, float(frac)))
    teach = cap.get("teacher_pass_frac")
    if teach and teach > 0:
        relative = max(0.0, min(1.0, float(frac) / max(float(teach), 0.5)))
    else:
        relative = absolute
    return max(0.0, min(1.0, 0.5 * absolute + 0.5 * relative))


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


def _axis_judge_probe(student: dict) -> float | None:
    """Teacher-as-judge normalized score in [0, 1]. 2026-04-23 shadow axis.

    Returns the ``normalized`` field from the eval script's judge probe
    payload: teacher scores each of 16 rotated prompts on a 1-5 rubric,
    valid scores are averaged, mapped via ``(mean - 1) / 4``. If too
    many prompts failed to parse (``n_valid < 8``) we drop the axis —
    that's a rubric/teacher drift signal and the telemetry is more
    meaningful than a noisy score. ``None`` if the probe didn't run or
    didn't report.
    """
    jp = student.get("judge_probe") or {}
    if not jp:
        return None
    norm = jp.get("normalized")
    if norm is None:
        return None
    if (jp.get("n_valid") or 0) < 8:
        return None
    return max(0.0, min(1.0, float(norm)))


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


def compute_axes(student: dict, king_kl: float | None = None,
                 king_rkl: float | None = None) -> dict[str, float | None]:
    """Compute the raw per-axis values for one student dict.

    Pulled out of ``compute_composite`` so that the teacher sanity gate
    can score the teacher itself on the same axes (by passing the teacher
    row from ``results['students']``). Returns a dict keyed by axis name;
    values are floats in [0, 1] or None if the axis couldn't be computed.

    The ``judge_probe`` key is always populated when the probe reported
    data, but it is only included in ``worst`` / ``weighted`` aggregation
    by ``compute_composite`` when ``JUDGE_AXIS_IN_COMPOSITE`` is truthy.
    """
    return {
        "on_policy_rkl": _axis_on_policy_rkl(student, king_rkl),
        "kl": _axis_kl(student, king_kl),
        "capability": _axis_capability(student),
        "length": _axis_length(student),
        "degeneracy": _axis_degeneracy(student),
        "judge_probe": _axis_judge_probe(student),
    }


def resolve_teacher_broken_axes(teacher_student_row: dict | None,
                                king_kl: float | None = None,
                                king_rkl: float | None = None) -> set[str]:
    """Identify axes where the teacher itself fails the sanity floor.

    If ``teacher_student_row`` is None (no teacher-as-student probe this
    round, e.g. the teacher-as-student pass wasn't added yet) returns an
    empty set — fail open. For any axis where the teacher scores a real
    value < ``TEACHER_SANITY_FLOOR`` we return that axis name so the
    caller can drop it from ranking. Axes where the teacher returns
    None are considered uncalibrated and also dropped defensively.
    """
    broken: set[str] = set()
    if not teacher_student_row:
        return broken
    teacher_axes = compute_axes(teacher_student_row, king_kl, king_rkl)
    for axis, val in teacher_axes.items():
        if axis not in AXIS_WEIGHTS:
            continue
        if val is None:
            continue
        if val < TEACHER_SANITY_FLOOR:
            broken.add(axis)
    return broken


def compute_composite(student: dict, king_kl: float | None = None,
                      king_rkl: float | None = None,
                      broken_axes: set[str] | None = None) -> dict:
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

    ``broken_axes`` (2026-04-23): axes where the teacher itself failed
    the sanity floor this round. They are computed + logged per-student
    but excluded from ``worst`` / ``weighted`` aggregation. Caller is
    responsible for passing the result of ``resolve_teacher_broken_axes``
    once per round (it only depends on the teacher row and the king
    anchors).
    """
    axes = compute_axes(student, king_kl, king_rkl)
    # Build effective weights: judge axis is shadow-only unless the
    # promote gate is on. Keeping this local to compute_composite so a
    # single env flip flows to every caller without touching AXIS_WEIGHTS.
    effective_weights = dict(AXIS_WEIGHTS)
    if JUDGE_AXIS_IN_COMPOSITE:
        effective_weights["judge_probe"] = JUDGE_AXIS_WEIGHT
    ranked = {
        k: v for k, v in axes.items()
        if v is not None
        and k in effective_weights
        and (not broken_axes or k not in broken_axes)
    }
    if not ranked:
        return {"version": COMPOSITE_SHADOW_VERSION, "axes": axes,
                "worst": None, "weighted": None, "present_count": 0,
                "broken_axes": sorted(broken_axes) if broken_axes else [],
                "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE}
    worst = min(ranked.values())
    total_w = sum(effective_weights[k] for k in ranked)
    weighted = sum(effective_weights[k] * v for k, v in ranked.items()) / total_w if total_w else None
    return {
        "version": COMPOSITE_SHADOW_VERSION,
        "axes": {k: (round(v, 4) if v is not None else None) for k, v in axes.items()},
        "worst": round(worst, 4),
        "weighted": round(weighted, 4) if weighted is not None else None,
        "present_count": len(ranked),
        "broken_axes": sorted(broken_axes) if broken_axes else [],
        "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE,
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
                                students_data: dict[Any, dict],
                                teacher_student_row: dict | None = None) -> None:
    """Mutates h2h_results in place to add ``composite`` per entry.

    ``students_data`` maps model_name -> the raw per-student dict from
    ``pod_eval_vllm.py`` output. We resolve each h2h entry's student by
    model name.

    ``teacher_student_row`` (optional, 2026-04-23) is the teacher's own
    per-student row — when pod_eval_vllm runs the teacher through the
    student probes (think/chat/capability/rkl) and deposits a row under
    the teacher's model name, pass that row here. Axes where the teacher
    falls below ``TEACHER_SANITY_FLOOR`` are dropped from every
    challenger's composite ranking this round with a note, preventing a
    miscalibrated axis from corrupting rankings (2026-04-19 failure
    class). If None / absent, every axis stays in play — fail open.
    """
    king_rkl = _resolve_king_rkl(king_kl, students_data, h2h_results)
    broken = resolve_teacher_broken_axes(teacher_student_row, king_kl, king_rkl)
    for entry in h2h_results:
        model = entry.get("model")
        if not model or model not in students_data:
            continue
        comp = compute_composite(students_data[model], king_kl, king_rkl, broken)
        if entry.get("disqualified") and not entry.get("is_king"):
            comp = {**comp, "worst": 0.0, "weighted": 0.0,
                    "disqualified": True, "dq_reason": entry.get("dq_reason")}
        entry["composite"] = comp
