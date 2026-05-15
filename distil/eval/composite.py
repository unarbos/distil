"""Composite scoring (v31.2 / v32.6).

Live axis surface (sums to ~1.0 with retired tiers all zero):

* Distillation tier (relative to king):
  ``on_policy_rkl``, ``top_k_overlap``, ``kl``, ``capability``,
  ``length``, ``degeneracy``
* Quality tier (teacher-as-judge rubrics):
  ``judge_probe``, ``long_form_judge``, ``long_gen_coherence``,
  ``chat_turns_probe``
* Discipline:
  ``reasoning_density``, ``calibration_bench``
* 11 v31 procedural skill axes (anti-Goodhart, no static items):
  ``v31_math_gsm_symbolic``, ``v31_math_competition``,
  ``v31_math_robustness``, ``v31_code_humaneval_plus``,
  ``v31_reasoning_logic_grid``, ``v31_reasoning_dyval_arith``,
  ``v31_long_context_ruler``, ``v31_knowledge_multi_hop_kg``,
  ``v31_ifeval_verifiable``, ``v31_truthfulness_calibration``,
  ``v31_consistency_paraphrase``

``final = α · worst_K_mean + (1 − α) · weighted`` with
``α = settings.composite_final_bottom_weight`` (default 0.75) and
``K = settings.worst_3_mean_k`` (default 3).

The retired Arena-v3, BENCH, BENCH_GROUP, CANARY and SHADOW axis tiers
are deleted entirely (they were all weight-zero in v31.2 production).
"""

from __future__ import annotations

import logging
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.eval.composite")

COMPOSITE_SCHEMA_VERSION = 32

# ── Constants ────────────────────────────────────────────────────────────

V31_AXIS_NAMES: tuple[str, ...] = (
    "v31_math_gsm_symbolic",
    "v31_math_competition",
    "v31_math_robustness",
    "v31_code_humaneval_plus",
    "v31_reasoning_logic_grid",
    "v31_reasoning_dyval_arith",
    "v31_long_context_ruler",
    "v31_knowledge_multi_hop_kg",
    "v31_ifeval_verifiable",
    "v31_truthfulness_calibration",
    "v31_consistency_paraphrase",
)

# Min valid items per axis; below the floor the axis drops out.
BENCH_MIN_VALID: dict[str, int] = {
    "v31_math_gsm_symbolic": 8,
    "v31_math_competition": 8,
    "v31_math_robustness": 8,
    "v31_code_humaneval_plus": 4,
    "v31_reasoning_logic_grid": 8,
    "v31_reasoning_dyval_arith": 8,
    "v31_long_context_ruler": 6,
    "v31_knowledge_multi_hop_kg": 8,
    "v31_ifeval_verifiable": 6,
    "v31_truthfulness_calibration": 8,
    "v31_consistency_paraphrase": 6,
    "calibration_bench": 4,
}

# Per-axis bench values whose baseline-relative dock applies (absolute
# pass_frac axes only — relative axes would double-penalise).
BASELINE_RELATIVE_PENALTY_AXES: frozenset[str] = frozenset(V31_AXIS_NAMES) | {"calibration_bench"}

# Min valid samples for the rubric-probe axes (drop the axis if fewer
# valid teacher rubric scores were collected this round).
JUDGE_PROBE_MIN_VALID = 4
LONG_FORM_JUDGE_MIN_VALID = 2
CHAT_TURNS_MIN_VALID = 2


# ── Per-axis extractors ─────────────────────────────────────────────────


def _clamp(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(1.0, f))


def _king_ratio(student: dict, field: str, king_ref: float | None) -> float | None:
    """``king_ref / student.field`` clamped to [0, 1]; ``None`` if missing."""
    v = student.get(field)
    if v is None or king_ref is None or king_ref <= 0:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0 or f in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(1.0, float(king_ref) / f))


def _axis_kl(student: dict, king_kl: float | None) -> float | None:
    return _king_ratio(student, "kl_global_avg", king_kl)


def _axis_top_k_overlap(student: dict) -> float | None:
    return _clamp(student.get("top_k_overlap_mean"))


def _axis_on_policy_rkl(student: dict, king_rkl: float | None) -> float | None:
    """king_rkl / student_rkl, clamped to [0, 1].

    On-policy RKL is robust to teacher-forced memorisation. Falls back to
    a 0.1-nat absolute anchor when no king is available (cold-start).
    """
    opr = student.get("on_policy_rkl") or {}
    rkl = opr.get("mean_rkl")
    if rkl is None:
        return None
    try:
        rkl_f = float(rkl)
    except (TypeError, ValueError):
        return None
    if rkl_f != rkl_f:
        return None
    if king_rkl is None or king_rkl <= 0:
        return 1.0 if rkl_f <= 0 else max(0.0, min(1.0, 0.1 / max(rkl_f, 1e-6)))
    return 1.0 if rkl_f <= 0 else max(0.0, min(1.0, float(king_rkl) / rkl_f))


def _axis_capability(student: dict) -> float | None:
    """0.5 · absolute + 0.5 · (frac / max(teacher, 0.5)).

    The 0.5 floor on the denominator stops a flaky teacher round from
    inflating the axis when the student matches a wrong teacher.
    """
    cap = student.get("capability") or {}
    frac = cap.get("pass_frac")
    if frac is None:
        return None
    absolute = _clamp(frac)
    if absolute is None:
        return None
    teach = cap.get("teacher_pass_frac")
    if teach and teach > 0:
        relative = max(0.0, min(1.0, float(frac) / max(float(teach), 0.5)))
    else:
        relative = absolute
    return max(0.0, min(1.0, 0.5 * absolute + 0.5 * relative))


def _axis_length(student: dict) -> float | None:
    la = student.get("length_axis") or {}
    return _clamp(la.get("penalty"))


def _axis_degeneracy(student: dict) -> float | None:
    """think_probe terminate + non-degenerate + self-bleu margin → [0, 1]."""
    tp = student.get("think_probe") or {}
    if not tp:
        return None
    tested = tp.get("prompts_tested") or 0
    if tested == 0:
        return None
    term = (tp.get("prompts_terminated") or 0) / tested
    degen = max(0.0, 1.0 - (tp.get("prompts_degenerate") or 0) / tested)
    sb = tp.get("self_bleu_across_prompts") or 0.0
    teach_sb = tp.get("teacher_self_bleu") or 0.0
    sb_margin = max(0.0, 0.9 - max(sb - teach_sb, 0.0))
    sb_score = min(1.0, sb_margin / 0.9)
    return 0.4 * term + 0.4 * degen + 0.2 * sb_score


def _judge_axis(student: dict, key: str, score_field: str, min_valid: int) -> float | None:
    payload = student.get(key) or {}
    if not payload:
        return None
    if (payload.get("n_valid") or 0) < min_valid:
        return None
    return _clamp(payload.get(score_field))


def _axis_judge_probe(s: dict) -> float | None:
    return _judge_axis(s, "judge_probe", "normalized", JUDGE_PROBE_MIN_VALID)


def _axis_long_form_judge(s: dict) -> float | None:
    return _judge_axis(s, "long_form_judge_probe", "normalized", LONG_FORM_JUDGE_MIN_VALID)


def _axis_long_gen_coherence(s: dict) -> float | None:
    return _judge_axis(s, "long_form_judge_probe", "coherence_factor", LONG_FORM_JUDGE_MIN_VALID)


def _axis_chat_turns_probe(s: dict) -> float | None:
    return _judge_axis(s, "chat_turns_probe", "normalized", CHAT_TURNS_MIN_VALID)


def _axis_bench_pass_frac(student: dict, axis_name: str) -> float | None:
    payload = student.get(axis_name) or {}
    if not payload or payload.get("error"):
        return None
    n = int(payload.get("n") or 0)
    if n < BENCH_MIN_VALID.get(axis_name, 4):
        return None
    return _clamp(payload.get("pass_frac"))


REASONING_DENSITY_TARGETS: dict[str, float] = {
    # v31 procedural axes (steady-state, the only ones distil's pipeline produces).
    "v31_math_gsm_symbolic": 400.0,
    "v31_math_competition": 500.0,
    "v31_math_robustness": 400.0,
    "v31_code_humaneval_plus": 300.0,
    "v31_reasoning_logic_grid": 200.0,
    "v31_reasoning_dyval_arith": 300.0,
    "v31_knowledge_multi_hop_kg": 80.0,
    "v31_ifeval_verifiable": 250.0,
    "v31_truthfulness_calibration": 60.0,
    "v31_consistency_paraphrase": 60.0,
    # Legacy bench targets — only present when rows produced by the prod
    # pipeline are scored by distil (transition / parity-test path). Distil's
    # own pipeline never emits these axes, so they drop out cleanly in
    # steady state. Kept here so distil reproduces prod's `reasoning_density`
    # bit-for-bit when reading prod-produced rows.
    "math_bench": 400.0,
    "code_bench": 300.0,
    "reasoning_bench": 150.0,
    "knowledge_bench": 30.0,
    "ifeval_bench": 250.0,
    "aime_bench": 800.0,
    "mbpp_bench": 250.0,
    "tool_use_bench": 300.0,
    "long_context_bench": 30.0,
    "robustness_bench": 400.0,
    "debug_bench": 350.0,
    "correction_bench": 350.0,
    "multi_doc_synthesis_bench": 60.0,
    "calibration_bench": 60.0,
    "refactor_bench": 300.0,
    "pragmatic_bench": 60.0,
}


def _axis_reasoning_density(student: dict) -> float | None:
    """Mean per-bench pass_frac × length_bonus, matching prod's target dict.

    `length_bonus = 1.0` at ≤ target tokens, `1/(1 + (ratio − 1))` above.
    Same formula as `scripts.validator.composite._axis_reasoning_density`.
    Missing or zero-token benches drop out (fail-open).
    """
    targets = REASONING_DENSITY_TARGETS
    scores: list[float] = []
    for name, target in targets.items():
        payload = student.get(name) or {}
        if not isinstance(payload, dict) or payload.get("error"):
            continue
        n = int(payload.get("n") or 0)
        # Default floor of 2 matches prod (`BENCH_MIN_VALID.get(axis, 2)`).
        # Per-axis overrides are still used when present in BENCH_MIN_VALID.
        if n < BENCH_MIN_VALID.get(name, 2):
            continue
        correct = int(payload.get("correct") or 0)
        if correct == 0:
            scores.append(0.0)
            continue
        mean_tok = float(payload.get("mean_gen_tokens_correct") or 0.0)
        if mean_tok <= 0 or target <= 0:
            continue
        ratio = mean_tok / target
        bonus = 1.0 if ratio <= 1.0 else 1.0 / (1.0 + (ratio - 1.0))
        scores.append((correct / n) * bonus)
    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


# ── Public composite API ────────────────────────────────────────────────


def compute_axes(
    student: dict,
    king_kl: float | None = None,
    king_rkl: float | None = None,
) -> dict[str, float | None]:
    """Per-axis values for one student in [0, 1] (or None when missing)."""
    out: dict[str, float | None] = {
        "on_policy_rkl": _axis_on_policy_rkl(student, king_rkl),
        "kl": _axis_kl(student, king_kl),
        "top_k_overlap": _axis_top_k_overlap(student),
        "capability": _axis_capability(student),
        "length": _axis_length(student),
        "degeneracy": _axis_degeneracy(student),
        "judge_probe": _axis_judge_probe(student),
        "long_form_judge": _axis_long_form_judge(student),
        "long_gen_coherence": _axis_long_gen_coherence(student),
        "chat_turns_probe": _axis_chat_turns_probe(student),
        "reasoning_density": _axis_reasoning_density(student),
        "calibration_bench": _axis_bench_pass_frac(student, "calibration_bench"),
    }
    for axis in V31_AXIS_NAMES:
        out[axis] = _axis_bench_pass_frac(student, axis)
    return out


def resolve_reference_broken_axes(reference_row: dict | None) -> set[str]:
    """Bench axes where the reference scored exactly 0 (eval-setup signal)."""
    if not reference_row:
        return set()
    broken: set[str] = set()
    for axis in V31_AXIS_NAMES:
        bench = reference_row.get(axis)
        if not isinstance(bench, dict):
            continue
        n = int(bench.get("n") or 0)
        pf = bench.get("pass_frac")
        if n > 0 and pf is not None and float(pf) <= 0.0:
            broken.add(axis)
    return broken


def resolve_teacher_broken_axes(
    teacher_row: dict | None,
    king_kl: float | None = None,
    king_rkl: float | None = None,
) -> set[str]:
    """Axes where the teacher itself scores below the sanity floor."""
    if not teacher_row:
        return set()
    teacher_axes = compute_axes(teacher_row, king_kl, king_rkl)
    weights = settings.axis_weights()
    floor = settings.teacher_sanity_floor
    return {
        axis
        for axis, val in teacher_axes.items()
        if axis in weights and val is not None and val < floor
    }


def _apply_baseline_penalty(
    axis: str,
    value: float | None,
    reference_value: float | None,
) -> float | None:
    """Dock a bench axis when the student regresses below the same-round reference."""
    if not settings.baseline_penalty_enabled:
        return value
    if axis not in BASELINE_RELATIVE_PENALTY_AXES:
        return value
    if value is None or reference_value is None:
        return value
    a, r = float(value), float(reference_value)
    if a >= r:
        return value
    return max(0.0, a - settings.baseline_penalty_alpha * (r - a))


def compute_composite(
    student: dict,
    *,
    king_kl: float | None = None,
    king_rkl: float | None = None,
    broken_axes: set[str] | None = None,
    reference_axes: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Compute the per-axis + composite score for one student.

    ``final = α · worst_K_mean + (1 − α) · weighted`` over present + non-broken axes.
    Missing axes drop out of both aggregations.
    """
    raw = compute_axes(student, king_kl=king_kl, king_rkl=king_rkl)
    if reference_axes:
        axes = {k: _apply_baseline_penalty(k, v, reference_axes.get(k)) for k, v in raw.items()}
    else:
        axes = dict(raw)

    weights = {k: w for k, w in settings.axis_weights().items() if w > 0}
    broken = broken_axes or set()

    ranked = {k: v for k, v in axes.items() if v is not None and k in weights and k not in broken}
    weighted_axes = {k: v for k, v in axes.items() if v is not None and k in weights}

    if not ranked:
        return {
            "version": COMPOSITE_SCHEMA_VERSION,
            "axes": axes,
            "axes_raw": raw if reference_axes else None,
            "worst": None,
            "worst_3_mean": None,
            "final": None,
            "final_alpha": settings.composite_final_bottom_weight,
            "weighted": None,
            "present_count": 0,
            "broken_axes": sorted(broken),
        }

    worst = min(ranked.values())
    sorted_vals = sorted(ranked.values())
    k_eff = min(settings.worst_3_mean_k, len(sorted_vals))
    worst_k_mean = sum(sorted_vals[:k_eff]) / k_eff
    total_w = sum(weights[k] for k in weighted_axes)
    weighted = sum(weights[k] * v for k, v in weighted_axes.items()) / total_w if total_w else None
    alpha = settings.composite_final_bottom_weight
    if weighted is not None:
        final = alpha * worst_k_mean + (1.0 - alpha) * weighted
    else:
        final = worst_k_mean

    return {
        "version": COMPOSITE_SCHEMA_VERSION,
        "axes": {k: (round(v, 4) if v is not None else None) for k, v in axes.items()},
        "axes_raw": (
            {k: (round(v, 4) if v is not None else None) for k, v in raw.items()}
            if reference_axes
            else None
        ),
        "worst": round(worst, 4),
        "worst_3_mean": round(worst_k_mean, 4),
        "final": round(final, 4),
        "final_alpha": round(alpha, 4),
        "weighted": round(weighted, 4) if weighted is not None else None,
        "present_count": len(ranked),
        "broken_axes": sorted(broken),
    }


# ── King selection + dethrone gate ──────────────────────────────────────


def select_king(composites: dict[str, dict]) -> str | None:
    """Pick the model with the highest ``final`` composite (skips DQ'd / None)."""
    best: tuple[float, str] | None = None
    for model_name, comp in composites.items():
        f = comp.get("final") if isinstance(comp, dict) else None
        if f is None:
            continue
        try:
            v = float(f)
        except (TypeError, ValueError):
            continue
        if v != v:
            continue
        if best is None or v > best[0]:
            best = (v, model_name)
    return best[1] if best else None


def is_dethrone(
    challenger_composite: dict,
    king_composite: dict | None,
    *,
    margin: float | None = None,
) -> tuple[bool, str]:
    """Return ``(is_dethrone, reason)`` for one challenger vs the seated king.

    Default margin: ``settings.composite_dethrone_margin`` (5%). Fails OPEN
    (no dethrone) when fewer than ``composite_dethrone_min_axes`` axes are
    present in either composite.
    """
    m = float(margin if margin is not None else settings.composite_dethrone_margin)
    cf = challenger_composite.get("final") if isinstance(challenger_composite, dict) else None
    if cf is None:
        return False, "challenger_no_final"
    if not isinstance(king_composite, dict) or king_composite.get("final") is None:
        return True, "no_king"
    kf = float(king_composite["final"])
    cf = float(cf)
    if (challenger_composite.get("present_count") or 0) < settings.composite_dethrone_min_axes:
        return False, "challenger_too_sparse"
    if (king_composite.get("present_count") or 0) < settings.composite_dethrone_min_axes:
        return True, "king_too_sparse"
    if cf >= kf * (1.0 + m):
        return True, f"final_gain {cf:.4f} >= king {kf:.4f} * (1+{m:.3f})"
    return False, f"margin_not_met (challenger={cf:.4f} king={kf:.4f})"
