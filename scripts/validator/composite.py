"""Multi-axis composite score computation — shadow mode.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible on the current SN97 king). The fix is to
score each student on several independent axes and combine them with a
worst-case rule, so that gaming any single axis penalizes overall rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: SHADOW. The validator logs this alongside KL and surfaces it to
miners via h2h_results; it does NOT yet decide the king. A 14-day grace
period starts when we ship this so miners can train against the new axes.
After the grace period the composite replaces ``kl`` as the ranking key.
"""
from __future__ import annotations

from typing import Any


AXIS_WEIGHTS = {
    "kl": 0.35,
    "capability": 0.35,
    "length": 0.15,
    "degeneracy": 0.15,
}

COMPOSITE_SHADOW_VERSION = 1


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


def compute_composite(student: dict, king_kl: float | None = None) -> dict:
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


def annotate_h2h_with_composite(h2h_results: list[dict], king_kl: float | None,
                                students_data: dict[Any, dict]) -> None:
    """Mutates h2h_results in place to add ``composite`` per entry.

    ``students_data`` maps model_name -> the raw per-student dict from
    ``pod_eval_vllm.py`` output. We resolve each h2h entry's student by
    model name.
    """
    for entry in h2h_results:
        model = entry.get("model")
        if not model or model not in students_data:
            continue
        comp = compute_composite(students_data[model], king_kl)
        if entry.get("disqualified") and not entry.get("is_king"):
            comp = {**comp, "worst": 0.0, "weighted": 0.0,
                    "disqualified": True, "dq_reason": entry.get("dq_reason")}
        entry["composite"] = comp
