"""Sparse top-K KL on the shared support of teacher + student.

We renormalise both distributions over the intersection of the teacher's
top-K and student's top-K, then compute KL(teacher || student) and
RKL(student || teacher). Mean over positions; mean over prompts.
"""

from __future__ import annotations

import math
from typing import Iterator


def _coerce_int_keys(d: dict) -> dict[int, float]:
    """JSON round-tripping stringifies dict keys.

    The teacher_logprobs payload is persisted in
    ``teacher_cache/round_<id>.json`` and re-loaded for Phase 2 student
    scoring. JSON has no concept of integer dict keys, so a payload
    written as ``{42: -1.2}`` round-trips as ``{"42": -1.2}``. Student
    logprobs come straight out of vLLM with native ``int`` keys, so
    without this coercion the two key spaces never intersect and KL +
    top-K overlap silently collapse to ``None`` / ``0`` — that's exactly
    the regression we hit on round ``1778892714`` where every student's
    ``kl_global_avg`` came back ``None`` despite a clean Phase 2 run.

    We accept either ``int`` or ``str(int)`` keys and drop anything that
    won't parse cleanly (defensive against, e.g., special teacher-API
    decode artifacts).
    """
    if not d:
        return {}
    out: dict[int, float] = {}
    for k, v in d.items():
        if isinstance(k, int):
            out[k] = v
            continue
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _renorm(d: dict[int, float], support: set[int]) -> dict[int, float]:
    if not d or not support:
        return {}
    log_probs = {tid: d[tid] for tid in support if tid in d}
    if not log_probs:
        return {}
    log_z = max(log_probs.values()) + math.log(
        sum(math.exp(lp - max(log_probs.values())) for lp in log_probs.values())
    )
    return {tid: lp - log_z for tid, lp in log_probs.items()}


def position_kl(teacher: dict, student: dict) -> float | None:
    """Return KL(teacher || student) on the shared support, or None if disjoint."""
    if not teacher or not student:
        return None
    teacher = _coerce_int_keys(teacher)
    student = _coerce_int_keys(student)
    support = set(teacher) & set(student)
    if not support:
        return None
    t = _renorm(teacher, support)
    s = _renorm(student, support)
    out = 0.0
    for tid in support:
        lt, ls = t.get(tid), s.get(tid)
        if lt is None or ls is None:
            continue
        out += math.exp(lt) * (lt - ls)
    return out


def position_rkl(teacher: dict, student: dict) -> float | None:
    """Return KL(student || teacher) — same as ``position_kl`` with args swapped."""
    return position_kl(student, teacher)


def _iter_positions(
    teacher_seq: list,
    student_seq: list,
) -> "Iterator[tuple[dict[int, float], dict[int, float]]]":
    """Yield aligned ``(teacher_pos_dict, student_pos_dict)`` pairs.

    Caller can pass either:
      * flat shape ``list[dict[int, float]]`` (one prompt, per-position), or
      * nested shape ``list[list[dict[int, float]]]`` (per-prompt then
        per-position).

    The pod's Phase 2 call site builds the nested shape (one teacher
    trace per prompt, each with N positions), and the legacy validator
    aggregated mean-over-positions *then* mean-over-prompts. We do
    "flatten then mean" which yields the SAME value when all prompts
    have equal position counts (true for our fixed max_new_tokens)
    and is robust if a prompt got truncated. Without the shape sniff,
    ``set(teacher)`` blew up on the nested shape with
    ``TypeError: unhashable type: 'dict'`` and every student row
    returned ``kl_global_avg=None`` (the 2026-05-15 round regression).
    """
    if not teacher_seq or not student_seq:
        return
    t0 = teacher_seq[0] if teacher_seq else None
    s0 = student_seq[0] if student_seq else None
    nested = isinstance(t0, list) and isinstance(s0, list)
    if nested:
        for t_prompt, s_prompt in zip(teacher_seq, student_seq, strict=False):
            for t_pos, s_pos in zip(t_prompt or [], s_prompt or [], strict=False):
                yield t_pos, s_pos
    else:
        for t_pos, s_pos in zip(teacher_seq, student_seq, strict=False):
            yield t_pos, s_pos


def average_kl(
    teacher: list,
    student: list,
) -> float | None:
    """Mean of position-level KL(teacher || student) over the joint trace."""
    vals: list[float] = []
    for t, s in _iter_positions(teacher, student):
        v = position_kl(t, s)
        if v is not None and v == v:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def average_rkl(
    teacher: list,
    student: list,
) -> float | None:
    """Mean of position-level KL(student || teacher) over the joint trace."""
    vals: list[float] = []
    for t, s in _iter_positions(teacher, student):
        v = position_rkl(t, s)
        if v is not None and v == v:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def top_k_overlap(
    teacher: list,
    student: list,
    *,
    k: int = 20,
) -> float | None:
    """Mean |top-K(teacher) ∩ top-K(student)| / K over the joint trace."""
    overlaps: list[float] = []
    for t, s in _iter_positions(teacher, student):
        if not t or not s:
            continue
        t = _coerce_int_keys(t)
        s = _coerce_int_keys(s)
        t_top = {tid for tid, _ in sorted(t.items(), key=lambda kv: -kv[1])[:k]}
        s_top = {tid for tid, _ in sorted(s.items(), key=lambda kv: -kv[1])[:k]}
        if not t_top:
            continue
        overlaps.append(len(t_top & s_top) / max(len(t_top), 1))
    return sum(overlaps) / len(overlaps) if overlaps else None
