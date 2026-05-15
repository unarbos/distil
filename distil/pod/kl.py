"""Sparse top-K KL on the shared support of teacher + student.

We renormalise both distributions over the intersection of the teacher's
top-K and student's top-K, then compute KL(teacher || student) and
RKL(student || teacher). Mean over positions; mean over prompts.
"""

from __future__ import annotations

import math


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


def position_kl(teacher: dict[int, float], student: dict[int, float]) -> float | None:
    """Return KL(teacher || student) on the shared support, or None if disjoint."""
    if not teacher or not student:
        return None
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


def position_rkl(teacher: dict[int, float], student: dict[int, float]) -> float | None:
    """Return KL(student || teacher) — same as ``position_kl`` with args swapped."""
    return position_kl(student, teacher)


def average_kl(
    teacher_per_pos: list[dict[int, float]],
    student_per_pos: list[dict[int, float]],
) -> float | None:
    vals: list[float] = []
    for t, s in zip(teacher_per_pos, student_per_pos, strict=False):
        v = position_kl(t, s)
        if v is not None and v == v:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def average_rkl(
    teacher_per_pos: list[dict[int, float]],
    student_per_pos: list[dict[int, float]],
) -> float | None:
    vals: list[float] = []
    for t, s in zip(teacher_per_pos, student_per_pos, strict=False):
        v = position_rkl(t, s)
        if v is not None and v == v:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def top_k_overlap(
    teacher_per_pos: list[dict[int, float]],
    student_per_pos: list[dict[int, float]],
    *,
    k: int = 20,
) -> float | None:
    overlaps: list[float] = []
    for t, s in zip(teacher_per_pos, student_per_pos, strict=False):
        if not t or not s:
            continue
        t_top = {tid for tid, _ in sorted(t.items(), key=lambda kv: -kv[1])[:k]}
        s_top = {tid for tid, _ in sorted(s.items(), key=lambda kv: -kv[1])[:k]}
        if not t_top:
            continue
        overlaps.append(len(t_top & s_top) / max(len(t_top), 1))
    return sum(overlaps) / len(overlaps) if overlaps else None
