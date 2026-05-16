"""Regression: KL must work when teacher_logprobs comes back from JSON.

The pod persists ``teacher_logprobs`` to ``teacher_cache/round_<id>.json``
between Phase 1 (teacher generation) and Phase 2 (per-student scoring),
which JSON-stringifies the integer token-id dict keys. Student logprobs
come straight out of vLLM with native ``int`` keys, so without int-key
coercion the support intersection is always empty and every student's
``kl_global_avg`` silently collapses to ``None`` — exactly the regression
we hit on round 1778892714 (king Foremost04 + UID 137 both vLLM-loaded
cleanly, ran 16 bench axes, but came back with kl=None and top_k=0).
"""

from __future__ import annotations

import json
import math

from distil.pod.kl import (
    average_kl,
    average_rkl,
    position_kl,
    top_k_overlap,
)


def _make_position(top_ids: list[int], top_logprobs: list[float]) -> dict[int, float]:
    return dict(zip(top_ids, top_logprobs, strict=True))


def test_position_kl_handles_string_keyed_teacher() -> None:
    """Teacher returns dict with str(int) keys (post JSON round-trip)."""
    teacher = _make_position([1, 2, 3], [-0.1, -1.5, -2.5])
    teacher_after_json = json.loads(json.dumps(teacher))
    assert all(isinstance(k, str) for k in teacher_after_json)

    student = _make_position([1, 2, 4], [-0.2, -1.2, -3.0])  # native int keys

    kl = position_kl(teacher_after_json, student)
    assert kl is not None and not math.isnan(kl)


def test_average_kl_after_json_roundtrip_is_finite() -> None:
    """Both flat and nested shapes must work after teacher JSON round-trip."""
    teacher_flat = [
        _make_position([1, 2, 3], [-0.1, -1.5, -2.5]),
        _make_position([1, 4, 5], [-0.2, -1.4, -2.4]),
    ]
    student_flat = [
        _make_position([1, 2, 3], [-0.15, -1.3, -2.7]),
        _make_position([1, 4, 6], [-0.25, -1.6, -2.2]),
    ]
    teacher_after_json = json.loads(json.dumps(teacher_flat))

    kl = average_kl(teacher_after_json, student_flat)
    rkl = average_rkl(teacher_after_json, student_flat)
    overlap = top_k_overlap(teacher_after_json, student_flat, k=3)

    assert kl is not None and kl >= 0.0
    assert rkl is not None and rkl >= 0.0
    assert overlap is not None and overlap > 0.0


def test_nested_shape_after_json_roundtrip() -> None:
    """Nested ``list[list[dict]]`` is the actual on-pod payload shape."""
    teacher_nested = [
        [
            _make_position([1, 2, 3], [-0.1, -1.5, -2.5]),
            _make_position([1, 4, 5], [-0.2, -1.4, -2.4]),
        ],
        [
            _make_position([1, 2, 7], [-0.15, -1.3, -3.0]),
        ],
    ]
    student_nested = [
        [
            _make_position([1, 2, 3], [-0.12, -1.6, -2.6]),
            _make_position([1, 4, 8], [-0.22, -1.5, -2.5]),
        ],
        [
            _make_position([1, 2, 9], [-0.18, -1.4, -2.9]),
        ],
    ]
    teacher_after_json = json.loads(json.dumps(teacher_nested))

    kl = average_kl(teacher_after_json, student_nested)
    overlap = top_k_overlap(teacher_after_json, student_nested, k=3)

    assert kl is not None and kl >= 0.0
    assert overlap is not None and overlap > 0.0


def test_disjoint_supports_still_returns_none() -> None:
    """If teacher + student truly share no token ids, KL is still None."""
    teacher = {1: -0.1, 2: -1.5}
    student = {99: -0.2, 100: -1.3}
    teacher_after_json = json.loads(json.dumps(teacher))
    assert position_kl(teacher_after_json, student) is None
