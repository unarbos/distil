"""Validator-host orchestration: composite, round, results, king, service.

This package owns the validator-host control flow. The GPU-pod runner
lives in :mod:`distil.pod`; the FastAPI dashboard backend in
:mod:`distil.api`.
"""

from distil.eval.composite import (
    BENCH_MIN_VALID,
    COMPOSITE_SCHEMA_VERSION,
    compute_axes,
    compute_composite,
    is_dethrone,
    resolve_reference_broken_axes,
    resolve_teacher_broken_axes,
    select_king,
)

__all__ = [
    "BENCH_MIN_VALID",
    "COMPOSITE_SCHEMA_VERSION",
    "compute_axes",
    "compute_composite",
    "is_dethrone",
    "resolve_reference_broken_axes",
    "resolve_teacher_broken_axes",
    "select_king",
]
