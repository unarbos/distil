"""reasoning_density — measure the fraction of reasoning tokens per axis.

This is a derived metric — it reads the per-axis ``mean_gen_tokens_correct``
already populated by the v31 axis runners and combines them into a single
``reasoning_density`` score. The aggregation logic actually lives in
:mod:`distil.eval.composite._axis_reasoning_density`; this file is a tiny
shim so callers can ``from distil.pod.probes.reasoning_density import run``.
"""

from __future__ import annotations

from distil.eval.composite import _axis_reasoning_density


def run(per_axis_results: dict[str, dict]) -> dict[str, float | None]:
    """Return a per-student-payload entry for ``reasoning_density``."""
    score = _axis_reasoning_density(per_axis_results)
    return {"score": score}
