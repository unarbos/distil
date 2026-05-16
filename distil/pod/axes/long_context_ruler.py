"""v31_long_context_ruler — RULER long-context needle-in-a-haystack suite.

Items: ``distil.pod.axes.v31.long_context_ruler.generate_items``.
Grader: ``distil.pod.axes.v31.long_context_ruler.grade_response``.
"""

from __future__ import annotations

from distil.pod.axes._runner import run_axis
from distil.pod.axes.v31 import long_context_ruler as _v31

MAX_TOKENS = 256
AXIS_NAME = "v31_long_context_ruler"


def _grade(text: str, item: dict) -> bool:
    return bool(_v31.grade_response(text, item.get("gold", "")))


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    return run_axis(
        engine,
        axis_name=AXIS_NAME,
        items=items,
        prompt_fn=lambda it: it["question"],
        grader=_grade,
        max_tokens=MAX_TOKENS,
        extra_item_keys=("task", "n_distractors", "needle_pos"),
    )
