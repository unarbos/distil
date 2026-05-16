"""v31_reasoning_logic_grid — Zebra-style constraint-satisfaction puzzles.

Items: ``distil.pod.axes.v31.reasoning_logic_grid.generate_items``.
Grader: ``distil.pod.axes.v31.reasoning_logic_grid.grade_response``.
"""

from __future__ import annotations

from distil.pod.axes._runner import run_axis
from distil.pod.axes.v31 import reasoning_logic_grid as _v31

MAX_TOKENS = 1024
AXIS_NAME = "v31_reasoning_logic_grid"


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
        extra_item_keys=("num_people", "num_attrs", "num_clues"),
    )
