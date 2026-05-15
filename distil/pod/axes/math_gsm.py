"""v31_math_gsm_symbolic — Apple GSM-Symbolic procedural arithmetic.

Items: ``distil.pod.axes.v31.math_gsm_symbolic.generate_items``.
Grader: shared ``_math.extract_answer`` + ``_math.score_answer``.
"""

from __future__ import annotations

from distil.pod.axes import _math
from distil.pod.axes._runner import run_axis
from distil.pod.axes.v31 import math_gsm_symbolic as _v31

MAX_TOKENS = 768
AXIS_NAME = "v31_math_gsm_symbolic"


def _grade(text: str, item: dict) -> bool:
    pred = _math.extract_answer(text, item.get("src", ""))
    return bool(_math.score_answer(pred, str(item.get("gold", ""))))


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    return run_axis(
        engine,
        axis_name=AXIS_NAME,
        items=items,
        prompt_fn=lambda it: _math.format_prompt(it["question"], it.get("src", "")),
        grader=_grade,
        max_tokens=MAX_TOKENS,
        extra_item_keys=("difficulty", "is_noop", "template"),
    )
