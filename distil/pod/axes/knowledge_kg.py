"""v31_knowledge_multi_hop_kg — synthetic multi-hop KG queries.

Items: ``distil.pod.axes.v31.knowledge_multi_hop_kg.generate_items``.
Grader: ``distil.pod.axes.v31.knowledge_multi_hop_kg.grade_response``
(takes an ``all_correct`` list so equivalent answers count).
"""

from __future__ import annotations

from distil.pod.axes._runner import run_axis
from distil.pod.axes.v31 import knowledge_multi_hop_kg as _v31

MAX_TOKENS = 256
AXIS_NAME = "v31_knowledge_multi_hop_kg"


def _grade(text: str, item: dict) -> bool:
    return bool(
        _v31.grade_response(
            text,
            item.get("gold", ""),
            all_correct=item.get("all_correct_answers") or [],
        )
    )


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    return run_axis(
        engine,
        axis_name=AXIS_NAME,
        items=items,
        prompt_fn=lambda it: it["question"],
        grader=_grade,
        max_tokens=MAX_TOKENS,
        extra_item_keys=("task",),
    )
