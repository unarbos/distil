"""Generic per-axis runner.

Each v31 axis is a pair (generator, grader). The generator is a pure
function that emits ``{question, gold, ...}`` items keyed on
``block_seed``; the grader is a pure function that takes
``(response_text, item)`` and returns ``bool``. This module wires them
to a vLLM engine and emits the standard ``BenchResult`` payload that
``distil.eval.composite`` consumes.

Why split it out: the v31 generators in ``distil/pod/axes/v31/`` are
**identical** to the production ones in ``scripts/v31/`` (a literal
copy with the cross-axis import paths re-pointed). Keeping the runner
separate from the generators means we can re-validate scoring against
production by importing either ``scripts.v31.X.generate_items`` or
``distil.pod.axes.v31.X.generate_items`` from the same harness.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from distil.pod.axes._base import BenchResult, generate_greedy

logger = logging.getLogger("distil.pod.axes._runner")

# A grader takes (response_text, item) -> bool. Items may include extra
# context (e.g. ``all_correct_answers`` for the KG axis).
Grader = Callable[[str, dict[str, Any]], bool]
PromptFn = Callable[[dict[str, Any]], str]


def run_axis(
    engine,
    *,
    axis_name: str,
    items: list[dict[str, Any]],
    prompt_fn: PromptFn,
    grader: Grader,
    max_tokens: int,
    extra_item_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Score ``items`` on ``engine``; return the standard payload."""
    if not items:
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}

    prompts = [prompt_fn(it) for it in items]
    try:
        gens = generate_greedy(engine, prompts, max_tokens=max_tokens)
    except Exception as exc:
        logger.exception(f"axis {axis_name} vllm generate failed: {exc}")
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "error": str(exc)[:200]}

    scored: list[dict[str, Any]] = []
    correct = 0
    completion_tokens = 0
    for it, (text, tok_ids) in zip(items, gens, strict=False):
        n_tok = len(tok_ids or ())
        completion_tokens += n_tok
        try:
            ok = bool(grader(text, it))
        except Exception as exc:
            logger.warning(f"axis {axis_name} grader crashed on item: {exc}")
            ok = False
        correct += int(ok)
        row: dict[str, Any] = {
            "src": it.get("src", ""),
            "gold": str(it.get("gold", ""))[:80],
            "ok": ok,
            "tokens": n_tok,
            "tail": (text or "")[-120:],
        }
        for k in extra_item_keys:
            if k in it:
                row[k] = it[k]
        scored.append(row)

    result = BenchResult(
        n=len(scored), correct=correct, completion_tokens=completion_tokens, items=scored
    )
    return result.as_dict()
