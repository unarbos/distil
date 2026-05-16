"""v31_code_humaneval_plus — EvalPlus-style augmented HumanEval items.

Each item: ``{prompt, test, entry_point, task_id, src, template,
n_test_cases}``. Generator: ``distil.pod.axes.v31.code_humaneval_plus``.
Grader: sandboxed subprocess that runs ``prompt + completion + test``
and calls ``check({entry_point})``.

The sandbox itself (``distil.pod.sandbox.run_humaneval``) handles:

* fenced code block extraction (```` ```python ... ``` ````)
* chat-style prose stripping (largest-parseable-window heuristic)
* auto-indent recovery for bare ``return ...`` bodies
* redundant ``def {entry_point}`` redeclarations
* per-sample nonce sentinel to prevent ``os._exit(0)`` spoofs

so this module passes the raw generation through untouched.
"""

from __future__ import annotations

import logging

from distil.pod.axes._base import BenchResult, generate_greedy
from distil.pod.axes.v31 import code_humaneval_plus as _v31
from distil.pod.sandbox import run_humaneval

logger = logging.getLogger("distil.pod.axes.code_humaneval")

MAX_TOKENS = 1024
AXIS_NAME = "v31_code_humaneval_plus"


def _build_prompt(item: dict) -> str:
    return (
        "Complete the following Python function. Output ONLY the function "
        "body (no extra explanation, no markdown fences, no surrounding code).\n\n"
        f"{item['prompt']}"
    )


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    if not items:
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}

    prompts = [_build_prompt(it) for it in items]
    try:
        gens = generate_greedy(engine, prompts, max_tokens=MAX_TOKENS)
    except Exception as exc:
        logger.exception(f"code vllm generate failed: {exc}")
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "error": str(exc)[:200]}

    scored: list[dict] = []
    correct = 0
    completion_tokens = 0
    for it, (text, tok_ids) in zip(items, gens, strict=False):
        n_tok = len(tok_ids or ())
        completion_tokens += n_tok
        completion = text or ""
        try:
            ok = run_humaneval(
                it["prompt"],
                completion,
                it["test"],
                entry_point=it["entry_point"],
                timeout_s=8.0,
            )
        except Exception as exc:
            logger.warning(f"code sandbox crashed: {exc}")
            ok = False
        scored.append(
            {
                "src": it.get("src", ""),
                "task_id": it.get("task_id"),
                "entry_point": it.get("entry_point"),
                "ok": bool(ok),
                "tokens": n_tok,
                "n_test_cases": it.get("n_test_cases"),
                "template": it.get("template"),
                "tail": completion[-160:],
            }
        )
        correct += int(ok)

    res = BenchResult(
        n=len(scored), correct=correct, completion_tokens=completion_tokens, items=scored
    )
    return res.as_dict()
