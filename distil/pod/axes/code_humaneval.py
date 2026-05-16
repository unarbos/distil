"""v31_code_humaneval_plus — EvalPlus-style augmented HumanEval items.

Each item: ``{prompt, test, entry_point, task_id, src, template,
n_test_cases}``. Generator: ``distil.pod.axes.v31.code_humaneval_plus``.
Grader: sandboxed subprocess that runs ``prompt + completion + test``
and checks ``check(candidate)`` returns None.
"""

from __future__ import annotations

import logging
import re

from distil.pod.axes._base import BenchResult, generate_greedy
from distil.pod.axes.v31 import code_humaneval_plus as _v31
from distil.pod.sandbox import run_humaneval

logger = logging.getLogger("distil.pod.axes.code_humaneval")

MAX_TOKENS = 1024
AXIS_NAME = "v31_code_humaneval_plus"

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_body(text: str, entry_point: str) -> str:
    """Pull the function body out of ``text`` for ``def {entry_point}``."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    body = m.group(1) if m else text
    if f"def {entry_point}" in body:
        idx = body.index(f"def {entry_point}")
        body = body[idx:]
    return body


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
        body = _extract_body(text or "", it["entry_point"])
        try:
            ok = run_humaneval(it["prompt"], body, it["test"], timeout_s=8.0)
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
                "tail": body[-160:],
            }
        )
        correct += int(ok)

    res = BenchResult(
        n=len(scored), correct=correct, completion_tokens=completion_tokens, items=scored
    )
    return res.as_dict()
