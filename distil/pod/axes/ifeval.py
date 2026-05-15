"""v31_ifeval_verifiable — Google IFEval verifiable-instructions axis.

Items: ``distil.pod.axes.v31.ifeval_verifiable.generate_items`` (each
item carries ``instruction_ids`` + ``kwargs`` listing which verifiers
must pass).
Grader: ``_ifeval_vendor.evaluate_item`` (vendored Google IFEval
verifier subset, Apache 2.0).
"""

from __future__ import annotations

import logging

from distil.pod.axes._base import BenchResult, generate_greedy
from distil.pod.axes._math import strip_thinking
from distil.pod.axes.v31 import _ifeval_vendor, ifeval_verifiable as _v31

logger = logging.getLogger("distil.pod.axes.ifeval")

MAX_TOKENS = 1024
AXIS_NAME = "v31_ifeval_verifiable"


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    if not items:
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}

    prompts = [it["prompt"] for it in items]
    try:
        gens = generate_greedy(engine, prompts, max_tokens=MAX_TOKENS)
    except Exception as exc:
        logger.exception(f"ifeval vllm generate failed: {exc}")
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "error": str(exc)[:200]}

    scored: list[dict] = []
    correct = 0
    completion_tokens = 0
    for it, (text, tok_ids) in zip(items, gens, strict=False):
        n_tok = len(tok_ids or ())
        completion_tokens += n_tok
        cleaned = strip_thinking(text or "")
        try:
            all_pass, per = _ifeval_vendor.evaluate_item(
                cleaned, it["instruction_ids"], it.get("kwargs") or []
            )
        except Exception as exc:
            logger.warning(f"ifeval evaluate_item crashed: {exc}")
            all_pass, per = False, []
        scored.append(
            {
                "src": it.get("src", ""),
                "instruction_ids": it.get("instruction_ids"),
                "per_instruction": per,
                "stack_depth": it.get("stack_depth"),
                "ok": bool(all_pass),
                "tokens": n_tok,
                "tail": (text or "")[-120:],
            }
        )
        correct += int(all_pass)

    res = BenchResult(
        n=len(scored), correct=correct, completion_tokens=completion_tokens, items=scored
    )
    return res.as_dict()
