"""v31_consistency_paraphrase — paraphrase-pair consistency axis.

Each item carries two paraphrased versions of the same math question
(``question`` / ``question_b``) plus a single ``gold``. We generate
one response per phrasing and score:

* 1.0 — both correct
* 0.5 — exactly one correct (surface-form fragility)
* 0.0 — neither correct

This deliberately differs from pure pass-frac for the other v31 axes —
a model that's right on one phrasing and wrong on the other is showing
partial memorisation, which is what this axis is built to catch.
"""

from __future__ import annotations

import logging

from distil.pod.axes._base import BenchResult, generate_greedy
from distil.pod.axes.v31 import consistency_paraphrase as _v31

logger = logging.getLogger("distil.pod.axes.consistency")

MAX_TOKENS = 512
AXIS_NAME = "v31_consistency_paraphrase"


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    if not items:
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}

    prompts_a = [it["question"] for it in items]
    prompts_b = [it["question_b"] for it in items]
    try:
        gens_a = generate_greedy(engine, prompts_a, max_tokens=MAX_TOKENS)
        gens_b = generate_greedy(engine, prompts_b, max_tokens=MAX_TOKENS)
    except Exception as exc:
        logger.exception(f"consistency vllm generate failed: {exc}")
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "error": str(exc)[:200]}

    scored: list[dict] = []
    total_score = 0.0
    both_correct = 0
    completion_tokens = 0
    for it, (a_text, a_tok), (b_text, b_tok) in zip(items, gens_a, gens_b, strict=False):
        completion_tokens += len(a_tok or ()) + len(b_tok or ())
        s = float(_v31.consistency_score(a_text or "", b_text or "", str(it.get("gold", ""))))
        total_score += s
        if s >= 0.999:
            both_correct += 1
        scored.append(
            {
                "src": it.get("src", ""),
                "gold": str(it.get("gold", ""))[:40],
                "score": round(s, 3),
                "ok": s >= 0.999,
                "tokens": len(a_tok or ()) + len(b_tok or ()),
                "tail_a": (a_text or "")[-80:],
                "tail_b": (b_text or "")[-80:],
                "template": it.get("template"),
                "difficulty": it.get("difficulty"),
            }
        )

    n = len(scored)
    res = BenchResult(
        n=n,
        correct=both_correct,
        completion_tokens=completion_tokens,
        items=scored,
        extra={"raw_consistency_mean": round(total_score / n, 4) if n else 0.0},
    )
    payload = res.as_dict()
    payload["pass_frac"] = round(total_score / n, 4) if n else 0.0
    return payload
