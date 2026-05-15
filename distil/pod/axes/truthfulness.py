"""v31_truthfulness_calibration — SimpleQA-style 3-way calibration axis.

Per-item classification: correct / incorrect / not_attempted. The axis
``pass_frac`` is the SimpleQA-normalised score
``(num_correct - num_incorrect) / num_items`` mapped to [0, 1] — so
confidently-wrong responses are penalised vs honest abstentions.

Items: ``distil.pod.axes.v31.truthfulness_calibration.generate_items``.
Classifier: ``distil.pod.axes.v31.truthfulness_calibration.classify_response``.
"""

from __future__ import annotations

import logging

from distil.pod.axes._base import BenchResult, generate_greedy
from distil.pod.axes.v31 import truthfulness_calibration as _v31

logger = logging.getLogger("distil.pod.axes.truthfulness")

MAX_TOKENS = 256
AXIS_NAME = "v31_truthfulness_calibration"


def run(engine, *, block_seed: int, n_items: int) -> dict:
    items = _v31.generate_items(block_seed, n_items)
    if not items:
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}

    prompts = [it["question"] for it in items]
    try:
        gens = generate_greedy(engine, prompts, max_tokens=MAX_TOKENS)
    except Exception as exc:
        logger.exception(f"truthfulness vllm generate failed: {exc}")
        return {"n": 0, "correct": 0, "pass_frac": 0.0, "error": str(exc)[:200]}

    scored: list[dict] = []
    n_correct = n_incorrect = n_abstain = 0
    completion_tokens = 0
    for it, (text, tok_ids) in zip(items, gens, strict=False):
        n_tok = len(tok_ids or ())
        completion_tokens += n_tok
        try:
            verdict = _v31.classify_response(text, it.get("gold", ""))
        except Exception as exc:
            logger.warning(f"truthfulness classifier crashed: {exc}")
            verdict = "incorrect"
        if verdict == "correct":
            n_correct += 1
        elif verdict == "not_attempted":
            n_abstain += 1
        else:
            n_incorrect += 1
        scored.append(
            {
                "src": it.get("src", ""),
                "gold": str(it.get("gold", ""))[:80],
                "verdict": verdict,
                "ok": verdict == "correct",
                "tokens": n_tok,
                "tail": (text or "")[-120:],
                "family": it.get("family"),
            }
        )

    n = len(scored)
    raw = (n_correct - n_incorrect) / n if n else 0.0
    pass_frac = max(0.0, min(1.0, (raw + 1.0) / 2.0))  # map [-1, 1] -> [0, 1]
    res = BenchResult(
        n=n,
        correct=n_correct,
        completion_tokens=completion_tokens,
        items=scored,
        extra={
            "incorrect": n_incorrect,
            "not_attempted": n_abstain,
            "raw_score": round(raw, 4),
        },
    )
    payload = res.as_dict()
    payload["pass_frac"] = round(pass_frac, 4)
    return payload
