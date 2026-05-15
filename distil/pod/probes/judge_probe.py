"""judge_probe — short single-turn quality rubric (1-5 scale)."""

from __future__ import annotations

import re
from typing import Any

from distil.pod.axes._base import block_seeded_rng, generate_greedy

PROMPTS = (
    "Explain why the sky appears blue, in 2-3 sentences.",
    "Write a short, friendly note to a colleague about a delayed report.",
    "Summarise the plot of Hamlet in three sentences.",
    "Describe how to make a basic vinaigrette.",
    "Explain the difference between TCP and UDP in plain English.",
)

RUBRIC = (
    "You are a strict grader. Rate the following response from 1 (poor) to 5 (excellent) "
    "on accuracy, helpfulness and clarity. Reply with ONLY the integer score.\n\n"
    "PROMPT: {prompt}\n\nRESPONSE: {response}\n\nScore (1-5):"
)


def _extract_score(text: str) -> int | None:
    m = re.search(r"\b([1-5])\b", text or "")
    return int(m.group(1)) if m else None


def run(student_engine, teacher_engine, *, block_seed: int, n_items: int) -> dict[str, Any]:
    rng = block_seeded_rng(block_seed, "judge_probe")
    prompts = [rng.choice(PROMPTS) for _ in range(n_items)]
    responses = generate_greedy(student_engine, prompts, max_tokens=256)
    judge_prompts = [
        RUBRIC.format(prompt=p, response=r[0]) for p, r in zip(prompts, responses, strict=False)
    ]
    judges = generate_greedy(teacher_engine, judge_prompts, max_tokens=8)
    scores = [_extract_score(t) for t, _ in judges]
    valid = [s for s in scores if s is not None]
    if not valid:
        return {"n": len(prompts), "n_valid": 0, "normalized": None}
    mean = sum(valid) / len(valid)
    return {
        "n": len(prompts),
        "n_valid": len(valid),
        "mean_score": round(mean, 3),
        "normalized": round((mean - 1) / 4, 4),
    }
