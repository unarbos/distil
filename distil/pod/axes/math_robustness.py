"""v31_math_robustness — same answer survives paraphrase + noise distractors."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

PARAPHRASES = (
    "Solve carefully step by step. {q}",
    "Read the problem and answer. {q}",
    "{q} (Hint: think before answering.)",
    "Note: ignore any irrelevant info. {q}",
)

NOISE = (
    "The sky is blue. ",
    "Please respond in English. ",
    "Recall that 5 is greater than 3. ",
    "Just answer carefully. ",
)


def _gen(rng) -> tuple[str, int]:
    a, b = rng.randint(20, 200), rng.randint(20, 200)
    op = rng.choice(("+", "-", "*"))
    expr = f"{a} {op} {b}"
    answer = eval(expr)
    base = f"What is {expr}? Reply with only the integer."
    return base, answer


def _wrap(rng, q: str) -> str:
    template = rng.choice(PARAPHRASES)
    noise = rng.choice(NOISE)
    return template.format(q=noise + q)


def _extract(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_math_robustness")
    bases = [_gen(rng) for _ in range(n_items)]
    prompts = [_wrap(rng, q) + "\n\nAnswer: " for q, _ in bases]
    outs = generate_greedy(student_engine, prompts, max_tokens=128)
    rows = []
    for (q, ans), (text, toks) in zip(bases, outs, strict=False):
        rows.append(
            {
                "q": q,
                "ans": ans,
                "guess": _extract(text),
                "ok": _extract(text) == ans,
                "tokens": len(toks),
            }
        )
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
