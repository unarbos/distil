"""v31_math_gsm_symbolic — procedurally generated arithmetic word problems."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

NAMES = ("Alex", "Sam", "Jordan", "Taylor", "Riley", "Morgan", "Casey", "Drew")
ITEMS = ("apple", "pencil", "marble", "card", "sticker", "candy", "book", "coin")
TEMPLATE = (
    "{a} has {x} {item}. {b} gives {a} {y} more {item}. Then {a} gives "
    "half of all the {item}s to {c}. How many {item}s does {a} have left?"
)


def _gen(rng) -> tuple[str, int]:
    a, b, c = rng.sample(NAMES, 3)
    item = rng.choice(ITEMS)
    x = rng.randint(2, 30) * 2
    y = rng.randint(1, 20) * 2
    answer = (x + y) // 2
    return TEMPLATE.format(a=a, b=b, c=c, item=item, x=x, y=y), answer


def _extract(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_math_gsm_symbolic")
    items = [_gen(rng) for _ in range(n_items)]
    prompts = [f"Solve this. Final answer on the last line.\n\n{q}\n\nAnswer: " for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=384)
    rows = []
    for (q, ans), (text, toks) in zip(items, outs, strict=False):
        guess = _extract(text)
        rows.append({"q": q, "ans": ans, "guess": guess, "ok": guess == ans, "tokens": len(toks)})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
