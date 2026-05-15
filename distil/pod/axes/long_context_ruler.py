"""v31_long_context_ruler — needle-in-a-haystack at the 32K context cap."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

_HAYSTACK_LINE = (
    "The sky was a uniform shade of grey, like wet concrete. The wind smelled of rain. "
)


def _gen(rng) -> tuple[str, str]:
    secret = f"The secret code is {rng.randint(10000, 99999)}-{rng.choice('ABCDEFGH')}."
    n_lines = rng.choice((600, 1200, 2000))  # ~12K – 28K tokens
    needle_pos = rng.randint(int(n_lines * 0.2), int(n_lines * 0.8))
    lines = []
    for i in range(n_lines):
        lines.append(secret if i == needle_pos else _HAYSTACK_LINE)
    haystack = "\n".join(lines)
    code = re.search(r"\d{5}-[A-Z]", secret).group(0)
    prompt = (
        "Read the document below carefully. Somewhere inside is a sentence "
        "starting with 'The secret code is'. Reply with ONLY the code "
        "(format: NNNNN-X).\n\n--- DOCUMENT ---\n"
        + haystack
        + "\n--- END DOCUMENT ---\n\nSecret code: "
    )
    return prompt, code


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_long_context_ruler")
    items = [_gen(rng) for _ in range(n_items)]
    prompts = [q for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=32)
    rows = []
    for (_q, ans), (text, toks) in zip(items, outs, strict=False):
        m = re.search(r"\d{5}-[A-Z]", text or "")
        guess = m.group(0) if m else ""
        rows.append({"ans": ans, "guess": guess, "ok": guess == ans, "tokens": len(toks)})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
