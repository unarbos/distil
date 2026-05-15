"""v31_ifeval_verifiable — programmatically-verifiable instruction following."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)


def _exact_words(rng):
    n = rng.randint(8, 24)
    return (
        f"Write exactly {n} English words about clouds.",
        lambda t: len([w for w in re.findall(r"[A-Za-z]+", t)]) == n,
    )


def _starts_with(rng):
    word = rng.choice(("Today", "Once", "Surprisingly", "Carefully", "Quickly"))
    return (
        f"Reply with a short sentence that starts with the word '{word}'.",
        lambda t: t.strip().startswith(word),
    )


def _no_letter(rng):
    letter = rng.choice("aeiou")
    return (
        f"Write a sentence about programming without using the letter '{letter}'.",
        lambda t: letter not in t.lower(),
    )


def _bullet_list(rng):
    n = rng.randint(3, 6)
    return (
        f"Reply with a Markdown bullet list of exactly {n} items about animals.",
        lambda t: sum(1 for line in t.splitlines() if line.lstrip().startswith(("-", "*"))) == n,
    )


_GENS = (_exact_words, _starts_with, _no_letter, _bullet_list)


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_ifeval_verifiable")
    items = [rng.choice(_GENS)(rng) for _ in range(n_items)]
    prompts = [q for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=256)
    rows = []
    for (q, check), (text, toks) in zip(items, outs, strict=False):
        ok = False
        try:
            ok = bool(check(text))
        except Exception:
            ok = False
        rows.append({"q": q[:80], "ok": ok, "tokens": len(toks)})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
