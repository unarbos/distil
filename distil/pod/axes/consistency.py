"""v31_consistency_paraphrase — paraphrase consistency on factual yes/no items."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

ITEMS = (
    ("Is water wet?", "yes"),
    ("Is fire cold?", "no"),
    ("Are humans mammals?", "yes"),
    ("Is the sun a planet?", "no"),
    ("Do birds have feathers?", "yes"),
    ("Is iron a noble gas?", "no"),
)

PARAPHRASES = (
    "Answer yes or no: {q}",
    "Reply 'yes' or 'no': {q}",
    "{q} (yes/no)",
    "True/false reformulated as yes/no: {q}",
)


def _gen(rng) -> tuple[list[str], str]:
    q, ans = rng.choice(ITEMS)
    prompts = [tpl.format(q=q) for tpl in PARAPHRASES]
    return prompts, ans


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_consistency_paraphrase")
    item_prompts: list[list[str]] = []
    item_answers: list[str] = []
    for _ in range(n_items):
        ps, a = _gen(rng)
        item_prompts.append(ps)
        item_answers.append(a)
    flat = [p for ps in item_prompts for p in ps]
    outs = generate_greedy(student_engine, flat, max_tokens=8)
    rows = []
    cursor = 0
    for ps, ans in zip(item_prompts, item_answers, strict=False):
        guesses = []
        for _ in ps:
            text, _toks = outs[cursor]
            cursor += 1
            m = re.search(r"\b(yes|no)\b", (text or "").lower())
            guesses.append(m.group(1) if m else "")
        consistent = all(g == ans for g in guesses)
        rows.append({"ans": ans, "guesses": guesses, "ok": consistent, "tokens": 0})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
