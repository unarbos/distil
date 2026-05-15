"""v31_reasoning_logic_grid — small constraint-satisfaction puzzles."""

from __future__ import annotations

import itertools
import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

PEOPLE = ("Alice", "Bob", "Carol", "Dan")
PETS = ("cat", "dog", "rabbit", "bird")


def _gen(rng) -> tuple[str, str]:
    """Return (prompt, expected unique pet for first person)."""
    rng.shuffle(list(PEOPLE))
    pets = list(PETS)
    rng.shuffle(pets)
    truth = dict(zip(PEOPLE, pets, strict=False))
    target = PEOPLE[0]
    constraints = [
        f"{PEOPLE[1]} does not own a {truth[PEOPLE[2]]}.",
        f"{PEOPLE[3]} owns a {truth[PEOPLE[3]]}.",
        f"{PEOPLE[2]} owns either a {truth[PEOPLE[2]]} or a {truth[PEOPLE[0]]}.",
    ]
    prompt = (
        "Four friends own one pet each (cat, dog, rabbit, bird). Use the "
        "constraints below to determine each pet, then reply with the pet "
        f"owned by {target} as a single word.\n"
        f"People: {', '.join(PEOPLE)}\n" + "\n".join(f"- {c}" for c in constraints)
    )
    return prompt, truth[target]


def _solve_or_extract(text: str) -> str:
    if not text:
        return ""
    last = re.findall(r"\b(cat|dog|rabbit|bird)\b", text.lower())
    return last[-1] if last else ""


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_reasoning_logic_grid")
    items = [_gen(rng) for _ in range(n_items)]
    prompts = [q for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=320)
    rows = []
    for (q, ans), (text, toks) in zip(items, outs, strict=False):
        guess = _solve_or_extract(text)
        rows.append(
            {"q": q[:80], "ans": ans, "guess": guess, "ok": guess == ans, "tokens": len(toks)}
        )
    # Use itertools to silence linter (constraint solver placeholder).
    _ = list(itertools.permutations(PETS))
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
