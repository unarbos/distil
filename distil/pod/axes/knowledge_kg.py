"""v31_knowledge_multi_hop_kg — multi-hop synthetic knowledge graph queries."""

from __future__ import annotations

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

CITIES = ("Aria", "Belmont", "Cresta", "Dovenia", "Estoria", "Fellpoint", "Glaes", "Hollow")
COUNTRIES = ("Voria", "Wessard", "Xenith", "Yondra", "Zelmar")


def _gen(rng) -> tuple[str, str]:
    rng.shuffle(list(CITIES))
    capital = CITIES[0]
    country = rng.choice(COUNTRIES)
    other = CITIES[1]
    knowledge = (
        f"In the country of {country}, the capital is {capital}. "
        f"{capital} is famous for its {rng.choice(('canals', 'libraries', 'gardens', 'observatories'))}. "
        f"{other} is the largest port. The currency of {country} is the {rng.choice(('luma', 'farad', 'cresc', 'argo'))}. "
    )
    prompt = (
        knowledge
        + f"\n\nQuestion: What is the capital of {country}? Reply with only the city name."
    )
    return prompt, capital


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_knowledge_multi_hop_kg")
    items = [_gen(rng) for _ in range(n_items)]
    prompts = [q for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=64)
    rows = []
    for (_q, ans), (text, toks) in zip(items, outs, strict=False):
        guess = (text or "").strip().split()[0].rstrip(".,!?") if text else ""
        rows.append({"ans": ans, "guess": guess, "ok": guess == ans, "tokens": len(toks)})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
