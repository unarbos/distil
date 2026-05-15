"""v31_math_competition — modular arithmetic / number-theory style items."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)


def _modular_gcd_lcm(rng) -> tuple[str, int]:
    a = rng.randint(20, 999)
    b = rng.randint(20, 999)

    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    g = gcd(a, b)
    return (
        f"Find gcd({a}, {b}). Reply with only the integer.",
        g,
    )


def _power_mod(rng) -> tuple[str, int]:
    base = rng.randint(2, 30)
    exp = rng.randint(2, 12)
    mod = rng.randint(7, 97)
    return (
        f"Compute {base}^{exp} mod {mod}. Reply with only the integer.",
        pow(base, exp, mod),
    )


def _arithmetic_sum(rng) -> tuple[str, int]:
    a = rng.randint(1, 50)
    d = rng.randint(1, 9)
    n = rng.randint(8, 20)
    s = n * (2 * a + (n - 1) * d) // 2
    return (
        f"What is the sum of the first {n} terms of the arithmetic sequence "
        f"starting at {a} with common difference {d}? Reply with only the integer.",
        s,
    )


_GENS = (_modular_gcd_lcm, _power_mod, _arithmetic_sum)


def _extract(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_math_competition")
    items = [rng.choice(_GENS)(rng) for _ in range(n_items)]
    prompts = [f"{q}\n\nAnswer: " for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=192)
    rows = []
    for (q, ans), (text, toks) in zip(items, outs, strict=False):
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
