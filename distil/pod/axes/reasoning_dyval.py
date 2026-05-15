"""v31_reasoning_dyval_arith — DyVal-style multi-step arithmetic graphs."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)


def _gen(rng) -> tuple[str, int]:
    n_ops = rng.randint(3, 6)
    bindings: list[str] = []
    last_var = None
    val: int | None = None
    for i in range(n_ops):
        var = chr(ord("a") + i)
        if last_var is None:
            v = rng.randint(2, 19)
            bindings.append(f"{var} = {v}")
            val = v
        else:
            op = rng.choice(("+", "-", "*"))
            x = rng.randint(2, 12)
            bindings.append(f"{var} = {last_var} {op} {x}")
            val = eval(f"{val} {op} {x}")
        last_var = var
    prompt = (
        "Compute the value of the last variable. Reply with only the integer.\n"
        + "\n".join(bindings)
        + f"\n\n{last_var} = ?"
    )
    return prompt, int(val)


def _extract(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_reasoning_dyval_arith")
    items = [_gen(rng) for _ in range(n_items)]
    prompts = [q for q, _ in items]
    outs = generate_greedy(student_engine, prompts, max_tokens=192)
    rows = []
    for (q, ans), (text, toks) in zip(items, outs, strict=False):
        rows.append(
            {
                "q": q[:80],
                "ans": ans,
                "guess": _extract(text),
                "ok": _extract(text) == ans,
                "tokens": len(toks),
            }
        )
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
