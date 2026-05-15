"""v31_code_humaneval_plus — sandboxed Python function completion."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    aggregate,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)
from distil.pod.sandbox import run_humaneval

PROBLEMS = (
    {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "tests": "def check(c):\n    assert c(1, 2) == 3\n    assert c(-3, 7) == 4\n",
    },
    {
        "prompt": 'def reverse(s):\n    """Reverse a string."""\n',
        "tests": "def check(c):\n    assert c('abc') == 'cba'\n    assert c('') == ''\n",
    },
    {
        "prompt": 'def is_prime(n):\n    """Return True iff n is prime."""\n',
        "tests": "def check(c):\n    assert c(2)\n    assert c(13)\n    assert not c(15)\n    assert not c(1)\n",
    },
    {
        "prompt": 'def factorial(n):\n    """Iterative factorial."""\n',
        "tests": "def check(c):\n    assert c(5) == 120\n    assert c(0) == 1\n",
    },
    {
        "prompt": 'def unique_sorted(xs):\n    """Return sorted unique values."""\n',
        "tests": "def check(c):\n    assert c([3, 1, 2, 1]) == [1, 2, 3]\n",
    },
)


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else (text or "")


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "v31_code_humaneval_plus")
    items = [rng.choice(PROBLEMS) for _ in range(n_items)]
    prompts = [
        f"Complete the function. Reply with only the function body.\n\n```python\n{p['prompt']}```"
        for p in items
    ]
    outs = generate_greedy(student_engine, prompts, max_tokens=384)
    rows = []
    for problem, (text, toks) in zip(items, outs, strict=False):
        body = _extract_code(text)
        ok = run_humaneval(problem["prompt"], body, problem["tests"], timeout_s=6.0)
        rows.append({"prompt": problem["prompt"][:80], "ok": ok, "tokens": len(toks)})
    return aggregate(rows, completion_tokens=estimate_completion_tokens(outs)).as_dict()
