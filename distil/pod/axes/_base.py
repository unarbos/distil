"""Shared infrastructure for v31 axis runners.

* :class:`BenchResult` — the standard ``{n, correct, pass_frac, ...}`` payload
* :func:`block_seeded_rng` — deterministic per-axis RNG keyed on ``(block, axis)``
* :func:`generate_greedy` — single batched ``LLM.generate`` call helper
* :func:`aggregate` — mean / pass-frac / token-count rollup
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchResult:
    """Standard axis payload."""

    n: int = 0
    correct: int = 0
    completion_tokens: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_frac(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def mean_gen_tokens_correct(self) -> float:
        toks = [it.get("tokens", 0) for it in self.items if it.get("ok")]
        return sum(toks) / len(toks) if toks else 0.0

    def as_dict(self) -> dict[str, Any]:
        out = {
            "n": self.n,
            "correct": self.correct,
            "pass_frac": round(self.pass_frac, 4),
            "completion_tokens": self.completion_tokens,
            "mean_gen_tokens_correct": round(self.mean_gen_tokens_correct, 1),
            "items": self.items[:64],
        }
        out.update(self.extra)
        return out


def block_seeded_rng(block_seed: int, axis_name: str) -> random.Random:
    """Per-axis deterministic RNG so different validators draw the same items."""
    h = hashlib.sha256(f"{block_seed}:{axis_name}".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def generate_greedy(
    engine,
    prompts: list[str],
    *,
    max_tokens: int,
    temperature: float = 0.0,
):
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    outs = engine.generate(prompts, params)
    return [(o.outputs[0].text, list(o.outputs[0].token_ids)) for o in outs]


def aggregate(items: list[dict[str, Any]], *, completion_tokens: int = 0) -> BenchResult:
    res = BenchResult()
    res.n = len(items)
    res.correct = sum(1 for it in items if it.get("ok"))
    res.completion_tokens = completion_tokens or sum(it.get("tokens", 0) for it in items)
    res.items = items
    return res


def estimate_completion_tokens(generations: list[tuple[str, list[int]]]) -> int:
    return sum(len(toks) for _, toks in generations)
