"""Activation fingerprint — functional copy detection.

We greedy-decode the same fixed seed prompts on every model and project the
top-1 token-id sequence into a small float vector. Cosine similarity above
``settings.activation_fp_threshold`` flags two models as functional copies.
"""

from __future__ import annotations

import hashlib

from distil.pod.axes._base import generate_greedy
from distil.settings import settings

SEED_PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def quicksort(arr):",
    "Once upon a time,",
    "import numpy as np\n",
    "The first president of the United States was",
    "Write a haiku about autumn:",
    "SELECT * FROM users WHERE",
)


def _project(token_ids: list[int], dim: int) -> list[float]:
    """Hash-project a token-id sequence into a fixed-length unit vector."""
    if not token_ids:
        return [0.0] * dim
    out = [0.0] * dim
    for t in token_ids:
        b = hashlib.sha256(int(t).to_bytes(4, "big", signed=False)).digest()
        for i in range(dim):
            byte = b[i % 32]
            sign = 1.0 if (byte & 1) else -1.0
            out[i] += sign * (byte / 255.0)
    norm = sum(x * x for x in out) ** 0.5 or 1.0
    return [x / norm for x in out]


def run(student_engine, *, dim: int | None = None) -> list[float]:
    d = int(dim or settings.activation_fp_dim)
    outs = generate_greedy(student_engine, list(SEED_PROMPTS), max_tokens=24)
    all_tokens: list[int] = []
    for _text, toks in outs:
        all_tokens.extend(toks)
    return _project(all_tokens, d)
