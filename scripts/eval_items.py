"""Procedural item-generation helpers for pod eval."""
from __future__ import annotations


def _rot_text(s: str, n: int) -> str:
    if not s:
        return s
    n = n % len(s)
    return s[n:] + s[:n]
