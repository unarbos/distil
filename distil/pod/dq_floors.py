"""Pod-side DQ checks: vocab-OOB, runaway repetition, length floor."""

from __future__ import annotations

from collections import Counter


def vocab_oob_rate(token_ids: list[int], vocab_size: int) -> float:
    if not token_ids:
        return 0.0
    n = sum(1 for t in token_ids if not (0 <= int(t) < vocab_size))
    return n / len(token_ids)


def is_runaway_repetition(text: str, *, window: int = 64, threshold: float = 0.6) -> bool:
    """A response is degenerate if any 16-char substring exceeds ``threshold`` of a sliding window."""
    if not text or len(text) < window:
        return False
    chunks = [text[i : i + 16] for i in range(0, len(text) - 16)]
    if not chunks:
        return False
    _most_common, count = Counter(chunks).most_common(1)[0]
    return count / len(chunks) >= threshold


def length_floor_ok(text: str, *, min_chars: int = 4) -> bool:
    return len((text or "").strip()) >= min_chars


def dq_response(
    text: str, *, vocab_size: int | None = None, token_ids: list[int] | None = None
) -> str | None:
    if not length_floor_ok(text):
        return "non_responsive"
    if is_runaway_repetition(text):
        return "runaway_repetition"
    if vocab_size and token_ids and vocab_oob_rate(token_ids, vocab_size) > 0.0:
        return "vocab_oob"
    return None
