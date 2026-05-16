"""Per-model timeouts + GPU/disk hygiene + CUDA-poisoned quarantine + log-stall detection.

The log-stall detector replicates the production fix that caught the
``diffuznik`` incident (a student model that hung in a tokenizer
``while True`` and silently stalled the whole round). The orchestrator
tails each shard's stdout and feeds the line stream into a
:class:`LineStallDetector` — when either the line stream goes silent
or the tail goes loop-y the orchestrator kills the shard and moves on.
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("distil.pod.watchdog")


class WallClock:
    """Counts down a budget; raises :class:`TimeoutError` on ``check()``."""

    def __init__(self, budget_s: float):
        self.budget_s = float(budget_s)
        self.start = time.time()

    def remaining(self) -> float:
        return max(0.0, self.budget_s - (time.time() - self.start))

    def check(self, label: str = "") -> None:
        if self.remaining() <= 0:
            raise TimeoutError(f"wall_clock_exceeded:{label}")


def free_gpu() -> None:
    """Best-effort GPU memory + CUDA-context cleanup between models."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:
        logger.debug(f"free_gpu torch path: {exc}")
    gc.collect()


def cuda_alive() -> bool:
    """Return False if the CUDA context is poisoned (model went OOM badly)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def disk_free_gb(path: str = "/home") -> float:
    return shutil.disk_usage(path).free / (1024**3)


@dataclass
class LineStallDetector:
    """Detects two stall modes in a streaming log tail.

    1. **Silence** — no new line for ``stale_after_s`` seconds.
    2. **Repeat loop** — the last ``repeat_window`` lines are all
       identical (the model has gone into a tight token-emission loop
       and is flooding stdout with the same suffix).

    Both modes return :class:`StallVerdict` so callers can log a clear
    reason and decide whether to ``SIGTERM`` the offending shard.
    """

    stale_after_s: float = 180.0  # Prod default: 3 min silence kills.
    repeat_window: int = 12  # Prod default: 12 identical tails ⇒ kill.
    _last_change_ts: float = field(default_factory=time.time)
    _tail: deque[str] = field(default_factory=lambda: deque(maxlen=32))

    def observe(self, line: str) -> None:
        line = (line or "").rstrip()
        if not line:
            return
        if not self._tail or line != self._tail[-1]:
            self._last_change_ts = time.time()
        self._tail.append(line)

    def repeat_tail(self) -> int:
        """How many consecutive identical lines are at the tail."""
        if not self._tail:
            return 0
        last = self._tail[-1]
        n = 0
        for line in reversed(self._tail):
            if line == last:
                n += 1
            else:
                break
        return n

    def check(self) -> "StallVerdict | None":
        silence_s = time.time() - self._last_change_ts
        if silence_s >= self.stale_after_s:
            return StallVerdict(reason="silence", stale_s=silence_s, repeat_tail=self.repeat_tail())
        rep = self.repeat_tail()
        if rep >= self.repeat_window:
            return StallVerdict(
                reason="repeat_loop", stale_s=silence_s, repeat_tail=rep, sample=self._tail[-1]
            )
        return None


@dataclass
class StallVerdict:
    reason: str  # "silence" | "repeat_loop"
    stale_s: float
    repeat_tail: int
    sample: str = ""

    def describe(self) -> str:
        if self.reason == "silence":
            return f"shard silent for {self.stale_s:.0f}s (>= threshold)"
        return (
            f"shard emitted the same line {self.repeat_tail}x in a row "
            f"(repeat-loop); last={self.sample[:80]!r}"
        )


__all__ = [
    "LineStallDetector",
    "StallVerdict",
    "WallClock",
    "cuda_alive",
    "disk_free_gb",
    "free_gpu",
]
