"""Per-model timeouts + GPU/disk hygiene + CUDA-poisoned quarantine."""

from __future__ import annotations

import gc
import logging
import shutil
import time

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
