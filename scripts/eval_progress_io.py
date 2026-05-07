"""Progress-file helpers shared by pod eval scripts.

The pod evaluator writes a small JSON progress file that the validator polls.
Keep this module dependency-light so it can be copied beside ``pod_eval.py`` on
remote GPU pods.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def atomic_json_write(path: str, data: Any) -> None:
    """Write JSON atomically with a one-shot error log on failure."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as handle:
            json.dump(data, handle, default=str, allow_nan=True)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        err_key = f"_atomic_json_write_err_{path}_{type(exc).__name__}_{str(exc)[:80]}"
        if not globals().get(err_key):
            globals()[err_key] = True
            print(
                f"[progress] {path} write failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


class DebouncedProgressWriter:
    """Coalesce high-frequency progress writes while preserving flush points."""

    def __init__(self, path: str, min_interval_s: float = 0.5):
        self.path = path
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._last_write = 0.0

    def write(self, data: Any, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval_s:
            return False
        atomic_json_write(self.path, data)
        self._last_write = now
        return True
