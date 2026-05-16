"""Small request-side helpers: rate-limit bucket + IP extractor + sanitisers."""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from threading import RLock

from fastapi import Request

_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "cf-connecting-ip")


def client_real_ip(request: Request) -> str:
    """Best-effort real client IP, distinguishing Caddy/CF proxies from peer IP."""
    for header in _PROXY_HEADERS:
        v = request.headers.get(header)
        if v:
            return v.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


class _Bucket:
    """Sliding-window per-key counter; thread-safe for FastAPI's worker pool."""

    def __init__(self, *, capacity: int, window_s: int = 60):
        self.capacity = int(capacity)
        self.window_s = int(window_s)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def hit(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_s
        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.capacity:
                return False
            q.append(now)
            return True


_SANITIZE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, *, max_chars: int = 8000) -> str:
    if not isinstance(text, str):
        return ""
    out = _SANITIZE_CONTROL.sub("", text)
    if len(out) > max_chars:
        out = out[:max_chars] + "…"
    return out


def join_chunks(chunks: Iterable[str]) -> str:
    return "".join(c for c in chunks if c)


def bg_refresh(fn, *, interval_s: int = 30, label: str = "bg"):
    """Decorator-like: run ``fn`` in a background daemon thread on a fixed interval."""
    import logging
    import threading

    log = logging.getLogger(f"distil.api.bg.{label}")

    def _loop():
        while True:
            try:
                fn()
            except Exception as exc:
                log.warning(f"bg refresh failed: {exc}")
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, name=f"distil-bg-{label}", daemon=True)
    t.start()
    return t
