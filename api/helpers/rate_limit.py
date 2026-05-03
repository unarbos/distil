"""Rate limiting utilities."""

import time as _rate_time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = _rate_time.time()
        window_start = now - self.window_sec
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter(max_requests=60, window_sec=60)
_chat_rate_limiter = RateLimiter(max_requests=10, window_sec=60)  # Stricter for chat
# 2026-05-02 (v30.5): the OpenAI-compatible endpoint at /v1/chat/completions
# is used by agent harnesses (Flue, OpenAI Agents SDK, Vercel AI SDK,
# LangChain, …) that fire many requests per second during a single
# tool-calling loop. The 10/min cap on _chat_rate_limiter dethrones any
# real agent within seconds. Agent traffic is (a) much cheaper than
# Open-WebUI human chat (no markdown rendering, no streaming UX) and
# (b) self-throttling via tool execution time, so a more generous cap is
# appropriate. 240/min ≈ 4/sec sustained, with 5×-burst headroom — fine
# for any realistic single-agent loop, still tight enough that a runaway
# would be visible in a minute.
_openai_api_rate_limiter = RateLimiter(max_requests=240, window_sec=60)
