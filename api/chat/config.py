"""Module config + one-time SDK bootstrap for the SN97 chat agent.

All ``DISTIL_CHAT_*`` env vars are read here at import time so any
imported helper sees the same values. The SDK setup
(``_ensure_sdk_configured``) is lazy: first call wires the OpenAI Agents
SDK at our vLLM endpoint, subsequent calls return the cached client.
"""

from __future__ import annotations

import logging
import os

from agents import (
    AsyncOpenAI,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

logger = logging.getLogger("distil.chat.config")

_VLLM_BASE_URL = os.environ.get(
    "DISTIL_CHAT_VLLM_URL", "http://127.0.0.1:8100/v1"
)
_VLLM_SERVED_MODEL = os.environ.get(
    "DISTIL_CHAT_VLLM_MODEL", "sn97-king"
)
_VLLM_API_KEY = os.environ.get("DISTIL_CHAT_VLLM_KEY", "sn97-no-auth")
_AGENT_MAX_TURNS = int(os.environ.get("DISTIL_CHAT_AGENT_MAX_TURNS", "10"))
# 2026-05-11: bumped 2048 -> 16384 so the king's <think>...</think> chain
# never truncates mid-reason. The chat pod runs vLLM with
# ``max_model_len=32768`` (see scripts/chat_pod/chat_server.py:355) so
# even a 4 K user prompt + 16 K thinking + 12 K answer fits comfortably.
# The hard ceiling enforced in the API layer is 30 K (max_model_len -
# headroom for prompt) — see ``_AGENT_MAX_TOKENS_CEILING`` below.
_AGENT_MAX_TOKENS_DEFAULT = int(os.environ.get("DISTIL_CHAT_AGENT_MAX_TOKENS", "16384"))
# Hard ceiling: leave 2 K of headroom under the chat pod's ``max_model_len``
# (32 K). Anything higher and vLLM rejects the request with a 400. We
# clamp every code path that propagates ``max_tokens`` into the agent
# loop against this number so a buggy client can't blow past it.
_AGENT_MAX_TOKENS_CEILING = int(os.environ.get("DISTIL_CHAT_AGENT_MAX_TOKENS_CEILING", "30720"))
_AGENT_DEFAULT_TEMPERATURE = float(os.environ.get("DISTIL_CHAT_AGENT_TEMPERATURE", "0.6"))


_sdk_configured = False
_vllm_client: AsyncOpenAI | None = None


def _ensure_sdk_configured() -> AsyncOpenAI:
    """Lazy one-time setup: point the SDK at our vLLM endpoint and
    disable tracing (no upstream OpenAI key)."""
    global _sdk_configured, _vllm_client
    if _sdk_configured:
        assert _vllm_client is not None  # for mypy; populated together
        return _vllm_client
    set_tracing_disabled(True)
    _vllm_client = AsyncOpenAI(base_url=_VLLM_BASE_URL, api_key=_VLLM_API_KEY)
    set_default_openai_client(_vllm_client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    _sdk_configured = True
    logger.info(
        "agent SDK configured: vllm=%s served_model=%s",
        _VLLM_BASE_URL, _VLLM_SERVED_MODEL,
    )
    return _vllm_client


__all__ = [
    "_AGENT_DEFAULT_TEMPERATURE",
    "_AGENT_MAX_TOKENS_CEILING",
    "_AGENT_MAX_TOKENS_DEFAULT",
    "_AGENT_MAX_TURNS",
    "_VLLM_API_KEY",
    "_VLLM_BASE_URL",
    "_VLLM_SERVED_MODEL",
    "_ensure_sdk_configured",
]
