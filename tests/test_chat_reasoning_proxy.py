"""Tests for the thin transport layer in ``api/routes/chat.py``.

The orchestration / tool-calling / Python-execution stack now lives in
``api.agent_runner`` and is exercised by ``tests/test_agent_runner.py``.
This file only verifies the bits ``chat.py`` is still responsible for:
the per-turn audit-log helper, the chat-pod transport error mapping,
and the request-payload normalizer. Anything more is a misplaced test."""

import asyncio
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
API = os.path.join(ROOT, "api")
# Order matters: keep API ahead of ROUTES so the new ``api/chat/`` package
# resolves before ``api/routes/chat.py`` (the route file). The bare-name
# ``import chat`` was historically resolved to the routes file via the
# ROUTES path entry; we now spell that qualified to avoid the clash.
ROUTES = os.path.join(API, "routes")
for path in (ROUTES, API, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
# Re-shuffle so API sits ahead of ROUTES in the resolution order, otherwise
# ``import chat`` lands on ``routes/chat.py`` and shadows the ``api/chat/``
# package that ``agent_runner`` depends on.
for path in (ROUTES, API):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, ROUTES)  # last
sys.path.insert(0, API)     # first

from routes import chat as chat_route  # noqa: E402


# ── Transport-error mapping ──────────────────────────────────────────────────

def test_local_chat_post_maps_pool_timeout_to_unavailable(monkeypatch):
    """Regression for the 2026-05-09 Internal Server Error bug.

    When the chat tunnel is up but the upstream vLLM is dead, dozens of
    concurrent requests pile up on the 64-slot connection pool and the
    65th hits ``httpx.PoolTimeout`` after 5 s. Pre-fix that leaked as
    a bare 500. The contract is: any transport-level failure (incl.
    PoolTimeout, RemoteProtocolError, WriteTimeout) must be mapped to
    :class:`_ChatPodUnavailable` so callers can return the documented
    503.
    """
    import httpx

    class _FakeClient:
        def __init__(self, exc):
            self._exc = exc

        async def post(self, *args, **kwargs):
            raise self._exc

    cases = [
        httpx.PoolTimeout("pool full"),
        httpx.RemoteProtocolError("upstream RST"),
        httpx.WriteTimeout("write stalled"),
        httpx.ConnectError("refused"),
    ]
    for exc in cases:
        monkeypatch.setattr(chat_route, "_get_http_client", lambda exc=exc: _FakeClient(exc))
        try:
            asyncio.run(chat_route._local_chat_post({"messages": []}, timeout=1.0))
        except chat_route._ChatPodUnavailable as e:
            assert exc.__class__.__name__ in str(e)
        else:
            raise AssertionError(
                f"_local_chat_post did not convert {exc!r} to _ChatPodUnavailable"
            )


def test_local_chat_post_maps_5xx_response_to_unavailable(monkeypatch):
    """vLLM 5xx (or a half-open tunnel returning 502) must surface as
    503, not propagate as a 500. ``chat-keeper`` then notices the pod
    is down on the next tick and triggers a recovery."""
    class _Resp:
        status_code = 503

        def json(self):  # pragma: no cover — should not be called
            return {}

    class _FakeClient:
        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(chat_route, "_get_http_client", lambda: _FakeClient())
    try:
        asyncio.run(chat_route._local_chat_post({"messages": []}, timeout=1.0))
    except chat_route._ChatPodUnavailable as e:
        assert "503" in str(e)
    else:
        raise AssertionError(
            "_local_chat_post did not surface 503 as _ChatPodUnavailable"
        )


# ── Payload normalizer ───────────────────────────────────────────────────────

def test_normalize_chat_payload_pins_served_model_and_enables_thinking():
    """Thinking must default ON so the chat reflects the king's actual
    reasoning capability (matches the held-out benchmark and bench
    axes). ``max_tokens`` defaults to 16 384 (16 K) — the v32.1
    reasoning-uncap default — so the king's <think>...</think> chain
    has full room to reason without truncation. The chat pod's vLLM
    runs with max_model_len=32 768, so 16 K leaves ample headroom for
    the prompt + thinking + answer. The served-model is pinned and a
    formatting system prompt is injected if the client didn't supply
    one."""
    payload = {
        "model": "client-supplied-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = chat_route._normalize_chat_payload(payload)
    assert out["model"] == chat_route.CHAT_POD_SERVED_MODEL
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert out["max_tokens"] == 16384
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert sys_msgs, "missing system formatting prompt was not injected"


def test_normalize_chat_payload_honours_explicit_thinking_off():
    """A client that explicitly opts out of thinking (low-latency
    use cases) must still get the no-think rendering."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = chat_route._normalize_chat_payload(payload)
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_normalize_chat_payload_preserves_existing_system_prompt():
    """If the client supplies a system prompt we must not stomp on it."""
    payload = {
        "messages": [
            {"role": "system", "content": "BE TERSE."},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 256,
    }
    out = chat_route._normalize_chat_payload(payload)
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "BE TERSE."
    assert out["max_tokens"] == 256


# ── Audit log: repeated-substring detector ───────────────────────────────────

def test_detect_repeated_substring_finds_duplicates():
    repeats = chat_route._detect_repeated_substring("abc" * 200)
    assert repeats > 0


def test_detect_repeated_substring_handles_short_input():
    assert chat_route._detect_repeated_substring("") == 0
    assert chat_route._detect_repeated_substring("short") == 0


def test_extract_message_content_promotes_thinking_when_content_empty():
    """Some vLLM builds emit an empty `content` and stuff the answer
    into `reasoning` instead. The transport helper must surface that
    as the visible content so the user is not left with nothing."""
    content, thinking = chat_route._extract_message_content(
        {"content": "", "reasoning": "the answer is 42"}
    )
    assert content == "the answer is 42"
    assert thinking is None


def test_extract_message_content_keeps_thinking_when_content_present():
    content, thinking = chat_route._extract_message_content(
        {"content": "the answer is 42", "reasoning": "step-by-step trace"}
    )
    assert content == "the answer is 42"
    assert thinking == "step-by-step trace"


def test_estimate_input_tokens_string_content():
    msgs = [{"role": "user", "content": "abcabcabc"}]
    assert chat_route._estimate_input_tokens(msgs) == (9 // 3) + 8


def test_estimate_input_tokens_handles_multimodal_parts():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "abcabc"},
            {"type": "text", "text": "xyzxyz"},
        ]},
    ]
    assert chat_route._estimate_input_tokens(msgs) == (6 // 3) + (6 // 3) + 8


def test_clamp_passes_small_payload_through():
    msgs = [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"},
    ]
    trimmed, clamped = chat_route._clamp_for_context_budget(msgs, 1024)
    assert trimmed == msgs
    assert clamped == 1024


def test_clamp_lowers_max_tokens_for_long_input():
    big_user = "x" * (28000 * 3)
    msgs = [{"role": "user", "content": big_user}]
    trimmed, clamped = chat_route._clamp_for_context_budget(msgs, 16384)
    assert trimmed == msgs, "single-turn message must never be dropped"
    estimated = chat_route._estimate_input_tokens(msgs)
    overhead = chat_route._CHAT_AGENT_OVERHEAD_TOKENS
    assert estimated + clamped + overhead <= chat_route._CHAT_MODEL_MAX_LEN
    assert clamped >= chat_route._CHAT_MIN_OUTPUT_TOKENS


def test_clamp_trims_oldest_history_when_needed():
    sys_msg = {"role": "system", "content": "system rules"}
    big = "x" * (8000 * 3)
    msgs = [
        sys_msg,
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "what did I just ask?"},
    ]
    trimmed, _ = chat_route._clamp_for_context_budget(msgs, 4096)
    assert trimmed[0] is sys_msg, "system prompt must always survive"
    assert trimmed[-1] == msgs[-1], "latest user turn must always survive"
    assert len(trimmed) < len(msgs), "stale history must be dropped"


def test_clamp_never_drops_only_user_message():
    huge = "x" * (60000 * 3)
    msgs = [{"role": "user", "content": huge}]
    trimmed, clamped = chat_route._clamp_for_context_budget(msgs, 16384)
    assert trimmed == msgs
    assert clamped == chat_route._CHAT_MIN_OUTPUT_TOKENS


def test_clamp_handles_invalid_max_tokens():
    msgs = [{"role": "user", "content": "hi"}]
    _, c1 = chat_route._clamp_for_context_budget(msgs, None)
    _, c2 = chat_route._clamp_for_context_budget(msgs, "abc")
    assert c1 == c2
    assert c1 >= chat_route._CHAT_MIN_OUTPUT_TOKENS
