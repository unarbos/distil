"""Hand-written multi-turn chat agent (replaces the OpenAI Agents SDK).

* Preflight intent dispatch (regex → tool) before the LLM is called for
  cheap deterministic intents (sn97 state, model info, web search,
  history). The tool result is appended as a system message.
* Loop up to ``settings.chat_max_turns`` times: call the chat pod's
  OpenAI-compatible endpoint, look for either a python fence or a native
  tool_call in the assistant message, dispatch via
  :data:`distil.api.tools.TOOL_REGISTRY`, append the result, and continue.
* Streaming variant ``stream_agent_chat`` yields SSE-formatted chunks
  for ``GET /api/chat/stream`` and ``POST /v1/chat/completions?stream=true``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx

from distil.api.helpers import sanitize_text
from distil.api.tools import TOOL_REGISTRY, preflight_intent
from distil.settings import settings

logger = logging.getLogger("distil.api.agent")

PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.S)
TOOL_CALL_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.chat_pod_url, timeout=settings.chat_timeout_s)


def _llm_chat(
    messages: list[dict], *, stream: bool = False, model: str | None = None
) -> dict | Iterator[bytes]:
    body = {
        "model": model or settings.chat_model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": stream,
    }
    with _client() as c:
        if stream:
            resp = c.send(c.build_request("POST", "/v1/chat/completions", json=body), stream=True)
            return resp.iter_raw()
        resp = c.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()


def _dispatch_python_fence(content: str) -> str | None:
    m = PYTHON_FENCE.search(content or "")
    if not m:
        return None
    result = TOOL_REGISTRY["python_exec"](m.group(1))
    return f'<tool_result tool="python_exec">{json.dumps(result)[:4000]}</tool_result>'


def _dispatch_tool_call(content: str) -> str | None:
    m = TOOL_CALL_TAG.search(content or "")
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
        name = call.get("name")
        args = call.get("arguments") or {}
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return (
                f'<tool_result tool="{name}">{json.dumps({"error": "unknown tool"})}</tool_result>'
            )
        result = fn(**args)
        return f'<tool_result tool="{name}">{json.dumps(result)[:4000]}</tool_result>'
    except Exception as exc:
        return f'<tool_result tool="unknown">{json.dumps({"error": str(exc)})}</tool_result>'


def _maybe_preflight(messages: list[dict]) -> list[dict]:
    user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not user_msg:
        return messages
    text = user_msg.get("content") or ""
    intent = preflight_intent(text)
    if intent and intent in TOOL_REGISTRY:
        try:
            result = (
                TOOL_REGISTRY[intent](query=text)
                if intent in ("sn97_state", "web_search")
                else TOOL_REGISTRY[intent](text)
            )
            return [
                *messages,
                {"role": "system", "content": f"[preflight {intent}]\n{json.dumps(result)[:2000]}"},
            ]
        except Exception as exc:
            logger.warning(f"preflight {intent} failed: {exc}")
    return messages


def run_agent_chat(messages: list[dict], *, model: str | None = None) -> dict:
    """Run the full multi-turn agent loop and return the final assistant message."""
    msgs = _maybe_preflight(messages)
    last_assistant: dict[str, Any] = {}
    for _ in range(settings.chat_max_turns):
        resp = _llm_chat(msgs, stream=False, model=model)
        choice = resp["choices"][0]
        msg = choice.get("message") or {}
        last_assistant = msg
        content = msg.get("content") or ""
        tool_block = _dispatch_python_fence(content) or _dispatch_tool_call(content)
        if not tool_block:
            break
        msgs = [*msgs, msg, {"role": "tool", "content": tool_block}]
    last_assistant["content"] = sanitize_text(last_assistant.get("content") or "")
    return last_assistant


def stream_agent_chat(messages: list[dict], *, model: str | None = None) -> Iterator[bytes]:
    """SSE stream — yields the chat pod's SSE bytes verbatim with a final ``[DONE]``."""
    msgs = _maybe_preflight(messages)
    chunks = _llm_chat(msgs, stream=True, model=model)
    yield from chunks  # type: ignore[misc]
    yield b"data: [DONE]\n\n"
