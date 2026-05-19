"""Streaming chat SSE bridges + true token-by-token agent loop.

Two SSE shapes are exposed:

  * :func:`stream_agent_chat_openai` — OpenAI ``chat.completion.chunk``
    deltas. Used by ``/v1/chat/completions`` for OpenAI-compatible
    clients (Open WebUI, Cursor, etc).

  * :func:`stream_agent_chat_legacy` — Distil dashboard's legacy
    ``{response, thinking, ...}`` shape. Used by ``/api/chat`` (the
    dashboard's chat panel still consumes this).

Both bridges share :func:`_run_agent_streaming` which drives the SDK's
``Runner.run_streamed`` and yields :class:`_StreamEvent` (``thinking``
/ ``content`` / ``error`` / ``done``) that each SSE shape projects
into its wire format.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from agents import Runner, RunConfig
from agents.exceptions import AgentsException
from agents import MaxTurnsExceeded, ModelBehaviorError
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from openai.types.responses import ResponseTextDeltaEvent

# Reasoning-text delta type isn't part of the public ``openai.types.responses``
# re-export on every SDK version; pull it lazily so we don't crash on older
# pinned openai pkgs that don't ship it yet.
try:
    from openai.types.responses import ResponseReasoningTextDeltaEvent  # type: ignore
except ImportError:  # pragma: no cover -- older openai pkg
    ResponseReasoningTextDeltaEvent = None  # type: ignore

from agent_tools import SN97AgentContext

from chat.agent_factory import _build_agent
from chat.config import (
    _AGENT_MAX_TOKENS_CEILING,
    _AGENT_MAX_TOKENS_DEFAULT,
    _AGENT_MAX_TURNS,
    _VLLM_SERVED_MODEL,
)
from chat.history import _latest_user_text, _openai_messages_to_agent_input
from chat.hooks import _StreamEvent, _format_tool_result_for_thinking
from chat.preflight import _inject_preflight_context, _preflight_tools
from chat.sanitizer import _StreamingTextSanitizer

logger = logging.getLogger("distil.chat.streaming")


def stream_agent_chat_openai(
    body: dict, king_uid: int | None, king_model: str | None,
):
    """OpenAI-format SSE stream (``data: {...chat.completion.chunk}``)."""
    messages = body.get("messages") or []
    inputs = _openai_messages_to_agent_input(messages)
    max_tokens = int(body.get("max_tokens") or _AGENT_MAX_TOKENS_DEFAULT)
    max_tokens = min(max(max_tokens, 64), _AGENT_MAX_TOKENS_CEILING)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    base = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": king_model or _VLLM_SERVED_MODEL,
        "king_uid": king_uid,
    }

    async def generate():
        if not inputs:
            yield _openai_sse_delta(base, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return
        yield _openai_sse_delta(base, {"role": "assistant"})
        tool_call_idx = 0
        last_content_text = ""
        async for ev in _run_agent_streaming(body, king_uid, king_model, max_tokens, inputs):
            if ev.kind == "thinking":
                phase = ev.extra.get("phase") if isinstance(ev.extra, dict) else None
                if phase == "reasoning":
                    yield _openai_sse_delta(base, {"reasoning": ev.text, "reasoning_content": ev.text})
                    continue
                if phase == "start":
                    name, args_preview = _parse_tool_call_event(ev.text)
                    if not name:
                        continue
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    yield _openai_sse_delta(base, {"tool_calls": [{"index": tool_call_idx, "id": call_id, "type": "function", "function": {"name": name, "arguments": args_preview or ""}}]})
                    tool_call_idx += 1
                continue
            if ev.kind == "content":
                last_content_text = ev.text
                yield _openai_sse_delta(base, {"content": ev.text})
            elif ev.kind == "error":
                yield "data: " + json.dumps({**base, "error": {"message": ev.text}}) + "\n\n"
            elif ev.kind == "done":
                if last_content_text and not last_content_text.endswith("\n\n"):
                    pad = "\n\n" if not last_content_text.endswith("\n") else "\n"
                    yield _openai_sse_delta(base, {"content": pad})
                yield _openai_sse_delta(base, {}, finish_reason="stop")
                break
        yield "data: [DONE]\n\n"

    return generate


_TOOL_CALL_PREVIEW_RE = re.compile(r"^calling\s+([A-Za-z0-9_]+)(?:\((.*)\))?$")


def _parse_tool_call_event(text: str) -> tuple[str | None, str | None]:
    """Parse a thinking-channel ``calling <name>(<preview>)`` line into
    ``(name, args_preview)`` for a native OpenAI ``tool_calls`` delta.
    Returns ``(None, None)`` on malformed input."""
    if not text:
        return None, None
    m = _TOOL_CALL_PREVIEW_RE.match(text.strip())
    if not m:
        return None, None
    return m.group(1), (m.group(2) or "")


def stream_agent_chat_legacy(
    body: dict, king_uid: int | None, king_model: str | None,
):
    """Legacy ``/api/chat`` SSE format (``data: {response, thinking, ...}``)."""
    messages = body.get("messages") or []
    inputs = _openai_messages_to_agent_input(messages)
    max_tokens = int(body.get("max_tokens") or _AGENT_MAX_TOKENS_DEFAULT)
    max_tokens = min(max(max_tokens, 64), _AGENT_MAX_TOKENS_CEILING)

    async def generate():
        if not inputs:
            yield 'data: {"error": "messages required"}\n\n'
            yield "data: [DONE]\n\n"
            return
        async for ev in _run_agent_streaming(body, king_uid, king_model, max_tokens, inputs):
            if ev.kind == "thinking":
                yield "data: " + json.dumps({"thinking": ev.text, "delta": True}) + "\n\n"
            elif ev.kind == "content":
                yield "data: " + json.dumps({
                    "response": ev.text, "delta": True,
                    "king_uid": king_uid, "king_model": king_model,
                }) + "\n\n"
            elif ev.kind == "error":
                yield "data: " + json.dumps({"error": ev.text}) + "\n\n"
            elif ev.kind == "done":
                break
        yield "data: [DONE]\n\n"

    return generate


def _openai_sse_delta(base: dict, delta: dict, *, finish_reason: str | None = None) -> str:
    return "data: " + json.dumps({
        **base,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }) + "\n\n"


def _format_tool_called_event(item: Any) -> str:
    """Produce a single-line ``calling <tool>(<args preview>)`` string
    for the streaming reasoning channel from a ``ToolCallItem``."""
    raw = getattr(item, "raw_item", None) or item
    name = getattr(raw, "name", None) or "tool"
    args = getattr(raw, "arguments", None) or ""
    if isinstance(args, str) and args:
        preview = args[:160].replace("\n", " ")
        return f"calling {name}({preview})"
    return f"calling {name}"


def _format_tool_output_event(item: Any) -> str:
    """Produce a one-line summary of a ``ToolCallOutputItem`` for the
    streaming reasoning channel."""
    raw = getattr(item, "raw_item", None)
    output = None
    tool_name = None
    if isinstance(raw, dict):
        output = raw.get("output")
        tool_name = raw.get("name")
    else:
        output = getattr(raw, "output", None) or getattr(item, "output", None)
        tool_name = getattr(raw, "name", None)
    tool_name = tool_name or "tool"
    return _format_tool_result_for_thinking(tool_name, output)


async def _run_agent_streaming(
    body: dict,
    king_uid: int | None,
    king_model: str | None,
    max_tokens: int,
    inputs: list[dict],
):
    """Drive the agent loop with TRUE token-by-token streaming via
    ``Runner.run_streamed()``.

    Yields ``_StreamEvent`` covering:
      * ``thinking``  — pre-flight + tool start/end + reasoning deltas
      * ``content``   — sanitized model text deltas (one event per
                        underlying ``ResponseTextDeltaEvent``)
      * ``error``     — agent loop failures
      * ``done``      — terminal, always emitted

    The stream is fully event-driven via ``result.stream_events()``;
    we never buffer the model output to chunk it after-the-fact."""
    user_text = _latest_user_text(body.get("messages") or [])
    preflight_blocks, preflight_trace = await _preflight_tools(
        user_text, king_uid, king_model,
    )
    inputs = _inject_preflight_context(inputs, preflight_blocks)

    for line in preflight_trace:
        yield _StreamEvent(
            kind="thinking", text=f"preflight: {line}",
            extra={"phase": "preflight"},
        )

    agent = _build_agent(king_uid, king_model, max_tokens)
    ctx = SN97AgentContext(king_uid=king_uid, king_model=king_model)
    sanitizer = _StreamingTextSanitizer()

    # Track turn boundaries so we can prefix subsequent model turns
    # with a clear separator. The king isn't always able to follow the
    # tool-only protocol, so it sometimes regenerates its full answer
    # after a tool call -- the separator makes it obvious to the user
    # that the prior text is being superseded.
    turn_idx = 0  # 0 = first model turn, 1+ = post-tool-call turns
    pending_tool_calls = 0  # set when we see tool_called events
    saw_text_in_turn = False

    # Live streaming run; ``result.stream_events`` yields raw model
    # deltas + run-item events as the agent loop progresses through
    # multi-turn tool calling.
    try:
        result = Runner.run_streamed(
            agent, inputs,
            context=ctx,
            max_turns=_AGENT_MAX_TURNS,
            run_config=RunConfig(tracing_disabled=True),
        )
    except Exception as exc:  # noqa: BLE001
        yield _StreamEvent(
            kind="error",
            text=f"agent setup failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        yield _StreamEvent(kind="done")
        return

    try:
        async for event in result.stream_events():
            # ── Raw model deltas ────────────────────────────────────
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    delta = data.delta or ""
                    if not delta:
                        continue
                    cleaned = sanitizer.feed(delta)
                    if cleaned:
                        # Emit a turn-boundary separator on the first
                        # token of a post-tool-call turn so the user can
                        # see that this is a regenerated answer.
                        if turn_idx > 0 and not saw_text_in_turn:
                            yield _StreamEvent(
                                kind="content",
                                text="\n\n---\n\n",
                                extra={"phase": "turn_separator"},
                            )
                        saw_text_in_turn = True
                        yield _StreamEvent(kind="content", text=cleaned)
                    continue
                if (
                    ResponseReasoningTextDeltaEvent is not None
                    and isinstance(data, ResponseReasoningTextDeltaEvent)
                ):
                    rd = getattr(data, "delta", "") or ""
                    if rd:
                        yield _StreamEvent(
                            kind="thinking", text=rd,
                            extra={"phase": "reasoning"},
                        )
                    continue
                # All other raw events (item-added, content-part-added,
                # function-args deltas, etc.) are surfaced as RunItem
                # stream events below; nothing else to forward.
                continue

            # ── Run-item events: tool calls, tool outputs, messages ─
            if isinstance(event, RunItemStreamEvent):
                if event.name == "tool_called":
                    pending_tool_calls += 1
                    yield _StreamEvent(
                        kind="thinking",
                        text=_format_tool_called_event(event.item),
                        extra={"phase": "start"},
                    )
                elif event.name == "tool_output":
                    pending_tool_calls = max(pending_tool_calls - 1, 0)
                    yield _StreamEvent(
                        kind="thinking",
                        text=_format_tool_output_event(event.item),
                        extra={"phase": "end"},
                    )
                    # When we just consumed all pending tool outputs,
                    # the next model turn is starting. Reset the
                    # per-turn flags so the separator + dedup fire on
                    # the next text delta we see.
                    if pending_tool_calls == 0:
                        turn_idx += 1
                        saw_text_in_turn = False
                        # Reset sanitizer so a stop-marker hit in a
                        # broken earlier turn doesn't permanently mute
                        # the next (corrected) turn.
                        sanitizer = _StreamingTextSanitizer()
                # message_output_created / handoff_* / reasoning_item_*
                # are not surfaced -- their content already streamed
                # via the raw deltas above.
                continue

            if isinstance(event, AgentUpdatedStreamEvent):
                # Agent transitions are internal; don't show to user.
                continue
    except MaxTurnsExceeded:
        text = (
            "I ran out of tool-calling turns before I could reach a final "
            "answer. Try rephrasing or breaking the question into smaller "
            "parts."
        )
        yield _StreamEvent(kind="content", text=text)
        yield _StreamEvent(kind="done")
        return
    except (AgentsException, ModelBehaviorError) as exc:
        logger.warning("streaming agent error: %s", exc)
        yield _StreamEvent(
            kind="error",
            text=f"agent error: {type(exc).__name__}: {str(exc)[:200]}",
        )
        yield _StreamEvent(kind="done")
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("unexpected streaming error: %s", exc)
        yield _StreamEvent(
            kind="error", text=f"unexpected error: {str(exc)[:200]}",
        )
        yield _StreamEvent(kind="done")
        return

    # Drain any held-back partial chat-template token.
    tail = sanitizer.flush()
    if tail:
        yield _StreamEvent(kind="content", text=tail)
    yield _StreamEvent(kind="done")


__all__ = [
    "_TOOL_CALL_PREVIEW_RE",
    "_format_tool_called_event",
    "_format_tool_output_event",
    "_openai_sse_delta",
    "_parse_tool_call_event",
    "_run_agent_streaming",
    "stream_agent_chat_legacy",
    "stream_agent_chat_openai",
]
