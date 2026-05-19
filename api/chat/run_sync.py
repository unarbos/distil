"""Non-streaming chat entry point.

Exposes :func:`run_agent_chat` which runs the OpenAI Agents SDK loop to
completion and returns an OpenAI-compatible ``chat.completion`` dict.
The streaming variant lives in :mod:`chat.streaming`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from agents import ItemHelpers, Runner, RunConfig
from agents.exceptions import AgentsException
from agents import MaxTurnsExceeded, ModelBehaviorError
from openai.types.responses import ResponseFunctionToolCall

from agent_tools import SN97AgentContext

from chat.agent_factory import _build_agent
from chat.config import (
    _AGENT_MAX_TOKENS_CEILING,
    _AGENT_MAX_TOKENS_DEFAULT,
    _AGENT_MAX_TURNS,
    _VLLM_SERVED_MODEL,
)
from chat.history import _latest_user_text, _openai_messages_to_agent_input
from chat.hooks import _StreamEvent, _StreamingHooks
from chat.preflight import _inject_preflight_context, _preflight_tools
from chat.sanitizer import (
    _STOP_TOKEN_RE,
    _split_think_blocks,
    _strip_fake_tool_output,
)

logger = logging.getLogger("distil.chat.run_sync")


@dataclass
class AgentRunOutput:
    text: str
    reasoning: str
    runtime_trace: list[str]
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]


def _summarise_run(result: Any, hooks_events: list[_StreamEvent]) -> AgentRunOutput:
    """Extract a clean (text, reasoning, trace) from a RunResult."""
    final_text = ""
    if hasattr(result, "final_output") and isinstance(result.final_output, str):
        final_text = result.final_output
    if not final_text:
        for item in reversed(getattr(result, "new_items", []) or []):
            text = ItemHelpers.extract_last_text(getattr(item, "raw_item", item))
            if text:
                final_text = text
                break

    visible, think_reasoning = _split_think_blocks(final_text)
    visible = _STOP_TOKEN_RE.sub("", visible)
    visible = _strip_fake_tool_output(visible).strip()

    # Extract the most recent non-empty python_exec stdout so we can fall
    # back to it when the model returns ONLY a code block (no prose) -- a
    # model failure mode where the user would otherwise see uninterpreted
    # Python source as the final answer.
    last_python_stdout: str | None = None
    last_python_failed = False
    for ev in hooks_events:
        if ev.kind != "thinking":
            continue
        if ev.extra.get("tool") == "python_exec" and ev.extra.get("phase") == "end":
            text_line = ev.text or ""
            if text_line.startswith("python_exec stdout:"):
                stdout = text_line[len("python_exec stdout:"):].strip()
                if stdout and stdout != "(empty)":
                    last_python_stdout = stdout
                    last_python_failed = False
            elif text_line.startswith("python_exec FAILED"):
                last_python_failed = True

    if (
        last_python_stdout is not None
        and visible.lstrip().startswith("```")
        and visible.rstrip().endswith("```")
        and "```" not in visible[3:-3]
    ):
        # The model returned only a single fenced block, with no prose
        # around it. Wrap the python output in a friendlier final answer.
        visible = (
            f"The Python sandbox ran your request and printed:\n\n"
            f"```\n{last_python_stdout[:8000]}\n```"
        )

    trace: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(result, "new_items", []) or []:
        raw = getattr(item, "raw_item", None)
        if isinstance(raw, ResponseFunctionToolCall):
            args_preview = (raw.arguments or "")[:160].replace("\n", " ")
            trace.append(f"called {raw.name}({args_preview})")
            tool_calls.append({
                "id": raw.call_id,
                "type": "function",
                "function": {"name": raw.name, "arguments": raw.arguments},
            })
    for ev in hooks_events:
        if ev.kind == "thinking" and ev.extra.get("phase") == "end":
            trace.append(ev.text)

    reasoning = think_reasoning
    if trace:
        reasoning = (
            (reasoning + "\n\nTool trace:\n" + "\n".join(f"- {t}" for t in trace)).strip()
            if reasoning
            else "Tool trace:\n" + "\n".join(f"- {t}" for t in trace)
        )

    usage_obj = getattr(result, "context_wrapper", None)
    usage_dict: dict[str, Any] | None = None
    try:
        usage = getattr(usage_obj, "usage", None) if usage_obj else None
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception:
        usage_dict = None

    return AgentRunOutput(
        text=visible,
        reasoning=reasoning,
        runtime_trace=trace,
        usage=usage_dict,
        tool_calls=tool_calls,
    )


async def run_agent_chat(
    body: dict, king_uid: int | None, king_model: str | None,
) -> dict[str, Any]:
    """Non-streaming entry. Runs the agent loop and returns an
    OpenAI-compatible ``chat.completion`` dict."""
    messages = body.get("messages") or []
    inputs = _openai_messages_to_agent_input(messages)
    if not inputs:
        return _empty_assistant_response(king_uid, king_model, "messages required")

    user_text = _latest_user_text(messages)
    preflight_blocks, preflight_trace = await _preflight_tools(
        user_text, king_uid, king_model,
    )
    inputs = _inject_preflight_context(inputs, preflight_blocks)

    max_tokens = int(body.get("max_tokens") or _AGENT_MAX_TOKENS_DEFAULT)
    max_tokens = min(max(max_tokens, 64), _AGENT_MAX_TOKENS_CEILING)
    agent = _build_agent(king_uid, king_model, max_tokens)
    ctx = SN97AgentContext(king_uid=king_uid, king_model=king_model)

    hooks_events: list[_StreamEvent] = []
    for line in preflight_trace:
        hooks_events.append(_StreamEvent(
            kind="thinking", text=f"preflight: {line}",
            extra={"phase": "preflight"},
        ))

    class _CollectHooks(_StreamingHooks):
        def __init__(self):
            class _NullQ:
                def put_nowait(self, _ev): hooks_events.append(_ev)
                async def put(self, ev): hooks_events.append(ev)
            super().__init__(_NullQ())  # type: ignore[arg-type]

    try:
        result = await Runner.run(
            agent,
            inputs,
            context=ctx,
            max_turns=_AGENT_MAX_TURNS,
            hooks=_CollectHooks(),
            run_config=RunConfig(tracing_disabled=True),
        )
    except MaxTurnsExceeded as exc:
        logger.warning("agent hit max turns: %s", exc)
        return _empty_assistant_response(
            king_uid, king_model,
            "I ran out of tool-calling turns before I could reach a final answer. "
            "Try rephrasing or breaking the question into smaller parts.",
        )
    except (AgentsException, ModelBehaviorError) as exc:
        logger.warning("agent error: %s", exc)
        return _empty_assistant_response(
            king_uid, king_model,
            f"Agent error: {type(exc).__name__}: {str(exc)[:200]}",
        )

    out = _summarise_run(result, hooks_events)
    # Prepend pre-flight trace lines into the reasoning so the dashboard
    # shows them above the SDK's tool trace.
    if preflight_trace:
        preflight_section = "Pre-flight tools:\n" + "\n".join(
            f"- {line}" for line in preflight_trace
        )
        out = AgentRunOutput(
            text=out.text,
            reasoning=(
                preflight_section + "\n\n" + out.reasoning
                if out.reasoning else preflight_section
            ),
            runtime_trace=preflight_trace + out.runtime_trace,
            usage=out.usage,
            tool_calls=out.tool_calls,
        )
    if not out.text:
        out = AgentRunOutput(
            text="(no answer; agent returned empty content)",
            reasoning=out.reasoning,
            runtime_trace=out.runtime_trace,
            usage=out.usage,
            tool_calls=out.tool_calls,
        )
    return _build_chat_completion_response(out, king_uid, king_model)


def _empty_assistant_response(
    king_uid: int | None, king_model: str | None, msg: str,
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": now,
        "model": king_model or _VLLM_SERVED_MODEL,
        "king_uid": king_uid,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": msg, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }


def _build_chat_completion_response(
    out: AgentRunOutput, king_uid: int | None, king_model: str | None,
) -> dict[str, Any]:
    now = int(time.time())
    message: dict[str, Any] = {
        "role": "assistant",
        "content": out.text,
        "tool_calls": [],
    }
    if out.reasoning:
        message["reasoning"] = out.reasoning
        message["reasoning_content"] = out.reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": now,
        "model": king_model or _VLLM_SERVED_MODEL,
        "king_uid": king_uid,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": out.usage or {
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
        },
    }


__all__ = [
    "AgentRunOutput",
    "_build_chat_completion_response",
    "_empty_assistant_response",
    "_summarise_run",
    "run_agent_chat",
]
