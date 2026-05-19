"""SDK RunHooks that feed the streaming SSE event queue.

The OpenAI Agents SDK fires :class:`RunHooks` callbacks at tool-call /
agent-step boundaries. We forward those into a per-request asyncio
Queue so the streaming bridge can interleave them as ``thinking``
deltas alongside the model's content stream.

:func:`_format_tool_result_for_thinking` produces the one-line tool-
result summary that shows up in the dashboard's reasoning panel.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agents import RunHooks, Tool

from agent_tools import SN97AgentContext


@dataclass
class _StreamEvent:
    kind: str  # "thinking", "content", "error", "done"
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class _StreamingHooks(RunHooks[SN97AgentContext]):
    """Pushes tool start/end markers to a queue so the SSE generator can
    forward them as ``thinking`` deltas while ``Runner.run`` executes."""

    def __init__(self, queue: asyncio.Queue):
        self._q = queue

    async def on_tool_start(self, context, agent, tool: Tool, *args, **kwargs):
        try:
            await self._q.put(_StreamEvent(
                kind="thinking",
                text=f"calling tool: {tool.name}",
                extra={"tool": tool.name, "phase": "start"},
            ))
        except Exception:
            pass

    async def on_tool_end(self, context, agent, tool: Tool, result: Any, *args, **kwargs):
        try:
            preview = _format_tool_result_for_thinking(tool.name, result)
            await self._q.put(_StreamEvent(
                kind="thinking",
                text=preview,
                extra={"tool": tool.name, "phase": "end"},
            ))
        except Exception:
            pass

    async def on_agent_start(self, context, agent):
        await self._q.put(_StreamEvent(
            kind="thinking",
            text=f"agent started ({agent.name})",
            extra={"phase": "agent_start"},
        ))

    async def on_agent_end(self, context, agent, output):
        await self._q.put(_StreamEvent(
            kind="thinking",
            text=f"agent finished ({agent.name})",
            extra={"phase": "agent_end"},
        ))


def _format_tool_result_for_thinking(tool_name: str, result: Any) -> str:
    """Produce a compact, user-safe summary of a tool result for the
    ``thinking`` channel. The model already gets the full result via the
    SDK; this is only for the dashboard's reasoning panel."""
    try:
        if isinstance(result, dict):
            if tool_name == "python_exec":
                stdout = (result.get("stdout") or "").strip()
                err = result.get("stderr_or_error")
                if err:
                    return f"python_exec FAILED: {str(err)[:200]}"
                return f"python_exec stdout: {stdout[:200] or '(empty)'}"
            if tool_name == "web_search":
                results = result.get("results") or []
                if not results:
                    return "web_search: no results"
                first = results[0]
                return f"web_search: {first.get('title', '?')} ({len(results)} hits)"
            if tool_name == "sn97_state":
                king = result.get("king") or {}
                return f"sn97_state: king UID={king.get('uid')} model={king.get('model')}"
            if tool_name == "model_info":
                if result.get("error"):
                    return f"model_info FAILED: {result['error']}"
                return f"model_info: {result.get('model_path')} params_b={result.get('params_b')}"
            return f"{tool_name}: {json.dumps(result)[:200]}"
        return f"{tool_name}: {str(result)[:200]}"
    except Exception:
        return f"{tool_name}: <unrenderable result>"


__all__ = [
    "_StreamEvent",
    "_StreamingHooks",
    "_format_tool_result_for_thinking",
]
