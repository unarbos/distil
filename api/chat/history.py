"""Conversation translation: OpenAI messages -> SDK Agent input.

The OpenAI Agents SDK expects ``{role, content}`` dicts as input. The
prod chat router forwards the raw ``messages`` array from the OpenAI
chat-completions request, which can contain system messages, multi-
modal content lists, derail-prone assistant turns from legacy clients,
and arbitrary long histories. We normalize all of that here.

Trimming rules (in order):

  1. Keep only the last :data:`_HISTORY_TURN_CAP` messages.
  2. Drop system messages (the Agent's own ``instructions`` IS the
     system prompt — a second system message gets merged or lost on
     some providers).
  3. Drop any assistant turn that contains a :data:`_DERAIL_MARKERS`
     substring (legacy tool names the king used to reference from an
     earlier prompt-engineering iteration).
  4. Strip ``<think>...</think>`` and ``Runtime trace...`` lines from
     assistant turns so the model isn't trying to extend its own
     hidden reasoning.
  5. Cap each message at :data:`_HISTORY_CHAR_CAP_PER_MSG` chars.
"""

from __future__ import annotations

import re

_HISTORY_CHAR_CAP_PER_MSG = 4000
_HISTORY_TURN_CAP = 12  # last N messages (user/assistant pairs)
_DERAIL_MARKERS = (
    "Use the tool ",
    "get_model_info",
    "get_subnet_overview",
    "get_leaderboard",
    "knowledge base",
    "Persistent File System",
    "Pyodide",
)
_CLIENT_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_RUNTIME_TRACE_LINE_RE = re.compile(
    r"(?im)^Runtime trace, not hidden model reasoning:.*(?:\n|$)"
)


def _strip_displayed_thinking(text: str) -> str:
    text = _CLIENT_THINK_RE.sub("", text or "")
    text = _RUNTIME_TRACE_LINE_RE.sub("", text)
    return text.strip()


def _openai_messages_to_agent_input(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style ``messages`` to the SDK's input format.

    Drops system messages (the agent's own ``instructions`` is the system
    prompt), trims displayed-thinking from prior assistant turns, and caps
    history size so we don't ship the whole transcript on every turn.
    """
    if not isinstance(messages, list):
        return []
    out: list[dict] = []
    for msg in messages[-_HISTORY_TURN_CAP:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            content = "\n".join(parts)
        if not isinstance(content, str):
            continue
        if role == "assistant":
            content = _strip_displayed_thinking(content)
            if not content or any(marker in content for marker in _DERAIL_MARKERS):
                continue
        content = content[:_HISTORY_CHAR_CAP_PER_MSG]
        out.append({"role": role, "content": content})
    return out


def _latest_user_text(messages: list[dict]) -> str:
    """Pull the last user message's text content as a flat string."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                return "\n".join(parts)
    return ""


__all__ = [
    "_CLIENT_THINK_RE",
    "_DERAIL_MARKERS",
    "_HISTORY_CHAR_CAP_PER_MSG",
    "_HISTORY_TURN_CAP",
    "_RUNTIME_TRACE_LINE_RE",
    "_latest_user_text",
    "_openai_messages_to_agent_input",
    "_strip_displayed_thinking",
]
