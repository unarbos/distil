"""Custom vLLM model wrapper + python-fence synthetic tool-call injection.

The SDK's :class:`OpenAIChatCompletionsModel` is subclassed here to:

1. **Strip ``tools`` / ``tool_choice`` / ``parallel_tool_calls``** from
   the request *before* it reaches vLLM. The current SN97 king isn't
   fine-tuned to emit native OpenAI-format tool_calls against a
   JSON-schema tools array, and forwarding the schema causes
   pathological loops on trivial prompts. See :func:`_strip_tools_for_vllm`
   for the full story (2026-05-19 bot incident).

2. **Rewrite ``\u0060\u0060\u0060python`` fences as synthetic
   ``python_exec`` tool calls** at the SDK boundary so the multi-turn
   sandbox loop fires automatically without the model needing to emit
   native tool_calls. The per-request ``_seen_codes`` set deduplicates
   re-emissions across turns so the model recapping its prior code
   doesn't trigger an infinite tool-call loop.

All exports keep the leading underscore from the pre-split
``agent_runner`` module so the test suite (which references them by
their original names) keeps working unchanged.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agents import OpenAIChatCompletionsModel
from agents.items import ModelResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


_PY_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

# Cap on how many fence -> python_exec injections we'll synthesize per
# assistant turn. The model occasionally emits 5+ fenced blocks in a
# single message which would otherwise produce a tool-call avalanche.
_MAX_PYTHON_FENCE_INJECTIONS = 3


# ── Fence extraction + dedup ────────────────────────────────────────────────


def _extract_python_fences(text: str) -> list[str]:
    """Pull up to ``_MAX_PYTHON_FENCE_INJECTIONS`` deduped python snippets."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _PY_FENCE_RE.finditer(text):
        code = (match.group(1) or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
        if len(out) >= _MAX_PYTHON_FENCE_INJECTIONS:
            break
    return out


def _has_native_python_tool_call(items: list[Any]) -> bool:
    for item in items:
        if isinstance(item, ResponseFunctionToolCall) and item.name == "python_exec":
            return True
    return False


def _make_synthetic_python_tool_call(idx: int, code: str) -> ResponseFunctionToolCall:
    # Use a fully opaque call id (not ``call_pyfence_*``) so the model's
    # next turn can't easily mimic the format and "narrate" fake tool
    # calls back at us. The previous ``call_pyfence_<idx>_<hex>`` style
    # was reproducible by token boundary so the model had learned to
    # echo it; switching to a single random hex blob breaks that pattern.
    del idx  # kept for backwards-compat ABI on older callers
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    return ResponseFunctionToolCall(
        id=f"fc_{uuid.uuid4().hex[:12]}",
        call_id=call_id,
        arguments=json.dumps({"code": code}),
        name="python_exec",
        type="function_call",
    )


def _normalize_python_code_for_dedup(code: str) -> str:
    """Build a dedup key that ignores trivial reformats (comments, blank
    lines, leading/trailing whitespace) so the model can't trick the
    ``_seen_codes`` guard by reformatting the same broken snippet."""
    lines: list[str] = []
    for raw in (code or "").splitlines():
        # Strip end-of-line comments and trailing whitespace; skip pure
        # blank/comment lines entirely.
        stripped = re.sub(r"\s*#.*$", "", raw).rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def _inject_python_fence_tool_calls(
    items: list[Any], seen_codes: set[str] | None = None,
) -> list[Any]:
    """Walk a model output list. For each assistant ``ResponseOutputMessage``
    whose text contains ``\u0060\u0060\u0060python`` fences AND for which the model
    didn't already emit a native ``python_exec`` tool call, append synthetic
    ``ResponseFunctionToolCall`` items so the SDK's tool loop runs the
    sandbox in the next turn.

    ``seen_codes`` deduplicates across the whole run: once a snippet has
    been executed (or scheduled) we never re-execute it, even if the model
    quotes it back verbatim in a follow-up turn. Dedup is whitespace-
    and comment-insensitive so trivial reformats are still caught.
    Without this guard the model's "here's the code I ran:
    ``\u0060\u0060\u0060python ... \u0060\u0060\u0060``" recap triggers another
    injection and the agent loop never terminates.
    """
    if _has_native_python_tool_call(items):
        return items

    new_items: list[Any] = []
    appended_calls = 0
    for item in items:
        new_items.append(item)
        if isinstance(item, ResponseOutputMessage) and appended_calls < _MAX_PYTHON_FENCE_INJECTIONS:
            text_parts = []
            for c in item.content or []:
                if isinstance(c, ResponseOutputText):
                    text_parts.append(c.text or "")
            joined = "".join(text_parts)
            for code in _extract_python_fences(joined):
                if appended_calls >= _MAX_PYTHON_FENCE_INJECTIONS:
                    break
                key = _normalize_python_code_for_dedup(code)
                if not key:
                    continue
                if seen_codes is not None and key in seen_codes:
                    continue
                if seen_codes is not None:
                    seen_codes.add(key)
                appended_calls += 1
                new_items.append(_make_synthetic_python_tool_call(appended_calls, code))
    return new_items


# ── vLLM tool/tool_choice scrubbing ─────────────────────────────────────────


def _strip_tools_for_vllm(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Strip ``tools`` / ``tool_choice`` / ``parallel_tool_calls`` from the
    SDK call before it reaches vLLM.

    Why: the current SN97 king is **not** fine-tuned to emit native
    OpenAI-format ``tool_calls`` against a JSON-schema tools array.
    When the SDK forwards ``tools=[python_exec]`` to vLLM, the model
    sees a tool schema in its system prompt, tries to use it for
    EVERYTHING (even trivial prompts like "who are you"), and collapses
    into pathological loops -- e.g.::

        "I:\\n\\n---\\n\\n---\\n\\n---..."
        "I am not you,\\n\\nI am not you,..."
        " ```python\\n ```python\\n ```python..."

    All three of those failure modes were observed live on 2026-05-19
    against the king (UID 68 / arboskiller/arbosv23) on the same
    "who are you" prompt — and all three disappeared when the same
    payload was sent to vLLM directly **without** the tools array.

    The SDK's ``get_response`` / ``stream_response`` signatures pass
    ``tools`` as the 4th positional argument and ``model_settings`` as
    the 3rd, but the SDK has shifted these around between versions; we
    walk both ``args`` and ``kwargs`` defensively so a future SDK bump
    doesn't silently regress to the broken behaviour.

    The python_exec sandbox keeps working because the model triggers
    it via ```python``` fences -- ``_inject_python_fence_tool_calls``
    synthesizes ``ResponseFunctionToolCall`` items at the SDK boundary
    AFTER the model has produced its text, so the multi-turn sandbox
    loop runs identically without the tools array being visible to the
    raw chat-completions request.
    """
    from agents.models.openai_chatcompletions import (
        OpenAIChatCompletionsModel as _Parent,
    )
    import inspect
    sig = inspect.signature(_Parent.get_response)
    params = list(sig.parameters.keys())  # includes 'self' at index 0

    # Locate positional indices for the things we want to scrub.
    try:
        tools_idx = params.index("tools") - 1
    except ValueError:
        tools_idx = -1
    try:
        model_settings_idx = params.index("model_settings") - 1
    except ValueError:
        model_settings_idx = -1

    args = list(args)
    # Scrub the positional ``tools`` slot if present.
    if 0 <= tools_idx < len(args):
        args[tools_idx] = []
    # Scrub the positional ``model_settings`` slot if present.
    if 0 <= model_settings_idx < len(args):
        ms = args[model_settings_idx]
        ms = _scrub_tool_choice(ms)
        args[model_settings_idx] = ms

    # Same scrub on the kwarg path.
    if "tools" in kwargs:
        kwargs["tools"] = []
    if "model_settings" in kwargs:
        kwargs["model_settings"] = _scrub_tool_choice(kwargs["model_settings"])

    return tuple(args), kwargs


def _scrub_tool_choice(ms):
    """Strip ``tool_choice`` and ``parallel_tool_calls`` from a
    ``ModelSettings`` so vLLM doesn't echo them back as part of the
    chat-template's tool-use scaffolding. We return a NEW instance
    (or modify in place if it's a dataclass) to avoid mutating shared
    state — the Agent's settings object is shared across turns.
    """
    if ms is None:
        return ms
    # Dataclass path: dataclasses.replace if available.
    try:
        import dataclasses
        if dataclasses.is_dataclass(ms):
            kwargs = {}
            for f in dataclasses.fields(ms):
                if f.name in ("tool_choice", "parallel_tool_calls"):
                    kwargs[f.name] = None
            if kwargs:
                return dataclasses.replace(ms, **kwargs)
            return ms
    except Exception:
        pass
    # Generic attribute mutation path.
    for attr in ("tool_choice", "parallel_tool_calls"):
        if hasattr(ms, attr):
            try:
                setattr(ms, attr, None)
            except Exception:
                pass
    return ms


# ── Custom Model class ──────────────────────────────────────────────────────


class _SN97ChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat-completions model wrapper that converts the king's
    ``\u0060\u0060\u0060python`` fences into ``python_exec`` tool calls so the SDK's
    multi-turn loop fires the sandbox automatically.

    Streaming intercept: at end-of-stream we mutate the final
    ``ResponseCompletedEvent`` to inject the same synthetic tool calls.

    A per-instance ``_seen_codes`` set deduplicates injections across the
    whole agent run so the model recapping its prior code doesn't trigger
    an infinite tool-call loop. Build a fresh model per request.

    The wrapper also scrubs ``tools`` / ``tool_choice`` /
    ``parallel_tool_calls`` from the outgoing request before it reaches
    vLLM — see :func:`_strip_tools_for_vllm` for why.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_codes: set[str] = set()

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        args, kwargs = _strip_tools_for_vllm(args, kwargs)
        response = await super().get_response(*args, **kwargs)
        new_items = _inject_python_fence_tool_calls(list(response.output), self._seen_codes)
        if len(new_items) != len(response.output):
            response.output = new_items
        return response

    async def stream_response(self, *args, **kwargs):
        args, kwargs = _strip_tools_for_vllm(args, kwargs)
        async for event in super().stream_response(*args, **kwargs):
            if isinstance(event, ResponseCompletedEvent) and event.response is not None:
                merged = _inject_python_fence_tool_calls(
                    list(event.response.output or []), self._seen_codes,
                )
                if len(merged) != len(event.response.output or []):
                    event.response.output = merged
            yield event


__all__ = [
    "_MAX_PYTHON_FENCE_INJECTIONS",
    "_PY_FENCE_RE",
    "_SN97ChatCompletionsModel",
    "_extract_python_fences",
    "_has_native_python_tool_call",
    "_inject_python_fence_tool_calls",
    "_make_synthetic_python_tool_call",
    "_normalize_python_code_for_dedup",
    "_scrub_tool_choice",
    "_strip_tools_for_vllm",
]
