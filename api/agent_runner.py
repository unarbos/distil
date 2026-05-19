"""SN97 chat agent runner.

Wraps the OpenAI Agents SDK so the chat surface gets:
- typed function tools (see ``agent_tools.py``)
- automatic multi-turn tool execution
- ``ContextWrapper``-based per-request state (king info, sandbox budget)
- streaming SSE bridge for ``/v1/chat/completions`` and ``/api/chat``

The king model (vLLM @ 127.0.0.1:8100) is not yet fine-tuned to emit
native OpenAI ``tool_calls``. Instead it writes fenced ``\u0060\u0060\u0060python``
blocks. ``_SN97ChatCompletionsModel`` converts those fences into synthetic
``python_exec`` tool calls so the SDK's normal multi-turn loop fires the
sandbox and feeds the real stdout back to the model in the next turn.

Streaming model:
- The non-streaming ``run_agent_chat`` (used by ``/api/chat`` non-stream
  + the legacy dashboard sync path) drives the agent via ``Runner.run``.
- The streaming bridges (``stream_agent_chat_openai`` for Open-WebUI on
  ``/v1/chat/completions``; ``stream_agent_chat_legacy`` for the v2
  dashboard's SSE) drive ``Runner.run_streamed`` for true token-by-
  token deltas.
- Tool start events surface as native OpenAI ``tool_calls`` deltas on
  the OpenAI bridge so Open-WebUI renders them as code blocks + tool-
  output pills; on the legacy bridge they go through a separate
  ``thinking`` channel for the dashboard's collapsible side panel.
- Only true model ``<think>...</think>`` deltas reach Open-WebUI's
  ``reasoning_content`` field. Pre-flight + tool-trace metadata stays
  off that channel because Open-WebUI inlines accumulated reasoning
  into a ``<details type="reasoning">`` block at stream end with HTML-
  entity-escaped content, and any ``>`` / ``"`` / ``'`` characters in
  trace lines would render as literal ``&gt;`` / ``&quot;`` / ``&#x27;``
  noise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import (
    Agent,
    AsyncOpenAI,
    ItemHelpers,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    RunHooks,
    Tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.exceptions import AgentsException
from agents.items import ModelResponse
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
# Reasoning-text delta type isn't part of the public ``openai.types.responses``
# re-export on every SDK version; pull it lazily so we don't crash on older
# pinned openai pkgs that don't ship it yet.
try:
    from openai.types.responses import ResponseReasoningTextDeltaEvent  # type: ignore
except ImportError:  # pragma: no cover -- older openai pkg
    ResponseReasoningTextDeltaEvent = None  # type: ignore

from agent_tools import (
    SN97AgentContext, python_exec, system_prompt_for,
)

logger = logging.getLogger("distil.agent_runner")

# ── Module config ────────────────────────────────────────────────────────────

_VLLM_BASE_URL = os.environ.get(
    "DISTIL_CHAT_VLLM_URL", "http://127.0.0.1:8100/v1"
)
_VLLM_SERVED_MODEL = os.environ.get(
    "DISTIL_CHAT_VLLM_MODEL", "sn97-king"
)
_VLLM_API_KEY = os.environ.get("DISTIL_CHAT_VLLM_KEY", "sn97-no-auth")
_AGENT_MAX_TURNS = int(os.environ.get("DISTIL_CHAT_AGENT_MAX_TURNS", "10"))
# 2026-05-11: bumped 2048 → 16384 so the king's <think>...</think> chain
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

_PY_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
# Patterns that decide which non-python tools to pre-flight. The current
# king isn't fine-tuned to emit native ``tool_calls``, so it cannot trigger
# web_search / sn97_state / model_info itself. Instead we detect intent in
# the user's text and call the helpers up-front, then inject the results
# as an authoritative system message so the agent loop has them as ground
# truth. python_exec is driven by ``\u0060\u0060\u0060python`` fences inside the
# model's output and stays in the SDK loop so the model can iterate.
_SN97_RE = re.compile(
    r"\b(sn\s?97|sn-97|subnet|king|leaderboard|miner|uid|eval|round|"
    r"validator|arbos|distil)\b",
    re.IGNORECASE,
)
_WEB_SEARCH_RE = re.compile(
    r"(?:"
    r"\b(?:search\s+(?:the\s+)?web|web\s+search|look\s+up|google\s+for|google\s+the|"
    r"duckduckgo|find\s+me|fetch\s+(?:the\s+)?(?:latest|news|results?))\b"
    r"|\b(?:right\s+now|today|tonight|this\s+(?:morning|afternoon|evening|week|month|year))\b"
    r"|\b(?:latest|breaking|current|today'?s|recent|live|real[-\s]?time)\b"
    r"|\b(?:weather|forecast|temperature|stock\s+price|share\s+price|exchange\s+rate|"
    r"headline[s]?|news|score)\b"
    r"|\b(?:bitcoin|btc|ethereum|eth|tesla|gold|silver|oil|sp500|s&p\s*500)\b"
    r"|\bprice\s+of\s+\w+\b|\b\w+\s+price\b"
    r")",
    re.IGNORECASE,
)
_MODEL_PATH_RE = re.compile(
    r'"model_path"\s*:\s*"([^"]+)"|'
    r"\b([A-Za-z0-9._-]{2,40}/[A-Za-z0-9._-]{2,100})\b"
)
_FAKE_TOOL_OUTPUT_RE = re.compile(
    # Matches a literal label like "**Tool Output:**" / "**Output:**" /
    # "Sandbox stdout:" / "**Python Sandbox Output:**" -- with various
    # leading markdown decoration -- followed by EITHER a fenced block on
    # the next line OR the value inline on the same line. Both are
    # leftovers from the model "narrating" what the runtime printed; the
    # runtime trace already shows the user that the code ran, so we drop
    # the label+payload pair.
    #
    # The regex is anchored with ``^`` at line start (MULTILINE) to avoid
    # eating bare "Output:" / "Result:" mid-sentence in legitimate prose.
    r"(?im)"
    r"^[*_]{0,3}\s*"  # line start, optional ** / __ decoration
    r"(?:"
    # Form 1: 1-3 explicit prefix words like Tool / Sandbox / Python
    # Sandbox / Tool Python Sandbox followed by Output / Result / stdout
    r"(?:(?:Tool|Sandbox|Runtime|Execution|Python|Code)\s+){1,3}"
    r"(?:Output|Result|stdout|response)"
    # Form 2: bare label as long as it stands at line start
    r"|(?:Output|Result|stdout|Returns?|Yields?)"
    r")"
    r"\s*[:;\-]?\s*"
    r"[*_]{0,3}\s*"
    r"(?:"
    # Form A: label on its own line, fenced block underneath.
    r"\n+```[a-z0-9_-]*\n.*?```\s*"
    # Form B: label immediately followed by value on the same line.
    r"|[ \t]+[^\n]{1,1500}(?:\n|$)"
    # Form C: label on its own line, then a single short value on the
    # next line (numbers, single tokens, quoted strings). Strictly
    # length-capped to avoid eating real prose; only drops things like
    # "**Output:**\n2491\n" or "Result:\nhi\n".
    r"|\n+[^\n`]{1,80}(?:\n\n|\n*$)"
    r")",
    re.DOTALL | re.MULTILINE,
)
_RUNNING_CODE_FILLER_RE = re.compile(
    r"(?im)^\s*(?:Running\s+(?:this\s+)?code\b|Executing\b|Let me\s+run\b|"
    r"I(?:'ll|\s+will)\s+run\b)[^\n]{0,160}\.{0,3}\s*$",
)
# Catch the model "narrating" tool calls back at us. The Hermes / Kimi
# chat templates expose synthetic tool-call ids and the ``## Return of
# <call_id>`` framing in the model's context, and weaker chat models
# learn to reproduce both as plain text. Drop those fragments wholesale
# -- the SDK already executed the real tool and we render it from
# ``runtime_trace`` separately.
_FAKE_TOOL_CALL_NARRATION_RE = re.compile(
    r"(?im)"
    r"(?:"
    # Form 1: literal "## Return of call_<...>" / "tool## Return of ..."
    r"(?:^|\n)\s*(?:tool)?#{1,4}\s*Return\s+of\s+(?:call|fc)_[A-Za-z0-9_]+[^\n]*"
    # Form 2: a fabricated tool-call invocation like
    # ``call_xxx{"code": "..."}`` or ``fc_xxx{...}``
    r"|(?:^|\n)\s*(?:call|fc)_[A-Za-z0-9_]+\s*\{[^\n]*\}"
    # Form 3: bracketed tool-message like ``<tool>{"name": ...}</tool>``
    r"|<\s*/?\s*tool[^>]*>"
    r")",
)
# Once the model emits ANY of those tool-narration fragments, the rest
# of the message is almost always a stream of fabricated stdouts and
# self-quoting. Truncate from the first such marker to the end so the
# user gets a clean answer instead of the model pretending to be the
# orchestrator.
_FAKE_TOOL_NARRATION_TRUNC_RE = re.compile(
    r"(?:tool##|##\s*Return\s+of|(?:^|\n)\s*(?:call|fc)_[a-z0-9]{6,}\s*\{)",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(
    # Match the standard XML-style ``<think>...</think>`` block AND the
    # Kimi-K2.6 chat-template variant which uses Unicode triangles
    # (``◁think▷ ... ◁/think▷``) instead of ASCII brackets. The Kimi
    # template emits these when ``enable_thinking=True`` is forwarded
    # via ``chat_template_kwargs`` and the upstream reasoning parser
    # only catches ``<think>`` literals — so the triangle variant
    # would otherwise leak into the user-visible answer.
    r"(?:<think\b[^>]*>(.*?)</think>|◁think▷(.*?)◁/think▷)",
    re.IGNORECASE | re.DOTALL,
)

_MAX_PYTHON_FENCE_INJECTIONS = 3


# ── One-time SDK config ──────────────────────────────────────────────────────

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


# ── Custom Model: rewrite python fences as synthetic tool calls ─────────────

def _strip_fake_tool_output(text: str) -> str:
    """Drop fabricated 'Tool Output:'/'Sandbox stdout:' fenced blocks AND
    'tool## Return of call_xxx' tool-call narration the model invents
    around its real ``\u0060\u0060\u0060python`` block. The SDK always shows the
    truth in ``runtime_trace``; the model's "let me also pretend to be
    the orchestrator" output is pure noise."""
    if not text:
        return text
    # Hard cut: if the model started narrating tool calls, the rest of
    # the message is fabricated. Truncate from the first marker so we
    # ship the clean prefix only.
    trunc = _FAKE_TOOL_NARRATION_TRUNC_RE.search(text)
    if trunc:
        text = text[:trunc.start()].rstrip()
    cleaned = _FAKE_TOOL_OUTPUT_RE.sub("", text)
    cleaned = _FAKE_TOOL_CALL_NARRATION_RE.sub("", cleaned)
    cleaned = _RUNNING_CODE_FILLER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _split_think_blocks(text: str) -> tuple[str, str]:
    """Promote inline ``<think>...</think>`` (or Kimi ``◁think▷...◁/think▷``)
    to a reasoning channel."""
    if not text:
        return text, ""
    lower = text.lower()
    if "<think" not in lower and "◁think▷" not in lower:
        return text, ""
    think_chunks: list[str] = []
    for m in _THINK_BLOCK_RE.finditer(text):
        # Group 1 = ``<think>...</think>`` body, Group 2 = ``◁think▷...◁/think▷`` body
        body = (m.group(1) or m.group(2) or "").strip()
        if body:
            think_chunks.append(body)
    visible = _THINK_BLOCK_RE.sub("", text).strip()
    reasoning = "\n\n".join(think_chunks).strip()
    return visible, reasoning


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


_STOP_TOKEN_RE = re.compile(
    # Common chat-template end-of-message markers that occasionally leak
    # past the vLLM tool/reasoning parsers. Strip them so the dashboard
    # doesn't render them as visible text. Pattern accepts:
    # * any bracketed `<|...|>` token whose body is alphanumeric /
    #   underscore (covers Hermes, Qwen, Kimi, Llama-3, Mistral
    #   templates plus the synthetic ``<|tool_call_argument_begin|>``
    #   / ``<|im_middle|>`` markers that bleed through tool-call
    #   parsing).
    # * unmatched Kimi-K2.6 triangle markers ``◁think▷`` / ``◁/think▷``
    #   / ``◁|tool_calls_section_begin|▷`` that occasionally leak when
    #   the model produces a half-open block (one of them gets
    #   stripped by ``_split_think_blocks``, the other doesn't).
    r"<\|[A-Za-z0-9_\-]+\|>|◁/?[A-Za-z0-9_\-|]+▷",
)


def _sanitize_assistant_text(text: str) -> str:
    """Strip ``<think>`` blocks, fake tool-output blocks, and stray
    chat-template stop tokens, then collapse blank lines so the
    user-facing text is clean."""
    visible, _ = _split_think_blocks(text or "")
    visible = _STOP_TOKEN_RE.sub("", visible)
    return _strip_fake_tool_output(visible).strip()


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


# ── Hooks → SSE queue ────────────────────────────────────────────────────────

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


# ── Conversation translation ────────────────────────────────────────────────

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


def _build_agent(king_uid: int | None, king_model: str | None, max_tokens: int) -> Agent[SN97AgentContext]:
    """Construct a fresh Agent for one chat request.

    A fresh Agent per request keeps the system prompt up-to-date with the
    live king (king info changes every eval round) and avoids any
    cross-request state on the Agent object.

    ``python_exec`` is the ONLY SDK tool exposed. The model triggers it
    by writing fenced ``\u0060\u0060\u0060python`` blocks; the custom model
    wrapper rewrites those into synthetic ``python_exec`` tool calls.
    Web search, SN97 live state and HuggingFace model info are pre-flighted
    by the runtime (see ``_preflight_tools``) and injected as ground-truth
    context; we deliberately don't surface them as SDK tools because the
    current king isn't fine-tuned to emit native ``tool_calls`` and any
    extra tool in the schema (e.g. summarise_history) just nudges it into
    invoking the wrong helper for trivial conversational turns.
    """
    _ensure_sdk_configured()
    client = _ensure_sdk_configured()
    model = _SN97ChatCompletionsModel(model=_VLLM_SERVED_MODEL, openai_client=client)
    return Agent[SN97AgentContext](
        name="sn97-chat",
        instructions=system_prompt_for(king_uid, king_model),
        model=model,
        model_settings=ModelSettings(
            temperature=_AGENT_DEFAULT_TEMPERATURE,
            top_p=0.9,
            max_tokens=max_tokens,
            tool_choice="auto",
            parallel_tool_calls=False,
            # Anti-loop: gentle repetition penalty so the king is less
            # likely to collapse into a "from math import asinh / from
            # math import atanh / ..." infinite repeat when its tool
            # result disagrees with its earlier guess. Kept SMALL so the
            # model's natural tool-using vocabulary ("I'll compute X
            # with python: `python_exec`") still flows.
            frequency_penalty=0.15,
            presence_penalty=0.0,
            # vLLM-specific knobs forwarded via ``extra_body`` so the
            # OpenAI client passes them as body fields instead of
            # validating them as Python kwargs (which would 400):
            #
            # * ``repetition_penalty=1.05`` — enough to break adversarial
            #   loops without hurting normal prose generation.
            # * ``chat_template_kwargs.enable_thinking=False`` —
            #   forces the king to answer directly in the content
            #   channel. We previously toggled this ON so the
            #   ``distil_kimi`` reasoning parser would split the
            #   ``<think>…</think>`` chain into a separate
            #   ``reasoning_content`` field, but: (a) the current king
            #   isn't fine-tuned on that template and produced
            #   degenerate "I:\n\n---\n\n---..." filler loops with
            #   thinking forced ON; (b) the reasoning parser didn't
            #   recognise the king's actual ``◁/think▷`` Kimi triangle
            #   tags anyway, so the "Thinking" pane stayed empty.
            #   Direct content output is the lesser of two evils until
            #   the chat pod re-deploys to a king whose template
            #   matches the parser. See also ``_THINK_BLOCK_RE`` below,
            #   which still strips any inline ``<think>`` block the
            #   model emits voluntarily.
            # * ``stop=["\\n```\\n"]`` + ``include_stop_str_in_output=True`` —
            #   closes the model off once it finishes a fenced code
            #   block. Without it the king regularly free-types
            #   "Tool Output: 17891344..." after the closing ``` and
            #   the SDK has to chase down the fake-tool-result text
            #   with a sanitizer pass. ``include_stop_str_in_output``
            #   is CRITICAL — without it vLLM swallows the closing
            #   ``` and ``_inject_python_fence_tool_calls`` can no
            #   longer match the fence (the dedup regex requires both
            #   opening and closing ```), so the synthetic
            #   ``python_exec`` call is never created and the SDK
            #   returns the code-only message to the user instead of
            #   looping back with the sandbox stdout.
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": ["\n```\n"],
                "include_stop_str_in_output": True,
            },
        ),
        tools=[python_exec],
    )


# ── Pre-flight non-python tools ─────────────────────────────────────────────
#
# The current king is NOT fine-tuned to emit native ``tool_calls``, so it
# can't invoke web_search / sn97_state / model_info on its own. We detect
# user intent up-front and call the underlying helpers, then inject the
# JSON results as an authoritative system message so the agent loop sees
# the data as ground truth. After the king is upgraded to emit native
# tool_calls these pre-flights become redundant -- the SDK loop will
# trigger the same helpers via ``tool_choice="auto"``.

def _latest_user_text(messages: list[dict]) -> str:
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


_SEARCH_PREFIX_RE = re.compile(
    r"^\s*("
    r"search\s+(?:the\s+)?web\s+for|web\s+search\s+for|"
    r"look\s+up|google\s+for|google\s+the|google|"
    r"find\s+me|fetch\s+(?:the\s+)?(?:latest|news|results?\s+(?:about|on|for))|"
    r"(?:can\s+you|could\s+you|please|tell\s+me)\s+"
    r"(?:search|find|look\s+up|google|fetch|tell\s+me)|"
    r"what(?:'s|\s+is)|what\s+are|what\s+were|"
    r"how\s+much\s+(?:is|does|costs?)|how\s+do|how\s+many|"
    r"who\s+(?:is|are|was|were)|where\s+(?:is|are|was|were)|"
    r"when\s+(?:is|are|was|were|did|does|do)|"
    r"explain|tell\s+me\s+about"
    r")\s+",
    re.IGNORECASE,
)


def _normalize_search_query(user_text: str) -> str:
    """Compact a chatty user prompt into a search-friendly query.

    DuckDuckGo's HTML endpoint returns zero hits when the query is too
    long, has trailing punctuation, or contains compound clauses like
    ``"X, and how much is Y worth?"``. We strip leading "what is",
    trailing punctuation, and chop everything after the first comma so
    the search engine sees the user's primary intent.
    """
    q = (user_text or "").strip()
    # Take only the first clause (before a comma or semicolon) -- the
    # rest is usually a follow-up like "and convert to USD" that the
    # search engine doesn't need.
    for sep in (",", ";", " and ", "?"):
        idx = q.find(sep) if sep != "?" else q.rfind(sep)
        if idx > 0 and (sep != "?" or idx == len(q) - 1):
            q = q[:idx]
            break
    q = q.strip()
    # Strip leading interrogative / search-request prefix.
    q = _SEARCH_PREFIX_RE.sub("", q).strip()
    # Strip trailing punctuation.
    q = q.rstrip("?!.,;:").strip()
    # Cap to ~12 words; longer queries usually have noise.
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    return q or (user_text or "").strip()


async def _preflight_tools(
    user_text: str, king_uid: int | None, king_model: str | None,
) -> tuple[list[str], list[str]]:
    """Run the non-python tools that the model can't invoke itself.

    Returns ``(context_blocks, runtime_trace)`` where ``context_blocks``
    is a list of pretty-printed result strings to inject as a system
    message and ``runtime_trace`` is human-readable lines for the
    reasoning panel.
    """
    blocks: list[str] = []
    trace: list[str] = []
    # Captured by the sn97_state branch below; used as a fallback
    # candidate for the implicit ``model_info`` pre-flight when the
    # user asks about size/params of "the king" without naming a path.
    king_model_path: str = ""
    if not user_text:
        return blocks, trace

    # web_search
    if _WEB_SEARCH_RE.search(user_text):
        try:
            from agent_tools import _parse_duckduckgo_html, _WEB_HEADERS, _WEB_TIMEOUT
            import httpx as _httpx
            import urllib.parse as _up
            q = _normalize_search_query(user_text)
            url = "https://duckduckgo.com/html/?" + _up.urlencode({"q": q})
            async with _httpx.AsyncClient(
                timeout=_WEB_TIMEOUT, headers=_WEB_HEADERS, follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                results = _parse_duckduckgo_html(resp.text, query=q, limit=5)
            blocks.append(
                "WEB_SEARCH_RESULT:\n" + json.dumps(
                    {"query": q, "results": results}, indent=2,
                )[:4000]
            )
            trace.append(f"web_search ({len(results)} hits)")
        except Exception as exc:
            blocks.append(f"WEB_SEARCH_ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            trace.append(f"web_search FAILED: {type(exc).__name__}")

    # sn97_state
    if _SN97_RE.search(user_text):
        try:
            from state_store import (
                eval_progress, h2h_latest, read_cache, top4_leaderboard,
                uid_hotkey_map,
            )
            top4 = top4_leaderboard() or {}
            king = dict(top4.get("king") or {})
            if king_uid is not None:
                king.setdefault("uid", king_uid)
            if king_model:
                king.setdefault("model", king_model)
            contenders = [dict(c) for c in (top4.get("contenders") or [])[:4]]
            progress = eval_progress() or {}
            h2h = h2h_latest() or {}
            payload: dict[str, Any] = {
                "king": king,
                "contenders": contenders,
                "eval_progress": {
                    "active": bool(progress.get("active") or progress.get("phase") not in (None, "", "idle")),
                    "phase": progress.get("phase") or progress.get("stage") or progress.get("status"),
                    "students_done": progress.get("students_done"),
                    "students_total": progress.get("students_total"),
                    "block": progress.get("completed_block") or h2h.get("block"),
                },
            }
            uid_match = re.search(r"\buid\s*[:=#]?\s*(\d{1,3})\b", user_text, re.IGNORECASE)
            if uid_match:
                with suppress(Exception):
                    uid = int(uid_match.group(1))
                    hotkey = uid_hotkey_map().get(str(uid))
                    commitments_data = read_cache("commitments", {}) or {}
                    commitments = commitments_data.get("commitments", commitments_data)
                    model = None
                    if hotkey and isinstance(commitments, dict):
                        c = commitments.get(hotkey) or {}
                        if isinstance(c, dict):
                            model = c.get("model") or c.get("repo")
                    payload["uid_lookup"] = {"uid": uid, "hotkey": hotkey, "model": model}
            blocks.append("SN97_LIVE_STATE:\n" + json.dumps(payload, indent=2)[:4000])
            trace.append(f"sn97_state (king UID={king.get('uid')})")
            # Capture king model path for the implicit ``model_info``
            # pre-flight below ("what's the king's param count?" etc.)
            with suppress(Exception):
                kmp = king.get("model")
                if isinstance(kmp, str) and "/" in kmp:
                    king_model_path = kmp
        except Exception as exc:
            trace.append(f"sn97_state FAILED: {type(exc).__name__}")

    # model_info -- explicit request OR implicit "size/params/bytes of
    # current king" follow-up that piggybacks on a prior sn97_state hit.
    explicit_model_info = (
        "get_model_info" in user_text
        or "model_path" in user_text
        or "tell me about" in user_text.lower()
    )
    implicit_king_size = bool(king_model_path) and bool(
        re.search(
            r"\b(size|params?|parameters?|bytes?|gb|mb|memory|"
            r"quantiz(?:e|ed|ation)|footprint|vram|disk)\b",
            user_text, re.IGNORECASE,
        )
    )
    if explicit_model_info or implicit_king_size:
        candidate_path = ""
        m = _MODEL_PATH_RE.search(user_text)
        if m:
            candidate_path = (m.group(1) or m.group(2) or "").strip()
        if not candidate_path and implicit_king_size:
            candidate_path = king_model_path or ""
        if "/" in candidate_path and len(candidate_path) >= 3:
            with suppress(Exception):
                from external import get_model_info as fetch_model_info_data
                info = fetch_model_info_data(candidate_path)
                if isinstance(info, dict):
                    blocks.append(
                        "MODEL_INFO_RESULT:\n" + json.dumps(
                            {**info, "model_path": candidate_path}, indent=2,
                        )[:2000]
                    )
                    trace.append(f"model_info ({candidate_path})")
    return blocks, trace


# ── Public entry points ─────────────────────────────────────────────────────

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


def _inject_preflight_context(
    inputs: list[dict], blocks: list[str],
) -> list[dict]:
    """Prepend a synthetic user turn carrying the pre-flight tool results.

    We use ``role=user`` (not ``system``) because the SDK passes the
    Agent's ``instructions`` as the only true system prompt; a second
    system message would either be merged or lost on some providers.
    Wrapping the data in a clearly delineated user turn keeps it visible
    to the model without poisoning the conversation history if the agent
    is long-running.
    """
    if not blocks:
        return inputs
    header = (
        "[runtime tool results — authoritative ground truth, pre-fetched "
        "by the chat orchestrator before this turn. Treat these values as "
        "true and reference them in your answer; do not contradict them "
        "or claim you cannot access them.]\n\n"
    )
    payload = header + "\n\n".join(blocks)
    augmented = list(inputs)
    # Splice before the LAST user turn so the data lands right next to
    # the question. Falls back to prepend if no user turn is present.
    insert_at = len(augmented)
    for i in range(len(augmented) - 1, -1, -1):
        if augmented[i].get("role") == "user":
            insert_at = i
            break
    augmented.insert(insert_at, {"role": "user", "content": payload})
    return augmented


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


# ── Streaming SSE bridges ───────────────────────────────────────────────────

# Markers that signal the model has begun fabricating tool-call narration.
# Once any of these appear in the streaming text, we suppress the rest of
# the visible content for that turn -- the SDK already ran the real tool
# and the runtime trace shows it.
#
# Plain-string list (cheap substring scan in the hot path); the
# corresponding regex below picks up structural variants the model
# invents that don't reduce to a stable substring.
_TRUNC_MARKERS = (
    "## Return of",
    "tool##",
    "call_pyfence_",
    "## Tool Output",
    "## Sandbox Output",
    "## Python Output",
    "## Runtime Output",
    "## Tool Result",
    "<tool>",
    "</tool>",
    "<|tool_call_argument",
    "<|im_start|>tool",
)
# Anchored regex for `call_<hex>` fragments and other model attempts to
# emit synthetic call-id text. We anchor at start-of-line so a real
# Python identifier like ``call_handler`` in legitimate code is left
# alone.
_TRUNC_RE = re.compile(
    r"(?im)"
    r"(?:^|\n)\s*(?:call|fc)_[a-f0-9]{8,}\s*[\{:]"
    r"|(?:^|\n)\s*\*\*\s*Tool\s+Output\s*\*?\*?:?",
)


class _StreamingTextSanitizer:
    """Buffer streaming text deltas and yield user-safe slices.

    Handles three failure modes mid-stream without forcing a full-buffer
    end-of-turn flush:

    1. **Partial chat-template stop tokens** (``<|im_end|>`` ,
       ``<|tool_call_argument_begin|>``, etc.). When the buffered tail
       contains an unclosed ``<|`` we hold back from that ``<|`` until
       either the matching ``|>`` arrives (in which case we strip the
       whole token) or we accumulate enough text to be sure it's not a
       template token.
    2. **Fake tool-call narration** (``## Return of call_*``,
       ``call_pyfence_*{...}``). Once the buffer contains any of
       :data:`_TRUNC_MARKERS` we permanently stop yielding visible text;
       the runtime trace already shows the truth and the model has gone
       off the rails.
    3. **Pathological line repetition** (the model collapsing into an
       infinite ``from math import asinh\nfrom math import atanh\n...``
       loop). When the same short line repeats too many times in a row
       we cut the stream and never recover.

    All other text is yielded immediately so the dashboard sees true
    token-by-token streaming.
    """

    # Longest known chat-template stop token is ``<|tool_call_argument_begin|>``
    # at 30 chars; bump to 64 for safety.
    _MAX_BUFFER = 64
    # Lines shorter than this are eligible for repeat-detection; longer
    # lines are often legit code blocks where some repetition is fine.
    _REPEAT_LINE_MAX_CHARS = 80
    # How many identical short lines in a row before we cut the stream.
    _REPEAT_LIMIT = 8

    def __init__(self):
        self._tail = ""
        self._stopped = False
        self._line_buf = ""
        self._last_line: str | None = None
        self._repeat_count = 0

    def feed(self, chunk: str) -> str:
        if self._stopped or not chunk:
            return ""
        combined = self._tail + chunk
        # Truncation marker -> permanent suppression. Scan all markers
        # and pick the EARLIEST match so we always cut at the first sign
        # of fake tool-call narration.
        earliest = -1
        for marker in _TRUNC_MARKERS:
            idx = combined.find(marker)
            if idx >= 0 and (earliest == -1 or idx < earliest):
                earliest = idx
        m = _TRUNC_RE.search(combined)
        if m and (earliest == -1 or m.start() < earliest):
            earliest = m.start()
        if earliest >= 0:
            self._stopped = True
            emit = combined[:earliest]
            self._tail = ""
            return _STOP_TOKEN_RE.sub("", emit).rstrip()
        # Strip any complete ``<|...|>`` chat-template tokens
        cleaned = _STOP_TOKEN_RE.sub("", combined)
        # Hold back from the last unclosed ``<|`` so the next chunk can
        # complete the token (and we strip it whole).
        last_lt = cleaned.rfind("<|")
        if last_lt >= 0 and "|>" not in cleaned[last_lt:]:
            partial_len = len(cleaned) - last_lt
            if partial_len > self._MAX_BUFFER:
                # Too long to be a template token; just emit it.
                self._tail = ""
                emit = cleaned
            else:
                self._tail = cleaned[last_lt:]
                emit = cleaned[:last_lt]
        else:
            self._tail = ""
            emit = cleaned
        # Run repetition detection on the visible emit and cut if needed.
        return self._guard_repetition(emit)

    def _guard_repetition(self, emit: str) -> str:
        """Watch for ``N`` consecutive identical short lines and stop
        if the threshold is hit. Returns the (possibly truncated) emit."""
        if not emit or self._stopped:
            return emit
        out_chars: list[str] = []
        for ch in emit:
            out_chars.append(ch)
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf.strip()
                self._line_buf = ""
                if not line:
                    self._last_line = None
                    self._repeat_count = 0
                    continue
                if len(line) <= self._REPEAT_LINE_MAX_CHARS and line == self._last_line:
                    self._repeat_count += 1
                    if self._repeat_count >= self._REPEAT_LIMIT:
                        self._stopped = True
                        # Drop the trailing repeated lines we just
                        # appended; emit only the prefix.
                        joined = "".join(out_chars)
                        # Strip back to the last position before the
                        # repeats started so we don't show the user
                        # 8 copies of the same line.
                        cut = joined.rfind(
                            "\n" + line, 0,
                            joined.rfind(line),
                        )
                        if cut < 0:
                            cut = joined.find(line)
                        return (joined[:cut] if cut >= 0 else joined).rstrip() + (
                            "\n\n*[stopped: model output collapsed into a "
                            "repeating loop]*"
                        )
                else:
                    self._last_line = line
                    self._repeat_count = 1
        return "".join(out_chars)

    def flush(self) -> str:
        if self._stopped:
            self._tail = ""
            return ""
        out = _STOP_TOKEN_RE.sub("", self._tail)
        self._tail = ""
        return out

    @property
    def stopped(self) -> bool:
        return self._stopped


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
    "AgentRunOutput",
    "run_agent_chat",
    "stream_agent_chat_legacy",
    "stream_agent_chat_openai",
]
