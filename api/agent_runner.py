"""Backwards-compat facade for the ``api/chat/`` sub-package.

This module used to be a 1750-line mega-module that mixed nine distinct
concerns (SDK bootstrap, sanitizer, model wrapper, hooks, preflight,
history translation, agent factory, sync entry, streaming bridges).
Each concern now lives in its own file under ``api/chat/``.

We keep this module as a re-export facade so:

  * ``api/routes/chat.py`` keeps importing ``run_agent_chat`` /
    ``stream_agent_chat_openai`` / ``stream_agent_chat_legacy`` from
    here (~3 line changes avoided).

  * ``tests/test_agent_runner.py`` (1055 lines, ~830 tests) keeps
    referencing ~20 private helpers via ``agent_runner._foo``
    (``_StreamingTextSanitizer``, ``_strip_tools_for_vllm``,
    ``_inject_python_fence_tool_calls``, etc.). The test file does
    ``import agent_runner`` and reaches for everything via attribute
    access, so as long as the names resolve at this module path the
    tests stay green.

The public surface (``__all__``) is unchanged: the four entry points
the rest of the codebase actually calls.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("distil.agent_runner")

# ── SDK + openai SDK types — re-exported because tests poke at them ─────────
from agents import (  # noqa: E402, F401
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
from agents.exceptions import AgentsException  # noqa: E402, F401
from agents.items import ModelResponse  # noqa: E402, F401
from agents.stream_events import (  # noqa: E402, F401
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from openai.types.responses import (  # noqa: E402, F401
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
try:
    from openai.types.responses import ResponseReasoningTextDeltaEvent  # noqa: E402, F401
except ImportError:  # pragma: no cover -- older openai pkg
    ResponseReasoningTextDeltaEvent = None  # type: ignore

# ── Chat SDK-context dataclass + the one SDK-registered tool ────────────────
from agent_tools import SN97AgentContext, python_exec, system_prompt_for  # noqa: E402, F401

# ── Sub-package re-exports ──────────────────────────────────────────────────
# Each module owns one concern. See its docstring for what it does.
from chat.sanitizer import (  # noqa: E402, F401
    _FAKE_TOOL_CALL_NARRATION_RE,
    _FAKE_TOOL_NARRATION_TRUNC_RE,
    _FAKE_TOOL_OUTPUT_RE,
    _RUNNING_CODE_FILLER_RE,
    _STOP_TOKEN_RE,
    _THINK_BLOCK_RE,
    _TRUNC_MARKERS,
    _TRUNC_RE,
    _StreamingTextSanitizer,
    _sanitize_assistant_text,
    _split_think_blocks,
    _strip_fake_tool_output,
)
from chat.model import (  # noqa: E402, F401
    _MAX_PYTHON_FENCE_INJECTIONS,
    _PY_FENCE_RE,
    _SN97ChatCompletionsModel,
    _extract_python_fences,
    _has_native_python_tool_call,
    _inject_python_fence_tool_calls,
    _make_synthetic_python_tool_call,
    _normalize_python_code_for_dedup,
    _scrub_tool_choice,
    _strip_tools_for_vllm,
)
from chat.config import (  # noqa: E402, F401
    _AGENT_DEFAULT_TEMPERATURE,
    _AGENT_MAX_TOKENS_CEILING,
    _AGENT_MAX_TOKENS_DEFAULT,
    _AGENT_MAX_TURNS,
    _VLLM_API_KEY,
    _VLLM_BASE_URL,
    _VLLM_SERVED_MODEL,
    _ensure_sdk_configured,
)
from chat.hooks import (  # noqa: E402, F401
    _StreamEvent,
    _StreamingHooks,
    _format_tool_result_for_thinking,
)
from chat.history import (  # noqa: E402, F401
    _CLIENT_THINK_RE,
    _DERAIL_MARKERS,
    _HISTORY_CHAR_CAP_PER_MSG,
    _HISTORY_TURN_CAP,
    _RUNTIME_TRACE_LINE_RE,
    _latest_user_text,
    _openai_messages_to_agent_input,
    _strip_displayed_thinking,
)
from chat.preflight import (  # noqa: E402, F401
    _MODEL_PATH_RE,
    _SEARCH_PREFIX_RE,
    _SN97_RE,
    _WEB_SEARCH_RE,
    _inject_preflight_context,
    _normalize_search_query,
    _preflight_tools,
)
from chat.agent_factory import _build_agent  # noqa: E402, F401
from chat.run_sync import (  # noqa: E402, F401
    AgentRunOutput,
    _build_chat_completion_response,
    _empty_assistant_response,
    _summarise_run,
    run_agent_chat,
)
from chat.streaming import (  # noqa: E402, F401
    _TOOL_CALL_PREVIEW_RE,
    _format_tool_called_event,
    _format_tool_output_event,
    _openai_sse_delta,
    _parse_tool_call_event,
    _run_agent_streaming,
    stream_agent_chat_legacy,
    stream_agent_chat_openai,
)


__all__ = [
    "AgentRunOutput",
    "run_agent_chat",
    "stream_agent_chat_legacy",
    "stream_agent_chat_openai",
]
