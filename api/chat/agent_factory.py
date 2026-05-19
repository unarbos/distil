"""Build a fresh OpenAI Agents SDK ``Agent`` per request.

A fresh Agent per request keeps the system prompt up-to-date with the
live king (king info changes every eval round) and avoids any
cross-request state on the Agent object.

The vLLM-specific ``extra_body`` knobs live here because they're
tightly coupled to the model wrapper in :mod:`chat.model`: changing
``enable_thinking`` here without updating the sanitizer in
:mod:`chat.sanitizer` will leak ``<think>`` tags into the user-visible
answer; changing the stop sequence without updating
:func:`_inject_python_fence_tool_calls` will break the synthetic
python-exec tool call injection.
"""

from __future__ import annotations

from agents import Agent, ModelSettings

from agent_tools import SN97AgentContext, python_exec, system_prompt_for

from chat.config import (
    _AGENT_DEFAULT_TEMPERATURE,
    _VLLM_SERVED_MODEL,
    _ensure_sdk_configured,
)
from chat.model import _SN97ChatCompletionsModel


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
            #   recognise the king's actual ``\u25c1/think\u25b7`` Kimi
            #   triangle tags anyway, so the "Thinking" pane stayed
            #   empty. Direct content output is the lesser of two evils
            #   until the chat pod re-deploys to a king whose template
            #   matches the parser. The sanitizer's ``_THINK_BLOCK_RE``
            #   still strips any inline ``<think>`` block the model
            #   emits voluntarily.
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


__all__ = ["_build_agent"]
