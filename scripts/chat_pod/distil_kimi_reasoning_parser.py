# SPDX-License-Identifier: Apache-2.0
"""Custom Kimi-family reasoning parser used by chat.arbos.life.

Why a custom parser:
  vLLM's stock ``KimiK2ReasoningParser`` (vllm/reasoning/kimi_k2_reasoning_parser.py)
  treats ANY model output without an explicit ``</think>`` token as 100%
  reasoning — it returns ``(reasoning=full_output, content=None)``. That
  matches the Kimi K2.6 reference model's contract (the official chat
  template auto-prepends ``<think>`` to every assistant turn so the model
  is structurally "inside thinking" until it emits ``</think>``), but it
  is broken UX for the weak distilled-student kings that win SN97
  rounds: those students rarely emit ``</think>`` on their own, so EVERY
  response landed in ``message.reasoning`` and Open-WebUI showed an
  empty answer with the actual content trapped inside the (collapsed
  by default) Thinking pane.

  We want the best of both worlds:
    * If the king DOES emit ``thoughts</think>answer`` (or just
      ``</think>answer``): split on ``</think>`` → reasoning shows in
      the Thinking pane, content shows as the answer. Same as stock.
    * If the king emits ``answer`` with NO ``</think>``: don't bury the
      answer in reasoning. Treat the whole output as content. Users see
      the answer (no Thinking pane that turn — that's accurate, the
      king didn't think).
    * If the king emits ``<think>thoughts</think>answer`` (literal tags
      in its output): also split — the parser strips the redundant
      leading ``<think>`` if present.
    * If the king emits a Kimi tool-call section (``<|tool_calls_section_begin|>``)
      without first closing ``</think>``: we mirror stock behavior and
      treat everything before the section as reasoning, since that IS
      what the Kimi training contract says (thinking implicitly ends at
      a tool call).

  Streaming follows the same rules: until we see ``</think>`` or a tool
  section, we stream tokens as ``content`` (NOT reasoning). When/if we
  later see a ``</think>``, we retroactively reclassify previous text
  as reasoning via the ``DeltaMessage(reasoning=..., content=...)``
  transition. Worst case the user sees the answer first and a Thinking
  pane appearing late — strictly better than no answer at all.

Wiring:
  ``scripts/chat_pod/chat_server.py`` copies this file into vLLM's
  ``vllm/reasoning/`` directory on the chat-king pod and registers it
  via ``--reasoning-parser distil_kimi`` for kimi_k2 / kimi_k25
  families. The registration uses vLLM's plugin auto-discovery in
  ``vllm/reasoning/__init__.py``: any class decorated with
  ``@ReasoningParserManager.register_module("distil_kimi")`` is loaded
  on parser import. We register both as a class decorator AND
  defensively at module import time so re-importing this file in a
  test or a partial reload doesn't double-register.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


@ReasoningParserManager.register_module("distil_kimi")
class DistilKimiReasoningParser(ReasoningParser):
    """Lenient Kimi-family parser — see module docstring."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )

        if self._end_token_id is None:
            raise RuntimeError(
                "DistilKimiReasoningParser could not locate the </think> "
                "token in the tokenizer — is this a Kimi-family model?"
            )

    # ------------------------------------------------------------------
    # End-of-reasoning detection
    # ------------------------------------------------------------------
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Reasoning ends when we see ``</think>`` OR a tool-call section.

        For our lenient model, we ALSO consider reasoning "ended" if we
        never saw a ``<think>`` opener at all — that means the model
        skipped thinking entirely and started straight on the answer.
        Marking is_reasoning_end=True in that case lets vLLM emit the
        first content delta as content (not reasoning), which is the
        whole point of the lenient mode.
        """
        end_id = self._end_token_id
        tool_id = self._tool_section_start_token_id
        start_id = self._start_token_id

        saw_start = False
        for tok in input_ids:
            if start_id is not None and tok == start_id:
                saw_start = True
            if tok == end_id:
                return True
            if tool_id is not None and tok == tool_id:
                return True
        # No </think> and no tool section. If we never saw <think>
        # either, the model emitted plain answer text — reasoning is
        # already "over" (it never started). If we saw <think> but no
        # close, we're still mid-thinking.
        return not saw_start

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Streaming variant — same logic against the delta tokens."""
        delta_set = set(delta_ids)
        if self._end_token_id in delta_set:
            return True
        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_set
        ):
            return True
        # If we never saw a <think> opener in the cumulative stream,
        # treat as "reasoning ended" so deltas flow as content.
        if self._start_token_id is None:
            return True
        for tok in input_ids:
            if tok == self._start_token_id:
                return False
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return the token IDs that should be classified as content."""
        end_id = self._end_token_id
        tool_id = self._tool_section_start_token_id
        start_id = self._start_token_id

        if end_id in input_ids:
            end_index = (
                len(input_ids) - 1 - input_ids[::-1].index(end_id)
            )
            return input_ids[end_index + 1 :]
        if tool_id is not None and tool_id in input_ids:
            tool_index = (
                len(input_ids) - 1 - input_ids[::-1].index(tool_id)
            )
            # Include the section tokens so the downstream tool-call
            # parser can find them.
            return input_ids[tool_index:]
        # No </think> and no tool section. Lenient mode: if we also
        # never saw <think>, treat the whole stream as content (the
        # model skipped thinking and went straight to the answer).
        if start_id is not None and start_id not in input_ids:
            return list(input_ids)
        # Saw <think> but no close — still reasoning, no content yet.
        return []

    # ------------------------------------------------------------------
    # Non-streaming path (used by the OpenAI /v1/chat/completions
    # endpoint when stream=false — Open-WebUI uses streaming, but the
    # health-check probes use non-streaming).
    # ------------------------------------------------------------------
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Return ``(reasoning, content)`` per the lenient rules.

        - ``</think>`` present  → split on it
        - tool section present  → text before is reasoning (matches
                                  stock parser; tool calls are content
                                  and are picked up by the tool-call
                                  parser downstream)
        - neither present       → ``(None, model_output)`` — content is
                                  the whole thing, no reasoning
        """
        # If the model echoed the leading <think> from the prompt back
        # into its output, strip it before we look for the close.
        text = model_output
        if text.startswith(self._start_token):
            text = text[len(self._start_token) :]

        end_index = text.find(self._end_token)
        if end_index != -1:
            reasoning = text[:end_index]
            content = text[end_index + len(self._end_token) :]
            return (
                reasoning if reasoning else None,
                content if content else None,
            )

        tool_index = text.find(self._tool_section_start_token)
        if tool_index != -1:
            reasoning = text[:tool_index]
            content = text[tool_index:]
            return (
                reasoning if reasoning else None,
                content if content else None,
            )

        # Lenient fallback: no </think>, no tool section. If the
        # ORIGINAL output had a <think> opener that we just stripped,
        # the model still believes it's mid-thinking — keep stock
        # behavior (return as reasoning) so we don't accidentally show
        # raw scratch as the answer. If there was no opener at all, the
        # model just answered without thinking — return as content.
        if model_output.startswith(self._start_token):
            return (text, None)
        return (None, text)

    # ------------------------------------------------------------------
    # Streaming path (Open-WebUI default)
    # ------------------------------------------------------------------
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Stream-friendly classifier.

        Decision tree per delta:
          1. If we already saw </think> or a tool section in
             ``previous_token_ids``: stream as content.
          2. If the delta crosses </think>: split — reasoning before,
             content after.
          3. If the delta crosses a tool section start: split — reasoning
             before, content (incl. section) after.
          4. If no <think> opener was ever observed in either previous
             or current text: stream as content (lenient). This is the
             whole point of the parser — never bury an answer in
             reasoning when the model didn't think at all.
          5. Otherwise: stream as reasoning (we're inside an open
             <think> block).
        """
        end_id = self._end_token_id
        tool_id = self._tool_section_start_token_id
        start_id = self._start_token_id

        # (1) Already past reasoning end → pure content.
        already_ended = False
        for tok in previous_token_ids:
            if tok == end_id:
                already_ended = True
                break
            if tool_id is not None and tok == tool_id:
                already_ended = True
                break
        if already_ended:
            return DeltaMessage(content=delta_text)

        # Skip emitting standalone <think>/</think> singletons.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            start_id,
            end_id,
        ):
            return None

        # (2) Delta crosses </think>.
        if end_id in delta_token_ids:
            end_pos = delta_text.find(self._end_token)
            reasoning = delta_text[:end_pos]
            content = delta_text[end_pos + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # (3) Delta crosses tool section.
        if tool_id is not None and tool_id in delta_token_ids:
            tool_pos = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_pos]
            content = delta_text[tool_pos:]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content,
            )

        # (4) Lenient fallback — never saw a <think> opener.
        # We check both previous and current cumulative text so we
        # don't misclassify the very first delta (which has empty
        # previous).
        if start_id is not None:
            saw_open = False
            for tok in current_token_ids:
                if tok == start_id:
                    saw_open = True
                    break
            if not saw_open:
                return DeltaMessage(content=delta_text)

        # (5) Inside an open <think> block — stream as reasoning.
        return DeltaMessage(reasoning=delta_text)
