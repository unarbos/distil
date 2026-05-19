"""Text sanitizer for the SN97 king's chat output.

Handles three failure modes:

1. **Fabricated tool-output narration** — the model "pretending to be
   the orchestrator" by writing ``**Tool Output:**`` / ``Sandbox stdout:``
   / ``## Return of call_xxx`` framing around its real
   ``\u0060\u0060\u0060python`` blocks. The SDK already runs the sandbox and
   surfaces results via ``runtime_trace``; the model's commentary is
   pure noise. Stripped by :func:`_strip_fake_tool_output`.

2. **Chat-template stop tokens leaking past vLLM** — bracketed
   ``<|...|>`` markers (Hermes / Qwen / Kimi / Llama-3 / Mistral) and
   Kimi-K2.6 triangle markers ``\u25c1think\u25b7``. Stripped by
   :data:`_STOP_TOKEN_RE`.

3. **Streaming text collapses into pathological loops** — same line
   repeating, or the model emits a fake ``call_<hex>{...}`` framing
   that the SDK won't catch. Detected and truncated by
   :class:`_StreamingTextSanitizer`.

All exports keep the leading underscore from the pre-split
``agent_runner`` module so the test suite (which references them by
their original names) keeps working unchanged.
"""

from __future__ import annotations

import re

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
    # (``\u25c1think\u25b7 ... \u25c1/think\u25b7``) instead of ASCII brackets. The Kimi
    # template emits these when ``enable_thinking=True`` is forwarded
    # via ``chat_template_kwargs`` and the upstream reasoning parser
    # only catches ``<think>`` literals — so the triangle variant
    # would otherwise leak into the user-visible answer.
    r"(?:<think\b[^>]*>(.*?)</think>|\u25c1think\u25b7(.*?)\u25c1/think\u25b7)",
    re.IGNORECASE | re.DOTALL,
)

_STOP_TOKEN_RE = re.compile(
    # Common chat-template end-of-message markers that occasionally leak
    # past the vLLM tool/reasoning parsers. Strip them so the dashboard
    # doesn't render them as visible text. Pattern accepts:
    # * any bracketed `<|...|>` token whose body is alphanumeric /
    #   underscore (covers Hermes, Qwen, Kimi, Llama-3, Mistral
    #   templates plus the synthetic ``<|tool_call_argument_begin|>``
    #   / ``<|im_middle|>`` markers that bleed through tool-call
    #   parsing).
    # * unmatched Kimi-K2.6 triangle markers ``\u25c1think\u25b7`` /
    #   ``\u25c1/think\u25b7`` / ``\u25c1|tool_calls_section_begin|\u25b7`` that
    #   occasionally leak when the model produces a half-open block
    #   (one of them gets stripped by ``_split_think_blocks``, the
    #   other doesn't).
    r"<\|[A-Za-z0-9_\-]+\|>|\u25c1/?[A-Za-z0-9_\-|]+\u25b7",
)

# ── Streaming-time truncation markers ───────────────────────────────────────
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


# ── Non-streaming sanitizers ────────────────────────────────────────────────


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
    """Promote inline ``<think>...</think>`` (or Kimi
    ``\u25c1think\u25b7...\u25c1/think\u25b7``) to a reasoning channel."""
    if not text:
        return text, ""
    lower = text.lower()
    if "<think" not in lower and "\u25c1think\u25b7" not in lower:
        return text, ""
    think_chunks: list[str] = []
    for m in _THINK_BLOCK_RE.finditer(text):
        # Group 1 = ``<think>...</think>`` body,
        # Group 2 = ``\u25c1think\u25b7...\u25c1/think\u25b7`` body.
        body = (m.group(1) or m.group(2) or "").strip()
        if body:
            think_chunks.append(body)
    visible = _THINK_BLOCK_RE.sub("", text).strip()
    reasoning = "\n\n".join(think_chunks).strip()
    return visible, reasoning


def _sanitize_assistant_text(text: str) -> str:
    """Strip ``<think>`` blocks, fake tool-output blocks, and stray
    chat-template stop tokens, then collapse blank lines so the
    user-facing text is clean."""
    visible, _ = _split_think_blocks(text or "")
    visible = _STOP_TOKEN_RE.sub("", visible)
    return _strip_fake_tool_output(visible).strip()


# ── Streaming sanitizer ─────────────────────────────────────────────────────


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


__all__ = [
    "_FAKE_TOOL_CALL_NARRATION_RE",
    "_FAKE_TOOL_NARRATION_TRUNC_RE",
    "_FAKE_TOOL_OUTPUT_RE",
    "_RUNNING_CODE_FILLER_RE",
    "_STOP_TOKEN_RE",
    "_THINK_BLOCK_RE",
    "_TRUNC_MARKERS",
    "_TRUNC_RE",
    "_StreamingTextSanitizer",
    "_sanitize_assistant_text",
    "_split_think_blocks",
    "_strip_fake_tool_output",
]
