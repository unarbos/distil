"""Tests for the OpenAI Agents SDK harness in api/agent_runner.py.

We test in two layers:

1. **Unit**: pure helpers — python-fence detection / dedup / synthetic
   ``ResponseFunctionToolCall`` injection, message conversion, fake tool
   output stripping. These don't touch the SDK or vLLM.

2. **Integration**: monkeypatch the SDK's ``Runner.run`` so we don't hit
   the live model, but we still exercise the full ``run_agent_chat`` and
   the legacy SSE bridge to catch regressions in the chat-completion
   shape and the streaming format.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
API = os.path.join(ROOT, "api")
ROUTES = os.path.join(API, "routes")
# Keep API ahead of ROUTES on sys.path: ``api/chat/`` (the new package
# hosting the chat-completions plumbing extracted out of agent_runner) and
# ``api/routes/chat.py`` (the FastAPI route module) both want the bare
# ``chat`` name. The package wins when API is first; otherwise
# ``from chat.sanitizer import …`` (inside agent_runner) explodes with
# ``'chat' is not a package``.
for path in (ROUTES, API, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
for path in (ROUTES, API):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, ROUTES)
sys.path.insert(0, API)

import agent_runner  # noqa: E402
import agent_tools  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


# ── Helpers to build SDK-shaped objects without a live model ─────────────────


def _msg(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[], logprobs=[])],
        role="assistant",
        type="message",
        status="completed",
    )


# ── Unit: python-fence helpers ───────────────────────────────────────────────


def test_extract_python_fences_dedup_and_cap():
    text = (
        "Here are some snippets:\n"
        "```python\nprint(1)\n```\n"
        "Same:\n```py\nprint(1)\n```\n"  # dedup
        "```python\nprint(2)\n```\n"
        "```python\nprint(3)\n```\n"
        "```python\nprint(4)\n```\n"  # past cap
    )
    out = agent_runner._extract_python_fences(text)
    assert out == ["print(1)", "print(2)", "print(3)"]


def test_inject_python_tool_calls_appends_one_per_fence():
    items = [_msg("```python\nprint(7*6)\n```")]
    seen: set[str] = set()
    out = agent_runner._inject_python_fence_tool_calls(items, seen)
    assert len(out) == 2
    assert isinstance(out[0], ResponseOutputMessage)
    assert isinstance(out[1], ResponseFunctionToolCall)
    assert out[1].name == "python_exec"
    args = json.loads(out[1].arguments)
    assert args == {"code": "print(7*6)"}


def test_inject_python_tool_calls_dedup_across_calls():
    """Same code in a follow-up turn must NOT trigger another tool call."""
    seen: set[str] = set()
    items1 = [_msg("```python\nprint(7*6)\n```")]
    out1 = agent_runner._inject_python_fence_tool_calls(items1, seen)
    assert any(isinstance(i, ResponseFunctionToolCall) for i in out1)
    items2 = [_msg("Earlier I ran:\n```python\nprint(7*6)\n```")]
    out2 = agent_runner._inject_python_fence_tool_calls(items2, seen)
    assert not any(isinstance(i, ResponseFunctionToolCall) for i in out2), (
        "second turn must not re-inject already-executed code"
    )


def test_inject_python_tool_calls_skips_when_native_call_present():
    items = [
        _msg("```python\nprint(1)\n```"),
        ResponseFunctionToolCall(
            id="fc_1", call_id="call_1",
            arguments=json.dumps({"code": "print(1)"}),
            name="python_exec", type="function_call",
        ),
    ]
    seen: set[str] = set()
    out = agent_runner._inject_python_fence_tool_calls(items, seen)
    # Native tool call already present -- we must not append a duplicate.
    fn_calls = [i for i in out if isinstance(i, ResponseFunctionToolCall)]
    assert len(fn_calls) == 1


# ── Unit: stripping fake tool output ────────────────────────────────────────


def test_strip_fake_tool_output_removes_label_plus_block():
    content = (
        "Here's the calculation.\n\n"
        "```python\nprint(4**8)\n```\n\n"
        "**Tool Output:**\n```\n12345\n```\n\n"
        "Sandbox Output:\n```text\nfake again\n```\n"
        "Done."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    assert "Tool Output" not in cleaned
    assert "12345" not in cleaned
    assert "Sandbox Output" not in cleaned
    assert "fake again" not in cleaned
    assert "```python" in cleaned
    assert "Here's the calculation." in cleaned
    assert "Done." in cleaned


def test_strip_fake_tool_output_handles_inline_value():
    """Form B: ``**Tool Output:** <number>`` on a single line."""
    content = (
        "I computed it.\n\n"
        "```python\nprint(2**10)\n```\n\n"
        "**Tool Output:** 1024\n\n"
        "So 2^10 = 1024."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    assert "Tool Output" not in cleaned
    assert "```python" in cleaned
    assert "So 2^10 = 1024." in cleaned


def test_strip_fake_tool_output_truncates_at_fake_tool_narration():
    """Production failure mode: the model writes a clean prose answer,
    then starts mimicking the SDK's tool-call format with garbage like
    ``call_pyfence_1_xxx{...}`` and ``## Return of call_pyfence_*``.
    Drop everything from the first such marker."""
    raw = (
        "The 2000th Fibonacci number is **4224...125**.\n\n"
        "I'll verify this with a different implementation.\n\n"
        "```python\nimport math\nprint(math.fib(2000))\n```\n\n"
        "call_pyfence_1_6c6fa8c9{\"code\": \"import math\\nprint(math.fib(2000))\"}"
        "tool## Return of call_pyfence_1_6c6fa8c9\n"
        "{'stdout': '2', 'stderr_or_error': '', 'exit_code': 0}"
        "tool## Return of call_pyfence_1_064295a2\n"
        "{'stdout': 'long fake digit string'}"
    )
    cleaned = agent_runner._strip_fake_tool_output(raw)
    # The legitimate answer + python block survive.
    assert "4224" in cleaned and "125" in cleaned
    assert "```python" in cleaned
    # The fake tool-call narration is gone.
    assert "call_pyfence_" not in cleaned
    assert "## Return of" not in cleaned
    assert "long fake digit string" not in cleaned


def test_strip_fake_tool_output_drops_inline_tool_xml():
    """Some Hermes-style models emit ``<tool>{...}</tool>`` inline."""
    raw = "The answer is 42. <tool>{\"name\": \"python_exec\"}</tool> Done."
    cleaned = agent_runner._strip_fake_tool_output(raw)
    assert "<tool" not in cleaned
    assert "The answer is 42." in cleaned and "Done." in cleaned


def test_streaming_sanitizer_passes_clean_text_immediately():
    """Plain text with no template tokens or markers must be yielded
    verbatim with at most one short trailing buffer."""
    s = agent_runner._StreamingTextSanitizer()
    out = s.feed("Hello, world!")
    assert out == "Hello, world!"
    assert s.flush() == ""


def test_streaming_sanitizer_strips_complete_template_tokens():
    """Complete ``<|...|>`` tokens delivered in a single chunk are
    stripped, neighbouring text is preserved."""
    s = agent_runner._StreamingTextSanitizer()
    out = s.feed("Final answer: 42<|im_end|>")
    assert out == "Final answer: 42"
    assert s.flush() == ""


def test_streaming_sanitizer_holds_back_partial_template_token():
    """A partial ``<|`` at the boundary must be held until the matching
    ``|>`` arrives -- otherwise the user sees half of the template tag."""
    s = agent_runner._StreamingTextSanitizer()
    a = s.feed("Final: 42 <|im_")
    # The sanitizer holds back from "<|" onwards.
    assert a == "Final: 42 "
    b = s.feed("end|>more")
    assert b == "more"
    assert s.flush() == ""


def test_streaming_sanitizer_truncates_at_fake_tool_narration():
    """Once the model starts emitting ``## Return of call_*`` style
    fake tool-call narration, suppress all further visible text."""
    s = agent_runner._StreamingTextSanitizer()
    pre = s.feed("The answer is 42.\n\n")
    assert pre == "The answer is 42.\n\n"
    cut = s.feed("tool## Return of call_pyfence_xxx\n{'stdout': '42'}")
    assert cut == ""  # everything from the marker onwards is dropped
    assert s.stopped is True
    # Subsequent chunks are also dropped.
    assert s.feed("more garbage") == ""
    assert s.flush() == ""


def test_streaming_sanitizer_truncates_at_tool_output_header():
    """Variants like ``## Tool Output: call_xxx`` and ``## Sandbox
    Output`` must also be cut -- the runtime trace already shows the
    real result and the model is hallucinating it."""
    for header in (
        "## Tool Output: call_7c316b3faa0d",
        "## Sandbox Output\n```",
        "## Python Output:",
        "## Runtime Output:",
    ):
        s = agent_runner._StreamingTextSanitizer()
        pre = s.feed("Result is 42.\n\n")
        assert pre == "Result is 42.\n\n"
        cut = s.feed(header + "\n{'foo': 'bar'}")
        assert cut == ""
        assert s.stopped, f"sanitizer must cut on header={header!r}"


def test_streaming_sanitizer_truncates_at_call_id_fragment():
    """When the model leaks a synthetic ``call_<hex>{...}`` fragment
    on its own line, cut there too."""
    s = agent_runner._StreamingTextSanitizer()
    pre = s.feed("Computed.\n\n")
    assert pre == "Computed.\n\n"
    s.feed("call_a1b2c3d4{\"stdout\": \"x\"}")
    assert s.stopped


def test_streaming_sanitizer_does_not_cut_legitimate_code_token():
    """A legitimate Python identifier like ``call_handler`` (no hex
    suffix, not on its own line) must NOT trip the call-id detector."""
    s = agent_runner._StreamingTextSanitizer()
    text = "Use `call_handler` to register a callback."
    out = s.feed(text)
    assert out == text
    assert not s.stopped


def test_normalize_search_query_strips_chatty_prefix_and_punctuation():
    """Without normalization DDG returns 0 hits for chatty queries
    like ``'what is the current bitcoin price?'``. We just need the
    cropped query to be shorter, free of trailing punctuation, and
    free of obvious leading interrogative cruft."""
    cases = [
        "what is the current bitcoin price?",
        "what's the price of ethereum right now?",
        "tell me about the weather in tokyo today",
        "search the web for latest AI news",
        "google for tesla stock price",
        "what is the bitcoin price, and convert 0.5 BTC to USD?",
    ]
    for raw in cases:
        got = agent_runner._normalize_search_query(raw)
        assert got, f"got empty query for input={raw!r}"
        assert not got.endswith(("?", "!", ".", ",", ";", ":")), got
        assert "?" not in got, got
        # Must be at most as long as the input (and usually shorter)
        assert len(got) <= len(raw)
    # Compound clause: the bit after the comma must be dropped.
    out = agent_runner._normalize_search_query(
        "what is the bitcoin price, and convert 0.5 BTC to USD?"
    )
    assert "convert" not in out
    assert "USD" not in out
    assert "bitcoin" in out.lower() and "price" in out.lower()
    # Search-verb prefix must be removed.
    out2 = agent_runner._normalize_search_query("search the web for latest AI news")
    assert "search the web" not in out2.lower()
    assert "latest AI news" in out2 or "latest ai news" in out2.lower()


def test_normalize_search_query_falls_back_to_original_when_empty():
    """If the cropping removes everything, return the original text so
    we never send an empty query to DDG."""
    assert agent_runner._normalize_search_query("") == ""
    # All pure punctuation -> falls back
    assert agent_runner._normalize_search_query("?!?") == "?!?"


def test_normalize_python_code_for_dedup_ignores_comments_and_whitespace():
    a = """
    def f():
        # comment 1
        return 42  # trailing
    print(f())
    """
    b = "def f():\n    return 42\nprint(f())"
    assert (
        agent_runner._normalize_python_code_for_dedup(a)
        == agent_runner._normalize_python_code_for_dedup(b)
    )


def test_normalize_python_code_for_dedup_separates_actually_different_code():
    a = "print(1)"
    b = "print(2)"
    assert (
        agent_runner._normalize_python_code_for_dedup(a)
        != agent_runner._normalize_python_code_for_dedup(b)
    )


def test_streaming_sanitizer_cuts_pathological_line_repetition():
    """Real prod failure: the model collapses into ``from math import
    asinh\\nfrom math import atanh\\n...`` for hundreds of lines. We
    must cut after a small number of identical lines so the user
    doesn't drown in the loop."""
    s = agent_runner._StreamingTextSanitizer()
    pre = s.feed("Here's the imports:\n")
    assert pre == "Here's the imports:\n"
    repeated = "from math import asinh\n"
    out = ""
    for _ in range(15):
        out += s.feed(repeated)
        if s.stopped:
            break
    assert s.stopped is True, "sanitizer must cut on infinite line repetition"
    assert "stopped" in out.lower() or "repeating" in out.lower()
    assert out.count("from math import asinh") < 10, (
        "must not let dozens of identical lines through"
    )


def test_streaming_sanitizer_emits_partial_when_clearly_not_a_token():
    """If we see ``<|`` but it grows past 64 chars without ``|>``, it
    can't be a chat-template token; emit it so we don't stall."""
    s = agent_runner._StreamingTextSanitizer()
    pre = s.feed("Look at this code: <|")
    assert pre == "Look at this code: "
    # 80 chars after the "<|", still no closing "|>"; sanitizer must
    # eventually flush rather than buffer forever.
    long_filler = "x" * 80
    out = s.feed(long_filler)
    assert "<|" in out + s.flush()


def test_stop_token_re_strips_chat_template_leakage():
    """Various chat-template stop tokens that leak past vLLM's parser
    must be removed before the user sees the answer."""
    raw = (
        "Final answer: 42<|im_end|>"
        "<|tool_call_argument_begin|>system<|im_middle|>recap follows"
        "<|end_of_turn|>"
    )
    cleaned = agent_runner._STOP_TOKEN_RE.sub("", raw)
    assert "<|" not in cleaned
    assert "Final answer: 42" in cleaned
    assert "system" in cleaned  # the literal word, not the bracketed token


def test_strip_fake_tool_output_strips_label_then_bare_value():
    """The model often writes ``**Output:**\\n<value>\\n`` (label on one
    line, value on next, no fence). The stripper must drop both."""
    content = (
        "Sure, computing.\n\n"
        "```python\nprint(47 * 53)\n```\n\n"
        "**Output:**\n2491\n\n"
        "So 47 * 53 = 2491."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    assert "**Output:**" not in cleaned
    # The stripper must drop the label+value pair...
    assert "**Output:**\n2491" not in cleaned
    # ...but the model's own python block + downstream prose stay.
    assert "```python" in cleaned
    assert "Sure, computing." in cleaned
    assert "So 47 * 53 = 2491." in cleaned


def test_strip_fake_tool_output_keeps_legitimate_output_section():
    """Don't false-positive on a section header like 'Output:' followed
    by a real multi-line explanation -- only the short bare-value form
    should be stripped."""
    content = (
        "Output:\n"
        "The function should return a list of integers, but ours returns "
        "a generator. We need to wrap it in list().\n\n"
        "Here's the fix..."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    assert "function should return" in cleaned, (
        "the stripper ate a legitimate prose paragraph"
    )


def test_strip_fake_tool_output_strips_multiword_prefix():
    """Real-world prod failure mode: model writes ``**Python Sandbox
    Output:**`` (two prefix words) above a fenced block."""
    content = (
        "Done.\n\n"
        "```python\nprint(12 * 13)\n```\n\n"
        "**Python Sandbox Output:**\n```\n156\n```\n\n"
        "So 12 * 13 = 156."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    assert "Python Sandbox Output" not in cleaned
    assert "```python" in cleaned
    assert "So 12 * 13 = 156." in cleaned


def test_strip_fake_tool_output_strips_bare_output_label():
    """Real-world prod failure mode: model writes a bare ``**Output:**``
    label (no ``Tool`` prefix) above a fenced block. Must still strip."""
    content = (
        "Here's the verification.\n\n"
        "```python\nprint(13 * 17)\n```\n\n"
        "**Output:**\n```\n221\n```\n\n"
        "So 13 * 17 = 221."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    # The label and the fake fenced body must be gone …
    assert "**Output:**" not in cleaned
    # … but the model's own python block + closing prose stay.
    assert "```python" in cleaned
    assert "So 13 * 17 = 221." in cleaned


def test_strip_fake_tool_output_does_not_eat_real_prose():
    """Bare ``Output:`` mid-sentence (not at line start) must not match —
    "The output of the algorithm: ..." is legitimate prose."""
    content = (
        "I computed the answer. The output of my analysis is a clear win "
        "for plan B. Result: positive trend overall."
    )
    cleaned = agent_runner._strip_fake_tool_output(content)
    # Both phrases should survive because they're embedded in normal prose.
    assert "output of my analysis" in cleaned
    # "Result: positive..." is a tricky case: our regex may strip it.
    # Document the actual behavior so future tweaks are intentional.
    # We accept that bare "Result:" at line start might be stripped.
    assert "win for plan B" in cleaned


def test_split_think_blocks_promotes_inline_thinking():
    text = "<think>let me think</think>The answer is 42."
    visible, reasoning = agent_runner._split_think_blocks(text)
    assert visible == "The answer is 42."
    assert reasoning == "let me think"


def test_split_think_blocks_handles_kimi_triangle_markers():
    """Kimi-K2.6 chat template emits Unicode triangle markers
    (``◁think▷ ... ◁/think▷``) instead of ``<think>...</think>``
    when ``enable_thinking=True`` is forwarded via
    ``chat_template_kwargs``. The split helper has to recognise both
    or the trace leaks into the user-visible answer."""
    text = "◁think▷step 1, then step 2◁/think▷The final answer is 42."
    visible, reasoning = agent_runner._split_think_blocks(text)
    assert visible == "The final answer is 42."
    assert reasoning == "step 1, then step 2"


def test_split_think_blocks_strips_unmatched_kimi_triangles_via_sanitize():
    """A half-emitted ``◁/think▷`` (close-tag with no opener — the
    chat-template prepended the implicit ``◁think▷`` so only the
    closer reaches the user) must be stripped by
    ``_sanitize_assistant_text`` so the user doesn't see the raw
    template token."""
    text = "Some preface ◁/think▷The clean answer."
    cleaned = agent_runner._sanitize_assistant_text(text)
    assert "◁/think▷" not in cleaned
    assert "Some preface" in cleaned
    assert "The clean answer." in cleaned


# ── Unit: message history conversion ────────────────────────────────────────


def test_openai_messages_to_agent_input_drops_system_and_caps_history():
    msgs = [{"role": "system", "content": "should be dropped"}]
    msgs.extend([{"role": "user", "content": f"q{i}"} for i in range(20)])
    out = agent_runner._openai_messages_to_agent_input(msgs)
    assert all(m["role"] != "system" for m in out)
    assert len(out) <= agent_runner._HISTORY_TURN_CAP
    # Most recent question must be present.
    assert out[-1]["content"] == "q19"


def test_openai_messages_strips_displayed_thinking_from_assistant_history():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "<think>secret reasoning</think>final answer",
        },
        {"role": "user", "content": "follow-up"},
    ]
    out = agent_runner._openai_messages_to_agent_input(msgs)
    asst = next(m for m in out if m["role"] == "assistant")
    assert asst["content"] == "final answer"
    assert "secret reasoning" not in asst["content"]


def test_openai_messages_drops_known_derail_markers():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Use the tool get_model_info to inspect."},
        {"role": "user", "content": "follow-up"},
    ]
    out = agent_runner._openai_messages_to_agent_input(msgs)
    # The derailed assistant turn must not poison future turns.
    assert all(m.get("role") != "assistant" or "Use the tool" not in m["content"] for m in out)


# ── Unit: function tools are well-formed ────────────────────────────────────


def test_tools_have_unique_names_and_required_schema():
    seen = set()
    for tool in agent_tools.ALL_TOOLS:
        assert tool.name not in seen, f"duplicate tool name {tool.name!r}"
        seen.add(tool.name)
        schema = tool.params_json_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        # additionalProperties=False is required for OpenAI strict-mode
        # tool schemas; both vLLM tool parsers we use need it too.
        assert schema.get("additionalProperties") is False, (
            f"tool {tool.name} must have additionalProperties=False"
        )


def test_python_exec_sandbox_blocks_unsafe_imports():
    """The python_exec tool's underlying subprocess must reject bad code."""
    out, err = agent_tools._run_python_subprocess(
        "import subprocess\nsubprocess.run(['ls'])"
    )
    assert out == ""
    assert err is not None
    assert "subprocess" in err.lower()


def test_python_exec_sandbox_runs_safe_code():
    out, err = agent_tools._run_python_subprocess("print(7 * 6)")
    assert err is None
    assert out.strip() == "42"


def test_python_exec_sandbox_handles_huge_int_print():
    """Regression: Python 3.11+ caps int->str at 4300 digits by default,
    which broke ``print(fib(2000))`` (418 digits is fine but factorials
    aren't). Our prelude calls ``sys.set_int_max_str_digits(0)``; verify
    a 5000-digit number prints cleanly."""
    out, err = agent_tools._run_python_subprocess("print(10 ** 5000)")
    assert err is None
    assert out.startswith("1") and out.endswith("0")
    assert len(out) >= 5001


def test_python_exec_sandbox_blocks_open_call():
    out, err = agent_tools._run_python_subprocess("open('/etc/passwd').read()")
    assert err is not None
    assert "open" in err.lower()


# ── Integration: ``run_agent_chat`` with mocked Runner.run ──────────────────


@dataclass
class _FakeRunResult:
    final_output: str = ""
    new_items: list = None  # type: ignore[assignment]
    context_wrapper: Any = None


@pytest.fixture
def patched_runner(monkeypatch):
    """Patch agents.Runner.run to a controllable async callable."""
    captured: dict[str, Any] = {}

    async def fake_run(agent, inputs, *, context, max_turns, hooks, run_config):
        captured["agent"] = agent
        captured["inputs"] = inputs
        captured["context"] = context
        captured["max_turns"] = max_turns
        captured["hooks"] = hooks
        captured["run_config"] = run_config
        return captured["next_result"]

    monkeypatch.setattr(agent_runner.Runner, "run", staticmethod(fake_run))
    return captured


def test_run_agent_chat_returns_chat_completion_dict(patched_runner):
    patched_runner["next_result"] = _FakeRunResult(
        final_output="The answer is 42.",
        new_items=[],
    )
    body = {
        "messages": [{"role": "user", "content": "what?"}],
        "max_tokens": 200,
    }
    data = asyncio.run(
        agent_runner.run_agent_chat(body, king_uid=72, king_model="moonshotai/Kimi-K2.6")
    )
    assert data["object"] == "chat.completion"
    assert data["king_uid"] == 72
    assert data["model"] == "moonshotai/Kimi-K2.6"
    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is 42."
    assert msg["tool_calls"] == []


def test_run_agent_chat_strips_fake_tool_output_from_final_text(patched_runner):
    patched_runner["next_result"] = _FakeRunResult(
        final_output=(
            "I computed it.\n\n"
            "```python\nprint(2*21)\n```\n\n"
            "**Tool Output:** 42\n\n"
            "So 2*21 = 42."
        ),
        new_items=[],
    )
    body = {"messages": [{"role": "user", "content": "what is 2*21?"}]}
    data = asyncio.run(
        agent_runner.run_agent_chat(body, king_uid=72, king_model="king/v1")
    )
    content = data["choices"][0]["message"]["content"]
    assert "Tool Output" not in content
    assert "```python" in content
    assert "So 2*21 = 42." in content


def test_run_agent_chat_promotes_inline_think_blocks_to_reasoning(patched_runner):
    patched_runner["next_result"] = _FakeRunResult(
        final_output="<think>let me think</think>The answer is 65536.",
        new_items=[],
    )
    body = {"messages": [{"role": "user", "content": "what is 4**8?"}]}
    data = asyncio.run(
        agent_runner.run_agent_chat(body, king_uid=72, king_model="king/v1")
    )
    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is 65536."
    assert "let me think" in msg["reasoning"]


def test_run_agent_chat_returns_apology_on_max_turns(patched_runner, monkeypatch):
    """When the SDK raises MaxTurnsExceeded, the user must get a clean
    apology message instead of a 500."""
    from agents import MaxTurnsExceeded

    async def raise_max(*a, **kw):
        raise MaxTurnsExceeded("Max turns (10) exceeded")

    monkeypatch.setattr(agent_runner.Runner, "run", staticmethod(raise_max))
    body = {"messages": [{"role": "user", "content": "infinite loop please"}]}
    data = asyncio.run(
        agent_runner.run_agent_chat(body, king_uid=72, king_model="king/v1")
    )
    content = data["choices"][0]["message"]["content"]
    assert "ran out" in content.lower() or "rephras" in content.lower()


def test_run_agent_chat_handles_empty_messages(patched_runner):
    patched_runner["next_result"] = _FakeRunResult(final_output="x", new_items=[])
    data = asyncio.run(
        agent_runner.run_agent_chat({"messages": []}, king_uid=72, king_model="k")
    )
    content = data["choices"][0]["message"]["content"]
    assert "messages required" in content.lower()


def test_run_agent_chat_passes_king_context_to_run(patched_runner):
    patched_runner["next_result"] = _FakeRunResult(final_output="ok", new_items=[])
    body = {"messages": [{"role": "user", "content": "hi"}]}
    asyncio.run(
        agent_runner.run_agent_chat(body, king_uid=99, king_model="some/model")
    )
    ctx = patched_runner["context"]
    assert ctx is not None
    assert ctx.king_uid == 99
    assert ctx.king_model == "some/model"


def test_inject_preflight_context_splices_before_last_user_turn():
    """Pre-flight tool results must appear immediately before the user
    question so the model sees them as the most recent ground truth."""
    inputs = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "current question"},
    ]
    blocks = ["WEB_SEARCH_RESULT:\n{...}", "SN97_LIVE_STATE:\n{...}"]
    out = agent_runner._inject_preflight_context(inputs, blocks)
    assert len(out) == len(inputs) + 1
    # The injection should sit between the assistant turn and the LATEST user turn.
    assert out[-1] == inputs[-1]
    spliced = out[-2]
    assert spliced["role"] == "user"
    assert "WEB_SEARCH_RESULT" in spliced["content"]
    assert "SN97_LIVE_STATE" in spliced["content"]
    assert "authoritative" in spliced["content"].lower()


def test_inject_preflight_context_noop_when_no_blocks():
    inputs = [{"role": "user", "content": "hi"}]
    out = agent_runner._inject_preflight_context(inputs, [])
    assert out == inputs


def test_preflight_runs_web_search_for_time_sensitive_queries(monkeypatch):
    """Pre-flight detects web-search intent and calls the real DDG helper.
    We patch httpx so the test doesn't hit the network."""
    captured = {}

    class _FakeResp:
        def __init__(self, body):
            self.text = body
        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, url):
            captured["url"] = url
            body = (
                '<div class="result"><h2 class="result__title">'
                '<a class="result__a" href="https://example.com/btc">'
                'Bitcoin price today</a></h2>'
                '<a class="result__snippet">$81,491</a></div>'
            )
            return _FakeResp(body)

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "AsyncClient", _FakeClient)
    blocks, trace = asyncio.run(
        agent_runner._preflight_tools(
            "what is the bitcoin price right now?", king_uid=72, king_model="k",
        )
    )
    assert any("WEB_SEARCH_RESULT" in b for b in blocks), blocks
    assert any("web_search" in t for t in trace)
    assert "duckduckgo" in captured["url"].lower()


def test_preflight_skips_web_search_for_general_questions(monkeypatch):
    """No web search for evergreen questions -- saves latency and avoids
    poisoning the agent with stale snippet text."""
    blocks, trace = asyncio.run(
        agent_runner._preflight_tools(
            "explain what a transformer is in simple terms",
            king_uid=72, king_model="k",
        )
    )
    assert not any("WEB_SEARCH_RESULT" in b for b in blocks)
    assert not any("web_search" in t for t in trace)


def test_summarise_run_substitutes_real_stdout_when_model_returns_only_code():
    """Model failure mode: returns just a fenced ``\u0060\u0060\u0060python ... \u0060\u0060\u0060``
    block with no prose. Show the user the actual stdout instead of raw
    code so they get an answer they can read."""
    result = _FakeRunResult(
        final_output=(
            "```python\n"
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n): a, b = b, a + b\n"
            "    return a\n"
            "print(fibonacci(10))\n"
            "```"
        ),
        new_items=[],
    )
    stdout_event = agent_runner._StreamEvent(
        kind="thinking",
        text="python_exec stdout: 55",
        extra={"tool": "python_exec", "phase": "end"},
    )
    out = agent_runner._summarise_run(result, [stdout_event])
    assert "55" in out.text
    assert "Python sandbox" in out.text
    assert "def fibonacci" not in out.text


def test_summarise_run_keeps_normal_text_with_python_block():
    """When the model writes prose AROUND a python block, leave it alone."""
    result = _FakeRunResult(
        final_output="The answer is 55. Here's the code:\n```python\nprint(55)\n```",
        new_items=[],
    )
    stdout_event = agent_runner._StreamEvent(
        kind="thinking",
        text="python_exec stdout: 55",
        extra={"tool": "python_exec", "phase": "end"},
    )
    out = agent_runner._summarise_run(result, [stdout_event])
    assert "The answer is 55." in out.text
    assert "```python" in out.text


def test_summarise_run_emits_tool_trace_in_reasoning():
    """Tool calls in the run result must surface as a reasoning trace."""
    @dataclass
    class _Item:
        raw_item: Any

    fc = ResponseFunctionToolCall(
        id="fc_1", call_id="call_1",
        arguments=json.dumps({"code": "print(2+2)"}),
        name="python_exec", type="function_call",
    )
    result = _FakeRunResult(
        final_output="The answer is 4.",
        new_items=[_Item(raw_item=fc)],
    )
    out = agent_runner._summarise_run(result, [])
    assert out.text == "The answer is 4."
    assert "called python_exec" in out.reasoning
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["function"]["name"] == "python_exec"


# ── Integration: streaming SSE shape ────────────────────────────────────────


def _patch_streaming(monkeypatch, events):
    """Replace ``_run_agent_streaming`` with an async generator that just
    yields the supplied ``_StreamEvent`` list. Lets us isolate the SSE
    format wrappers from the SDK ``Runner.run_streamed`` machinery.

    Since the chat-package split (2026-05-19), the streaming bridges live
    in ``chat.streaming`` and reference ``_run_agent_streaming`` from
    that module's own namespace; ``agent_runner._run_agent_streaming`` is
    just a re-export. We patch BOTH binding points so the fake wins
    regardless of which symbol path the SSE bridge resolves through.
    """
    async def fake(body, king_uid, king_model, max_tokens, inputs):
        for ev in events:
            yield ev
    monkeypatch.setattr(agent_runner, "_run_agent_streaming", fake)
    import chat.streaming
    monkeypatch.setattr(chat.streaming, "_run_agent_streaming", fake)


def test_stream_agent_chat_openai_emits_role_then_content_then_done(monkeypatch):
    """Smoke: the OpenAI-format SSE stream emits role, content, [DONE]."""
    _patch_streaming(monkeypatch, [
        agent_runner._StreamEvent(kind="content", text="hello "),
        agent_runner._StreamEvent(kind="content", text="world"),
        agent_runner._StreamEvent(kind="done"),
    ])
    gen = agent_runner.stream_agent_chat_openai(
        {"messages": [{"role": "user", "content": "hi"}]},
        king_uid=72,
        king_model="king/v1",
    )

    async def collect():
        chunks: list[str] = []
        async for c in gen():
            chunks.append(c)
        return chunks

    chunks = asyncio.run(collect())
    body = "".join(chunks)
    assert "data: " in body
    assert "[DONE]" in body
    # Role chunk must come before any content chunk.
    role_idx = body.find('"role": "assistant"')
    content_idx = body.find('"content": "hello "')
    assert role_idx >= 0 and content_idx >= 0
    assert role_idx < content_idx
    # Both content deltas must be emitted as separate SSE chunks.
    assert '"content": "hello "' in body
    assert '"content": "world"' in body


def test_stream_agent_chat_openai_only_forwards_true_reasoning(monkeypatch):
    """Open-WebUI inlines accumulated reasoning_content deltas into a
    <details type="reasoning"> block at stream end and HTML-escapes the
    inner text. So pre-flight / tool-start / tool-end traces would leak
    as &gt; / &quot; / &#x27; noise inside that block. Only true model
    <think> deltas (phase=reasoning) reach reasoning_content; tool
    starts become native OpenAI tool_calls deltas."""
    _patch_streaming(monkeypatch, [
        agent_runner._StreamEvent(kind="thinking", text="preflight: web_search (5 hits)", extra={"phase": "preflight"}),
        agent_runner._StreamEvent(kind="thinking", text="calling python_exec(...)", extra={"phase": "start"}),
        agent_runner._StreamEvent(kind="thinking", text="python_exec stdout: 42", extra={"phase": "end"}),
        agent_runner._StreamEvent(kind="thinking", text="Let me think about this carefully.", extra={"phase": "reasoning"}),
        agent_runner._StreamEvent(kind="content", text="The answer is 42."),
        agent_runner._StreamEvent(kind="done"),
    ])
    gen = agent_runner.stream_agent_chat_openai(
        {"messages": [{"role": "user", "content": "hi"}]},
        king_uid=1, king_model="m",
    )

    async def collect():
        out: list[str] = []
        async for c in gen():
            out.append(c)
        return out

    body = "".join(asyncio.run(collect()))
    assert "Let me think about this carefully" in body
    assert '"reasoning_content": "Let me think about this carefully' in body
    assert '"reasoning_content": "preflight' not in body
    assert '"reasoning_content": "python_exec stdout' not in body
    assert '"reasoning_content": "calling python_exec' not in body
    assert '"tool_calls"' in body
    assert '"name": "python_exec"' in body
    assert "The answer is 42." in body
    assert "[DONE]" in body


def test_stream_agent_chat_openai_pads_content_with_blank_line(monkeypatch):
    """Content stream must end with \\n\\n so any HTML block Open-WebUI
    inlines (reasoning collapsible, etc.) sits on its own markdown line
    and parses as block-level HTML, not as inline text inside the last
    paragraph (which would render the raw <details ...> tag literally)."""
    _patch_streaming(monkeypatch, [
        agent_runner._StreamEvent(kind="content", text="answer with no trailing newline"),
        agent_runner._StreamEvent(kind="done"),
    ])
    gen = agent_runner.stream_agent_chat_openai(
        {"messages": [{"role": "user", "content": "hi"}]},
        king_uid=1, king_model="m",
    )

    async def collect():
        out: list[str] = []
        async for c in gen():
            out.append(c)
        return out

    body = "".join(asyncio.run(collect()))
    pad_idx = body.find('"content": "\\n\\n"')
    finish_idx = body.find('"finish_reason": "stop"')
    assert pad_idx >= 0, body
    assert pad_idx < finish_idx


def test_parse_tool_call_event_extracts_name_and_args():
    name, args = agent_runner._parse_tool_call_event('calling python_exec({"code": "print(42)"})')
    assert name == "python_exec"
    assert args == '{"code": "print(42)"}'
    name, args = agent_runner._parse_tool_call_event("calling web_search")
    assert name == "web_search"
    assert args == ""
    assert agent_runner._parse_tool_call_event("") == (None, None)
    assert agent_runner._parse_tool_call_event("not a tool call") == (None, None)
    assert agent_runner._parse_tool_call_event("preflight: ...") == (None, None)


def test_stream_agent_chat_legacy_emits_response_chunks(monkeypatch):
    _patch_streaming(monkeypatch, [
        agent_runner._StreamEvent(kind="content", text="legacy "),
        agent_runner._StreamEvent(kind="content", text="hi"),
        agent_runner._StreamEvent(kind="done"),
    ])
    gen = agent_runner.stream_agent_chat_legacy(
        {"messages": [{"role": "user", "content": "hi"}]},
        king_uid=72,
        king_model="king/v1",
    )

    async def collect():
        out: list[str] = []
        async for c in gen():
            out.append(c)
        return out

    chunks = asyncio.run(collect())
    body = "".join(chunks)
    assert '"response":' in body
    assert '"response": "legacy "' in body
    assert '"response": "hi"' in body
    assert body.endswith("data: [DONE]\n\n")


def test_stream_agent_chat_legacy_forwards_thinking_events(monkeypatch):
    _patch_streaming(monkeypatch, [
        agent_runner._StreamEvent(kind="thinking", text="preflight: sn97_state"),
        agent_runner._StreamEvent(kind="content", text="King is UID 5"),
        agent_runner._StreamEvent(kind="done"),
    ])
    gen = agent_runner.stream_agent_chat_legacy(
        {"messages": [{"role": "user", "content": "who's king?"}]},
        king_uid=5, king_model="x/y",
    )

    async def collect():
        out: list[str] = []
        async for c in gen():
            out.append(c)
        return out

    body = "".join(asyncio.run(collect()))
    assert '"thinking": "preflight: sn97_state"' in body
    assert '"response": "King is UID 5"' in body


# ── Tools: behavioural guards ───────────────────────────────────────────────


def test_parse_duckduckgo_html_extracts_titles_and_snippets():
    """The DDG parser is hot-path for ``web_search`` -- a regression here
    means every news / price / weather query returns garbage."""
    body = """
    <div class="result"><h2 class="result__title">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fbtc">
        Bitcoin price today
      </a>
    </h2>
      <a class="result__snippet" href="...">Bitcoin is currently $81,491.41 USD.</a>
    </div>
    <div class="result"><h2 class="result__title">
      <a class="result__a" href="https://b.example/news">Tech news</a>
    </h2>
    <div class="result__snippet">Top tech stories today.</div>
    </div>
    """
    parsed = agent_tools._parse_duckduckgo_html(body, query="btc", limit=5)
    assert len(parsed) == 2
    titles = [r["title"] for r in parsed]
    urls = [r["url"] for r in parsed]
    snippets = [r["snippet"] for r in parsed]
    assert "Bitcoin price today" in titles
    assert "https://a.example/btc" in urls
    assert any("$81,491.41" in s for s in snippets)
    assert "Tech news" in titles


def test_python_code_safety_check_rejects_obvious_bad_patterns():
    safe, reason = agent_tools._python_code_is_safe("print(1+1)")
    assert safe and reason is None

    safe, reason = agent_tools._python_code_is_safe("import requests; requests.get('http://x')")
    assert not safe
    assert "requests" in (reason or "")

    safe, reason = agent_tools._python_code_is_safe("eval('2+2')")
    assert not safe
    assert "eval" in (reason or "")
