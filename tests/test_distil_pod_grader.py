"""Regression: Phase 3 rubric grading must work through the API teacher.

Before the fix, ``_phase_judge`` unconditionally called
``start_teacher(spec["teacher_repo"], ...)`` and tried to load the
teacher onto the local pod GPUs. With ``DISTIL_TEACHER_MODE=api`` the
teacher is Kimi-K2.6 (1T params) which doesn't fit on the 8xB200 pod,
so vLLM crashed with ``RuntimeError: Engine core initialization
failed`` every Phase 3 — judge / long-form / chat-turns axes came back
``None`` for every student, sinking the king's ``worst-3 mean`` and
triggering a bogus dethrone on round 1778895344.

The fix routes rubric grading through :class:`APIGrader` (greedy text-
only chat completions via OpenRouter) when teacher_mode=api. These
tests verify the three probe ``grade_*`` entry points accept either a
:class:`Grader` adapter or a raw vLLM engine (back-compat).
"""

from __future__ import annotations

from distil.pod.grader import Grader, VLLMGrader
from distil.pod.probes import chat_turns_probe, judge_probe, long_form_judge


class _StubGrader:
    """Records the prompts it sees and returns canned single-digit responses."""

    def __init__(self, response: str = "4") -> None:
        self.response = response
        self.calls: list[tuple[list[str], int]] = []

    def greedy(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        self.calls.append((list(prompts), max_tokens))
        return [self.response] * len(prompts)


class _StubVLLMEngine:
    """Mimics enough of vllm.LLM for VLLMGrader to wrap without crashing.

    We monkey-patch ``generate_greedy`` in :class:`VLLMGrader.greedy`
    by going through a fake engine that returns deterministic outputs.
    """


def test_judge_probe_grade_responses_accepts_grader() -> None:
    stub = _StubGrader(response="5")
    collected = [
        {"prompt": "Explain TCP", "response": "TCP is a connection-oriented protocol.", "tokens": 8},
        {"prompt": "Define API", "response": "An interface for programs to talk.", "tokens": 7},
    ]
    out = judge_probe.grade_responses(stub, collected)
    assert out["n"] == 2
    assert out["n_valid"] == 2
    assert out["mean_score"] == 5.0
    assert out["normalized"] == 1.0
    # rubric prompts were forwarded with max_tokens=8
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == 8


def test_long_form_judge_grade_responses_accepts_grader() -> None:
    stub = _StubGrader(response="3")
    collected = [
        {"prompt": "Write 300 words on testing.", "response": "Testing is foundational. " * 80, "tokens": 200},
    ]
    out = long_form_judge.grade_responses(stub, collected)
    assert out["n"] == 1
    assert out["n_valid"] == 1
    # normalized = ((3 - 1) / 4) * coherence_factor (∈ [0,1]); just non-None + ≤ 0.5
    assert out["normalized"] is not None and 0.0 <= out["normalized"] <= 0.5
    assert out["coherence_factor"] is not None


def test_chat_turns_grade_dialogues_accepts_grader() -> None:
    stub = _StubGrader(response="4")
    collected = [{"dialogue": "USER: hi\nASSISTANT: hello\n", "opener": "hi", "followup": "?"}]
    out = chat_turns_probe.grade_dialogues(stub, collected)
    assert out["n"] == 1
    assert out["n_valid"] == 1
    assert out["normalized"] == 0.75


def test_grader_back_compat_accepts_raw_vllm_engine(monkeypatch) -> None:
    """Old code paths that pass a raw vLLM engine still work (wrapped in
    :class:`VLLMGrader` automatically). We monkey-patch generate_greedy
    so this test doesn't need an actual vLLM install at runtime."""

    def fake_generate_greedy(engine, prompts, *, max_tokens):
        del engine, max_tokens
        return [("2", []) for _ in prompts]

    monkeypatch.setattr("distil.pod.axes._base.generate_greedy", fake_generate_greedy)

    class FakeEngine:
        pass

    out = judge_probe.grade_responses(
        FakeEngine(), [{"prompt": "x", "response": "y", "tokens": 1}]
    )
    assert out["n_valid"] == 1
    assert out["mean_score"] == 2.0


def test_grader_returning_empty_text_collapses_to_n_valid_zero() -> None:
    """API returning '' for every prompt should yield normalized=None."""
    stub = _StubGrader(response="")
    out = judge_probe.grade_responses(
        stub, [{"prompt": "x", "response": "y", "tokens": 1}]
    )
    assert out["n"] == 1
    assert out["n_valid"] == 0
    assert out["normalized"] is None


def test_grader_protocol_is_satisfied_by_stub() -> None:
    """Make sure the duck-typed Grader protocol matches both Stub + VLLMGrader.

    Grader is a non-runtime-checkable Protocol — we verify the ``.greedy``
    attribute exists with the right signature instead of using
    ``isinstance``.
    """
    stub = _StubGrader()
    assert hasattr(stub, "greedy") and callable(stub.greedy)
    assert hasattr(VLLMGrader(object()), "greedy")
    # And the duck-typing in grade_responses works (no Grader import needed)
    out = judge_probe.grade_responses(
        stub, [{"prompt": "x", "response": "y", "tokens": 1}]
    )
    assert out["n_valid"] == 1
    # Suppress unused import warning while keeping the import visible
    assert Grader is not None
