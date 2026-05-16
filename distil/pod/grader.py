"""Grader adapter for Phase 3 rubric judging.

Phase 3 (judge / long_form_judge / chat_turns) feeds rubric prompts to
the teacher and expects a short text response (typically a 1-5 digit).
Pre-cutover the teacher was a local vLLM engine, so the probes called
:func:`distil.pod.axes._base.generate_greedy` directly. After the
``DISTIL_TEACHER_MODE=api`` cutover the teacher is Kimi-K2.6 routed
through OpenRouter (it doesn't fit locally), so the rubric pass has to
go over HTTP too -- otherwise vLLM crashes trying to load 1T params on
an 8xB200 pod and judge axes silently come back ``None`` for every
student, which sinks ``worst-3 mean`` enough to trigger bogus
dethrones (the 2026-05-16 round_1778895344 post-mortem).

This module exposes a single thin protocol -- :class:`Grader` -- that
either a vLLM engine or the OpenRouter chat API can implement, and the
three probes call ``grader.greedy(prompts, max_tokens=N)`` regardless
of which mode the validator is in.
"""

from __future__ import annotations

from typing import Protocol


class Grader(Protocol):
    """Greedy text-only completion adapter."""

    def greedy(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        """Return one short greedy continuation per prompt, same order."""
        ...


class VLLMGrader:
    """Wrap a local vLLM ``LLM`` engine in the :class:`Grader` protocol."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def greedy(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        from distil.pod.axes._base import generate_greedy

        gens = generate_greedy(self.engine, prompts, max_tokens=max_tokens)
        return [text or "" for text, _ in gens]


class APIGrader:
    """Route rubric prompts through the OpenAI-compatible teacher API."""

    def greedy(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        from distil.pod.teacher_api import greedy_batch_api

        return greedy_batch_api(prompts, max_new_tokens=max_tokens)


__all__ = ["APIGrader", "Grader", "VLLMGrader"]
