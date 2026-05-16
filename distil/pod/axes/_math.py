"""Shared answer-extraction + scoring for math-style v31 axes.

Used by ``math_gsm`` (v31_math_gsm_symbolic) and ``math_robustness``
(v31_math_robustness) — both axes emit gold via the standard ``#### N``
GSM8K marker or the ``\\boxed{...}`` MATH-500 marker, so a single
extractor + grader keeps the two axes apples-to-apples.

This is a faithful, decoupled copy of the matching helpers in
``scripts/pod_eval_vllm.py`` (``_extract_boxed`` / ``_math_extract_answer`` /
``_math_score_one`` / ``_math_format_prompt``); the regex set is the
authoritative one used in production today.
"""

from __future__ import annotations

import re

_MATH_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MATH_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_MATH_ANSWER_PHRASE_RE = re.compile(
    r"(?:the\s+)?answer\s*(?:is|=|:)\s*\$?([^\s\n\.]+)",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRAIL_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
_THINK_NARRATIVE_RE = re.compile(r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop ``<think>...</think>`` / "Thinking Process:" leaders before extraction."""
    if not text:
        return ""
    if "<think>" in text:
        text = _THINK_BLOCK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = _THINK_TRAIL_RE.sub("", text, count=1)
    if text.lstrip().startswith("Thinking Process:"):
        text = _THINK_NARRATIVE_RE.sub("", text, count=1)
    return text.strip()


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` (nested braces one deep)."""
    last = None
    for m in _MATH_BOXED_START_RE.finditer(text):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[i:j].strip()
                    break
            j += 1
    return last


def format_prompt(question: str, src: str) -> str:
    """Nudge the model toward a deterministic final-answer format."""
    if src == "math500":
        return (
            f"{question}\n\n"
            "Solve the problem and end your response with "
            "'\\boxed{ANSWER}' where ANSWER is the final simplified result."
        )
    return (
        f"{question}\n\n"
        "Solve step by step and end with '#### N' where N is the final numeric answer."
    )


def extract_answer(text: str, src: str = "") -> str:
    """Pull the numeric/boxed answer from a model generation."""
    cleaned = strip_thinking(text or "")
    if not cleaned:
        return ""
    if src == "math500":
        boxed = extract_boxed(cleaned)
        if boxed:
            return boxed.rstrip(".")
    if "####" in cleaned:
        m = re.search(r"####\s*([^\n]+)", cleaned)
        if m:
            tail = m.group(1).strip().rstrip(".")
            tm = _MATH_NUMBER_RE.search(tail)
            if tm:
                return tm.group(0)
            return tail
    m = _MATH_ANSWER_PHRASE_RE.search(cleaned)
    if m:
        frag = m.group(1).strip().rstrip(".,")
        tm = _MATH_NUMBER_RE.search(frag)
        if tm:
            return tm.group(0)
        if frag:
            return frag
    nums = _MATH_NUMBER_RE.findall(cleaned)
    if nums:
        return nums[-1]
    return cleaned.strip().splitlines()[-1].strip() if cleaned else ""


def score_answer(pred: str, gold: str) -> int:
    """1 if pred equals gold under normalisation, else 0."""
    if not pred:
        return 0
    p = pred.replace(",", "").replace("$", "").strip().rstrip(".")
    g = gold.replace(",", "").replace("$", "").strip().rstrip(".")
    if p == g:
        return 1
    try:
        return 1 if abs(float(p) - float(g)) < 1e-6 else 0
    except (TypeError, ValueError):
        return 0


__all__ = [
    "extract_answer",
    "extract_boxed",
    "format_prompt",
    "score_answer",
    "strip_thinking",
]
