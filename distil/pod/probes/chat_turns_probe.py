"""chat_turns_probe — multi-turn dialogue evaluation (3 turns, 1-5 rubric)."""

from __future__ import annotations

import re
from typing import Any

from distil.pod.axes._base import block_seeded_rng, generate_greedy

OPENERS = (
    "Hi! I'm trying to learn how Docker volumes work. Can you start with the basics?",
    "I want to plan a 3-day trip to Lisbon. Where should I start?",
    "Can you help me understand how DNS resolution works step by step?",
    "What's a good way to start learning Rust if I already know Python?",
)

FOLLOWUPS = (
    "Could you give a small concrete example?",
    "How does that change in production vs. local dev?",
    "What's the most common mistake people make with this?",
    "How would I debug it when it goes wrong?",
)

RUBRIC = (
    "Grade this 3-turn dialogue on coherence, consistency and helpfulness. "
    "Reply with ONLY an integer 1-5.\n\n--- DIALOGUE ---\n{dialogue}\n--- END ---\n\nScore (1-5):"
)


def _format_turn(role: str, msg: str) -> str:
    return f"{role}: {msg.strip()}\n"


def _extract_score(text: str) -> int | None:
    m = re.search(r"\b([1-5])\b", text or "")
    return int(m.group(1)) if m else None


def run(student_engine, teacher_engine, *, block_seed: int, n_items: int) -> dict[str, Any]:
    rng = block_seeded_rng(block_seed, "chat_turns_probe")
    dialogues: list[str] = []
    for _ in range(n_items):
        opener = rng.choice(OPENERS)
        followup = rng.choice(FOLLOWUPS)
        # Turn 1
        r1, _ = generate_greedy(student_engine, [opener], max_tokens=200)[0]
        # Turn 2
        prompt2 = (
            _format_turn("USER", opener)
            + _format_turn("ASSISTANT", r1)
            + _format_turn("USER", followup)
        )
        r2, _ = generate_greedy(student_engine, [prompt2], max_tokens=200)[0]
        # Turn 3
        prompt3 = (
            prompt2 + _format_turn("ASSISTANT", r2) + _format_turn("USER", "Thanks. Anything else?")
        )
        r3, _ = generate_greedy(student_engine, [prompt3], max_tokens=160)[0]
        dialogues.append(prompt3 + _format_turn("ASSISTANT", r3))
    judge_prompts = [RUBRIC.format(dialogue=d) for d in dialogues]
    judges = generate_greedy(teacher_engine, judge_prompts, max_tokens=8)
    scores = [_extract_score(t) for t, _ in judges]
    valid = [s for s in scores if s is not None]
    if not valid:
        return {"n": len(dialogues), "n_valid": 0, "normalized": None}
    mean = sum(valid) / len(valid)
    return {
        "n": len(dialogues),
        "n_valid": len(valid),
        "mean_score": round(mean, 3),
        "normalized": round((mean - 1) / 4, 4),
    }
