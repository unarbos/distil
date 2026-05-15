"""long_form_judge_probe — 300-500 word essay rubric × statistical coherence factor."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from distil.pod.axes._base import block_seeded_rng, generate_greedy

PROMPTS = (
    "Write a 300-400 word essay on whether artificial general intelligence is a net good for humanity.",
    "Write a 300-400 word reflection on the role of public libraries in the 21st century.",
    "Write a 300-400 word piece on the most important lessons from the COVID-19 pandemic.",
    "Write a 300-400 word essay on whether remote work is sustainable for software teams.",
)

RUBRIC = (
    "Grade this 300-500 word essay on structure, depth, coherence and length adherence. "
    "Reply with ONLY an integer 1-5.\n\nPROMPT: {prompt}\n\nESSAY: {response}\n\nScore (1-5):"
)


def _coherence_factor(text: str) -> float:
    """6-signal statistical coherence: length, repetition, line variance, etc."""
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    if n_words < 50:
        return 0.0
    length_score = 1.0 if 250 <= n_words <= 600 else max(0.0, 1.0 - abs(n_words - 425) / 425)
    counts = Counter(w.lower() for w in words)
    top_freq = max(counts.values()) / n_words
    repetition_score = max(0.0, 1.0 - top_freq * 5)
    sentences = re.split(r"[.!?]+", text)
    nonblank = [s.strip() for s in sentences if s.strip()]
    n_sent = max(len(nonblank), 1)
    sent_score = min(1.0, n_sent / 8)
    avg_len = n_words / n_sent
    avg_len_score = max(0.0, 1.0 - abs(avg_len - 18) / 18)
    unique_ratio = len(set(counts)) / max(n_words, 1)
    diversity_score = min(1.0, unique_ratio * 2)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_score = min(1.0, len(paragraphs) / 3)
    return round(
        sum(
            [
                length_score,
                repetition_score,
                sent_score,
                avg_len_score,
                diversity_score,
                paragraph_score,
            ]
        )
        / 6,
        4,
    )


def _extract_score(text: str) -> int | None:
    m = re.search(r"\b([1-5])\b", text or "")
    return int(m.group(1)) if m else None


def run(student_engine, teacher_engine, *, block_seed: int, n_items: int) -> dict[str, Any]:
    rng = block_seeded_rng(block_seed, "long_form_judge_probe")
    prompts = [rng.choice(PROMPTS) for _ in range(n_items)]
    responses = generate_greedy(student_engine, prompts, max_tokens=600)
    judge_prompts = [
        RUBRIC.format(prompt=p, response=r[0]) for p, r in zip(prompts, responses, strict=False)
    ]
    judges = generate_greedy(teacher_engine, judge_prompts, max_tokens=8)
    scores = [_extract_score(t) for t, _ in judges]
    coherences = [_coherence_factor(r[0]) for r in responses]
    valid_pairs = [(s, c) for s, c in zip(scores, coherences, strict=False) if s is not None]
    if not valid_pairs:
        return {"n": len(prompts), "n_valid": 0, "normalized": None, "coherence_factor": None}
    mean_norm = sum(((s - 1) / 4) * c for s, c in valid_pairs) / len(valid_pairs)
    mean_coh = sum(c for _, c in valid_pairs) / len(valid_pairs)
    return {
        "n": len(prompts),
        "n_valid": len(valid_pairs),
        "normalized": round(mean_norm, 4),
        "coherence_factor": round(mean_coh, 4),
    }
