"""calibration_bench — Brier score on yes/no factual items with confidence."""

from __future__ import annotations

import re

from distil.pod.axes._base import (
    BenchResult,
    block_seeded_rng,
    estimate_completion_tokens,
    generate_greedy,
)

ITEMS = (
    ("Is the Eiffel Tower in Paris?", "yes"),
    ("Is Mars larger than Earth?", "no"),
    ("Is Mount Everest the tallest mountain?", "yes"),
    ("Is the Sahara a tropical rainforest?", "no"),
    ("Are dolphins mammals?", "yes"),
    ("Is gold magnetic?", "no"),
)


def run(student_engine, *, block_seed: int, n_items: int):
    rng = block_seeded_rng(block_seed, "calibration_bench")
    items = [rng.choice(ITEMS) for _ in range(n_items)]
    prompts = [
        f"Answer yes or no AND give a confidence between 0 and 1. "
        f"Format: 'yes 0.9' or 'no 0.7'.\n\nQ: {q}\nA: "
        for q, _ in items
    ]
    outs = generate_greedy(student_engine, prompts, max_tokens=16)
    rows = []
    brier_total = 0.0
    correct = 0
    for (q, ans), (text, toks) in zip(items, outs, strict=False):
        m = re.search(r"\b(yes|no)\b\s+([01](?:\.\d+)?)", (text or "").lower())
        if not m:
            rows.append({"q": q, "ans": ans, "ok": False, "tokens": len(toks)})
            continue
        guess, conf_s = m.group(1), float(m.group(2))
        conf_for_yes = conf_s if guess == "yes" else (1.0 - conf_s)
        target = 1.0 if ans == "yes" else 0.0
        brier_total += (conf_for_yes - target) ** 2
        ok = guess == ans
        correct += int(ok)
        rows.append(
            {"q": q[:60], "ans": ans, "guess": guess, "conf": conf_s, "ok": ok, "tokens": len(toks)}
        )
    brier = brier_total / max(len(rows), 1)
    score = max(0.0, 1.0 - 2.0 * brier)
    res = BenchResult(
        n=len(rows),
        correct=correct,
        completion_tokens=estimate_completion_tokens(outs),
        items=rows,
        extra={"brier": round(brier, 4), "calibration_score": round(score, 4)},
    )
    return res.as_dict()
