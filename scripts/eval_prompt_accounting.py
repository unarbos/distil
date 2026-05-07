"""Prompt accounting helpers for eval quality gates."""

import os


def min_teacher_logprob_coverage() -> float:
    raw = os.environ.get("DISTIL_MIN_TEACHER_LOGPROB_COVERAGE", "0.5")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.5
    return min(1.0, max(0.0, value))


def min_effective_teacher_prompts() -> int:
    raw = os.environ.get("DISTIL_MIN_EFFECTIVE_TEACHER_PROMPTS", "30")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 30
    return max(1, value)


def teacher_logprob_coverage_stats(total_prompts: int, usable_prompts: int) -> dict:
    total = max(0, int(total_prompts or 0))
    usable = max(0, int(usable_prompts or 0))
    coverage = (usable / total) if total > 0 else 0.0
    return {
        "n_teacher_prompts_total": total,
        "n_teacher_prompts_with_logprobs": usable,
        "n_teacher_prompts_dropped_missing_logprobs": max(0, total - usable),
        "teacher_logprob_coverage": round(coverage, 6),
    }


def validate_teacher_logprob_coverage(total_prompts: int, usable_prompts: int) -> dict:
    stats = teacher_logprob_coverage_stats(total_prompts, usable_prompts)
    min_prompts = min_effective_teacher_prompts()
    min_coverage = min_teacher_logprob_coverage()
    usable = stats["n_teacher_prompts_with_logprobs"]
    coverage = stats["teacher_logprob_coverage"]
    if usable < min_prompts or coverage < min_coverage:
        raise RuntimeError(
            "API teacher logprob coverage below quality floor: "
            f"{usable}/{stats['n_teacher_prompts_total']} usable "
            f"({coverage:.1%}); require at least {min_prompts} prompts "
            f"and {min_coverage:.0%} coverage"
        )
    return stats
