import pytest

from scripts import eval_prompt_accounting


def test_teacher_logprob_coverage_stats_counts_missing_prompts():
    stats = eval_prompt_accounting.teacher_logprob_coverage_stats(300, 241)

    assert stats == {
        "n_teacher_prompts_total": 300,
        "n_teacher_prompts_with_logprobs": 241,
        "n_teacher_prompts_dropped_missing_logprobs": 59,
        "teacher_logprob_coverage": 0.803333,
    }


def test_validate_teacher_logprob_coverage_accepts_healthy_subset(monkeypatch):
    monkeypatch.setenv("DISTIL_MIN_TEACHER_LOGPROB_COVERAGE", "0.5")
    monkeypatch.setenv("DISTIL_MIN_EFFECTIVE_TEACHER_PROMPTS", "30")

    stats = eval_prompt_accounting.validate_teacher_logprob_coverage(300, 180)

    assert stats["teacher_logprob_coverage"] == 0.6


def test_validate_teacher_logprob_coverage_rejects_degraded_provider(monkeypatch):
    monkeypatch.setenv("DISTIL_MIN_TEACHER_LOGPROB_COVERAGE", "0.5")
    monkeypatch.setenv("DISTIL_MIN_EFFECTIVE_TEACHER_PROMPTS", "30")

    with pytest.raises(RuntimeError, match="below quality floor"):
        eval_prompt_accounting.validate_teacher_logprob_coverage(300, 29)

    with pytest.raises(RuntimeError, match="below quality floor"):
        eval_prompt_accounting.validate_teacher_logprob_coverage(300, 100)
