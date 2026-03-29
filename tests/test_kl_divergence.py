"""Unit tests for KL-divergence computation."""
import math
from eval.kl_divergence import compute_kl_divergence


def test_identical_distributions():
    """KL(P || P) should be 0."""
    logprobs = [{"a": math.log(0.7), "b": math.log(0.3)}]
    assert abs(compute_kl_divergence(logprobs, logprobs)) < 1e-10


def test_different_distributions():
    """KL should be positive for different distributions."""
    teacher = [{"a": math.log(0.9), "b": math.log(0.1)}]
    student = [{"a": math.log(0.5), "b": math.log(0.5)}]
    kl = compute_kl_divergence(teacher, student)
    assert kl > 0


def test_empty_returns_inf():
    """Empty logprobs should return inf."""
    assert compute_kl_divergence([], []) == float("inf")


def test_multiple_positions():
    """Average across multiple positions."""
    teacher = [
        {"a": math.log(0.8), "b": math.log(0.2)},
        {"a": math.log(0.6), "b": math.log(0.4)},
    ]
    student = [
        {"a": math.log(0.5), "b": math.log(0.5)},
        {"a": math.log(0.5), "b": math.log(0.5)},
    ]
    kl = compute_kl_divergence(teacher, student)
    assert kl > 0
    assert math.isfinite(kl)
