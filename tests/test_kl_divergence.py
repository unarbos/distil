"""Tests for KL-divergence computation."""

import math

import pytest

from eval.kl_divergence import compute_kl_divergence


class TestComputeKLDivergence:
    """Test suite for the per-token KL-divergence calculator."""

    def test_identical_distributions_give_zero(self):
        """KL(P || P) == 0 for any distribution P."""
        teacher = [{"a": math.log(0.7), "b": math.log(0.3)}]
        student = [{"a": math.log(0.7), "b": math.log(0.3)}]
        kl = compute_kl_divergence(teacher, student)
        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_known_kl_value(self):
        """Verify against a hand-computed KL-divergence."""
        # P = [0.5, 0.5], Q = [0.25, 0.75]
        # KL(P||Q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
        #          = 0.5*ln(2) + 0.5*ln(2/3)
        #          ≈ 0.5*0.6931 + 0.5*(-0.4055) ≈ 0.1438
        teacher = [{"x": math.log(0.5), "y": math.log(0.5)}]
        student = [{"x": math.log(0.25), "y": math.log(0.75)}]
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        kl = compute_kl_divergence(teacher, student)
        assert kl == pytest.approx(expected, rel=1e-6)

    def test_multiple_positions_averaged(self):
        """KL should be averaged across token positions."""
        pos1_t = {"a": math.log(0.9), "b": math.log(0.1)}
        pos1_s = {"a": math.log(0.9), "b": math.log(0.1)}  # identical → 0
        pos2_t = {"a": math.log(0.5), "b": math.log(0.5)}
        pos2_s = {"a": math.log(0.25), "b": math.log(0.75)}

        kl = compute_kl_divergence([pos1_t, pos2_t], [pos1_s, pos2_s])
        # Position 1 contributes ~0, position 2 contributes ~0.1438
        # Average should be ~0.0719
        assert kl > 0
        assert kl < 0.2

    def test_empty_returns_inf(self):
        """Empty logprobs → inf (no data)."""
        assert compute_kl_divergence([], []) == float("inf")
        assert compute_kl_divergence([], [{"a": -1.0}]) == float("inf")

    def test_missing_tokens_handled(self):
        """Tokens present in teacher but not student (and vice versa) are handled."""
        teacher = [{"a": math.log(0.9), "b": math.log(0.1)}]
        student = [{"a": math.log(0.8), "c": math.log(0.2)}]
        kl = compute_kl_divergence(teacher, student)
        assert kl >= 0
        assert math.isfinite(kl)

    def test_non_negative(self):
        """KL divergence is always ≥ 0."""
        import random

        random.seed(42)
        for _ in range(20):
            n_tokens = random.randint(2, 10)
            tokens = [f"t{i}" for i in range(n_tokens)]
            t_vals = [random.random() for _ in tokens]
            s_vals = [random.random() for _ in tokens]
            t_total = sum(t_vals)
            s_total = sum(s_vals)
            teacher = [{t: math.log(v / t_total) for t, v in zip(tokens, t_vals)}]
            student = [{t: math.log(v / s_total) for t, v in zip(tokens, s_vals)}]
            kl = compute_kl_divergence(teacher, student)
            assert kl >= -1e-10  # allow tiny floating-point noise

    def test_length_mismatch_uses_minimum(self):
        """When teacher and student have different lengths, use the shorter."""
        teacher = [
            {"a": math.log(0.7), "b": math.log(0.3)},
            {"a": math.log(0.5), "b": math.log(0.5)},
            {"a": math.log(0.6), "b": math.log(0.4)},
        ]
        student = [
            {"a": math.log(0.7), "b": math.log(0.3)},
        ]
        kl = compute_kl_divergence(teacher, student)
        # Only 1 position compared → should be ~0 (identical first position)
        assert kl == pytest.approx(0.0, abs=1e-8)
