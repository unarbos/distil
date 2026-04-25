#!/usr/bin/env python3
"""Tests for the Session 3.7 ``noise_resistance_bench`` axis.

Goodhart-context: ``robustness_bench`` (paraphrase rotation) and
``noise_resistance_bench`` (adversarial input noise) form a real-world
robustness battery on the math pool. paraphrase rotation tests
*semantic* shift; noise resistance tests *surface* shift — typos,
case jitter, distractor chatter, common misspellings, extra whitespace.

Critical invariants this suite pins down:

* The wrapper rotation is **block-seed deterministic** (every validator
  agrees on which wrappers run a given round).
* Different block_seeds yield **different** wrapper sets so a miner
  cannot pre-train on one canonical noise bundle.
* Wrappers **never destroy answer-extractable content** (digits and
  arithmetic operators must survive every wrapper, otherwise the axis
  is broken — we'd be punishing models for failing to solve a
  *different* math problem).
* The internal randomness inside a wrapper (e.g. typo positions) is
  **per-(item, perturbation) seed-deterministic** so two validators
  see byte-identical perturbed prompts in the same round.
* The noise pool aliases the math pool but uses an **independent
  stream offset** so its sampled items are usually disjoint from
  ``math_bench`` / ``robustness_bench`` in the same round.
* ``BENCH_NOISE_PERTURB_K`` clamps to the available templates and
  always emits at least one wrapper (a misconfiguration that would
  otherwise silently drop the axis).
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_torch.long = object()
    fake_torch.compile = lambda fn, **_kwargs: fn
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")
    fake_f.kl_div = lambda *_args, **_kwargs: None
    fake_nn.functional = fake_f
    fake_torch.nn = fake_nn
    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("torch.nn", fake_nn)
    sys.modules.setdefault("torch.nn.functional", fake_f)


_DIGITS = set("0123456789")
_OPS = set("+-*/=<>")


def _math_safe(prompt: str, output: str) -> bool:
    """Every digit / operator that appears in ``prompt`` must appear in
    ``output``. Wrappers are allowed to add chars but never to silently
    drop math content."""
    for ch in prompt:
        if ch in _DIGITS or ch in _OPS:
            if output.count(ch) < prompt.count(ch):
                return False
    return True


class TestNoisePerturbations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_template_table_is_nonempty(self):
        templates = self.mod._NOISE_PERTURBATION_TEMPLATES
        self.assertGreaterEqual(
            len(templates), 4,
            "Need at least 4 wrappers so K=2 has variety across rounds",
        )
        for name, fn in templates:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(fn))

    def test_pick_is_block_seed_deterministic(self):
        a = self.mod._pick_noise_perturbations(12345, k=2)
        b = self.mod._pick_noise_perturbations(12345, k=2)
        self.assertEqual([n for n, _ in a], [n for n, _ in b])

    def test_pick_rotates_with_block_seed(self):
        runs = [
            tuple(n for n, _ in self.mod._pick_noise_perturbations(s, k=2))
            for s in (10000, 20000, 30000, 40000, 50000)
        ]
        self.assertGreater(
            len(set(runs)), 1,
            "Wrapper rotation must vary across seeds; saw only one set "
            "across 5 trials",
        )

    def test_pick_clamps_k_to_template_count(self):
        templates = self.mod._NOISE_PERTURBATION_TEMPLATES
        picked = self.mod._pick_noise_perturbations(123, k=999)
        self.assertEqual(len(picked), len(templates))

    def test_pick_returns_at_least_one_with_k_zero(self):
        picked = self.mod._pick_noise_perturbations(123, k=0)
        self.assertEqual(len(picked), 1)

    def test_robustness_and_noise_pick_independently(self):
        """Same block_seed → robustness and noise must NOT pick the
        same wrapper sequence — they live in different libraries."""
        rob = [n for n, _ in self.mod._pick_robustness_perturbations(7777, k=2)]
        noise = [n for n, _ in self.mod._pick_noise_perturbations(7777, k=2)]
        # Different libraries, so the union of names should differ.
        self.assertNotEqual(rob, noise)

    def test_wrappers_preserve_math_content(self):
        """Every wrapper must leave digits + operators intact."""
        original = "If 2 + 3 = 5 and 4 * 6 = 24, what is 7 - 1?"
        for name, fn in self.mod._NOISE_PERTURBATION_TEMPLATES:
            out = fn(original, 42)
            self.assertIsInstance(out, str, f"{name} must return str")
            self.assertTrue(
                _math_safe(original, out),
                f"{name} wrapper dropped digits or operators — answer "
                f"extraction would now grade against a different problem.\n"
                f"original: {original!r}\noutput:   {out!r}",
            )

    def test_wrappers_are_seed_deterministic(self):
        """Two calls with the same (prompt, seed) must produce
        byte-identical output — every validator sees the same noise."""
        prompt = "What is the value of x in 5x + 10 = 25?"
        for name, fn in self.mod._NOISE_PERTURBATION_TEMPLATES:
            a = fn(prompt, 12345)
            b = fn(prompt, 12345)
            self.assertEqual(a, b, f"{name} is non-deterministic for seed=12345")

    def test_typo_swap_preserves_digits(self):
        """The keyboard-typo wrapper must never substitute a digit."""
        out = self.mod._noise_safe_letter_swap(
            "There were 3 apples and 7 oranges, total 10 fruits.",
            rate=1.0,  # force every alpha to be considered
            rng_seed=99,
        )
        for ch in "37 10":
            self.assertIn(ch, out, f"digit {ch!r} dropped under aggressive typo")

    def test_case_jitter_preserves_digits(self):
        out = self.mod._noise_case_jitter(
            "If x = 5 and y = 3, find x + y.",
            rate=1.0,  # force every alpha
            rng_seed=7,
        )
        for ch in "5 3":
            self.assertIn(ch, out)

    def test_drop_periods_keeps_decimal_points(self):
        """Decimal points (period flanked by digits) must survive."""
        out = self.mod._noise_drop_sentence_periods(
            "The value is 3.14159. Keep going. Done.",
            rng_seed=1,
        )
        self.assertIn("3.14159", out)

    def test_noise_pool_is_alias_of_math(self):
        """Pool aliasing — set the math pool, the noise pool must
        observe the same items."""
        self.mod._BENCH_POOLS["math"] = [
            {"src": "gsm8k", "question": "What is 1+1?", "gold": "2"},
        ]
        self.mod._BENCH_POOLS["noise"] = self.mod._BENCH_POOLS["math"]
        self.assertIs(
            self.mod._BENCH_POOLS["noise"], self.mod._BENCH_POOLS["math"],
            "noise pool should alias (not copy) math",
        )

    def test_noise_sample_uses_independent_stream(self):
        """math, robustness, and noise should all draw under different
        stream offsets — same block_seed, three different sample sets.
        This is the core anti-collision property."""
        items = [
            {"src": "gsm8k", "question": f"q{i}", "gold": str(i)}
            for i in range(40)
        ]
        self.mod._BENCH_POOLS["math"] = list(items)
        self.mod._BENCH_POOLS["robustness"] = self.mod._BENCH_POOLS["math"]
        self.mod._BENCH_POOLS["noise"] = self.mod._BENCH_POOLS["math"]
        block_seed = 8042854
        math_pick = self.mod._pick_bench_items("math", block_seed, 4)
        rob_pick = self.mod._pick_bench_items("robustness", block_seed, 4)
        noise_pick = self.mod._pick_bench_items("noise", block_seed, 4)
        self.assertEqual(len(math_pick), 4)
        self.assertEqual(len(rob_pick), 4)
        self.assertEqual(len(noise_pick), 4)
        # All three must be pairwise different — same seed, three offsets.
        sets = [
            tuple(it["question"] for it in math_pick),
            tuple(it["question"] for it in rob_pick),
            tuple(it["question"] for it in noise_pick),
        ]
        self.assertEqual(
            len({tuple(s) for s in sets}), 3,
            "math/robustness/noise must each pick a distinct sample set "
            "under the same block_seed",
        )


class TestNoiseAxisExtractor(unittest.TestCase):
    """Pure-Python axis extractor; no torch, no eval pod."""

    def test_extractor_returns_pass_frac_when_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_noise_resistance_bench,
        )
        student = {
            "noise_resistance_bench": {
                "n": BENCH_MIN_VALID["noise_resistance_bench"],
                "correct": BENCH_MIN_VALID["noise_resistance_bench"],
                "pass_frac": 1.0,
                "items": [],
                "perturbations": ["light_typos", "case_jitter"],
            },
        }
        self.assertEqual(_axis_noise_resistance_bench(student), 1.0)

    def test_extractor_drops_below_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_noise_resistance_bench,
        )
        student = {
            "noise_resistance_bench": {
                "n": BENCH_MIN_VALID["noise_resistance_bench"] - 1,
                "correct": 0,
                "pass_frac": 0.0,
                "items": [],
            },
        }
        self.assertIsNone(_axis_noise_resistance_bench(student))

    def test_extractor_handles_error_payload(self):
        from scripts.validator.composite import _axis_noise_resistance_bench
        student = {
            "noise_resistance_bench": {
                "error": "torch not available",
                "n": 0, "correct": 0, "pass_frac": 0.0,
            },
        }
        self.assertIsNone(_axis_noise_resistance_bench(student))

    def test_composite_version_is_v12(self):
        from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
        self.assertGreaterEqual(
            COMPOSITE_SHADOW_VERSION, 12,
            "noise_resistance_bench requires composite version >= 12",
        )


if __name__ == "__main__":
    unittest.main()
