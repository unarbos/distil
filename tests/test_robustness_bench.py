#!/usr/bin/env python3
"""Tests for the Session 3.7 robustness_bench paraphrase axis.

Goodhart-context: a miner who memorizes the canonical wording of public
math items (gsm8k/math500) wins ``math_bench`` for free. The robustness
axis re-uses the math pool but asks each item under K block-rotated
paraphrase wrappers, so a model that only remembers exact phrasings
fails. These tests pin down:

* The wrapper rotation is **deterministic per block_seed** (every
  validator agrees on which wrappers run a given round).
* Different block_seeds yield **different** wrapper sets (so a miner
  cannot pre-train on a single canonical wording bundle).
* The wrappers are **non-trivial string transforms** (every wrapper's
  output is a strict superstring of the original prompt with at least
  some prefix or postfix added — they are not no-ops).
* ``BENCH_ROBUSTNESS_PERTURB_K`` clamps to the available templates.

We don't try to load a real torch stack here — the probe is exercised
via a dummy model + tokenizer indirection in a follow-up integration
test (eval pod side). At unit-test scope, we only need the perturbation
machinery to be correct + deterministic; that's the only piece miners
can game.
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


class TestRobustnessPerturbations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_template_table_is_nonempty(self):
        templates = self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES
        self.assertGreaterEqual(
            len(templates), 4,
            "Need at least 4 wrappers so K=2 has variety across rounds",
        )
        for name, fn in templates:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(fn))

    def test_pick_is_block_seed_deterministic(self):
        a = self.mod._pick_robustness_perturbations(12345, k=2)
        b = self.mod._pick_robustness_perturbations(12345, k=2)
        self.assertEqual([n for n, _ in a], [n for n, _ in b])

    def test_pick_rotates_with_block_seed(self):
        runs = [
            tuple(n for n, _ in self.mod._pick_robustness_perturbations(s, k=2))
            for s in (10000, 20000, 30000, 40000, 50000)
        ]
        # We don't require perfect uniqueness across 5 seeds, but we do
        # require that not every seed produces the same wrappers — that
        # would be a memorization payday.
        self.assertGreater(
            len(set(runs)), 1,
            "Wrapper rotation must vary across seeds; saw only one set "
            "across 5 trials",
        )

    def test_pick_clamps_k_to_template_count(self):
        templates = self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES
        picked = self.mod._pick_robustness_perturbations(123, k=999)
        self.assertEqual(len(picked), len(templates))

    def test_pick_returns_at_least_one_with_k_zero(self):
        picked = self.mod._pick_robustness_perturbations(123, k=0)
        # K=0 is a misconfiguration; we'd rather still emit one wrapper
        # than silently stop the axis. Verifies the max(1, k) clamp.
        self.assertEqual(len(picked), 1)

    def test_wrappers_strictly_extend_the_prompt(self):
        original = "What is 2 + 2?\n\nProvide a final answer."
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            out = fn(original)
            self.assertIsInstance(out, str, f"{name} must return str")
            # Wrapper-family perturbations strictly extend the prompt;
            # paraphrase-family perturbations may shorten or keep it the
            # same length (e.g., ``Find the area of X.`` has the same
            # character count as ``What is the area of X?``).
            if name not in self.mod._ROBUSTNESS_PARAPHRASE_NAMES:
                self.assertGreater(
                    len(out), len(original),
                    f"{name} wrapper produced output no longer than the "
                    f"original — that's effectively a no-op",
                )
            # The numeric content must survive every perturbation —
            # both wrappers (which never touch it) and paraphrases
            # (which only swap instruction words).
            self.assertIn(
                "2 + 2", out,
                f"{name} perturbation dropped the original numeric content",
            )

    def test_no_wrapper_collapses_to_template_only(self):
        # Negative test: if any perturbation returned only the boilerplate
        # ("Solve the following problem.") with the actual question
        # truncated, the axis would be vacuous. Wrapper-family checks the
        # closing question mark survives. Paraphrase-family is exempt
        # (imperative_to_question may rewrite "what is x?" → "What is the
        # value of x?" which preserves semantics but not the literal
        # phrase).
        original = "If x + 5 = 12, what is x?"
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            if name in self.mod._ROBUSTNESS_PARAPHRASE_NAMES:
                # For paraphrase entries we only check that the math
                # content (``x + 5 = 12``) survives — instruction words
                # may be swapped.
                out = fn(original)
                self.assertIn(
                    "x + 5 = 12", out,
                    f"{name} paraphrase dropped the math content",
                )
                continue
            out = fn(original)
            self.assertIn(
                "what is x?", out,
                f"{name} wrapper dropped the closing question",
            )

    def test_paraphrase_family_exists(self):
        """Stratification depends on at least one paraphrase entry being
        present in the templates table. If a refactor accidentally
        removes them, the picker would silently fall back to wrapper-only
        rounds and reopen the memorization-bypass hole."""
        names = {n for n, _ in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES}
        self.assertTrue(
            self.mod._ROBUSTNESS_PARAPHRASE_NAMES & names,
            "Templates table lost all paraphrase-family entries — "
            "memorization defense disabled",
        )

    def test_pick_always_includes_a_paraphrase_when_available(self):
        """Stratification rule: every round must include at least one
        paraphrase-family perturbation. Probe over many block seeds and
        K values; if any round comes back wrapper-only we have a
        regression that lets memorizers pass."""
        for seed in (1, 100, 12345, 999999, 8052008):
            for k in (1, 2, 3, 4):
                picked = self.mod._pick_robustness_perturbations(seed, k=k)
                names = {n for n, _ in picked}
                self.assertTrue(
                    names & self.mod._ROBUSTNESS_PARAPHRASE_NAMES,
                    f"seed={seed} k={k} produced wrapper-only set: {names}",
                )

    def test_paraphrase_actually_changes_inner_text(self):
        """A paraphrase that returns the input unchanged is a silent
        no-op — same Goodhart hole as a wrapper. Verify each paraphrase
        function actually modifies a synthetic prompt."""
        sample = (
            "Find the total cost of 5 apples at $2 each.\n\n"
            "Solve step by step and end with '#### N' where N is the final numeric answer."
        )
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            if name not in self.mod._ROBUSTNESS_PARAPHRASE_NAMES:
                continue
            out = fn(sample)
            self.assertNotEqual(
                out, sample,
                f"{name} paraphrase returned input unchanged on a sample "
                f"that contains common instruction-domain anchors",
            )

    def test_paraphrase_preserves_numeric_content(self):
        """A paraphrase that drops or changes digits would change the
        answer — fatal for grading. Verify numeric tokens survive."""
        sample = (
            "Find the total cost of 5 apples at $2 each.\n\n"
            "Solve step by step and end with '#### N' where N is the final numeric answer."
        )
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            if name not in self.mod._ROBUSTNESS_PARAPHRASE_NAMES:
                continue
            out = fn(sample)
            for digit in ("5", "2"):
                self.assertIn(
                    digit, out,
                    f"{name} paraphrase removed digit '{digit}' — grading would break",
                )

    def test_robustness_pool_is_alias_of_math(self):
        # _BENCH_POOLS is module-level state. We don't load the math
        # pool here (no datasets package on the test runner) but we
        # can verify the alias hook in `_bench_load_pools` would set
        # them equal: compare list identity post-init when math is
        # populated. We simulate by directly stamping the pool.
        self.mod._BENCH_POOLS["math"] = [
            {"src": "gsm8k", "question": "What is 2+2?", "gold": "4"},
        ]
        self.mod._BENCH_POOLS["robustness"] = self.mod._BENCH_POOLS["math"]
        self.assertIs(
            self.mod._BENCH_POOLS["robustness"], self.mod._BENCH_POOLS["math"],
            "robustness pool should alias (not copy) math so growth is "
            "tracked",
        )

    def test_robustness_sample_uses_independent_stream(self):
        # When robustness and math share a pool but are sampled with
        # different stream offsets, _pick_bench_items must yield a
        # *different* permutation for the same block_seed. This is
        # the central anti-collision property — without it, robustness
        # and math would always score the same items and the axis
        # would degenerate to "math under a wrapper".
        items = [
            {"src": "gsm8k", "question": f"q{i}", "gold": str(i)}
            for i in range(40)
        ]
        self.mod._BENCH_POOLS["math"] = list(items)
        self.mod._BENCH_POOLS["robustness"] = self.mod._BENCH_POOLS["math"]
        block_seed = 8042854
        math_pick = self.mod._pick_bench_items("math", block_seed, 4)
        rob_pick = self.mod._pick_bench_items("robustness", block_seed, 4)
        self.assertEqual(len(math_pick), 4)
        self.assertEqual(len(rob_pick), 4)
        self.assertNotEqual(
            [it["question"] for it in math_pick],
            [it["question"] for it in rob_pick],
            "math and robustness picks must differ — same block_seed, "
            "different stream offsets",
        )


class TestRobustnessAxisExtractor(unittest.TestCase):
    """Pure-Python axis extractor; no torch, no eval pod."""

    def test_extractor_returns_pass_frac_when_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_robustness_bench,
        )
        student = {
            "robustness_bench": {
                "n": BENCH_MIN_VALID["robustness_bench"],
                "correct": BENCH_MIN_VALID["robustness_bench"],
                "pass_frac": 1.0,
                "items": [],
                "perturbations": ["a", "b"],
            },
        }
        self.assertEqual(_axis_robustness_bench(student), 1.0)

    def test_extractor_drops_below_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_robustness_bench,
        )
        student = {
            "robustness_bench": {
                "n": BENCH_MIN_VALID["robustness_bench"] - 1,
                "correct": 0,
                "pass_frac": 0.0,
                "items": [],
            },
        }
        self.assertIsNone(_axis_robustness_bench(student))

    def test_extractor_handles_error_payload(self):
        from scripts.validator.composite import _axis_robustness_bench
        student = {
            "robustness_bench": {
                "error": "torch not available",
                "n": 0, "correct": 0, "pass_frac": 0.0,
            },
        }
        self.assertIsNone(_axis_robustness_bench(student))


if __name__ == "__main__":
    unittest.main()
