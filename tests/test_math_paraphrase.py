#!/usr/bin/env python3
"""Regression tests for the round-22 math_bench paraphrase.

Pre-v22 ``math_bench`` (GSM8K + MATH-500) used the canonical problem
wording verbatim. The pool is 1 819 public items (1 319 GSM8K test +
500 MATH-500). A miner who pre-trains on the public datasets can build
a ``{problem_text → answer}`` lookup keyed on canonical wording and
saturate the largest single bench weight (0.12) without doing math.

v22+ extends the round-21 AIME paraphrase recipe to math_bench AND the
two math-derived axes (``tool_use_bench`` reuses numerically-tractable
math items; ``self_consistency_bench`` reuses hard math items). These
tests pin the contract:

* ``_paraphrase_math_problem`` is deterministic per (question, block_seed).
* ``_paraphrase_aime_problem`` remains as a backwards-compatible alias.
* GSM8K-style natural-language items survive paraphrase (numbers and
  ``####`` markers preserved).
* MATH-500-style LaTeX items survive paraphrase (LaTeX blocks and
  ``\\boxed{}`` markers preserved).
* The paraphrase rotates per block_seed (different round → different
  wording) AT LEAST SOMETIMES.
"""

import os
import re
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    import types
    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = lambda *a, **kw: __import__("contextlib").nullcontext()
    torch_mod.manual_seed = lambda *a, **kw: None
    for fn in ("float32", "bfloat16", "float16"):
        setattr(torch_mod, fn, fn)
    torch_mod.tensor = lambda *a, **kw: None
    torch_mod.is_tensor = lambda x: False
    torch_mod.empty_cache = lambda *a, **kw: None

    class _R:
        @staticmethod
        def get_rng_state():
            return None

        @staticmethod
        def set_rng_state(s):
            return None

    torch_mod.random = _R()
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *a, **kw: None,
        empty_cache=lambda *a, **kw: None,
    )
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_mod.functional

    transformers_mod = types.ModuleType("transformers")
    for cls in ("AutoTokenizer", "AutoModelForCausalLM", "AutoConfig"):
        setattr(
            transformers_mod, cls,
            type(cls, (), {"from_pretrained": classmethod(lambda *a, **kw: None)}),
        )
    sys.modules["transformers"] = transformers_mod

    bittensor_mod = types.ModuleType("bittensor")
    bittensor_mod.logging = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    sys.modules["bittensor"] = bittensor_mod


_install_torch_stub()
import pod_eval_vllm as pev  # noqa: E402


# Realistic GSM8K-style items (natural-language word problems with no
# LaTeX and a ``####`` final-answer marker after the reasoning trace).
# In practice the dataset stores question + answer separately, so the
# question alone is the natural-language stem; the ``####`` lives in
# the answer field. Test against the question text the model sees.
_GSM8K_SAMPLES = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total does it take?",
    "Find the number of marbles in the jar if Sarah has 12 red marbles, "
    "twice as many blue marbles, and 5 fewer green marbles than blue.",
    "Calculate the total cost when each apple costs $0.50 and Maria buys "
    "3 dozen apples.",
)

# MATH-500-style items: LaTeX-heavy, often with ``\\boxed{}`` answer or
# ``$ … $`` inline math. Imperative closing sentence is common.
_MATH500_SAMPLES = (
    r"Let $a, b, c$ be positive real numbers with $a + b + c = 1$. "
    r"Find the maximum value of $\sqrt{a} + \sqrt{b} + \sqrt{c}$.",
    r"Determine all integers $n$ such that $\frac{n+5}{n-1}$ is also an integer.",
    r"Compute $\sum_{k=1}^{100} \frac{1}{k(k+1)}$.",
    r"What is the value of $\binom{10}{3}$?",
    r"Find the area of the triangle with vertices $(0, 0)$, $(4, 0)$, and "
    r"$(2, 3)$, expressed as $\boxed{N}$ for an integer $N$.",
)


class TestMathParaphraseDeterminism(unittest.TestCase):
    def test_same_seed_same_paraphrase(self):
        """Cross-validator agreement: same (question, block_seed) → same output."""
        for q in _GSM8K_SAMPLES + _MATH500_SAMPLES:
            a = pev._paraphrase_math_problem(q, block_seed=12345)
            b = pev._paraphrase_math_problem(q, block_seed=12345)
            self.assertEqual(a, b)

    def test_aime_alias_still_works(self):
        """The round-21 ``_paraphrase_aime_problem`` name must keep
        working as an alias so existing imports don't break."""
        for q in _GSM8K_SAMPLES + _MATH500_SAMPLES:
            self.assertEqual(
                pev._paraphrase_aime_problem(q, block_seed=99),
                pev._paraphrase_math_problem(q, block_seed=99),
            )

    def test_input_not_mutated(self):
        for q in _GSM8K_SAMPLES + _MATH500_SAMPLES:
            snap = q
            pev._paraphrase_math_problem(q, block_seed=42)
            self.assertEqual(q, snap)

    def test_none_block_seed_passes_through(self):
        for q in _GSM8K_SAMPLES + _MATH500_SAMPLES:
            self.assertEqual(pev._paraphrase_math_problem(q, block_seed=None), q)

    def test_empty_string_passes_through(self):
        self.assertEqual(pev._paraphrase_math_problem("", block_seed=42), "")


class TestMathParaphrasePreservesNumbers(unittest.TestCase):
    """The math content (numbers, LaTeX, currency) must survive
    untouched. Otherwise we'd change the answer along with the wording."""

    def test_digits_preserved_gsm8k(self):
        """All digit runs in GSM8K-style questions survive."""
        for q in _GSM8K_SAMPLES:
            original_nums = re.findall(r"\d+", q)
            for seed in (1, 2, 3, 17, 199):
                paraphrased = pev._paraphrase_math_problem(q, block_seed=seed)
                paraphrased_nums = re.findall(r"\d+", paraphrased)
                self.assertEqual(
                    sorted(original_nums), sorted(paraphrased_nums),
                    f"digits changed under paraphrase at seed={seed}\n"
                    f"  input : {q}\n  output: {paraphrased}",
                )

    def test_currency_preserved_gsm8k(self):
        """Currency markers like $0.50 / $2 survive verbatim."""
        for q in _GSM8K_SAMPLES:
            currencies = re.findall(r"\$\d+(?:\.\d+)?", q)
            if not currencies:
                continue
            for seed in (1, 2, 17, 99):
                paraphrased = pev._paraphrase_math_problem(q, block_seed=seed)
                for c in currencies:
                    self.assertIn(c, paraphrased)

    def test_latex_blocks_preserved(self):
        """``$...$`` and ``\\boxed{...}`` blocks survive verbatim."""
        for q in _MATH500_SAMPLES:
            inline_math = re.findall(r"\$[^$]+\$", q)
            for seed in (1, 17, 199):
                paraphrased = pev._paraphrase_math_problem(q, block_seed=seed)
                paraphrased_inline = re.findall(r"\$[^$]+\$", paraphrased)
                self.assertEqual(
                    sorted(inline_math), sorted(paraphrased_inline),
                    f"inline LaTeX changed at seed={seed}\n"
                    f"  input : {q}\n  output: {paraphrased}",
                )

    def test_boxed_format_preserved(self):
        """``\\boxed{...}`` markers must survive paraphrase verbatim."""
        for q in _MATH500_SAMPLES:
            if r"\boxed" not in q:
                continue
            for seed in (1, 17, 199):
                paraphrased = pev._paraphrase_math_problem(q, block_seed=seed)
                self.assertIn(
                    r"\boxed", paraphrased,
                    f"\\boxed marker dropped at seed={seed}",
                )


class TestMathParaphraseRotation(unittest.TestCase):
    """Across many seeds, at least some questions get a different
    wording. Synonym table is small so not every problem rotates, but
    some rotation must be observable to confirm wiring is hot."""

    def test_at_least_some_rotation_gsm8k(self):
        any_rotated = False
        for q in _GSM8K_SAMPLES:
            seen = set()
            for seed in range(0, 200, 7):
                seen.add(pev._paraphrase_math_problem(q, block_seed=seed))
            if len(seen) >= 2:
                any_rotated = True
                break
        self.assertTrue(
            any_rotated,
            "no GSM8K-style question rotated wording across 30 seeds — "
            "paraphrase wiring may be cold",
        )

    def test_at_least_some_rotation_math500(self):
        any_rotated = False
        for q in _MATH500_SAMPLES:
            seen = set()
            for seed in range(0, 200, 7):
                seen.add(pev._paraphrase_math_problem(q, block_seed=seed))
            if len(seen) >= 2:
                any_rotated = True
                break
        self.assertTrue(
            any_rotated,
            "no MATH-500-style question rotated wording across 30 seeds — "
            "paraphrase wiring may be cold",
        )

    def test_imperative_gsm8k_gets_multiple_variants(self):
        """A typical imperative-shape GSM8K problem ('Find ...') should
        produce at least two distinct wordings across many seeds."""
        q = _GSM8K_SAMPLES[2]  # "Find the number of marbles ..."
        seen = set()
        for seed in range(0, 256):
            seen.add(pev._paraphrase_math_problem(q, block_seed=seed))
        self.assertGreaterEqual(
            len(seen), 2,
            f"only {len(seen)} variants from imperative GSM8K — "
            f"paraphrase too narrow",
        )


class TestMathParaphraseRobustness(unittest.TestCase):
    """Edge cases that historically broke the paraphrase pipeline."""

    def test_handles_latex_with_backslashes(self):
        """Bug discovered round 21: ``\\sqrt{k}`` in body triggered
        ``re.error: bad escape \\s`` in ``_imperative_to_question``.
        Test pins the fix."""
        q = r"Find the smallest positive integer $k$ such that $\sqrt{k}$ is rational."
        for seed in range(0, 64):
            try:
                pev._paraphrase_math_problem(q, block_seed=seed)
            except Exception as e:  # pragma: no cover - regression fence
                self.fail(f"paraphrase raised at seed={seed}: {e}")

    def test_handles_dollar_amount_no_collision(self):
        """``$2`` is currency, ``$x$`` is LaTeX; the paraphrase must
        not confuse them."""
        q = "How much money does Maria save if each apple costs $0.50 and she buys 8?"
        for seed in range(0, 32):
            paraphrased = pev._paraphrase_math_problem(q, block_seed=seed)
            self.assertIn("$0.50", paraphrased)
            self.assertIn("8", paraphrased)


class TestMathParaphraseAppliedAtRoundStart(unittest.TestCase):
    """``set_bench_block_seed`` must wire the paraphrase into the math /
    tool_use / self_consistency sample lists. Without this the helper
    is wired but never called from the live eval path."""

    def test_math_samples_mutate_per_round(self):
        """Manually populate the math pool, set two different
        block_seeds, and confirm at least some samples differ."""
        # Cleanly install a small synthetic math pool and reset state.
        pev._BENCH_POOLS["math"] = [
            {
                "src": "gsm8k",
                "question": "Find the total cost when each apple costs 2 dollars and Sue buys 5.",
                "gold": "10",
            },
            {
                "src": "gsm8k",
                "question": "Calculate the area of a circle with radius 7. Express the answer in square units.",
                "gold": "153.94",
            },
            {
                "src": "math500",
                "question": (
                    r"Let $x$ be a positive integer. Determine the number of integer "
                    r"solutions to $x^2 < 100$."
                ),
                "gold": "9",
            },
        ]
        # Pre-populate the other pools the round-start path expects so
        # the iteration doesn't crash on absent keys.
        for k in (
            "code", "reasoning", "knowledge", "ifeval", "aime", "mbpp",
            "tool_use", "self_consistency", "arc", "truthful", "robustness",
            "noise",
        ):
            pev._BENCH_POOLS.setdefault(k, [])
        # Fresh per-round state so set_bench_block_seed actually picks.
        pev._BENCH_BLOCK_SEED = None
        for k in list(pev._BENCH_SAMPLES.keys()):
            pev._BENCH_SAMPLES[k] = []
        # Force the bench-load path to skip (we already populated pools).
        pev._BENCH_POOLS_LOADED = True

        try:
            pev.set_bench_block_seed(11111)
            samples_a = list(pev._BENCH_SAMPLES["math"])
            pev.set_bench_block_seed(22222)
            samples_b = list(pev._BENCH_SAMPLES["math"])
        finally:
            # Reset so other tests don't see our synthetic pool.
            pev._BENCH_POOLS["math"] = []
            for k in list(pev._BENCH_SAMPLES.keys()):
                pev._BENCH_SAMPLES[k] = []
            pev._BENCH_BLOCK_SEED = None
            pev._BENCH_POOLS_LOADED = False

        # At least one of the chosen items should have rotated wording
        # between the two block seeds (helper is hot).
        a_questions = {it["question"] for it in samples_a}
        b_questions = {it["question"] for it in samples_b}
        self.assertNotEqual(
            a_questions, b_questions,
            "math_bench samples did not rotate between block_seeds — "
            "paraphrase wiring NOT hot from set_bench_block_seed",
        )


if __name__ == "__main__":
    unittest.main()
