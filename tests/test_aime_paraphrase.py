#!/usr/bin/env python3
"""Regression tests for the round-21 AIME problem paraphrase.

Pre-v21 ``aime_bench`` used the canonical AIME problem wording verbatim.
Pool is ~90 public items from ``HuggingFaceH4/aime_2025`` +
``Maxwell-Jia/AIME_2024`` + ``AI-MO/aimo-validation-aime`` with integer
answers 0–999. A miner who pre-trains on the public datasets can build a
``{problem_text → answer}`` lookup keyed on canonical wording.

v21+ wraps each AIME problem with the same math-domain-safe paraphrase
helpers ``robustness_bench`` already uses. These tests pin the contract:

* ``_paraphrase_aime_problem`` is deterministic per (question, block_seed).
* The paraphrase rotates per block_seed (different round → different
  wording) AT LEAST SOMETIMES (the synonym table is small so some
  problems will paraphrase identically).
* Numbers, LaTeX (``$...$``, ``\\boxed{}``), and ``####`` markers are
  preserved verbatim — the math content survives untouched.
* ``block_seed=None`` returns the question unchanged.
* Both helpers (``_apply_instruction_synonyms`` and
  ``_imperative_to_question``) get a chance to fire on real-shape AIME
  problems.
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


# Realistic AIME problem stems. Mix imperatives (Find …) with non-
# imperative ("Let S be …") so we exercise both paraphrase paths.
_AIME_SAMPLES = (
    "Find the number of ordered pairs $(m, n)$ of positive integers such that $m^2 + n^2 < 100$.",
    "Determine the smallest positive integer $N$ such that the digit sum of $N$ equals 21.",
    "Calculate the value of $\\sum_{k=1}^{1000} \\lfloor \\sqrt{k} \\rfloor$.",
    "Compute the number of polynomials $P(x)$ with integer coefficients such that $P(2)=5$ and $P(3)=7$.",
    "Let $S$ be the set of all positive integers $n$ for which $n^2 + n + 41$ is prime. "
    "Find the smallest element of $S$.",
)


class TestAimeParaphraseDeterminism(unittest.TestCase):
    def test_same_seed_same_paraphrase(self):
        """Cross-validator agreement: same (question, block_seed) → same output."""
        for q in _AIME_SAMPLES:
            a = pev._paraphrase_aime_problem(q, block_seed=12345)
            b = pev._paraphrase_aime_problem(q, block_seed=12345)
            self.assertEqual(a, b)

    def test_input_not_mutated(self):
        for q in _AIME_SAMPLES:
            snap = q
            pev._paraphrase_aime_problem(q, block_seed=42)
            self.assertEqual(q, snap)


class TestAimeParaphraseRotation(unittest.TestCase):
    """Across many seeds, at least some questions get a different wording."""

    def test_at_least_some_rotation(self):
        """At least one of the realistic samples paraphrases differently
        between seeds A and B for some pair (A, B). Synonym table is
        small so universal rotation isn't guaranteed, but some rotation
        must be observable to confirm the wiring is hot."""
        any_rotated = False
        for q in _AIME_SAMPLES:
            seen = set()
            for seed in range(0, 200, 7):
                seen.add(pev._paraphrase_aime_problem(q, block_seed=seed))
            if len(seen) >= 2:
                any_rotated = True
                break
        self.assertTrue(
            any_rotated,
            "no AIME question rotated wording across 30 seeds — paraphrase wiring may be cold",
        )

    def test_imperative_questions_get_at_least_two_variants(self):
        """A typical imperative-shape AIME problem ('Find ...') should
        produce at least two distinct wordings across many seeds."""
        q = _AIME_SAMPLES[0]  # 'Find the number of ordered pairs ...'
        seen = set()
        for seed in range(0, 256):
            seen.add(pev._paraphrase_aime_problem(q, block_seed=seed))
        self.assertGreaterEqual(
            len(seen), 2,
            f"only {len(seen)} variants from imperative AIME — paraphrase too narrow",
        )


class TestAimeParaphrasePreservesMath(unittest.TestCase):
    """The math content (numbers, LaTeX, boxed format) must survive
    untouched. Otherwise we'd change the answer along with the wording."""

    def test_digits_preserved(self):
        """All digit runs in the original survive the paraphrase."""
        for q in _AIME_SAMPLES:
            for seed in (1, 2, 3, 100, 9999):
                out = pev._paraphrase_aime_problem(q, block_seed=seed)
                self.assertEqual(
                    re.findall(r"\d+", q),
                    re.findall(r"\d+", out),
                    f"digit runs changed for seed={seed}, q={q!r}, out={out!r}",
                )

    def test_latex_inline_preserved(self):
        """LaTeX inline math (``$...$``) survives the paraphrase."""
        for q in _AIME_SAMPLES:
            for seed in (1, 42, 7777):
                out = pev._paraphrase_aime_problem(q, block_seed=seed)
                # The same set of $-delimited tokens must appear in
                # both — order can shift if the imperative rewrite
                # moves "the body" but the set is invariant.
                self.assertEqual(
                    sorted(re.findall(r"\$[^$]+\$", q)),
                    sorted(re.findall(r"\$[^$]+\$", out)),
                    f"LaTeX changed: seed={seed}, q={q!r}, out={out!r}",
                )

    def test_boxed_format_preserved(self):
        """``\\boxed{...}`` must not be touched."""
        q = "Compute the value. Express your answer as $\\boxed{m+n}$."
        out = pev._paraphrase_aime_problem(q, block_seed=42)
        self.assertIn("\\boxed{m+n}", out)


class TestAimeParaphraseEdgeCases(unittest.TestCase):
    def test_block_seed_none_passes_through(self):
        """Dev/replay mode: no seed → no rotation."""
        for q in _AIME_SAMPLES:
            self.assertEqual(pev._paraphrase_aime_problem(q, block_seed=None), q)

    def test_empty_string(self):
        out = pev._paraphrase_aime_problem("", block_seed=42)
        self.assertEqual(out, "")

    def test_no_imperative_no_synonym_passes_through(self):
        """A non-imperative problem with no synonym-table words returns
        unchanged. The paraphrase helpers degrade gracefully."""
        # No imperative verb (Find/Calculate/Compute/Determine) and no
        # words from the synonym table (find/calculate/compute/
        # determine/solve/the question/the problem/answer/each/every/
        # total/how many/how much/what is) appear here.
        q = "Suppose $x$ is a positive real. Express your reply modulo 1000."
        out = pev._paraphrase_aime_problem(q, block_seed=12345)
        self.assertEqual(out, q)

    def test_hex_block_seed(self):
        """Hex-string block_seed works (canonical bittensor format)."""
        q = _AIME_SAMPLES[0]
        out = pev._paraphrase_aime_problem(q, block_seed="0xdeadbeef")
        self.assertEqual(re.findall(r"\d+", q), re.findall(r"\d+", out))


if __name__ == "__main__":
    unittest.main()
