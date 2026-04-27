#!/usr/bin/env python3
"""Regression tests for the round-23 code_bench / mbpp_bench paraphrase.

Pre-v23 ``code_bench`` (HumanEval, 164 public items) and ``mbpp_bench``
(MBPP+, 378 public items) used the canonical prompt verbatim. Both ship
the gold test harness with the dataset, so a miner who pre-trains on
the public datasets can build a ``{prompt → solution}`` lookup keyed on
the canonical docstring / description and saturate both axes without
ever compiling Python.

v23 introduces ``_paraphrase_code_problem`` — a structurally aware
paraphrase that classifies each prompt line as PROSE or CODE and
rotates ONLY prose lines. These tests pin the contract:

* Determinism: same ``(prompt, block_seed)`` → identical output. Cross-
  validator agreement.
* HumanEval invariants: function signatures, ``>>> doctest`` lines,
  doctest output lines, ``import``/``from`` statements, and bare triple-
  quote markers are preserved verbatim.
* MBPP invariants: ``assert`` lines (the test harness) are preserved
  verbatim; the natural-language description is rotated.
* No identifier corruption: variable / function / module names in code
  lines are untouched even when they contain words from the synonym
  table (e.g. ``find`` in ``"abc".find("b")``).
* Rotation: across many seeds, at least some prompts get a different
  wording (synonym table is small but non-trivially populated).
* Backward-compat: existing math paraphrase still works (no regression
  from the ``extra_table`` extension).
* Round-start wiring: ``set_bench_block_seed`` applies the paraphrase
  to ``code`` and ``mbpp`` sample lists.
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


# ── HumanEval-style sample prompts ────────────────────────────────────
# Realistic HumanEval prompts as the model sees them: imports, function
# signature, docstring with prose + ``>>>`` doctests + outputs, closing
# triple-quote. Indentation matches the dataset format (4-space inside
# the docstring).

_HUMANEVAL_HAS_CLOSE = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

_HUMANEVAL_FIND_LARGEST = '''def find_largest(numbers):
    """Find the largest number in the list.

    The function should compute the maximum value and return it.
    >>> find_largest([3, 1, 4, 1, 5, 9, 2])
    9
    """
'''

_HUMANEVAL_STRIP_TARGET = '''def strip_target(s: str, target: str) -> str:
    """Given a string s, remove every occurrence of the target substring.
    >>> strip_target("abcabc", "b")
    'acac'
    """
'''


# Edge case: ``find`` appears INSIDE a doctest as ``"abc".find("b")``.
# Naive global synonym swap would corrupt this to ``"abc".determine("b")``
# and break the test. v23 must NOT touch the doctest line.
_HUMANEVAL_FIND_IN_DOCTEST = '''def index_of(haystack: str, needle: str) -> int:
    """Return the index of needle in haystack.

    Find the first match.
    >>> "abcabc".find("b")
    1
    >>> index_of("abcabc", "b")
    1
    """
'''


# ── MBPP-style sample prompts ─────────────────────────────────────────
# MBPP+ items use plain English with ``assert`` test cases inline.
# Three realistic shapes: simple function, multi-line description,
# function with helper assertion.

_MBPP_LCS = """Write a python function to find the longest common subsequence of two strings using dynamic programming.
Your code should pass these tests:
assert lcs('abc', 'abd') == 'ab'
assert lcs('xyz', 'xyz') == 'xyz'
assert lcs('abc', 'def') == ''
"""

_MBPP_IS_PRIME = """Write a python function to check if a given number is prime.
Your code should pass these tests:
assert is_prime(2) == True
assert is_prime(4) == False
assert is_prime(17) == True
"""

_MBPP_COUNT_VOWELS = """Write a function that counts vowels in a string. The function should return the total count.
Your code should pass these tests:
assert count_vowels('hello') == 2
assert count_vowels('xyz') == 0
"""


_HUMANEVAL_SAMPLES = (
    _HUMANEVAL_HAS_CLOSE,
    _HUMANEVAL_FIND_LARGEST,
    _HUMANEVAL_STRIP_TARGET,
    _HUMANEVAL_FIND_IN_DOCTEST,
)

_MBPP_SAMPLES = (
    _MBPP_LCS,
    _MBPP_IS_PRIME,
    _MBPP_COUNT_VOWELS,
)


# ── Tests: Determinism ────────────────────────────────────────────────


class TestCodeParaphraseDeterminism(unittest.TestCase):
    """Cross-validator agreement: same (prompt, block_seed) → same output."""

    def test_same_seed_same_paraphrase_humaneval(self):
        for p in _HUMANEVAL_SAMPLES:
            a = pev._paraphrase_code_problem(p, block_seed=12345)
            b = pev._paraphrase_code_problem(p, block_seed=12345)
            self.assertEqual(a, b)

    def test_same_seed_same_paraphrase_mbpp(self):
        for p in _MBPP_SAMPLES:
            a = pev._paraphrase_code_problem(p, block_seed=99)
            b = pev._paraphrase_code_problem(p, block_seed=99)
            self.assertEqual(a, b)

    def test_input_not_mutated(self):
        for p in _HUMANEVAL_SAMPLES + _MBPP_SAMPLES:
            snap = p
            pev._paraphrase_code_problem(p, block_seed=42)
            self.assertEqual(p, snap)

    def test_none_block_seed_passes_through(self):
        for p in _HUMANEVAL_SAMPLES + _MBPP_SAMPLES:
            self.assertEqual(
                pev._paraphrase_code_problem(p, block_seed=None), p,
            )

    def test_empty_string_passes_through(self):
        self.assertEqual(pev._paraphrase_code_problem("", block_seed=42), "")

    def test_whitespace_only_passes_through(self):
        self.assertEqual(
            pev._paraphrase_code_problem("   \n  \n", block_seed=42),
            "   \n  \n",
        )


# ── Tests: HumanEval-style invariants (preserve code lines) ──────────


class TestHumanEvalInvariants(unittest.TestCase):
    """Function signatures, doctests, imports, and triple-quote markers
    must survive paraphrase verbatim."""

    def test_function_signature_preserved(self):
        for p in _HUMANEVAL_SAMPLES:
            for seed in (1, 17, 199, 9999):
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                # Each signature line that starts with `def ` must appear
                # verbatim in the output.
                for line in p.split("\n"):
                    if line.lstrip().startswith("def "):
                        self.assertIn(
                            line, out,
                            f"signature mutated at seed={seed}\n"
                            f"  expected: {line!r}",
                        )

    def test_imports_preserved(self):
        for seed in (1, 17, 199):
            out = pev._paraphrase_code_problem(_HUMANEVAL_HAS_CLOSE, block_seed=seed)
            self.assertIn("from typing import List", out)

    def test_doctest_inputs_preserved(self):
        """Lines starting with ``>>>`` (doctest inputs) must survive verbatim.
        These contain canonical-form code that the model uses to infer
        I/O behavior; rewriting them would corrupt the spec."""
        for p in _HUMANEVAL_SAMPLES:
            doctests = [ln for ln in p.split("\n") if ln.lstrip().startswith(">>> ")]
            for seed in (1, 17, 199, 9999):
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                for dt in doctests:
                    self.assertIn(
                        dt, out,
                        f"doctest mutated at seed={seed}\n"
                        f"  expected: {dt!r}",
                    )

    def test_doctest_outputs_preserved(self):
        """Lines IMMEDIATELY after a ``>>>`` (doctest expected output) must
        survive verbatim. These are the gold values; rewriting them would
        give the wrong answer."""
        for p in _HUMANEVAL_SAMPLES:
            lines = p.split("\n")
            output_lines = []
            prev_was_doctest = False
            for ln in lines:
                stripped = ln.lstrip()
                if prev_was_doctest and stripped not in ("", '"""', "'''"):
                    output_lines.append(ln)
                prev_was_doctest = stripped.startswith(">>> ")
            for seed in (1, 17, 199):
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                for ol in output_lines:
                    self.assertIn(
                        ol, out,
                        f"doctest output mutated at seed={seed}\n"
                        f"  expected: {ol!r}",
                    )

    def test_find_in_doctest_not_corrupted(self):
        """Critical regression test: the word ``find`` appears INSIDE a
        doctest as ``"abcabc".find("b")``. Naive global synonym swap
        would rewrite this to ``"abcabc".determine("b")`` and break the
        test harness. Confirms that v23's line-by-line classification
        protects code identifiers."""
        for seed in range(0, 256):
            out = pev._paraphrase_code_problem(
                _HUMANEVAL_FIND_IN_DOCTEST, block_seed=seed,
            )
            # The exact doctest line must survive; the prose line ABOVE
            # may legitimately rewrite "Find" → "Determine"/"Calculate"/
            # "Compute".
            self.assertIn(
                '>>> "abcabc".find("b")', out,
                f"doctest containing 'find' was corrupted at seed={seed}\n"
                f"output: {out!r}",
            )
            # The doctest output is also preserved.
            for line in out.split("\n"):
                if line.strip() == "1":
                    break
            else:  # pragma: no cover - regression fence
                self.fail(f"doctest output '1' missing at seed={seed}")

    def test_triple_quote_markers_preserved(self):
        """Both opening and closing ``\"\"\"`` markers must survive.
        Removing them would break Python parsing of the prompt."""
        for p in _HUMANEVAL_SAMPLES:
            for seed in (1, 17, 199):
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                self.assertEqual(
                    p.count('"""'), out.count('"""'),
                    f"triple-quote count changed at seed={seed}",
                )


# ── Tests: MBPP-style invariants (preserve assert lines) ─────────────


class TestMBPPInvariants(unittest.TestCase):
    """``assert`` lines (the gold test harness) must survive verbatim
    even when they contain words from the synonym table."""

    def test_assert_lines_preserved(self):
        for p in _MBPP_SAMPLES:
            assert_lines = [
                ln for ln in p.split("\n")
                if ln.lstrip().startswith("assert ")
            ]
            self.assertGreater(
                len(assert_lines), 0,
                "test setup error: MBPP sample has no assert lines",
            )
            for seed in (1, 17, 199, 9999):
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                for al in assert_lines:
                    self.assertIn(
                        al, out,
                        f"assert line mutated at seed={seed}\n"
                        f"  expected: {al!r}",
                    )

    def test_function_names_preserved_in_asserts(self):
        """The function name in ``assert lcs(...)`` must NOT be touched
        even though there's no syntactic difference between an assert
        line containing ``find`` and a prose line containing ``find``."""
        # _MBPP_IS_PRIME has `assert is_prime(2) == True`. If `is_prime`
        # were replaced (e.g. via a hypothetical ``("is_prime", ...)``
        # entry — we don't have one, but the structural guarantee should
        # hold) the test would fail.
        for seed in range(0, 64):
            out = pev._paraphrase_code_problem(_MBPP_IS_PRIME, block_seed=seed)
            self.assertIn("assert is_prime(2) == True", out)
            self.assertIn("assert is_prime(4) == False", out)

    def test_test_harness_preamble_can_rotate(self):
        """The natural-language preamble ('Write a python function ...')
        is a prose line and SHOULD be eligible for paraphrase."""
        seen = set()
        for seed in range(0, 256):
            out = pev._paraphrase_code_problem(_MBPP_LCS, block_seed=seed)
            preamble = out.split("\n")[0]
            seen.add(preamble)
        # At least 2 variants across 256 seeds confirms the preamble
        # rotates. With "write a python function" + "find" in the
        # preamble, the candidate set is rich enough that we expect
        # multiple hits.
        self.assertGreaterEqual(
            len(seen), 2,
            f"MBPP preamble didn't rotate across 256 seeds — "
            f"paraphrase wiring may be cold (seen={seen!r})",
        )


# ── Tests: Rotation (paraphrase actually fires) ──────────────────────


class TestCodeParaphraseRotation(unittest.TestCase):
    """Across many seeds, at least some prompts get a different wording.
    This guards against a future change that accidentally makes
    paraphrase a no-op (e.g. dropping all code-table entries)."""

    def test_at_least_one_humaneval_rotates(self):
        any_rotated = False
        for p in _HUMANEVAL_SAMPLES:
            seen = set()
            for seed in range(0, 200, 7):
                seen.add(pev._paraphrase_code_problem(p, block_seed=seed))
            if len(seen) >= 2:
                any_rotated = True
                break
        self.assertTrue(
            any_rotated,
            "no HumanEval-style prompt rotated wording across 30 seeds — "
            "paraphrase wiring may be cold",
        )

    def test_at_least_one_mbpp_rotates(self):
        any_rotated = False
        for p in _MBPP_SAMPLES:
            seen = set()
            for seed in range(0, 200, 7):
                seen.add(pev._paraphrase_code_problem(p, block_seed=seed))
            if len(seen) >= 2:
                any_rotated = True
                break
        self.assertTrue(
            any_rotated,
            "no MBPP-style prompt rotated wording across 30 seeds — "
            "paraphrase wiring may be cold",
        )

    def test_humaneval_find_largest_rotates(self):
        """The ``Find the largest`` opener has both ``find`` (math table)
        and a closing ``compute`` (math table) — should rotate freely."""
        seen = set()
        for seed in range(0, 256):
            seen.add(
                pev._paraphrase_code_problem(_HUMANEVAL_FIND_LARGEST, block_seed=seed)
            )
        self.assertGreaterEqual(
            len(seen), 2,
            f"only {len(seen)} variants from imperative HumanEval prompt — "
            f"paraphrase too narrow",
        )


# ── Tests: Backward compatibility (math paraphrase still works) ──────


class TestBackwardCompatibility(unittest.TestCase):
    """The ``extra_table`` extension on ``_apply_instruction_synonyms``
    must not regress math paraphrase callers."""

    def test_apply_instruction_synonyms_no_extra_table(self):
        """Default call (no extra_table) behaves exactly as before."""
        text = "Find the value of x given the constraint."
        a = pev._apply_instruction_synonyms(text, seed=42)
        b = pev._apply_instruction_synonyms(text, seed=42, extra_table=())
        self.assertEqual(a, b)

    def test_math_paraphrase_still_works(self):
        """Existing math paraphrase callers continue to work."""
        q = "Find the area of the triangle with sides 3, 4, and 5."
        out = pev._paraphrase_math_problem(q, block_seed=42)
        # Output is non-empty, deterministic, and preserves the digits.
        self.assertTrue(out)
        self.assertEqual(
            sorted(re.findall(r"\d+", q)),
            sorted(re.findall(r"\d+", out)),
        )

    def test_aime_alias_still_works(self):
        """The round-21 ``_paraphrase_aime_problem`` alias must survive."""
        q = r"Find the smallest positive integer $k$ such that $\sqrt{k}$ is rational."
        self.assertEqual(
            pev._paraphrase_aime_problem(q, block_seed=99),
            pev._paraphrase_math_problem(q, block_seed=99),
        )


# ── Tests: Round-start wiring ────────────────────────────────────────


class TestCodeParaphraseAppliedAtRoundStart(unittest.TestCase):
    """``set_bench_block_seed`` must wire the paraphrase into the code /
    mbpp sample lists. Without this the helper is dead code on the
    critical path."""

    def test_set_bench_block_seed_wires_code_generation(self):
        """v27: code/mbpp samples must come from the procedural generator
        keyed on block_seed (not from a static pool with paraphrasing).
        This is a strictly stronger Goodhart defense than v23 paraphrase.
        """
        src = open(pev.__file__).read()
        self.assertIn(
            '_BENCH_SAMPLES["code"] = _generate_code_items(block_seed', src,
            "set_bench_block_seed does not wire _generate_code_items — "
            "v27 procedural generation will not take effect at round-start",
        )

    def test_wiring_covers_code_and_mbpp(self):
        """Both ``code`` and ``mbpp`` sample lists must be procedurally
        generated per round (v27 — replaces v23 paraphrase wiring)."""
        src = open(pev.__file__).read()
        code_pattern = re.compile(
            r'_BENCH_SAMPLES\["code"\]\s*=\s*_generate_code_items\(\s*block_seed',
            re.DOTALL,
        )
        mbpp_pattern = re.compile(
            r'_BENCH_SAMPLES\["mbpp"\]\s*=\s*_generate_code_items\(\s*\n?\s*'
            r'block_seed\s*\^\s*0x4D42',
            re.DOTALL,
        )
        self.assertTrue(
            code_pattern.search(src),
            "code_bench procedural wiring missing from set_bench_block_seed",
        )
        self.assertTrue(
            mbpp_pattern.search(src),
            "mbpp_bench procedural wiring missing from set_bench_block_seed",
        )


# ── Tests: Edge-case robustness ──────────────────────────────────────


class TestCodeParaphraseRobustness(unittest.TestCase):
    """Edge cases that historically caused issues in similar pipelines."""

    def test_handles_only_signature_no_docstring(self):
        """Some prompts may have just a signature with no docstring."""
        p = "def foo(x):\n    pass\n"
        for seed in range(0, 32):
            try:
                out = pev._paraphrase_code_problem(p, block_seed=seed)
                self.assertIn("def foo(x):", out)
            except Exception as e:  # pragma: no cover
                self.fail(f"paraphrase raised at seed={seed}: {e}")

    def test_handles_consecutive_doctests_no_blank_line(self):
        """Two ``>>>`` lines back-to-back (rare but possible) should not
        confuse the doctest-output classifier."""
        p = '''def f(x):
    """ Compute f(x).
    >>> f(1)
    2
    >>> f(2)
    4
    """
'''
        for seed in (1, 17, 199):
            out = pev._paraphrase_code_problem(p, block_seed=seed)
            # All four doctest lines (2 inputs + 2 outputs) preserved.
            self.assertIn(">>> f(1)", out)
            self.assertIn(">>> f(2)", out)
            # Outputs must survive at exactly the same indentation.
            for line in p.split("\n"):
                if line.strip() in ("2", "4"):
                    self.assertIn(line, out)

    def test_handles_multi_line_docstring_prose(self):
        """A docstring with multiple prose lines (no doctests) should
        rotate independently per line."""
        p = '''def helper(x):
    """ Compute the answer.
    The function should determine the result.
    """
'''
        out = pev._paraphrase_code_problem(p, block_seed=42)
        # No code line is mutated.
        self.assertIn("def helper(x):", out)
        # At least one prose line is rotated (or the original survives —
        # but the function should not raise).
        self.assertTrue(out)

    def test_handles_decorators(self):
        """``@decorator`` lines are code; do not paraphrase."""
        p = '''@cache
def fib(n):
    """ Find the n-th Fibonacci number. """
'''
        for seed in (1, 17, 199):
            out = pev._paraphrase_code_problem(p, block_seed=seed)
            self.assertIn("@cache", out)
            self.assertIn("def fib(n):", out)

    def test_handles_class_definition(self):
        """``class`` lines are code; do not paraphrase."""
        p = '''class Solver:
    """ Find the optimum. """
    def solve(self):
        pass
'''
        for seed in (1, 17, 199):
            out = pev._paraphrase_code_problem(p, block_seed=seed)
            self.assertIn("class Solver:", out)
            self.assertIn("    def solve(self):", out)


if __name__ == "__main__":
    unittest.main()
