#!/usr/bin/env python3
"""
Regression tests for the MBPP-sandbox compatibility wrap.

MBPP+ tests are standalone asserts (e.g. ``assert foo(1) == 2``) but the
HumanEval sandbox expects a ``check(candidate)`` function. ``mbpp_bench_probe``
in ``scripts/pod_eval_vllm.py`` wraps the raw assert block into a ``check``
body before handing it to the sandbox. These tests lock in that wrap:

  1. A passing generation is reported as passed.
  2. A failing generation is reported as failed (not spuriously passing).
  3. A generation with a syntax error is reported as failed.
  4. The entry-point extraction logic in ``_bench_load_pools`` correctly
     ignores helper wrappers like ``math.isclose`` and ``set`` when picking
     the function name from an assertion.

These tests mirror the inline wrap + extraction logic to avoid depending on
pod-only imports (vllm, bittensor). If the source is refactored, keep this
file in sync.
"""

import re
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import humaneval_sandbox as hs


def _wrap_for_sandbox(test: str, entry_point: str) -> str:
    """Mirror of ``_wrap_for_sandbox`` inside ``mbpp_bench_probe``."""
    indented = "\n".join(
        (f"    {line}" if line.strip() else "")
        for line in test.splitlines()
    )
    return f"def check(candidate):\n{indented}\n    return True\n"


def _extract_entry_point(tests: str) -> str:
    """Mirror of the entry-point regex inside ``_bench_load_pools``."""
    _helpers = {
        "set", "round", "abs", "len", "isinstance", "sorted",
        "tuple", "list", "dict", "str", "int", "float", "frozenset",
        "all", "any",
    }
    _modules = {
        "math", "numpy", "np", "os", "sys", "collections",
        "itertools", "functools",
    }
    for cand_m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(", tests):
        cand = cand_m.group(1)
        base = cand.split(".")[-1] if "." in cand else cand
        top = cand.split(".")[0]
        if top in _modules:
            continue
        if base in _helpers:
            continue
        return base
    return ""


class TestMbppWrap(unittest.TestCase):
    """End-to-end contract tests for the MBPP-sandbox wrap."""

    def test_passing_solution(self):
        tests = (
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)"
        )
        completion = (
            "def similar_elements(a, b):\n"
            "    return tuple(sorted(set(a) & set(b)))\n"
        )
        wrapped = _wrap_for_sandbox(tests, "similar_elements")
        result = hs.run_sample("", completion, wrapped, "similar_elements")
        self.assertTrue(result.passed, f"Expected pass, got {result.reason}")

    def test_failing_solution(self):
        tests = "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)"
        bad = "def similar_elements(a, b):\n    return tuple()\n"
        wrapped = _wrap_for_sandbox(tests, "similar_elements")
        result = hs.run_sample("", bad, wrapped, "similar_elements")
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.reason)

    def test_syntax_error_solution(self):
        tests = "assert foo(1) == 1"
        broken = "def foo(x):\n    return [[\n"
        wrapped = _wrap_for_sandbox(tests, "foo")
        result = hs.run_sample("", broken, wrapped, "foo")
        self.assertFalse(result.passed)
        self.assertIn("SyntaxError", result.reason)


class TestMbppEntryPointExtraction(unittest.TestCase):
    """Ensure the entry-point regex ignores helpers like ``math.isclose``."""

    def test_direct_call(self):
        tests = "assert similar_elements((1,), (1,)) == (1,)"
        self.assertEqual(_extract_entry_point(tests), "similar_elements")

    def test_math_isclose_wrapped(self):
        tests = "assert math.isclose(volume(1, 2, 3), 6.0, rel_tol=1e-6)"
        self.assertEqual(_extract_entry_point(tests), "volume")

    def test_set_wrapped(self):
        tests = "assert set(uniq([1, 2, 2, 3])) == {1, 2, 3}"
        self.assertEqual(_extract_entry_point(tests), "uniq")

    def test_multiple_asserts_first_wins(self):
        tests = (
            "assert foo(1) == 1\n"
            "assert foo(2) == 2\n"
        )
        self.assertEqual(_extract_entry_point(tests), "foo")

    def test_numpy_wrapper_skipped(self):
        tests = "assert numpy.allclose(matmul([[1]],[[1]]), [[1]])"
        self.assertEqual(_extract_entry_point(tests), "matmul")


class TestMbppProseStripping(unittest.TestCase):
    """Goodhart hardening (2026-04-26 round 18): MBPP must accept correct
    code wrapped in chat-style prose. A model that produces a working
    function but prefixes it with "Sure! Here is..." or appends "Hope
    this helps!" should be graded on its code, not on prompt-format
    compliance (which we measure separately in ``ifeval_bench``).
    """

    TESTS = (
        "assert is_sorted([1, 2, 3, 4]) == True\n"
        "assert is_sorted([4, 3, 2, 1]) == False\n"
    )
    GOOD_FN = (
        "def is_sorted(lst):\n"
        "    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n"
    )

    def _run(self, gen):
        wrapped = _wrap_for_sandbox(self.TESTS, "is_sorted")
        return hs.run_sample("", gen, wrapped, "is_sorted")

    def test_clean_function_passes(self):
        """Baseline: clean def with no prose passes (pre-fix behavior)."""
        result = self._run(self.GOOD_FN)
        self.assertTrue(result.passed, f"clean def must pass: {result.reason}")

    def test_leading_prose_recovers(self):
        """``Sure! Here is the function:\\n\\ndef ...`` must pass."""
        chatty = (
            "Sure! Here is a Python function that solves this:\n"
            "\n"
            + self.GOOD_FN
        )
        result = self._run(chatty)
        self.assertTrue(result.passed, f"chatty leading prose must recover: {result.reason}")

    def test_trailing_prose_recovers(self):
        """``def ...\\n\\nHope this helps!`` must pass."""
        chatty = (
            self.GOOD_FN
            + "\n"
            + "This function checks if a list is sorted ascendingly. Hope this helps!\n"
        )
        result = self._run(chatty)
        self.assertTrue(result.passed, f"trailing prose must recover: {result.reason}")

    def test_both_prose_sides_recovers(self):
        """Both leading and trailing prose must be stripped."""
        chatty = (
            "Here is the function:\n"
            "\n"
            + self.GOOD_FN
            + "\n"
            + "That should work for your use case.\n"
        )
        result = self._run(chatty)
        self.assertTrue(result.passed, f"both-side prose must recover: {result.reason}")

    def test_fenced_block_still_works(self):
        """Existing fence-stripping must continue to work after prose rule."""
        fenced = (
            "```python\n"
            + self.GOOD_FN
            + "```\n"
        )
        result = self._run(fenced)
        self.assertTrue(result.passed, f"fenced block must pass: {result.reason}")

    def test_fenced_block_with_prose(self):
        """Combined: prose + fenced block should pass."""
        chatty_fenced = (
            "Sure, here's my solution:\n"
            "\n"
            "```python\n"
            + self.GOOD_FN
            + "```\n"
            "\n"
            "Let me know if you need adjustments!\n"
        )
        result = self._run(chatty_fenced)
        self.assertTrue(result.passed, f"fenced+prose must recover: {result.reason}")

    def test_prose_recovery_does_not_promote_wrong_code(self):
        """Prose stripping must NEVER make a wrong solution pass."""
        bad_fn = (
            "Sure! Here is the function:\n"
            "\n"
            "def is_sorted(lst):\n"
            "    return False\n"  # wrong — always returns False
            "\n"
            "Hope it helps!\n"
        )
        result = self._run(bad_fn)
        self.assertFalse(result.passed, "wrong logic must still fail")
        self.assertIn("AssertionError", result.reason)

    def test_module_constant_preserved(self):
        """A module-level constant before the def must be preserved."""
        with_const = (
            "Here's my solution using a lookup table:\n"
            "\n"
            "_TABLE = {True: True, False: False}\n"
            "\n"
            "def is_sorted(lst):\n"
            "    return _TABLE[all(lst[i] <= lst[i+1] for i in range(len(lst)-1))]\n"
            "\n"
            "Done!\n"
        )
        result = self._run(with_const)
        self.assertTrue(result.passed, f"module constant must be preserved: {result.reason}")

    def test_helper_function_preserved(self):
        """A helper def before the entry point must be preserved."""
        with_helper = (
            "I'll use a helper:\n"
            "\n"
            "def _pair_ok(a, b):\n"
            "    return a <= b\n"
            "\n"
            "def is_sorted(lst):\n"
            "    return all(_pair_ok(lst[i], lst[i+1]) for i in range(len(lst)-1))\n"
        )
        result = self._run(with_helper)
        self.assertTrue(result.passed, f"helper before entry must be preserved: {result.reason}")

    def test_imports_preserved(self):
        """A leading ``from X import Y`` must be preserved."""
        with_import = (
            "Here you go:\n"
            "\n"
            "from itertools import pairwise\n"
            "\n"
            "def is_sorted(lst):\n"
            "    return all(a <= b for a, b in pairwise(lst))\n"
        )
        result = self._run(with_import)
        self.assertTrue(result.passed, f"leading import must be preserved: {result.reason}")

    def test_unrecoverable_gen_falls_through(self):
        """Truly unparseable gen must remain unparseable (we never invent code)."""
        garbage = "this is just english prose with no code at all"
        result = self._run(garbage)
        self.assertFalse(result.passed)


class TestExtractPythonBlockUnit(unittest.TestCase):
    """Unit tests for ``_extract_python_block`` itself (no sandbox)."""

    def test_already_parseable_returned_unchanged(self):
        gen = "def foo():\n    return 1\n"
        out = hs._extract_python_block(gen, must_contain="def foo(")
        self.assertEqual(out, gen)

    def test_strips_leading_prose(self):
        gen = "Sure!\n\ndef foo():\n    return 1\n"
        out = hs._extract_python_block(gen, must_contain="def foo(")
        self.assertNotIn("Sure!", out)
        self.assertIn("def foo()", out)

    def test_strips_trailing_prose(self):
        gen = "def foo():\n    return 1\n\nDone!\n"
        out = hs._extract_python_block(gen, must_contain="def foo(")
        self.assertNotIn("Done!", out)
        self.assertIn("def foo()", out)

    def test_keeps_leading_constants(self):
        gen = "Hi there!\n\n_X = 5\n\ndef foo():\n    return _X\n\nDone! 🎉"
        out = hs._extract_python_block(gen, must_contain="def foo(")
        self.assertIn("_X = 5", out)
        self.assertIn("def foo()", out)
        self.assertNotIn("Hi there!", out)
        self.assertNotIn("Done!", out)

    def test_no_must_contain_match_returns_original(self):
        """If we never find a parseable region containing ``must_contain``,
        the original is returned (we never guess)."""
        gen = "Sure! Here is the function:\n\ndef bar():\n    return 1\n"
        out = hs._extract_python_block(gen, must_contain="def foo(")
        self.assertEqual(out, gen)

    def test_empty_input(self):
        self.assertEqual(hs._extract_python_block("", must_contain="def foo("), "")
        self.assertEqual(hs._extract_python_block(None, must_contain="def foo("), None)

    def test_no_must_contain_arg_works(self):
        """Without ``must_contain``, returns first parseable contiguous
        region (still strips leading prose)."""
        gen = "english.\n\ndef foo():\n    return 1\n"
        out = hs._extract_python_block(gen)
        self.assertIn("def foo()", out)
        self.assertNotIn("english.", out)


if __name__ == "__main__":
    unittest.main()
