#!/usr/bin/env python3
"""
Regression tests for the auto-indent recovery in ``humaneval_sandbox``.

The HumanEval-style "Output only the function body" prompt asks models to
emit the body of a function whose ``def`` line is already in the prompt.
Many models comply but emit the body **without leading indentation**:

  prompt:    "def incr_list(l):\n    \"\"\"...\"\"\"\n"
  bad gen:   "return [x + 1 for x in l]"

When the sandbox concatenates ``prompt + gen``, that ``return`` is at
column 0 — outside the prompt's ``def`` block — so Python raises
``SyntaxError: 'return' outside function``. Without the recovery this
counts as a code_bench failure even though the model produced a
semantically correct solution.

The recovery in ``_assemble_program`` indents every line of an
unindented gen by 4 spaces, but ONLY when the prompt itself contains
``def {entry}(`` (HumanEval pattern). Empty-prompt MBPP-style usage
that passes a complete function in the gen must NOT trigger the
recovery (covered by ``test_mbpp_sandbox_wrap.py``).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import humaneval_sandbox as hs


HE42_PROMPT = (
    "def incr_list(l):\n"
    "    \"\"\"Return list with elements incremented by 1.\n"
    "    >>> incr_list([1, 2, 3])\n"
    "    [2, 3, 4]\n"
    "    \"\"\"\n"
)
HE42_TEST = (
    "def check(candidate):\n"
    "    assert candidate([1, 2, 3]) == [2, 3, 4]\n"
    "    assert candidate([5, 2, 5, 2, 3, 3, 9, 0, 123]) == "
    "[6, 3, 6, 3, 4, 4, 10, 1, 124]\n"
)


class TestHumanEvalAutoIndent(unittest.TestCase):
    """Lock in the unindented-body recovery and its conservative scope."""

    def test_unindented_body_recovers(self):
        """Bare ``return ...`` (no leading indent) must compile and pass."""
        bad_gen = "return [x + 1 for x in l]"
        result = hs.run_sample(HE42_PROMPT, bad_gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"unindented body should recover: {result.reason}")

    def test_indented_body_unchanged(self):
        """Properly indented bodies should still pass (no double-indent)."""
        good_gen = "    return [x + 1 for x in l]"
        result = hs.run_sample(HE42_PROMPT, good_gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"indented body should pass: {result.reason}")

    def test_full_function_in_gen_unchanged(self):
        """A gen that includes ``def`` should still work (existing branch)."""
        full_gen = (
            "def incr_list(l):\n"
            "    return [x + 1 for x in l]\n"
        )
        result = hs.run_sample(HE42_PROMPT, full_gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"full-def gen should pass: {result.reason}")

    def test_empty_prompt_with_full_function(self):
        """MBPP-style: prompt='', gen='def foo(): ...'. Recovery must NOT fire."""
        prompt = ""
        gen = "def foo(x):\n    return x + 1\n"
        test = "def check(candidate):\n    assert candidate(1) == 2\n"
        result = hs.run_sample(prompt, gen, test, "foo")
        self.assertTrue(result.passed, f"empty-prompt full-fn should pass: {result.reason}")

    def test_unindented_body_wrong_logic_still_fails(self):
        """Recovery shouldn't promote a wrong solution to passing."""
        bad_logic = "return [x + 2 for x in l]"
        result = hs.run_sample(HE42_PROMPT, bad_logic, HE42_TEST, "incr_list")
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.reason)

    def test_multiline_unindented_body(self):
        """Multi-line bare body should be uniformly indented."""
        multiline = (
            "result = []\n"
            "for x in l:\n"
            "    result.append(x + 1)\n"
            "return result\n"
        )
        result = hs.run_sample(HE42_PROMPT, multiline, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"multiline body should recover: {result.reason}")


class TestHumanEvalProseStripping(unittest.TestCase):
    """Goodhart hardening (2026-04-26 round 18): HumanEval-style prompts
    must accept correctly-indented bodies wrapped in chat-style prose.
    The model can produce a working completion but prefix it with
    "Here's the body:" or append "Hope this helps!" — those must not
    flip a real-skill signal into a SyntaxError.
    """

    BODY = "    return [x + 1 for x in l]\n"

    def test_trailing_prose_stripped(self):
        gen = self.BODY + "\nThat should do it!\n"
        result = hs.run_sample(HE42_PROMPT, gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"trailing prose must recover: {result.reason}")

    def test_leading_prose_stripped(self):
        gen = "Here's the body:\n" + self.BODY
        result = hs.run_sample(HE42_PROMPT, gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"leading prose must recover: {result.reason}")

    def test_both_prose_sides_stripped(self):
        gen = "Here's the body:\n" + self.BODY + "\nHope it helps!\n"
        result = hs.run_sample(HE42_PROMPT, gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"both prose sides must recover: {result.reason}")

    def test_trailing_stray_fence_with_prose(self):
        """Round 18 fix for ``code\\n```\\nepilogue``: the closing fence
        must be peeled off and the leading code preserved (not the
        trailing prose). Older logic took the wrong side of the split."""
        gen = self.BODY + "```\n\nLet me know!\n"
        result = hs.run_sample(HE42_PROMPT, gen, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"stray closing fence must recover: {result.reason}")

    def test_prose_recovery_does_not_promote_wrong_logic(self):
        """Prose stripping must never make a wrong solution pass."""
        bad = "    return [x + 2 for x in l]\n\nHope this helps!\n"
        result = hs.run_sample(HE42_PROMPT, bad, HE42_TEST, "incr_list")
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.reason)

    def test_clean_body_unchanged(self):
        """Baseline: clean body without prose still passes."""
        result = hs.run_sample(HE42_PROMPT, self.BODY, HE42_TEST, "incr_list")
        self.assertTrue(result.passed, f"clean body must pass: {result.reason}")


if __name__ == "__main__":
    unittest.main()
