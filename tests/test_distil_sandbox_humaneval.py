"""Regression tests for ``distil.pod.sandbox.run_humaneval``.

The ``v31_code_humaneval_plus`` axis universally scored 0.0 for every
miner (king + reference + every challenger) from the cutover until
2026-05-16 because:

    src = prompt + completion + test + "check(candidate)"

referenced an undefined name ``candidate`` (the test block defines
``def check(candidate):`` but never assigns ``candidate`` itself). Every
sandbox subprocess died with ``NameError: name 'candidate' is not
defined`` → ``ok=False`` → ``pass_frac=0.0``.

The fix invokes ``check({entry_point})`` directly using the known
function name, and adds three Goodhart-recovery passes that legacy
prod (``scripts/humaneval_sandbox.py``) already had:

* fence stripping (``Sure!\\n```python ... ```\\nHope this helps!``)
* parseable-window prose trimming
* auto-indent recovery for bare ``return ...`` bodies

Plus a per-sample nonce sentinel so ``import os; os._exit(0)`` cannot
spoof a pass.
"""

from __future__ import annotations

import unittest

from distil.pod.sandbox import run_humaneval


_PROMPT = (
    "def count_target_abc(items, target):\n"
    '    """Count target in items."""\n'
)
_TEST = (
    "def check(candidate):\n"
    "    assert candidate([1, 2, 1], 1) == 2\n"
    "    assert candidate([], 0) == 0\n"
    "    assert candidate([0, 0, 0], 0) == 3\n"
)
_ENTRY = "count_target_abc"


def _run(completion: str) -> bool:
    return run_humaneval(_PROMPT, completion, _TEST, entry_point=_ENTRY, timeout_s=5.0)


class TestSandboxBasics(unittest.TestCase):
    """Pre-fix every one of these returned False with a NameError."""

    def test_indented_body_passes(self):
        self.assertTrue(_run("    return sum(1 for x in items if x == target)\n"))

    def test_wrong_logic_still_fails(self):
        self.assertFalse(_run("    return 999\n"))

    def test_empty_completion_fails(self):
        self.assertFalse(_run(""))

    def test_missing_entry_point_returns_false(self):
        self.assertFalse(
            run_humaneval(_PROMPT, "    return 0\n", _TEST, entry_point="")
        )


class TestSandboxRecovery(unittest.TestCase):
    """Format-mismatch recoveries: a correct solution emitted in the
    wrong format must still count as a pass."""

    def test_unindented_body_recovered(self):
        self.assertTrue(_run("return sum(1 for x in items if x == target)\n"))

    def test_multiline_unindented_body(self):
        multi = (
            "result = 0\n"
            "for x in items:\n"
            "    if x == target:\n"
            "        result += 1\n"
            "return result\n"
        )
        self.assertTrue(_run(multi))

    def test_fenced_prose_stripped(self):
        gen = (
            "Sure! Here you go:\n"
            "```python\n"
            "    return sum(1 for x in items if x == target)\n"
            "```\n"
            "Hope this helps!"
        )
        self.assertTrue(_run(gen))

    def test_trailing_prose_stripped(self):
        gen = (
            "    return sum(1 for x in items if x == target)\n"
            "\nThat should do it!\n"
        )
        self.assertTrue(_run(gen))

    def test_leading_prose_stripped(self):
        gen = (
            "Here's the body:\n"
            "    return sum(1 for x in items if x == target)\n"
        )
        self.assertTrue(_run(gen))

    def test_full_def_redeclaration(self):
        gen = (
            "def count_target_abc(items, target):\n"
            "    return sum(1 for x in items if x == target)\n"
        )
        self.assertTrue(_run(gen))

    def test_recovery_does_not_promote_wrong_logic(self):
        gen = (
            "Sure! Here is the wrong answer:\n"
            "```python\n"
            "    return -1\n"
            "```\n"
        )
        self.assertFalse(_run(gen))


class TestSandboxAntiSpoof(unittest.TestCase):
    """The sentinel + ``check({entry})`` direct call must defeat the
    common spoofs from prior audits (``os._exit(0)``, monkey-patching
    ``check`` to a no-op, etc.)."""

    def test_os_exit_zero_does_not_pass(self):
        self.assertFalse(_run("    import os; os._exit(0)\n"))

    def test_sys_exit_zero_does_not_pass(self):
        self.assertFalse(_run("    import sys; sys.exit(0)\n"))

    def test_check_monkeypatch_does_not_pass(self):
        """A model that tries to override ``check`` from inside its body
        must NOT pass. The assembly order puts the gen BEFORE the test
        block, so the genuine ``def check(candidate)`` is the last one
        defined and the model's override is shadowed.

        Pre-fix this would have been impossible to defeat — without an
        entry_point-based ``check({entry}) `` call the test never ran
        at all. Locking in the post-fix behaviour as a regression."""
        gen = (
            "    pass\n"
            "\n"
            "def check(*a, **kw):\n"
            "    return None\n"
        )
        self.assertFalse(_run(gen))

    def test_assertion_failure_does_not_pass(self):
        gen = "    return target + 999\n"
        self.assertFalse(_run(gen))


class TestSandboxTimeout(unittest.TestCase):
    def test_infinite_loop_times_out_as_fail(self):
        gen = "    while True:\n        pass\n"
        self.assertFalse(_run(gen))


if __name__ == "__main__":
    unittest.main()
