#!/usr/bin/env python3
"""Regression tests for the round-19 capability_probe procedural rebalance.

Pre-v19 the capability axis (composite weight 0.25, second-highest) drew
24 of its 36 items per round from a static trivia pool baked into the
open-source ``pod_eval_vllm.py``. Round 18 logs caught a clear
memorization attack: ``ty4321/cc`` scored capability=1.000 perfect while
bombing math_bench=0.5, code_bench=0.5, aime=0.0, knowledge_bench=0.5.
v19+ flips the static/procedural ratio (12 static + 24 procedural) and
broadens the procedural generator beyond arithmetic to number theory,
string ops, list ops, and comparison. Every procedural item is freshly
sampled per round so the (operands, items) tuple cannot be memorised.

These tests pin the contract:

* ``_procedural_capability_prompts`` produces n items, each scorable.
* Items rotate per ``block_seed`` (different seed ⇒ different items).
* Items are deterministic per ``block_seed`` (same seed ⇒ same items).
* Generated answers actually match what the scorer accepts.
* The category distribution is broad (not just arithmetic).
* ``build_capability_prompts`` emits a 12 + 24 mix by default.
"""

import os
import random
import re
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    """Provide a ``torch`` stub so we can import ``pod_eval_vllm`` on the
    validator host (where torch is intentionally absent). The capability
    prompt builders never touch torch — but pod_eval_vllm imports torch
    at module top level."""
    import types

    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = lambda *a, **kw: __import__("contextlib").nullcontext()
    torch_mod.manual_seed = lambda *a, **kw: None
    torch_mod.float32 = "float32"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float16 = "float16"
    torch_mod.tensor = lambda *a, **kw: None
    torch_mod.is_tensor = lambda x: False
    torch_mod.empty_cache = lambda *a, **kw: None

    class _RandomStub:
        @staticmethod
        def get_rng_state():
            return None

        @staticmethod
        def set_rng_state(s):
            return None

    torch_mod.random = _RandomStub()
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
    transformers_mod.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": classmethod(lambda *a, **kw: None)}
    )
    transformers_mod.AutoModelForCausalLM = type(
        "AutoModelForCausalLM",
        (),
        {"from_pretrained": classmethod(lambda *a, **kw: None)},
    )
    transformers_mod.AutoConfig = type(
        "AutoConfig",
        (),
        {"from_pretrained": classmethod(lambda *a, **kw: None)},
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


class TestProceduralCapabilityPrompts(unittest.TestCase):
    """Lock the procedural-prompts contract."""

    def test_n_items_returned(self):
        """Function honours the requested count."""
        rng = random.Random(123)
        for n in (1, 5, 12, 24, 100):
            out = pev._procedural_capability_prompts(rng, n)
            self.assertEqual(len(out), n, f"expected {n} items, got {len(out)}")

    def test_zero_returns_empty(self):
        rng = random.Random(0)
        self.assertEqual(pev._procedural_capability_prompts(rng, 0), [])

    def test_each_item_is_scorable(self):
        """Every emitted item must include 'q', 'a', 'kind' and use a
        kind the scorer understands. Pre-v19 only 'int' was used; v19
        also uses 'word' (even/odd) and 'yesno' (divisible_by)."""
        rng = random.Random(456)
        out = pev._procedural_capability_prompts(rng, 100)
        valid_kinds = {"int", "word", "yesno", "phrase", "format_re",
                       "word_count", "rhyme", "mc", "word_alt"}
        for item in out:
            self.assertIn("q", item)
            self.assertIn("a", item)
            self.assertIn("kind", item)
            self.assertIn(item["kind"], valid_kinds,
                          f"unrecognised kind: {item['kind']!r}")
            self.assertTrue(item["q"].strip(), "empty question")
            self.assertTrue(str(item["a"]).strip(), "empty answer")

    def test_deterministic_per_block_seed(self):
        """Same rng seed ⇒ identical prompts (cross-validator agreement)."""
        rng_a = random.Random(7777)
        rng_b = random.Random(7777)
        out_a = pev._procedural_capability_prompts(rng_a, 24)
        out_b = pev._procedural_capability_prompts(rng_b, 24)
        self.assertEqual(
            [(it["q"], it["a"]) for it in out_a],
            [(it["q"], it["a"]) for it in out_b],
        )

    def test_rotates_across_block_seeds(self):
        """Different seeds ⇒ at least mostly-different prompts. Permits
        accidental collisions on small N but not full identity."""
        out_a = pev._procedural_capability_prompts(random.Random(111), 24)
        out_b = pev._procedural_capability_prompts(random.Random(222), 24)
        same = sum(
            1 for a, b in zip(out_a, out_b) if a["q"] == b["q"] and a["a"] == b["a"]
        )
        self.assertLess(same, len(out_a), "every item identical across seeds")

    def test_arithmetic_answers_are_correct(self):
        """For arithmetic kinds we can re-derive the answer from the
        prompt. Also confirms our generator and scorer agree."""
        rng = random.Random(999)
        out = pev._procedural_capability_prompts(rng, 200)
        arithmetic_checked = 0
        for item in out:
            q = item["q"]
            a = item["a"]
            m = re.match(
                r"^What is (-?\d+)\s+([+\-*/])\s+(-?\d+)\?",
                q,
            )
            if m:
                left, op, right = int(m.group(1)), m.group(2), int(m.group(3))
                if op == "+":
                    expected = left + right
                elif op == "-":
                    expected = left - right
                elif op == "*":
                    expected = left * right
                else:
                    expected = left // right
                self.assertEqual(int(a), expected,
                                 f"arithmetic {q!r} should be {expected}, got {a!r}")
                arithmetic_checked += 1
            mod_m = re.match(r"^What is (-?\d+) mod (-?\d+)\?", q)
            if mod_m:
                left, right = int(mod_m.group(1)), int(mod_m.group(2))
                self.assertEqual(int(a), left % right)
                arithmetic_checked += 1
        self.assertGreater(arithmetic_checked, 0,
                           "no arithmetic items generated in 200 samples")

    def test_string_op_answers_are_correct(self):
        """count_chars / count_vowels answers must match the literal word."""
        rng = random.Random(424242)
        out = pev._procedural_capability_prompts(rng, 200)
        chars_checked = 0
        vowels_checked = 0
        for item in out:
            q = item["q"]
            a = item["a"]
            m_chars = re.match(
                r"^How many characters are in the word '([a-z]+)'\?", q
            )
            if m_chars:
                w = m_chars.group(1)
                self.assertEqual(int(a), len(w),
                                 f"count_chars '{w}' should be {len(w)}, got {a!r}")
                chars_checked += 1
            m_vow = re.match(
                r"^How many vowels are in the word '([a-z]+)'\?", q
            )
            if m_vow:
                w = m_vow.group(1)
                expected = sum(1 for c in w if c in "aeiou")
                self.assertEqual(int(a), expected,
                                 f"count_vowels '{w}' should be {expected}, got {a!r}")
                vowels_checked += 1
        self.assertGreater(chars_checked + vowels_checked, 0,
                           "no string-op items generated in 200 samples")

    def test_list_op_answers_are_correct(self):
        rng = random.Random(31415)
        out = pev._procedural_capability_prompts(rng, 300)
        checked = 0
        for item in out:
            q = item["q"]
            a = item["a"]
            m_min = re.match(r"^What is the minimum of:\s*([0-9, ]+)\?", q)
            m_max = re.match(r"^What is the maximum of:\s*([0-9, ]+)\?", q)
            m_count = re.match(
                r"^How many even numbers are in this list:\s*([0-9, ]+)\?", q
            )
            if m_min:
                vals = [int(x) for x in m_min.group(1).split(",")]
                self.assertEqual(int(a), min(vals))
                checked += 1
            elif m_max:
                vals = [int(x) for x in m_max.group(1).split(",")]
                self.assertEqual(int(a), max(vals))
                checked += 1
            elif m_count:
                vals = [int(x) for x in m_count.group(1).split(",")]
                self.assertEqual(int(a), sum(1 for v in vals if v % 2 == 0))
                checked += 1
        self.assertGreater(checked, 0, "no list-op items generated in 300 samples")

    def test_categories_are_broad(self):
        """At N=200 we should see at least 5 distinct *category* templates,
        not just arithmetic. Uses a coarse per-template-prefix bucketing."""
        rng = random.Random(0xDEAD)
        out = pev._procedural_capability_prompts(rng, 200)
        prefixes = set()
        for item in out:
            q = item["q"]
            for label, pat in (
                ("add", r"^What is \d+ \+ \d+"),
                ("sub", r"^What is \d+ - \d+"),
                ("mul", r"^What is \d+ \* \d+"),
                ("div", r"^What is \d+ / \d+"),
                ("mod", r"^What is \d+ mod \d+"),
                ("power", r"^What is \d+ to the power of \d+"),
                ("sum_digits", r"^What is the sum of the digits"),
                ("even_or_odd", r"^Is \d+ even or odd"),
                ("divisible_by", r"^Is \d+ divisible by"),
                ("count_chars", r"^How many characters are in the word"),
                ("count_vowels", r"^How many vowels are in the word"),
                ("list_min", r"^What is the minimum of:"),
                ("list_max", r"^What is the maximum of:"),
                ("count_evens", r"^How many even numbers are in this list:"),
                ("which_larger", r"^Which is larger:"),
            ):
                if re.match(pat, q):
                    prefixes.add(label)
        self.assertGreaterEqual(
            len(prefixes), 5,
            f"only {len(prefixes)} categories observed in 200 items: {prefixes!r}",
        )


class TestBuildCapabilityPromptsMix(unittest.TestCase):
    """Verify the round-19 12-static + 24-procedural mix."""

    def setUp(self):
        self._orig_n = pev.CAPABILITY_PROBE_N
        self._orig_proc = pev.CAPABILITY_PROBE_N_PROC_MATH

    def tearDown(self):
        pev.CAPABILITY_PROBE_N = self._orig_n
        pev.CAPABILITY_PROBE_N_PROC_MATH = self._orig_proc

    def test_default_counts(self):
        """Default behaviour must be the round-19 12+24 split."""
        prompts = pev.build_capability_prompts(block_seed=20260426)
        self.assertEqual(
            len(prompts),
            pev.CAPABILITY_PROBE_N + pev.CAPABILITY_PROBE_N_PROC_MATH,
        )
        self.assertGreaterEqual(
            pev.CAPABILITY_PROBE_N_PROC_MATH, pev.CAPABILITY_PROBE_N,
            "round 19 contract: procedural items must be >= static items",
        )

    def test_deterministic_per_block_seed(self):
        # Static items don't always carry an ``a`` key (e.g. format_re items
        # rely on accept_re alone); compare the full dict instead.
        a = pev.build_capability_prompts(block_seed=42)
        b = pev.build_capability_prompts(block_seed=42)
        self.assertEqual(a, b)

    def test_rotates_across_block_seeds(self):
        a = pev.build_capability_prompts(block_seed=111)
        b = pev.build_capability_prompts(block_seed=222)
        same = sum(1 for x, y in zip(a, b) if x["q"] == y["q"])
        self.assertLess(same, len(a))

    def test_static_pool_share_capped(self):
        """At most CAPABILITY_PROBE_N items overlap with the static
        pool — the rest must be procedural (block-seeded)."""
        prompts = pev.build_capability_prompts(block_seed=99999)
        static_qs = {it["q"] for it in pev._CAPABILITY_STATIC_POOL}
        from_static = sum(1 for it in prompts if it["q"] in static_qs)
        self.assertLessEqual(from_static, pev.CAPABILITY_PROBE_N)


if __name__ == "__main__":
    unittest.main()
