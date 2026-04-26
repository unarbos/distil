#!/usr/bin/env python3
"""Regression tests for the round-24 per-round BBH inline-MC option shuffle.

Pre-v24 ``reasoning_bench`` (BBH, weight 0.08) shipped each multi-choice
subtask with a fixed correct-letter per item, encoded inline in the
question text as ``Options:\\n(A) ...\\n(B) ...``. ~12 of the 21 BBH
subtasks (logical_deduction_*, tracking_shuffled_objects_*,
disambiguation_qa, geometric_shapes, hyperbaton, movie_recommendation,
penguins_in_a_table, ruin_names, snarks, temporal_sequences) ship the
same fixed-letter format. Schema-version-0 records hit
``reasoning_bench=0.88`` paired with ``arc_bench=0`` / ``code_bench=0``
— the textbook saturated-on-memorisable-axis Goodhart signature.

Round 20's ``_shuffle_mc_options_for_round`` helper closes this attack
for ARC / MMLU-Pro / TruthfulQA, but BBH stores the options *inline*
in the question text rather than as a separate ``options`` field, so a
dedicated parser/shuffler is required. This is what
``_shuffle_bbh_mc_options`` provides in v24.

These tests pin the contract:

* MC-shape detection — only items with an inline ``Options:\\n(A) ...``
  block and a canonical ``"(X)"`` gold letter get shuffled.
* Gold remap correctness — after the shuffle, ``gold`` points at where
  the original correct answer's text actually landed.
* Determinism — same ``(item, block_seed)`` → same shuffled question.
* Cross-validator agreement — the per-item key is
  ``block_seed XOR sha256(question)``, so two validators with the same
  block_seed produce identical output.
* Per-round rotation — across many seeds the gold letter visits every
  label, breaking ``{question_text → letter}`` lookups.
* Per-question independence — two different questions in the same
  round shuffle independently.
* Boolean / numeric subtasks (boolean_expressions, object_counting,
  web_of_lies, navigate, etc.) have no inline options block — they
  must pass through unchanged so the helper degrades gracefully on
  the entire BBH pool.
* Schema preservation — the rebuilt question keeps the canonical
  ``Options:\\n(A) ...\\n(B) ...`` shape so the existing answer-
  extraction regex (``\\(?[A-Z]\\)?``) keeps working.
* ``set_bench_block_seed`` integration — reasoning_bench samples
  actually get shuffled at round-start.
* Anti-memorisation — a miner who memorised the canonical letter
  loses signal under v24.
"""

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    """Mirror the stub used by ``test_mc_option_shuffle`` so we can
    import ``pod_eval_vllm`` without a real torch / transformers /
    bittensor install on the test runner."""
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


# ─────────────────────────────────────────────────────────────────────
# Fixture builders — these mimic the exact wire-format from the
# ``lukaemon/bbh`` HF dataset (see test in __main__ for an annotated
# real sample). Each fixture mirrors the
# ``{src, question, gold}`` shape produced at pool-load time in
# ``_load_bench_pools``.
# ─────────────────────────────────────────────────────────────────────


def _logical_deduction_item(gold="(A)"):
    """Mirrors ``bbh/logical_deduction_three_objects`` real format."""
    question = (
        "The following paragraphs each describe a set of three objects "
        "arranged in a fixed order. The statements are logically "
        "consistent within each paragraph. In a golf tournament, there "
        "were three golfers: Amy, Eli, and Eve. Eve finished above Amy. "
        "Eli finished below Amy.\n"
        "Options:\n"
        "(A) Amy finished last\n"
        "(B) Eli finished last\n"
        "(C) Eve finished last"
    )
    return {"src": "bbh/logical_deduction_three_objects", "question": question, "gold": gold}


def _hyperbaton_item(gold="(B)"):
    """Mirrors ``bbh/hyperbaton``."""
    question = (
        "Which sentence has the correct adjective order:\n"
        "Options:\n"
        "(A) red small triangular paper Russian car\n"
        "(B) small triangular red Russian paper car"
    )
    return {"src": "bbh/hyperbaton", "question": question, "gold": gold}


def _movie_recommendation_item(gold="(C)"):
    """Mirrors ``bbh/movie_recommendation`` (4 options)."""
    question = (
        "Find a movie similar to The Matrix, Inception, Interstellar, "
        "and The Dark Knight:\n"
        "Options:\n"
        "(A) Mamma Mia\n"
        "(B) The Notebook\n"
        "(C) The Prestige\n"
        "(D) Cars 2"
    )
    return {"src": "bbh/movie_recommendation", "question": question, "gold": gold}


def _seven_options_item(gold="(D)"):
    """Mirrors ``bbh/logical_deduction_seven_objects`` (7 options).
    Stress-tests larger label ranges since ``range(64)`` over 7 letters
    is more sensitive to RNG bias than over 3-4 letters."""
    question = (
        "Seven runners ran a race. Determine the order:\n"
        "Options:\n"
        "(A) Alice finished first\n"
        "(B) Bob finished first\n"
        "(C) Carol finished first\n"
        "(D) Dave finished first\n"
        "(E) Eve finished first\n"
        "(F) Frank finished first\n"
        "(G) Grace finished first"
    )
    return {"src": "bbh/logical_deduction_seven_objects", "question": question, "gold": gold}


def _boolean_item(gold="True"):
    """Mirrors ``bbh/boolean_expressions`` — no inline options block,
    boolean gold. Must pass through unchanged."""
    return {
        "src": "bbh/boolean_expressions",
        "question": "not ( ( not not True ) ) is",
        "gold": gold,
    }


def _object_counting_item(gold="6"):
    """Mirrors ``bbh/object_counting`` — numeric gold, no options."""
    return {
        "src": "bbh/object_counting",
        "question": "I have a flute, a piano, a trombone, "
                    "four violins, a clarinet, and a drum. "
                    "How many musical instruments do I have?",
        "gold": gold,
    }


def _navigate_item(gold="No"):
    """Mirrors ``bbh/navigate`` — Yes/No gold, no options."""
    return {
        "src": "bbh/navigate",
        "question": "Take 5 steps. Turn right. Take 3 steps. "
                    "Are you back at the starting point?",
        "gold": gold,
    }


def _gold_text(item: dict) -> str:
    """Helper: extract the *content* (not the letter) of the gold
    option from an MC-format BBH item. The shuffled item must have
    its ``gold`` letter pointing at this same text in the new order.
    """
    import re as _re
    block = pev._BBH_OPTION_BLOCK_RE.search(item["question"])
    assert block, f"item is not MC-format: {item}"
    gold_letter = item["gold"].strip("()").upper()
    for line in block.group("options").splitlines():
        m = pev._BBH_OPTION_LINE_RE.match(line.strip())
        if m and m.group(1).upper() == gold_letter:
            return m.group(2).rstrip()
    raise AssertionError(f"gold letter {gold_letter} not in options")


def _parse_shuffled_options(item: dict) -> list[tuple[str, str]]:
    """Parse the inline options out of the (possibly shuffled)
    item.question. Returns ``[(label, text), ...]`` in display order."""
    block = pev._BBH_OPTION_BLOCK_RE.search(item["question"])
    assert block, f"shuffled item lost its options block: {item['question']!r}"
    out = []
    for line in block.group("options").splitlines():
        m = pev._BBH_OPTION_LINE_RE.match(line.strip())
        if m:
            out.append((m.group(1).upper(), m.group(2).rstrip()))
    return out


# ─────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────


class TestBBHShuffleCorrectness(unittest.TestCase):
    """The shuffled item must remain self-consistent: gold letter
    points at the original gold text in the new ordering, no options
    are dropped, no extraneous options appear."""

    def test_logical_deduction_gold_remap(self):
        original = _logical_deduction_item(gold="(A)")
        original_gold_text = _gold_text(original)
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=12345)
        new_options = _parse_shuffled_options(shuf)
        new_gold_letter = shuf["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)

    def test_hyperbaton_gold_remap_two_options(self):
        """Stress-test smallest valid MC pool (n=2)."""
        original = _hyperbaton_item(gold="(B)")
        original_gold_text = _gold_text(original)
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=98765)
        new_options = _parse_shuffled_options(shuf)
        new_gold_letter = shuf["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)

    def test_movie_four_options_gold_remap(self):
        original = _movie_recommendation_item(gold="(C)")
        original_gold_text = _gold_text(original)
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=42)
        new_options = _parse_shuffled_options(shuf)
        new_gold_letter = shuf["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)

    def test_seven_options_gold_remap(self):
        original = _seven_options_item(gold="(D)")
        original_gold_text = _gold_text(original)
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=314159)
        new_options = _parse_shuffled_options(shuf)
        new_gold_letter = shuf["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)

    def test_labels_remain_canonical_letters(self):
        """A/B/C labels stay A/B/C — only the text content rotates."""
        for fixture, _gold in (
            (_logical_deduction_item, "(A)"),
            (_hyperbaton_item, "(B)"),
            (_movie_recommendation_item, "(C)"),
        ):
            shuf = pev._shuffle_bbh_mc_options(fixture(), block_seed=1)
            options = _parse_shuffled_options(shuf)
            labels = [lbl for lbl, _ in options]
            n = len(labels)
            expected = [chr(ord("A") + i) for i in range(n)]
            self.assertEqual(labels, expected)

    def test_no_options_lost(self):
        for fixture in (_logical_deduction_item, _movie_recommendation_item, _seven_options_item):
            for seed in (1, 2, 3, 4, 5, 100, 9999, 10**9 + 7):
                original = fixture()
                original_texts = [t for _, t in _parse_shuffled_options(original)]
                shuf = pev._shuffle_bbh_mc_options(original, block_seed=seed)
                shuffled_texts = [t for _, t in _parse_shuffled_options(shuf)]
                self.assertEqual(len(shuffled_texts), len(original_texts))
                self.assertEqual(set(shuffled_texts), set(original_texts))

    def test_question_stem_preserved(self):
        """The narrative leading up to ``Options:`` is byte-identical
        to the original — only the option block is rewritten."""
        original = _logical_deduction_item(gold="(A)")
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=2024)
        original_stem = original["question"].split("Options:")[0]
        shuf_stem = shuf["question"].split("Options:")[0]
        self.assertEqual(original_stem, shuf_stem)

    def test_canonical_format_preserved(self):
        """Output must keep ``Options:\\n(A) text\\n(B) text\\n...`` so
        the answer-extraction regex (``_BBH_PAREN_RE``) keeps working
        without modification."""
        original = _movie_recommendation_item(gold="(C)")
        shuf = pev._shuffle_bbh_mc_options(original, block_seed=777)
        self.assertIn("Options:\n(A) ", shuf["question"])
        self.assertIn("\n(B) ", shuf["question"])
        self.assertIn("\n(C) ", shuf["question"])
        self.assertIn("\n(D) ", shuf["question"])


class TestBBHShuffleDeterminism(unittest.TestCase):
    """Cross-validator agreement: the per-item key
    ``block_seed XOR sha256(question)`` is stable across processes,
    machines, and Python sessions."""

    def test_same_seed_same_output(self):
        item = _logical_deduction_item()
        a = pev._shuffle_bbh_mc_options(item, block_seed=12345)
        b = pev._shuffle_bbh_mc_options(item, block_seed=12345)
        self.assertEqual(a["question"], b["question"])
        self.assertEqual(a["gold"], b["gold"])

    def test_input_not_mutated(self):
        item = _logical_deduction_item(gold="(B)")
        snap_question = item["question"]
        snap_gold = item["gold"]
        pev._shuffle_bbh_mc_options(item, block_seed=42)
        self.assertEqual(item["question"], snap_question)
        self.assertEqual(item["gold"], snap_gold)

    def test_hex_block_seed(self):
        """Hex-string block_seeds (canonical bittensor format) work."""
        item = _logical_deduction_item()
        original_gold_text = _gold_text(item)
        out = pev._shuffle_bbh_mc_options(item, block_seed="0xdeadbeef")
        new_options = _parse_shuffled_options(out)
        new_gold_letter = out["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)


class TestBBHShuffleRotation(unittest.TestCase):
    """Across many seeds the gold letter visits every label, breaking
    ``{question_text → letter}`` memorisation."""

    def test_three_options_gold_rotates(self):
        item = _logical_deduction_item(gold="(A)")
        seen = set()
        for seed in range(64):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold"])
        self.assertEqual(seen, {"(A)", "(B)", "(C)"})

    def test_two_options_gold_rotates(self):
        item = _hyperbaton_item(gold="(A)")
        seen = set()
        for seed in range(32):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold"])
        self.assertEqual(seen, {"(A)", "(B)"})

    def test_four_options_gold_rotates(self):
        item = _movie_recommendation_item(gold="(C)")
        seen = set()
        for seed in range(80):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold"])
        self.assertEqual(seen, {"(A)", "(B)", "(C)", "(D)"})

    def test_seven_options_gold_rotates(self):
        item = _seven_options_item(gold="(D)")
        seen = set()
        for seed in range(256):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold"])
        self.assertEqual(seen, {"(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"})

    def test_two_questions_shuffle_independently(self):
        """Two different MC questions in the same round get distinct
        permutations — the shuffle is per-question, not a global rotation."""
        item_a = _logical_deduction_item(gold="(A)")
        item_b = _movie_recommendation_item(gold="(C)")
        seed = 31415
        shuf_a = pev._shuffle_bbh_mc_options(item_a, block_seed=seed)
        shuf_b = pev._shuffle_bbh_mc_options(item_b, block_seed=seed)
        opts_a = _parse_shuffled_options(shuf_a)
        opts_b = _parse_shuffled_options(shuf_b)
        self.assertNotEqual([t for _, t in opts_a], [t for _, t in opts_b])

    def test_two_subtasks_same_question_unlikely(self):
        """If two pool items share question text (vanishingly rare in
        practice but worth testing), the shuffle is stable across them
        — same key → same permutation."""
        item_a = {
            "src": "bbh/foo",
            "question": _logical_deduction_item()["question"],
            "gold": "(A)",
        }
        item_b = {
            "src": "bbh/bar",
            "question": _logical_deduction_item()["question"],
            "gold": "(B)",
        }
        seed = 12345
        shuf_a = pev._shuffle_bbh_mc_options(item_a, block_seed=seed)
        shuf_b = pev._shuffle_bbh_mc_options(item_b, block_seed=seed)
        opts_a = _parse_shuffled_options(shuf_a)
        opts_b = _parse_shuffled_options(shuf_b)
        self.assertEqual([t for _, t in opts_a], [t for _, t in opts_b])


class TestBBHNonMCPassthrough(unittest.TestCase):
    """Non-MC BBH subtasks (boolean / numeric / yes-no gold) have no
    inline options block — the helper must pass them through unchanged
    so the entire BBH pool remains scorable in v24."""

    def test_boolean_expressions_passes_through(self):
        item = _boolean_item(gold="True")
        out = pev._shuffle_bbh_mc_options(item, block_seed=12345)
        self.assertEqual(out["question"], item["question"])
        self.assertEqual(out["gold"], item["gold"])

    def test_object_counting_passes_through(self):
        item = _object_counting_item(gold="6")
        out = pev._shuffle_bbh_mc_options(item, block_seed=99)
        self.assertEqual(out, item)

    def test_navigate_passes_through(self):
        item = _navigate_item(gold="No")
        out = pev._shuffle_bbh_mc_options(item, block_seed=42)
        self.assertEqual(out, item)

    def test_web_of_lies_passes_through(self):
        """Yes/No gold without an Options: block."""
        item = {
            "src": "bbh/web_of_lies",
            "question": "Alice tells the truth. Bob says Alice lies. "
                        "Carol says Bob lies. Does Carol tell the truth?",
            "gold": "Yes",
        }
        out = pev._shuffle_bbh_mc_options(item, block_seed=42)
        self.assertEqual(out, item)


class TestBBHShuffleEdgeCases(unittest.TestCase):
    def test_block_seed_none_passes_through(self):
        """Dev / replay mode: no seed → no rotation."""
        item = _logical_deduction_item()
        out = pev._shuffle_bbh_mc_options(item, block_seed=None)
        self.assertEqual(out, item)

    def test_empty_question_passes_through(self):
        item = {"src": "bbh/foo", "question": "", "gold": "(A)"}
        out = pev._shuffle_bbh_mc_options(item, block_seed=1)
        self.assertEqual(out, item)

    def test_empty_gold_passes_through(self):
        item = {
            "src": "bbh/foo",
            "question": _logical_deduction_item()["question"],
            "gold": "",
        }
        out = pev._shuffle_bbh_mc_options(item, block_seed=1)
        self.assertEqual(out, item)

    def test_non_letter_gold_passes_through(self):
        """If gold isn't ``"(X)"`` shape we don't try to remap it."""
        item = _logical_deduction_item()
        item["gold"] = "Eli finished last"
        out = pev._shuffle_bbh_mc_options(item, block_seed=1)
        self.assertEqual(out, item)

    def test_gold_letter_not_in_options_passes_through(self):
        """If gold is ``(Z)`` but options only run A-C, we punt."""
        item = _logical_deduction_item()
        item["gold"] = "(Z)"
        out = pev._shuffle_bbh_mc_options(item, block_seed=1)
        self.assertEqual(out, item)

    def test_single_option_passes_through(self):
        """Helper requires >=2 options — pathological 1-option items
        are returned unchanged rather than risking a remap."""
        item = {
            "src": "bbh/foo",
            "question": "Pick the only choice:\nOptions:\n(A) only one",
            "gold": "(A)",
        }
        out = pev._shuffle_bbh_mc_options(item, block_seed=1)
        self.assertEqual(out, item)

    def test_extra_keys_preserved(self):
        item = _logical_deduction_item()
        item["extra"] = "preserve_me"
        item["src"] = "bbh/logical_deduction_three_objects"
        out = pev._shuffle_bbh_mc_options(item, block_seed=42)
        self.assertEqual(out["src"], "bbh/logical_deduction_three_objects")
        self.assertEqual(out["extra"], "preserve_me")

    def test_options_with_special_characters(self):
        """Option text containing parentheses, asterisks, math symbols
        survives a round-trip through the parser."""
        item = {
            "src": "bbh/foo",
            "question": "Pick the right one:\nOptions:\n"
                        "(A) f(x) = x^2 + 1\n"
                        "(B) g(x) = (x-1)(x+1)\n"
                        "(C) h(x) = x*sin(x)",
            "gold": "(B)",
        }
        original_gold_text = _gold_text(item)
        out = pev._shuffle_bbh_mc_options(item, block_seed=99)
        new_options = _parse_shuffled_options(out)
        new_gold_letter = out["gold"].strip("()").upper()
        new_gold_text = next(t for lbl, t in new_options if lbl == new_gold_letter)
        self.assertEqual(new_gold_text, original_gold_text)


class TestBBHShuffleAntiMemorisation(unittest.TestCase):
    """A miner who memorised the canonical correct-letter from the
    public ``lukaemon/bbh`` dataset should lose signal under v24 —
    they should be right on roughly ``1/n`` of rounds where ``n`` is
    the option count, the floor of pure random guessing."""

    def test_three_option_memoriser_loses_signal(self):
        """A memoriser who learned ``(A)`` is correct (the canonical
        gold for ``_logical_deduction_item``) should now be correct
        on roughly 1/3 of rounds."""
        item = _logical_deduction_item(gold="(A)")
        memorised = "(A)"
        n_rounds = 300
        n_correct = 0
        for r in range(n_rounds):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=r * 17 + 3)
            if shuf["gold"] == memorised:
                n_correct += 1
        ratio = n_correct / n_rounds
        # Expected ~33%; allow generous slack for sampling noise.
        self.assertGreater(ratio, 0.20, f"memorised score {ratio:.2%} suspiciously low")
        self.assertLess(ratio, 0.50, f"memorised score {ratio:.2%} suspiciously high")

    def test_four_option_memoriser_loses_signal(self):
        """4-option BBH (movie_recommendation): expected ~25%."""
        item = _movie_recommendation_item(gold="(C)")
        memorised = "(C)"
        n_rounds = 300
        n_correct = 0
        for r in range(n_rounds):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=r * 7919 + 11)
            if shuf["gold"] == memorised:
                n_correct += 1
        ratio = n_correct / n_rounds
        self.assertGreater(ratio, 0.13, f"memorised score {ratio:.2%} suspiciously low")
        self.assertLess(ratio, 0.40, f"memorised score {ratio:.2%} suspiciously high")

    def test_oracle_solver_unaffected(self):
        """A solver who actually parses the options and identifies the
        correct content should score 100% regardless of letter rotation
        — that's the whole point. We simulate this by computing
        ``gold_text`` from the original item, then looking up which
        letter holds that text in the shuffled item, and confirming
        we always recover ``shuf['gold']``.
        """
        item = _movie_recommendation_item(gold="(C)")
        original_gold_text = _gold_text(item)
        for r in range(60):
            shuf = pev._shuffle_bbh_mc_options(item, block_seed=r * 31 + 1)
            new_options = _parse_shuffled_options(shuf)
            oracle_letter = next(
                lbl for lbl, t in new_options if t == original_gold_text
            )
            self.assertEqual(f"({oracle_letter})", shuf["gold"])


class TestBBHShuffleSetBenchBlockSeedIntegration(unittest.TestCase):
    """End-to-end check: ``set_bench_block_seed`` must apply
    ``_shuffle_bbh_mc_options`` to ``_BENCH_SAMPLES['reasoning']`` so
    when ``reasoning_bench_probe`` reads the samples, it sees the
    rotated letters. This is the one wire-up that closes the
    Goodhart vector in production."""

    def setUp(self):
        # Save and clear bench samples / pools we touch so each test
        # is hermetic.
        self._saved_pool_reasoning = list(pev._BENCH_POOLS.get("reasoning") or [])
        self._saved_samples = dict(pev._BENCH_SAMPLES)
        pev._BENCH_SAMPLES.clear()

    def tearDown(self):
        pev._BENCH_POOLS["reasoning"] = self._saved_pool_reasoning
        pev._BENCH_SAMPLES.clear()
        pev._BENCH_SAMPLES.update(self._saved_samples)

    def test_reasoning_samples_are_shuffled_at_round_start(self):
        """v27 supersedes the v20 BBH-pool-shuffle wiring with a procedural
        generator (``_generate_reasoning_items``), so the static
        ``_BENCH_POOLS['reasoning']`` is no longer consulted at round
        start. Instead, verify that:
          1. ``set_bench_block_seed`` populates ``_BENCH_SAMPLES['reasoning']``
             without reading from ``_BENCH_POOLS['reasoning']``.
          2. Each sample carries a well-formed gold (parenthesized letter or
             plain letter) and an inline ``Options:`` block, matching the
             contract that ``reasoning_bench_probe`` consumes.
          3. Different ``block_seed`` values produce different sample sets
             (rotation is real, not a no-op)."""
        pev._BENCH_POOLS["reasoning"] = []
        pev.set_bench_block_seed(4242)
        samples_a = list(pev._BENCH_SAMPLES.get("reasoning") or [])
        self.assertGreater(len(samples_a), 0, "reasoning samples missing")
        for sample in samples_a:
            self.assertIn("gold", sample, "reasoning sample missing 'gold'")
            self.assertIn("question", sample, "reasoning sample missing 'question'")
            self.assertTrue(
                str(sample.get("src", "")).startswith("procedural_reasoning"),
                f"reasoning sample src={sample.get('src')!r} not procedural — "
                "v27 wiring regressed back to static pool",
            )
        pev.set_bench_block_seed(9999)
        samples_b = list(pev._BENCH_SAMPLES.get("reasoning") or [])
        self.assertGreater(len(samples_b), 0)
        keys_a = [(s["question"], s["gold"]) for s in samples_a]
        keys_b = [(s["question"], s["gold"]) for s in samples_b]
        self.assertNotEqual(
            keys_a, keys_b,
            "different block_seeds produced identical reasoning samples — "
            "procedural rotation appears to be wired off",
        )

    def test_set_bench_block_seed_deterministic_across_runs(self):
        """Two separate calls to ``set_bench_block_seed`` with the same
        seed and the same pool produce the same samples — guarantees
        cross-validator agreement on reasoning_bench."""
        pool = [
            _logical_deduction_item(gold="(A)"),
            _movie_recommendation_item(gold="(C)"),
        ]
        pev._BENCH_POOLS["reasoning"] = pool

        pev.set_bench_block_seed(31415)
        run1 = [(s["question"], s["gold"]) for s in (pev._BENCH_SAMPLES.get("reasoning") or [])]

        pev.set_bench_block_seed(31415)
        run2 = [(s["question"], s["gold"]) for s in (pev._BENCH_SAMPLES.get("reasoning") or [])]

        self.assertEqual(run1, run2)

    def test_set_bench_block_seed_varies_with_block_seed(self):
        """Different block_seeds produce different samples (or at
        least different golds for at least one item) — proves rotation
        is wired through, not silently skipped."""
        pool = [
            _logical_deduction_item(gold="(A)"),
            _movie_recommendation_item(gold="(C)"),
            _seven_options_item(gold="(D)"),
        ]
        pev._BENCH_POOLS["reasoning"] = pool

        gold_sets = []
        for seed in (1, 7, 99, 12345, 7654321):
            pev.set_bench_block_seed(seed)
            samples = pev._BENCH_SAMPLES.get("reasoning") or []
            gold_sets.append(tuple(sorted(s["gold"] for s in samples)))

        unique_orderings = set(gold_sets)
        self.assertGreater(
            len(unique_orderings), 1,
            "all block_seeds produced identical golds — no rotation observed",
        )


class TestBBHShuffleSchemaVersionGate(unittest.TestCase):
    """``COMPOSITE_SHADOW_VERSION`` must reflect v24 so the king
    selection filter quarantines pre-v24 records."""

    def test_composite_shadow_version_at_least_24(self):
        from validator.composite import COMPOSITE_SHADOW_VERSION
        self.assertGreaterEqual(
            COMPOSITE_SHADOW_VERSION, 24,
            "COMPOSITE_SHADOW_VERSION must be bumped to >=24 so the "
            "king selection filter quarantines pre-v24 records that "
            "were graded against the old (un-rotated) BBH letters.",
        )

    def test_king_selection_min_version_at_least_24(self):
        from validator.single_eval import _KING_SELECTION_MIN_VERSION
        self.assertGreaterEqual(
            _KING_SELECTION_MIN_VERSION, 24,
            "_KING_SELECTION_MIN_VERSION must be >=24 so Tier-1 king "
            "candidates are gated on the v24 schema.",
        )


if __name__ == "__main__":
    unittest.main()
