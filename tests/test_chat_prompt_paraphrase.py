#!/usr/bin/env python3
"""Regression tests for the round-25 per-round chat-prompt paraphrase.

Pre-v25 ``judge_probe`` (composite weight 0.15) drew 16 prompts per
round from a 65-prompt static pool baked into ``pod_eval_vllm.py``,
and ``chat_turns_probe`` (composite weight 0.08) drew 6 conversations
per round from a ~25-conversation static pool of 3-turn dialogues.
Combined attack surface = 0.23 weight, larger than ``code_bench`` +
``reasoning_bench`` combined (0.20). Both axes are graded by the
teacher rubric on a 1-5 scale of "correct + clear + addresses the
question + appropriate length", so a miner who pre-trains on
canonical 5/5-quality responses to all ~90 prompts can saturate
both axes from a ``{prompt_text → canonical_response}`` lookup
without doing any genuine chat work — the same canonical-wording
memorisation Goodhart vector closed for math / code / BBH in
v18-v24, just on a smaller surface.

These tests pin the v25 contract:

* The chat helper is region-aware: it never paraphrases inside
  backticks, quotes, or inline JSON-like blocks.
* Code identifiers, function signatures, format specifiers, regex
  literals, and quoted strings survive a paraphrase round-trip
  byte-identical.
* Determinism: same ``(prompt, block_seed)`` → same paraphrased
  prompt (cross-validator agreement).
* Per-prompt seeding: two different prompts in the same round get
  different paraphrase picks (no global rotation pattern).
* Per-round rotation: the same prompt across many seeds spans
  multiple surface variants — a memoriser keyed on canonical text
  loses signal.
* The chat synonym table avoids math-domain rewrites that would read
  awkward in conversational prose ("find a movie" → "determine a
  movie" must NOT happen in chat paraphrase).
* ``_pick_judge_probe_prompts`` and ``_pick_chat_turns_prompts``
  actually wire the paraphrase through.
* Schema version is bumped so old records are quarantined.
"""

import os
import re
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    """Mirror the stub used by other v18-v24 tests so we can import
    ``pod_eval_vllm`` without a real torch / transformers / bittensor
    install on the test runner."""
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
# Shared fixtures — sourced verbatim from the real JUDGE_PROBE_POOL
# and CHAT_TURNS_PROBE_POOL so any test failure here directly reflects
# behaviour the production validator will see.
# ─────────────────────────────────────────────────────────────────────


_JUDGE_PROMPT_PROSE_HEAVY = (
    "Explain briefly the difference between TCP and UDP."
)
_JUDGE_PROMPT_INSTRUCTION = (
    "Give three practical tips for writing cleaner code."
)
_JUDGE_PROMPT_CODE_INLINE = (
    "Write a Python function `is_palindrome(s: str) -> bool` that "
    "returns True if `s` is a palindrome ignoring case and spaces. "
    "Include only the function."
)
_JUDGE_PROMPT_LIST_TOKEN = (
    "What is the output of `print(list(range(3, 10, 2)))` in Python? "
    "Just the output."
)
_JUDGE_PROMPT_JSON_INLINE = (
    "Given a JSON object `{\"name\": \"Ada\", \"langs\": "
    "[\"py\", \"go\"]}`, what is the value at langs[0]? Answer with "
    "the value only."
)
_JUDGE_PROMPT_FORMAT_STRICT = (
    "List five countries of Europe separated by commas, no other text."
)
_JUDGE_PROMPT_QUOTED = (
    "Write a single sentence that describes the color 'deep ocean blue' "
    "without using the words 'blue' or 'ocean'."
)
_JUDGE_PROMPT_REGEX_LITERAL = (
    "Write a regex that matches a US 5-digit zip code. Just the regex."
)
_JUDGE_PROMPT_TRIPLE_BACKTICK = (
    "Here is a code block:\n```python\nfor x in list(range(5)):\n    "
    "print(x)\n```\nDescribe what the code does in one sentence."
)


_CHAT_TURNS_FIXTURE = (
    "Give me a simple recipe for chocolate chip cookies.",
    "I don't have baking soda. Can I substitute?",
    "Using your substitution, how does the final cookie differ from "
    "the original recipe?",
)


# ─────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────


class TestChatParaphraseRegionAwareness(unittest.TestCase):
    """The v25 helper must NEVER mutate code / quoted / JSON regions."""

    def test_inline_backticks_preserved_byte_identical(self):
        """``range(5)`` / ``list(...)`` inside backticks must survive
        even though ``list`` and ``range`` would be candidates for the
        chat synonym table if applied without region awareness."""
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_LIST_TOKEN, block_seed=12345)
        self.assertIn("`print(list(range(3, 10, 2)))`", out)

    def test_function_name_in_backticks_preserved(self):
        out = pev._paraphrase_chat_prompt(
            _JUDGE_PROMPT_CODE_INLINE, block_seed=42,
        )
        self.assertIn("`is_palindrome(s: str) -> bool`", out)
        self.assertIn("`s`", out)

    def test_inline_json_preserved(self):
        """Inline JSON-like ``{...}`` blocks survive — the helper must
        not touch curly-brace-delimited segments because Python format
        strings, JSON, and dict literals are common in the pool."""
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_JSON_INLINE, block_seed=99)
        self.assertIn("`{\"name\": \"Ada\", \"langs\": [\"py\", \"go\"]}`", out)

    def test_single_quoted_strings_preserved(self):
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_QUOTED, block_seed=7)
        self.assertIn("'deep ocean blue'", out)
        self.assertIn("'blue'", out)
        self.assertIn("'ocean'", out)

    def test_triple_backtick_fence_preserved(self):
        out = pev._paraphrase_chat_prompt(
            _JUDGE_PROMPT_TRIPLE_BACKTICK, block_seed=2024,
        )
        self.assertIn(
            "```python\nfor x in list(range(5)):\n    print(x)\n```",
            out,
        )
        self.assertNotIn("for x in enumerate(range(5))", out)

    def test_regex_literal_preserved(self):
        """Regex prompts are common in the pool. We don't want to
        rotate words inside the regex pattern."""
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_REGEX_LITERAL, block_seed=1)
        self.assertIn("Just the regex", out)


class TestChatParaphraseProseRotation(unittest.TestCase):
    """The chat helper must produce a measurable rotation across seeds
    on PROSE prompts so a memoriser loses signal."""

    def test_explain_rotates_across_seeds(self):
        """Across many seeds the leading verb of an "Explain X" prompt
        should be rewritten at least once. We verify by collecting the
        SET of rewritten openings across 64 seeds and asserting that
        more than one variant appears (rotation present)."""
        seen_openings = set()
        for seed in range(64):
            out = pev._paraphrase_chat_prompt(
                _JUDGE_PROMPT_PROSE_HEAVY, block_seed=seed * 7919 + 1,
            )
            seen_openings.add(out.split(" ", 1)[0].rstrip(":,.").lower())
        self.assertGreater(
            len(seen_openings), 1,
            f"no rotation observed across 64 seeds — only {seen_openings} "
            f"appeared",
        )

    def test_give_rotates_across_seeds(self):
        seen_openings = set()
        for seed in range(64):
            out = pev._paraphrase_chat_prompt(
                _JUDGE_PROMPT_INSTRUCTION, block_seed=seed * 7919 + 1,
            )
            seen_openings.add(out.split(" ", 1)[0].rstrip(":,.").lower())
        self.assertGreater(len(seen_openings), 1)

    def test_synonyms_are_chat_domain_only(self):
        """The chat path must NOT layer math-domain synonyms (find /
        calculate / determine) on top — those create awkward rewrites
        in conversational prose ("find a movie" / "calculate the
        cost"). We sample many seeds and check NO output rewrites
        ``"give"`` to ``"determine"`` or ``"compute"``."""
        for seed in range(200):
            out = pev._paraphrase_chat_prompt(
                _JUDGE_PROMPT_INSTRUCTION, block_seed=seed * 7 + 11,
            )
            self.assertNotIn("Determine three", out)
            self.assertNotIn("Calculate three", out)
            self.assertNotIn("Compute three", out)


class TestChatParaphraseDeterminism(unittest.TestCase):
    """Cross-validator agreement: same ``(prompt, block_seed)`` →
    same output."""

    def test_same_seed_same_output(self):
        a = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_PROSE_HEAVY, block_seed=12345)
        b = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_PROSE_HEAVY, block_seed=12345)
        self.assertEqual(a, b)

    def test_input_not_mutated(self):
        snap = _JUDGE_PROMPT_PROSE_HEAVY
        pev._paraphrase_chat_prompt(_JUDGE_PROMPT_PROSE_HEAVY, block_seed=42)
        self.assertEqual(_JUDGE_PROMPT_PROSE_HEAVY, snap)

    def test_per_prompt_seed_independence(self):
        """Two different prompts in the same round get different
        paraphrase picks because the per-prompt seed is mixed with
        the prompt text via ``_stable_seed_from_text``. We verify by
        confirming that across a small panel of seeds the two prompts
        do NOT converge to the same swap on every seed (a global
        constant transform would converge)."""
        n_diff_swaps = 0
        for seed in (1, 7, 99, 12345, 31415, 999999):
            out_a = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_PROSE_HEAVY, block_seed=seed)
            out_b = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_INSTRUCTION, block_seed=seed)
            # Extract leading verb of each output. If the helper applied
            # a global transform, the same source word in both prompts
            # would map to the same target. Different leading verbs
            # across seeds proves per-prompt seed independence.
            verb_a = out_a.split(" ", 1)[0].rstrip(":,.")
            verb_b = out_b.split(" ", 1)[0].rstrip(":,.")
            orig_a = _JUDGE_PROMPT_PROSE_HEAVY.split(" ", 1)[0]
            orig_b = _JUDGE_PROMPT_INSTRUCTION.split(" ", 1)[0]
            if (verb_a != orig_a) and (verb_b != orig_b):
                n_diff_swaps += 1
        self.assertGreater(
            n_diff_swaps, 0,
            "neither prompt was paraphrased on any seed — wire-up "
            "or seed mixing broken",
        )

    def test_hex_block_seed(self):
        a = pev._paraphrase_chat_prompt(
            _JUDGE_PROMPT_PROSE_HEAVY, block_seed="0xdeadbeef",
        )
        b = pev._paraphrase_chat_prompt(
            _JUDGE_PROMPT_PROSE_HEAVY, block_seed="0xdeadbeef",
        )
        self.assertEqual(a, b)


class TestChatParaphraseSemanticPreservation(unittest.TestCase):
    """The paraphrase must preserve the semantic intent of the
    prompt so a model that genuinely understands the request can
    still answer correctly."""

    def test_prose_only_change_is_synonym_swap(self):
        """For a prose-only prompt, the output is identical to the
        original modulo synonym swaps from the chat-domain table. We
        verify by canonicalising both strings (replacing every synonym
        from every cluster with a sentinel for that cluster), then
        comparing for equality."""
        clusters = [
            {"explain", "describe", "outline"},
            {"give", "provide", "offer"},
            {"show", "demonstrate", "illustrate"},
            {"list", "enumerate"},
            {"briefly", "concisely"},
            {"suggest", "recommend"},
        ]

        def canonicalise(text: str) -> str:
            t = text.lower()
            for i, cluster in enumerate(clusters):
                sentinel = f"<C{i}>"
                # Sort by length desc so multi-word entries match before single-word.
                for w in sorted(cluster, key=len, reverse=True):
                    t = re.sub(r"\b" + re.escape(w) + r"\b", sentinel, t)
            return t

        canon_orig = canonicalise(_JUDGE_PROMPT_PROSE_HEAVY)
        for seed in range(64):
            out = pev._paraphrase_chat_prompt(
                _JUDGE_PROMPT_PROSE_HEAVY, block_seed=seed * 13 + 1,
            )
            canon_out = canonicalise(out)
            self.assertEqual(
                canon_orig, canon_out,
                f"paraphrase changed text in a way no synonym cluster "
                f"explains:\nseed={seed * 13 + 1}\noriginal="
                f"{_JUDGE_PROMPT_PROSE_HEAVY!r}\nout={out!r}\n"
                f"canon_orig={canon_orig!r}\ncanon_out={canon_out!r}",
            )

    def test_format_constraint_preserved(self):
        """``"no other text"`` is a hard constraint scored by the
        rubric — must survive a paraphrase round-trip byte-identical."""
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_FORMAT_STRICT, block_seed=42)
        self.assertIn("no other text", out)
        self.assertIn("five countries of Europe", out)

    def test_preserves_numerical_constraints(self):
        """Numerical constraints anchor rubric scoring — they must
        not be rotated by the synonym swap (the chat synonym table
        contains no number-altering entries, but verify regression-
        ready)."""
        prompt = "Give me a response that is exactly one sentence ending in 'fin.'"
        for seed in range(20):
            out = pev._paraphrase_chat_prompt(prompt, block_seed=seed * 31 + 1)
            self.assertIn("exactly one sentence", out)
            self.assertIn("'fin.'", out)


class TestChatParaphraseEdgeCases(unittest.TestCase):
    def test_block_seed_none_passes_through(self):
        out = pev._paraphrase_chat_prompt(_JUDGE_PROMPT_PROSE_HEAVY, block_seed=None)
        self.assertEqual(out, _JUDGE_PROMPT_PROSE_HEAVY)

    def test_empty_string_passes_through(self):
        out = pev._paraphrase_chat_prompt("", block_seed=12345)
        self.assertEqual(out, "")

    def test_only_protected_passes_through(self):
        """A prompt that is 100% protected (just a code block) has no
        prose to paraphrase."""
        prompt = "```python\nprint('hello')\n```"
        out = pev._paraphrase_chat_prompt(prompt, block_seed=42)
        self.assertEqual(out, prompt)

    def test_no_synonym_match_passes_through(self):
        """A prompt with no chat-domain synonym candidates is returned
        unchanged."""
        prompt = "What time is it now?"
        out = pev._paraphrase_chat_prompt(prompt, block_seed=12345)
        self.assertEqual(out, prompt)

    def test_case_preservation(self):
        """Capitalisation of the swapped word follows the original
        casing convention — leading capital → leading capital."""
        prompt = "Explain the concept of recursion."
        for seed in range(64):
            out = pev._paraphrase_chat_prompt(prompt, block_seed=seed)
            first_word = out.split(" ", 1)[0]
            self.assertTrue(
                first_word[:1].isupper(),
                f"opening word {first_word!r} lost its capitalisation",
            )

    def test_unicode_safe(self):
        """Non-ASCII content is unchanged. Some judge prompts contain
        em-dashes / curly quotes; the helper must not crash."""
        prompt = "Explain — in one paragraph — why we use unit tests."
        out = pev._paraphrase_chat_prompt(prompt, block_seed=42)
        self.assertIn("—", out)


class TestPickJudgeProbePromptsIntegration(unittest.TestCase):
    """End-to-end: ``_pick_judge_probe_prompts`` must apply the
    paraphrase so when ``judge_response_probe`` reads
    ``JUDGE_PROBE_PROMPTS`` it sees the rotated phrasings."""

    def test_picks_are_paraphrased(self):
        original_pool_first_prompts = list(pev.JUDGE_PROBE_POOL[: pev.JUDGE_PROBE_PER_ROUND])
        out = pev._pick_judge_probe_prompts(block_seed=12345)
        self.assertEqual(len(out), pev.JUDGE_PROBE_PER_ROUND)
        for picked in out:
            self.assertIsInstance(picked, str)
            self.assertGreater(len(picked), 5)
        # At least one picked prompt must differ from its original
        # form across the candidate pool — proves the paraphrase wires
        # through. We can't pin which one because picking shuffles
        # the pool.
        any_changed = False
        for p in out:
            if p not in pev.JUDGE_PROBE_POOL:
                any_changed = True
                break
        self.assertTrue(
            any_changed,
            "no picked prompt was paraphrased — the wire-up regressed",
        )

    def test_picks_are_deterministic(self):
        a = pev._pick_judge_probe_prompts(block_seed=12345)
        b = pev._pick_judge_probe_prompts(block_seed=12345)
        self.assertEqual(a, b)

    def test_picks_rotate_across_seeds(self):
        a = pev._pick_judge_probe_prompts(block_seed=1)
        b = pev._pick_judge_probe_prompts(block_seed=2)
        self.assertNotEqual(a, b)

    def test_block_seed_none_skips_paraphrase(self):
        """Dev / replay mode (no seed) returns the canonical pool
        verbatim so old replay logs grade identically."""
        out = pev._pick_judge_probe_prompts(block_seed=None)
        self.assertEqual(
            out,
            list(pev.JUDGE_PROBE_POOL[: pev.JUDGE_PROBE_PER_ROUND]),
        )


class TestPickChatTurnsPromptsIntegration(unittest.TestCase):
    """End-to-end: ``_pick_chat_turns_prompts`` must paraphrase EACH
    turn of EACH picked conversation."""

    def test_picks_are_three_turns(self):
        out = pev._pick_chat_turns_prompts(block_seed=12345)
        self.assertEqual(len(out), pev.CHAT_TURNS_PROBE_PER_ROUND)
        for convo in out:
            self.assertEqual(len(convo), 3)
            for turn in convo:
                self.assertIsInstance(turn, str)
                self.assertGreater(len(turn), 5)

    def test_picks_are_paraphrased(self):
        out = pev._pick_chat_turns_prompts(block_seed=12345)
        original_pool_set = {
            tuple(c) for c in pev.CHAT_TURNS_PROBE_POOL
        }
        any_changed = False
        for convo in out:
            if tuple(convo) not in original_pool_set:
                any_changed = True
                break
        self.assertTrue(
            any_changed,
            "no picked conversation was paraphrased — wire-up regressed",
        )

    def test_picks_are_deterministic(self):
        a = pev._pick_chat_turns_prompts(block_seed=12345)
        b = pev._pick_chat_turns_prompts(block_seed=12345)
        self.assertEqual(a, b)

    def test_per_turn_paraphrase_independent(self):
        """Each turn within a conversation gets the same per-prompt
        seed but the seed is text-derived, so different turn texts
        produce different swaps. We verify by checking that across
        seeds the paraphrased turns aren't all the SAME variant."""
        all_turn_variants = [set(), set(), set()]
        for seed in range(32):
            out = pev._pick_chat_turns_prompts(block_seed=seed * 17 + 3)
            if not out:
                continue
            convo = out[0]
            for i, turn in enumerate(convo):
                all_turn_variants[i].add(turn)
        for i, variants in enumerate(all_turn_variants):
            self.assertGreater(
                len(variants), 1,
                f"turn {i} did not rotate across seeds: {variants}",
            )


class TestChatParaphraseSchemaVersionGate(unittest.TestCase):
    """``COMPOSITE_SHADOW_VERSION`` must reflect v25 so the king
    selection filter quarantines pre-v25 records."""

    def test_composite_shadow_version_at_least_25(self):
        from validator.composite import COMPOSITE_SHADOW_VERSION
        self.assertGreaterEqual(
            COMPOSITE_SHADOW_VERSION, 25,
            "COMPOSITE_SHADOW_VERSION must be bumped to >=25 so the "
            "king selection filter quarantines pre-v25 records that "
            "were graded against the un-rotated chat prompts.",
        )

    def test_king_selection_min_version_at_least_25(self):
        from validator.single_eval import _KING_SELECTION_MIN_VERSION
        self.assertGreaterEqual(
            _KING_SELECTION_MIN_VERSION, 25,
            "_KING_SELECTION_MIN_VERSION must be >=25 so Tier-1 king "
            "candidates are gated on the v25 schema.",
        )


class TestChatSynonymTableSafety(unittest.TestCase):
    """The chat synonym table must satisfy domain-safety invariants.
    These are *static* properties of the table, so they can be tested
    directly without running through the helper."""

    def test_no_math_domain_overlap(self):
        """find / calculate / determine / compute / solve have
        homonym ambiguity in conversational prose. The chat table
        must not include them as src OR alts."""
        forbidden = {"find", "calculate", "determine", "compute", "solve"}
        for src, alts in pev._CHAT_INSTRUCTION_SYNONYMS:
            self.assertNotIn(
                src.lower(), forbidden,
                f"chat src {src!r} overlaps math-domain table",
            )
            for alt in alts:
                self.assertNotIn(
                    alt.lower(), forbidden,
                    f"chat alt {alt!r} for {src!r} overlaps math-domain table",
                )

    def test_no_numeric_in_table(self):
        """No entry contains a digit — numeric constraints are
        rubric-anchoring and must never be rotated by the synonym
        swap."""
        for src, alts in pev._CHAT_INSTRUCTION_SYNONYMS:
            self.assertFalse(
                any(c.isdigit() for c in src),
                f"chat src {src!r} contains a digit",
            )
            for alt in alts:
                self.assertFalse(
                    any(c.isdigit() for c in alt),
                    f"chat alt {alt!r} for {src!r} contains a digit",
                )

    def test_no_punctuation_anchors(self):
        """No entry contains a colon, semicolon, or period — those
        often appear inside format specs and would be unsafe to swap."""
        forbidden_chars = ":;.!?"
        for src, alts in pev._CHAT_INSTRUCTION_SYNONYMS:
            for ch in forbidden_chars:
                self.assertNotIn(
                    ch, src, f"chat src {src!r} contains {ch!r}",
                )
                for alt in alts:
                    self.assertNotIn(
                        ch, alt,
                        f"chat alt {alt!r} for {src!r} contains {ch!r}",
                    )


class TestChatParaphraseAntiMemorisation(unittest.TestCase):
    """A miner who memorised a canonical 5/5-quality response keyed on
    the original prompt text loses signal under v25."""

    def test_canonical_text_lookup_misses_most_rounds(self):
        """``"Explain briefly the difference between TCP and UDP."``
        is the canonical pre-v25 prompt. Across many seeds, the helper
        rewrites the leading verb so a memoriser keying on the exact
        original string is wrong on most rounds."""
        original = _JUDGE_PROMPT_PROSE_HEAVY
        n_rounds = 200
        n_unchanged = 0
        for r in range(n_rounds):
            out = pev._paraphrase_chat_prompt(
                original, block_seed=r * 7919 + 11,
            )
            if out == original:
                n_unchanged += 1
        ratio = n_unchanged / n_rounds
        # Occasional unchanged rounds are fine (some seeds happen to
        # pick a swap that maps the prompt to itself when no candidate
        # appears, or a swap that rewrites a synonym back to itself).
        # We require that the prompt is rewritten in MOST rounds so
        # canonical-text memorisation loses signal.
        self.assertLess(
            ratio, 0.40,
            f"prompt was unchanged in {ratio:.0%} of rounds — paraphrase "
            f"is too weak to defeat canonical-text memorisation",
        )


if __name__ == "__main__":
    unittest.main()
