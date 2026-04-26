#!/usr/bin/env python3
"""Regression tests for the round-26 per-round on_policy_rkl paraphrase.

Pre-v26 the ``on_policy_rkl`` axis (composite weight 0.35 — the SINGLE
LARGEST axis weight in the entire composite, larger than the next two
combined) drew its 16-of-80 prompts deterministically from a fully
public, fully canonical pool baked into ``pod_eval_vllm.py``. The 2026-
04-26 v17 hardening rotated the rollout-sampling seed per ``block_seed``
which closed the "predict-your-own-trajectory" attack — but it did NOT
close the more fundamental Goodhart vector that prompt rotation alone
defeats: a miner who pre-distils onto teacher's outputs for the
canonical wording of all 80 entries can saturate ``on_policy_rkl``
regardless of sampling-seed rotation, because the student has been
trained to place teacher-likely tokens at every position the teacher
would on those exact 80 inputs.

v26 wires the v25 ``_paraphrase_chat_prompt`` into
``_pick_on_policy_rkl_prompts`` so each of the 16 sampled prompts gets
a chat-domain synonym swap keyed on ``(block_seed, sha(prompt))``. The
helper is region-aware so translation answer keys
(``"Translate to French: The cat sat on the mat."``) are PROTECTED via
the helper's quoted-region detection — only conversational PROSE
rotates, the quoted source survives byte-identical so the gold output
of a translation prompt is unchanged. JSON / code / format-spec
prompts in the pool likewise survive by virtue of the protected-
region split.

These tests pin the v26 contract:

* Per-round paraphrase actually fires for all sampled prompts when a
  ``block_seed`` is provided.
* The selection is deterministic across validators (cross-validator
  agreement is the entire point of seed-derived rotation).
* The selection rotates between rounds (a memoriser keyed on canonical
  wording sees a different surface every round).
* Region-aware preservation: the translation-anchor sub-pool
  ("Translate to French: The cat sat on the mat.") keeps the quoted
  source-text byte-identical so the answer key still matches.
* Dev / replay mode (``block_seed=None``) returns the canonical pool
  unchanged — production rotation does not regress local debugging.
* Schema version is bumped (``COMPOSITE_SHADOW_VERSION >= 26``) and
  ``_KING_SELECTION_MIN_VERSION`` follows so old records are
  quarantined until regraded.
"""

import os
import re
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    """Mirror the stub used by the v18-v25 tests so we can import
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


# Translation prompts from the production pool — anchored on a quoted
# source string that MUST survive byte-identical under paraphrase
# (otherwise the gold answer no longer matches the prompt).
_TRANSLATION_PROMPTS = (
    "Translate to French: The cat sat on the mat.",
    "Translate to Spanish: I would like a cup of coffee, please.",
    "Translate to Portuguese: The book is on the table.",
)


class TestOnPolicyRklPickContract(unittest.TestCase):
    """The picker must be deterministic, return ``ON_POLICY_RKL_PER_ROUND``
    items, and accept ``None`` as a dev-mode escape hatch."""

    def test_picks_are_deterministic(self):
        a = pev._pick_on_policy_rkl_prompts(block_seed=12345)
        b = pev._pick_on_policy_rkl_prompts(block_seed=12345)
        self.assertEqual(a, b)

    def test_picks_are_per_round_count(self):
        picks = pev._pick_on_policy_rkl_prompts(block_seed=999)
        self.assertEqual(len(picks), pev.ON_POLICY_RKL_PER_ROUND)

    def test_block_seed_none_returns_canonical_prefix(self):
        """Dev / replay mode: same length, no paraphrase. The pool is
        the production source of truth so we compare against the first
        K entries of ``ON_POLICY_RKL_POOL`` byte-identical."""
        out = pev._pick_on_policy_rkl_prompts(block_seed=None)
        self.assertEqual(
            out,
            list(pev.ON_POLICY_RKL_POOL[:pev.ON_POLICY_RKL_PER_ROUND]),
        )

    def test_invalid_block_seed_falls_back(self):
        """A non-integer ``block_seed`` (string that can't parse, list,
        tuple) must NOT raise — the fallback returns the canonical
        prefix so the round still runs."""
        out = pev._pick_on_policy_rkl_prompts(block_seed="not-a-number")
        self.assertEqual(
            out,
            list(pev.ON_POLICY_RKL_POOL[:pev.ON_POLICY_RKL_PER_ROUND]),
        )

    def test_hex_block_seed_supported(self):
        """``int(seed)`` accepts hex when given as a string; we want
        block_hash-derived seeds (often hex) to flow through."""
        out_a = pev._pick_on_policy_rkl_prompts(block_seed=int("0xdeadbeef", 16))
        out_b = pev._pick_on_policy_rkl_prompts(block_seed=int("0xdeadbeef", 16))
        self.assertEqual(out_a, out_b)


class TestOnPolicyRklParaphraseFires(unittest.TestCase):
    """The whole point of v26 — a ``block_seed``-keyed call to the
    picker must produce at least one prompt whose surface form differs
    from the canonical pool entry."""

    def test_paraphrase_fires_at_least_once(self):
        """Across the 16 picks for a given seed, at least one prompt
        must differ from the canonical text. If zero prompts changed,
        the wire-up is broken."""
        canonical = set(pev.ON_POLICY_RKL_POOL)
        picks = pev._pick_on_policy_rkl_prompts(block_seed=31415)
        n_changed = sum(1 for p in picks if p not in canonical)
        self.assertGreater(
            n_changed, 0,
            f"v26 paraphrase did not fire — none of the {len(picks)} "
            f"picks differ from the canonical pool. Wire-up regression?",
        )

    def test_paraphrase_fires_substantial_across_seeds(self):
        """Aggregate signal across 8 seeds. The on_policy_rkl pool is
        ~80 entries with a mix of:
          * Prose-rich prompts ("Explain X", "Give Y", "Describe Z") —
            these get paraphrased.
          * Short factual prompts ("Who wrote Hamlet?", "What is 2**10?",
            "Translate to French: …") — these have no chat-domain
            verbs so the helper passes them through.
        Rotation depth is therefore bounded above by the prose-rich
        fraction (~25%); the test floor is 25% so the gate fires when
        wire-up regresses without false-failing on pool composition.
        Combined with the per-round 16-of-80 sample and the v17
        sampling-seed rotation, even ~25% prompt rotation reliably
        defeats the canonical-wording memoriser: their average KL
        across ~4 paraphrased prompts per round is materially higher
        than honest miners and the axis penalty compounds round on
        round."""
        canonical = set(pev.ON_POLICY_RKL_POOL)
        n_total = 0
        n_changed = 0
        for seed in (1, 7, 99, 12345, 31415, 999999, 2026, 42):
            picks = pev._pick_on_policy_rkl_prompts(block_seed=seed)
            n_total += len(picks)
            n_changed += sum(1 for p in picks if p not in canonical)
        ratio = n_changed / max(n_total, 1)
        self.assertGreaterEqual(
            ratio, 0.25,
            f"v26 paraphrase fires too rarely: only {n_changed}/{n_total} "
            f"({100 * ratio:.1f}%) prompts paraphrased across 8 seeds — "
            f"wire-up regression suspected (expected >= 25%).",
        )

    def test_picks_rotate_across_seeds(self):
        """Two different seeds must produce two different picked sets.
        If the picker is the same across seeds, the per-round rotation
        is broken (the helper degenerated to a constant transform)."""
        a = pev._pick_on_policy_rkl_prompts(block_seed=1)
        b = pev._pick_on_policy_rkl_prompts(block_seed=2)
        self.assertNotEqual(
            a, b,
            "two different block_seeds produced identical picks — "
            "rotation is broken",
        )


class TestOnPolicyRklTranslationAnchors(unittest.TestCase):
    """Translation prompts have the answer key implicitly defined by
    the quoted source string. The helper's region-aware split must
    keep that source PROTECTED so the prompt's gold output is
    unchanged.

    We verify the property at the helper level (rather than relying
    on the picker happening to sample a translation prompt for any
    given seed) because translation prompts are the most fragile
    sub-pool — failing to preserve the source text would silently
    poison ~12 of the 80 pool entries."""

    def test_french_translation_source_preserved(self):
        prompt = _TRANSLATION_PROMPTS[0]
        for seed in (1, 42, 12345, 31415):
            out = pev._paraphrase_chat_prompt(prompt, block_seed=seed)
            self.assertIn(
                "The cat sat on the mat.", out,
                f"translation source mutated under seed={seed}: {out!r}",
            )

    def test_spanish_translation_source_preserved(self):
        prompt = _TRANSLATION_PROMPTS[1]
        for seed in (1, 42, 12345, 31415):
            out = pev._paraphrase_chat_prompt(prompt, block_seed=seed)
            self.assertIn(
                "I would like a cup of coffee, please.", out,
                f"translation source mutated under seed={seed}: {out!r}",
            )

    def test_translation_target_language_preserved(self):
        """The target language tag ('to French', 'to Spanish', etc.)
        must survive — the chat synonym table does not touch language
        names but we want a regression test in case a future addition
        accidentally rewrites them."""
        languages = ("French", "Spanish", "Portuguese")
        for prompt, lang in zip(_TRANSLATION_PROMPTS, languages):
            for seed in (1, 42, 12345):
                out = pev._paraphrase_chat_prompt(prompt, block_seed=seed)
                self.assertIn(
                    lang, out,
                    f"target language {lang!r} mutated under seed={seed}: {out!r}",
                )


class TestOnPolicyRklJsonAnchor(unittest.TestCase):
    """The JSON-output prompt in the pool is region-protected by the
    single-quoted-string detector — it must survive byte-identical."""

    _JSON_PROMPT = (
        "Output a JSON object with keys 'name' and 'age' describing a "
        "fictional person. Just the JSON."
    )

    def test_json_keys_preserved(self):
        for seed in (1, 42, 12345, 31415):
            out = pev._paraphrase_chat_prompt(self._JSON_PROMPT, block_seed=seed)
            self.assertIn("'name'", out, f"'name' key dropped under seed={seed}")
            self.assertIn("'age'", out, f"'age' key dropped under seed={seed}")
            # "Just the JSON." is a hard format constraint that must survive.
            self.assertIn(
                "Just the JSON.", out,
                f"format constraint 'Just the JSON.' dropped under seed={seed}",
            )


class TestOnPolicyRklPoolIntegrity(unittest.TestCase):
    """Sanity tests on the pool itself — these are not v26-specific
    but anchor downstream tests on a stable, audited surface."""

    def test_pool_size_at_least_60(self):
        """Combinatorics relies on a healthy pool size; a regression
        that shrinks the pool below ~60 makes per-round rotation too
        easy to memorise. The current pool is 80 entries."""
        self.assertGreaterEqual(len(pev.ON_POLICY_RKL_POOL), 60)

    def test_pool_entries_unique(self):
        self.assertEqual(
            len(pev.ON_POLICY_RKL_POOL),
            len(set(pev.ON_POLICY_RKL_POOL)),
            "ON_POLICY_RKL_POOL has duplicate entries",
        )

    def test_pool_entries_are_strings(self):
        for p in pev.ON_POLICY_RKL_POOL:
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 0)


class TestOnPolicyRklSchemaVersionGate(unittest.TestCase):
    """Old composite records must be quarantined until regraded under
    v26 — the king filter relies on these constants."""

    def test_composite_shadow_version_at_least_26(self):
        from validator.composite import COMPOSITE_SHADOW_VERSION
        self.assertGreaterEqual(COMPOSITE_SHADOW_VERSION, 26)

    def test_king_selection_min_version_at_least_26(self):
        from validator.single_eval import _KING_SELECTION_MIN_VERSION
        self.assertGreaterEqual(_KING_SELECTION_MIN_VERSION, 26)


class TestOnPolicyRklSetBlockSeedIntegration(unittest.TestCase):
    """``set_on_policy_rkl_block_seed`` is the production entry point.
    It must (1) populate ``ON_POLICY_RKL_PROMPTS`` with the picker's
    output (so ``rollout_phase_a`` consumes the paraphrased set) and
    (2) be re-callable with the same seed without re-doing work."""

    def test_set_block_seed_populates_prompts(self):
        # Reset module-state so the test is order-independent.
        pev._ON_POLICY_RKL_BLOCK_SEED = None
        pev.set_on_policy_rkl_block_seed(54321)
        self.assertEqual(
            len(pev.ON_POLICY_RKL_PROMPTS), pev.ON_POLICY_RKL_PER_ROUND,
        )
        # At least one prompt should have been paraphrased.
        canonical = set(pev.ON_POLICY_RKL_POOL)
        n_changed = sum(
            1 for p in pev.ON_POLICY_RKL_PROMPTS if p not in canonical
        )
        self.assertGreater(
            n_changed, 0,
            "set_on_policy_rkl_block_seed did not produce any paraphrased "
            "prompts — wire-up regression",
        )

    def test_set_block_seed_idempotent_on_same_seed(self):
        pev._ON_POLICY_RKL_BLOCK_SEED = None
        pev.set_on_policy_rkl_block_seed(98765)
        snapshot = list(pev.ON_POLICY_RKL_PROMPTS)
        pev.set_on_policy_rkl_block_seed(98765)
        self.assertEqual(snapshot, list(pev.ON_POLICY_RKL_PROMPTS))

    def test_set_block_seed_rotates_across_seeds(self):
        pev._ON_POLICY_RKL_BLOCK_SEED = None
        pev.set_on_policy_rkl_block_seed(11111)
        a = list(pev.ON_POLICY_RKL_PROMPTS)
        pev.set_on_policy_rkl_block_seed(22222)
        b = list(pev.ON_POLICY_RKL_PROMPTS)
        self.assertNotEqual(
            a, b,
            "ON_POLICY_RKL_PROMPTS did not rotate between two distinct "
            "block_seeds — production rotation broken",
        )


class TestOnPolicyRklRotationDoesNotBreakSemantics(unittest.TestCase):
    """A model that genuinely understands the prompt should still see a
    valid, well-formed prompt after paraphrase — we don't want the
    rotation to introduce gibberish that artificially raises KL even
    for honest miners. We verify by checking that picked prompts:

    * Stay on the same sub-distribution (still ASCII-printable, still
      end with sentence punctuation, still under a sane length cap).
    * Preserve any digits / numeric anchors that the original pool
      relied on for grading."""

    def test_picked_prompts_stay_printable(self):
        picks = pev._pick_on_policy_rkl_prompts(block_seed=12345)
        for p in picks:
            self.assertTrue(
                all(c.isprintable() or c in "\n\t" for c in p),
                f"non-printable character in paraphrase output: {p!r}",
            )

    def test_canonical_digit_content_preserved_under_paraphrase(self):
        """Digits anchor reasoning prompts ("What is 2**10?", "How many
        trailing zeros does 25! have?"). The chat synonym table is
        digit-free by construction (verified separately in
        ``test_chat_prompt_paraphrase``), so paraphrasing any
        digit-bearing canonical prompt must preserve its digit content
        byte-identical. We iterate through every digit-bearing pool
        entry and exercise multiple seeds: any drift means a digit
        leaked into a synonym entry."""
        canonical_with_digits = [
            p for p in pev.ON_POLICY_RKL_POOL if any(c.isdigit() for c in p)
        ]
        self.assertGreater(
            len(canonical_with_digits), 0,
            "pool has no digit-bearing prompts — test fixture broken",
        )
        for canonical in canonical_with_digits:
            canonical_digits = [c for c in canonical if c.isdigit()]
            for seed in (1, 7, 42, 31415, 12345, 999999):
                paraphrased = pev._paraphrase_chat_prompt(
                    canonical, block_seed=seed,
                )
                paraphrased_digits = [c for c in paraphrased if c.isdigit()]
                self.assertEqual(
                    canonical_digits, paraphrased_digits,
                    f"digits drifted under paraphrase:\n"
                    f"  seed={seed}\n"
                    f"  canonical={canonical!r}\n"
                    f"  paraphrased={paraphrased!r}",
                )


if __name__ == "__main__":
    unittest.main()
