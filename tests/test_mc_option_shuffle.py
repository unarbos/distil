#!/usr/bin/env python3
"""Regression tests for the round-20 per-round MC option shuffle.

Pre-v20 the ARC and MMLU-Pro pools shipped with a fixed correct-letter per
question (the raw dataset order) and ``truthful_bench`` only shuffled
per-question at pool-load time. A miner who pre-trained on the public
``allenai/ai2_arc`` and ``TIGER-Lab/MMLU-Pro`` datasets could build a
``{question_text → correct_letter}`` lookup and saturate ``arc_bench``
without parsing the options. Round 18 caught this in the wild: 8 distinct
miners scored ``arc_bench=1.000`` while their ``knowledge_bench`` sat at
0.0–0.25 — a textbook letter-memorisation signature.

These tests pin the contract:

* ``_shuffle_mc_options_for_round`` produces a coherent shuffled item.
* The shuffled gold_letter matches the shuffled option position.
* Determinism: same (item, block_seed) → same shuffle.
* Cross-validator agreement: same item from two validators with the
  same block_seed → same shuffle.
* Per-round rotation: same item, different block_seeds → different
  letter for the correct answer (with high probability).
* Per-question independence: two different questions in the same round
  shuffle independently.
* ARC-shape and MMLU-shape are both supported.
* Items without an MC shape pass through untouched.
* ``block_seed=None`` returns the item unchanged (dev/replay mode).
"""

import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _install_torch_stub():
    """Identical to ``test_capability_procedural`` — the shuffle helper
    lives in ``pod_eval_vllm`` whose top-level imports include torch."""
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


def _arc_item(q="What planet is closest to the sun?"):
    """An ARC-shape item with gold letter 'A' (Mercury at index 0)."""
    return {
        "src": "arc-challenge",
        "question": q,
        "labels": ["A", "B", "C", "D"],
        "texts": ["Mercury", "Venus", "Earth", "Mars"],
        "gold_letter": "A",
    }


def _mmlu_item(q="Which gas is most abundant in Earth's atmosphere?"):
    """An MMLU-shape item with gold letter 'B' (Nitrogen at index 1)."""
    return {
        "src": "mmlu-pro",
        "question": q,
        "options": ["Oxygen", "Nitrogen", "Argon", "Carbon Dioxide"],
        "gold_letter": "B",
        "category": "chem",
    }


class TestShuffleCorrectness(unittest.TestCase):
    """The shuffled item must remain self-consistent."""

    def test_arc_gold_letter_matches_text(self):
        """After shuffling an ARC item, gold_letter must point at the
        original gold answer's text in the new ordering."""
        original = _arc_item()
        original_gold_text = original["texts"][
            original["labels"].index(original["gold_letter"])
        ]
        shuffled = pev._shuffle_mc_options_for_round(original, block_seed=12345)
        new_gold_idx = shuffled["labels"].index(shuffled["gold_letter"])
        self.assertEqual(shuffled["texts"][new_gold_idx], original_gold_text)
        self.assertEqual(set(shuffled["texts"]), set(original["texts"]))

    def test_mmlu_gold_letter_matches_option(self):
        original = _mmlu_item()
        original_gold_text = original["options"][
            ord(original["gold_letter"]) - ord("A")
        ]
        shuffled = pev._shuffle_mc_options_for_round(original, block_seed=98765)
        new_gold_idx = ord(shuffled["gold_letter"]) - ord("A")
        self.assertEqual(shuffled["options"][new_gold_idx], original_gold_text)
        self.assertEqual(set(shuffled["options"]), set(original["options"]))

    def test_labels_remain_canonical_letters(self):
        """ARC keeps A/B/C/D as labels — only the texts reorder."""
        shuffled = pev._shuffle_mc_options_for_round(_arc_item(), block_seed=42)
        self.assertEqual(shuffled["labels"], ["A", "B", "C", "D"])

    def test_no_options_lost(self):
        """Length and set of options are preserved."""
        for item, key in (
            (_arc_item(), "texts"),
            (_mmlu_item(), "options"),
        ):
            for seed in (1, 2, 3, 4, 5, 100, 9999):
                shuffled = pev._shuffle_mc_options_for_round(item, block_seed=seed)
                self.assertEqual(len(shuffled[key]), len(item[key]))
                self.assertEqual(set(shuffled[key]), set(item[key]))


class TestShuffleDeterminism(unittest.TestCase):
    """Cross-validator agreement: same input → same output."""

    def test_arc_deterministic(self):
        item = _arc_item()
        a = pev._shuffle_mc_options_for_round(item, block_seed=12345)
        b = pev._shuffle_mc_options_for_round(item, block_seed=12345)
        self.assertEqual(a, b)

    def test_mmlu_deterministic(self):
        item = _mmlu_item()
        a = pev._shuffle_mc_options_for_round(item, block_seed=12345)
        b = pev._shuffle_mc_options_for_round(item, block_seed=12345)
        self.assertEqual(a, b)

    def test_input_not_mutated(self):
        item = _arc_item()
        snap = {
            "labels": list(item["labels"]),
            "texts": list(item["texts"]),
            "gold_letter": item["gold_letter"],
        }
        pev._shuffle_mc_options_for_round(item, block_seed=42)
        self.assertEqual(item["labels"], snap["labels"])
        self.assertEqual(item["texts"], snap["texts"])
        self.assertEqual(item["gold_letter"], snap["gold_letter"])


class TestShuffleRotation(unittest.TestCase):
    """Per-round rotation: different seeds → different letter
    distribution. We can't demand "always different" because some seeds
    happen to leave a small option list partly fixed; we demand that
    over many seeds the gold letter spans a wide range and that for a
    given item the gold letter changes for at least one of every two
    seeds."""

    def test_arc_letter_rotates_across_seeds(self):
        """Across many seeds the gold letter visits every label."""
        item = _arc_item()
        seen = set()
        for seed in range(64):
            shuf = pev._shuffle_mc_options_for_round(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold_letter"])
        self.assertEqual(seen, {"A", "B", "C", "D"})

    def test_mmlu_letter_rotates_across_seeds(self):
        item = _mmlu_item()
        seen = set()
        for seed in range(64):
            shuf = pev._shuffle_mc_options_for_round(item, block_seed=seed * 7919 + 1)
            seen.add(shuf["gold_letter"])
        self.assertEqual(seen, {"A", "B", "C", "D"})

    def test_two_questions_shuffle_independently(self):
        """In the same round, two different questions get distinct
        permutations — the shuffle must not be a global rotation."""
        item_a = _arc_item(q="Which planet has rings?")
        item_b = _arc_item(q="Which planet is the largest?")
        seed = 31415
        shuf_a = pev._shuffle_mc_options_for_round(item_a, block_seed=seed)
        shuf_b = pev._shuffle_mc_options_for_round(item_b, block_seed=seed)
        self.assertNotEqual(shuf_a["texts"], shuf_b["texts"])


class TestShuffleEdgeCases(unittest.TestCase):
    def test_block_seed_none_passes_through(self):
        """Dev/replay mode: no seed → no rotation."""
        item = _arc_item()
        out = pev._shuffle_mc_options_for_round(item, block_seed=None)
        self.assertEqual(out, item)

    def test_unrecognised_shape_passes_through(self):
        item = {"foo": "bar", "question": "?"}
        out = pev._shuffle_mc_options_for_round(item, block_seed=1)
        self.assertEqual(out, item)

    def test_missing_gold_letter_passes_through(self):
        item = {
            "question": "x",
            "labels": ["A", "B"],
            "texts": ["a", "b"],
            "gold_letter": "Z",
        }
        out = pev._shuffle_mc_options_for_round(item, block_seed=1)
        self.assertEqual(out, item)

    def test_hex_block_seed(self):
        """Hex-string block_seed (canonical bittensor format) works."""
        item = _arc_item()
        out = pev._shuffle_mc_options_for_round(item, block_seed="0xdeadbeef")
        new_gold_idx = out["labels"].index(out["gold_letter"])
        original_gold_text = item["texts"][item["labels"].index(item["gold_letter"])]
        self.assertEqual(out["texts"][new_gold_idx], original_gold_text)

    def test_preserves_extra_keys(self):
        """Extra fields like 'src' and 'category' must survive."""
        item = _mmlu_item()
        out = pev._shuffle_mc_options_for_round(item, block_seed=42)
        self.assertEqual(out["src"], item["src"])
        self.assertEqual(out["category"], item["category"])
        self.assertEqual(out["question"], item["question"])


class TestShuffleAntiMemorization(unittest.TestCase):
    """The whole point of round 20: across multiple rounds, the
    correct letter for a given question varies. A miner who learned
    the v19 raw-dataset letter ('A' for our test ARC item) is wrong on
    most rounds."""

    def test_v19_static_lookup_loses_signal(self):
        """Memorising the raw-letter answer ('A' here) is correct on
        only ~1/4 of rounds (uniform distribution over A/B/C/D)."""
        item = _arc_item()
        memorised = item["gold_letter"]
        n_rounds = 200
        n_correct = 0
        for r in range(n_rounds):
            shuf = pev._shuffle_mc_options_for_round(item, block_seed=r * 17 + 3)
            if shuf["gold_letter"] == memorised:
                n_correct += 1
        ratio = n_correct / n_rounds
        self.assertGreater(ratio, 0.10, f"memorised score {ratio:.2%} suspiciously low")
        self.assertLess(ratio, 0.40, f"memorised score {ratio:.2%} suspiciously high")


if __name__ == "__main__":
    unittest.main()
