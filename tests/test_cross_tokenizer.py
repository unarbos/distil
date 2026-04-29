#!/usr/bin/env python3
"""Tests for the cross-tokenizer helper used by the Stage-2 Kimi K2.6
teacher swap. The helper is scaffolding; these tests verify the
shape/contract matches the existing teacher-cache schema and that the
edit-distance helper handles edge cases.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _MockTokenizer:
    """Minimal HF-tokenizer-shape mock for unit tests."""

    def __init__(self, vocab_to_id: dict[str, int]):
        self.vocab_to_id = vocab_to_id
        self.id_to_vocab = {v: k for k, v in vocab_to_id.items()}

    def __call__(self, text: str, add_special_tokens: bool = True,
                 return_tensors=None):
        ids: list[int] = []
        cur = ""
        for ch in text:
            cur += ch
            if cur in self.vocab_to_id:
                ids.append(self.vocab_to_id[cur])
                cur = ""
        # Trailing chars: try byte-level fallback (use first char id 0)
        if cur:
            ids.append(0)
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.id_to_vocab.get(int(i), "") for i in ids)


class TestEditDistance(unittest.TestCase):
    def test_identical_strings_zero_distance(self):
        from eval.cross_tokenizer import _normalised_edit_distance
        self.assertEqual(_normalised_edit_distance("hello", "hello"), 0.0)

    def test_completely_different(self):
        from eval.cross_tokenizer import _normalised_edit_distance
        d = _normalised_edit_distance("hello", "world")
        self.assertGreater(d, 0.5)

    def test_empty_strings(self):
        from eval.cross_tokenizer import _normalised_edit_distance
        self.assertEqual(_normalised_edit_distance("", ""), 0.0)
        self.assertEqual(_normalised_edit_distance("a", ""), 1.0)
        self.assertEqual(_normalised_edit_distance("", "a"), 1.0)

    def test_one_character_off(self):
        from eval.cross_tokenizer import _normalised_edit_distance
        d = _normalised_edit_distance("hello", "hellz")
        self.assertAlmostEqual(d, 0.2, places=2)


class TestRetokenize(unittest.TestCase):
    def setUp(self):
        # Both tokenizers see the same vocab so round-trip is lossless.
        self.tok = _MockTokenizer({
            "the": 1, "quick": 2, " ": 3, "brown": 4, "fox": 5,
            "h": 10, "e": 11, "l": 12, "o": 13,
        })

    def test_round_trip_round_trips(self):
        from eval.cross_tokenizer import (
            decode_with_kimi_tokenizer, retokenize_to_qwen, round_trip_drift,
        )
        kimi_ids = [1, 3, 2, 3, 4, 3, 5]  # "the quick brown fox"
        text = decode_with_kimi_tokenizer(kimi_ids, self.tok)
        self.assertEqual(text, "the quick brown fox")
        qwen_ids = retokenize_to_qwen(text, self.tok)
        self.assertEqual(qwen_ids, kimi_ids)
        self.assertEqual(round_trip_drift(text, qwen_ids, self.tok), 0.0)

    def test_handles_empty_inputs(self):
        from eval.cross_tokenizer import (
            decode_with_kimi_tokenizer, retokenize_to_qwen,
        )
        self.assertEqual(decode_with_kimi_tokenizer([], self.tok), "")
        self.assertEqual(retokenize_to_qwen("", self.tok), [])

    def test_handles_none_tokenizer(self):
        from eval.cross_tokenizer import (
            decode_with_kimi_tokenizer, retokenize_to_qwen,
        )
        self.assertEqual(decode_with_kimi_tokenizer([1, 2, 3], None), "")
        self.assertEqual(retokenize_to_qwen("hello", None), [])


class TestAlignLogprobs(unittest.TestCase):
    def test_align_returns_correct_schema(self):
        from eval.cross_tokenizer import align_logprobs_kimi_to_qwen
        tok = _MockTokenizer({
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4,
        })
        kimi_top_k_indices = [[0, 1, 2], [3, 4, 0]]
        kimi_top_k_logprobs = [[-0.1, -1.0, -2.0], [-0.3, -0.5, -3.0]]
        result = align_logprobs_kimi_to_qwen(
            kimi_top_k_indices, kimi_top_k_logprobs,
            tok, tok, k=3,
        )
        self.assertIn("indices", result)
        self.assertIn("values", result)
        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(len(result["values"]), 2)

    def test_align_handles_none_tokenizer(self):
        from eval.cross_tokenizer import align_logprobs_kimi_to_qwen
        result = align_logprobs_kimi_to_qwen(
            [[0, 1]], [[-0.1, -1.0]], None, None, k=3,
        )
        self.assertEqual(result, {"indices": [], "values": []})


class TestDriftSummary(unittest.TestCase):
    def test_summary_basic_stats(self):
        from eval.cross_tokenizer import stage2_drift_summary
        s = stage2_drift_summary([0.0, 0.05, 0.10, 0.15, 0.20])
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 0.10, places=2)
        self.assertEqual(s["max"], 0.20)

    def test_summary_empty(self):
        from eval.cross_tokenizer import stage2_drift_summary
        s = stage2_drift_summary([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["mean"])


if __name__ == "__main__":
    unittest.main()
