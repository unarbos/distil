#!/usr/bin/env python3
"""Tests for knowledge_bench v2 — open-ended factual reasoning generator.

v30 (2026-04-29). Replaced the legacy MMLU-Pro 10-way MC pool with
procedural fact-like items (price tables, transitive ordering, container
counting, alphabet/calendar/weekday/unit/roman conventions). Verifies:

  * All 8 subtypes appear across a moderate sample (round-stable).
  * Generation is deterministic in block_seed (cross-validator parity).
  * Per-item ``accept`` regex correctly recognises the gold answer
    embedded in plausible model phrasings.
  * Per-item ``accept`` regex correctly rejects boundary cases like
    ``16`` vs ``116`` or ``R`` vs ``RR`` so a substring leak doesn't
    score the model spuriously correct.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_torch.long = object()
    fake_torch.compile = lambda fn, **_kwargs: fn
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")
    fake_f.kl_div = lambda *_args, **_kwargs: None
    fake_nn.functional = fake_f
    fake_torch.nn = fake_nn
    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("torch.nn", fake_nn)
    sys.modules.setdefault("torch.nn.functional", fake_f)


class TestKnowledgeV2Generation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_generation_covers_all_subtypes(self):
        items = self.mod._generate_knowledge_v2_items(20260429, 24)
        seen = {it["category"] for it in items}
        expected = {
            "price_table_total", "transitive_order", "container_count",
            "alphabet_position", "calendar_offset", "day_of_week_offset",
            "unit_convert", "roman_numeral",
        }
        self.assertEqual(seen, expected,
            "all 8 subtypes must appear in a 24-item sample")

    def test_generation_is_block_seed_deterministic(self):
        a = self.mod._generate_knowledge_v2_items(12345, 8)
        b = self.mod._generate_knowledge_v2_items(12345, 8)
        c = self.mod._generate_knowledge_v2_items(12346, 8)
        # Compiled regex objects don't compare cleanly, so compare the
        # serialisable fields only.
        def _shape(items):
            return [
                (it["src"], it["category"], it["question"], it["gold"], it["accept_src"])
                for it in items
            ]
        self.assertEqual(_shape(a), _shape(b),
            "same block_seed must produce identical items")
        self.assertNotEqual(_shape(a), _shape(c),
            "different block_seed must produce different items")

    def test_each_item_has_required_fields(self):
        items = self.mod._generate_knowledge_v2_items(99887, 8)
        for it in items:
            self.assertIn("src", it)
            self.assertIn("category", it)
            self.assertIn("question", it)
            self.assertIn("gold", it)
            self.assertIn("accept", it)
            self.assertIn("accept_src", it)
            self.assertTrue(it["question"], "question must be non-empty")
            self.assertTrue(it["gold"], "gold must be non-empty")
            self.assertTrue(it["accept"], "accept patterns must be non-empty")

    def test_grader_accepts_canonical_gold(self):
        """For every generated item, the canonical gold string by itself
        is accepted by the grader. This is the minimum viable correctness:
        a model that emits exactly the gold gets credit."""
        items = self.mod._generate_knowledge_v2_items(20260429, 24)
        for it in items:
            ok = self.mod._knowledge_v2_grade_one(str(it["gold"]), it)
            self.assertEqual(ok, 1,
                f"gold '{it['gold']}' must match its own accept pattern "
                f"for {it['category']}")

    def test_grader_accepts_gold_in_sentence(self):
        """A model that emits a short conversational wrap around the gold
        ('The answer is X.') still gets credit."""
        items = self.mod._generate_knowledge_v2_items(20260429, 24)
        for it in items:
            wrapped = f"The answer is {it['gold']}."
            ok = self.mod._knowledge_v2_grade_one(wrapped, it)
            self.assertEqual(ok, 1,
                f"wrapped gold '{wrapped}' must match for {it['category']}")

    def test_grader_rejects_boundary_substring(self):
        """A substring of the gold inside a longer number/word must NOT
        match. This is the critical correctness for the boundary-protected
        accept patterns: gold='8' inside '180' is a different number,
        gold='R' inside 'RR' is a different identifier."""
        # Force a price_table_total item with small gold to test
        # numeric boundary protection.
        items = self.mod._generate_knowledge_v2_items(20260429, 64)
        small_gold_items = [it for it in items
                            if it["category"] in ("price_table_total",
                                                  "container_count",
                                                  "unit_convert")
                            and it["gold"].isdigit()
                            and 1 <= int(it["gold"]) <= 9]
        self.assertTrue(small_gold_items,
            "test fixture should include at least one single-digit gold")
        for it in small_gold_items:
            confounder = f"{it['gold']}{it['gold']}"  # double-digit "11", "22", etc.
            ok = self.mod._knowledge_v2_grade_one(confounder, it)
            self.assertEqual(ok, 0,
                f"confounder '{confounder}' must NOT match gold "
                f"'{it['gold']}' for {it['category']}")

    def test_grader_rejects_unrelated_text(self):
        items = self.mod._generate_knowledge_v2_items(20260429, 24)
        for it in items:
            # The string 'zorgblat' is procedurally improbable; every
            # gold should reject it.
            ok = self.mod._knowledge_v2_grade_one("zorgblat", it)
            self.assertEqual(ok, 0,
                f"unrelated text must not match gold '{it['gold']}' "
                f"for {it['category']}")

    def test_grader_handles_empty_input(self):
        items = self.mod._generate_knowledge_v2_items(20260429, 8)
        for it in items:
            self.assertEqual(self.mod._knowledge_v2_grade_one("", it), 0)
            self.assertEqual(self.mod._knowledge_v2_grade_one("   ", it), 0)


if __name__ == "__main__":
    unittest.main()
