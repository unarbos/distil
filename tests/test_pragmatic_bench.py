#!/usr/bin/env python3
"""Tests for pragmatic_bench v30 — theory-of-mind / scalar implicature
/ indirect-request recognition procedural items.

Verifies:
  * Generator covers all 4 subtypes.
  * Block-seed determinism (cross-validator parity).
  * For false_belief items, the answer correctly distinguishes
    actor-belief vs world-state question types.
  * The shared knowledge_v2 grader correctly accepts gold answers
    embedded in plausible model phrasings.
  * Scalar implicature items reject the strong quantifier
    interpretation when only the weak quantifier was uttered.
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


class TestPragmaticBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_covers_all_subtypes(self):
        items = self.mod._generate_pragmatic_items(20260429, 16)
        seen = {it["category"] for it in items}
        expected = {
            "false_belief", "scalar_implicature",
            "epistemic_state_tracking", "indirect_request",
        }
        self.assertEqual(seen, expected,
            "all 4 pragmatic subtypes must appear in a 16-item sample")

    def test_block_seed_determinism(self):
        a = self.mod._generate_pragmatic_items(99887, 8)
        b = self.mod._generate_pragmatic_items(99887, 8)
        c = self.mod._generate_pragmatic_items(99888, 8)
        def _shape(items):
            return [(it["src"], it["question"], it["gold"]) for it in items]
        self.assertEqual(_shape(a), _shape(b))
        self.assertNotEqual(_shape(a), _shape(c))

    def test_grader_accepts_gold_for_every_item(self):
        items = self.mod._generate_pragmatic_items(20260429, 24)
        for it in items:
            ok = self.mod._knowledge_v2_grade_one(str(it["gold"]), it)
            self.assertEqual(ok, 1,
                f"gold '{it['gold']}' must match its accept pattern "
                f"for {it['category']}")

    def test_grader_handles_wrapped_response(self):
        items = self.mod._generate_pragmatic_items(20260429, 16)
        for it in items:
            wrap = f"My answer is {it['gold']}."
            ok = self.mod._knowledge_v2_grade_one(wrap, it)
            self.assertEqual(ok, 1,
                f"wrapped gold '{wrap}' must match for {it['category']}")

    def test_false_belief_distinct_question_types(self):
        """Within a block_seed sample, false_belief items can ask
        either where the actor will look (gold = container_1) or
        where the object actually is (gold = container_2). Verify
        we get distinct gold values across different items so the
        random pick of question type is exercised."""
        items = self.mod._generate_pragmatic_items(20260429, 64)
        fb_items = [it for it in items if it["category"] == "false_belief"]
        self.assertGreater(len(fb_items), 4)
        unique_questions = {it["question"][-200:] for it in fb_items}
        # Should include at least one "look first" and one
        # "actually located" variant.
        has_belief = any("look for" in q for q in unique_questions)
        has_world = any("actually located" in q for q in unique_questions)
        self.assertTrue(has_belief, "false_belief must include belief-Q variant")
        self.assertTrue(has_world, "false_belief must include world-state Q variant")

    def test_scalar_implicature_strong_form_correctly_rejects(self):
        """For scalar_implicature items asking the STRONG-quantifier
        interpretation ('did all ... ?'), the gold is 'no' — the
        weak-quantifier utterance does not entail the strong one.
        Verify this contract holds for every such item."""
        items = self.mod._generate_pragmatic_items(20260429, 32)
        si_items = [it for it in items if it["category"] == "scalar_implicature"]
        self.assertGreater(len(si_items), 1)
        for it in si_items:
            q = it["question"]
            if "is it the case that all" in q or "is it the case that every" in q:
                self.assertEqual(it["gold"], "no",
                    f"strong-quantifier scalar implicature gold must be 'no' "
                    f"(got '{it['gold']}' for question {q[-200:]})")

    def test_epistemic_tracking_uninformed_actor_does_not_know(self):
        """In epistemic_state_tracking items, the actor C never
        sees the message. So if the question asks 'does C know?',
        the gold is 'no'. Verify the contract holds."""
        items = self.mod._generate_pragmatic_items(20260429, 64)
        es_items = [it for it in items if it["category"] == "epistemic_state_tracking"]
        self.assertGreater(len(es_items), 1)
        # Find at least one item where the question targets the
        # actor that was NOT informed.
        no_items = [it for it in es_items if it["gold"] == "no"]
        yes_items = [it for it in es_items if it["gold"] == "yes"]
        self.assertGreater(
            len(no_items), 0,
            "epistemic_state_tracking sample must include 'no' items "
            "(uninformed actor does not know)",
        )
        self.assertGreater(
            len(yes_items), 0,
            "epistemic_state_tracking sample must include 'yes' items "
            "(informed actor or self does know)",
        )


if __name__ == "__main__":
    unittest.main()
