#!/usr/bin/env python3
"""Tests for the block-seeded procedural benchmark generator."""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub():
    """`pod_eval_vllm` only needs torch at import time for dtype defaults here."""
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


class TestProceduralBenchGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_generator_covers_all_task_families(self):
        items = self.mod._generate_procedural_items(8042152, 10)
        srcs = {it["src"] for it in items}
        self.assertEqual(len(items), 10)
        self.assertTrue({
            "procedural/reasoning",
            "procedural/instruction",
            "procedural/retrieval",
            "procedural/table",
            "procedural/constraint",
        }.issubset(srcs))

    def test_generation_is_block_seed_deterministic(self):
        a = self.mod._generate_procedural_items(12345, 8)
        b = self.mod._generate_procedural_items(12345, 8)
        c = self.mod._generate_procedural_items(12346, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_gold_answers_are_exactly_matchable(self):
        for item in self.mod._generate_procedural_items(8042152, 10):
            gold = str(item["answer"])
            self.assertTrue(gold)
            self.assertTrue(self.mod._answer_exact_in_text(gold, f"answer: {gold}"))

    def test_strict_grader_rejects_verbose_answer(self):
        gold = "42"
        self.assertTrue(self.mod._answer_exact_in_text(gold, "42", strict=True))
        self.assertTrue(self.mod._answer_exact_in_text(gold, r"\boxed{42}", strict=True))
        self.assertFalse(self.mod._answer_exact_in_text(gold, "The answer is 42.", strict=True))


if __name__ == "__main__":
    unittest.main()
