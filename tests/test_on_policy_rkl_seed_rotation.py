#!/usr/bin/env python3
"""Tests for the Session 3.10 on_policy_rkl per-round seed rotation.

Goodhart context: ``on_policy_rkl`` is the highest-weight axis in the
composite score (it directly measures distillation quality, the entire
point of subnet 97). Pre-3.10 the rollout-sampling seed was the
constant ``ON_POLICY_RKL_SEED=42``. Combined with the prompt-pool
rotation, that meant ``torch.manual_seed(42 + p_idx)`` was the SAME
across rounds for every prompt position. A miner who:

  1. Knew the 80-prompt pool (it's in source — public),
  2. Could iterate over ``set_on_policy_rkl_block_seed`` to enumerate
     which prompt subsets each block produces (also public),
  3. Ran their candidate model with seed=42+p_idx for each prompt,

…could pre-compute the EXACT token sequence their model would emit
during evaluation and surgically train weights so that those exact
tokens align with teacher-high-probability tokens. That's a direct
attack on the metric, not on the underlying objective.

The fix derives a per-round sampling seed from block_seed (XOR with
the base seed). The student's sampling trajectory now varies between
rounds, so per-round overfitting is impractical (a miner would have to
retrain inside the round window, ~1h, after seeing the block_seed).

These tests pin down:
  * Determinism — every validator computes the same derived seed for a
    given block_seed.
  * Rotation — different block_seeds produce different derived seeds.
  * Coupling — calling ``set_on_policy_rkl_block_seed`` updates BOTH the
    prompt pool AND the derived seed (otherwise prompts could rotate
    while the seed stays fixed, half-defeating the defense).
  * Backward compat — passing an explicit ``seed=`` to
    ``on_policy_rollouts`` still works.
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


class TestOnPolicyRklSeedRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def setUp(self):
        # Reset module-level state so each test starts from a clean slate.
        self.mod._ON_POLICY_RKL_BLOCK_SEED = None
        self.mod.ON_POLICY_RKL_DERIVED_SEED = self.mod.ON_POLICY_RKL_SEED

    def test_set_block_seed_updates_derived_seed(self):
        """Calling set_on_policy_rkl_block_seed must update the derived
        sampling seed; a stale seed defeats the rotation."""
        before = self.mod.ON_POLICY_RKL_DERIVED_SEED
        self.mod.set_on_policy_rkl_block_seed(8052008)
        after = self.mod.ON_POLICY_RKL_DERIVED_SEED
        self.assertNotEqual(
            before, after,
            "Derived seed unchanged after set_block_seed — rotation "
            "broken; rounds would still share the same sampling path.",
        )

    def test_derived_seed_is_deterministic_per_block_seed(self):
        """Every validator must compute the same derived seed for a
        given block_seed; otherwise validators disagree on rollouts."""
        seeds_a = []
        seeds_b = []
        for bs in (10000, 20000, 30000, 40000, 50000):
            self.mod._ON_POLICY_RKL_BLOCK_SEED = None
            self.mod.set_on_policy_rkl_block_seed(bs)
            seeds_a.append(self.mod.ON_POLICY_RKL_DERIVED_SEED)
            self.mod._ON_POLICY_RKL_BLOCK_SEED = None
            self.mod.set_on_policy_rkl_block_seed(bs)
            seeds_b.append(self.mod.ON_POLICY_RKL_DERIVED_SEED)
        self.assertEqual(seeds_a, seeds_b)

    def test_derived_seed_rotates_across_block_seeds(self):
        """Different block_seeds must produce different derived seeds.
        If they collided (which won't happen with XOR + 32-bit space)
        the rotation would silently degenerate."""
        seeds: set[int] = set()
        for bs in (1, 2, 100, 12345, 999999, 8052008, 8062008):
            self.mod._ON_POLICY_RKL_BLOCK_SEED = None
            self.mod.set_on_policy_rkl_block_seed(bs)
            seeds.add(self.mod.ON_POLICY_RKL_DERIVED_SEED)
        self.assertGreaterEqual(
            len(seeds), 6,
            f"Only {len(seeds)} distinct derived seeds across 7 block_seeds; "
            "expected near-perfect uniqueness from XOR rotation",
        )

    def test_derived_seed_xors_base(self):
        """The rotation is XOR(base, block_seed) — flipping any bit of
        the block_seed flips the same bit of the derived seed. This is
        the cheapest reversible scrambler we can audit by inspection."""
        base = int(self.mod.ON_POLICY_RKL_SEED)
        for bs in (1, 0xDEADBEEF, 0x12345678, 8052008):
            self.mod._ON_POLICY_RKL_BLOCK_SEED = None
            self.mod.set_on_policy_rkl_block_seed(bs)
            expected = (base ^ (bs & 0xFFFFFFFF)) & 0xFFFFFFFF
            self.assertEqual(
                self.mod.ON_POLICY_RKL_DERIVED_SEED, expected,
                f"Derived seed for block_seed={bs} was "
                f"{self.mod.ON_POLICY_RKL_DERIVED_SEED:#x}, expected "
                f"{expected:#x}",
            )

    def test_set_block_seed_idempotent_with_same_value(self):
        """Re-calling with the SAME block_seed must be a no-op
        (preserves the derived seed). Otherwise repeated calls within a
        round (mid-eval) could regenerate the seed and corrupt the
        already-running sampling stream."""
        self.mod.set_on_policy_rkl_block_seed(8052008)
        first = self.mod.ON_POLICY_RKL_DERIVED_SEED
        self.mod.set_on_policy_rkl_block_seed(8052008)
        self.mod.set_on_policy_rkl_block_seed(8052008)
        self.assertEqual(self.mod.ON_POLICY_RKL_DERIVED_SEED, first)

    def test_none_block_seed_leaves_derived_seed_unchanged(self):
        """A None block_seed must NOT zero-out the derived seed (would
        make the sampler fall back to seed 0 across the cluster)."""
        self.mod.set_on_policy_rkl_block_seed(8052008)
        first = self.mod.ON_POLICY_RKL_DERIVED_SEED
        self.mod.set_on_policy_rkl_block_seed(None)
        self.assertEqual(self.mod.ON_POLICY_RKL_DERIVED_SEED, first)

    def test_invalid_block_seed_leaves_derived_seed_unchanged(self):
        """Garbage block_seed values (string, dict) must degrade
        gracefully — keep the previous derived seed rather than
        crashing the entire eval pipeline."""
        self.mod.set_on_policy_rkl_block_seed(8052008)
        first = self.mod.ON_POLICY_RKL_DERIVED_SEED
        for bad in ("not-a-seed", object(), {"x": 1}):
            try:
                self.mod._ON_POLICY_RKL_BLOCK_SEED = None
                self.mod.set_on_policy_rkl_block_seed(bad)
            except Exception:
                pass
            self.assertEqual(
                self.mod.ON_POLICY_RKL_DERIVED_SEED, first,
                f"Bad block_seed {bad!r} corrupted the derived seed; "
                "the rollouts pipeline would silently desync.",
            )

    def test_prompt_rotation_couples_with_seed_rotation(self):
        """Both pool AND seed must update together — a half-update would
        leave one defense disabled. Verify pool and seed both change
        when set_block_seed is called with a fresh value."""
        # Establish baseline.
        self.mod.set_on_policy_rkl_block_seed(1)
        prompts_1 = list(self.mod.ON_POLICY_RKL_PROMPTS)
        seed_1 = self.mod.ON_POLICY_RKL_DERIVED_SEED
        # Rotate.
        self.mod._ON_POLICY_RKL_BLOCK_SEED = None
        self.mod.set_on_policy_rkl_block_seed(99999)
        prompts_2 = list(self.mod.ON_POLICY_RKL_PROMPTS)
        seed_2 = self.mod.ON_POLICY_RKL_DERIVED_SEED
        self.assertNotEqual(
            prompts_1, prompts_2,
            "Prompt pool unchanged across block_seeds — pool rotation "
            "broken",
        )
        self.assertNotEqual(
            seed_1, seed_2,
            "Derived seed unchanged across block_seeds — seed rotation "
            "broken",
        )


if __name__ == "__main__":
    unittest.main()
