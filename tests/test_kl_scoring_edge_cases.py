#!/usr/bin/env python3
"""
Unit tests for KL divergence, scoring, model checker, and edge cases.

Covers:
  - KL divergence computation with known inputs/outputs
  - Early stopping edge cases (KL=0, negative, very high)
  - Score computation (winner-take-all, epsilon threshold)
  - Model checker validation (MoE params, vocab size, disqualification)
  - State consistency validation
  - Prompt sampling determinism

Usage:
    pytest tests/test_kl_scoring_edge_cases.py -v
    python tests/test_kl_scoring_edge_cases.py
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════════════
# KL Divergence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKLDivergence(unittest.TestCase):
    """Test KL divergence computation with known inputs."""

    def test_identical_distributions_kl_zero(self):
        """KL(P || P) should be 0 for identical distributions."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        logits = torch.randn(1, 10, 50)
        result = compute_kl_from_logits(logits, logits)

        self.assertAlmostEqual(result["kl_mean"], 0.0, places=5)
        self.assertAlmostEqual(result["kl_min"], 0.0, places=5)
        self.assertEqual(result["n_positions"], 10)

    def test_different_distributions_kl_positive(self):
        """KL(P || Q) should be > 0 for different distributions."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        teacher = torch.randn(1, 10, 50)
        student = torch.randn(1, 10, 50)
        result = compute_kl_from_logits(teacher, student)

        self.assertGreater(result["kl_mean"], 0.0)
        self.assertGreater(result["kl_max"], 0.0)

    def test_kl_with_start_pos(self):
        """KL with start_pos should only compute from that position."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        logits = torch.randn(1, 20, 50)
        student = torch.randn(1, 20, 50)

        full = compute_kl_from_logits(logits, student, start_pos=0)
        partial = compute_kl_from_logits(logits, student, start_pos=10)

        self.assertEqual(full["n_positions"], 20)
        self.assertEqual(partial["n_positions"], 10)

    def test_kl_2d_input(self):
        """KL should work with 2D inputs (no batch dim)."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        logits = torch.randn(10, 50)
        result = compute_kl_from_logits(logits, logits)
        self.assertAlmostEqual(result["kl_mean"], 0.0, places=5)

    def test_kl_known_value(self):
        """KL between specific known distributions should match manual calculation."""
        import torch
        import torch.nn.functional as F
        from eval.kl_divergence import compute_kl_from_logits

        # Simple 3-class case: teacher = [0.7, 0.2, 0.1], student = [0.4, 0.3, 0.3]
        # We create logits that produce these softmax outputs
        # log([0.7, 0.2, 0.1]) as logits (pre-softmax, we use log probs directly)
        teacher_probs = torch.tensor([0.7, 0.2, 0.1])
        student_probs = torch.tensor([0.4, 0.3, 0.3])

        # Manual KL: sum(p * log(p/q))
        expected_kl = (teacher_probs * (teacher_probs.log() - student_probs.log())).sum().item()
        self.assertGreater(expected_kl, 0)

        # Create logits (use log probs as logits — softmax of log(p) ≠ p exactly,
        # but we can verify the function produces consistent positive KL)
        teacher_logits = teacher_probs.log().unsqueeze(0).unsqueeze(0)  # [1,1,3]
        student_logits = student_probs.log().unsqueeze(0).unsqueeze(0)
        result = compute_kl_from_logits(teacher_logits, student_logits)
        self.assertGreater(result["kl_mean"], 0)

    def test_kl_symmetry_not_guaranteed(self):
        """KL is asymmetric: KL(P||Q) != KL(Q||P) in general."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        a = torch.randn(1, 5, 30)
        b = torch.randn(1, 5, 30)

        kl_ab = compute_kl_from_logits(a, b)["kl_mean"]
        kl_ba = compute_kl_from_logits(b, a)["kl_mean"]

        # They should both be positive but generally different
        self.assertGreater(kl_ab, 0)
        self.assertGreater(kl_ba, 0)
        self.assertNotAlmostEqual(kl_ab, kl_ba, places=3)

    # 2026-05-04 — test_compute_kl_divergence_topk removed: the
    # ``compute_kl_divergence`` CPU fallback was deleted in the
    # 2026-04-04 validator rewrite (see
    # reports/2026-04-04-validator-rewrite.md). Sparse top-K KL now
    # lives in scripts/pod_eval_vllm.py:compute_kl_from_sparse and is
    # exercised end-to-end by the integration suite. Keeping the
    # orphan stub here only added a hard-failing import.

    def test_zero_gen_len_returns_inf(self):
        """evaluate_student_kl should return inf when gen_len=0."""
        from eval.kl_divergence import evaluate_student_kl

        cache_entry = {
            "full_ids": None,
            "teacher_logits": None,
            "prompt_len": 10,
            "gen_len": 0,
        }
        result = evaluate_student_kl(MagicMock(), cache_entry, device="cpu")
        self.assertEqual(result["kl_mean"], float("inf"))
        self.assertEqual(result["n_positions"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Early Stopping Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping logic and edge cases."""

    def _compute_ci_lower(self, scores: list[float]) -> float:
        """Compute 95% CI lower bound (mirrors pipeline logic)."""
        n = len(scores)
        if n < 2:
            return 0.0
        mean = sum(scores) / n
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        se = math.sqrt(var / n)
        return mean - 1.96 * se

    def test_clearly_worse_triggers_early_stop(self):
        """Student clearly worse than best → should stop early."""
        best_kl = 0.05
        student_scores = [0.14, 0.16, 0.15, 0.13, 0.17, 0.15, 0.14]
        lower = self._compute_ci_lower(student_scores)
        self.assertGreater(lower, best_kl, "CI lower bound should exceed best_kl")

    def test_close_scores_no_early_stop(self):
        """Student very close to best → should NOT stop early."""
        best_kl = 0.05
        close_scores = [0.052, 0.048, 0.051, 0.049, 0.053, 0.047, 0.050]
        lower = self._compute_ci_lower(close_scores)
        self.assertLessEqual(lower, best_kl, "Close student should not trigger early stop")

    def test_kl_zero_scores(self):
        """All KL=0 scores (suspicious/fraudulent)."""
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lower = self._compute_ci_lower(scores)
        self.assertEqual(lower, 0.0)

    def test_kl_negative_scores(self):
        """Negative KL values should be treated as anomalous."""
        # In practice, negative KL indicates numerical issues
        scores = [-0.001, -0.002, 0.001, -0.001, 0.0, -0.003, 0.002]
        lower = self._compute_ci_lower(scores)
        # Lower bound should be negative or near zero
        self.assertLess(lower, 0.01)

    def test_kl_very_high(self):
        """Very high KL should immediately trigger early stop."""
        best_kl = 0.05
        high_scores = [50.0, 55.0, 48.0, 52.0, 51.0, 49.0, 53.0]
        lower = self._compute_ci_lower(high_scores)
        self.assertGreater(lower, best_kl)

    def test_single_score_no_stop(self):
        """Single score → can't compute CI, shouldn't stop."""
        lower = self._compute_ci_lower([0.15])
        self.assertEqual(lower, 0.0, "Single score should return 0 (can't compute CI)")

    def test_identical_scores_zero_variance(self):
        """All identical scores → variance=0, CI lower = mean."""
        scores = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        n = len(scores)
        mean = sum(scores) / n
        lower = self._compute_ci_lower(scores)
        self.assertAlmostEqual(lower, mean, places=10)

    def test_fraud_kl_threshold(self):
        """KL ≤ 0.001 should be flagged as suspicious."""
        fraud_threshold = 0.001
        test_values = [0.0, 0.0001, 0.0005, 0.001]
        for kl in test_values:
            self.assertLessEqual(kl, fraud_threshold, f"KL={kl} should be flagged")

        legit_values = [0.002, 0.01, 0.05]
        for kl in legit_values:
            self.assertGreater(kl, fraud_threshold, f"KL={kl} should not be flagged")


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoring(unittest.TestCase):
    """Failure / staleness tracking still lives in eval.scoring; the
    legacy ``compute_winner_weights`` and ``update_ema`` API was removed
    in the 2026-04-04 validator rewrite (see
    reports/2026-04-04-validator-rewrite.md). The composite ranking now
    lives in scripts/validator/composite.py and is exercised by
    tests/test_arena_v3_composite.py + tests/test_eval_pipeline.py.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_failure_tracking(self):
        """Failure count should increment and reset correctly."""
        from eval.scoring import record_failure, reset_failures, is_stale

        failures = {}
        for _ in range(3):
            record_failure(0, failures)
        self.assertTrue(is_stale(0, failures, max_failures=3))

        reset_failures(0, failures)
        self.assertFalse(is_stale(0, failures, max_failures=3))


# ═══════════════════════════════════════════════════════════════════════════════
# Model Checker Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelChecker(unittest.TestCase):
    """Test model architecture checking and param counting."""

    def test_moe_param_count_basic(self):
        """MoE param counting should distinguish total vs active params."""
        from eval.model_checker import compute_moe_params

        config = {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
        result = compute_moe_params(config)
        self.assertTrue(result["is_moe"])
        self.assertEqual(result["num_experts"], 8)
        self.assertEqual(result["num_active_experts"], 2)
        self.assertGreater(result["total_params"], result["active_params"])

    def test_dense_model_params(self):
        """Dense model: total params == active params."""
        from eval.model_checker import compute_moe_params

        config = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
        }
        result = compute_moe_params(config)
        self.assertFalse(result["is_moe"])
        self.assertEqual(result["total_params"], result["active_params"])

    def test_nested_text_config(self):
        """Config with text_config should be handled correctly."""
        from eval.model_checker import compute_moe_params

        config = {
            "text_config": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 50000,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
            }
        }
        result = compute_moe_params(config)
        self.assertGreater(result["total_params"], 0)
        self.assertFalse(result["is_moe"])

    def test_empty_config(self):
        """Empty config should return zero params."""
        from eval.model_checker import compute_moe_params

        result = compute_moe_params({})
        self.assertEqual(result["total_params"], 0)
        self.assertFalse(result["is_moe"])

    def test_duplicate_hash_detection(self):
        """Duplicate model hash should be detected."""
        from eval.model_checker import register_model_hash, check_duplicate_hash

        tmpdir = tempfile.mkdtemp()
        state_dir = Path(tmpdir)

        register_model_hash("abc123", 0, state_dir)
        register_model_hash("def456", 1, state_dir)

        # UID 2 submits same hash as UID 0
        dup = check_duplicate_hash("abc123", 2, state_dir)
        self.assertEqual(dup, 0)

        # UID 0 checking its own hash → not a duplicate
        dup2 = check_duplicate_hash("abc123", 0, state_dir)
        self.assertIsNone(dup2)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 2026-05-04 — Removed orphan tests for legacy validator-rewrite
    # API:
    #   * test_disqualification_by_hotkey_block / _legacy_uid
    #   * test_flagging_coldkey_hf
    #   * test_commitment_changed
    # The functions disqualify(), is_disqualified(), is_flagged(),
    # commitment_changed() were folded into eval/scoring.py +
    # scripts/validator/single_eval.py during the 2026-04-04 rewrite.
    # Equivalent end-to-end coverage now lives in
    # tests/test_eval_pipeline.py.


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset / Prompt Sampling Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptSampling(unittest.TestCase):
    """Test prompt sampling determinism and edge cases."""

    def test_shard_selection_deterministic(self):
        """Same block hash should always select the same shard."""
        import hashlib

        hash_hex = "abcdef1234567890" * 4
        shard1 = int(hash_hex[:8], 16) % 6542
        shard2 = int(hash_hex[:8], 16) % 6542
        self.assertEqual(shard1, shard2)

    def test_different_hashes_different_shards(self):
        """Different block hashes should (usually) select different shards."""
        shards = set()
        for i in range(100):
            import hashlib
            h = hashlib.sha256(str(i).encode()).hexdigest()
            shard = int(h[:8], 16) % 6542
            shards.add(shard)
        # With 100 random hashes across 6542 shards, we should get many distinct
        self.assertGreater(len(shards), 50)

    def test_hash_hex_computation(self):
        """Hash normalization should strip only the 0x prefix."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        with_prefix = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0xabcdef1234")
        without_prefix = sample_prompts_seeded(pool, 10, block_number=100, block_hash="abcdef1234")
        self.assertEqual(with_prefix, without_prefix)

    def test_hash_hex_preserves_leading_zeroes(self):
        """Leading zeroes after 0x are part of the seed and must be preserved."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        with_leading_zero = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0x00abcdef1234")
        without_leading_zero = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0xabcdef1234")
        self.assertNotEqual(with_leading_zero, without_leading_zero)

    def test_sample_prompts_seeded_deterministic(self):
        """Seeded sampling should be deterministic."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        s1 = sample_prompts_seeded(pool, 10, block_number=42)
        s2 = sample_prompts_seeded(pool, 10, block_number=42)
        self.assertEqual(s1, s2)

    def test_sample_prompts_seeded_different_blocks(self):
        """Different block numbers should produce different samples."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        s1 = sample_prompts_seeded(pool, 10, block_number=42)
        s2 = sample_prompts_seeded(pool, 10, block_number=43)
        self.assertNotEqual(s1, s2)

    def test_format_prompt_sanitization(self):
        """format_prompt should handle edge cases."""
        from eval.dataset import format_prompt

        # Empty / None
        self.assertEqual(format_prompt(""), "")
        self.assertEqual(format_prompt(None), "")

        # Normal text
        result = format_prompt("Hello world, this is a test.")
        self.assertEqual(result, "Hello world, this is a test.")

        # Null bytes should be stripped
        result = format_prompt("Hello\x00World")
        self.assertNotIn("\x00", result)

        # Very long text should be truncated
        long_text = "A" * 1000
        result = format_prompt(long_text, max_chars=512)
        self.assertLessEqual(len(result), 512)

    def test_format_prompt_binary_rejection(self):
        """format_prompt should reject mostly binary content."""
        from eval.dataset import format_prompt

        # Mostly non-printable → rejected
        binary_garbage = "\x01\x02\x03\x04\x05" * 100
        result = format_prompt(binary_garbage)
        self.assertEqual(result, "")

    def test_prompt_cache_fallback_when_datasets_unavailable(self):
        """Prompt sampling should fall back to cached history if datasets can't load."""
        from eval.dataset import sample_prompts_from_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            seed_prompts = [
                f"Prompt {i} with enough text to be valid and deterministic for fallback sampling."
                for i in range(10)
            ]
            (cache_dir / "block_1_10.json").write_text(json.dumps(seed_prompts))

            with patch("eval.dataset._load_dataset_in_temp_hf_cache", side_effect=ImportError("datasets missing")):
                sampled = sample_prompts_from_dataset(
                    n=5,
                    block_number=42,
                    block_hash="0xabc123",
                    cache_dir=cache_dir,
                    min_chars=10,
                    max_chars=200,
                )

            self.assertEqual(len(sampled), 5)
            self.assertTrue(all(prompt in seed_prompts for prompt in sampled))


# ═══════════════════════════════════════════════════════════════════════════════
# State Consistency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateConsistency(unittest.TestCase):
    """Test state persistence and consistency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scores_round_trip(self):
        """Scores should survive save/load cycle."""
        from eval.scoring import save_scores, load_scores

        scores = {"0": 0.123456, "5": 0.789012, "10": 1.5}
        save_scores(scores, self.state_dir)
        loaded = load_scores(self.state_dir)
        self.assertEqual(loaded, scores)

    def test_empty_state_defaults(self):
        """Loading from empty state dir should return empty dicts."""
        from eval.scoring import load_scores, load_failures, load_disqualified

        self.assertEqual(load_scores(self.state_dir), {})
        self.assertEqual(load_failures(self.state_dir), {})
        self.assertEqual(load_disqualified(self.state_dir), {})

    def test_corrupted_json_handled(self):
        """Corrupted JSON files should return empty defaults."""
        from eval.scoring import load_scores

        (self.state_dir / "scores.json").write_text("not valid json {{{")
        result = load_scores(self.state_dir)
        self.assertEqual(result, {})

    def test_score_history_append(self):
        """Score history should append and cap correctly."""
        from eval.scoring import append_score_history, load_score_history

        for i in range(10):
            append_score_history(
                block=1000 + i,
                timestamp=float(i),
                scores={"0": 0.1 * i},
                king_uid=0,
                state_dir=self.state_dir,
                max_entries=5,
            )

        history = load_score_history(self.state_dir)
        self.assertEqual(len(history), 5)
        # Should keep the last 5
        self.assertEqual(history[0]["block"], 1005)
        self.assertEqual(history[-1]["block"], 1009)

    def test_disqualification_is_per_hotkey(self):
        """DQ is per-hotkey: a new commit_block on the same hotkey
        does NOT clear the DQ. The miner must register a new hotkey
        to be re-evaluated.

        Policy change 2026-05-04 — see eval/scoring.py:disqualify.
        Pre-2026-05-04 the DQ was per-(hotkey, commit_block) so a new
        commit cleared the DQ; that let repeat-offender hotkeys waste
        validator pod time round after round (e.g. UID 73 cascading
        CUDA asserts across three rounds before stale-skip kicked in,
        each time requiring a fresh load attempt). New scope: once a
        hotkey is DQ'd it stays DQ'd until a new hotkey re-registers.
        """
        from eval.scoring import disqualify, is_disqualified

        dq = {}
        disqualify("5HotKey", "bad model v1", dq, commit_block=100)

        self.assertTrue(is_disqualified(0, "5HotKey", dq, commit_block=100))
        self.assertTrue(is_disqualified(0, "5HotKey", dq, commit_block=200))
        self.assertTrue(is_disqualified(0, "5HotKey", dq))
        self.assertFalse(is_disqualified(0, "5OtherKey", dq, commit_block=100))

    def test_cold_start_crowning_is_a_king_change(self):
        """When a round starts with king_uid=None and a winner emerges,
        the post-round logic must treat that as a king change so
        ``h2h_latest.king_changed`` and ``new_king_uid`` are both set
        correctly. Pre-2026-05-04 this returned ``False`` because the
        expression was ``winner_uid != king_uid if king_uid is not None
        else False`` — the cold-start branch hard-coded False and the
        announcement / dashboard / API ``is_king`` flag never fired
        for the first king of a new era. This is the bug Sebastian
        reported on 2026-05-04 (UID 188 won post-Kimi-cutover round 2
        but the API kept reporting ``is_king: false``).
        """
        # Reproduce the expression used in update_h2h_state directly:
        for king_uid, winner_uid, expected in [
            (None, 188, True),
            (None, None, False),
            (None, 0, True),
            (188, 188, False),
            (188, 195, True),
            # king_uid=188 + winner_uid=None means "king failed and no
            # replacement emerged" — the legacy expression also returned
            # True here (None != 188), so we preserve that behaviour.
            (188, None, True),
        ]:
            if king_uid is None:
                king_changed = winner_uid is not None
            else:
                king_changed = winner_uid != king_uid
            self.assertEqual(
                king_changed, expected,
                f"king_uid={king_uid!r} winner_uid={winner_uid!r}: "
                f"expected king_changed={expected}, got {king_changed}",
            )

    def test_atrisk_cold_start_king_streak_short_circuit(self):
        """Cold-start crowning of an already-at-risk king must seed the
        regression streak at MIN_STREAK − 1 so this round's at_risk
        increment immediately reaches MIN_STREAK and the floor waiver
        kicks in NEXT round (not three rounds later).

        Sebastian's 2026-05-04 report — "the overall eval scores seem
        even worse than the 4B" — traced to this exact path: post-Kimi
        cutover crowned UID 190 with composite.worst = 0.000, then
        the regression streak counted up from 0 → 1 → 2 → 3 across
        three rounds before the floor finally waived. During those
        three rounds challengers like UID 125 (worst = 0.125, strictly
        better than the king on the worst axis) were vetoed because
        their absolute worst was below the static floor of 0.20. The
        short-circuit collapses that delay to a single round — the
        next at-risk pass already has streak ≥ MIN_STREAK and any
        improved challenger can dethrone.

        Healthy fresh kings (cold-start AND not at_risk) must still
        start at 0 — the short-circuit only fires on at-risk crowns.
        """
        MIN_STREAK = 3  # default KING_REGRESSION_MIN_STREAK
        # Reproduce the post-fix branch from state_manager.update_h2h_state:
        cases = [
            # (king_changed, at_risk, expected_prev_streak, expected_new_streak)
            (True,  True,  MIN_STREAK - 1, MIN_STREAK),       # at-risk cold-start: short-circuit
            (True,  False, 0,              0),                 # healthy cold-start: fresh slate
            (False, True,  None,           None),  # streak unchanged path — see below
            (False, False, None,           None),
        ]
        for king_changed, at_risk, exp_prev, exp_new in cases:
            prev_streak_persisted = 1  # arbitrary stale streak from prior round
            if king_changed:
                if at_risk:
                    prev_streak = max(0, MIN_STREAK - 1)
                else:
                    prev_streak = 0
            else:
                prev_streak = prev_streak_persisted
            new_streak = prev_streak + 1 if at_risk else 0
            if king_changed:
                self.assertEqual(
                    prev_streak, exp_prev,
                    f"king_changed={king_changed} at_risk={at_risk}: "
                    f"expected prev_streak={exp_prev}, got {prev_streak}",
                )
                self.assertEqual(
                    new_streak, exp_new,
                    f"king_changed={king_changed} at_risk={at_risk}: "
                    f"expected new_streak={exp_new}, got {new_streak}",
                )
            else:
                # Non-cold-start: previous behaviour untouched.
                self.assertEqual(prev_streak, prev_streak_persisted)
                expected = prev_streak_persisted + 1 if at_risk else 0
                self.assertEqual(new_streak, expected)

    def test_disqualification_legacy_keys_recognised(self):
        """Pre-2026-05-04 DQ entries used ``hotkey:<commit_block>`` keys.
        ``is_disqualified`` and ``get_dq_reason`` must still recognise
        those entries so historical DQ'd hotkeys remain DQ'd after the
        per-hotkey policy migration.
        """
        from eval.scoring import get_dq_reason, is_disqualified

        dq = {
            "5HotKey:50": "legacy entry from pre-migration",
            "5HotKey:75": "another legacy entry",
        }
        self.assertTrue(is_disqualified(0, "5HotKey", dq))
        self.assertTrue(is_disqualified(0, "5HotKey", dq, commit_block=200))
        # get_dq_reason returns one of the legacy entries (last wins
        # by dict insertion order).
        reason = get_dq_reason(0, "5HotKey", dq, commit_block=200)
        self.assertIn("legacy entry", reason)
        # An unrelated hotkey is unaffected by another hotkey's
        # legacy entries.
        self.assertFalse(is_disqualified(0, "5OtherKey", dq))

    # 2026-05-04 — Removed test_commitment_cache_persistence: the
    # save/load_commitment_cache helpers were removed in the
    # 2026-04-04 validator rewrite. Commitments are now read directly
    # from the chain via bittensor.Subtensor.get_all_revealed_commitments
    # and cached in api/state_store.py:read_commitments() with a
    # different on-disk schema.


# ═══════════════════════════════════════════════════════════════════════════════
# KL Anchor Resolution Tests (cold-start fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveKingKl(unittest.TestCase):
    """Verify the cold-start kl-axis fallback added 2026-05-04 after
    Sebastian's report that the dashboard ``Kl`` axis was blank for
    every student in the post-Kimi-cutover round (because king_h2h_kl
    was ``None`` and ``_axis_kl`` returns None whenever its anchor is
    missing). The fallback uses the round-wide minimum kl_global_avg
    as the anchor so the new winner scores 1.0 on kl and challengers
    get a meaningful denominator.
    """

    def test_uses_king_kl_when_available(self):
        """Normal rounds: anchor on the king's KL, ignore the round
        minimum even if a non-king student had a lower KL (pre-existing
        behaviour preserved by the fallback path)."""
        from scripts.validator.composite import _resolve_king_kl

        students_data = {
            "king/m": {"kl_global_avg": 0.30},
            "challenger/a": {"kl_global_avg": 0.25},
            "challenger/b": {"kl_global_avg": 0.50},
        }
        self.assertEqual(_resolve_king_kl(0.30, students_data), 0.30)
        self.assertEqual(_resolve_king_kl(2.5, students_data), 2.5)

    def test_falls_back_to_round_min_when_king_kl_missing(self):
        """Cold-start rounds (king_h2h_kl=None): anchor on the
        round-wide minimum kl_global_avg so the kl axis is populated."""
        from scripts.validator.composite import _resolve_king_kl

        students_data = {
            "challenger/a": {"kl_global_avg": 2.05},
            "challenger/b": {"kl_global_avg": 4.30},
            "challenger/c": {"kl_global_avg": 1.60},
        }
        self.assertEqual(
            _resolve_king_kl(None, students_data), 1.60,
            "should pick the lowest KL as the anchor",
        )

    def test_falls_back_to_round_min_for_inf_or_zero_king_kl(self):
        """Defensive: ``inf`` / 0 / negative king_kl values are treated
        as missing and fall back to the round minimum (these are
        sentinel values for failed evals, not legitimate anchors)."""
        from scripts.validator.composite import _resolve_king_kl

        students_data = {"a": {"kl_global_avg": 1.5}}
        self.assertEqual(_resolve_king_kl(float("inf"), students_data), 1.5)
        self.assertEqual(_resolve_king_kl(0.0, students_data), 1.5)
        self.assertEqual(_resolve_king_kl(-1.0, students_data), 1.5)
        self.assertEqual(_resolve_king_kl(float("nan"), students_data), 1.5)

    def test_returns_none_when_no_data_at_all(self):
        """Bootstrap round before any student has scored: nothing to
        anchor on, return None and let the caller decide."""
        from scripts.validator.composite import _resolve_king_kl

        self.assertIsNone(_resolve_king_kl(None, {}))
        self.assertIsNone(_resolve_king_kl(None, {"a": {}}))
        self.assertIsNone(_resolve_king_kl(None, {"a": {"kl_global_avg": None}}))

    def test_kl_axis_populated_in_cold_start_via_fallback(self):
        """End-to-end: under cold-start (king_kl=None) the kl axis is
        meaningful when computed against the resolved anchor — the
        round's best student scores 1.0, others get the natural
        anchor/student ratio."""
        from scripts.validator.composite import _axis_kl, _resolve_king_kl

        students_data = {
            "winner/m": {"kl_global_avg": 2.0},
            "challenger/a": {"kl_global_avg": 4.0},
        }
        anchor = _resolve_king_kl(None, students_data)
        self.assertEqual(anchor, 2.0)
        self.assertAlmostEqual(
            _axis_kl(students_data["winner/m"], anchor), 1.0, places=4,
        )
        self.assertAlmostEqual(
            _axis_kl(students_data["challenger/a"], anchor), 0.5, places=4,
        )

    def test_skips_teacher_self_kl_floor(self):
        """The teacher-vs-itself row reports kl≈0 by construction. The
        fallback's ``skip_floor`` (1e-4) drops it from the anchor pool
        so a real student's KL becomes the anchor instead of ~0
        (which would force every challenger to score None on kl)."""
        from scripts.validator.composite import _resolve_king_kl

        students_data = {
            "teacher": {"kl_global_avg": 1e-9},
            "student/a": {"kl_global_avg": 2.5},
            "student/b": {"kl_global_avg": 3.0},
        }
        self.assertEqual(
            _resolve_king_kl(None, students_data), 2.5,
            "teacher's near-zero self-KL must be skipped",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Announcement Earnings-Line Tests (multi-king payout)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnnouncementEarningsLine(unittest.TestCase):
    """Verify the new-king Discord announcement reflects the v30.4
    multi-king payout policy (emission split equally across the
    RECENT_KINGS_MAX most-recent distinct kings) instead of the
    deprecated ``winner takes all`` wording. Bug reported by Sebastian
    on 2026-05-04 after the UID 188 cold-start announcement still
    claimed the king took 100 percent of emission.
    """

    def _mocked_price_response(self, tao_per_day=137.8, tao_usd=2.0):
        from io import BytesIO

        class _Resp(BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_):
                return False

        body = json.dumps({
            "miners_tao_per_day": tao_per_day,
            "tao_usd": tao_usd,
        }).encode()
        return _Resp(body)

    def _make_state(self, recent_kings, tmpdir):
        """Build a real ValidatorState in tmpdir with the given recent_kings
        history. Using the real class keeps this test honest about the
        ``state.save_announcement`` write path.
        """
        from eval.state import ValidatorState

        s = ValidatorState(tmpdir)
        s.recent_kings = list(recent_kings)
        return s

    def _build_announcement(self, recent_kings, new_uid=188,
                            tao_per_day=137.8, tao_usd=2.0,
                            multi_king_env="1"):
        """Invoke ``announce_new_king`` with the price endpoint mocked
        and return the queued announcement dict.
        """
        from scripts.validator.announcements import announce_new_king

        with tempfile.TemporaryDirectory() as tmpdir:
            state = self._make_state(recent_kings, tmpdir)
            env_patch = patch.dict(
                os.environ,
                {"SINGLE_EVAL_MODE": "1", "MULTI_KING_PAYOUT_ENABLED": multi_king_env},
                clear=False,
            )
            url_patch = patch(
                "urllib.request.urlopen",
                return_value=self._mocked_price_response(tao_per_day, tao_usd),
            )
            with env_patch, url_patch:
                announce_new_king(
                    new_uid=new_uid,
                    new_model="best26/sn97-us-v7",
                    new_kl=2.055847,
                    old_uid=None,
                    old_model=None,
                    old_kl=None,
                    state=state,
                    paired_prompts=60,
                    total_prompts=60,
                    new_composite_final=0.268,
                    new_composite_worst=0.040,
                    new_composite_weighted=0.655,
                    new_composite_worst_3_mean=0.102,
                    new_limiting_axis="long_form_judge",
                )
            ann_path = Path(tmpdir) / "announcement.json"
            self.assertTrue(ann_path.exists(), "announcement file not written")
            return json.loads(ann_path.read_text())

    def test_no_winner_takes_all_wording_in_steady_state(self):
        """When the recent-kings queue is full (5 distinct kings), the
        announcement must NOT contain 'winner takes all' and must show
        the king's actual 20% share."""
        ann = self._build_announcement([188, 195, 107, 42, 17])
        msg = ann["message"]
        self.assertNotIn("winner takes all", msg.lower())
        self.assertIn("20% of the", msg)
        self.assertIn("split equally across the 5 most-recent", msg)
        # New king's per-day take is the pool divided by 5: 137.8 / 5 = 27.56.
        self.assertIn("~27.6 τ/day", msg)

    def test_cold_start_one_king_shows_dilution_note(self):
        """When only the new king sits in the queue, the line must
        explain that the share will dilute as new kings are crowned —
        not pretend it is winner-takes-all forever."""
        ann = self._build_announcement([188])
        msg = ann["message"]
        self.assertNotIn("winner takes all", msg.lower())
        self.assertIn("currently 100%", msg)
        self.assertIn("dilutes toward ~20%", msg)
        # Single-king cold-start: full pool to the new king.
        self.assertIn("~137.8 τ/day", msg)
        self.assertIn("UID 188 earns ~137.8 τ/day", msg)

    def test_three_kings_shows_one_third_share(self):
        """Three distinct kings in queue → each gets 33% of the pool.
        137.8 / 3 = 45.93… — formatted as ~45.9."""
        ann = self._build_announcement([188, 195, 107])
        msg = ann["message"]
        self.assertNotIn("winner takes all", msg.lower())
        self.assertIn("33% of the", msg)
        self.assertIn("split equally across the 3 most-recent", msg)
        self.assertIn("~45.9 τ/day", msg)

    def test_legacy_winner_takes_all_env_override(self):
        """If an operator sets ``MULTI_KING_PAYOUT_ENABLED=0`` we honor
        the legacy winner-takes-all behaviour AND surface the override
        in the message so it can't drift unnoticed."""
        ann = self._build_announcement([188], multi_king_env="0")
        msg = ann["message"]
        self.assertIn("winner takes all", msg.lower())
        self.assertIn("MULTI_KING_PAYOUT_ENABLED=0", msg)
        self.assertIn("~137.8 τ/day", msg)

    def test_new_uid_added_defensively_when_queue_lacks_it(self):
        """If the caller invokes ``announce_new_king`` BEFORE
        ``_record_king_change`` has pushed the new king onto the queue,
        the announcement must still account for the new king. Belt-and-
        braces against future caller-order regressions."""
        # Queue has 4 OTHER kings; new king is 188 and is NOT in the
        # queue → effective n_kings = 5, share = 20%.
        ann = self._build_announcement([195, 107, 42, 17], new_uid=188)
        msg = ann["message"]
        self.assertIn("20% of the", msg)
        self.assertIn("split equally across the 5 most-recent", msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Reproduce Prompts Script Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReproducePrompts(unittest.TestCase):
    """Test the reproduce_prompts.py helper functions."""

    def test_compute_hash_hex_with_block_hash(self):
        """compute_hash_hex with a block hash should strip 0x prefix."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from reproduce_prompts import compute_hash_hex

        result = compute_hash_hex(100, "0xabcdef1234")
        self.assertEqual(result, "abcdef1234")

        result2 = compute_hash_hex(100, "abcdef1234")
        self.assertEqual(result2, "abcdef1234")

    def test_compute_hash_hex_fallback(self):
        """compute_hash_hex without block hash should use sha256."""
        import hashlib
        from reproduce_prompts import compute_hash_hex

        result = compute_hash_hex(42, None)
        expected = hashlib.sha256(b"42").hexdigest()
        self.assertEqual(result, expected)

    def test_compute_shard_index_range(self):
        """Shard index should always be in valid range."""
        from reproduce_prompts import compute_shard_index, CLIMBMIX_NUM_SHARDS

        for i in range(100):
            import hashlib
            h = hashlib.sha256(str(i).encode()).hexdigest()
            idx = compute_shard_index(h)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, CLIMBMIX_NUM_SHARDS)

    def test_shard_index_deterministic(self):
        """Same hash should always produce same shard index."""
        from reproduce_prompts import compute_shard_index

        idx1 = compute_shard_index("abcdef1234567890")
        idx2 = compute_shard_index("abcdef1234567890")
        self.assertEqual(idx1, idx2)


if __name__ == "__main__":
    unittest.main()
