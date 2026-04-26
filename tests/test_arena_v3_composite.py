#!/usr/bin/env python3
"""Unit tests for Arena v3 composite scoring (Session 2 + Session 3 axes
+ Pareto dominance).

Covers:
  * Session 2 bench axes (math/code/reasoning/knowledge/ifeval) promoted
    to production ranking.
  * Session 3 axes (aime/mbpp/tool_use/self_consistency/arc/truthful/
    long_context/procedural) — live by default, still overrideable by env.
  * JUDGE_AXIS_IN_COMPOSITE / BENCH_AXES_IN_COMPOSITE / ARENA_V3_AXES_IN_COMPOSITE
    default values (v2 prod, v3 prod).
  * Pareto majority dominance: wins/losses/ties, margin, insufficient-axes
    fail-open, and the soft-Pareto decision (majority win AND net wins ≥ 0).
  * Teacher sanity gate correctly includes promoted v2/v3 axes.

Usage:
    pytest tests/test_arena_v3_composite.py -v
    python tests/test_arena_v3_composite.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _make_student(
    *,
    kl=0.3,
    rkl=0.1,
    cap_frac=0.8,
    teacher_cap=0.9,
    length_penalty=0.9,
    think_passed=True,
    judge_norm=0.7,
    judge_n_valid=12,
    bench: dict | None = None,
) -> dict:
    """Fabricate a pod_eval_vllm student payload for axis tests."""
    out: dict = {
        "kl_global_avg": kl,
        "on_policy_rkl": {"mean_rkl": rkl},
        "capability": {
            "pass_frac": cap_frac,
            "teacher_pass_frac": teacher_cap,
        },
        "length_axis": {"penalty": length_penalty, "ratio": 1.1},
        "think_probe": {
            "prompts_tested": 5,
            "prompts_terminated": 5 if think_passed else 2,
            "prompts_degenerate": 0 if think_passed else 2,
            "self_bleu_across_prompts": 0.3,
            "teacher_self_bleu": 0.3,
            "pass": think_passed,
        },
        "judge_probe": {
            "normalized": judge_norm,
            "n_valid": judge_n_valid,
            "n": 16,
        },
    }
    if bench:
        for axis_name, pass_frac in bench.items():
            n = 8 if axis_name not in ("code_bench", "aime_bench", "mbpp_bench",
                                      "tool_use_bench", "self_consistency_bench") else 4
            out[axis_name] = {
                "n": n,
                "correct": int(round(pass_frac * n)),
                "pass_frac": pass_frac,
                "items": [],
            }
    return out


class TestSession2Promoted(unittest.TestCase):
    """Session 2 bench axes are production (BENCH_AXES_IN_COMPOSITE=1)."""

    def setUp(self):
        import scripts.validator.composite as _c
        self._saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        self._saved_judge = _c.JUDGE_AXIS_IN_COMPOSITE
        self._saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        self._saved_rd = _c.REASONING_DENSITY_IN_COMPOSITE
        self._saved_chat = _c.CHAT_TURNS_AXIS_IN_COMPOSITE
        _c.BENCH_AXES_IN_COMPOSITE = True
        _c.JUDGE_AXIS_IN_COMPOSITE = True
        _c.ARENA_V3_AXES_IN_COMPOSITE = False
        _c.REASONING_DENSITY_IN_COMPOSITE = False
        _c.CHAT_TURNS_AXIS_IN_COMPOSITE = False

    def tearDown(self):
        import scripts.validator.composite as _c
        _c.BENCH_AXES_IN_COMPOSITE = self._saved_bench
        _c.JUDGE_AXIS_IN_COMPOSITE = self._saved_judge
        _c.ARENA_V3_AXES_IN_COMPOSITE = self._saved_v3
        _c.REASONING_DENSITY_IN_COMPOSITE = self._saved_rd
        _c.CHAT_TURNS_AXIS_IN_COMPOSITE = self._saved_chat

    def test_bench_axes_lower_worst(self):
        """A student passing KL but failing math_bench should have low worst."""
        from scripts.validator.composite import compute_composite
        student = _make_student(bench={
            "math_bench": 0.10,     # fails
            "code_bench": 0.90,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.85,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLess(comp["worst"], 0.15,
            "worst should be dragged down by math_bench=0.10")
        self.assertTrue(comp["bench_in_composite"])
        self.assertFalse(comp["arena_v3_in_composite"])

    def test_judge_in_composite(self):
        """judge_probe lowers worst when promoted and below other axes."""
        from scripts.validator.composite import compute_composite
        student = _make_student(judge_norm=0.15, bench={
            "math_bench": 0.85,
            "code_bench": 0.75,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.80,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLess(comp["worst"], 0.2)
        self.assertTrue(comp["judge_in_composite"])

    def test_v3_not_in_composite(self):
        """v3 axes are shown in axes dict but excluded from worst/weighted."""
        from scripts.validator.composite import compute_composite
        student = _make_student(bench={
            "math_bench": 0.85,
            "code_bench": 0.75,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.80,
            "aime_bench": 0.01,       # catastrophically bad
            "tool_use_bench": 0.05,
            "self_consistency_bench": 0.05,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertGreater(comp["worst"], 0.20,
            "v3 shadow axes must NOT pull down worst")
        self.assertIn("aime_bench", comp["axes"])
        self.assertEqual(comp["axes"]["aime_bench"], 0.01)


class TestSession3Production(unittest.TestCase):
    """Session 3 axes enter composite by default but can still be disabled."""

    def test_v3_gate_promoted(self):
        import scripts.validator.composite as _c
        saved = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.85,
                "code_bench": 0.75,
                "reasoning_bench": 0.80,
                "knowledge_bench": 0.70,
                "ifeval_bench": 0.80,
                "aime_bench": 0.05,  # bad — now should pull worst down
                "mbpp_bench": 0.6,
                "tool_use_bench": 0.3,
                "self_consistency_bench": 0.7,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(comp["worst"], 0.06,
                "aime=0.05 must now be the worst axis when v3 is promoted")
            self.assertTrue(comp["arena_v3_in_composite"])
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench

    def test_v3_axes_populated_when_data_present(self):
        from scripts.validator.composite import compute_axes
        student = _make_student(bench={
            "aime_bench": 0.25,
            "mbpp_bench": 0.55,
            "tool_use_bench": 0.4,
            "self_consistency_bench": 0.6,
            "arc_bench": 0.72,
        })
        axes = compute_axes(student, king_kl=0.3, king_rkl=0.1)
        self.assertEqual(axes["aime_bench"], 0.25)
        self.assertEqual(axes["mbpp_bench"], 0.55)
        self.assertEqual(axes["tool_use_bench"], 0.4)
        self.assertEqual(axes["self_consistency_bench"], 0.6)
        self.assertEqual(axes["arc_bench"], 0.72)

    def test_arc_bench_is_v3_live_by_default(self):
        """arc_bench is in ARENA_V3_AXIS_WEIGHTS and gates worst by default."""
        import scripts.validator.composite as _c
        self.assertIn("arc_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("arc_bench", _c.BENCH_MIN_VALID)
        self.assertTrue(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "arc_bench": 0.05,
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(comp["worst"], 0.06,
                             "arc_bench=0.05 must pull worst down in live mode")
        self.assertEqual(comp["axes"]["arc_bench"], 0.05)

    def test_truthful_bench_is_v3_live_by_default(self):
        """truthful_bench (Session 3.4) follows live v3 semantics."""
        import scripts.validator.composite as _c
        self.assertIn("truthful_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("truthful_bench", _c.BENCH_MIN_VALID)
        self.assertIn("truthful_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        self.assertTrue(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "truthful_bench": 0.05,
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(comp["worst"], 0.06)
        self.assertEqual(comp["axes"]["truthful_bench"], 0.05)

    def test_truthful_bench_gates_worst_when_promoted(self):
        """truthful_bench is in the worst-axis rule when Arena v3 is promoted."""
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
                "knowledge_bench": 0.8, "ifeval_bench": 0.8,
                "aime_bench": 0.6, "mbpp_bench": 0.6,
                "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
                "arc_bench": 0.6,
                "truthful_bench": 0.1,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(
                comp["worst"], 0.11,
                "truthful_bench=0.1 must now pull worst down",
            )
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench

    def test_long_context_bench_is_v3_live_by_default(self):
        """long_context_bench (Session 3.5) follows live v3 semantics."""
        import scripts.validator.composite as _c
        self.assertIn("long_context_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("long_context_bench", _c.BENCH_MIN_VALID)
        self.assertIn("long_context_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        self.assertTrue(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "long_context_bench": 0.05,
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(comp["worst"], 0.06)
        self.assertEqual(comp["axes"]["long_context_bench"], 0.05)

    def test_long_context_bench_gates_worst_when_promoted(self):
        """long_context_bench joins worst-axis rule when Arena v3 is promoted."""
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
                "knowledge_bench": 0.8, "ifeval_bench": 0.8,
                "aime_bench": 0.6, "mbpp_bench": 0.6,
                "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
                "arc_bench": 0.6, "truthful_bench": 0.6,
                "long_context_bench": 0.1,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(
                comp["worst"], 0.11,
                "long_context_bench=0.1 must now pull worst down",
            )
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench

    def test_procedural_bench_is_v3_live(self):
        """procedural_bench is a fresh block-seeded axis and gates worst."""
        import scripts.validator.composite as _c
        self.assertIn("procedural_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("procedural_bench", _c.BENCH_MIN_VALID)
        self.assertIn("procedural_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "aime_bench": 0.6, "mbpp_bench": 0.6,
            "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
            "arc_bench": 0.6, "truthful_bench": 0.6,
            "long_context_bench": 0.6, "procedural_bench": 0.05,
            "robustness_bench": 0.6, "noise_resistance_bench": 0.6,
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(comp["worst"], 0.06)
        self.assertEqual(comp["axes"]["procedural_bench"], 0.05)

    def test_robustness_bench_is_v3_live(self):
        """robustness_bench (Session 3.7) gates the composite worst.

        Session 3.7 adds a math-pool reuse axis that asks each item under
        K block-rotated paraphrase wrappers. A model that overfits the
        canonical math wording has high math_bench but low
        robustness_bench → composite worst is dragged down by the
        robustness axis.
        """
        import scripts.validator.composite as _c
        self.assertIn("robustness_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("robustness_bench", _c.BENCH_MIN_VALID)
        student = _make_student(bench={
            "math_bench": 0.95, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "aime_bench": 0.6, "mbpp_bench": 0.6,
            "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
            "arc_bench": 0.6, "truthful_bench": 0.6,
            "long_context_bench": 0.6, "procedural_bench": 0.6,
            "robustness_bench": 0.10,  # canonical-only memorizer
            "noise_resistance_bench": 0.6,
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(
            comp["worst"], 0.11,
            "robustness_bench=0.10 must drag worst below 0.11 — a "
            "miner who only memorizes canonical wording cannot win",
        )
        self.assertEqual(comp["axes"]["robustness_bench"], 0.10)

    def test_noise_resistance_bench_is_v3_live(self):
        """noise_resistance_bench (Session 3.7) gates the composite worst.

        Sibling axis to robustness_bench: same pool (alias of math),
        independent stream offset, but the perturbations are
        adversarial input noise (typos, case jitter, distractor chatter)
        rather than semantic paraphrase. A model that breaks under
        light real-world chat noise has its composite worst dragged
        down here even if math_bench scores high.
        """
        import scripts.validator.composite as _c
        self.assertIn("noise_resistance_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("noise_resistance_bench", _c.BENCH_MIN_VALID)
        self.assertIn("noise_resistance_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        student = _make_student(bench={
            "math_bench": 0.95, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "aime_bench": 0.6, "mbpp_bench": 0.6,
            "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
            "arc_bench": 0.6, "truthful_bench": 0.6,
            "long_context_bench": 0.6, "procedural_bench": 0.6,
            "robustness_bench": 0.6,
            "noise_resistance_bench": 0.05,  # brittle-to-noise distillation
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLessEqual(
            comp["worst"], 0.06,
            "noise_resistance_bench=0.05 must drag worst below 0.06 — a "
            "model that fails on typos/distractors cannot win",
        )
        self.assertEqual(comp["axes"]["noise_resistance_bench"], 0.05)


class TestParetoDominance(unittest.TestCase):
    """Soft Pareto dominance semantics."""

    def test_pareto_wins_clear(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.8, "length": 0.9, "degeneracy": 0.9,
            "on_policy_rkl": 0.85, "judge_probe": 0.8,
            "math_bench": 0.7, "code_bench": 0.6, "reasoning_bench": 0.7,
            "knowledge_bench": 0.6, "ifeval_bench": 0.7,
        }
        king = {k: v - 0.10 for k, v in challenger.items()}
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertTrue(out["pareto_wins"])
        self.assertGreaterEqual(out["n_wins"], 6)
        self.assertEqual(out["n_losses"], 0)

    def test_pareto_loses_on_losses(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.4, "length": 0.3, "degeneracy": 0.4,
            "on_policy_rkl": 0.85, "judge_probe": 0.4,
            "math_bench": 0.7, "code_bench": 0.3, "reasoning_bench": 0.4,
            "knowledge_bench": 0.4, "ifeval_bench": 0.4,
        }
        king = {
            "kl": 0.85, "capability": 0.6, "length": 0.7, "degeneracy": 0.7,
            "on_policy_rkl": 0.80, "judge_probe": 0.6,
            "math_bench": 0.5, "code_bench": 0.5, "reasoning_bench": 0.6,
            "knowledge_bench": 0.6, "ifeval_bench": 0.6,
        }
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertFalse(out["pareto_wins"])
        self.assertIn("n_losses", out)

    def test_pareto_insufficient_axes_fails_open(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {"kl": 0.9, "capability": 0.8}
        king = {"kl": 0.85, "capability": 0.7}
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertFalse(out["pareto_wins"])
        self.assertTrue(out["reason"].startswith("insufficient"))

    def test_pareto_ties_within_margin(self):
        from scripts.validator.composite import compute_pareto_dominance
        # All axes within margin → all ties → no pareto wins, no losses.
        c = {f"a{i}": 0.5 for i in range(6)}
        k = {f"a{i}": 0.505 for i in range(6)}
        # Inject the expected axis names via monkey-patch so the function
        # actually considers them — compute_pareto_dominance iterates
        # known axis names, so use real ones:
        challenger = {
            "kl": 0.50, "capability": 0.50, "length": 0.50, "degeneracy": 0.50,
            "on_policy_rkl": 0.50, "judge_probe": 0.50,
        }
        king = {k: v + 0.005 for k, v in challenger.items()}
        out = compute_pareto_dominance(challenger, king, margin=0.02)
        self.assertEqual(out["n_wins"], 0)
        self.assertEqual(out["n_losses"], 0)
        self.assertEqual(out["n_ties"], 6)
        self.assertFalse(out["pareto_wins"])

    def test_pareto_includes_shadow_when_enabled(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.8, "length": 0.9, "degeneracy": 0.9,
            "on_policy_rkl": 0.85, "judge_probe": 0.8,
            "aime_bench": 0.6, "mbpp_bench": 0.7, "tool_use_bench": 0.6,
            "self_consistency_bench": 0.7,
        }
        king = {k: v - 0.10 for k, v in challenger.items()}
        out_with_shadow = compute_pareto_dominance(
            challenger, king, margin=0.02, include_shadow=True,
        )
        out_no_shadow = compute_pareto_dominance(
            challenger, king, margin=0.02, include_shadow=False,
        )
        self.assertGreater(out_with_shadow["comparable"], out_no_shadow["comparable"])


class TestBenchExtractor(unittest.TestCase):
    """Per-axis extractors correctly respect BENCH_MIN_VALID."""

    def test_below_min_valid_drops(self):
        from scripts.validator.composite import _axis_bench_pass_frac
        student = {"math_bench": {"n": 2, "correct": 1, "pass_frac": 0.5}}
        self.assertIsNone(_axis_bench_pass_frac(student, "math_bench"))

    def test_errored_bench_drops(self):
        from scripts.validator.composite import _axis_bench_pass_frac
        student = {"math_bench": {
            "n": 8, "correct": 4, "pass_frac": 0.5, "error": "boom",
        }}
        self.assertIsNone(_axis_bench_pass_frac(student, "math_bench"))

    def test_v3_uses_smaller_floor(self):
        from scripts.validator.composite import _axis_bench_pass_frac, BENCH_MIN_VALID
        self.assertLess(BENCH_MIN_VALID["aime_bench"], BENCH_MIN_VALID["math_bench"])
        student = {"aime_bench": {"n": 3, "correct": 0, "pass_frac": 0.0}}
        # aime floor is 3, so n=3 is enough.
        self.assertEqual(_axis_bench_pass_frac(student, "aime_bench"), 0.0)


class TestTeacherSanityGate(unittest.TestCase):
    """Teacher sanity gate includes v2 promoted + v3 axes (when promoted)."""

    def test_teacher_broken_math_axis_dropped(self):
        import scripts.validator.composite as _c
        saved = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.BENCH_AXES_IN_COMPOSITE = True
            teacher_row = _make_student(bench={
                "math_bench": 0.10,    # teacher fails — axis miscalibrated
                "code_bench": 0.90,
                "reasoning_bench": 0.90,
                "knowledge_bench": 0.90,
                "ifeval_bench": 0.90,
            })
            broken = _c.resolve_teacher_broken_axes(
                teacher_row, king_kl=0.1, king_rkl=0.05,
            )
            self.assertIn("math_bench", broken,
                "teacher scoring 0.10 on math should mark the axis broken")
        finally:
            _c.BENCH_AXES_IN_COMPOSITE = saved

    def test_teacher_v3_axis_not_checked_when_shadow(self):
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = False
            teacher_row = _make_student(bench={
                "aime_bench": 0.10,
                "tool_use_bench": 0.10,
                "self_consistency_bench": 0.10,
            })
            broken = _c.resolve_teacher_broken_axes(
                teacher_row, king_kl=0.1, king_rkl=0.05,
            )
            self.assertNotIn("aime_bench", broken)
            self.assertNotIn("tool_use_bench", broken)
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3


class TestReferenceBrokenAxes(unittest.TestCase):
    """Reference-broken-axes filter: drop bench axes the base model
    cannot even partially attempt. Audit 2026-04-26 found the reference
    Qwen-4B scored pass_frac=0 on aime/code/tool_use/noise_resistance
    every round, locking ``worst() == 0`` for all challengers."""

    def test_reference_zero_axes_marked_broken(self):
        from scripts.validator.composite import resolve_reference_broken_axes
        ref = _make_student(bench={
            "aime_bench": 0.0,         # truncation → eval-broken
            "code_bench": 0.0,         # truncation → eval-broken
            "tool_use_bench": 0.0,     # bad prompt → eval-broken
            "math_bench": 0.5,         # genuine signal — ref can do
            "reasoning_bench": 0.75,   # genuine signal
            "arc_bench": 1.0,          # genuine signal
        })
        broken = resolve_reference_broken_axes(ref)
        self.assertEqual(broken, {"aime_bench", "code_bench", "tool_use_bench"})
        self.assertNotIn("math_bench", broken)
        self.assertNotIn("reasoning_bench", broken)
        self.assertNotIn("arc_bench", broken)

    def test_partial_reference_axis_kept(self):
        """Reference scoring 0.25 (= 1/4) is still useful signal — a
        student outperforming the reference there is meaningfully better.
        Only the *exact zero* floor is treated as eval-broken."""
        from scripts.validator.composite import resolve_reference_broken_axes
        ref = _make_student(bench={
            "math_bench": 0.25,
            "code_bench": 0.25,
        })
        broken = resolve_reference_broken_axes(ref)
        self.assertEqual(broken, set())

    def test_missing_reference_returns_empty(self):
        """Fail open if reference row is absent."""
        from scripts.validator.composite import resolve_reference_broken_axes
        self.assertEqual(resolve_reference_broken_axes(None), set())
        self.assertEqual(resolve_reference_broken_axes({}), set())

    def test_axis_with_n_zero_not_marked_broken(self):
        """If the axis didn't run at all (n=0), don't mark it broken
        from the reference side — that's a different failure mode
        (probe didn't run) handled elsewhere."""
        from scripts.validator.composite import resolve_reference_broken_axes
        ref = {
            "aime_bench": {"n": 0, "correct": 0, "pass_frac": 0.0},
            "code_bench": {"n": 3, "correct": 0, "pass_frac": 0.0},
        }
        broken = resolve_reference_broken_axes(ref)
        self.assertEqual(broken, {"code_bench"})

    def test_relative_axes_never_in_broken_set(self):
        """KL/RKL/capability/length/degeneracy reference at 1.0 by
        construction; the broken-axes filter only inspects bench axes."""
        from scripts.validator.composite import resolve_reference_broken_axes
        ref = {
            "kl_global_avg": 0.0,
            "on_policy_rkl": {"mean_rkl": 0.0},
            "capability": {"pass_frac": 1.0},
            "length_axis": {"penalty": 1.0},
            # No bench axes at all.
        }
        broken = resolve_reference_broken_axes(ref)
        self.assertEqual(broken, set())

    def test_annotate_drops_reference_broken_from_worst(self):
        """End-to-end: annotate_h2h_with_composite must drop reference-
        broken bench axes from ``worst`` so the gate doesn't degenerate."""
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        saved_rd = _c.REASONING_DENSITY_IN_COMPOSITE
        saved_chat = _c.CHAT_TURNS_AXIS_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            # Disable rd/chat for this test — _make_student doesn't
            # populate them and we want to isolate the bench-axis drop.
            _c.REASONING_DENSITY_IN_COMPOSITE = False
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = False
            ref_name = "Qwen/Qwen3.5-4B"
            students_data = {
                ref_name: _make_student(kl=0.0, rkl=0.0, cap_frac=1.0,
                    bench={
                        "aime_bench": 0.0,        # ref zero → drop
                        "code_bench": 0.0,        # ref zero → drop
                        "math_bench": 0.5,        # ref OK
                        "reasoning_bench": 0.75,
                        "knowledge_bench": 0.5,
                        "ifeval_bench": 0.5,
                        "arc_bench": 1.0,
                        "truthful_bench": 1.0,
                        "long_context_bench": 1.0,
                        "procedural_bench": 0.5,
                        "robustness_bench": 0.5,
                        "noise_resistance_bench": 0.33,
                        "self_consistency_bench": 0.33,
                        "mbpp_bench": 0.5,
                        "tool_use_bench": 0.33,
                    }),
                "challenger/m": _make_student(kl=0.2, rkl=0.05, cap_frac=0.8,
                    bench={
                        "aime_bench": 0.0,        # broken — should NOT drag worst
                        "code_bench": 0.0,        # broken — should NOT drag worst
                        "math_bench": 0.5,
                        "reasoning_bench": 0.75,
                        "knowledge_bench": 0.5,
                        "ifeval_bench": 0.5,
                        "arc_bench": 1.0,
                        "truthful_bench": 1.0,
                        "long_context_bench": 1.0,
                        "procedural_bench": 0.5,
                        "robustness_bench": 0.5,
                        "noise_resistance_bench": 0.33,
                        "self_consistency_bench": 0.33,
                        "mbpp_bench": 0.5,
                        "tool_use_bench": 0.33,
                    }),
            }
            h2h_results = [
                {"uid": -1, "model": ref_name, "is_reference": True, "is_king": False},
                {"uid": 100, "model": "challenger/m", "is_king": False},
            ]
            _c.annotate_h2h_with_composite(
                h2h_results, king_kl=0.2,
                students_data=students_data,
                reference_model=ref_name,
                reference_uid=-1,
            )
            challenger = next(r for r in h2h_results if r["uid"] == 100)
            comp = challenger["composite"]
            # Without the reference-broken filter, worst would be 0.0
            # (aime_bench / code_bench dragging). With it, worst should
            # equal the next-lowest axis (noise_resistance_bench=0.33,
            # self_consistency_bench=0.33, or tool_use_bench=0.33 — note
            # tool_use only stays in if not reference-broken; here ref has
            # tool_use=0.33 so it's kept).
            self.assertGreater(comp["worst"], 0.0,
                f"Reference-broken aime/code should drop from worst, "
                f"got {comp['worst']} with axes {comp['axes']}")
            self.assertIn("aime_bench", comp.get("broken_axes", []))
            self.assertIn("code_bench", comp.get("broken_axes", []))
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench
            _c.REASONING_DENSITY_IN_COMPOSITE = saved_rd
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = saved_chat

    def test_worst_drops_broken_but_weighted_keeps_them(self):
        """Asymmetric filter (refined 2026-04-26): the worst-axis gate
        excludes broken axes (else it degenerates to 0=0=0), but the
        weighted aggregator KEEPS them so a student who beats the
        reference on a broken axis still gets credit.
        """
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        saved_rd = _c.REASONING_DENSITY_IN_COMPOSITE
        saved_chat = _c.CHAT_TURNS_AXIS_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            _c.REASONING_DENSITY_IN_COMPOSITE = False
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = False

            ref_bench = {
                "aime_bench": 0.0,
                "tool_use_bench": 0.0,
                "math_bench": 0.5,
                "reasoning_bench": 0.5,
                "knowledge_bench": 0.5,
                "ifeval_bench": 0.5,
                "arc_bench": 0.5,
                "truthful_bench": 0.5,
                "long_context_bench": 0.5,
                "procedural_bench": 0.5,
                "robustness_bench": 0.5,
                "noise_resistance_bench": 0.5,
                "self_consistency_bench": 0.5,
                "mbpp_bench": 0.5,
                "code_bench": 0.5,
            }
            # Two students with identical NON-broken axes but differ on
            # the broken (aime/tool_use) axes:
            #   * weak: 0 on both broken axes (same as ref).
            #   * strong: 1.0 on both broken axes (genuinely better).
            # If weighted ignores broken axes, weak == strong on
            # weighted. If weighted keeps them, strong > weak.
            ref_name = "Qwen/Qwen3.5-4B"
            students_data = {
                ref_name: _make_student(kl=0.0, rkl=0.0, cap_frac=1.0, bench=ref_bench),
                "weak/m": _make_student(
                    kl=0.2, rkl=0.05, cap_frac=0.8,
                    bench={**ref_bench, "aime_bench": 0.0, "tool_use_bench": 0.0},
                ),
                "strong/m": _make_student(
                    kl=0.2, rkl=0.05, cap_frac=0.8,
                    bench={**ref_bench, "aime_bench": 1.0, "tool_use_bench": 1.0},
                ),
            }
            h2h_results = [
                {"uid": -1, "model": ref_name, "is_reference": True, "is_king": False},
                {"uid": 100, "model": "weak/m", "is_king": False},
                {"uid": 101, "model": "strong/m", "is_king": False},
            ]
            _c.annotate_h2h_with_composite(
                h2h_results, king_kl=0.2,
                students_data=students_data,
                reference_model=ref_name,
                reference_uid=-1,
            )
            weak = next(r for r in h2h_results if r["uid"] == 100)["composite"]
            strong = next(r for r in h2h_results if r["uid"] == 101)["composite"]

            # broken_axes set is identical for both (it's round-local).
            self.assertSetEqual(set(weak["broken_axes"]), set(strong["broken_axes"]))
            self.assertIn("aime_bench", weak["broken_axes"])
            self.assertIn("tool_use_bench", weak["broken_axes"])

            # WORST: identical for both (broken axes excluded).
            self.assertEqual(weak["worst"], strong["worst"],
                "worst() should drop broken axes — weak and strong tie there")

            # WEIGHTED: strong > weak (broken axes still contribute).
            self.assertGreater(strong["weighted"], weak["weighted"],
                f"weighted() should KEEP broken axes so genuine wins are "
                f"rewarded: strong={strong['weighted']} weak={weak['weighted']}")
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench
            _c.REASONING_DENSITY_IN_COMPOSITE = saved_rd
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = saved_chat


class TestAnnotateH2HWithPareto(unittest.TestCase):
    """annotate_h2h_with_composite attaches the pareto sub-dict per row."""

    def test_pareto_attached_to_non_king(self):
        from scripts.validator.composite import annotate_h2h_with_composite
        students_data = {
            "king/model": _make_student(
                kl=0.2, rkl=0.08, cap_frac=0.85,
                bench={"math_bench": 0.6, "code_bench": 0.5,
                       "reasoning_bench": 0.7, "knowledge_bench": 0.6,
                       "ifeval_bench": 0.7},
            ),
            "chall/model": _make_student(
                kl=0.19, rkl=0.07, cap_frac=0.9,
                bench={"math_bench": 0.75, "code_bench": 0.65,
                       "reasoning_bench": 0.80, "knowledge_bench": 0.72,
                       "ifeval_bench": 0.78},
            ),
        }
        h2h = [
            {"uid": 10, "model": "king/model", "is_king": True, "kl": 0.2},
            {"uid": 11, "model": "chall/model", "is_king": False, "kl": 0.19},
        ]
        annotate_h2h_with_composite(h2h, king_kl=0.2, students_data=students_data)
        king_row = next(r for r in h2h if r["is_king"])
        chall_row = next(r for r in h2h if not r["is_king"])
        self.assertNotIn("pareto", (king_row.get("composite") or {}),
            "king should not have pareto (vs self)")
        pareto = (chall_row.get("composite") or {}).get("pareto")
        self.assertIsNotNone(pareto)
        self.assertIn("pareto_wins", pareto)
        self.assertIn("comparable", pareto)
        self.assertGreater(pareto["comparable"], 3)


class TestKingHealth(unittest.TestCase):
    """King regression telemetry (2026-04-24, distil-97 leeroyjkin)."""

    def _build_h2h(self, king_bench, base_bench):
        king_data = _make_student(kl=0.2, rkl=0.1, cap_frac=0.85, bench=king_bench)
        base_data = _make_student(kl=0.22, rkl=0.12, cap_frac=0.80, bench=base_bench)
        students_data = {"king/m": king_data, "base/m": base_data}
        h2h = [
            {"uid": 10, "model": "king/m", "is_king": True, "kl": 0.2},
            {"uid": -1, "model": "base/m", "is_king": False, "kl": 0.22},
        ]
        return h2h, students_data

    def test_healthy_king_no_at_risk_flag(self):
        from scripts.validator.composite import annotate_h2h_with_composite
        h2h, sd = self._build_h2h(
            king_bench={"math_bench": 0.75, "code_bench": 0.70,
                        "reasoning_bench": 0.72, "knowledge_bench": 0.68,
                        "ifeval_bench": 0.70},
            base_bench={"math_bench": 0.40, "code_bench": 0.35,
                        "reasoning_bench": 0.42, "knowledge_bench": 0.38,
                        "ifeval_bench": 0.40},
        )
        annotate_h2h_with_composite(h2h, king_kl=0.2, students_data=sd,
            reference_model="base/m", reference_uid=-1)
        health = next(r for r in h2h if r["is_king"])["composite"]["king_health"]
        self.assertFalse(health["at_risk"])
        self.assertFalse(health["below_floor"])
        self.assertFalse(health["worse_than_base"])

    def test_king_below_floor(self):
        from scripts.validator.composite import annotate_h2h_with_composite
        h2h, sd = self._build_h2h(
            king_bench={"math_bench": 0.60, "code_bench": 0.55,
                        "reasoning_bench": 0.70, "knowledge_bench": 0.05,
                        "ifeval_bench": 0.60},
            base_bench={"math_bench": 0.40, "code_bench": 0.35,
                        "reasoning_bench": 0.42, "knowledge_bench": 0.40,
                        "ifeval_bench": 0.40},
        )
        annotate_h2h_with_composite(h2h, king_kl=0.2, students_data=sd,
            reference_model="base/m", reference_uid=-1)
        health = next(r for r in h2h if r["is_king"])["composite"]["king_health"]
        self.assertTrue(health["below_floor"])
        self.assertTrue(health["worse_than_base"])
        self.assertTrue(health["at_risk"])
        # king_worst_axis can be any low-scoring axis, including shadow
        # axes like reasoning_density that derive from knowledge=0.05.
        # Just check that it's populated and points at something real.
        self.assertIsNotNone(health["king_worst_axis"])

    def test_worse_than_base_only(self):
        from scripts.validator.composite import annotate_h2h_with_composite
        h2h, sd = self._build_h2h(
            king_bench={"math_bench": 0.35, "code_bench": 0.30,
                        "reasoning_bench": 0.32, "knowledge_bench": 0.28,
                        "ifeval_bench": 0.30},
            base_bench={"math_bench": 0.55, "code_bench": 0.50,
                        "reasoning_bench": 0.52, "knowledge_bench": 0.48,
                        "ifeval_bench": 0.50},
        )
        annotate_h2h_with_composite(h2h, king_kl=0.2, students_data=sd,
            reference_model="base/m", reference_uid=-1)
        health = next(r for r in h2h if r["is_king"])["composite"]["king_health"]
        self.assertFalse(health["below_floor"])
        self.assertTrue(health["worse_than_base"])
        self.assertTrue(health["at_risk"])

    def test_gate_disabled_by_default(self):
        from scripts.validator.composite import KING_REGRESSION_GATE
        self.assertIsInstance(KING_REGRESSION_GATE, bool)


class TestAxisSummaryStats(unittest.TestCase):
    """Informational balance scores (2026-04-25, non-gating)."""

    def test_spread_low_for_balanced_student(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.10, rkl=0.10, cap_frac=0.85,
            bench={
                "math_bench": 0.80, "code_bench": 0.80,
                "reasoning_bench": 0.80, "knowledge_bench": 0.80,
                "ifeval_bench": 0.80,
            },
        )
        comp = compute_composite(student, king_kl=0.10, king_rkl=0.10)
        self.assertIsNotNone(comp["axis_spread"])
        self.assertLess(comp["axis_spread"], 0.15,
            "balanced student should have low axis spread")

    def test_spread_high_for_narrow_specialist(self):
        from scripts.validator.composite import compute_composite
        # Huge gap between bench axes (rotation-memorized) and relative axes
        # (never learned).
        student = _make_student(
            kl=1.0, rkl=1.5, cap_frac=0.2,
            bench={
                "math_bench": 1.00, "code_bench": 1.00,
                "reasoning_bench": 1.00, "knowledge_bench": 1.00,
                "ifeval_bench": 1.00,
            },
        )
        comp = compute_composite(student, king_kl=1.0, king_rkl=1.0)
        self.assertGreater(comp["axis_spread"], 0.20,
            "narrow specialist (all bench high, cap low) should have wide spread")

    def test_bench_vs_rel_gap_positive_when_bench_high(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.5, rkl=1.0, cap_frac=0.3,  # rel axes much worse than king
            bench={
                "math_bench": 0.95, "code_bench": 0.95,  # bench axes high
                "reasoning_bench": 0.95, "knowledge_bench": 0.95,
                "ifeval_bench": 0.95,
            },
        )
        # King has low KL/RKL, so this student normalizes poorly on rel axes.
        comp = compute_composite(student, king_kl=0.1, king_rkl=0.1)
        self.assertIsNotNone(comp["bench_vs_rel_gap"])
        self.assertGreater(comp["bench_vs_rel_gap"], 0.10,
            "bench >> rel should show a positive gap (potential rotation-memorization)")

    def test_bench_vs_rel_gap_near_zero_for_balanced(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.1, rkl=0.1, cap_frac=0.85,
            bench={
                "math_bench": 0.75, "code_bench": 0.80,
                "reasoning_bench": 0.78, "knowledge_bench": 0.76,
                "ifeval_bench": 0.77,
            },
        )
        comp = compute_composite(student, king_kl=0.1, king_rkl=0.1)
        # rel mean ~0.95 (all axes at 1.0 except capability=0.85/1.0=0.944)
        # bench mean ~0.77 → gap ≈ -0.18
        self.assertIsNotNone(comp["bench_vs_rel_gap"])
        self.assertLess(abs(comp["bench_vs_rel_gap"]), 0.30,
            "balanced student shouldn't show suspiciously large gap")


def _bench_with_tokens(pass_frac: float, n: int,
                       mean_tokens_correct: float,
                       mean_tokens_all: float | None = None) -> dict:
    """Fabricate a bench payload complete with Session 3.2 token stats."""
    return {
        "n": n,
        "correct": int(round(pass_frac * n)),
        "pass_frac": pass_frac,
        "items": [],
        "mean_gen_tokens_correct": mean_tokens_correct,
        "mean_gen_tokens": mean_tokens_all if mean_tokens_all is not None
                           else mean_tokens_correct,
    }


class TestReasoningDensity(unittest.TestCase):
    """Session 3.2 (2026-04-25) — reasoning_density bell-curve axis."""

    def _student_with_bench_tokens(self, bench_payloads: dict) -> dict:
        student = _make_student()
        for k, v in bench_payloads.items():
            student[k] = v
        return student

    def test_axis_high_for_efficient_correct_student(self):
        from scripts.validator.composite import _axis_reasoning_density
        student = self._student_with_bench_tokens({
            "math_bench":     _bench_with_tokens(0.80, 8, 300),   # under 400 target
            "code_bench":     _bench_with_tokens(0.75, 4, 200),   # under 300 target
            "knowledge_bench": _bench_with_tokens(0.70, 8, 20),   # under 30 target
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNotNone(rd)
        self.assertGreater(rd, 0.60,
            "efficient correct student should score high on reasoning_density")

    def test_axis_drops_for_over_thinker(self):
        from scripts.validator.composite import _axis_reasoning_density
        student = self._student_with_bench_tokens({
            "math_bench":     _bench_with_tokens(0.80, 8, 1600),  # 4x target
            "code_bench":     _bench_with_tokens(0.75, 4, 1200),  # 4x target
            "knowledge_bench": _bench_with_tokens(0.70, 8, 120),  # 4x target
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNotNone(rd)
        self.assertLess(rd, 0.35,
            "4x over-thinker should get much lower reasoning_density")

    def test_axis_none_when_no_bench_meets_min_valid(self):
        from scripts.validator.composite import _axis_reasoning_density
        # Benches with n < BENCH_MIN_VALID are skipped entirely; if no
        # bench qualifies there's no data to score → None.
        student = self._student_with_bench_tokens({
            "math_bench": _bench_with_tokens(0.0, 1, 0),  # n=1 < min 4
            "code_bench": _bench_with_tokens(0.0, 1, 0),  # n=1 < min 2
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNone(rd)

    def test_axis_zero_when_all_benches_wrong(self):
        from scripts.validator.composite import _axis_reasoning_density
        # Some benches have correct=0 with known mean_tokens → score 0
        # Others have 0 correct → contribute 0.
        # As long as *any* bench meets min_valid, we return something.
        student = self._student_with_bench_tokens({
            "math_bench": _bench_with_tokens(0.0, 8, 500,
                                             mean_tokens_all=500),
            "code_bench": _bench_with_tokens(0.0, 4, 400,
                                             mean_tokens_all=400),
        })
        rd = _axis_reasoning_density(student)
        # Both benches have correct=0 (per _axis_reasoning_density logic
        # that's a 0 contribution). Mean of zeros → 0.0.
        self.assertIsNotNone(rd)
        self.assertEqual(rd, 0.0)

    def test_density_can_be_disabled(self):
        from scripts.validator.composite import compute_composite
        import scripts.validator.composite as _c
        # Save + force shadow.
        saved = _c.REASONING_DENSITY_IN_COMPOSITE
        _c.REASONING_DENSITY_IN_COMPOSITE = False
        try:
            student = self._student_with_bench_tokens({
                # 10x target → density would be 0.1 × pass_frac → ~0.08
                "math_bench": _bench_with_tokens(0.85, 8, 4000),
            })
            comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertIsNotNone(comp["axes"]["reasoning_density"])
            # Despite a terrible density score, worst is not dragged down
            # because the axis isn't in ``ranked`` (shadow).
            self.assertGreater(comp["worst"], 0.50)
        finally:
            _c.REASONING_DENSITY_IN_COMPOSITE = saved

    def test_promoted_density_gates_worst(self):
        from scripts.validator.composite import compute_composite
        import scripts.validator.composite as _c
        saved = _c.REASONING_DENSITY_IN_COMPOSITE
        _c.REASONING_DENSITY_IN_COMPOSITE = True
        try:
            student = self._student_with_bench_tokens({
                "math_bench": _bench_with_tokens(0.85, 8, 4000),  # 10x
                "code_bench": _bench_with_tokens(0.80, 4, 3000),
                "knowledge_bench": _bench_with_tokens(0.80, 8, 300),
            })
            comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertIsNotNone(comp["axes"]["reasoning_density"])
            # Worst axis now drops because reasoning_density is in ranked.
            self.assertLess(comp["axes"]["reasoning_density"], 0.30)
            self.assertLessEqual(
                comp["worst"], comp["axes"]["reasoning_density"] + 1e-4,
                "promoted reasoning_density should be able to set worst",
            )
        finally:
            _c.REASONING_DENSITY_IN_COMPOSITE = saved


class TestChatTurnsProbe(unittest.TestCase):
    """Session 3.3 (2026-04-25, LIVE) — multi-turn coherence axis."""

    def _student_with_chat_turns(self, normalized, n_valid=5, n=6, n_turns=3) -> dict:
        s = _make_student()
        s["chat_turns_probe"] = {
            "normalized": normalized,
            "mean_score": 1 + 4 * (normalized if normalized is not None else 0.0),
            "n_valid": n_valid,
            "n": n,
            "n_turns": n_turns,
            "in_composite": False,
        }
        return s

    def test_axis_returns_normalized_when_enough_valid(self):
        from scripts.validator.composite import _axis_chat_turns_probe
        s = self._student_with_chat_turns(normalized=0.72, n_valid=5)
        self.assertAlmostEqual(_axis_chat_turns_probe(s), 0.72, places=3)

    def test_axis_none_when_too_few_valid(self):
        from scripts.validator.composite import (
            CHAT_TURNS_MIN_VALID, _axis_chat_turns_probe,
        )
        s = self._student_with_chat_turns(
            normalized=0.9, n_valid=max(0, CHAT_TURNS_MIN_VALID - 1))
        self.assertIsNone(_axis_chat_turns_probe(s))

    def test_axis_none_when_probe_missing(self):
        from scripts.validator.composite import _axis_chat_turns_probe
        s = _make_student()
        s.pop("chat_turns_probe", None)
        self.assertIsNone(_axis_chat_turns_probe(s))

    def test_chat_turns_live_by_default(self):
        from scripts.validator.composite import compute_composite
        s = self._student_with_chat_turns(normalized=0.1, n_valid=5)
        s.update(_make_student(bench={
            "math_bench": 0.9, "code_bench": 0.9,
            "reasoning_bench": 0.9, "knowledge_bench": 0.9,
            "ifeval_bench": 0.9,
        }))
        comp = compute_composite(s, king_kl=0.3, king_rkl=0.1)
        self.assertIsNotNone(comp["axes"]["chat_turns_probe"])
        self.assertTrue(comp["chat_turns_in_composite"])
        self.assertLess(comp["worst"], 0.20,
            "chat_turns_probe=0.1 must drag worst when live")

    def test_promoted_chat_turns_gates_worst(self):
        import scripts.validator.composite as _c
        from scripts.validator.composite import compute_composite
        saved = _c.CHAT_TURNS_AXIS_IN_COMPOSITE
        _c.CHAT_TURNS_AXIS_IN_COMPOSITE = True
        try:
            s = _make_student(bench={
                "math_bench": 0.9, "code_bench": 0.9,
                "reasoning_bench": 0.9, "knowledge_bench": 0.9,
                "ifeval_bench": 0.9,
            })
            s["chat_turns_probe"] = {
                "normalized": 0.12, "mean_score": 1.48,
                "n_valid": 5, "n": 6, "n_turns": 3,
                "in_composite": True,
            }
            comp = compute_composite(s, king_kl=0.3, king_rkl=0.1)
            self.assertTrue(comp["chat_turns_in_composite"])
            self.assertLess(comp["worst"], 0.20,
                "promoted chat_turns axis must be able to set worst")
        finally:
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
