"""Regression test for the ``/api/composite-config`` endpoint.

Miners asked for a machine-readable spec of the composite-score axes
and dethrone thresholds (#distil 2026-05-16, flag #18). Without it
they had to read ``distil/eval/composite.py`` source to know which
axes contribute to ``final``, which are telemetry-only, what the
per-axis minimum sample floor is, and what dethrone margin is in
force. The endpoint surfaces the canonical schema so third-party
tooling can pin against the same definitions the validator uses.
"""

from __future__ import annotations

import unittest

from distil.api.routes import composite_config


class TestCompositeConfigEndpoint(unittest.TestCase):
    def test_schema_version_is_v32(self):
        cfg = composite_config()
        self.assertEqual(cfg["schema_version"], 32)

    def test_axis_partition_complete(self):
        """All eleven v31 procedural axes must be exposed."""
        cfg = composite_config()
        v31 = cfg["axes"]["v31_procedural"]
        for axis in (
            "v31_math_gsm_symbolic",
            "v31_math_competition",
            "v31_math_robustness",
            "v31_code_humaneval_plus",
            "v31_reasoning_logic_grid",
            "v31_reasoning_dyval_arith",
            "v31_long_context_ruler",
            "v31_knowledge_multi_hop_kg",
            "v31_ifeval_verifiable",
            "v31_truthfulness_calibration",
            "v31_consistency_paraphrase",
        ):
            self.assertIn(axis, v31, f"missing v31 axis: {axis}")

    def test_dethrone_thresholds_present(self):
        cfg = composite_config()
        dt = cfg["dethrone"]
        self.assertIn("margin", dt)
        self.assertIn("min_axes", dt)
        self.assertIn("floor", dt)
        self.assertIsInstance(dt["margin"], (int, float))
        self.assertIsInstance(dt["floor"], (int, float))

    def test_final_blend_has_formula_and_alpha(self):
        cfg = composite_config()
        fb = cfg["final_blend"]
        self.assertIn("formula", fb)
        self.assertIn("alpha_bottom", fb)
        self.assertIn("worst_k", fb)

    def test_single_eval_policy_present(self):
        cfg = composite_config()
        se = cfg["single_eval"]
        self.assertIn("max_load_failures", se)
        self.assertIn("max_per_round", se)
        self.assertGreaterEqual(se["max_load_failures"], 1)

    def test_bench_min_valid_floors_for_each_v31_axis(self):
        cfg = composite_config()
        floors = cfg["bench_min_valid"]
        for axis in cfg["axes"]["v31_procedural"]:
            self.assertIn(axis, floors, f"no floor declared for {axis}")
            self.assertGreaterEqual(floors[axis], 1)


if __name__ == "__main__":
    unittest.main()
