#!/usr/bin/env python3
"""Focused tests for maintenance-round challenger capping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class FakeState:
    def __init__(self):
        self.top4_leaderboard = {
            "phase": "maintenance",
            "contenders": [
                {"uid": 10, "h2h_kl": 0.10},
                {"uid": 11, "h2h_kl": 0.11},
                {"uid": 12, "h2h_kl": 0.12},
                {"uid": 13, "h2h_kl": 0.13},
                {"uid": 14, "h2h_kl": 0.14},
                {"uid": 15, "h2h_kl": 0.15},
            ],
        }
        self.scores = {
            "10": 0.10,
            "11": 0.11,
            "12": 0.12,
            "13": 0.13,
            "14": 0.14,
            "15": 0.15,
            "20": 0.09,
            "21": 0.095,
            "22": 0.16,
        }
        self.evaluated_uids = set(self.scores)


class TestChallengerCap(unittest.TestCase):
    def test_cap_protects_top_h2h_but_keeps_new_submissions(self):
        import scripts.validator.challengers as ch

        saved_cap = ch.MAINTENANCE_CHALLENGER_CAP
        saved_protected = ch.PROTECTED_H2H_CONTENDERS
        try:
            ch.MAINTENANCE_CHALLENGER_CAP = 8
            ch.PROTECTED_H2H_CONTENDERS = 3
            state = FakeState()
            challengers = {
                48: {"model": "king", "commit_block": 1000},
                10: {"model": "h2h-1", "commit_block": 900},
                11: {"model": "h2h-2", "commit_block": 900},
                12: {"model": "h2h-3", "commit_block": 900},
                13: {"model": "h2h-4", "commit_block": 900},
                14: {"model": "h2h-5", "commit_block": 900},
                15: {"model": "h2h-6", "commit_block": 900},
                20: {"model": "dormant-best", "commit_block": 800},
                21: {"model": "dormant-good", "commit_block": 800},
                30: {"model": "newer-new", "commit_block": 1200},
                31: {"model": "older-new", "commit_block": 1100},
            }

            ch.cap_challengers(challengers, state, king_uid=48)

            self.assertEqual(len(challengers), 8)
            self.assertIn(48, challengers)
            self.assertTrue({10, 11, 12}.issubset(challengers))
            self.assertTrue({30, 31}.issubset(challengers))
            self.assertIn(20, challengers)
            self.assertNotIn(15, challengers)
        finally:
            ch.MAINTENANCE_CHALLENGER_CAP = saved_cap
            ch.PROTECTED_H2H_CONTENDERS = saved_protected


if __name__ == "__main__":
    unittest.main()
