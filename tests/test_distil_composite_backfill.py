"""Regression tests for ``composite_backfill.backfill_missing_composites``.

UIDs flagged as ``evaluated`` without a corresponding entry in
``composite_scores.json`` surface as ``eval_status:
evaluated_no_composite`` in the API and earn zero emission. Pre-fix
(2026-05-16) the distil cutover rebuilt ``composite_scores.json`` from
scratch but kept ``evaluated_uids.json``, leaving ~120 historical UIDs
stuck in that state. UID 214 (was king at block 8163228) was the
public-facing example flagged in #distil.

This recovery walks ``state.h2h_history`` newest-first and copies the
most-recent valid composite back into ``state.composite_scores`` for
each affected UID. The sweep is idempotent and runs at the top of
``service._round`` so a redeploy can never permanently zero-out a
miner's emission share by accident.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from distil.eval.composite_backfill import backfill_missing_composites


def _state(*, evaluated_uids, composite_scores, h2h_history):
    return SimpleNamespace(
        evaluated_uids=list(evaluated_uids),
        composite_scores=dict(composite_scores),
        h2h_history=list(h2h_history),
    )


class TestComposiBackfill(unittest.TestCase):
    def test_backfills_uid_with_h2h_composite(self):
        state = _state(
            evaluated_uids=["214"],
            composite_scores={},
            h2h_history=[
                {
                    "block": 8163228,
                    "results": [
                        {
                            "uid": 214,
                            "model": "Bittoby1040/Happy_Const@abc",
                            "composite": {
                                "final": 0.45,
                                "worst": 0.22,
                                "axes": {"kl": 1.0},
                            },
                        }
                    ],
                }
            ],
        )
        backfilled = backfill_missing_composites(state)
        self.assertEqual(len(backfilled), 1)
        self.assertEqual(backfilled[0]["uid"], 214)
        self.assertIn("214", state.composite_scores)
        self.assertEqual(state.composite_scores["214"]["final"], 0.45)

    def test_idempotent_when_already_present(self):
        state = _state(
            evaluated_uids=["47"],
            composite_scores={"47": {"final": 0.5}},
            h2h_history=[
                {
                    "block": 1,
                    "results": [
                        {"uid": 47, "composite": {"final": 0.9}},
                    ],
                }
            ],
        )
        backfilled = backfill_missing_composites(state)
        self.assertEqual(backfilled, [])
        self.assertEqual(state.composite_scores["47"]["final"], 0.5)

    def test_skips_uid_without_h2h_composite(self):
        state = _state(
            evaluated_uids=["999"],
            composite_scores={},
            h2h_history=[
                {
                    "block": 1,
                    "results": [
                        {"uid": 999, "composite": None},
                        {"uid": 999, "composite": {"final": None}},
                    ],
                }
            ],
        )
        backfilled = backfill_missing_composites(state)
        self.assertEqual(backfilled, [])
        self.assertNotIn("999", state.composite_scores)

    def test_picks_newest_round_for_uid(self):
        state = _state(
            evaluated_uids=["7"],
            composite_scores={},
            h2h_history=[
                {
                    "block": 100,
                    "results": [{"uid": 7, "composite": {"final": 0.2}}],
                },
                {
                    "block": 200,
                    "results": [{"uid": 7, "composite": {"final": 0.6}}],
                },
                {
                    "block": 150,
                    "results": [{"uid": 7, "composite": {"final": 0.4}}],
                },
            ],
        )
        backfilled = backfill_missing_composites(state)
        self.assertEqual(state.composite_scores["7"]["final"], 0.6)
        self.assertEqual(backfilled[0]["block"], 200)

    def test_preserves_model_metadata_for_audit(self):
        state = _state(
            evaluated_uids=["7"],
            composite_scores={},
            h2h_history=[
                {
                    "block": 50,
                    "results": [
                        {
                            "uid": 7,
                            "model": "user/repo@deadbeef",
                            "composite": {"final": 0.3},
                        }
                    ],
                }
            ],
        )
        backfill_missing_composites(state)
        e = state.composite_scores["7"]
        self.assertEqual(e["model"], "user/repo")
        self.assertEqual(e["revision"], "deadbeef")
        self.assertEqual(e["block"], 50)

    def test_skips_uids_not_in_evaluated_uids(self):
        state = _state(
            evaluated_uids=["1"],
            composite_scores={},
            h2h_history=[
                {
                    "block": 10,
                    "results": [
                        {"uid": 1, "composite": {"final": 0.5}},
                        {"uid": 2, "composite": {"final": 0.6}},
                    ],
                }
            ],
        )
        backfilled = backfill_missing_composites(state)
        uids = [b["uid"] for b in backfilled]
        self.assertIn(1, uids)
        self.assertNotIn(2, uids)


if __name__ == "__main__":
    unittest.main()
