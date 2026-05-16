"""Regression tests for the ``top4_leaderboard.json`` schema.

The dashboard's ``/api/miner/{uid}`` endpoint (in ``api/routes/miners.py``)
sets ``is_king`` and ``in_top5`` by destructuring:

    top4 = top4_leaderboard()
    king = top4.get("king") or {}
    contenders = top4.get("contenders") or []
    is_king = king.get("uid") == uid
    in_top5 = uid in {king.get("uid"), *(c.get("uid") for c in contenders)}

Pre-fix, ``_refresh_top4`` only wrote ``{updated_at, rows}`` — no
``king`` field, no ``contenders``, and no ``uid`` on the row dicts.
The endpoint always returned ``is_king: false`` and ``in_top5: false``
for every UID including the actual king (UID 47), flagged in #distil
2026-05-16 by ``coffiee`` and ``aizaysi``.

This test locks in the new schema:

* ``top4_leaderboard`` MUST contain ``king``, ``contenders``, ``rows``.
* ``king`` MUST carry ``uid`` so the dashboard equality check fires.
* ``contenders[*]`` MUST carry ``uid`` so ``in_top5`` resolves.
* Row 0 in ``rows`` may not be the king (sparse-round case) — the
  king must come from ``h2h_latest.king_uid``, not from the top-final
  ranking, otherwise a high-scoring challenger one round before
  dethrone would be mislabeled.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from distil.eval.results import _refresh_top4


def _comp(final: float, *, kl: float | None = 0.5, disqualified: bool = False) -> dict:
    return {
        "final": final,
        "worst_3_mean": final * 0.8,
        "weighted": final * 1.05,
        "present_count": 22,
        "disqualified": disqualified,
        "axes": {"kl": kl} if kl is not None else {},
    }


def _state(*, h2h_king_uid: int | None, composite_scores: dict[str, dict]):
    s = MagicMock()
    s.h2h_latest = {"king_uid": h2h_king_uid} if h2h_king_uid is not None else {}
    s.composite_scores = composite_scores
    s.top4_leaderboard = {}
    return s


class TestTop4SchemaShape(unittest.TestCase):
    def test_writes_king_contenders_and_rows(self):
        s = _state(
            h2h_king_uid=47,
            composite_scores={
                "47": {"model": "Foremost04/will_king_v3_4", "revision": "abc", "final": 0.31},
                "99": {"model": "Doomate/xeon-v37", "revision": "def", "final": 0.55},
            },
        )
        composites = {
            "Foremost04/will_king_v3_4@abc": _comp(0.31),
            "Doomate/xeon-v37@def": _comp(0.55),
        }
        _refresh_top4(s, composites)
        t4 = s.top4_leaderboard
        self.assertIn("king", t4)
        self.assertIn("contenders", t4)
        self.assertIn("rows", t4)

    def test_king_carries_uid_for_is_king_endpoint(self):
        s = _state(
            h2h_king_uid=47,
            composite_scores={
                "47": {"model": "Foremost04/will_king_v3_4", "revision": "abc", "final": 0.31},
            },
        )
        composites = {"Foremost04/will_king_v3_4@abc": _comp(0.31)}
        _refresh_top4(s, composites)
        self.assertEqual(s.top4_leaderboard["king"].get("uid"), 47)

    def test_contenders_carry_uid_for_in_top5_endpoint(self):
        s = _state(
            h2h_king_uid=47,
            composite_scores={
                "47": {"model": "k", "revision": "a", "final": 0.31},
                "99": {"model": "c1", "revision": "b", "final": 0.55},
                "100": {"model": "c2", "revision": "c", "final": 0.45},
            },
        )
        composites = {
            "k@a": _comp(0.31),
            "c1@b": _comp(0.55),
            "c2@c": _comp(0.45),
        }
        _refresh_top4(s, composites)
        uids = {c.get("uid") for c in s.top4_leaderboard["contenders"]}
        self.assertIn(99, uids)
        self.assertIn(100, uids)

    def test_king_picked_from_h2h_latest_not_top_final(self):
        """The h2h-canonical king is the seated one even if a fresh
        contender has a higher composite this round (dethrone happens
        in a later step). Pre-fix this would silently mislabel.
        """
        s = _state(
            h2h_king_uid=47,
            composite_scores={
                "47": {"model": "king_model", "revision": "a", "final": 0.31},
                "99": {"model": "challenger", "revision": "b", "final": 0.92},
            },
        )
        composites = {
            "king_model@a": _comp(0.31),
            "challenger@b": _comp(0.92),
        }
        _refresh_top4(s, composites)
        self.assertEqual(s.top4_leaderboard["king"].get("uid"), 47)
        contender_uids = {c.get("uid") for c in s.top4_leaderboard["contenders"]}
        self.assertIn(99, contender_uids)
        self.assertNotIn(47, contender_uids)

    def test_filters_failed_phase2_rows_from_rows(self):
        s = _state(
            h2h_king_uid=47,
            composite_scores={"47": {"model": "k", "revision": "a", "final": 0.5}},
        )
        composites = {
            "k@a": _comp(0.5),
            "ghost@x": _comp(0.0, kl=None),
            "dq@y": _comp(0.7, disqualified=True),
        }
        _refresh_top4(s, composites)
        names = [r["name"] for r in s.top4_leaderboard["rows"]]
        self.assertIn("k@a", names)
        self.assertNotIn("ghost@x", names)
        self.assertNotIn("dq@y", names)

    def test_king_missing_from_composites_still_resolved(self):
        """The king may be missing from this round's composites if their
        commitment changed mid-round. ``top4.king`` should still be
        populated from ``composite_scores`` so the dashboard never
        flashes ``is_king: false`` for the actual seated king.
        """
        s = _state(
            h2h_king_uid=47,
            composite_scores={
                "47": {
                    "model": "king_model",
                    "revision": "a",
                    "final": 0.31,
                    "worst_3_mean": 0.20,
                    "weighted": 0.55,
                    "present_count": 22,
                },
                "99": {"model": "c", "revision": "b", "final": 0.55},
            },
        )
        composites = {"c@b": _comp(0.55)}
        _refresh_top4(s, composites)
        self.assertEqual(s.top4_leaderboard["king"].get("uid"), 47)
        self.assertEqual(s.top4_leaderboard["king"].get("final"), 0.31)


if __name__ == "__main__":
    unittest.main()
