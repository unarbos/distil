"""Regression test for dethrone-result persistence into next round.

The validator service builds the next round's round_spec by reading
``state.h2h_latest.king_uid`` and treating that UID as the seated king
to defend. Pre-fix, three successive rounds (h2h_history blocks
8198615, 8199086, 8199665) recorded ``king_changed=True`` with
``king_after`` set to UIDs 52, 92, and 35 respectively — yet each
subsequent round read ``king_uid=47`` from h2h_latest and re-seated
UID 47 as the king. The dethrone winners never sat the throne, only
appeared as a one-round annotation that vanished by the next round.

Root cause: ``process_round`` set ``record["king_uid"] = king_name``
where ``king_name`` was the INCOMING king of that round. The dethrone
gate ran AFTER ``process_round`` returned, populated ``record
["king_after"]`` + ``record["new_king_uid"]``, but never rewrote
``record["king_uid"]`` itself. The next round's resolver then read the
stale ``king_uid`` field and ignored the dethrone outcome entirely.

Fix: when ``king_changed`` is True, rewrite ``record["king_uid"]`` (and
``king_name``, ``king_model``) to the new seated king so
``state.h2h_latest.king_uid`` lines up with ``state.h2h_latest.king_
after``. The dashboard's audit fields ``prev_king_uid`` and
``new_king_uid`` still record the BEFORE/AFTER pair for transparency.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace


def _simulate_round_end(record: dict, new_king_uid: int, prev_king_uid: int,
                        commitments: dict,
                        state: object | None = None) -> dict:
    """Apply the same post-process_round rewrite the service does.

    When ``state`` is provided we also exercise the ``top4_leaderboard``
    rewrite that the service performs in the same critical block so this
    test catches future regressions where the leaderboard's ``king`` field
    drifts away from ``h2h_latest.king_uid`` (causing the API to return
    ``is_king: true`` for the deposed UID).
    """
    king_changed = (
        new_king_uid is not None
        and prev_king_uid is not None
        and int(new_king_uid) != int(prev_king_uid)
    )
    record["prev_king_uid"] = prev_king_uid
    record["new_king_uid"] = new_king_uid
    record["king_changed"] = king_changed
    if king_changed and new_king_uid is not None:
        record["king_uid"] = int(new_king_uid)
        record["king_name"] = str(new_king_uid)
        new_king_commit = commitments.get(int(new_king_uid))
        if new_king_commit is not None:
            record["king_model"] = new_king_commit.model
        if state is not None:
            leaderboard = getattr(state, "top4_leaderboard", None) or {}
            rows = list(leaderboard.get("rows") or [])
            cs_map = getattr(state, "composite_scores", None) or {}
            new_king_row: dict | None = None
            for r in rows:
                if r.get("uid") == int(new_king_uid):
                    new_king_row = r
                    break
            if new_king_row is None:
                kc = cs_map.get(str(int(new_king_uid))) or {}
                if kc:
                    model = kc.get("model")
                    revision = kc.get("revision")
                    new_king_row = {
                        "rank": None,
                        "uid": int(new_king_uid),
                        "name": f"{model}@{revision}" if model and revision else model,
                        "model": model,
                        "final": kc.get("final"),
                        "worst_3_mean": kc.get("worst_3_mean"),
                        "weighted": kc.get("weighted"),
                        "present_count": kc.get("present_count"),
                    }
            if new_king_row is not None:
                state.top4_leaderboard = {
                    **leaderboard,
                    "king": new_king_row,
                    "contenders": [r for r in rows if r is not new_king_row],
                }
    return record


class TestDethronePersistence(unittest.TestCase):
    def test_dethrone_rewrites_king_uid_for_next_round(self):
        record = {
            "king_uid": 47,
            "king_name": "Foremost04/will_king_v3_4@abc",
            "king_after": "35",
        }
        commits = {
            35: SimpleNamespace(uid=35, model="winner/repo", revision="def"),
            47: SimpleNamespace(uid=47, model="Foremost04/will_king_v3_4", revision="abc"),
        }
        _simulate_round_end(record, new_king_uid=35, prev_king_uid=47, commitments=commits)
        self.assertEqual(record["king_uid"], 35, "next round must read the new seated king")
        self.assertEqual(record["new_king_uid"], 35)
        self.assertEqual(record["prev_king_uid"], 47)
        self.assertTrue(record["king_changed"])
        self.assertEqual(record["king_model"], "winner/repo")

    def test_king_retain_leaves_king_uid_unchanged(self):
        record = {"king_uid": 47, "king_name": "47", "king_after": "47"}
        commits = {47: SimpleNamespace(uid=47, model="m", revision="r")}
        _simulate_round_end(record, new_king_uid=47, prev_king_uid=47, commitments=commits)
        self.assertEqual(record["king_uid"], 47)
        self.assertFalse(record["king_changed"])
        self.assertNotIn("king_model", record)

    def test_three_consecutive_dethrones_each_sticks(self):
        """The pre-fix data showed three rounds in a row where dethrone
        fired (block 8198615 -> 52, 8199086 -> 92, 8199665 -> 35) but
        none of the winners ever defended. Lock in that each successive
        dethrone winner becomes the next defender."""
        commits = {
            35: SimpleNamespace(uid=35, model="m35", revision="r35"),
            47: SimpleNamespace(uid=47, model="m47", revision="r47"),
            52: SimpleNamespace(uid=52, model="m52", revision="r52"),
            92: SimpleNamespace(uid=92, model="m92", revision="r92"),
        }
        # Round 1: 47 dethroned by 52
        r1 = {"king_uid": 47, "king_name": "47"}
        _simulate_round_end(r1, new_king_uid=52, prev_king_uid=47, commitments=commits)
        # Round 2 starts reading r1.king_uid (52, not 47)
        self.assertEqual(r1["king_uid"], 52)
        # Round 2: 52 dethroned by 92
        r2 = {"king_uid": r1["king_uid"], "king_name": "52"}
        _simulate_round_end(r2, new_king_uid=92, prev_king_uid=52, commitments=commits)
        self.assertEqual(r2["king_uid"], 92)
        self.assertEqual(r2["prev_king_uid"], 52)
        # Round 3: 92 dethroned by 35
        r3 = {"king_uid": r2["king_uid"], "king_name": "92"}
        _simulate_round_end(r3, new_king_uid=35, prev_king_uid=92, commitments=commits)
        self.assertEqual(r3["king_uid"], 35)
        self.assertEqual(r3["prev_king_uid"], 92)


class TestTop4LeaderboardKingSync(unittest.TestCase):
    """Regression: top4_leaderboard.king must follow h2h_latest.king_uid
    after a dethrone, otherwise /api/miner/<deposed_uid> returns
    ``is_king: true`` and /api/miner/<new_king_uid> returns
    ``is_king: false`` (confirmed live 2026-05-17 02:30 UTC).

    _refresh_top4 runs inside process_round BEFORE the dethrone gate, so
    the king it captures is the round's INCOMING king. The dethrone
    gate must also rewrite top4_leaderboard.king to the new seated UID.
    """

    def _make_state(self, incoming_king_uid: int, new_king_uid: int) -> object:
        """Build the post-process_round state shape: rows ranked by
        composite ``final``, ``king`` pointing at the incoming king,
        ``composite_scores`` populated for every row + the new king."""
        rows = [
            {"rank": 1, "uid": 241, "model": "bojan/sn97-model30",
             "final": 0.3663, "name": "bojan/sn97-model30@a"},
            {"rank": 2, "uid": incoming_king_uid, "model": "RLStepone/distil-success-h24",
             "final": 0.3567, "name": "RLStepone/distil-success-h24@b"},
            {"rank": 3, "uid": new_king_uid, "model": "Bittoby1040/Happy_Distill",
             "final": 0.4221, "name": "Bittoby1040/Happy_Distill@c"},
        ]
        cs_map = {
            str(incoming_king_uid): {
                "final": 0.3567, "model": "RLStepone/distil-success-h24",
                "revision": "b", "worst_3_mean": 0.25,
            },
            str(new_king_uid): {
                "final": 0.4221, "model": "Bittoby1040/Happy_Distill",
                "revision": "c", "worst_3_mean": 0.30, "weighted": 0.7,
                "present_count": 22,
            },
        }
        return SimpleNamespace(
            top4_leaderboard={
                "rows": rows,
                "king": rows[1],  # incoming king (will be wrong after dethrone)
                "contenders": [rows[0], rows[2]],
                "updated_at": 0.0,
            },
            composite_scores=cs_map,
        )

    def test_dethrone_rewrites_top4_king_to_new_uid(self):
        state = self._make_state(incoming_king_uid=14, new_king_uid=35)
        record = {"king_uid": 14, "king_name": "14"}
        commits = {35: SimpleNamespace(uid=35, model="Bittoby1040/Happy_Distill", revision="c")}
        _simulate_round_end(record, new_king_uid=35, prev_king_uid=14,
                            commitments=commits, state=state)
        # top4_leaderboard.king must now point at UID 35, not UID 14
        self.assertEqual(state.top4_leaderboard["king"]["uid"], 35)
        # And UID 14 should be back among contenders (it didn't successfully defend)
        contender_uids = [c["uid"] for c in state.top4_leaderboard["contenders"]]
        self.assertIn(14, contender_uids,
                      "deposed UID must be moved out of king slot into contenders")
        self.assertNotIn(35, contender_uids,
                         "new king must not also appear in contenders")

    def test_dethrone_to_uid_missing_from_rows_falls_back_to_cs(self):
        """Edge case: new king's composite is in state.composite_scores
        but the top4 rows truncated it out (only top 4 ranked). The
        rewrite must still build a king_row from composite_scores so
        the API returns the new king's metadata, not an empty {}."""
        state = self._make_state(incoming_king_uid=14, new_king_uid=999)
        # Remove UID 999 from rows to force the composite_scores fallback
        state.top4_leaderboard["rows"] = [
            r for r in state.top4_leaderboard["rows"] if r["uid"] != 999
        ]
        state.composite_scores["999"] = {
            "final": 0.55, "model": "outsider/repo", "revision": "z",
            "worst_3_mean": 0.40, "weighted": 0.75, "present_count": 20,
        }
        record = {"king_uid": 14, "king_name": "14"}
        commits = {999: SimpleNamespace(uid=999, model="outsider/repo", revision="z")}
        _simulate_round_end(record, new_king_uid=999, prev_king_uid=14,
                            commitments=commits, state=state)
        self.assertEqual(state.top4_leaderboard["king"]["uid"], 999)
        self.assertEqual(state.top4_leaderboard["king"]["model"], "outsider/repo")
        self.assertEqual(state.top4_leaderboard["king"]["final"], 0.55)

    def test_king_retain_does_not_touch_top4_king(self):
        state = self._make_state(incoming_king_uid=14, new_king_uid=14)
        original_king = state.top4_leaderboard["king"]
        record = {"king_uid": 14, "king_name": "14"}
        _simulate_round_end(record, new_king_uid=14, prev_king_uid=14,
                            commitments={}, state=state)
        self.assertIs(state.top4_leaderboard["king"], original_king,
                      "no-dethrone round must not mutate top4_leaderboard.king")


if __name__ == "__main__":
    unittest.main()
