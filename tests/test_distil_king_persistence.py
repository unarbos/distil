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
                        commitments: dict) -> dict:
    """Apply the same post-process_round rewrite the service does."""
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


if __name__ == "__main__":
    unittest.main()
