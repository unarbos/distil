"""Regression test: same-model UID collision silently shifts the king.

Observed live on 2026-05-19 between blocks 8219156–8220027. Three rounds
in a row had the seated king AND a non-king challenger committing the
*same* ``model@revision``:

  block=8219156  king_uid=100, challenger=101  (togetherness/spprofound-trainer-12)
  block=8219553  king_uid=253, challenger=15   (power612/lol)
  block=8220027  king_uid=254, challenger=95   (best26/sn97-ms-v11)

The pre-fix ``_build_uid_index`` loop iterated ``spec["students"]`` and
naively overwrote ``uid_index[name]`` on every iteration. With two
students sharing one ``name`` key, the non-king student (iterated later)
won the slot. ``process_round`` then resolved ``record["king_uid"]`` via
the bad lookup, stamping the colliding UID instead of the actual seated
king. That value propagated to ``h2h_latest.king_uid`` and the NEXT
round's ``_resolve_seated_king`` treated the colliding UID as the seat.

Two layers of defence pin this regression:

1. ``_build_uid_index`` runs a king-first pass so the king's row is
   always seated and the non-king pass uses ``setdefault``.
2. ``process_round`` accepts ``seated_king_uid`` from the caller and
   prefers it over the index lookup for the record stamp.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from distil.eval.round import select_challengers
from distil.eval.service import _build_uid_index


def _mk_commit(uid: int, model: str, revision: str, hotkey: str | None = None):
    return SimpleNamespace(
        uid=uid,
        model=model,
        revision=revision,
        key=f"{model}@{revision}",
        hotkey=hotkey or f"5HK{uid:03d}",
        coldkey=f"5CK{uid:03d}",
        commit_block=10_000 + uid,
    )


class TestUidIndexKingCollision(unittest.TestCase):
    def test_king_wins_collision_when_listed_first(self):
        """King in spec position 0, colliding challenger at position 4."""
        spec = {
            "students": [
                {"name": "shared/model@rev1", "uid": 100, "hotkey": "5K100",
                 "revision": "rev1", "is_king": True},
                {"name": "other/a@x", "uid": 56, "hotkey": "5K56",
                 "revision": "x", "is_king": False},
                {"name": "other/b@y", "uid": 33, "hotkey": "5K33",
                 "revision": "y", "is_king": False},
                {"name": "other/c@z", "uid": 16, "hotkey": "5K16",
                 "revision": "z", "is_king": False},
                # ← Same-model challenger AFTER the king. Pre-fix this
                #   row would have overwritten the king's entry.
                {"name": "shared/model@rev1", "uid": 101, "hotkey": "5K101",
                 "revision": "rev1", "is_king": False},
            ],
        }
        commits = {
            100: _mk_commit(100, "shared/model", "rev1"),
            56: _mk_commit(56, "other/a", "x"),
            33: _mk_commit(33, "other/b", "y"),
            16: _mk_commit(16, "other/c", "z"),
            101: _mk_commit(101, "shared/model", "rev1"),
        }
        uid_to_hotkey = {uid: c.hotkey for uid, c in commits.items()}

        idx = _build_uid_index(spec, commits, uid_to_hotkey)

        # The king's entry MUST resolve to the king's UID.
        self.assertEqual(
            idx["shared/model@rev1"]["uid"], 100,
            "king's row was clobbered by colliding challenger (UID 101)",
        )
        self.assertEqual(idx["shared/model@rev1"]["hotkey"], "5K100")

    def test_king_wins_collision_when_listed_last(self):
        """King in spec position 4 (last), colliding challenger first.

        Order-independence matters: even if select_challengers builds the
        spec with the king at the END, the king-first PASS must still win
        the collision.
        """
        spec = {
            "students": [
                {"name": "shared/model@rev1", "uid": 101, "hotkey": "5K101",
                 "revision": "rev1", "is_king": False},
                {"name": "other/a@x", "uid": 56, "hotkey": "5K56",
                 "revision": "x", "is_king": False},
                {"name": "shared/model@rev1", "uid": 100, "hotkey": "5K100",
                 "revision": "rev1", "is_king": True},
            ],
        }
        commits = {
            100: _mk_commit(100, "shared/model", "rev1"),
            56: _mk_commit(56, "other/a", "x"),
            101: _mk_commit(101, "shared/model", "rev1"),
        }
        idx = _build_uid_index(
            spec, commits, {uid: c.hotkey for uid, c in commits.items()}
        )
        self.assertEqual(idx["shared/model@rev1"]["uid"], 100)

    def test_non_king_non_king_collision_keeps_first_listed(self):
        """Two challengers (no king involved) with same model: spec-
        order tie-break — earlier student wins via setdefault."""
        spec = {
            "students": [
                {"name": "kingmodel@kr", "uid": 7, "is_king": True,
                 "revision": "kr"},
                {"name": "shared@s", "uid": 200, "is_king": False,
                 "revision": "s"},
                # Same model committed by a different UID, listed later.
                {"name": "shared@s", "uid": 201, "is_king": False,
                 "revision": "s"},
            ],
        }
        commits = {
            7: _mk_commit(7, "kingmodel", "kr"),
            200: _mk_commit(200, "shared", "s"),
            201: _mk_commit(201, "shared", "s"),
        }
        idx = _build_uid_index(
            spec, commits, {uid: c.hotkey for uid, c in commits.items()}
        )
        self.assertEqual(idx["kingmodel@kr"]["uid"], 7)
        self.assertEqual(
            idx["shared@s"]["uid"], 200,
            "non-king-vs-non-king collision should keep first-listed "
            "(setdefault semantics)",
        )

    def test_no_collision_round_unchanged(self):
        """Healthy round with unique model names — index resolves
        every UID exactly as written in the spec."""
        spec = {
            "students": [
                {"name": "k@1", "uid": 10, "is_king": True, "revision": "1"},
                {"name": "a@1", "uid": 11, "is_king": False, "revision": "1"},
                {"name": "b@1", "uid": 12, "is_king": False, "revision": "1"},
            ],
        }
        commits = {
            10: _mk_commit(10, "k", "1"),
            11: _mk_commit(11, "a", "1"),
            12: _mk_commit(12, "b", "1"),
        }
        idx = _build_uid_index(
            spec, commits, {uid: c.hotkey for uid, c in commits.items()}
        )
        self.assertEqual(idx["k@1"]["uid"], 10)
        self.assertEqual(idx["a@1"]["uid"], 11)
        self.assertEqual(idx["b@1"]["uid"], 12)

    def test_commitment_fallback_does_not_override_spec(self):
        """A UID with a chain commitment but NOT in the spec gets
        added via setdefault — must not shadow a spec entry."""
        spec = {
            "students": [
                {"name": "shared/m@r", "uid": 50, "is_king": True,
                 "revision": "r"},
            ],
        }
        commits = {
            50: _mk_commit(50, "shared/m", "r"),
            # Same model committed by an off-spec UID.
            99: _mk_commit(99, "shared/m", "r"),
        }
        idx = _build_uid_index(
            spec, commits, {uid: c.hotkey for uid, c in commits.items()}
        )
        self.assertEqual(idx["shared/m@r"]["uid"], 50,
                         "off-spec commitment must not override seated king")


class _FakeState:
    """Minimal ValidatorState shim for select_challengers."""

    def __init__(self):
        self.evaluated_uids: set[str] = set()
        self.composite_scores: dict[str, dict] = {}
        self.dq: dict = {}

    def is_disqualified(self, hotkey, *, uid=None):
        return False

    def record_failure(self, uid, key):
        return 0


def _mk_commitment(uid: int, model: str, revision: str, block: int = 10_000):
    """Build a Commitment-like object that select_challengers can consume."""
    return SimpleNamespace(
        uid=uid,
        model=model,
        revision=revision,
        key=f"{model}@{revision}",
        hotkey=f"5HK{uid:03d}",
        coldkey=f"5CK{uid:03d}",
        block=block,
        commit_block=block,
    )


class TestSelectChallengersDedup(unittest.TestCase):
    """Regression: a copycat that re-commits the king's exact
    ``model@revision`` must NOT enter the round as a challenger.

    Pre-fix, ``select_challengers`` only checked UID-level filters
    (king_uid skip, DQ, composite_scores, evaluated_uids). Two UIDs
    sharing one ``model@revision`` both passed the filters because they
    have different UIDs, and the spec built around both — triggering
    the uid_index collision downstream.
    """

    def test_copycat_of_king_is_skipped(self):
        commits = {
            10: _mk_commitment(10, "shared/m", "rev1", block=100),  # king
            11: _mk_commitment(11, "shared/m", "rev1", block=200),  # copycat
            12: _mk_commitment(12, "other/x", "rx", block=150),     # legit
        }
        state = _FakeState()
        chosen = select_challengers(
            commits, state, king_uid=10, n=5, skip_hf_check=True
        )
        uids = sorted(c.uid for c in chosen)
        self.assertEqual(uids, [12],
                         "copycat (UID 11 with king's model@revision) "
                         "must be excluded from the round")

    def test_two_copycats_dedup_keeps_oldest_committer(self):
        """Two non-king UIDs commit the same model; FIFO sort means the
        older committer wins the eval slot, the younger is skipped."""
        commits = {
            5: _mk_commitment(5, "king/k", "kr", block=50),   # king
            6: _mk_commitment(6, "copied/m", "cr", block=200),  # later commit
            7: _mk_commitment(7, "copied/m", "cr", block=100),  # earlier
        }
        state = _FakeState()
        chosen = select_challengers(
            commits, state, king_uid=5, n=5, skip_hf_check=True
        )
        uids = sorted(c.uid for c in chosen)
        self.assertEqual(uids, [7],
                         "FIFO winner (older commit block) keeps the "
                         "eval slot; later same-model committer skipped")

    def test_unique_models_all_accepted(self):
        commits = {
            5: _mk_commitment(5, "king/k", "kr", block=50),
            6: _mk_commitment(6, "a/m", "r", block=100),
            7: _mk_commitment(7, "b/m", "r", block=200),
            8: _mk_commitment(8, "c/m", "r", block=300),
        }
        state = _FakeState()
        chosen = select_challengers(
            commits, state, king_uid=5, n=5, skip_hf_check=True
        )
        self.assertEqual(sorted(c.uid for c in chosen), [6, 7, 8])

    def test_king_not_in_seen_keys_when_no_king(self):
        """No seated king (cold start) — dedup still works among
        challengers."""
        commits = {
            6: _mk_commitment(6, "a/m", "r1", block=100),
            7: _mk_commitment(7, "a/m", "r1", block=200),
            8: _mk_commitment(8, "b/m", "r2", block=300),
        }
        state = _FakeState()
        chosen = select_challengers(
            commits, state, king_uid=None, n=5, skip_hf_check=True
        )
        self.assertEqual(sorted(c.uid for c in chosen), [6, 8])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
