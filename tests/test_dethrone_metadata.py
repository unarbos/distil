"""Regression: dethrone-rewrite must stamp ``king_name`` as
``model@revision``, not as a bare UID-string.

Pre-fix (2026-05-20 Discord report, Arbos summary item #4) the
``_rewrite_record_for_dethrone`` helper wrote::

    record["king_name"] = str(new_king_uid)

so every dethrone round produced ``king_name: "104"`` instead of
``king_name: "best26/sn97-ms-v14@cf5d1a3d..."``. The v2 dashboard then
displayed the UID where the model name should have been, and the
sn97-bot summarised the symptom as "h2h metadata bug — king_name /
king_model inheriting from previous king instead of current".

The fix mirrors the PRE-dethrone schema from
``distil.eval.results.process_round`` (which writes the proper
``model@revision`` form), pulling the canonical id off
``commitment.key`` (falling back to ``f"{model}@{revision}"`` when the
``key`` attribute is missing).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


from distil.eval.service import _rewrite_record_for_dethrone  # noqa: E402


class _FakeState:
    def __init__(self):
        self.top4_leaderboard: dict = {}
        self.composite_scores: dict = {}


def _commit(uid: int, model: str, revision: str = "rev_abc") -> SimpleNamespace:
    """Mimics ``distil.chain.commitments.Commitment``."""
    return SimpleNamespace(
        uid=uid,
        model=model,
        revision=revision,
        key=f"{model}@{revision}",
        hotkey=f"5H{uid:03d}",
    )


def test_dethrone_rewrite_writes_model_at_revision_to_king_name():
    """The dethrone rewrite must produce ``king_name = "model@revision"``."""
    record: dict = {
        "block": 8220560,
        # process_round left these stamped on the PRE-dethrone king
        "king_uid": 183,
        "king_name": "const0312/arbosarbos@old_rev",
        "king_model": "const0312/arbosarbos",
    }
    commitments = {
        183: _commit(183, "const0312/arbosarbos", "old_rev"),
        104: _commit(104, "best26/sn97-ms-v14", "cf5d1a3d"),
    }
    state = _FakeState()
    _rewrite_record_for_dethrone(
        state=state,
        record=record,
        new_king_uid=104,
        commitments=commitments,
    )
    assert record["king_uid"] == 104
    # The critical assertion: king_name is the model@revision pair,
    # NOT the bare UID-string. Matches the schema process_round uses
    # for the PRE-dethrone case (results.py:673).
    assert record["king_name"] == "best26/sn97-ms-v14@cf5d1a3d"
    assert record["king_model"] == "best26/sn97-ms-v14"


def test_dethrone_rewrite_falls_back_when_commit_missing_key_attr():
    """Older commitment shape with no ``.key`` attribute still gets a
    proper ``model@revision`` string (constructed inline)."""
    record: dict = {"king_uid": 1, "king_name": "old@rev", "king_model": "old"}
    bare = SimpleNamespace(uid=2, model="alice/model", revision="r99")
    # Intentionally NO ``key`` attribute on the commitment.
    commitments = {1: _commit(1, "old", "rev"), 2: bare}
    state = _FakeState()
    _rewrite_record_for_dethrone(
        state=state,
        record=record,
        new_king_uid=2,
        commitments=commitments,
    )
    assert record["king_name"] == "alice/model@r99"
    assert record["king_model"] == "alice/model"


def test_dethrone_rewrite_does_not_collapse_revision_to_main_when_present():
    """Defensive: the ``or 'main'`` fallback must NOT trigger for the
    happy path. A real revision must be preserved verbatim."""
    record: dict = {"king_uid": 1, "king_name": "old", "king_model": "old"}
    commitments = {
        1: _commit(1, "old", "rev"),
        2: SimpleNamespace(uid=2, model="bob/v1", revision="38f26a463619b4f04f69d88397604404997f34f2"),
    }
    state = _FakeState()
    _rewrite_record_for_dethrone(
        state=state,
        record=record,
        new_king_uid=2,
        commitments=commitments,
    )
    assert record["king_name"] == "bob/v1@38f26a463619b4f04f69d88397604404997f34f2"


def test_dethrone_rewrite_fallback_uid_string_when_commitment_missing(caplog):
    """When the new_king_uid is NOT in commitments (shouldn't happen in
    prod, but the dethrone gate's pre-check is the only guarantee),
    we log a warning and fall back to a UID-string. Never crash."""
    record: dict = {"king_uid": 1, "king_name": "old", "king_model": "old"}
    commitments = {1: _commit(1, "old", "rev")}  # no entry for UID 999
    state = _FakeState()
    import logging
    with caplog.at_level(logging.WARNING):
        _rewrite_record_for_dethrone(
            state=state,
            record=record,
            new_king_uid=999,
            commitments=commitments,
        )
    assert record["king_uid"] == 999
    assert record["king_name"] == "999"  # defensive fallback
    assert any("no commitment for new_king_uid" in r.message for r in caplog.records)
