"""Regression: ``_emit_defense_announcement`` fires when the king
holds against real challengers, and skips otherwise.

Driven by the 2026-05-20 Discord ask: "automated king change + defense
announcements in this channel so you don't have to ask me every time".
The dethrone announcer already fires on king-change; this covers the
defense path.

Gating contract:
  * ``king_changed`` must be False (defense only — dethrone is a
    separate announcement).
  * The round must include at least one student that is NOT the king
    AND has a numeric ``composite.final`` (i.e. successfully scored).
  * DQ'd / failed students are excluded from the challenger count.
  * King-only re-eval rounds (no challengers) produce NO defense post.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

REPO = os.path.dirname(os.path.dirname(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from distil.eval.service import _emit_defense_announcement  # noqa: E402


class _FakeState:
    def __init__(self):
        self.composite_scores: dict = {}


def _commit(uid: int, model: str) -> SimpleNamespace:
    return SimpleNamespace(
        uid=uid,
        model=model,
        revision="rev",
        key=f"{model}@rev",
        hotkey=f"5H{uid:03d}",
    )


def _capture_defense_call(record, *, king_uid, king_changed):
    """Helper: run ``_emit_defense_announcement`` with the announce
    transport patched to a Mock so we can assert call args."""
    state = _FakeState()
    commitments = {king_uid: _commit(king_uid, "king/model")}
    with mock.patch("distil.eval.announce.announce_king_defense") as m:
        _emit_defense_announcement(
            state=state,
            record=record,
            king_changed=king_changed,
            king_uid=king_uid,
            commitments=commitments,
        )
    return m


def test_defense_fires_with_one_scored_challenger():
    record = {
        "block": 100,
        "king_model": "king/model",
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
            {"uid": 2, "composite": {"final": 0.40}, "model": "chal/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=False)
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["king_uid"] == 1
    assert kwargs["top_challenger_uid"] == 2
    assert kwargs["king_composite_final"] == 0.50
    assert kwargs["top_challenger_final"] == 0.40
    assert kwargs["n_challengers"] == 1


def test_defense_picks_strongest_challenger_when_many():
    """``top_challenger`` is the highest-scoring non-king student."""
    record = {
        "block": 100,
        "king_model": "king/model",
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
            {"uid": 2, "composite": {"final": 0.30}, "model": "weak/model"},
            {"uid": 3, "composite": {"final": 0.45}, "model": "strong/model"},
            {"uid": 4, "composite": {"final": 0.20}, "model": "weakest/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=False)
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["top_challenger_uid"] == 3
    assert kwargs["top_challenger_final"] == 0.45
    assert kwargs["n_challengers"] == 3


def test_defense_skips_dq_and_failed_challengers():
    """DQ'd / failed students are not real challengers — must not count."""
    record = {
        "block": 100,
        "king_model": "king/model",
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
            {"uid": 2, "composite": {"final": 0.49}, "model": "dq/model",
             "disqualified": True},
            {"uid": 3, "composite": {"final": 0.48}, "model": "failed/model",
             "status": "failed"},
            {"uid": 4, "composite": {"final": 0.30}, "model": "real/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=False)
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["top_challenger_uid"] == 4
    assert kwargs["n_challengers"] == 1


def test_defense_skips_when_king_changed():
    """Dethrone path already handles announcement — defense must NOT fire."""
    record = {
        "block": 100,
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
            {"uid": 2, "composite": {"final": 0.40}, "model": "chal/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=True)
    m.assert_not_called()


def test_defense_skips_when_no_challengers():
    """King-only re-eval round (no other students) — no defense to announce."""
    record = {
        "block": 100,
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=False)
    m.assert_not_called()


def test_defense_skips_when_all_challengers_lack_final():
    """A challenger that failed to score (no composite.final) doesn't count."""
    record = {
        "block": 100,
        "students": [
            {"uid": 1, "composite": {"final": 0.50}, "model": "king/model"},
            {"uid": 2, "composite": {"final": None}, "model": "no-score/model"},
            {"uid": 3, "model": "no-composite/model"},
        ],
    }
    m = _capture_defense_call(record, king_uid=1, king_changed=False)
    m.assert_not_called()


def test_defense_falls_back_to_composite_scores_for_king_final():
    """If the round record doesn't surface the king's composite, the
    defense announcer pulls it from ``state.composite_scores``."""
    record = {
        "block": 100,
        "king_model": "king/model",
        "students": [
            {"uid": 1, "model": "king/model"},  # king has no composite in record
            {"uid": 2, "composite": {"final": 0.40}, "model": "chal/model"},
        ],
    }
    state = _FakeState()
    state.composite_scores = {"1": {"final": 0.55, "model": "king/model"}}
    commitments = {1: _commit(1, "king/model")}
    with mock.patch("distil.eval.announce.announce_king_defense") as m:
        _emit_defense_announcement(
            state=state,
            record=record,
            king_changed=False,
            king_uid=1,
            commitments=commitments,
        )
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["king_composite_final"] == 0.55
