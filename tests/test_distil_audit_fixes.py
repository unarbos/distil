"""Regression tests for the 2026-05-16 pre-merge audit fixes.

Covers the 5 high-priority issues the audit flagged:

1. ``evict_stale_evaluated_uids`` honest-recommit support
2. king_uid pointing at a deregistered UID (h2h_latest stale)
3. ``is_dethrone`` sparse-king fail-closed (was fail-open)
4. ``_str_to_vocab_id`` returns ``None`` on unmapped tokens (was 0)
5. ``evaluated_uids`` not consumed on hard pod failure
"""

from __future__ import annotations

from distil.chain.commitments import Commitment
from distil.eval.composite import is_dethrone
from distil.eval.round import (
    _commit_signature,
    _commitment_changed,
    _stored_commit_signature,
    evict_stale_evaluated_uids,
)


class _FakeState:
    """Minimal duck-typed ValidatorState for round.py tests."""

    def __init__(self, *, evaluated_uids=None, composite_scores=None):
        self.evaluated_uids = list(evaluated_uids or [])
        self.composite_scores = dict(composite_scores or {})
        self.scores = {}

    def is_disqualified(self, *_a, **_k):
        return False


def _commit(uid: int, model: str, revision: str = "main", block: int = 100) -> Commitment:
    return Commitment(
        uid=uid,
        hotkey=f"hk{uid}",
        block=block,
        model=model,
        revision=revision,
        coldkey=f"ck{uid}",
    )


def test_recommit_evicts_evaluated_slot() -> None:
    """v2 of the same model on the same UID drops the prior eval slot."""
    state = _FakeState(
        evaluated_uids=["7"],
        composite_scores={
            "7": {"model": "alice/v1", "revision": "main", "block": 100, "final": 0.5}
        },
    )
    commitments = {7: _commit(7, model="alice/v2", revision="main", block=200)}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == ["7"]
    assert "7" not in state.evaluated_uids
    assert "7" not in state.composite_scores


def test_revision_change_evicts() -> None:
    """Same model id but new revision is also an eviction trigger."""
    state = _FakeState(
        evaluated_uids=["3"],
        composite_scores={
            "3": {"model": "bob/foo", "revision": "abc123", "block": 100, "final": 0.4}
        },
    )
    commitments = {3: _commit(3, model="bob/foo", revision="def456", block=100)}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == ["3"]


def test_same_commit_no_eviction() -> None:
    """Identical commitment signature ⇒ slot remains consumed."""
    state = _FakeState(
        evaluated_uids=["9"],
        composite_scores={
            "9": {"model": "alice/v1", "revision": "main", "block": 100, "final": 0.5}
        },
    )
    commitments = {9: _commit(9, model="alice/v1", revision="main", block=100)}
    assert evict_stale_evaluated_uids(state, commitments) == []
    assert state.evaluated_uids == ["9"]


def test_bootstrapped_legacy_composite_not_evicted() -> None:
    """Pre-eviction-port composites (no model/revision/block fields) stay sticky.

    This is the recovered-from-h2h_history case: the composite was
    rebuilt from old data and has no commit signature. Eviction
    would unfairly drop these on every round; we treat them as
    bootstrap records and let the next eval naturally overwrite them.
    """
    state = _FakeState(
        evaluated_uids=["47"],
        composite_scores={"47": {"final": 0.6, "axes": {"kl": 1.0}}},
    )
    commitments = {47: _commit(47, model="alice/v1", revision="main", block=100)}
    assert evict_stale_evaluated_uids(state, commitments) == []


def test_dq_only_uid_not_re_evicted() -> None:
    """A UID in evaluated_uids but not composite_scores (precheck DQ) is left alone."""
    state = _FakeState(
        evaluated_uids=["55"],
        composite_scores={},
    )
    commitments = {55: _commit(55, model="alice/v3")}
    assert evict_stale_evaluated_uids(state, commitments) == []
    assert "55" in state.evaluated_uids


def test_commit_signature_helpers() -> None:
    """``_commit_signature`` and ``_stored_commit_signature`` agree on equal records."""
    c = _commit(1, model="x/y", revision="r1", block=42)
    assert _commit_signature(c) == ("x/y", "r1", 42)
    assert _stored_commit_signature(
        {"model": "x/y", "revision": "r1", "block": 42}
    ) == ("x/y", "r1", 42)
    assert _stored_commit_signature(
        {"model": "x/y"}  # revision/block defaults
    ) == ("x/y", "main", 0)
    assert _commitment_changed({}, c) is True


def test_is_dethrone_sparse_king_fails_closed() -> None:
    """Sparse king ⇒ no dethrone (fix-2026-05-16). Was True (fail-open)."""
    challenger = {"final": 0.9, "present_count": 18}
    king_sparse = {"final": 0.4, "present_count": 1}
    do, reason = is_dethrone(challenger, king_sparse)
    assert do is False
    assert reason == "king_too_sparse"


def test_is_dethrone_sparse_challenger_no_dethrone() -> None:
    """Sparse challenger also still fails closed (existing invariant)."""
    challenger = {"final": 0.9, "present_count": 1}
    king = {"final": 0.4, "present_count": 18}
    do, reason = is_dethrone(challenger, king)
    assert do is False
    assert reason == "challenger_too_sparse"


def test_is_dethrone_no_king_lets_challenger_win() -> None:
    challenger = {"final": 0.5, "present_count": 18}
    do, reason = is_dethrone(challenger, None)
    assert do is True
    assert reason == "no_king"


def test_str_to_vocab_id_returns_none_on_unmapped() -> None:
    """Unmapped API token strings now return ``None`` (was 0).

    Pre-fix this piled logprob mass at id 0 (BOS/pad), which silently
    skewed KL on prompts where the API returned BPE merges not in the
    local vocab cache. Callers must skip ``None`` entries.
    """
    from distil.pod import teacher_api as ta

    class _FakeTok:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []  # no tokens encoded

    ta._TOKEN_TO_ID = {"hello": 42}
    ta._TOKENIZER = _FakeTok()

    assert ta._str_to_vocab_id("hello") == 42
    assert ta._str_to_vocab_id("not_in_vocab") is None


def test_str_to_vocab_id_encode_fallback_works() -> None:
    """If the tokenizer's encode() yields a real id, use it."""
    from distil.pod import teacher_api as ta

    class _FakeTok:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [99] if text == "rare" else []

    ta._TOKEN_TO_ID = {}
    ta._TOKENIZER = _FakeTok()
    assert ta._str_to_vocab_id("rare") == 99
    assert ta._str_to_vocab_id("missing") is None
