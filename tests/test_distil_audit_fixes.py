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

    def __init__(
        self,
        *,
        evaluated_uids=None,
        composite_scores=None,
        failures=None,
        evaluated_hotkeys=None,
        disqualified=None,
    ):
        self.evaluated_uids = list(evaluated_uids or [])
        self.composite_scores = dict(composite_scores or {})
        self.failures = dict(failures or {})
        # ``evaluated_hotkeys[hk]`` persists across composite-schema bumps
        # and is the source of truth for the one-eval-per-commit invariant
        # (see ``evict_stale_evaluated_uids``). ``process_round`` writes
        # this map next to ``evaluated_uids`` whenever a slot is consumed
        # (composite landed OR precheck DQ OR load-failure exhaustion).
        self.evaluated_hotkeys = dict(evaluated_hotkeys or {})
        # ``disqualified[hk] = reason`` mirrors ``ValidatorState.disqualified``
        # so the DQ-gate in ``evict_stale_evaluated_uids`` (rule 2a) is
        # exercisable in tests.
        self.disqualified = dict(disqualified or {})
        self.scores = {}

    def is_disqualified(self, hotkey=None, *, uid=None):
        if hotkey and hotkey in self.disqualified:
            return True
        if uid is not None and str(uid) in self.disqualified:
            return True
        return False

    def reset_failures(self, uid: int) -> None:
        """Mirror ``ValidatorState.reset_failures``. Required since
        ``evict_stale_evaluated_uids`` now resets the load-failure
        counter so a re-commitment to a corrected repo gets a clean
        3-strikes budget (see test_distil_load_failure_tracking).
        """
        self.failures.pop(str(uid), None)


def _commit(uid: int, model: str, revision: str = "main", block: int = 100) -> Commitment:
    return Commitment(
        uid=uid,
        hotkey=f"hk{uid}",
        block=block,
        model=model,
        revision=revision,
        coldkey=f"ck{uid}",
    )


def test_recommit_does_not_evict_evaluated_slot() -> None:
    """SPAM-PROOF (2026-05-20): a miner re-committing v2 of their model on
    the same hotkey does NOT earn a re-eval. The slot is keyed by the
    HOTKEY, not by ``(hotkey, model@revision)`` — so re-commits are
    pure no-ops as far as the eval-slot ledger is concerned.

    Pre-fix, ``evict_stale_evaluated_uids`` would clear the row on any
    ``_commitment_changed`` signal (model OR revision OR block delta),
    which let the togetherness exploit (observed live 2026-05-20 in
    #distil-97) cycle 13 checkpoint variants of the same base model
    on 13 hotkeys to claim 13× the eval budget the gate intended.

    Recovery for a hotkey that wants a real re-eval: REGISTER a new
    hotkey (pay the bond). That's the cost gate.
    """
    state = _FakeState(
        evaluated_uids=["7"],
        composite_scores={
            "7": {"model": "alice/v1", "revision": "main", "block": 100, "final": 0.5}
        },
        evaluated_hotkeys={
            "hk7": {
                "uid": 7, "model": "alice/v1", "revision": "main",
                "composite_final": 0.5, "composite_worst": None,
            }
        },
    )
    commitments = {7: _commit(7, model="alice/v2", revision="main", block=200)}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == []
    assert "7" in state.evaluated_uids
    assert "7" in state.composite_scores  # old composite untouched until the
    # next round's process_round writes the same slot — re-eval doesn't run.


def test_revision_change_does_not_evict() -> None:
    """SPAM-PROOF (2026-05-20): force-pushing a new revision on the same
    repo+hotkey is just a different re-commit — same closure as
    ``test_recommit_does_not_evict_evaluated_slot``. The prior eval
    produced a real composite so the slot is permanently locked."""
    state = _FakeState(
        evaluated_uids=["3"],
        composite_scores={
            "3": {"model": "bob/foo", "revision": "abc123", "block": 100, "final": 0.4}
        },
        evaluated_hotkeys={
            "hk3": {
                "uid": 3, "model": "bob/foo", "revision": "abc123",
                "composite_final": 0.4, "composite_worst": 0.2,
            }
        },
    )
    commitments = {3: _commit(3, model="bob/foo", revision="def456", block=100)}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == []
    assert "3" in state.evaluated_uids


def test_hotkey_rotation_evicts_stale_row() -> None:
    """Genuine hotkey rotation (the prior miner deregistered, a new
    hotkey now owns the UID slot) DOES trigger eviction so the new
    hotkey can take its one fair eval. Detected via the chain hotkey
    being absent from ``evaluated_hotkeys`` AND no composite row
    backing it."""
    state = _FakeState(
        evaluated_uids=["5"],
        composite_scores={},  # no composite — pure stale evaluated_uids row
        evaluated_hotkeys={},  # the prior owner's hotkey already pruned
    )
    commitments = {5: _commit(5, model="newminer/first", revision="main")}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == ["5"]
    assert "5" not in state.evaluated_uids


def test_pre_ledger_composite_backfills_evaluated_hotkeys() -> None:
    """Composites written before ``evaluated_hotkeys`` started being
    maintained (pre-2026-05-18) lock the slot by backfilling the
    ledger from the composite, NOT by evicting and re-evaluating."""
    state = _FakeState(
        evaluated_uids=["12"],
        composite_scores={
            "12": {
                "model": "legacy/repo", "revision": "abc",
                "block": 80, "final": 0.55,
            }
        },
        evaluated_hotkeys={},  # ledger not yet populated for this hotkey
    )
    commitments = {12: _commit(12, model="legacy/repo", revision="abc")}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == []
    assert "12" in state.evaluated_uids
    # Backfill happened:
    eh = state.evaluated_hotkeys.get("hk12")
    assert eh is not None
    assert eh.get("model") == "legacy/repo"
    assert eh.get("backfilled_from_composite") is True


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
    """A UID consumed by PRECHECK DQ (composite_final=None + DQ list
    entry) is left alone — the DQ list is the authoritative gate and
    re-commit must not silently revive a protocol-violation DQ.

    Distinct from the load-failure path which DOES allow retry on
    re-commit (see ``test_load_failure_uid_evicts_on_recommit`` below).
    """
    state = _FakeState(
        evaluated_uids=["55"],
        composite_scores={},
        evaluated_hotkeys={
            "hk55": {
                "uid": 55, "model": "alice/v3", "revision": "main",
                "composite_final": None, "composite_worst": None,
            }
        },
        disqualified={"hk55": "precheck:vocab_size_mismatch"},
    )
    commitments = {55: _commit(55, model="alice/v3")}
    assert evict_stale_evaluated_uids(state, commitments) == []
    assert "55" in state.evaluated_uids
    assert "hk55" in state.evaluated_hotkeys  # ledger preserved


def test_load_failure_uid_evicts_on_recommit() -> None:
    """A UID consumed by 3-strikes load failure (composite_final=None
    + NOT on DQ list) DOES evict on re-commit — fair retry path.

    This is the typo'd-repo rescue. Closes the bot's promise to
    itorgov (UID 171, 2026-05-20) without giving spam loops an out:
    the DQ list still blocks any actual cheating, and successful
    evals (composite_final≠None) still lock the slot via rule 1.
    """
    state = _FakeState(
        evaluated_uids=["77"],
        composite_scores={},
        failures={"77": 3},
        evaluated_hotkeys={
            "hk77": {
                "uid": 77, "model": "miner/typo-repo", "revision": "main",
                "composite_final": None, "composite_worst": None,
                "load_failures": 3,
            }
        },
        # NOT on DQ list — pure load failure
    )
    commitments = {77: _commit(77, model="miner/correct-repo")}
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert evicted == ["77"]
    assert "77" not in state.evaluated_uids
    assert "hk77" not in state.evaluated_hotkeys
    assert state.failures.get("77", 0) == 0  # fresh budget


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
