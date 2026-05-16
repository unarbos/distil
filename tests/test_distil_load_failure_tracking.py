"""Regression test for the 2026-05-16 ghost-model-monopoly bug.

Symptom: in round 1778935073 (and the two preceding rounds) the same
10 challenger UIDs (148, 124, 112, 125, 151, 123, 128, 130, 137, 139)
all returned ``kl=None`` / ``worst=None`` and were NOT marked
disqualified. The dashboard's Rounds tab then crashed (separate fix
in 88201c9 → bb4b76b) and every fresh commitment was starved out of
challenger slots because ``select_challengers`` is FIFO by
``commit_block`` and these ghost UIDs always retried first.

Root cause: the pod-side Phase-2 ``_phase_student`` catches
``OSError`` / ``RepositoryNotFoundError`` and writes
``{"name", "uid", "hotkey", "error": "..."}`` to its shard JSON. The
host's ``process_round`` then computed a composite with every axis
None, did NOT call ``record_failure`` (the legacy hook was dead
code), and the audit fix in 88201c9 explicitly preserved the
single-eval slot for any row with ``worst is None``. Together these
two correct-in-isolation behaviours formed a starvation loop.

This test pins the fix:

1.  Three consecutive Phase-2 load failures on the same UID consume
    the slot (``evaluated_uids`` grows) so ``select_challengers``
    stops re-selecting it.
2.  A single failure does NOT consume the slot — transient HF / vLLM
    blips still get retried (audit-fix invariant preserved).
3.  A successful round resets the failure counter to zero — the next
    failure starts the strike count fresh.
4.  Honest re-commitment (``evict_stale_evaluated_uids``) clears the
    failure counter so the new repo gets a clean 3-strikes budget.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from distil.eval.results import process_round
from distil.eval.round import evict_stale_evaluated_uids
from distil.settings import settings
from distil.state.files import ValidatorState


def _state() -> ValidatorState:
    d = Path(tempfile.mkdtemp(prefix="distil-load-fail-"))
    return ValidatorState(state_dir=d)


def _failing_row(name: str, uid: int, hotkey: str, error: str) -> dict:
    """Mimic the shape ``_phase_student`` writes when the model fails to
    load (HF 404, vLLM init crash, OSError "not a valid model
    identifier", etc.). No axis payloads, just identity + error.
    """
    return {
        "name": name,
        "uid": uid,
        "hotkey": hotkey,
        "error": error,
    }


def _good_row(name: str, uid: int, hotkey: str) -> dict:
    """A minimal Phase-2 row that produces a non-None ``composite.worst``.
    We give it one v31 axis with a passing payload — enough for
    ``compute_composite`` to land at least one axis and clear the
    "worst is None" gate that controls slot consumption.
    """
    return {
        "name": name,
        "uid": uid,
        "hotkey": hotkey,
        "kl_global_avg": 0.5,
        "v31_math_gsm_symbolic": {
            "n": 20,
            "correct": 16,
            "pass_frac": 0.8,
            "mean_gen_tokens_correct": 200.0,
        },
    }


GHOST_ERROR = (
    "OSError: slowsnake/kimi-43043 is not a local folder and is not a "
    "valid model identifier listed on 'https://huggingface.co/models'"
)


def test_first_failure_does_not_consume_slot():
    """Transient HF / vLLM blips MUST stay retryable."""
    state = _state()
    pod_results = {
        "slowsnake/kimi-43043@rev1": _failing_row(
            "slowsnake/kimi-43043@rev1", 124, "5E6tg8LEux", GHOST_ERROR
        ),
    }
    process_round(
        state=state,
        pod_results=pod_results,
        king_name=None,
        reference_name=None,
        teacher_name=None,
        block=100,
        block_hash="0xaa",
    )
    assert state.failures.get("124") == 1, (
        f"first load failure must increment strike counter; "
        f"got failures={state.failures}"
    )
    assert "124" not in state.evaluated_uids, (
        "slot must NOT be burned on a single failure (preserves the "
        "audit-fix invariant that transient blips retry next round); "
        f"got evaluated_uids={state.evaluated_uids}"
    )


def test_max_strikes_consumes_slot():
    """After ``max_load_failures`` consecutive failures the slot MUST
    be consumed so the ghost UID stops crowding out fresh commits.
    """
    state = _state()
    row_factory = lambda: {
        "slowsnake/kimi-43043@rev1": _failing_row(
            "slowsnake/kimi-43043@rev1", 124, "5E6tg8LEux", GHOST_ERROR
        ),
    }
    for strike in range(1, settings.max_load_failures + 1):
        process_round(
            state=state,
            pod_results=row_factory(),
            king_name=None,
            reference_name=None,
            teacher_name=None,
            block=100 + strike,
            block_hash=f"0x{strike:02x}",
        )
        assert state.failures.get("124") == strike, (
            f"strike {strike}: counter desync; failures={state.failures}"
        )
    assert "124" in state.evaluated_uids, (
        f"after {settings.max_load_failures} strikes the slot must be "
        f"consumed; got evaluated_uids={state.evaluated_uids}"
    )


def test_success_resets_strike_counter():
    """A successful round (composite.worst lands) MUST reset failures
    to zero so the next blip starts a fresh strike sequence.
    """
    state = _state()
    state.record_failure(124, "slowsnake/kimi-43043@rev1")
    state.record_failure(124, "slowsnake/kimi-43043@rev1")
    assert state.failures.get("124") == 2

    pod_results = {
        "slowsnake/kimi-43043@rev1": _good_row(
            "slowsnake/kimi-43043@rev1", 124, "5E6tg8LEux"
        ),
    }
    process_round(
        state=state,
        pod_results=pod_results,
        king_name=None,
        reference_name=None,
        teacher_name=None,
        block=100,
        block_hash="0xaa",
    )
    assert state.failures.get("124", 0) == 0, (
        f"successful evaluation must reset failures counter; "
        f"got failures={state.failures}"
    )


def test_recommit_evicts_failure_counter():
    """``evict_stale_evaluated_uids`` MUST clear the failure counter
    so a re-commitment to a working repo gets a clean budget.
    """
    from distil.eval.round import Commitment

    state = _state()
    state.failures = {"124": settings.max_load_failures}
    state.evaluated_uids = ["124"]
    state.composite_scores = {
        "124": {
            "model": "slowsnake/kimi-43043",
            "revision": "rev1",
            "block": 100,
            "worst": None,
            "final": None,
        }
    }
    commitments = {
        124: Commitment(
            uid=124,
            hotkey="5E6tg8LEux",
            model="slowsnake/kimi-NEW-CORRECTED-REPO",
            revision="rev2",
            block=200,
        ),
    }
    evicted = evict_stale_evaluated_uids(state, commitments)
    assert "124" in evicted, f"re-commitment must evict; got {evicted}"
    assert state.failures.get("124", 0) == 0, (
        f"re-commitment must reset failures counter so the new repo "
        f"has a fresh 3-strikes budget; got failures={state.failures}"
    )


def test_status_detail_is_surfaced_on_load_failure():
    """The result row must carry a ``status_detail`` string describing
    the load failure so the dashboard can render
    "load_failed (N/M): OSError: ..." instead of a blank ``kl=null``
    row.
    """
    state = _state()
    pod_results = {
        "slowsnake/kimi-43043@rev1": _failing_row(
            "slowsnake/kimi-43043@rev1", 124, "5E6tg8LEux", GHOST_ERROR
        ),
    }
    record = process_round(
        state=state,
        pod_results=pod_results,
        king_name=None,
        reference_name=None,
        teacher_name=None,
        block=100,
        block_hash="0xaa",
    )
    rows = record.get("results") or []
    assert rows, f"process_round produced no results; record={record}"
    row = rows[0]
    sd = row.get("status_detail") or ""
    assert "load_failed" in sd, (
        f"result row missing load_failed status_detail; got: {row}"
    )
    assert "1/" in sd, (
        f"status_detail must include strike count (got: {sd!r})"
    )
