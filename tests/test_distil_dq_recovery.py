"""Regression test for the 2026-05-16 ``integrity:HF-404`` DQ
auto-recovery sweeper.

Symptom (aizaysi, UID 233 / ``RLStepone/distil-success-h19``):
legacy precheck disqualified miners whose HF repo briefly 404'd
with a stable ``integrity: Model <repo> no longer exists on
HuggingFace (404)`` reason. The new ``distil/`` package no longer
writes ``integrity:`` DQs at all, but the 36 legacy rows persisted
in ``state.disqualified`` and never cleared — even after the
miner restored the repo and HF returned 200 again.

This sweeper runs at the top of every round; it HEAD-checks each
``integrity:.*404`` DQ and drops the row when the repo is now
reachable. Network/timeout/transient errors fail open so a
transient HF blip never auto-clears a real DQ.

Pinned invariants:

1.  A DQ whose model is now 200 on HF gets cleared.
2.  A DQ whose model is still 404 is left alone.
3.  Non-``integrity:`` DQs are never touched (the sweeper is
    scoped to one historical legacy reason category, not a
    blanket DQ-clearer).
4.  Network errors fail open — no auto-clear on a non-200 response.
5.  Cleared rows return a structured record suitable for logging.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from distil.eval.dq_recovery import sweep_integrity_dq_recoveries
from distil.state.files import ValidatorState


def _state() -> ValidatorState:
    d = Path(tempfile.mkdtemp(prefix="distil-dq-recover-"))
    return ValidatorState(state_dir=d)


@pytest.fixture
def fake_head(monkeypatch):
    """Replace the HF HEAD with a controllable callback."""
    calls: list[str] = []

    def install(responses: dict[str, tuple[bool, int]]):
        def fake(repo: str) -> tuple[bool, int]:
            calls.append(repo)
            return responses.get(repo, (False, 0))

        import distil.eval.dq_recovery as mod
        monkeypatch.setattr(mod, "_hf_head_ok", fake)
        return calls

    return install


def test_clears_restored_repo(fake_head):
    """A 200-on-HF repo with an integrity-404 DQ MUST be cleared."""
    calls = fake_head({"alice/model-restored": (True, 200)})
    state = _state()
    state.disqualified["hk_alice"] = (
        "integrity: Model alice/model-restored no longer exists on HuggingFace (404)"
    )
    cleared = sweep_integrity_dq_recoveries(state)
    assert len(cleared) == 1
    assert cleared[0]["model"] == "alice/model-restored"
    assert "hk_alice" not in state.disqualified, (
        f"DQ row must be removed; got {state.disqualified}"
    )
    assert "alice/model-restored" in calls


def test_leaves_still_404_alone(fake_head):
    """A 404-on-HF repo MUST stay DQ'd."""
    fake_head({"bob/ghost": (False, 404)})
    state = _state()
    reason = (
        "integrity: Model bob/ghost no longer exists on HuggingFace (404)"
    )
    state.disqualified["hk_bob"] = reason
    cleared = sweep_integrity_dq_recoveries(state)
    assert cleared == []
    assert state.disqualified.get("hk_bob") == reason


def test_ignores_non_integrity_dq(fake_head):
    """``arch:`` / ``copy:`` / ``one_eval_per_registration`` / etc.
    DQs MUST be ignored — the sweeper is HF-recovery scoped only.
    """
    # ``fake_head`` would assert if called for the wrong repo; we use
    # a captured-call list to verify HF was never queried.
    calls = fake_head({"any/repo": (True, 200)})
    state = _state()
    state.disqualified["hk_arch"] = "arch: total_params_too_large:2_000_000_000_000>1_500_000_000_000"
    state.disqualified["hk_copy"] = "copy: activation similarity 0.99 vs uid=42"
    state.disqualified["hk_one_eval"] = "one_eval_per_registration"
    cleared = sweep_integrity_dq_recoveries(state)
    assert cleared == []
    assert calls == [], f"HF must not be queried for non-integrity DQs; got {calls}"
    assert len(state.disqualified) == 3, "all 3 non-integrity DQs preserved"


def test_network_error_fails_open(fake_head):
    """A network/timeout error (returns ``(False, 0)``) MUST NOT
    clear a DQ — only a concrete 200 in hand should.
    """
    fake_head({"flaky/repo": (False, 0)})
    state = _state()
    reason = "integrity: Model flaky/repo no longer exists on HuggingFace (404)"
    state.disqualified["hk_flaky"] = reason
    cleared = sweep_integrity_dq_recoveries(state)
    assert cleared == []
    assert state.disqualified.get("hk_flaky") == reason


def test_mixed_batch(fake_head):
    """When a real workload is processed (some restored, some still
    404, some unrelated DQs) the sweeper MUST correctly partition.
    """
    fake_head({
        "good/restored-a": (True, 200),
        "good/restored-b": (True, 200),
        "ghost/dead": (False, 404),
        "flaky/timeout": (False, 0),
    })
    state = _state()
    state.disqualified["hk_a"] = "integrity: Model good/restored-a no longer exists on HuggingFace (404)"
    state.disqualified["hk_b"] = "integrity: Model good/restored-b no longer exists on HuggingFace (404)"
    state.disqualified["hk_dead"] = "integrity: Model ghost/dead no longer exists on HuggingFace (404)"
    state.disqualified["hk_timeout"] = "integrity: Model flaky/timeout no longer exists on HuggingFace (404)"
    state.disqualified["hk_arch"] = "arch: vocab_too_large:300000>256000"
    cleared = sweep_integrity_dq_recoveries(state)
    cleared_models = {c["model"] for c in cleared}
    assert cleared_models == {"good/restored-a", "good/restored-b"}, (
        f"only 200-confirmed restored repos must be cleared; got {cleared_models}"
    )
    assert "hk_a" not in state.disqualified
    assert "hk_b" not in state.disqualified
    assert "hk_dead" in state.disqualified
    assert "hk_timeout" in state.disqualified
    assert "hk_arch" in state.disqualified


def test_handles_shorter_reason_format(fake_head):
    """Older state snapshots used ``integrity: HuggingFace 404 for
    <repo>`` instead of the long form. The sweeper MUST parse both.
    """
    fake_head({"vintage/repo": (True, 200)})
    state = _state()
    state.disqualified["hk_old"] = "integrity: HuggingFace 404 for vintage/repo"
    cleared = sweep_integrity_dq_recoveries(state)
    assert len(cleared) == 1
    assert cleared[0]["model"] == "vintage/repo"
