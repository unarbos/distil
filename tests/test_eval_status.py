import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "api"
for path in (str(API_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_status import build_eval_statuses, failure_matches_commitment  # noqa: E402


def _no_dq(uid, hotkey, commitment, dq_data):
    return None


def test_failure_matches_commitment_requires_same_revision_when_present():
    assert failure_matches_commitment(
        "owner/model@abc",
        {"model": "owner/model", "revision": "abc"},
    )
    assert not failure_matches_commitment(
        "owner/model@abc",
        {"model": "owner/model", "revision": "def"},
    )
    assert failure_matches_commitment("owner/model", {"model": "owner/model"})


def test_build_eval_statuses_orders_primary_states():
    king_uid, block, statuses = build_eval_statuses(
        scores_data={"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}},
        dq_data={},
        failures_map={"2": 3},
        failure_models_map={"2": "b/model@rev"},
        evaluated_uids=["6"],
        uid_map={
            "1": "hk1",
            "2": "hk2",
            "3": "hk3",
            "4": "hk4",
            "5": "hk5",
            "6": "hk6",
            "7": "hk7",
        },
        commitments={
            "hk2": {"model": "b/model", "revision": "rev"},
            "hk3": {"model": "c/model"},
            "hk4": {"model": "d/model"},
            "hk5": {"model": "e/model"},
            "hk6": {"model": "f/model"},
            "hk7": {"model": "g/model"},
        },
        h2h_tracker={"7": {"king_uid": 1, "block": 90}},
        latest={"king_uid": 1, "block": 100},
        composite_scores={"5": {"final": 0.42, "version": "v1", "ts": 123}},
        progress={
            "phase": "loading_student",
            "current_student": "c/model",
            "eval_order": [
                {"uid": 3, "model": "c/model"},
                {"uid": 4, "model": "d/model"},
            ],
        },
        backlog={"round_cap": 10, "pending": [{"uid": 4, "status": "deferred", "commit_block": 77}]},
        epoch_blocks=5,
        dq_reason_for_commitment=_no_dq,
    )

    assert king_uid == 1
    assert block == 100
    assert statuses["1"]["status"] == "king"
    assert statuses["2"]["status"] == "skipped_stale"
    assert statuses["3"]["status"] == "running"
    assert statuses["4"]["status"] == "queued_active_round"
    assert statuses["5"]["status"] == "scored"
    assert statuses["6"]["status"] == "evaluated_no_composite"
    assert statuses["7"] == {"status": "tested", "epochs_ago": 2}


def test_build_eval_statuses_handles_disqualified_and_no_commitment():
    def dq_reason(uid, hotkey, commitment, dq_data):
        return "bad" if uid == 8 else None

    _king_uid, _block, statuses = build_eval_statuses(
        scores_data={},
        dq_data={},
        failures_map={},
        failure_models_map={},
        evaluated_uids=[],
        uid_map={"8": "hk8", "9": "hk9"},
        commitments={"hk8": {"model": "bad/model"}},
        h2h_tracker={},
        latest={},
        composite_scores={},
        progress={},
        backlog={},
        epoch_blocks=5,
        dq_reason_for_commitment=dq_reason,
    )

    assert statuses["8"] == {"status": "disqualified", "reason": "bad"}
    assert statuses["9"] == {"status": "no_commitment"}


# ── Regression tests for the queued-count bug (Discord 2026-05-20) ───
#
# Before this fix, the dashboard's eval-status route only checked
# ``evaluated_uids.json`` (a UID-list). That list gets cleared whenever
# a UID's composite is evicted by a schema bump, so previously-honest
# evaluations re-appeared as ``status: queued`` — observed live as
# ~60 UIDs flagged queued despite having been scored on their CURRENT
# (model, revision). The fix passes ``evaluated_hotkeys.json`` (the
# authoritative one-eval-per-commit ledger, hotkey-keyed and
# preserved across composite schema bumps) and re-classifies as
# ``evaluated_no_composite`` when the ledger entry matches the
# miner's current commit.


def test_build_eval_statuses_reclassifies_evicted_composite_via_hotkey_ledger():
    """Previously-evaluated UID whose composite row was evicted should
    surface as ``evaluated_no_composite``, NOT ``queued``."""
    _king_uid, _block, statuses = build_eval_statuses(
        scores_data={"10": {}},
        dq_data={},
        failures_map={},
        failure_models_map={},
        evaluated_uids=[],  # cleared by schema bump
        uid_map={"10": "hk10"},
        commitments={"hk10": {"model": "alice/v2", "revision": "abc123"}},
        h2h_tracker={},
        latest={"king_uid": 99, "block": 100},
        composite_scores={},  # evicted by schema bump
        progress={},
        backlog={},
        epoch_blocks=5,
        dq_reason_for_commitment=_no_dq,
        evaluated_hotkeys={
            "hk10": {"model": "alice/v2", "revision": "abc123"}
        },
    )
    assert statuses["10"] == {"status": "evaluated_no_composite"}


def test_build_eval_statuses_keeps_queued_when_hotkey_ledger_has_different_commit():
    """Hotkey was evaluated on an OLDER commit; current commit is new.
    The miner correctly stays as ``queued`` (re-eval candidate)."""
    _king_uid, _block, statuses = build_eval_statuses(
        scores_data={},  # never produced a score for this UID
        dq_data={},
        failures_map={},
        failure_models_map={},
        evaluated_uids=[],
        uid_map={"11": "hk11"},
        commitments={"hk11": {"model": "bob/v3", "revision": "newest"}},
        h2h_tracker={},
        latest={"king_uid": 99, "block": 100},
        composite_scores={},
        progress={},
        backlog={},
        epoch_blocks=5,
        dq_reason_for_commitment=_no_dq,
        evaluated_hotkeys={
            "hk11": {"model": "bob/v3", "revision": "oldrev"}
        },
    )
    assert statuses["11"] == {"status": "queued"}


def test_build_eval_statuses_keeps_queued_when_hotkey_not_in_ledger():
    """Hotkey is fully new — no prior eval. Stays ``queued`` (correctly)."""
    _king_uid, _block, statuses = build_eval_statuses(
        scores_data={},
        dq_data={},
        failures_map={},
        failure_models_map={},
        evaluated_uids=[],
        uid_map={"12": "hk12"},
        commitments={"hk12": {"model": "carol/v1", "revision": "first"}},
        h2h_tracker={},
        latest={"king_uid": 99, "block": 100},
        composite_scores={},
        progress={},
        backlog={},
        epoch_blocks=5,
        dq_reason_for_commitment=_no_dq,
        evaluated_hotkeys={},  # never evaluated
    )
    assert statuses["12"] == {"status": "queued"}


def test_build_eval_statuses_treats_missing_evaluated_hotkeys_as_legacy_behaviour():
    """``evaluated_hotkeys=None`` (older callers) reverts to the legacy
    list-based check; nothing should crash."""
    _king_uid, _block, statuses = build_eval_statuses(
        scores_data={"13": {}},
        dq_data={},
        failures_map={},
        failure_models_map={},
        evaluated_uids=["13"],
        uid_map={"13": "hk13"},
        commitments={"hk13": {"model": "dave/v1"}},
        h2h_tracker={},
        latest={"king_uid": 99, "block": 100},
        composite_scores={},
        progress={},
        backlog={},
        epoch_blocks=5,
        dq_reason_for_commitment=_no_dq,
        # evaluated_hotkeys omitted — must default safely.
    )
    assert statuses["13"] == {"status": "evaluated_no_composite"}
