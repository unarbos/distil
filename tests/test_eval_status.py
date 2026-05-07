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
