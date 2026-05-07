import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "api"
for path in (str(API_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import state_store  # noqa: E402
from routes import evaluation  # noqa: E402


def test_normalize_eval_progress_surfaces_effective_prompt_total(monkeypatch):
    monkeypatch.setattr(state_store.time, "time", lambda: 2000.0)

    progress = state_store.normalize_eval_progress({
        "active": True,
        "phase": "loading_student",
        "started_at": 1000.0,
        "students_total": 10,
        "prompts_total": 300,
        "teacher_prompts_done": 182,
        "completed": [
            {"student_name": "talent-richer/hope_king", "status": "scored"},
            {"student_name": "talent-richer/wahaha-3", "status": "scored"},
        ],
        "current": {
            "student_name": "william97127/Top9",
            "stage": "chat_turns_probe",
        },
        "pod": {"prompts_total": 182, "teacher_prompts_done": 182},
    })

    assert progress["students_done"] == 2
    assert progress["progress_fraction"] == 0.2
    assert progress["effective_prompts_total"] == 182
    assert progress["prompts_total"] == 300
    assert progress["phase_detail"] == (
        "student scoring 2/10 (william97127/Top9: chat_turns_probe)"
    )


def test_normalize_eval_progress_recomputes_stale_derived_fields(monkeypatch):
    monkeypatch.setattr(state_store.time, "time", lambda: 1120.0)

    progress = state_store.normalize_eval_progress({
        "active": True,
        "phase": "api_generating",
        "started_at": 1000.0,
        "teacher_started_at": 1060.0,
        "teacher_prompts_done": 30,
        "prompts_total": 300,
        "phase_detail": "stale old phase",
        "progress_fraction": 0.99,
        "phase_eta_s": 99999,
    })

    assert progress["phase_detail"] == "teacher API generation 30/300"
    assert progress["progress_fraction"] == 0.1
    assert progress["teacher_prompts_per_min"] == 30.0
    assert progress["phase_eta_s"] == 540.0


def test_normalize_eval_progress_uses_student_elapsed_for_eta(monkeypatch):
    monkeypatch.setattr(state_store.time, "time", lambda: 2000.0)

    progress = state_store.normalize_eval_progress({
        "active": True,
        "phase": "loading_student",
        "students_total": 4,
        "completed": [
            {"student_name": "a", "elapsed_s": 600},
            {"student_name": "b", "started_at": 1200, "finished_at": 2100},
        ],
        "current": {"student_name": "c", "stage": "bench_battery:math_bench"},
    })

    assert progress["students_done"] == 2
    assert progress["phase_detail"] == "student scoring 2/4 (c: bench_battery:math_bench)"
    assert progress["phase_eta_s"] == 1500.0


def test_queue_marks_pod_completed_dicts_and_current_student(monkeypatch):
    monkeypatch.setattr(evaluation, "eval_progress", lambda: {
        "active": True,
        "phase": "loading_student",
        "students_total": 3,
        "prompts_total": 300,
        "completed": [
            {"student_name": "talent-richer/hope_king", "status": "scored"},
        ],
        "current": {
            "student_name": "william97127/Top9",
            "stage": "chat_turns_probe",
        },
        "eval_order": [
            {"uid": 221, "model": "talent-richer/hope_king", "role": "king"},
            {"uid": 219, "model": "william97127/Top9", "role": "challenger"},
            {"uid": 167, "model": "best26/sn97-vs-v3", "role": "challenger"},
        ],
        "pod": {"prompts_total": 182, "teacher_prompts_done": 182},
    })
    monkeypatch.setattr(evaluation, "current_round", lambda: {"models_to_eval": {}})
    monkeypatch.setattr(evaluation, "top4_leaderboard", lambda: {"contenders": []})
    monkeypatch.setattr(evaluation, "read_state", lambda filename, default=None: default or {})

    response = evaluation.get_queue()
    payload = json.loads(response.body)

    assert payload["effective_prompts_total"] == 182
    assert [slot["status"] for slot in payload["slots"]] == [
        "done",
        "running",
        "pending",
    ]


def test_queue_accepts_uid_completion_and_deferred_backlog(monkeypatch):
    monkeypatch.setattr(evaluation, "eval_progress", lambda: {
        "active": True,
        "phase": "loading_student",
        "students_total": 2,
        "students_done": 1,
        "completed": [221],
        "current_model": "william97127/Top9",
        "eval_order": [
            {"uid": 221, "model": "talent-richer/hope_king", "role": "king"},
            {"uid": 219, "model": "william97127/Top9", "role": "challenger"},
        ],
    })
    monkeypatch.setattr(evaluation, "current_round", lambda: {"models_to_eval": {}})
    monkeypatch.setattr(evaluation, "top4_leaderboard", lambda: {"contenders": []})

    def fake_read_state(filename, default=None):
        if filename == "eval_backlog.json":
            return {"pending": [
                {"uid": 167, "model": "best26/sn97-vs-v3", "status": "deferred"},
            ]}
        return default or {}

    monkeypatch.setattr(evaluation, "read_state", fake_read_state)

    payload = json.loads(evaluation.get_queue().body)

    assert [slot["status"] for slot in payload["slots"][:2]] == ["done", "running"]
    assert payload["slots"][2]["uid"] == 167
    assert payload["slots"][2]["status"] == "deferred"
