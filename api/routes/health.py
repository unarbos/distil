"""Health check and root redirect endpoints."""

import os
import subprocess
import time as _time

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from config import NETUID, STATE_DIR
from helpers.sanitize import _safe_json_load

# Compute git revision once at import time
# Try git first, fall back to REVISION file (for non-git deployments)
try:
    _code_revision = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stderr=subprocess.DEVNULL,
        timeout=5,
    ).decode().strip()
except Exception:
    _code_revision = None

if not _code_revision:
    for _rev_path in [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "REVISION"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REVISION"),
    ]:
        try:
            with open(_rev_path) as _f:
                _code_revision = _f.read().strip() or None
            if _code_revision:
                break
        except Exception:
            pass

router = APIRouter()


@router.get("/", include_in_schema=False)
def root():
    """Redirect to interactive API docs."""
    return RedirectResponse(url="/docs")


@router.get("/api/health", tags=["Overview"], summary="Service health and quick status",
         description="""One-stop health check that returns the current state of the validator and subnet.

Response includes:
- `status`: `ok` if the API is running
- `king_uid` / `king_kl`: Current king and their KL score (lower = better)
- `n_scored` / `n_disqualified`: Number of active vs disqualified miners
- `last_eval_block` / `last_eval_age_min`: When the last eval happened
- `eval_active`: Whether an evaluation round is in progress right now
- `eval_progress`: Detailed progress if eval is active (phase, students done, current KL, etc.)

This is the best endpoint to start with - gives you a quick overview of the entire subnet state.
""")
def health():
    last_eval_block = None
    last_eval_age_min = None
    eval_active = False
    king_uid = None
    king_kl = None
    n_scored = 0
    n_dq = 0
    eval_students_done = 0
    eval_students_total = 0
    prog = {}
    try:
        h2h = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
        last_eval_block = h2h.get("block")
        ts = h2h.get("timestamp")
        if ts:
            last_eval_age_min = round((_time.time() - ts) / 60, 1)
        king_uid = h2h.get("king_uid")
        # Get king KL from scores
        scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
        n_scored = len(scores)
        if king_uid and str(king_uid) in scores:
            king_kl = scores[str(king_uid)]
        dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
        n_dq = len(dq)
        prog = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
        eval_active = prog.get("active", False)
        if eval_active:
            eval_students_done = len(prog.get("completed", []))
            eval_students_total = prog.get("students_total", 0)
    except Exception:
        pass
    return {
        "status": "ok",
        "netuid": NETUID,
        "dethrone_method": "paired_t_test",
        "king_uid": king_uid,
        "king_kl": round(king_kl, 6) if king_kl else None,
        "n_scored": n_scored,
        "n_disqualified": n_dq,
        "last_eval_block": last_eval_block,
        "last_eval_age_min": last_eval_age_min,
        "eval_active": eval_active,
        "code_revision": _code_revision,
        "eval_progress": {
            "phase": prog.get("phase"),
            "students_total": prog.get("students_total"),
            "students_done": len(prog.get("completed", [])),
            "prompts_total": prog.get("prompts_total"),
            "current_student": prog.get("current", {}).get("student_name") if isinstance(prog.get("current"), dict) else None,
            "current_prompt": prog.get("current", {}).get("prompts_done") if isinstance(prog.get("current"), dict) else None,
            "current_kl": prog.get("current", {}).get("kl_running_mean") if isinstance(prog.get("current"), dict) else None,
            "current_best": prog.get("current", {}).get("best_kl_so_far") if isinstance(prog.get("current"), dict) else None,
            "teacher_prompts_done": prog.get("teacher_prompts_done"),
        } if eval_active else None,
    }
