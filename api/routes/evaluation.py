"""Evaluation endpoints: H2H, leaderboard, eval progress, history, benchmarks, announcements."""

import json
import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import STATE_DIR
from helpers.cache import _get_stale
from helpers.sanitize import _sanitize_floats, _safe_json_load

router = APIRouter()


@router.get("/api/leaderboard", tags=["Evaluation"], summary="Top-4 leaderboard",
         description="Returns the top-4 leaderboard - current king and contenders. Dethronement uses paired t-test (p < 0.05).")
def get_leaderboard():
    top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {}) or {}
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_hotkey_map.json"), {})
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    cumulative = _safe_json_load(os.path.join(STATE_DIR, "cumulative_scores.json"), {})

    def _enrich(entry):
        """Fill in model name and KL from live state if missing."""
        if not entry:
            return entry
        uid = entry.get("uid")
        if uid is None:
            return entry
        # Model name
        if not entry.get("model"):
            hotkey = uid_map.get(str(uid))
            if hotkey and hotkey in commitments:
                c = commitments[hotkey]
                entry["model"] = c.get("model") or c.get("repo")
        # KL score
        if not entry.get("h2h_kl") and str(uid) in scores:
            entry["h2h_kl"] = scores[str(uid)]
        # Cumulative score
        cum = cumulative.get(str(uid))
        if cum and isinstance(cum, dict):
            entry["cumulative_score"] = cum.get("cumulative_kl_diff")
            entry["cumulative_rounds"] = cum.get("rounds")
        return entry

    king_data = dict(top4.get("king") or {}) if top4.get("king") else None
    # Override king from h2h_latest if top4 is stale
    if h2h_latest.get("king_uid") is not None:
        if not king_data or king_data.get("uid") != h2h_latest["king_uid"]:
            king_data = {"uid": h2h_latest["king_uid"], "kl": h2h_latest.get("king_kl")}
    king_data = _enrich(king_data)

    contenders = [_enrich(dict(c)) for c in (top4.get("contenders") or [])]
    # Filter out reference model (UID -1) and king from contenders
    king_uid = king_data.get("uid") if king_data else None
    contenders = [c for c in contenders if c.get("uid") not in (-1, king_uid)]
    # If contenders are empty or stale, rebuild from scores
    if not contenders or not any(c.get("h2h_kl") for c in contenders):
        scored = [(int(uid), kl) for uid, kl in scores.items()
                  if int(uid) not in (-1, king_uid or -999) and kl is not None]
        scored.sort(key=lambda x: x[1])
        contenders = [_enrich({"uid": uid, "h2h_kl": kl}) for uid, kl in scored[:4]]

    leaderboard = {
        "king": king_data,
        "contenders": contenders,
        "phase": top4.get("phase", "unknown"),
        "initial_eval_complete": top4.get("initial_eval_complete", False),
        "completed_at": top4.get("completed_at"),
    }

    # Reference model baseline (Qwen3.5-4B, UID -1)
    ref_kl = None
    for r in h2h_latest.get("results", []):
        if r.get("uid") == -1:
            ref_kl = r.get("kl")
            break

    return JSONResponse(
        content=_sanitize_floats({
            "leaderboard": leaderboard,
            "phase": leaderboard["phase"],
            "reference_baseline": {
                "model": "Qwen/Qwen3.5-4B",
                "kl": ref_kl,
                "description": "Undistilled base model (no training)",
            } if ref_kl else None,
        }),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


# ── Announcement helpers ──────────────────────────────────────────────────────

def _is_announcement_claimed(ann: dict) -> bool:
    """Check if an announcement has already been claimed, using the claims log."""
    claims_path = os.path.join(STATE_DIR, "announcement_claims.json")
    claims = _safe_json_load(claims_path, [])
    ann_ts = ann.get("timestamp", 0)
    ann_type = ann.get("type", "")
    for claim in claims:
        if claim.get("timestamp") == ann_ts and claim.get("type") == ann_type:
            return True
    return False


def _record_announcement_claim(ann: dict):
    """Record that an announcement was claimed, in a separate file rsync won't overwrite."""
    claims_path = os.path.join(STATE_DIR, "announcement_claims.json")
    claims = _safe_json_load(claims_path, [])
    claims.append({
        "timestamp": ann.get("timestamp", 0),
        "type": ann.get("type", ""),
        "claimed_at": time.time(),
    })
    # Keep only last 50 claims
    claims = claims[-50:]
    with open(claims_path, "w") as f:
        json.dump(claims, f, indent=2)


@router.get("/api/announcement", tags=["Evaluation"], summary="Pending announcements",
         description="Returns pending announcements (e.g., new king crowned). Returns `{type: null}` if none pending.")
def get_announcement():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True) and not _is_announcement_claimed(ann):
                return ann
        except Exception:
            pass
    return {"type": None}


@router.post("/api/announcement/claim", tags=["Evaluation"], summary="Claim pending announcement",
          description="Atomically reads and marks an announcement as posted. Returns the announcement content, or `{type: null}` if none pending. "
                      "Uses a claims log to prevent re-posting after rsync overwrites.")
def claim_announcement():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True) and not _is_announcement_claimed(ann):
                # Record the claim FIRST (idempotent protection)
                _record_announcement_claim(ann)
                # Also mark posted in the file (best effort - rsync may overwrite)
                ann["posted"] = True
                with open(ann_path, "w") as f:
                    json.dump(ann, f, indent=2)
                return ann
        except Exception:
            pass
    return {"type": None}


@router.post("/api/announcement/posted", tags=["Evaluation"], summary="Mark announcement as posted",
          description="Marks the current announcement as posted. Legacy endpoint - prefer `/api/announcement/claim`.")
def mark_announcement_posted():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            _record_announcement_claim(ann)
            ann["posted"] = True
            with open(ann_path, "w") as f:
                json.dump(ann, f, indent=2)
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}
    return {"ok": True, "note": "no announcement"}


@router.get("/api/eval-progress", tags=["Evaluation"], summary="Live evaluation progress",
         description="""Shows what the validator is currently doing in real-time.

When `active: true`, the response includes:
- `phase`: Current eval phase (e.g. `teacher_generation`, `student_eval`)
- `students_total`: How many miners are being evaluated
- `completed[]`: UIDs that have finished this round
- `current`: Details on the student being evaluated right now (name, prompts done, running KL mean)
- `prompts_total`: Total prompts in this round

When `active: false`, the validator is idle between rounds.
""")
def get_eval_progress():
    progress_path = os.path.join(STATE_DIR, "eval_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": False}


@router.get("/api/h2h-latest", tags=["Evaluation"], summary="Latest head-to-head round",
         description="""Returns results from the most recent evaluation round where miners compete against the king.

Response includes:
- `block`: Block when this round was scored
- `king_uid`: Current king's UID
- `king_h2h_kl`: King's KL score in this round
- `king_global_kl`: King's smoothed global KL
- `p_value`: Paired t-test p-value for the challenger vs king comparison
- `n_prompts`: Number of prompts used
- `results[]`: Array of `{uid, model, kl, is_king, vs_king}` for each evaluated miner
- `king_changed`: Whether the king was dethroned this round (requires p < 0.05)
""")
def get_h2h_latest():
    path = os.path.join(STATE_DIR, "h2h_latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return _sanitize_floats(data)
        except Exception:
            pass
    return {"error": "No H2H data yet"}


@router.get("/api/h2h-history", tags=["Evaluation"], summary="Head-to-head round history",
         description="Returns evaluation rounds with pagination. Supports `?limit=N` (default 50, max 200) and `?page=N` (1-indexed, default 1). Returns newest rounds first when paginated.")
def get_h2h_history(limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            # Annotate rounds with exploit info and extract t-test data to top level
            for entry in data:
                if entry.get("king_changed") and entry.get("new_king_uid"):
                    result_uids = [r.get("uid") for r in entry.get("results", [])]
                    if entry["new_king_uid"] not in result_uids:
                        entry["_exploit"] = True
                        entry["_exploit_note"] = "King promoted from cached scores without evaluation this round (fixed in 579b17b)"
                # Extract best challenger t-test info to top level for dashboard
                if entry.get("t_stat") is None:
                    best_tt = None
                    for r in entry.get("results", []):
                        tt = r.get("t_test")
                        if (
                            tt and isinstance(tt, dict) and tt.get("p") is not None
                            and r.get("dethrone_eligible", True)
                        ):
                            if best_tt is None or tt["p"] < best_tt["p"]:
                                best_tt = tt
                    if best_tt:
                        entry["t_stat"] = best_tt.get("t")
                        entry["p_value"] = best_tt.get("p")
                        entry["t_test_n"] = best_tt.get("n")
                        entry["t_test_mean_delta"] = best_tt.get("mean_delta")
            total = len(data)
            # Reverse so newest first, then paginate
            data_rev = list(reversed(data))
            start = (page - 1) * limit
            end = start + limit
            page_data = data_rev[start:end]
            return JSONResponse(
                content=_sanitize_floats({"rounds": page_data, "total": total, "page": page, "limit": limit, "has_more": end < total}),
                headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
            )
        except Exception:
            pass
    return JSONResponse(
        content={"rounds": [], "total": 0, "page": 1, "limit": limit, "has_more": False},
        headers={"Cache-Control": "public, max-age=10"},
    )


@router.get("/api/king-history", tags=["Evaluation"], summary="King dethronement history",
         description="Returns the chain of king changes (dethronements). Each entry shows the block, new king, and the dethroned UID with margin of victory.")
def get_king_history():
    """Extract all king changes from h2h_history.json."""
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if not os.path.exists(path):
        return JSONResponse(content=[], headers={"Cache-Control": "public, max-age=10"})
    try:
        with open(path) as f:
            history = json.load(f)
    except Exception:
        return JSONResponse(content=[], headers={"Cache-Control": "public, max-age=10"})

    changes = []
    for entry in history:
        if not entry.get("king_changed"):
            continue
        new_king_uid = entry.get("new_king_uid") or entry.get("king_uid")
        prev_king_uid = entry.get("prev_king_uid")
        new_king_model = entry.get("king_model")
        new_king_kl = entry.get("king_h2h_kl") or entry.get("king_kl")
        old_king_model = None
        old_king_kl = None
        winning_result = None
        for r in entry.get("results", []):
            if r.get("uid") == new_king_uid:
                winning_result = r
                new_king_model = r.get("model") or new_king_model
                new_king_kl = r.get("kl", new_king_kl)
            if r.get("uid") == prev_king_uid:
                old_king_model = r.get("model")
                old_king_kl = r.get("kl")
        margin = None
        if new_king_kl is not None and old_king_kl is not None and old_king_kl > 0:
            margin = round((old_king_kl - new_king_kl) / old_king_kl, 6)
        # Detect exploit rounds: new king not in results
        result_uids = [r.get("uid") for r in entry.get("results", [])]
        is_exploit = new_king_uid not in result_uids and entry.get("king_changed")
        change = {
            "block": entry.get("block"),
            "timestamp": entry.get("timestamp"),
            "old_king_uid": prev_king_uid,
            "new_king_uid": new_king_uid,
            "old_king_model": old_king_model,
            "new_king_model": new_king_model,
            "old_king_kl": old_king_kl,
            "new_king_kl": new_king_kl,
            "n_prompts": entry.get("n_prompts"),
            "paired_prompts": (winning_result or {}).get("paired_prompts"),
            "prompts_scored": (winning_result or {}).get("prompts_scored"),
            "dethrone_eligible": (winning_result or {}).get("dethrone_eligible"),
            "p_value": ((winning_result or {}).get("t_test") or {}).get("p"),
            "margin": margin,
        }
        if is_exploit:
            change["_exploit"] = True
        changes.append(change)
    changes.reverse()
    return JSONResponse(
        content=_sanitize_floats(changes),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/eval-stats", tags=["Evaluation"], summary="Eval round statistics",
         description="""Returns statistics about recent evaluation rounds including timing, model counts, and KL trends.

Useful for monitoring eval pipeline health and performance over time.
""")
def get_eval_stats():
    h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    if not h2h_history:
        return JSONResponse(content={"rounds": 0}, headers={"Cache-Control": "public, max-age=30"})

    recent = h2h_history[-20:]  # last 20 rounds
    timings = [r.get("elapsed_seconds") for r in recent if r.get("elapsed_seconds")]
    student_counts = [r.get("n_students") or len(r.get("results", [])) for r in recent]
    king_kls = [r.get("king_kl") for r in recent if r.get("king_kl")]
    dethronements = sum(1 for r in recent if r.get("king_changed"))

    # Time between rounds
    timestamps = [r.get("timestamp") for r in recent if r.get("timestamp")]
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1) if timestamps[i+1] > timestamps[i]]

    stats = {
        "total_rounds": len(h2h_history),
        "recent_rounds": len(recent),
        "dethronements_recent": dethronements,
        "timing": {
            "avg_seconds": round(sum(timings) / len(timings), 1) if timings else None,
            "min_seconds": round(min(timings), 1) if timings else None,
            "max_seconds": round(max(timings), 1) if timings else None,
            "rounds_with_timing": len(timings),
        },
        "models_per_round": {
            "avg": round(sum(student_counts) / len(student_counts), 1) if student_counts else None,
            "min": min(student_counts) if student_counts else None,
            "max": max(student_counts) if student_counts else None,
        },
        "king_kl_trend": [round(kl, 6) for kl in king_kls],
        "round_interval": {
            "avg_minutes": round(sum(intervals) / len(intervals) / 60, 1) if intervals else None,
            "min_minutes": round(min(intervals) / 60, 1) if intervals else None,
            "max_minutes": round(max(intervals) / 60, 1) if intervals else None,
        },
        "last_round": {
            "block": recent[-1].get("block"),
            "timestamp": recent[-1].get("timestamp"),
            "king_uid": recent[-1].get("king_uid"),
            "elapsed_seconds": recent[-1].get("elapsed_seconds"),
            "n_students": recent[-1].get("n_students") or len(recent[-1].get("results", [])),
        },
    }
    return JSONResponse(
        content=_sanitize_floats(stats),
        headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"},
    )


@router.get("/api/eval-status", tags=["Evaluation"], summary="Eval status for all miners",
         description="""Returns why each miner is or isn't being evaluated.
Statuses: king, queued, tested, stale, untested, disqualified.""")
def get_eval_status():
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
    h2h_tracker = _safe_json_load(os.path.join(STATE_DIR, "h2h_tested_against_king.json"), {})
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    current_king_uid = h2h_latest.get("king_uid")
    current_block = h2h_latest.get("block", 0)
    stale_threshold = 50

    result = {}
    for uid_str in scores:
        if uid_str in dq:
            result[uid_str] = {"status": "disqualified"}
            continue
        if current_king_uid is not None and int(uid_str) == current_king_uid:
            result[uid_str] = {"status": "king"}
            continue
        tracker_entry = h2h_tracker.get(uid_str, {})
        if tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
            last_block = tracker_entry["block"]
            epochs_since = (current_block - last_block) // 360 if current_block > last_block else 0
            if epochs_since < stale_threshold:
                result[uid_str] = {"status": "tested", "epochs_ago": epochs_since}
            else:
                result[uid_str] = {"status": "stale", "epochs_ago": epochs_since}
        else:
            result[uid_str] = {"status": "untested"}
    return JSONResponse(
        content={"king_uid": current_king_uid, "block": current_block, "statuses": result},
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/history", tags=["Evaluation"], summary="Score history over time",
         description="Returns historical KL scores for all miners over time. Supports `?limit=N` (default 50) to return only the latest N entries. Response includes `full_eval_block` if a full eval round exists.")
def get_history(limit: int = 50):
    limit = max(1, min(limit, 500))
    history_path = os.path.join(STATE_DIR, "score_history.json")
    entries = []
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                data = json.load(f)
            entries = data[-limit:] if len(data) > limit else data
        except Exception:
            pass

    # Find full_eval block from h2h_history.
    full_eval_block = None
    h2h_path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(h2h_path):
        try:
            with open(h2h_path) as f:
                h2h_data = json.load(f)
            full_eval_round = next((r for r in reversed(h2h_data) if r.get("type") == "full_eval"), None)
            if full_eval_round:
                raw_block = full_eval_round.get("block")
                full_eval_ts = full_eval_round.get("timestamp")
                if isinstance(raw_block, int) and raw_block < 100_000_000:
                    full_eval_block = raw_block
                elif full_eval_ts and entries:
                    nearest = min(entries, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
                    full_eval_block = nearest.get("block")
                elif full_eval_ts and os.path.exists(history_path):
                    with open(history_path) as f:
                        all_history = json.load(f)
                    if all_history:
                        nearest = min(all_history, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
                        full_eval_block = nearest.get("block")
        except Exception:
            pass

    return JSONResponse(
        content={"entries": entries, "full_eval_block": full_eval_block},
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )


@router.get("/api/eval-data", tags=["Evaluation"], summary="Eval data (prompts + completions)",
         description="Returns eval round data. Use `?list=true` for available files, or `?file=<name>` for a specific round.")
def get_eval_data(list: bool = False, file: str = None):
    data_dir = os.path.join(STATE_DIR, "eval_data")
    latest = os.path.join(STATE_DIR, "eval_data_latest.json")
    if list:
        if not os.path.exists(data_dir):
            return {"files": []}
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")], reverse=True)
        return {"files": files, "count": len(files)}
    if file:
        safe_name = os.path.basename(file)
        path = os.path.join(data_dir, safe_name)
        if not os.path.exists(path):
            return JSONResponse(content={"error": "File not found"}, status_code=404)
    else:
        path = latest
    if not os.path.exists(path):
        return JSONResponse(content={"error": "No eval data available"}, status_code=404)
    try:
        with open(path) as f:
            return JSONResponse(content=json.load(f), headers={"Cache-Control": "public, max-age=60"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/benchmarks", tags=["Evaluation"], summary="Benchmark results for king models",
         description="Returns benchmark scores for evaluated king models. Scores are from lm-eval-harness full eval sets.")
def get_benchmarks():
    benchmarks_dir = os.path.join(STATE_DIR, "benchmarks")
    if not os.path.exists(benchmarks_dir):
        return JSONResponse(content={"models": [], "baseline": None}, headers={"Cache-Control": "public, max-age=60"})
    models = []
    baseline = None
    for fname in sorted(os.listdir(benchmarks_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(benchmarks_dir, fname)) as f:
                data = json.load(f)
            if data.get("is_baseline"):
                baseline = data
            else:
                models.append(data)
        except Exception:
            pass
    return JSONResponse(
        content=_sanitize_floats({"models": models, "baseline": baseline}),
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )
