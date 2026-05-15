"""Public dashboard routes (the 12 endpoints the frontend actually calls)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from distil.api.external import hf_model_info, tao_price
from distil.state.store import store

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, Any]:
    progress = store.eval_progress()
    return {
        "ok": True,
        "phase": progress.get("phase"),
        "round_id": progress.get("round_id"),
        "updated_at": progress.get("updated_at"),
    }


@router.get("/leaderboard")
def leaderboard() -> dict[str, Any]:
    return store.top4_leaderboard()


@router.get("/scores")
def scores() -> dict[str, Any]:
    return {"scores": store.scores(), "composite_scores": store.composite_scores()}


@router.get("/h2h-latest")
def h2h_latest() -> dict[str, Any]:
    return store.h2h_latest()


@router.get("/h2h-history")
def h2h_history(limit: int = 50) -> list[dict]:
    rows = store.h2h_history()
    return rows[-int(limit) :]


@router.get("/king-history")
def king_history() -> dict[str, Any]:
    rows = store.h2h_history()
    out: list[dict] = []
    last_king: str | None = None
    for r in rows:
        k = r.get("king_after") or r.get("king_name")
        if k != last_king:
            out.append(
                {
                    "king": k,
                    "block": r.get("block"),
                    "ts": r.get("ts"),
                    "reason": r.get("king_reason"),
                }
            )
            last_king = k
    return {"history": out[-50:]}


@router.get("/eval-progress")
def eval_progress() -> dict[str, Any]:
    return store.eval_progress()


@router.get("/metagraph")
def metagraph() -> dict[str, Any]:
    return {"uid_hotkey_map": store.uid_hotkey_map(), "disqualified": store.disqualified()}


@router.get("/price")
def price() -> dict[str, Any]:
    return tao_price()


@router.get("/incidents")
def incidents(tail: int = 100) -> list[dict]:
    return store.incidents(tail=int(tail))


@router.get("/model-info/{owner}/{name}")
def model_info(owner: str, name: str) -> dict[str, Any]:
    info = hf_model_info(f"{owner}/{name}")
    if "error" in info:
        raise HTTPException(status_code=502, detail=info["error"])
    return info


@router.get("/miner/{uid}")
def miner_detail(uid: int) -> dict[str, Any]:
    uid_map = store.uid_hotkey_map() or {}
    hotkey = uid_map.get(str(uid))
    if not hotkey:
        raise HTTPException(status_code=404, detail="uid_not_found")
    composites = store.composite_scores()
    rounds = [
        r
        for r in store.h2h_history()
        if any(s.get("uid") == int(uid) for s in r.get("students") or [])
    ]
    return {
        "uid": int(uid),
        "hotkey": hotkey,
        "disqualified": store.disqualified().get(hotkey, ""),
        "composite": next(
            (c for k, c in composites.items() if (c or {}).get("uid") == int(uid)), None
        ),
        "rounds": rounds[-20:],
    }


# ── New telemetry surface (improvement #5) ──────────────────────────────


@router.get("/telemetry/timings")
def telemetry_timings(window: int = 100) -> dict[str, Any]:
    """Rolling per-bench wall-time + tokens/sec window."""
    rows = store.h2h_history()
    timings: list[dict] = []
    for r in rows[-int(window) :]:
        for t in r.get("per_bench_timing") or []:
            timings.append({**t, "round_block": r.get("block")})
    return {"window": int(window), "rows": timings}


@router.get("/telemetry/overview")
def telemetry_overview() -> dict[str, Any]:
    """High-level overview for the Live tab."""
    progress = store.eval_progress()
    last_eval = store.last_eval()
    return {
        "phase": progress.get("phase"),
        "round_id": progress.get("round_id"),
        "n_prompts": progress.get("n_prompts"),
        "phase_timings": (progress.get("phase_timings") or [])[-20:],
        "updated_at": progress.get("updated_at"),
        "last_eval": last_eval,
    }
