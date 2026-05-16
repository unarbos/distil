"""Public dashboard routes (the 12 endpoints the frontend actually calls)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from distil.api.external import hf_model_info, tao_price
from distil.eval.composite import BENCH_MIN_VALID, V31_AXIS_NAMES
from distil.eval.round import MAX_CHALLENGERS_PER_ROUND
from distil.settings import settings
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
        if any(s.get("uid") == int(uid) for s in (r.get("results") or r.get("students") or []))
    ]
    return {
        "uid": int(uid),
        "hotkey": hotkey,
        "disqualified": store.disqualified().get(hotkey, ""),
        # ``state.composite_scores`` is UID-keyed (``str(uid)`` → row).
        # The stored row does NOT include a ``uid`` field, so a
        # ``(c for k, c in composites.items() if c.get('uid') == int(uid))``
        # generator (the pre-fix code) always returned ``None`` and
        # the miner profile page rendered an empty composite block
        # for every UID. Direct UID-key lookup matches the writer in
        # ``results.process_round``.
        "composite": composites.get(str(uid)),
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


@router.get("/composite-config")
def composite_config() -> dict[str, Any]:
    """Composite-score weights and dethrone thresholds (read-only).

    Surfaces the canonical axis schema, per-axis minimum sample floors,
    final-score blend coefficient, and dethrone gate parameters so
    miners can reproduce ``state.composite_scores`` locally without
    digging through ``distil/eval/composite.py`` source.

    Miners flagged in #distil 2026-05-16 that the dashboard had no
    machine-readable spec for the v32 schema, leading to confusion
    about which axes were weighted vs telemetry, which floors apply,
    and how the worst-3 mean interacts with the weighted mean to
    produce ``final``. Adding this avoids guesswork and lets
    third-party tooling pin against the same definitions the validator
    uses.
    """
    return {
        "schema_version": 32,
        "axes": {
            "core": [
                "on_policy_rkl",
                "kl",
                "top_k_overlap",
                "capability",
                "length",
            ],
            "judge": [
                "judge_probe",
                "long_form_judge",
                "long_gen_coherence",
                "chat_turns_probe",
            ],
            "discipline": [
                "reasoning_density",
                "calibration_bench",
            ],
            "v31_procedural": list(V31_AXIS_NAMES),
        },
        "bench_min_valid": dict(BENCH_MIN_VALID),
        "final_blend": {
            "alpha_bottom": settings.composite_final_bottom_weight,
            "worst_k": settings.worst_3_mean_k,
            "formula": (
                "final = alpha_bottom * worst_K_mean + "
                "(1 - alpha_bottom) * weighted_mean"
            ),
        },
        "dethrone": {
            "margin": settings.composite_dethrone_margin,
            "min_axes": settings.composite_dethrone_min_axes,
            "floor": settings.composite_dethrone_floor,
        },
        "single_eval": {
            "max_per_round": MAX_CHALLENGERS_PER_ROUND,
            "max_load_failures": settings.max_load_failures,
        },
    }
