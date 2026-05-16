"""Round planner — pick challengers + king-paired re-eval.

A round always includes the seated king (so we re-anchor reference axes
on the same prompts) plus FIFO-selected challengers. Models that have
never been evaluated come first; among already-evaluated models the
oldest-evaluated is re-tested first. Stale composite-score schema
versions are evicted before scheduling.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from distil.chain.commitments import Commitment
from distil.eval.composite import COMPOSITE_SCHEMA_VERSION
from distil.settings import settings
from distil.state.files import ValidatorState

logger = logging.getLogger("distil.eval.round")

# Per-round challenger cap. Matches the legacy
# ``scripts/validator/single_eval.SINGLE_EVAL_MAX_PER_ROUND`` default
# (10) so the distil cutover doesn't regress how many miners get
# scored per round. Override via env ``DISTIL_MAX_CHALLENGERS_PER_ROUND``
# (or the legacy alias ``SINGLE_EVAL_MAX_PER_ROUND``) without code change.
import os as _os

MAX_CHALLENGERS_PER_ROUND = int(
    _os.environ.get("DISTIL_MAX_CHALLENGERS_PER_ROUND")
    or _os.environ.get("SINGLE_EVAL_MAX_PER_ROUND")
    or 10
)


def _model_key(c: Commitment) -> str:
    return c.key


def evict_stale_composites(state: ValidatorState) -> int:
    """Drop composite_scores rows from older schema versions."""
    drop: list[str] = []
    for k, v in (state.composite_scores or {}).items():
        if not isinstance(v, dict):
            drop.append(k)
            continue
        ver = v.get("version")
        if ver is not None and int(ver) < COMPOSITE_SCHEMA_VERSION:
            drop.append(k)
    for k in drop:
        state.composite_scores.pop(k, None)
    return len(drop)


def select_challengers(
    commitments: dict[int, Commitment],
    state: ValidatorState,
    *,
    king_uid: int | None,
    n: int = MAX_CHALLENGERS_PER_ROUND,
) -> list[Commitment]:
    """Pick challengers for the round (legacy ``single_eval`` semantics).

    A UID is **eligible** when:

      * it isn't the seated king (the king runs as paired re-eval, not
        as a challenger),
      * it isn't disqualified,
      * it isn't already in ``state.composite_scores`` (UID-keyed),
      * it isn't already in ``state.evaluated_uids`` (str-set of UIDs
        whose single-eval slot is spent).

    Eligible UIDs are sorted FIFO by ``commit_block`` (oldest first) and
    capped at ``n``. We deliberately DO NOT pad the round with already-
    evaluated UIDs — that's the legacy "one eval per commitment"
    contract. If only 2 fresh commits exist this round, only 2
    challengers run. Empty result = round is a king-only re-eval.

    Earlier distil revisions padded with re-evals when fewer than ``n``
    new commits existed; that produced phantom "10 challengers per
    round" output and silently re-scored UIDs that had already taken
    their one shot — see the dashboard regression flagged on 2026-05-15.
    """
    evaluated = {str(u) for u in (state.evaluated_uids or [])}
    candidates: list[Commitment] = []
    for uid, c in commitments.items():
        if king_uid is not None and int(uid) == int(king_uid):
            continue
        if state.is_disqualified(c.hotkey, uid=c.uid):
            continue
        uid_str = str(uid)
        if uid_str in state.composite_scores:
            continue
        if uid_str in evaluated:
            continue
        candidates.append(c)
    candidates.sort(key=lambda c: (int(getattr(c, "block", 0) or 0), int(c.uid)))
    return candidates[:n]


def build_round_spec(
    *,
    block: int,
    block_hash: str | None,
    teacher_repo: str,
    reference_repo: str,
    king: Commitment | None,
    challengers: list[Commitment],
) -> dict[str, Any]:
    """JSON-serializable spec uploaded to the GPU pod."""
    students: list[dict[str, Any]] = []
    if king is not None:
        students.append(
            {
                "name": _model_key(king),
                "repo": king.model,
                "revision": king.revision,
                "uid": king.uid,
                "hotkey": king.hotkey,
                "is_king": True,
            }
        )
    for c in challengers:
        students.append(
            {
                "name": _model_key(c),
                "repo": c.model,
                "revision": c.revision,
                "uid": c.uid,
                "hotkey": c.hotkey,
                "is_king": False,
            }
        )
    return {
        "round_id": int(time.time()),
        "block": block,
        "block_hash": block_hash,
        "teacher_repo": teacher_repo,
        "reference_repo": reference_repo,
        "n_prompts": settings.eval_n_prompts,
        "max_new_tokens": settings.eval_max_new_tokens,
        "per_axis_n": settings.eval_per_axis_n,
        "teacher_top_k": settings.teacher_top_k,
        "student_prompt_logprobs": settings.student_prompt_logprobs,
        "students": students,
        "vllm": {
            "max_model_len": settings.vllm_max_model_len,
            "enable_chunked_prefill": settings.vllm_enable_chunked_prefill,
            "gpu_memory_utilization": settings.vllm_gpu_memory_utilization,
            "dtype": settings.vllm_dtype,
            "max_logprobs": settings.vllm_max_logprobs,
        },
    }
