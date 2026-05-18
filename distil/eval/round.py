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


def _commit_signature(c: Commitment) -> tuple[str, str, int]:
    """``(model, revision, block)`` for the current on-chain commitment."""
    return (
        str(getattr(c, "model", "") or ""),
        str(getattr(c, "revision", "") or "main"),
        int(getattr(c, "block", 0) or 0),
    )


def _stored_commit_signature(record: dict) -> tuple[str, str, int]:
    """Same shape, pulled from a stored composite record."""
    return (
        str(record.get("model") or ""),
        str(record.get("revision") or "main"),
        int(record.get("block") or 0),
    )


def _commitment_changed(stored: dict | None, current: Commitment) -> bool:
    """Match legacy ``scripts/validator/single_eval.commitment_changed``.

    Bootstrap records (no ``block`` AND no ``revision`` stored) compare
    on model name only — that's how distil bootstrapped composites
    recovered from ``h2h_history`` (no per-row commit signature) stay
    valid until they're naturally overwritten.
    """
    if not stored:
        return True
    cur_model, cur_rev, cur_block = _commit_signature(current)
    stored_model = str(stored.get("model") or "")
    if stored_model and stored_model != cur_model:
        return True
    if stored.get("_bootstrapped"):
        return False
    # Pre-eviction-port records (no signature fields at all) also
    # behave as bootstrapped — model-only compare, no revision/block.
    if not stored.get("model") and not stored.get("revision") and not stored.get("block"):
        return False
    stored_rev = str(stored.get("revision") or "main")
    if stored_rev != cur_rev:
        return True
    if stored.get("block") is None:
        return False
    return int(stored.get("block") or 0) != cur_block


def evict_stale_evaluated_uids(
    state: ValidatorState, commitments: dict[int, Commitment]
) -> list[str]:
    """Drop ``evaluated_uids`` + ``composite_scores`` rows whose on-chain
    commitment has moved since the last eval. Returns the list of evicted
    UID strings.

    This is the legacy ``scripts/validator/single_eval.evict_stale_
    evaluated_uids`` invariant ported into distil — without it, a miner
    who pushes v2 of their model on the same UID is starved forever
    because :func:`select_challengers` short-circuits on
    ``uid in state.evaluated_uids``.

    The implementation matches legacy behavior for the three observed
    cases:

    * **honest re-commit** — stored composite has ``model``/``revision``/
      ``block`` fields, all three set; new commitment differs on any
      one of them ⇒ evict.
    * **DQ-only row** — UID is in ``evaluated_uids`` but ``composite_
      scores`` lookup is ``None`` (precheck DQ wrote the DQ row but no
      composite). The DQ row's commit_block is the source of truth for
      retry; we leave the UID consumed here and let the DQ-clear path
      handle re-eval.
    * **bootstrapped legacy row** — pre-2026-05-16 composites recovered
      from ``h2h_history`` (no commit signature). Stay sticky; model-
      name-only compare so a same-UID model-name change still evicts.
    """
    evicted: list[str] = []
    composite_scores = state.composite_scores or {}
    evaluated_hotkeys = getattr(state, "evaluated_hotkeys", {}) or {}
    for uid, c in commitments.items():
        uid_str = str(uid)
        in_eu = uid_str in (state.evaluated_uids or [])
        in_cs = uid_str in composite_scores
        if not (in_eu or in_cs):
            continue
        stored = composite_scores.get(uid_str)
        if stored is None:
            # In ``evaluated_uids`` but no composite row. STRICT
            # one-eval-per-commit invariant: only evict when the
            # CURRENT chain commit has NOT been evaluated against this
            # hotkey before. Use ``evaluated_hotkeys[hk]`` as the
            # source of truth — it persists across composite-schema
            # bumps (which ``evict_stale_composites`` blew away).
            #
            # Two branches eligible for eviction:
            #   1. No ``evaluated_hotkeys`` entry for this hotkey at
            #      all — brand-new hotkey / hotkey rotation; the
            #      ``evaluated_uids`` row is stale book-keeping from
            #      whoever held this UID before.
            #   2. Entry exists but ``(model, revision)`` differs from
            #      the current chain commit — same hotkey pushed a
            #      new model and the composite row was lost to a
            #      schema bump or never written (e.g. orchestrator
            #      crashed between the slot consume and composite
            #      write).
            #
            # Earlier revisions of this function were too aggressive
            # and evicted any ``in_eu && composite=None`` UID
            # regardless of whether the same commit had already been
            # evaluated. That re-queued 41 miners on commits they had
            # already taken their fair eval against and burned ~10x
            # OpenRouter teacher-API spend for no scoring change. See
            # the 2026-05-18 #distil-97 post-mortem.
            hk = getattr(c, "hotkey", None)
            eh_entry = evaluated_hotkeys.get(hk) if hk else None
            if eh_entry:
                eh_model = (eh_entry.get("model") or "").strip().lower()
                eh_rev = (eh_entry.get("revision") or "main").strip().lower()
                cur_model = (getattr(c, "model", "") or "").strip().lower()
                cur_rev = (getattr(c, "revision", "main") or "main").strip().lower()
                # Same commit if model matches and revisions agree on
                # at least their 7-char short SHA prefix (mirrors how
                # bittensor commitments report revisions).
                rev_match = (
                    eh_rev[:7] == cur_rev[:7]
                    or (eh_rev in ("", "main") and cur_rev in ("", "main"))
                )
                if eh_model == cur_model and rev_match:
                    continue  # SAME commit; one-eval-per-commit holds.
                logger.info(
                    f"evicting UID {uid_str} (hotkey {hk!r}): same hotkey "
                    f"pushed new commit {cur_model}@{cur_rev[:7]} "
                    f"(prev was {eh_model}@{eh_rev[:7]})"
                )
            else:
                logger.info(
                    f"evicting UID {uid_str} (hotkey {hk!r}): no prior "
                    f"evaluated_hotkeys entry — new miner / hotkey rotation"
                )
            try:
                state.evaluated_uids.remove(uid_str)
            except ValueError:
                pass
            evicted.append(uid_str)
            continue
        if not _commitment_changed(stored, c):
            continue
        try:
            state.evaluated_uids.remove(uid_str)
        except ValueError:
            pass
        state.composite_scores.pop(uid_str, None)
        # Also reset the load-failure counter: when the miner pushes a
        # fresh commitment they get a clean 3-strikes budget against
        # the new repo. Without this reset a UID that hit 3 strikes on
        # a typo'd repo (e.g. ``slowsnake/kimi-43043`` deleted from HF)
        # would inherit the 3-strike state to their corrected repo and
        # burn their slot on the very first transient HF blip.
        state.reset_failures(int(uid))
        evicted.append(uid_str)
    if evicted:
        logger.info(
            f"evicted {len(evicted)} stale evaluated UIDs (re-committed): {evicted}"
        )
    return evicted


def _hf_repo_reachable(repo: str, revision: str | None) -> tuple[bool, str]:
    """Lightweight HEAD-only check: does ``{repo}@{revision}`` resolve?

    Returns ``(ok, reason)``. ``reason`` is a short failure tag suitable
    for logging/DQ trail ("hf_404", "hf_403", "hf_timeout", ...). On
    network errors we FAIL OPEN — the pod-side load attempt is the
    authoritative gate and a transient ConnectionError on the host
    shouldn't kick out a valid commitment.

    Only used by :func:`select_challengers` to short-circuit ghost
    models (HF 404 / deleted repos) BEFORE they burn ~25 min of teacher
    API tokens + 8×B200 GPU time in Phase 1. The 3-strikes counter in
    ``process_round`` is the long-term gate; this preflight check
    cancels the next strike from happening at all when the model is
    obviously gone.
    """
    import requests

    rev = revision or "main"
    url = f"https://huggingface.co/{repo}/resolve/{rev}/config.json"
    token = settings.hf_dl_token or _os.environ.get("HF_TOKEN") or ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.head(url, allow_redirects=True, timeout=8, headers=headers)
    except Exception as exc:
        return True, f"head_check_skipped:{type(exc).__name__}"
    if r.status_code == 200:
        return True, "ok"
    if r.status_code in (401, 403):
        return False, f"hf_{r.status_code}"
    if r.status_code == 404:
        return False, "hf_404"
    return True, f"head_check_indeterminate:{r.status_code}"


def select_challengers(
    commitments: dict[int, Commitment],
    state: ValidatorState,
    *,
    king_uid: int | None,
    n: int = MAX_CHALLENGERS_PER_ROUND,
    skip_hf_check: bool = False,
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
    # HF preflight: walk the FIFO list HEAD-checking config.json on each
    # candidate's repo@revision. Skip any that 404 — they're ghost
    # commitments (deleted repo, typo, never uploaded) and would burn a
    # full round's teacher tokens + GPU time to find out the same thing.
    # The 3-strikes counter in ``process_round`` is the long-term gate
    # for transient blips; this is the cheap up-front gate for
    # permanently-missing models. ``skip_hf_check`` is provided as an
    # escape hatch for offline tests and the unit suite (where HF
    # network calls would be wrong to make).
    if skip_hf_check:
        return candidates[:n]
    accepted: list[Commitment] = []
    skipped: list[tuple[int, str, str]] = []
    for c in candidates:
        if len(accepted) >= n:
            break
        ok, reason = _hf_repo_reachable(c.model, getattr(c, "revision", None))
        if ok:
            accepted.append(c)
            continue
        # Permanent-failure surface (404 or auth-locked). Record a
        # failure strike + log so the dashboard / Discord audits can
        # see the skipped UID without trawling the pod logs. We don't
        # consume the slot here — the 3-strikes process_round path
        # owns that — but a confirmed 404 obviously rolls the counter
        # forward.
        uid = int(c.uid)
        try:
            n_strikes = state.record_failure(uid, f"{c.model}@{c.revision}")
        except Exception:  # pragma: no cover — state always provides record_failure
            n_strikes = 0
        skipped.append((uid, c.model, f"{reason}/strike={n_strikes}"))
    if skipped:
        logger.warning(
            f"select_challengers: skipped {len(skipped)} candidate(s) failing "
            f"HF preflight: {skipped}"
        )
    return accepted


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
