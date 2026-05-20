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
    """Drop ``evaluated_uids`` + ``composite_scores`` rows ONLY when the
    HOTKEY owning a UID-position has rotated. Returns the list of
    evicted UID strings.

    **STRICT one-registration-one-eval invariant** (operator policy,
    2026-05-20):

        Each on-chain registration gets EXACTLY ONE eval. A hotkey is
        a registration. Re-committing on the same hotkey — for any
        reason, including typo'd repos, force-pushes, model swaps,
        revision bumps, mid-eval pod crashes — does NOT earn another
        eval. The slot is owned by the hotkey, period.

        To get a second eval: deregister, re-register with a new
        hotkey, pay the bond again. That's the cost gate.

    The function evicts in two scenarios:

      1. **UID-rotation, ledger present** — the chain hotkey at this
         UID-slot differs from the hotkey recorded in
         ``evaluated_hotkeys`` under the same UID. The prior holder
         deregistered, a new hotkey claimed the slot, and that new
         hotkey is entitled to its own one fair eval. We drop the
         stale ``composite_scores`` row + ``evaluated_uids`` flag so
         ``select_challengers`` can queue the new commitment. The old
         hotkey's ``evaluated_hotkeys`` entry is kept (with its uid
         field now stale) as a historical record — if the old hotkey
         ever re-registers, it stays permanently locked out per the
         one-eval-per-hotkey rule.

      2. **UID-rotation, pre-ledger** — composite row exists but its
         stored ``model@revision`` doesn't match the chain
         commitment, and no ``evaluated_hotkeys`` entry exists for
         THIS UID at all. The composite was written before ledger
         maintenance started AND belongs to a different commitment,
         which strongly implies the prior hotkey holder. Treat as
         rotation and evict.

    Pre-ledger composites where the stored ``model@revision`` DOES
    match the chain commitment (same hotkey, no rotation, just
    written before the ledger was maintained) auto-backfill the
    ledger so the slot is properly locked going forward.

    What we explicitly do NOT evict on (each of these was an exploit
    or operator-policy concession we've now rolled back):

      * Model name change on the same hotkey
        (togetherness ckp2100 → ckp2200 — observed 2026-05-20)
      * Revision change on the same hotkey
        (force-push to a new sha while keeping the repo)
      * commit_block delta on the same hotkey
      * composite_final is None (3-strikes load failure / typo'd repo /
        mid-eval crash). Operator policy: typo your repo, you eat the
        DQ. Register again to retry.

    History
    -------
    2026-05-20 evening (commit 2320310) — the strict policy shipped
    WITHOUT the UID-rotation detection branch. Backfill rule 2 wrote
    ``evaluated_hotkeys[new_hotkey] = stored_composite`` whenever the
    chain hotkey at a recycled UID was new but composite_scores had
    a row from the prior holder — poisoning the ledger for the new
    hotkey and locking it out of challenger selection. Caught in
    Discord by @hotshot9411 / @justesting0996 within ~3h. Fixed by
    adding the prior-hotkey reverse index lookup below.
    """
    evicted: list[str] = []
    composite_scores = state.composite_scores or {}
    evaluated_hotkeys = getattr(state, "evaluated_hotkeys", {}) or {}

    # Reverse index: uid -> [hotkeys in evaluated_hotkeys whose entry
    # carries that uid]. Used to detect UID rotation: if the chain
    # hotkey at a UID is NOT in the ledger but some OTHER hotkey is
    # logged against the same UID, the slot's prior holder
    # deregistered and a new hotkey took over.
    uid_to_prior_hks: dict[int, list[str]] = {}
    for h, e in evaluated_hotkeys.items():
        if not isinstance(e, dict):
            continue
        u = e.get("uid")
        if u is None:
            continue
        try:
            uid_to_prior_hks.setdefault(int(u), []).append(h)
        except (TypeError, ValueError):
            continue

    def _evict_stale_rows(uid_str: str, uid_int: int) -> None:
        try:
            state.evaluated_uids.remove(uid_str)
        except ValueError:
            pass
        try:
            state.composite_scores.pop(uid_str, None)
        except Exception:  # noqa: BLE001 — pop on a non-dict is a logic bug, not a runtime
            pass
        state.reset_failures(uid_int)

    for uid, c in commitments.items():
        uid_str = str(uid)
        in_eu = uid_str in (state.evaluated_uids or [])
        in_cs = uid_str in composite_scores
        if not (in_eu or in_cs):
            continue
        hk = getattr(c, "hotkey", None)
        eh_entry = evaluated_hotkeys.get(hk) if hk else None

        # rule 1: chain hotkey is in the ledger → slot is locked
        # REGARDLESS of the composite_final value. Successful eval, DQ,
        # load-failure exhaustion, mid-eval crash — they all consume
        # the one-and-only eval slot for this hotkey-registration.
        if eh_entry is not None:
            continue

        # rule 2a: UID rotation detected via ledger. Some OTHER hotkey
        # (different from chain_hk) has an evaluated_hotkeys entry
        # pointing at this UID-slot — the prior holder used their
        # slot, the slot now belongs to chain_hk, and chain_hk has
        # never been evaluated. Evict the stale composite/uid rows so
        # the new hotkey enters the queue. Keep the old hotkey's
        # ledger entry intact so it stays locked if it ever
        # re-registers.
        prior_hks = [h for h in uid_to_prior_hks.get(int(uid), []) if h != hk]
        if prior_hks:
            logger.info(
                f"evicting UID {uid_str}: chain hotkey "
                f"{(hk or '?')[:14]}… is new; prior ledger holder(s) "
                f"{[(h or '?')[:14] + '…' for h in prior_hks]} consumed their slot — "
                f"UID rotation"
            )
            _evict_stale_rows(uid_str, int(uid))
            evicted.append(uid_str)
            continue

        stored = composite_scores.get(uid_str)
        if stored is not None:
            # rule 2b vs 2c: composite exists, ledger has no entry for
            # this UID at all. Distinguish "pre-ledger same hotkey
            # (backfill)" from "pre-ledger UID rotation (evict)" using
            # the commitment signature: if stored model@revision
            # matches the chain commitment, the composite belongs to
            # the current hotkey and we backfill the ledger; if it
            # doesn't match, the composite is from a prior holder of
            # this UID-slot and we evict.
            stored_model = stored.get("model")
            stored_rev = stored.get("revision") or "main"
            chain_model = getattr(c, "model", None)
            chain_rev = getattr(c, "revision", None) or "main"
            same_commit = (
                stored_model is not None
                and stored_model == chain_model
                and stored_rev == chain_rev
            )
            if same_commit:
                # rule 2b: pre-ledger composite from the same hotkey —
                # backfill so the slot is properly locked going forward.
                if hk:
                    state.evaluated_hotkeys[hk] = {
                        "uid": int(uid),
                        "model": stored.get("model"),
                        "revision": stored.get("revision") or "main",
                        "coldkey": getattr(c, "coldkey", None),
                        "evaluated_at_block": stored.get("block"),
                        "evaluated_at_ts": stored.get("evaluated_at"),
                        "composite_final": stored.get("final"),
                        "composite_worst": stored.get("worst"),
                        "backfilled_from_composite": True,
                    }
                continue
            # rule 2c: stored commitment differs from chain commitment
            # AND no ledger entry — pre-ledger UID rotation. Evict.
            logger.info(
                f"evicting UID {uid_str}: chain commitment "
                f"({chain_model}@{chain_rev[:12]}) doesn't match stored "
                f"composite ({stored_model}@{stored_rev[:12]}) and ledger "
                f"has no entry for this UID — pre-ledger UID rotation"
            )
            _evict_stale_rows(uid_str, int(uid))
            evicted.append(uid_str)
            continue

        # rule 3: no ledger entry AND no composite — orphan
        # evaluated_uids row (composite was deleted but the flag
        # wasn't). Genuine hotkey rotation with no scoring history.
        # Drop the orphan flag so the new hotkey gets queued.
        logger.info(
            f"evicting UID {uid_str}: chain hotkey "
            f"{(hk or '?')[:14]}… has no evaluated_hotkeys entry and "
            f"no composite row — orphan evaluated_uids flag"
        )
        _evict_stale_rows(uid_str, int(uid))
        evicted.append(uid_str)
    if evicted:
        logger.info(
            f"evicted {len(evicted)} UIDs (hotkey rotation): {evicted}"
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
        whose single-eval slot is spent),
      * its ``model@revision`` doesn't collide with the king or another
        already-accepted challenger in this round (the older committer
        wins, FIFO).

    Eligible UIDs are sorted FIFO by ``commit_block`` (oldest first) and
    capped at ``n``. We deliberately DO NOT pad the round with already-
    evaluated UIDs — that's the legacy "one eval per commitment"
    contract. If only 2 fresh commits exist this round, only 2
    challengers run. Empty result = round is a king-only re-eval.

    Earlier distil revisions padded with re-evals when fewer than ``n``
    new commits existed; that produced phantom "10 challengers per
    round" output and silently re-scored UIDs that had already taken
    their one shot — see the dashboard regression flagged on 2026-05-15.

    Same-``model@revision`` dedup (added 2026-05-19) closes the
    uid_index collision exploit: when two miners commit the exact same
    repo@revision, the pod produces a single result row keyed by name
    and ``uid_index[name]`` resolves to one of the two UIDs (the king-
    first pass keeps the king authoritative for the king's row, but
    challenger-vs-challenger collisions still attribute the result to
    whichever student was iterated first in the spec). The cleaner
    answer is to not let both UIDs into the round at all — the FIFO
    seeded with the king's key keeps the seated king's identity intact
    and only the OLDEST committer of any duplicate model gets the
    eval slot. Same-model challengers committed AFTER the FIFO winner
    keep their on-chain commitment but stay queued until they either
    push a genuinely-new commit or the FIFO winner gets re-evaluated
    and drops out.
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

    # Reserve the seated king's ``model@revision`` so a copycat that
    # committed the same repo can't enter the round as a challenger
    # and clobber the king's uid_index row. Without this guard the
    # bug observed live on 2026-05-19 (blocks 8219156 / 8219553 /
    # 8220027) fires every time a fresh copycat commitment lands —
    # the dethrone gate uses ``state.king`` so the LIVE seat is
    # fine, but the END-of-round writeback stamps the colliding
    # UID into ``h2h_latest.king_uid`` and the next round resolves
    # the seat to that UID. See the king-first pass in
    # ``service._build_uid_index`` for the secondary safety net.
    seen_keys: set[str] = set()
    king_c = commitments.get(int(king_uid)) if king_uid is not None else None
    if king_c is not None and getattr(king_c, "key", None):
        seen_keys.add(king_c.key)

    # Per-coldkey cap (added 2026-05-20 in response to the togetherness
    # sybil swarm — 13 hotkeys all under the same coldkey filling every
    # challenger slot for a round with checkpoint variants of the same
    # base model). Hard ceiling of ``MAX_PER_COLDKEY`` distinct UIDs
    # from any one coldkey in any single round. Doesn't block
    # registration (that's bittensor's gate) — only stops one
    # coldkey from monopolizing scoring bandwidth.
    MAX_PER_COLDKEY = 2
    coldkey_counts: dict[str, int] = {}
    if king_c is not None and getattr(king_c, "coldkey", None):
        coldkey_counts[king_c.coldkey] = coldkey_counts.get(king_c.coldkey, 0) + 1

    def _coldkey_cap_blocks(c: "Commitment") -> bool:
        ck = getattr(c, "coldkey", None)
        if not ck:
            return False
        return coldkey_counts.get(ck, 0) >= MAX_PER_COLDKEY

    if skip_hf_check:
        # Test-mode shortcut: still dedup by ``model@revision`` AND
        # apply the per-coldkey cap so the selection contract is
        # identical under tests + production.
        accepted_fast: list[Commitment] = []
        for c in candidates:
            if len(accepted_fast) >= n:
                break
            key = getattr(c, "key", None)
            if key and key in seen_keys:
                logger.info(
                    f"select_challengers: skipping uid={c.uid} — duplicate "
                    f"model@revision {key!r} already in round"
                )
                continue
            if _coldkey_cap_blocks(c):
                logger.info(
                    f"select_challengers: skipping uid={c.uid} — per-coldkey "
                    f"cap reached for coldkey {(c.coldkey or '?')[:14]}…"
                )
                continue
            if key:
                seen_keys.add(key)
            ck = getattr(c, "coldkey", None)
            if ck:
                coldkey_counts[ck] = coldkey_counts.get(ck, 0) + 1
            accepted_fast.append(c)
        return accepted_fast

    # HF preflight: walk the FIFO list HEAD-checking config.json on each
    # candidate's repo@revision. Skip any that 404 — they're ghost
    # commitments (deleted repo, typo, never uploaded) and would burn a
    # full round's teacher tokens + GPU time to find out the same thing.
    # The 3-strikes counter in ``process_round`` is the long-term gate
    # for transient blips; this is the cheap up-front gate for
    # permanently-missing models.
    accepted: list[Commitment] = []
    skipped: list[tuple[int, str, str]] = []
    duplicates: list[tuple[int, str]] = []
    coldkey_capped: list[tuple[int, str]] = []
    for c in candidates:
        if len(accepted) >= n:
            break
        key = getattr(c, "key", None)
        if key and key in seen_keys:
            duplicates.append((int(c.uid), key))
            continue
        if _coldkey_cap_blocks(c):
            coldkey_capped.append((int(c.uid), getattr(c, "coldkey", "?") or "?"))
            continue
        ok, reason = _hf_repo_reachable(c.model, getattr(c, "revision", None))
        if ok:
            if key:
                seen_keys.add(key)
            ck = getattr(c, "coldkey", None)
            if ck:
                coldkey_counts[ck] = coldkey_counts.get(ck, 0) + 1
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
    if duplicates:
        logger.warning(
            f"select_challengers: skipped {len(duplicates)} duplicate-model "
            f"candidate(s) (same model@revision as king or earlier accepted "
            f"challenger): {duplicates}"
        )
    if coldkey_capped:
        logger.warning(
            f"select_challengers: skipped {len(coldkey_capped)} candidate(s) "
            f"at per-coldkey cap (max {MAX_PER_COLDKEY} per coldkey/round): "
            f"{coldkey_capped}"
        )
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
