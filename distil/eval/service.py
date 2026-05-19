"""Validator service loop — fetch chain → eval → score → set_weights.

Each iteration:
  1. Fetch metagraph + commitments (with retry).
  2. Evict stale composite scores; select challengers (FIFO + king re-eval).
  3. Acquire a Lium pod, upload code, run the eval.
  4. Process results into composites + state shards.
  5. Resolve king + dethrone gate; build emission weights.
  6. ``set_weights`` (skip on ``--dry-run``).
  7. Sleep until the next round (one round per ~360 blocks ≈ 70 minutes).
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from distil.chain.commitments import fetch_revealed, parse_commitments
from distil.chain.metagraph import fetch_metagraph, get_subtensor
from distil.chain.weights import SetWeightsError, set_weights
from distil.eval.king import build_emission_weights, resolve_king
from distil.eval.pod import (
    acquire_pod,
    attach_pod,
    install_runtime,
    run_eval_on_pod,
    upload_runtime,
)
from distil.eval.composite_backfill import backfill_missing_composites
from distil.eval.dq_recovery import sweep_integrity_dq_recoveries
from distil.eval.results import process_round
from distil.eval.round import (
    build_round_spec,
    evict_stale_composites,
    evict_stale_evaluated_uids,
    select_challengers,
)
from distil.settings import settings
from distil.state.files import ValidatorState, log_event

logger = logging.getLogger("distil.eval.service")

# Legacy fallback only. Real value comes from ``settings.round_interval_s``
# (env: ``DISTIL_ROUND_INTERVAL_S``) and defaults to 0 → back-to-back rounds.
# Historically 4200s (~70 min / one Bittensor super-block) to pace
# ``set_weights`` calls; in 2026-05-18 we dropped this to 0 so the eval
# backlog clears without "validator is in scheduled sleep" FUD in #distil-97.
ROUND_INTERVAL_S = 4200

_STOP = False


def _install_signal_handlers() -> None:
    def _on_sig(signum, frame):
        global _STOP
        _STOP = True
        logger.info(f"signal {signum}; stopping after this round")

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)


def _wallet():
    import bittensor as bt

    return bt.wallet(name=settings.wallet_name, hotkey=settings.hotkey_name)


def _pod_context(round_id: int):
    """Persistent-pod when configured, ephemeral otherwise.

    Persistent mode reuses an existing Lium pod (lium_pod_name) across
    every round so teacher/student weights stay cached — this is what
    drops a round from ~6 h of cold downloads to ~47 min of evals.
    Falls back to ephemeral if no pod name is configured.
    """
    if settings.eval_persistent_pod and settings.lium_pod_name:
        logger.info(f"attaching to persistent pod {settings.lium_pod_name!r}")
        return attach_pod(settings.lium_pod_name)
    logger.info("provisioning ephemeral pod (no DISTIL_LIUM_POD_NAME set)")
    return acquire_pod(label=f"r{round_id}")


def _round(state: ValidatorState, *, dry_run: bool) -> None:
    sub = get_subtensor()
    mg, block, block_hash = fetch_metagraph(sub, settings.netuid)
    revealed = fetch_revealed(sub, settings.netuid)
    commitments, uid_to_hotkey, _ = parse_commitments(mg, revealed, n_uids=len(mg.hotkeys))

    state.uid_hotkey_map = {str(uid): hk for uid, hk in uid_to_hotkey.items()}
    n_evict = evict_stale_composites(state)
    if n_evict:
        logger.info(f"evicted {n_evict} stale composite rows (schema bump)")
    # Honest re-commit support: if a miner pushed v2 of their model on
    # the same UID, drop the prior evaluated_uids/composite row so the
    # challenger picker picks them up again. Without this, miners are
    # silently starved after their first eval — matches legacy
    # ``scripts/validator/single_eval.evict_stale_evaluated_uids``.
    n_recommit = len(evict_stale_evaluated_uids(state, commitments))
    if n_recommit:
        logger.info(f"evicted {n_recommit} re-committed UIDs (slot reset)")

    # Legacy ``integrity:HF 404`` DQs are otherwise permanent. When a
    # miner restores their HF repo (aizaysi did this for
    # ``RLStepone/distil-success-h19`` on 2026-05-16) we want the DQ
    # to clear automatically rather than requiring a human to edit
    # ``disqualified.json``. The sweeper HEAD-checks each
    # ``integrity:.*404`` row and drops the DQ when HF returns 200.
    try:
        cleared = sweep_integrity_dq_recoveries(state)
    except Exception as exc:  # pragma: no cover — fail open
        logger.warning(f"dq_recovery sweeper raised: {type(exc).__name__}: {exc}")
        cleared = []
    if cleared:
        for entry in cleared:
            logger.info(
                f"dq_recovery: cleared {entry['hotkey'][:20]}... "
                f"({entry['model']!r}) — HF restored"
            )

    # Backfill any ``evaluated_uids`` entry missing a composite by
    # walking ``h2h_history`` newest-first. The cutover from prod's
    # ``scripts/validator`` rebuilt ``composite_scores.json`` from
    # scratch but preserved ``evaluated_uids.json`` — left ~120 UIDs
    # flagged "evaluated" with no composite, surfacing as
    # ``eval_status: evaluated_no_composite`` + zero emission. The
    # legacy h2h composite is enough for emission-share purposes and
    # heals the mismatch within one round of redeploy.
    try:
        backfilled = backfill_missing_composites(state)
    except Exception as exc:  # pragma: no cover — fail open
        logger.warning(f"composite_backfill raised: {type(exc).__name__}: {exc}")
        backfilled = []
    if backfilled:
        logger.info(
            f"composite_backfill: restored {len(backfilled)} composite_scores "
            f"entries from h2h_history"
        )

    # Resolve seated king. Mirrors the legacy
    # ``scripts/validator/service._resolve_king`` precedence rules so
    # the cutover doesn't inadvertently dethrone the king every round:
    #
    #   1) PRIMARY  — ``state.h2h_latest.king_uid`` if that uid is
    #      still a valid commitment on the chain. The seated king
    #      keeps the crown until the paired-test dethrone gate
    #      decides otherwise, which happens AFTER scoring this round
    #      (see ``distil.eval.results.publish_round``), NOT at round-
    #      spec build time.
    #   2) FALLBACK — best ``final`` in composite_scores (UID-keyed).
    #      Only used on a cold-start validator with no h2h_latest yet.
    #
    # The previous code called ``resolve_king`` with
    # ``current_king_model=None`` which short-circuits the dethrone
    # gate and unconditionally returns the highest composite scorer —
    # i.e. every round would dethrone the current king. The legacy
    # validator never did that, and the dashboard would have flipped
    # on every cutover round.
    # ``king_name`` is the dethrone-gate's "current king" string (the
    # downstream resolve_king() call uses it as
    # ``current_king_model=``). Distil's composite_scores is UID-keyed,
    # so by convention king_name here is ``str(uid)`` — the UID-as-
    # string — matching what ``select_king`` returns. Both branches
    # MUST bind king_name (or the f-string log_event below and the
    # downstream dethrone-gate call hit UnboundLocalError, which was
    # crashing every distil round at this exact line until 22:46 UTC).
    king_name: str | None = None
    king_uid: int | None = None
    # Filter ``composite_scores`` to only UIDs with a current on-chain
    # commitment before any ``resolve_king`` call. Without this filter,
    # a UID that deregistered (or whose commitment became unparseable)
    # but still has a stored ``composite_final`` will beat every live
    # miner in ``select_king``'s highest-final scan, the
    # ``commitments.get(uid)`` lookup will then return ``None``, and
    # the self-heal fallback calls ``resolve_king`` *again on the same
    # unfiltered dict*, getting the same dead UID back → ``king_uid``
    # collapses to ``None`` and the round runs king-less. The seat
    # then stays empty for every subsequent round because each round
    # writes ``h2h_latest.king_uid=null``, which gates out the
    # ``h2h_latest`` fast path next time around.
    #
    # 2026-05-18: this was the live failure mode after UID 119
    # (composite_final=0.4209, scored 2026-05-17 12:42 UTC) lost its
    # chain commitment overnight — every round since 14:00 UTC ran
    # king-less, emission weights collapsed to zeros, and the bot
    # surfaced "king_uid: None" across the channel.
    valid_composites = {
        k: v
        for k, v in (state.composite_scores or {}).items()
        if isinstance(k, str)
        and k.isdigit()
        and int(k) in commitments
    }
    h2h_king_uid = (state.h2h_latest or {}).get("king_uid")
    if h2h_king_uid is not None and int(h2h_king_uid) in commitments:
        king_uid = int(h2h_king_uid)
        king_name = str(king_uid)
        _king_reason = "h2h_latest"
    else:
        king_name, _king_reason = resolve_king(
            valid_composites, current_king_model=None
        )
        if king_name is not None:
            try:
                # composite_scores is UID-keyed (matches legacy writer
                # ``single_eval.py:state.composite_scores[uid_str] = record``).
                king_uid = int(king_name)
            except (TypeError, ValueError):
                king_uid = next(
                    (c.uid for c in commitments.values() if c.key == king_name),
                    None,
                )
    king_commitment = commitments.get(king_uid) if king_uid is not None else None
    # Self-healing: if h2h_latest pointed at a UID that no longer has
    # a valid on-chain commitment (deregistered, model deleted, hotkey
    # rotated), fall back to the composite-scores top scorer. Without
    # this, ``build_round_spec`` runs a king-less round and downstream
    # paired-KL anchors are missing — silently corrupts every
    # subsequent dethrone gate.
    if king_uid is not None and king_commitment is None:
        logger.warning(
            f"seated king uid={king_uid} no longer in commitments; "
            "falling back to composite-scores top scorer"
        )
        fallback_name, _ = resolve_king(valid_composites, current_king_model=None)
        king_name = None
        king_uid = None
        if fallback_name is not None:
            try:
                king_uid = int(fallback_name)
            except (TypeError, ValueError):
                king_uid = next(
                    (c.uid for c in commitments.values() if c.key == fallback_name),
                    None,
                )
            king_commitment = commitments.get(king_uid) if king_uid is not None else None
            if king_commitment is not None:
                king_name = str(king_uid)
            else:
                king_uid = None

    challengers = select_challengers(commitments, state, king_uid=king_uid)
    if not challengers and king_commitment is None:
        logger.info("no challengers and no king; skipping round")
        return

    # Resume-on-attach: if we have an in-progress round and the chain block
    # hasn't advanced past the round-completion deadline, re-use the prior
    # round_spec rather than building a fresh one. Lets a crashed-restarted
    # validator continue an eval that's still running on the persistent pod.
    #
    # Resume is GATED on the prior spec containing the same set of
    # students we'd ship today. If the set differs (e.g. the king
    # changed, the challenger pool rotated, or the previous attempt
    # was built before a bugfix that altered selection rules) we
    # ABANDON the stale spec and rebuild from scratch. Without this
    # gate, a 1-student in-progress round from a pre-fix attempt
    # would keep getting "resumed" forever even after distil started
    # selecting king + 10 challengers properly.
    def _fresh_spec() -> dict[str, Any]:
        return build_round_spec(
            block=block,
            block_hash=block_hash,
            teacher_repo=settings.teacher_repo,
            reference_repo=settings.reference_repo,
            king=king_commitment,
            challengers=challengers,
        )

    in_progress = state.current_round or {}
    spec = None
    if in_progress.get("round_id") and not in_progress.get("completed"):
        age_min = (time.time() - in_progress.get("started_at", 0)) / 60
        prev_spec = in_progress.get("spec") or {}
        prev_uids = {int(s.get("uid", -1)) for s in (prev_spec.get("students") or [])}
        fresh = _fresh_spec()
        fresh_uids = {int(s.get("uid", -1)) for s in (fresh.get("students") or [])}
        if age_min >= settings.eval_round_max_minutes * 2:
            logger.warning(
                f"in-progress round {in_progress['round_id']} is {age_min:.1f} min old; "
                f"abandoning and starting fresh"
            )
            spec = fresh
        elif prev_uids != fresh_uids:
            logger.warning(
                f"in-progress round {in_progress['round_id']} students "
                f"{sorted(prev_uids)} != fresh {sorted(fresh_uids)}; "
                f"abandoning (likely built pre-bugfix) and starting fresh"
            )
            spec = fresh
        else:
            logger.info(
                f"resume: reusing in-progress round {in_progress['round_id']} "
                f"(age {age_min:.1f} min, students match)"
            )
            spec = prev_spec
    else:
        spec = _fresh_spec()

    out_dir = Path(settings.state_dir) / "_rounds" / f"round_{spec['round_id']}"
    log_event(f"starting round block={block} king={king_name} challengers={len(challengers)}")

    state.current_round = {
        "round_id": spec["round_id"],
        "block": block,
        "king_name": king_name,
        "n_challengers": len(challengers),
        "started_at": time.time(),
        "completed": False,
        "spec": spec,
    }
    state.save()

    t_round_start = time.time()
    with _pod_context(spec["round_id"]) as pod:
        upload_runtime(pod)
        install_runtime(pod)
        results = run_eval_on_pod(
            pod=pod,
            round_spec=spec,
            out_dir=out_dir,
            timeout_s=settings.eval_round_max_minutes * 60,
            n_gpus=settings.eval_n_gpus,
        )
    dur_min = (time.time() - t_round_start) / 60
    logger.info(f"round {spec['round_id']} eval finished in {dur_min:.1f} min")

    # Build the ``model@revision -> {uid, hotkey, ...}`` lookup that
    # ``process_round`` uses to map each pod result row back to a
    # specific (uid, hotkey) for state writes (composite_scores,
    # evaluated_hotkeys, activation_fingerprints, DQs).
    #
    # CRITICAL: when two miners commit the **same** model@revision under
    # **different** hotkeys, ``commitments.values()`` has two entries
    # that produce the same ``c.key``. A naive dict comprehension keeps
    # only the last-iterated one — and which one wins is whatever Python
    # gave us from the metagraph iteration (typically uid-ascending,
    # i.e. the *later* committer). Result of the bug: ``select_challengers``
    # picks the fresh UID (no composite, no evaluated_hotkeys entry),
    # the round_spec is built with that UID, the pod evaluates the
    # model, but ``process_round`` looks the result up in this
    # ``uid_index`` and writes the composite under the OTHER UID. The
    # fresh UID never gets marked evaluated, gets re-selected every
    # round, and burns 1 of the 3 challenger slots indefinitely.
    # Observed 2026-05-18 on UID 25 / UID 28 both holding
    # ``RLStepone/distil-b300-training-h25@5b20b59...``: UID 25 was in
    # the spec for 8+ consecutive rounds but every result was credited
    # to UID 28.
    #
    # Fix: prefer the (uid, hotkey) that's actually in ``spec["students"]``
    # for this round. That's the UID we scheduled, so that's the UID
    # whose state should be mutated. Fall back to ``commitments.values()``
    # for any model row not in the spec (king/reference/teacher).
    uid_index: dict[str, dict] = {}
    for s in spec.get("students", []) or []:
        name = s.get("name")
        uid_val = s.get("uid")
        if not name or uid_val is None:
            continue
        # spec entries carry uid/hotkey/revision/is_king but not
        # commit_block / coldkey — fetch those from commitments[uid].
        c = commitments.get(int(uid_val))
        uid_index[name] = {
            "uid": int(uid_val),
            "hotkey": s.get("hotkey") or uid_to_hotkey.get(int(uid_val)),
            "coldkey": getattr(c, "coldkey", None) if c else None,
            "commit_block": getattr(c, "commit_block", None) if c else None,
            "revision": s.get("revision") or (getattr(c, "revision", "main") if c else "main"),
        }
    # Fallback: include any chain commitment NOT in the spec, but use
    # ``setdefault`` so the spec winners stay authoritative.
    for c in commitments.values():
        uid_index.setdefault(
            c.key,
            {
                "uid": c.uid,
                "hotkey": uid_to_hotkey.get(c.uid),
                "coldkey": getattr(c, "coldkey", None),
                "commit_block": getattr(c, "commit_block", None),
                "revision": getattr(c, "revision", "main"),
            },
        )
    # Two different "king" identifiers here, easily confused:
    #
    #   * ``king_key`` (model_name@revision) — matches the keys in
    #     ``pod_results`` and ``commitments[].key``. Used by
    #     ``process_round`` to find the king's row inside the pod's
    #     results.json so its KL/RKL anchors the relative axes.
    #   * ``king_name``  (UID-as-string)     — matches the keys in
    #     ``state.composite_scores`` (UID-keyed, legacy schema). Used
    #     by the dethrone gate ``resolve_king``.
    #
    # Before this split, ``king_name`` was both — and ``rows.get(uid_str)``
    # in process_round returned None, so the king was never found in
    # results.json and the round-min was used as anchor instead of the
    # king's KL: a silent but big regression.
    king_key = king_commitment.key if king_commitment is not None else None
    record = process_round(
        state=state,
        pod_results=results,
        king_name=king_key,
        # ``reference_name`` is the local-loadable baseline (Qwen3.5-4B,
        # used for the baseline-penalty axis). ``teacher_name`` is the
        # cloud-API teacher (Kimi-K2.6) the round was graded against.
        # Pre-fix these two were swapped — currently harmless because
        # neither repo appears as a student key in ``pod_results`` (so
        # both rows resolved to None), but the broken-axis detection
        # and baseline-penalty reference would have silently used the
        # wrong row the moment either repo was added to the students
        # list, which is what happens during a teacher-rotation round.
        reference_name=spec["reference_repo"],
        teacher_name=spec["teacher_repo"],
        block=block,
        block_hash=block_hash,
        uid_index=uid_index,
        timings=results.get("__per_bench_timing__"),
    )
    state.current_round = {**state.current_round, "completed": True, "completed_at": time.time()}
    state.save()
    # End-of-round dethrone gate. MUST use the same commitments-filtered
    # composites that round-start used (see ``valid_composites`` above),
    # otherwise a dead UID with a stale-but-high stored
    # ``composite_final`` will beat the freshly-seated king here, the
    # record-rewrite at line ~425 will write ``king_uid=<dead_uid>`` into
    # ``h2h_latest``, and the next round's fast path will replay the
    # dead UID's lookup. 2026-05-18: this was the second half of the UID 119
    # incident — even after ``valid_composites`` blocked UID 119 from
    # being seated at round start, the unfiltered ``resolve_king`` here
    # promoted UID 119 to ``new_king_uid`` at round end (0.4209 stale >
    # 0.4175 fresh on UID 92) and the record-rewrite at line 425
    # silently re-introduced the dead UID to ``h2h_latest``. Recomputed
    # because ``process_round`` may have just inserted a fresh composite
    # for the seated king.
    valid_composites_end = {
        k: v
        for k, v in (state.composite_scores or {}).items()
        if isinstance(k, str)
        and k.isdigit()
        and int(k) in commitments
    }
    new_king, why = resolve_king(valid_composites_end, current_king_model=king_name)
    record["king_after"] = new_king
    record["king_reason"] = why

    # Persist legacy dethrone-context fields back onto the round record
    # AFTER resolve_king. The dashboard's RoundsPanel / king-history
    # endpoint reads ``prev_king_uid`` + ``new_king_uid`` + ``king_changed``
    # to render the "dethrone" annotation and the past-reigns chart;
    # without these every round renders as a king-retain regardless of
    # actual outcome. ``dethrone_method`` is the human label for WHY
    # the gate flipped (e.g. ``no_king``, ``final_gain ...``, ``margin_not_met``).
    prev_king_uid = king_uid  # the seat going INTO this round
    new_king_uid: int | None = prev_king_uid
    if new_king and new_king != king_name:
        # ``new_king`` is a UID-string out of resolve_king (composite_scores
        # is UID-keyed); convert to an int and look up the commitment.
        try:
            new_king_uid = int(new_king)
        except (TypeError, ValueError):
            new_king_uid = next(
                (c.uid for c in commitments.values() if c.key == new_king),
                None,
            )
        if new_king_uid is not None and new_king_uid in commitments:
            state.push_king(new_king_uid)
            king_uid = new_king_uid
            king_name = str(new_king_uid)
    record["prev_king_uid"] = prev_king_uid
    record["new_king_uid"] = new_king_uid
    king_changed = (
        new_king_uid is not None
        and prev_king_uid is not None
        and int(new_king_uid) != int(prev_king_uid)
    )
    record["king_changed"] = king_changed
    record["dethrone_method"] = why
    # CRITICAL: when dethrone fires, rewrite ``record["king_uid"]`` (and
    # the matching name+model fields) to the NEW seated king so the
    # next round's resolver (``service._round`` line ~167 reads
    # ``state.h2h_latest.king_uid``) picks the dethrone winner instead
    # of replaying the deposed king. Without this rewrite, three
    # successive rounds (blocks 8198615, 8199086, 8199665) recorded
    # ``king_changed=True`` with ``king_after`` set to 52, 92, and 35
    # respectively — but each next round read ``king_uid=47`` and
    # re-seated UID 47 anyway. ``king_after`` and ``new_king_uid``
    # become PURELY informational (post-gate audit fields) while
    # ``king_uid`` is the canonical seated-king field that drives the
    # next round.
    if king_changed and new_king_uid is not None:
        record["king_uid"] = int(new_king_uid)
        record["king_name"] = str(new_king_uid)
        # Look up the new king's model+revision so the dashboard +
        # downstream consumers don't render the stale deposed-king
        # model on h2h_latest.king_model.
        new_king_commit = commitments.get(int(new_king_uid))
        if new_king_commit is not None:
            record["king_model"] = new_king_commit.model
        # ``top4_leaderboard.king`` was already written by
        # ``_refresh_top4`` inside ``process_round`` BEFORE the dethrone
        # gate ran above — so its ``king`` field still names the deposed
        # UID, and ``/api/miner/<deposed>`` returns ``is_king: true``
        # while ``/api/miner/<new_king>`` returns ``is_king: false``.
        # Confirmed in #distil 2026-05-17 ("UID 35 leaderboard king,
        # UID 14 still is_king:true on API"). The contender rows ranked
        # by composite ``final`` are still valid (dethrone doesn't move
        # any composite scores) so only the ``king`` field and the
        # ``contenders`` partitioning need a post-dethrone rewrite.
        leaderboard = getattr(state, "top4_leaderboard", None) or {}
        rows = list(leaderboard.get("rows") or [])
        cs_map = getattr(state, "composite_scores", None) or {}
        new_king_row: dict | None = None
        for r in rows:
            if r.get("uid") == int(new_king_uid):
                new_king_row = r
                break
        if new_king_row is None:
            kc = cs_map.get(str(int(new_king_uid))) or {}
            if kc:
                model = kc.get("model")
                revision = kc.get("revision")
                new_king_row = {
                    "rank": None,
                    "uid": int(new_king_uid),
                    "name": f"{model}@{revision}" if model and revision else model,
                    "model": model,
                    "final": kc.get("final"),
                    "worst_3_mean": kc.get("worst_3_mean"),
                    "weighted": kc.get("weighted"),
                    "present_count": kc.get("present_count"),
                }
        if new_king_row is not None:
            state.top4_leaderboard = {
                **leaderboard,
                "updated_at": time.time(),
                "king": new_king_row,
                "contenders": [r for r in rows if r is not new_king_row],
            }
    # The record was already appended by process_round; rewrite the
    # last h2h_history row + h2h_latest with the dethrone-context
    # fields so the dashboard reads the same payload the validator
    # used to set weights.
    if state.h2h_history:
        state.h2h_history[-1] = record
    state.h2h_latest = record
    state.save()

    # Dethrone announcement — fire AFTER state.save() so the dashboard
    # banner (``GET /api/announcement``) and the state-file-derived
    # claim endpoint both see a consistent view, and so a Discord-side
    # outage can never roll back the round bookkeeping. See
    # ``distil/eval/announce.py`` for why this is best-effort: a
    # network hiccup or a missing bot token must never crash the
    # round or block ``set_weights`` from firing.
    if king_changed and new_king_uid is not None:
        try:
            from distil.eval.announce import announce_new_king
            cs = state.composite_scores or {}
            new_cs = cs.get(str(int(new_king_uid))) or {}
            prev_cs = (
                cs.get(str(int(prev_king_uid)))
                if prev_king_uid is not None else {}
            ) or {}
            prev_model: str | None = None
            if prev_king_uid is not None:
                prev_commit = commitments.get(int(prev_king_uid))
                if prev_commit is not None:
                    prev_model = getattr(prev_commit, "model", None)
                if not prev_model:
                    prev_model = prev_cs.get("model")
            new_model: str | None = record.get("king_model")
            if not new_model:
                new_commit = commitments.get(int(new_king_uid))
                if new_commit is not None:
                    new_model = getattr(new_commit, "model", None)
            announce_new_king(
                new_uid=int(new_king_uid),
                new_model=new_model,
                prev_uid=int(prev_king_uid) if prev_king_uid is not None else None,
                prev_model=prev_model,
                new_composite_final=new_cs.get("final"),
                prev_composite_final=prev_cs.get("final"),
                dethrone_method=why,
                block=record.get("block"),
                state_dir=settings.state_dir,
            )
        except Exception as exc:
            # Defensive: announce must never propagate. Log only.
            logger.warning(f"announce_new_king failed (non-fatal): {exc}")

    weights = build_emission_weights(
        n_uids=len(mg.hotkeys),
        king_uid=king_uid,
        recent_kings=state.recent_kings,
        state=state,
        uid_hotkey_map=state.uid_hotkey_map,
    )
    if dry_run:
        logger.info(f"[dry-run] weights would be: max={max(weights):.3f} king_uid={king_uid}")
        return
    if not weights or sum(weights) == 0:
        logger.info("no weights to set; skipping set_weights")
        return
    try:
        set_weights(
            sub,
            _wallet(),
            netuid=settings.netuid,
            n_uids=len(mg.hotkeys),
            weights=weights,
            label=f"king={king_uid}",
        )
        log_event(f"set_weights ok; king_uid={king_uid}", level="info")
    except SetWeightsError as exc:
        logger.error(f"set_weights failed: {exc}")
        log_event(f"set_weights failed: {exc}", level="error")


def run(
    *,
    wallet_name: str | None = None,
    hotkey_name: str | None = None,
    once: bool = False,
    dry_run: bool = False,
) -> int:
    if wallet_name:
        settings.wallet_name = wallet_name
    if hotkey_name:
        settings.hotkey_name = hotkey_name
    _install_signal_handlers()
    state = ValidatorState.load(settings.state_dir)
    while not _STOP:
        try:
            _round(state, dry_run=dry_run)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.exception(f"round failed: {exc}")
            # The journal sometimes loses logger.exception() multi-line
            # tracebacks (likely due to a missing handler config), so
            # ALSO mirror the formatted traceback into validator_log.json
            # via log_event — the first 1200 chars are enough to locate
            # the offending frame without scrolling.
            log_event(f"round crashed: {exc}", level="error")
            log_event(f"round traceback:\n{tb[-1200:]}", level="error")
            time.sleep(60)
            if once:
                return 1
        if once:
            return 0
        # Read each iteration so an operator can hot-tune via env +
        # validator restart without a code change.
        try:
            sleep_s = int(getattr(settings, "round_interval_s", ROUND_INTERVAL_S))
        except (TypeError, ValueError):
            sleep_s = ROUND_INTERVAL_S
        sleep_s = max(0, sleep_s)
        if sleep_s == 0:
            logger.info(
                "inter-round sleep disabled (round_interval_s=0); "
                "starting next round immediately"
            )
            continue
        logger.info(f"sleeping {sleep_s}s before next round")
        for _ in range(sleep_s):
            if _STOP:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(run(once="--once" in sys.argv, dry_run="--dry-run" in sys.argv))
