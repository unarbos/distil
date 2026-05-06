import logging
import json
import os
import time

from eval.scoring import disqualify
from eval.state import ValidatorState
from scripts.eval_policy import policy_env
from scripts.validator.config import MAX_KL_THRESHOLD, TOP_N_ALWAYS_INCLUDE
from scripts.validator import single_eval as single_eval_mod
from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
from scripts.validator.single_eval import (
    bootstrap_composite_from_h2h,
    evict_stale_evaluated_uids,
    is_single_eval_mode,
)

logger = logging.getLogger("distillation.remote_validator")


def _write_eval_backlog(state: ValidatorState, *, cap: int, pending: dict, kept: dict, deferred: list[int]) -> None:
    try:
        state_dir = getattr(state, "state_dir", None)
        if state_dir is None:
            return
        kept_set = {int(uid) for uid in kept.keys()}
        pending_rows = []
        for uid, info in sorted(
            pending.items(),
            key=lambda kv: (int((kv[1] or {}).get("commit_block") or 0), int(kv[0])),
        ):
            pending_rows.append({
                "uid": int(uid),
                "model": (info or {}).get("model"),
                "revision": (info or {}).get("revision"),
                "commit_block": (info or {}).get("commit_block"),
                "status": "queued" if int(uid) in kept_set else "deferred",
            })
        payload = {
            "updated_at": time.time(),
            "round_cap": cap,
            "pending_total": len(pending),
            "queued_uids": [int(uid) for uid in kept.keys()],
            "deferred_uids": [int(uid) for uid in deferred],
            "pending": pending_rows,
        }
        path = state_dir / "eval_backlog.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("single-eval: failed to persist eval backlog: %s", exc)


# 2026-04-24 (distil-97): once the subnet enters steady-state (all ~65
# valid models in ``state.scores``), ``select_challengers`` yields zero
# P1/P3 candidates because every UID is considered "already evaluated".
# ``add_top5_contenders`` then fills with the 4 H2H leaderboard slots
# and the round settles at 5-6 models — fine for tracking the king vs
# top-4 but blind to any dormant miner whose global KL (measured vs an
# earlier king on a different prompt set) is actually better than the
# current king's H2H KL. Without re-rotation, the subnet ranking
# silently goes stale and dormant miners with legitimately better
# models cannot regain the crown without re-uploading.
#
# ``DORMANT_ROTATION_N`` adds that many dormant miners per round,
# filtered to those whose ``state.scores[uid]`` beats the current
# king's h2h_kl (so we only spend compute on candidates who could
# plausibly win). Default 2 = ~16 extra minutes per round with
# shadow axes off, fits inside the 60-75min target.
DORMANT_ROTATION_N = int(policy_env("DORMANT_ROTATION_N", "2"))

# Maintenance rounds should keep the crown under pressure without turning every
# block into a multi-hour full sweep. The first few H2H contenders are sticky;
# lower leaderboard slots still enter the candidate pool, but new submissions
# and high-scoring dormant models can beat them for capped slots.
MAINTENANCE_CHALLENGER_CAP = int(policy_env("MAINTENANCE_CHALLENGER_CAP", "12"))
PROTECTED_H2H_CONTENDERS = int(
    policy_env("PROTECTED_H2H_CONTENDERS", str(min(4, max(1, TOP_N_ALWAYS_INCLUDE - 1))))
)


# 2026-04-24 (distil-97): evict H2H leaderboard contenders that fail precheck
# repeatedly. Scenario we keep hitting: a miner submits a public model, wins
# into the top-4 leaderboard, then privates the repo (restricted/gated on HF).
# Validator can never re-verify it, the entry sits there as a ghost blocking
# a real slot and spamming the TOP-CONTENDER REGRESSION CHECK warning every
# round. UID 64 (sampleratez/3406940) has been stuck like this for 4+ rounds.
# After this many consecutive precheck failures we drop the entry from the
# persisted leaderboard. The counter resets the moment precheck passes again,
# so transient HF blips (see 60317bb) don't evict anyone unfairly.
LB_PRECHECK_EVICTION_STREAK = int(policy_env("LB_PRECHECK_EVICTION_STREAK", "3"))


def select_challengers(valid_models, state: ValidatorState, king_uid, king_kl,
                       epoch_count: int, trust_king_kl: bool = True):
    """Pick challengers for the round.

    ``trust_king_kl`` = False disables the ``best_ever > king_kl*2`` prune.
    Set this when the king was picked from a stale cached score (the old H2H
    leaderboard expired and `_resolve_king` fell back to `state.scores`) —
    in that case ``king_kl`` can be artificially low (scores were measured
    against a different king, prompt set, or even a different model later
    re-uploaded under the same UID) and tightens the skip threshold so
    aggressively that genuinely competitive UIDs never get re-evaluated.

    When ``SINGLE_EVAL_MODE=1`` the planner runs a stripped-down version
    that only returns commitments not yet scored (or whose on-chain commit
    has changed since the last eval). The legacy king-pairing prune and
    the full-eval P1B branch are skipped entirely — re-evaluation is
    explicitly disallowed.
    """
    if is_single_eval_mode():
        evict_stale_evaluated_uids(state, valid_models)
        challengers = {}
        # Force-eligible UIDs:
        #
        #   - The current king ALWAYS gets force-eligible (2026-04-27).
        #     Even with procedural per-round bench items the king's
        #     stored composite came from a different prompt sample than
        #     this round's challengers, so worst-axis comparison is
        #     cross-sample. SE on bench axes with n=8-12 binomial items
        #     is ~0.14, which is bigger than the dethrone margin.
        #     Re-evaluating the king on the same prompts as challengers
        #     restores paired evaluation. (Discord 2026-04-27, coffieex
        #     +crypsick: "the variance is extreme across shards".)
        #
        #   - Historically: also re-eval'd on schema bumps. Now
        #     redundant because we re-eval every round, but the
        #     code path still works as a safety net if king-in-round
        #     ever gets disabled.
        force_eligible: set[str] = set()
        if king_uid is not None:
            king_record = (state.composite_scores or {}).get(str(king_uid))
            force_eligible.add(str(king_uid))
            if isinstance(king_record, dict):
                try:
                    king_version = int(king_record.get("version") or 0)
                except (TypeError, ValueError):
                    king_version = 0
                if king_version < int(COMPOSITE_SHADOW_VERSION):
                    logger.info(
                        f"single-eval: forcing king UID {king_uid} re-eval "
                        f"(stored composite version {king_version} < "
                        f"current schema {COMPOSITE_SHADOW_VERSION}); ensures "
                        f"like-for-like comparison against challengers."
                    )
                else:
                    logger.info(
                        f"single-eval: king UID {king_uid} included in "
                        f"this round (paired re-eval on shared prompts)."
                    )
        for uid, info in valid_models.items():
            uid_str = str(uid)
            model_name = info["model"]
            if info.get("is_reference"):
                continue
            if model_name in state.permanently_bad_models:
                state.evaluated_uids.add(uid_str)
                continue
            if uid_str in state.composite_scores and uid_str not in force_eligible:
                continue
            # Strict no-re-eval: a UID in evaluated_uids has been through a
            # full round once already. Even if its score row got dropped
            # later (DQ revert, partial-state reset, etc.), per single-eval
            # policy it should not run again unless its commitment changed
            # — and ``evict_stale_evaluated_uids`` already pulled out the
            # commitment-changed entries above. The previous filter required
            # both ``evaluated_uids`` AND ``scores`` to be set, which let
            # historical UIDs sneak back into the queue when state was
            # partially rebuilt.
            if uid_str in state.evaluated_uids and uid_str not in force_eligible:
                continue
            challengers[uid] = info
        # FIFO cap: oldest commitment first. Without this the planner
        # queues every pending new commit at once and rounds bloat to 8h
        # of pod compute. The cap forces rotation across rounds so each
        # individual round stays in the 60–75 min target. We read the
        # live cap from the single_eval module each call so unit tests
        # (and operators editing the env at runtime) can override it
        # without restarting the planner.
        cap = int(single_eval_mod.SINGLE_EVAL_MAX_PER_ROUND)
        pending_before_cap = dict(challengers)
        deferred: list[int] = []
        if challengers and cap > 0 and len(challengers) > cap:
            ordered = sorted(
                challengers.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("commit_block") or 0),
                    kv[0],
                ),
            )
            kept = dict(ordered[:cap])
            deferred = [uid for uid, _ in ordered[cap:]]
            logger.info(
                f"single-eval: capping round at {cap} of {len(challengers)} "
                f"pending new commitments (FIFO by commit_block); deferred "
                f"to next round: {deferred}"
            )
            challengers = kept
        _write_eval_backlog(
            state,
            cap=cap,
            pending=pending_before_cap,
            kept=challengers,
            deferred=deferred,
        )
        if challengers:
            n_king = 1 if (king_uid is not None and str(king_uid) in {str(u) for u in challengers}) else 0
            n_others = len(challengers) - n_king
            logger.info(
                f"single-eval: {n_others} new commitment(s) to evaluate"
                + (" + king (paired re-eval, 2026-04-27 fairness fix)" if n_king else "")
                + " (no top-N rotation, no dormant rotation)"
            )
        else:
            logger.info(
                "single-eval: no new commitments this round — round will be a no-op "
                "(king retains crown, weights stay)"
            )
        return challengers
    challengers = {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.evaluated_uids and uid_str in state.scores:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        best_ever = state.model_score_history.get(model_name, {}).get("best_kl")
        if trust_king_kl and best_ever is not None and king_kl < float("inf"):
            skip_threshold = max(king_kl * 2.0, king_kl + 0.05)
            if best_ever > skip_threshold:
                state.evaluated_uids.add(uid_str)
                continue
        challengers[uid] = info
    if king_uid is None:
        return challengers
    p1_new = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info["model"] in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        if state.scores.get(uid_str) is not None:
            continue
        if uid_str in state.evaluated_uids:
            continue
        p1_new.append(uid)
    for uid in p1_new:
        challengers[uid] = valid_models[uid]
    if p1_new:
        logger.info(f"🎯 SMART CHALLENGER: {len(p1_new)} new submission(s) — Priority 1: never evaluated")
    if state.top4_leaderboard.get("phase") == "initial_eval":
        full_eval_kl_cutoff = 0.12
        p1b = []
        for uid, info in valid_models.items():
            if uid == king_uid or uid in challengers:
                continue
            if info["model"] in state.permanently_bad_models:
                continue
            uid_str = str(uid)
            global_kl = state.scores.get(uid_str)
            if global_kl is None or global_kl <= 0 or global_kl > full_eval_kl_cutoff:
                continue
            h2h_record = state.h2h_tested_against_king.get(uid_str, {})
            if h2h_record.get("king_uid") == king_uid:
                continue
            p1b.append((uid, global_kl))
        if p1b:
            p1b.sort(key=lambda x: x[1])
            for uid, _ in p1b:
                challengers[uid] = valid_models[uid]
            logger.info(f"🏆 FULL EVAL: {len(p1b)} scored models added (untested vs new king, KL<=0.12)")
    return challengers


def add_top5_contenders(challengers, valid_models, state: ValidatorState, king_uid):
    """Always include top contenders in every eval round.

    Uses the latest round's H2H leaderboard (``top4_leaderboard.contenders``)
    first — these were ranked on the same prompt set as the current king and
    are the only fair cross-round comparison. Falls back to ``state.scores``
    only when no H2H leaderboard exists yet (e.g. fresh state after migration).

    The previous behaviour ranked purely by ``state.scores`` which mixes KL
    from different prompt sets and silently bumped genuine top-4 contenders
    off the round when newer challengers happened to have better-looking
    cross-round raw KL. Reported by Topaz (2026-04-17).

    No-op when ``SINGLE_EVAL_MODE=1``: the new policy is one-eval-per-
    commitment, so re-pinning H2H contenders into the round is exactly the
    behavior the flag exists to disable.
    """
    if is_single_eval_mode():
        return
    if king_uid is None:
        return
    contenders_added = 0

    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if lb_contenders:
        for entry in lb_contenders:
            uid = entry.get("uid")
            if uid is None or uid == king_uid or uid in challengers:
                continue
            if uid in valid_models:
                challengers[uid] = valid_models[uid]
                contenders_added += 1
        if contenders_added:
            logger.info(
                f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
                f"to eval (from H2H leaderboard)"
            )
        return

    scored = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is not None and 0 < kl < float("inf"):
            scored.append((uid, kl))
    scored.sort(key=lambda x: x[1])
    for uid, kl in scored[:TOP_N_ALWAYS_INCLUDE - 1]:
        challengers[uid] = valid_models[uid]
        contenders_added += 1
    if contenders_added:
        logger.info(
            f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
            f"to eval (from global scores — fallback)"
        )


def add_dormant_rotation(challengers, valid_models, state: ValidatorState,
                         king_uid, king_kl):
    """Rotate in ``DORMANT_ROTATION_N`` dormant miners whose global KL beats
    the current king.

    Rationale: once the subnet is steady-state, no new P1/P3 fires and the
    round shrinks to king+top-4. Miners who scored very well against an
    earlier king sit in ``state.scores`` forever without re-entering the
    ranking. This function picks the N best dormant scorers whose KL is
    below the current king's h2h_kl, so they can either:
      (a) confirm they're genuinely strong and climb back into the top-N,
      (b) show their old score was noise from an easier prompt set and
          settle back out of the running next round.

    Defensive filters:
      * skip king, skip current challengers, skip permanently_bad_models
      * require ``state.scores[uid] < king_kl`` (no point re-testing
        already-worse models)
      * require uid in ``valid_models`` (passed precheck this round)

    Opt-out: set ``DORMANT_ROTATION_N=0`` in the validator env to disable.
    Also a no-op when ``SINGLE_EVAL_MODE=1`` — dormant rotation is itself a
    re-eval mechanism and is incompatible with one-eval-per-commitment.
    """
    if is_single_eval_mode():
        return
    if king_uid is None or DORMANT_ROTATION_N <= 0:
        return
    if king_kl is None or king_kl == float("inf"):
        return
    candidates = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info.get("model") in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is None or kl <= 0 or kl >= float("inf"):
            continue
        if kl >= king_kl:
            continue
        candidates.append((uid, kl))
    candidates.sort(key=lambda x: x[1])
    added = []
    for uid, kl in candidates[:DORMANT_ROTATION_N]:
        challengers[uid] = valid_models[uid]
        added.append((uid, kl))
    if added:
        roster = ", ".join(f"UID {u}(kl={k:.4f})" for u, k in added)
        logger.info(
            f"♻️  Dormant rotation: added {len(added)} of {len(candidates)} "
            f"candidates better than king_kl={king_kl:.4f}: {roster}"
        )


def cap_challengers(challengers, state: ValidatorState, king_uid):
    # Single-eval mode: every commitment in the round is a never-evaluated
    # new submission, so the cap doesn't apply (there's nothing to truncate
    # — the natural ceiling is "however many new commits arrived"). The
    # registration burn cost is the spam control instead of an artificial
    # per-round limit.
    if is_single_eval_mode():
        return
    phase = state.top4_leaderboard.get("phase", "maintenance")
    max_cap = 80 if phase == "initial_eval" else MAINTENANCE_CHALLENGER_CAP
    if len(challengers) <= max_cap:
        return
    logger.warning(f"{len(challengers)} challengers exceeds cap of {max_cap} (phase={phase}). Truncating.")
    king_entry = challengers.pop(king_uid, None)
    # Preserve only the strongest H2H contenders. Previously every stored H2H
    # contender was pinned, so a six-slot leaderboard plus dormant rotation
    # could crowd out newer commits and make maintenance rounds too slow. The
    # remaining H2H entries still compete below, but do not override P1/new.
    lb_entries = [
        entry for entry in (state.top4_leaderboard.get("contenders") or [])
        if entry.get("uid") is not None and entry.get("uid") != king_uid
    ]
    lb_rank = {entry.get("uid"): i for i, entry in enumerate(lb_entries)}
    protected_uids = {
        entry.get("uid") for entry in lb_entries[:max(0, PROTECTED_H2H_CONTENDERS)]
    }
    protected = {uid: info for uid, info in challengers.items() if uid in protected_uids}
    remaining = {uid: info for uid, info in challengers.items() if uid not in protected_uids}

    def priority(item):
        uid, info = item
        uid_str = str(uid)
        score = state.scores.get(uid_str)
        is_new = score is None and uid_str not in state.evaluated_uids
        is_lb = uid in lb_rank
        commit_block = int((info or {}).get("commit_block") or 0)
        # Lower tuple sorts first:
        #   0: never-evaluated/new submissions, newest first
        #   1: scored dormant candidates by best known KL
        #   2: unprotected H2H contenders by H2H rank
        #   3: everything else
        if is_new:
            return (0, -commit_block, uid)
        if score is not None and 0 < score < float("inf"):
            return (1, float(score), -commit_block, uid)
        if is_lb:
            return (2, lb_rank[uid], -commit_block, uid)
        return (3, -commit_block, uid)

    sorted_remaining = sorted(remaining.items(), key=priority)
    slots_for_remaining = max(0, max_cap - len(protected) - (1 if king_entry else 0))
    challengers.clear()
    challengers.update(protected)
    challengers.update(dict(sorted_remaining[:slots_for_remaining]))
    if king_entry:
        challengers[king_uid] = king_entry
    if protected:
        logger.info(
            f"cap_challengers: protected {len(protected)} top-contender(s) "
            f"from truncation: {sorted(protected)}; cap={max_cap}"
        )


def assert_top_contenders_present(challengers, valid_models, state: ValidatorState, king_uid):
    """Regression guard: loud WARNING if any H2H leaderboard contender is absent from the
    eval round despite being a valid known model. Topaz's top-4 bug silently dropped
    genuine contenders for several rounds before being noticed — never again.

    Also handles auto-eviction of ghost contenders that persistently fail precheck
    (``LB_PRECHECK_EVICTION_STREAK``) — see module docstring for rationale.

    No-op when ``SINGLE_EVAL_MODE=1``: there's no notion of "top contenders that
    must reappear every round" — each commitment is evaluated exactly once.
    """
    if is_single_eval_mode():
        return
    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if not lb_contenders:
        return
    missing = []
    forced = []
    evicted = []
    kept = []
    for entry in lb_contenders:
        uid = entry.get("uid")
        if uid is None or uid == king_uid:
            kept.append(entry)
            continue
        in_valid = uid in valid_models
        model = (valid_models.get(uid) or {}).get("model") if in_valid else entry.get("model")
        if uid in challengers or in_valid:
            if entry.get("precheck_fail_streak"):
                entry["precheck_fail_streak"] = 0
            if uid in challengers:
                kept.append(entry)
                continue
            # If a valid H2H leaderboard contender was lost during cap/planning,
            # force it back into the round instead of merely warning. These are
            # the exact UIDs whose absence makes the crown under-tested.
            if in_valid:
                challengers[uid] = valid_models[uid]
                forced.append({"uid": uid, "model": model, "h2h_kl": entry.get("h2h_kl") or entry.get("kl")})
                kept.append(entry)
                continue
        if not in_valid:
            entry["precheck_fail_streak"] = int(entry.get("precheck_fail_streak", 0)) + 1
            if entry["precheck_fail_streak"] >= LB_PRECHECK_EVICTION_STREAK:
                evicted.append({"uid": uid, "model": model,
                                "streak": entry["precheck_fail_streak"]})
                continue
        missing.append({
            "uid": uid,
            "model": model,
            "in_valid_models": in_valid,
            "in_bad_list": model in state.permanently_bad_models if model else None,
            "h2h_kl": entry.get("h2h_kl") or entry.get("kl"),
            "precheck_fail_streak": entry.get("precheck_fail_streak", 0),
        })
        kept.append(entry)
    if forced:
        roster = ", ".join(f"UID {e['uid']} ({e['model']})" for e in forced)
        logger.warning(
            f"🛡️  Forced {len(forced)} valid H2H leaderboard contender(s) "
            f"back into the eval round after cap/planning: {roster}"
        )
    if evicted:
        state.top4_leaderboard["contenders"] = kept
        try:
            state.save_top4()
        except Exception as exc:
            logger.warning(f"failed to persist leaderboard after eviction: {exc}")
        roster = ", ".join(f"UID {e['uid']} ({e['model']}, streak={e['streak']})" for e in evicted)
        logger.warning(
            f"🪦 Evicted {len(evicted)} ghost contender(s) from H2H leaderboard "
            f"after {LB_PRECHECK_EVICTION_STREAK}+ consecutive precheck failures: {roster}"
        )
    if missing:
        logger.warning(
            f"⚠️  TOP-CONTENDER REGRESSION CHECK: {len(missing)} H2H leaderboard "
            f"contender(s) NOT in this round: {missing}"
        )
    else:
        logger.info(
            f"✅ top-contender check: all {len(lb_contenders) - len(evicted)} H2H "
            f"leaderboard contender(s) present in round"
        )


def check_models_exist(models_to_eval, uid_to_hotkey, state: ValidatorState, commitments: dict):
    removed = []
    for uid in list(models_to_eval.keys()):
        model_repo = models_to_eval[uid]["model"]
        try:
            import urllib.request

            req = urllib.request.Request(f"https://huggingface.co/api/models/{model_repo}", method="HEAD")
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.warning(f"UID {uid} ({model_repo}): deleted from HF — DQ")
                hotkey = models_to_eval[uid].get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                commit_block = models_to_eval[uid].get("commit_block")
                disqualify(hotkey, f"Model {model_repo} no longer exists on HuggingFace (404)", state.dq_reasons, commit_block=commit_block)
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                state.evaluated_uids.add(str(uid))
                removed.append(uid)
    for uid in removed:
        models_to_eval.pop(uid, None)
    return removed
