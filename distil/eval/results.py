"""Process the per-student JSON the pod returns into the validator's state.

* Compute composite scores for each student (with broken-axes + baseline penalty)
* Persist the head-to-head round record + history
* SHA256 dedupe (DQ exact-weight duplicates)
* Activation-fingerprint near-copy DQ — both within-round AND against historical
  fingerprints in ``state/activation_fingerprints.json`` (ports prod's
  ``scripts.validator.precheck.check_activation_fingerprint``).
* Long-form derail DQ — per-hotkey permanent DQ when most long-form responses
  fall below the coherence floor (ports prod's ``_check_long_form_derail_dq``).
* Composite dethrone floor — single-axis floor for dethroning eligibility.

The runtime state mutated here is the same ``distil.state.files.ValidatorState``
dataclass the legacy code wrote — all on-disk filenames + JSON shapes match
prod, so the API + dashboard read the same files unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Iterable

from distil.eval.composite import (
    compute_axes,
    compute_composite,
    resolve_reference_broken_axes,
    resolve_teacher_broken_axes,
)
from distil.settings import settings
from distil.state.files import ValidatorState

logger = logging.getLogger("distil.eval.results")


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolve_anchor(rows: dict[str, dict], king_name: str | None, key: str) -> float | None:
    """Pick the anchor for relative axes (king if seated, else round-min).

    Guarded against non-dict rows because results.json contains meta
    entries like ``__finished_at__: <float>`` mixed in with the
    student records — iterating ``rows.values()`` without an
    ``isinstance`` check raised ``'float' object has no attribute 'get'``
    on every cutover round at publish time.
    """
    if king_name and isinstance(kr := rows.get(king_name), dict):
        v = kr.get(key)
        try:
            f = float(v)
            if f > 0 and f == f and f not in (float("inf"), float("-inf")):
                return f
        except (TypeError, ValueError):
            pass
    best: float | None = None
    for row in rows.values():
        if not isinstance(row, dict) or row.get("is_teacher"):
            continue
        try:
            v = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
    return best


def _opr_anchor(rows: dict[str, dict], king_name: str | None) -> float | None:
    """on_policy_rkl uses the round-wide best (lowest) RKL when a non-king reported."""
    best = None
    for _name, row in rows.items():
        if not isinstance(row, dict):
            continue
        opr = row.get("on_policy_rkl") or {}
        if not isinstance(opr, dict):
            continue
        rkl = opr.get("mean_rkl")
        if rkl is None:
            continue
        try:
            v = float(rkl)
        except (TypeError, ValueError):
            continue
        if v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
    return best


# ── Activation-fingerprint near-copy DQ ─────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _layer_sim(fp_a: dict[str, list[float]], fp_b: dict[str, list[float]]) -> float | None:
    """Mean cosine sim across layers both fingerprints share. None if no overlap."""
    if not isinstance(fp_a, dict) or not isinstance(fp_b, dict):
        return None
    shared = [k for k in fp_a.keys() if k in fp_b]
    if not shared:
        return None
    sims = [_cosine(fp_a[k], fp_b[k]) for k in shared]
    sims = [s for s in sims if s == s]  # drop NaN
    if not sims:
        return None
    return sum(sims) / len(sims)


def _check_activation_copy(
    *,
    uid: str | None,
    hotkey: str | None,
    model_name: str,
    fingerprint: dict | None,
    commit_block: int | None,
    coldkey: str | None,
    uid_index: dict[str, dict],
    state: ValidatorState,
) -> tuple[bool, dict[str, Any] | None]:
    """Return ``(is_copy_to_dq, evidence)`` for one student.

    ``is_copy_to_dq`` is True only when THIS uid was the later committer (the
    "copy" side). When THIS uid was the earlier committer, returns False but
    surfaces an evidence dict the caller can log — the later-committed uid
    will be DQ'd when its row is processed.

    Same-coldkey matches are reported but never DQ'd (self-copy carve-out).
    Unknown commit-block ordering fails open (skip DQ) — prod behavior.
    """
    if not isinstance(fingerprint, dict):
        return False, None
    layer_fps = fingerprint.get("layer_fingerprints") or {}
    if not isinstance(layer_fps, dict) or not layer_fps:
        return False, None

    threshold = settings.activation_fp_threshold
    best_sim = 0.0
    best_uid: str | None = None
    best_record: dict | None = None
    for stored_uid, stored in state.activation_fingerprints.items():
        if not isinstance(stored, dict):
            continue
        if str(stored_uid) == str(uid):
            continue
        sim = _layer_sim(layer_fps, stored.get("layer_fingerprints") or {})
        if sim is None:
            continue
        if sim > best_sim:
            best_sim = sim
            best_uid = stored_uid
            best_record = stored
    if best_sim < threshold or best_record is None:
        return False, None

    other_model = best_record.get("model") or "unknown"
    other_block = best_record.get("commit_block")
    other_coldkey = best_record.get("coldkey")
    # Same-coldkey carve-out: never DQ a miner against themselves
    if coldkey and other_coldkey and coldkey == other_coldkey:
        logger.info(
            "uid=%s (%s) activation-matches uid=%s (%s) at sim=%.4f but shares coldkey; "
            "no DQ (self-copy carve-out)",
            uid, model_name, best_uid, other_model, best_sim,
        )
        return False, None

    # Resolve commit-block ordering. Fail open if unknown.
    if commit_block is None or other_block is None:
        logger.warning(
            "uid=%s activation-match with uid=%s sim=%.4f but commit_block unknown "
            "(my=%s, other=%s); skipping DQ to avoid false positives",
            uid, best_uid, best_sim, commit_block, other_block,
        )
        return False, None
    try:
        my_b = float(commit_block)
        other_b = float(other_block)
    except (TypeError, ValueError):
        return False, None

    evidence = {
        "match_uid": best_uid,
        "match_model": other_model,
        "sim": best_sim,
        "my_block": int(my_b),
        "other_block": int(other_b),
    }
    if my_b > other_b:
        # I committed later → I am the copy → DQ me
        return True, evidence
    # I committed first → no DQ for me; the other side gets it
    logger.info(
        "uid=%s (%s) activation-matches uid=%s (%s) sim=%.4f but I committed first; "
        "uid=%s will be DQ'd when its row is processed",
        uid, model_name, best_uid, other_model, best_sim, best_uid,
    )
    return False, evidence


# ── Long-form derail DQ ─────────────────────────────────────────────────


def _check_long_form_derail(student_row: dict) -> tuple[bool, str | None]:
    """Return ``(should_dq, reason)`` based on ``long_form_judge_probe.per_prompt``.

    DQ when > ``long_form_derail_dq_ratio`` of long-form responses score
    below ``long_form_derail_dq_threshold`` coherence. Ports prod's
    ``_check_long_form_derail_dq`` minus the king-exemption (handled by caller).
    """
    if not settings.long_form_derail_dq_enabled:
        return False, None
    lf = student_row.get("long_form_judge_probe") or {}
    per_prompt = lf.get("per_prompt") or []
    if not per_prompt:
        return False, None
    thr = settings.long_form_derail_dq_threshold
    derailed = sum(
        1 for r in per_prompt
        if isinstance(r, dict)
        and isinstance(r.get("coherence"), (int, float))
        and r["coherence"] < thr
    )
    if derailed / len(per_prompt) <= settings.long_form_derail_dq_ratio:
        return False, None
    coh = lf.get("coherence_factor")
    coh_str = f"{coh:.3f}" if isinstance(coh, (int, float)) else "n/a"
    reason = (
        f"long_form_incoherence: {derailed}/{len(per_prompt)} long-form responses "
        f"derailed (coherence<{thr:.2f}; aggregate coherence_factor={coh_str}). "
        f"Model produces word salad past ~500 tokens. DQ scope is per-hotkey: a "
        f"new on-chain commit on the SAME hotkey will NOT clear this DQ. "
        f"Register a fresh hotkey with a model that doesn't derail."
    )
    return True, reason


# ── Public entrypoint ────────────────────────────────────────────────────


def process_round(
    *,
    state: ValidatorState,
    pod_results: dict[str, dict[str, Any]],
    king_name: str | None,
    reference_name: str | None,
    teacher_name: str | None,
    block: int,
    block_hash: str | None,
    uid_index: dict[str, dict] | None = None,
    timings: list[dict] | None = None,
) -> dict[str, Any]:
    """Mutate ``state`` in place; return the round record.

    Parameters
    ----------
    state
        Loaded :class:`ValidatorState` (will be mutated + saved).
    pod_results
        ``{model_name: row}`` from ``eval_results_merged.json``.
    king_name, reference_name, teacher_name
        Names matching keys in ``pod_results`` (None → not seated this round).
    block, block_hash
        Anchor block for this round (recorded into the h2h_history entry).
    uid_index
        Optional ``{model_name: {uid, hotkey, coldkey, commit_block, revision}}``
        used by the activation-fingerprint history DQ. If not provided,
        history dedup falls open (no DQ).
    timings
        Optional per-bench timing list to record in h2h.
    """
    # Drop meta-keys (``__finished_at__`` etc.) that the pod orchestrator
    # mixes into results.json alongside per-student records — they
    # would otherwise blow up every iteration in this function.
    students: dict[str, dict] = {
        k: v for k, v in pod_results.items() if isinstance(v, dict)
    }
    teacher_row = students.get(teacher_name) if teacher_name else None
    reference_row = students.get(reference_name) if reference_name else None
    uid_index = uid_index or {}

    king_kl = _resolve_anchor(students, king_name, "kl_global_avg")
    king_rkl = _opr_anchor(students, king_name)
    broken = resolve_reference_broken_axes(reference_row) | resolve_teacher_broken_axes(
        teacher_row,
        king_kl=king_kl,
        king_rkl=king_rkl,
    )
    reference_axes: dict[str, float | None] | None = None
    if reference_row is not None:
        reference_axes = compute_axes(reference_row, king_kl=king_kl, king_rkl=king_rkl)

    composites: dict[str, dict] = {}
    dq_events: list[dict[str, Any]] = []

    for name, row in students.items():
        if row.get("is_teacher"):
            continue

        idx = uid_index.get(name) or {}
        uid = idx.get("uid") if idx else row.get("uid")
        hotkey = idx.get("hotkey") if idx else row.get("hotkey")
        coldkey = idx.get("coldkey")
        commit_block = idx.get("commit_block")
        is_king = (name == king_name)

        # SHA256 exact-weight duplicate check (always-on, cross-round).
        sha = row.get("weights_sha256")
        if sha and not is_king:
            prior = state.model_hashes.get(sha)
            if prior and prior != name:
                state.disqualify(hotkey or str(uid or name), f"duplicate_weights_of:{prior}")
                dq_events.append({"uid": uid, "name": name, "kind": "duplicate_weights", "of": prior})
            else:
                state.model_hashes[sha] = name

        # Activation-fingerprint cross-round + within-round near-copy.
        fp = row.get("activation_fingerprint")
        if isinstance(fp, dict) and fp.get("layer_fingerprints") and not is_king:
            should_dq, evidence = _check_activation_copy(
                uid=str(uid) if uid is not None else None,
                hotkey=hotkey,
                model_name=name,
                fingerprint=fp,
                commit_block=commit_block,
                coldkey=coldkey,
                uid_index=uid_index,
                state=state,
            )
            if should_dq and evidence:
                reason = (
                    f"activation_copy: cosine={evidence['sim']:.4f} matched uid="
                    f"{evidence['match_uid']} ({evidence['match_model']}), "
                    f"committed at block {evidence['my_block']} (theirs: {evidence['other_block']})"
                )
                state.disqualify(hotkey or str(uid or name), reason)
                dq_events.append({"uid": uid, "name": name, "kind": "activation_copy",
                                  **evidence})

        # Long-form derail (king-exempt by composite design; the king has
        # cleared the floor in past rounds and re-runs preserve seat).
        if not is_king:
            should_dq, reason = _check_long_form_derail(row)
            if should_dq and reason:
                state.disqualify(hotkey or str(uid or name), reason)
                dq_events.append({"uid": uid, "name": name, "kind": "long_form_derail"})

        comp = compute_composite(
            row,
            king_kl=king_kl,
            king_rkl=king_rkl,
            broken_axes=broken,
            reference_axes=reference_axes if name != reference_name else None,
        )
        comp["evaluated_at"] = time.time()
        # Commit signature: ``model`` / ``revision`` / ``block`` are how
        # ``distil.eval.round.evict_stale_evaluated_uids`` detects an
        # honest re-commitment on the same UID (legacy
        # ``scripts/validator/single_eval.commitment_changed``). Without
        # these fields the stored composite is treated as bootstrapped
        # and the UID stays consumed forever, which silently starves
        # miners who push v2 of their model.
        model_repo = name.split("@", 1)[0] if "@" in name else name
        comp["model"] = row.get("model") or model_repo
        revision_str = ""
        if "@" in name:
            revision_str = name.split("@", 1)[1]
        comp["revision"] = row.get("revision") or revision_str or "main"
        if commit_block is not None:
            comp["block"] = int(commit_block)
        if state.is_disqualified(hotkey, uid=int(uid) if uid is not None else None):
            comp["disqualified"] = True
            comp["dq_reason"] = state.dq_reason(hotkey, uid=int(uid) if uid is not None else None)
        composites[name] = comp
        # ``state.composite_scores`` is UID-keyed to match the legacy
        # writer (scripts/validator/single_eval.py:
        # ``state.composite_scores[uid_str] = record``). The in-memory
        # ``composites`` dict above stays model-name-keyed because the
        # h2h record + DQ logic below indexes by name.
        #
        # CRITICAL: only persist when Phase 2 actually produced a KL
        # signal. The legacy merge_composite_scores does the equivalent
        # of ``if comp.get('worst') is None: continue`` -- but that's
        # only a *partial* guard: if Phase 2 vLLM-loaded the student
        # OK but the JSON-stringified teacher_logprobs keys made every
        # position's intersection empty (kl=None, top_k_overlap=0), the
        # bench axes alone produce a small-but-positive ``worst`` and
        # the guard falls through. The student's composite then
        # overwrites the king's prior composite with a low score and
        # ``resolve_king`` promotes whatever stale score is on disk.
        # We hit both modes in the 2026-05-16 round_1778892714 post-
        # mortem: 8/10 challengers failed at vLLM load (composite all
        # null, fine) but king 47 + UID 137 vLLM-loaded, ran 16 bench
        # axes, and STILL came back with kl=None — and the bench-only
        # composite was lower than UID 83's stale prior, triggering a
        # bogus dethrone. So we now require BOTH ``worst`` to be
        # computable AND the KL axis to be real (or the model to be
        # explicitly DQ'd) before writing the row.
        kl_axis = (comp.get("axes") or {}).get("kl")
        kl_failed = kl_axis is None and not comp.get("disqualified")
        worst_missing = comp.get("worst") is None and not comp.get("disqualified")
        if worst_missing or kl_failed:
            logger.warning(
                f"uid={uid} ({name}): composite skipped "
                f"(worst={comp.get('worst')!r}, kl_axis={kl_axis!r}, "
                f"dq={comp.get('disqualified')}) — preserving prior record"
            )
        else:
            state.composite_scores[str(uid) if uid is not None else name] = comp

        # Update the cross-round fingerprint store so future rounds see this uid.
        if isinstance(fp, dict) and fp.get("layer_fingerprints"):
            state.activation_fingerprints[str(uid) if uid is not None else name] = {
                "model": name,
                "layer_fingerprints": fp.get("layer_fingerprints"),
                "n_layers": fp.get("n_layers"),
                "hidden_size": fp.get("hidden_size"),
                "commit_block": commit_block,
                "coldkey": coldkey,
                "updated": time.time(),
            }

    # ``king_uid`` / ``king_model`` are required for next-round king
    # resolution (the validator service reads h2h_latest.king_uid as
    # the PRIMARY source for the seated king — see ``_round`` in
    # ``distil.eval.service``). Without them, every restart falls
    # back to the composite-scores top scorer and silently dethrones
    # the seated king. The legacy ``h2h_latest`` schema also exposed
    # both fields to the dashboard.
    king_uid_resolved: int | None = None
    king_model_resolved: str | None = None
    if king_name:
        idx = uid_index.get(king_name) or {}
        if idx.get("uid") is not None:
            king_uid_resolved = int(idx["uid"])
        elif (students.get(king_name) or {}).get("uid") is not None:
            king_uid_resolved = int(students[king_name]["uid"])
        # Strip the @revision suffix to get the bare model id for
        # the dashboard's "king_model" widget.
        king_model_resolved = king_name.split("@", 1)[0]

    # Per-student result rows in the LEGACY ``results[]`` shape — what
    # the dashboard + prod ``/api/h2h-history`` / ``/api/miner/{uid}``
    # / ``/api/king-history`` already index by. Field names match
    # ``scripts/validator/results.py`` so the cutover is byte-shape
    # compatible.
    results: list[dict[str, Any]] = []
    n_prompts = 0
    for name, row in students.items():
        if row.get("is_teacher"):
            continue
        idx = uid_index.get(name) or {}
        uid = idx.get("uid") if idx else row.get("uid")
        hotkey = idx.get("hotkey") if idx else row.get("hotkey")
        comp = composites.get(name)
        is_king = (name == king_name)
        is_ref = (name == reference_name)
        # H2H KL = the global-avg KL on this round's prompts. Lower
        # better. The dashboard's bout-card renders this as the
        # "raw_kl" column in the per-round axis grid.
        kl = row.get("kl_global_avg")
        try:
            kl_val = float(kl) if kl is not None else float("inf")
        except (TypeError, ValueError):
            kl_val = float("inf")
        n_prompts = max(n_prompts, int(row.get("n_prompts") or 0))
        prompts_scored = int(row.get("n_prompts") or 0)
        result_row: dict[str, Any] = {
            "uid": uid,
            "model": (idx.get("model") if idx else None) or name.split("@", 1)[0],
            "kl": kl_val if kl_val != float("inf") else None,
            "is_king": is_king,
            "is_reference": is_ref,
            "prompts_scored": prompts_scored,
            "prompts_total": prompts_scored,
            "paired_prompts": prompts_scored,
            "dethrone_eligible": (not is_king) and (comp or {}).get("disqualified") is not True,
            "early_stopped": False,
            "composite": comp,
            "axes_summary": (comp or {}).get("axes"),
            "hotkey": hotkey,
            "name": name,
        }
        # Phase-2 load failure tracking. The pod-side ``_phase_student``
        # catches HF 404 / vLLM init crash / OSError and writes
        # ``{"name", "uid", "hotkey", "error": "..."}`` to its shard
        # output (no axis payloads). Pre-fix these rows fed
        # ``compute_composite`` which returned ``worst=None`` for every
        # axis — the row was then NOT marked disqualified, NOT added to
        # ``evaluated_uids`` (audit fix preserves the slot for transient
        # blips), and NOT counted as a failure. The same UIDs kept
        # being re-selected by ``select_challengers`` every round,
        # burning ~25 min of teacher API budget + 8×B200 GPU time per
        # round (10 ghost UIDs in 2026-05-16 round 1778935073). We
        # now record a failure counter per UID; after
        # ``settings.max_load_failures`` consecutive strikes we mark
        # the slot consumed so the UID stops crowding out fresh
        # commits. ``evict_stale_evaluated_uids`` resets the counter
        # on honest re-commitment (different model / revision / block).
        load_failed = bool(row.get("error")) and not (comp or {}).get("worst")
        load_succeeded = (comp or {}).get("worst") is not None
        load_failures_after = None
        if uid is not None and not is_ref and not is_king:
            if load_succeeded:
                state.reset_failures(int(uid))
            elif load_failed:
                load_failures_after = state.record_failure(int(uid), name)
                err_short = str(row.get("error") or "")[:200]
                result_row["error"] = err_short
                result_row["status"] = "load_failed"
                result_row["status_detail"] = (
                    f"load_failed ({load_failures_after}/"
                    f"{settings.max_load_failures}): {err_short}"
                )
                dq_events.append({
                    "uid": uid,
                    "name": name,
                    "kind": "load_failed",
                    "strike": load_failures_after,
                    "error": err_short,
                })
        if (comp or {}).get("disqualified"):
            result_row["disqualified"] = True
            result_row["dq_reason"] = (comp or {}).get("dq_reason")
        results.append(result_row)
        # Legacy single-eval policy: mark this UID as having spent its
        # one slot ONLY when the eval actually produced a usable result.
        # ``state.evaluated_uids`` is a list-backed set on disk; the
        # challenger picker (``distil.eval.round.select_challengers``)
        # skips any UID present here, matching prod's
        # ``scripts.validator.single_eval._evict_stale_evaluated_uids``.
        #
        # Pre-fix: every non-reference row was appended unconditionally
        # even when the pod returned ``error`` (HF 404, OOM, transient
        # network, vLLM init crash) — the miner's one shot was burned
        # without ever scoring. The 2026-05-16 audit flagged this as a
        # honest-mistake-becomes-permanent-DQ class of bug. Now we
        # require ``composite_worst`` to be present (the same predicate
        # used by ``process_round`` to decide whether to persist the
        # composite row), so transient infra failures retry on the
        # next round and the slot is only consumed on a real score.
        # Three-strikes load-failure consumption (see failure-tracking
        # block above): after ``max_load_failures`` consecutive Phase-2
        # crashes on the same commitment we burn the slot to stop the
        # UID from monopolising challenger picks forever.
        composite_landed = (comp or {}).get("worst") is not None
        load_exhausted = (
            load_failures_after is not None
            and load_failures_after >= settings.max_load_failures
        )
        slot_consumed = (
            composite_landed
            or (comp or {}).get("disqualified")
            or load_exhausted
        )
        if uid is not None and not is_ref and slot_consumed:
            uid_str = str(uid)
            if uid_str not in state.evaluated_uids:
                state.evaluated_uids.append(uid_str)
            if hotkey and hotkey not in state.evaluated_hotkeys:
                state.evaluated_hotkeys[hotkey] = {
                    "uid": int(uid),
                    "model": result_row["model"],
                    "evaluated_at_ts": time.time(),
                    "evaluated_at_block": block,
                    "composite_final": (comp or {}).get("final"),
                    "composite_worst": (comp or {}).get("worst"),
                    "load_failures": load_failures_after,
                }

    # King KL anchor — the "king_kl" / "king_h2h_kl" / "king_global_kl"
    # the dashboard surfaces. Falls back to the resolved anchor when
    # the seated king didn't report a row this round.
    if king_name and isinstance(students.get(king_name), dict):
        king_row = students[king_name]
        try:
            king_kl_val = float(king_row.get("kl_global_avg") or 0.0)
            if king_kl_val <= 0:
                king_kl_val = king_kl or 0.0
        except (TypeError, ValueError):
            king_kl_val = king_kl or 0.0
    else:
        king_kl_val = king_kl or 0.0

    record = {
        "block": block,
        "block_hash": block_hash,
        # Two timestamp aliases for back-compat: ``ts`` was distil's
        # original name, ``timestamp`` is the legacy h2h_history /
        # /api/h2h-history field name the dashboard reads.
        "ts": time.time(),
        "timestamp": time.time(),
        "type": "single_eval",
        "king_uid": king_uid_resolved,
        "king_name": king_name,
        "king_model": king_model_resolved,
        "king_kl": king_kl_val or None,
        "king_h2h_kl": king_kl_val or None,
        "king_global_kl": king_kl_val or None,
        "reference_name": reference_name,
        "teacher_name": teacher_name,
        "broken_axes": sorted(broken),
        "dq_events": dq_events,
        "n_prompts": n_prompts,
        "n_students": len(results),
        # ``results`` is the legacy field name. ``students`` is kept as
        # an alias so callers that already migrated to the new schema
        # keep working through the dashboard cutover.
        "results": results,
        "students": results,
        "per_bench_timing": timings or [],
    }
    state.append_round(record)
    state.save()
    _refresh_top4(state, composites)
    return record


# ── Top-4 leaderboard ────────────────────────────────────────────────────


def _refresh_top4(state: ValidatorState, composites: dict[str, dict]) -> None:
    # Match the persistence guard in ``process_round``: rows whose KL
    # axis is None (bench-only / failed Phase 2) don't reach
    # ``state.composite_scores``, so they must not bypass that filter
    # into top4_leaderboard either — otherwise the dashboard's "top
    # contenders" panel can briefly show a low-score row that exists
    # in-memory but is *not* part of the dethrone state on disk.
    def _keep(c: dict) -> bool:
        if c.get("final") is None:
            return False
        if c.get("disqualified"):
            return False
        if (c.get("axes") or {}).get("kl") is None:
            return False
        return True

    # Pull the canonical king from ``h2h_latest`` rather than guessing
    # from ``composites`` ordering — the reigning king might be missing
    # from the current round's composites (sparse-axes round, eviction,
    # etc.) and the dashboard would mislabel rank 1 as the king.
    h2h_latest = getattr(state, "h2h_latest", None) or {}
    king_uid_canonical = h2h_latest.get("king_uid")

    # Best-effort UID lookup: ``composites`` is name-keyed
    # (model@revision) but the legacy dashboard wants ``{uid, model}``
    # per row. ``state.composite_scores`` is UID-keyed and stores the
    # ``model`` field on each row, so build a reverse map.
    cs = getattr(state, "composite_scores", None) or {}
    name_to_uid: dict[str, int] = {}
    for uid_str, comp in (cs.items() if isinstance(cs, dict) else []):
        m = (comp or {}).get("model")
        rev = (comp or {}).get("revision")
        if m and rev:
            name_to_uid[f"{m}@{rev}"] = int(uid_str)
        elif m:
            name_to_uid[m] = int(uid_str)

    ranked = sorted(
        ((name, c) for name, c in composites.items() if _keep(c)),
        key=lambda kv: kv[1]["final"],
        reverse=True,
    )[:4]

    def _row(rank: int, name: str, c: dict) -> dict:
        uid = name_to_uid.get(name)
        if uid is None:
            base_name = name.split("@", 1)[0] if "@" in name else name
            uid = name_to_uid.get(base_name)
        return {
            "rank": rank,
            "uid": uid,
            "name": name,
            "model": name.split("@", 1)[0] if "@" in name else name,
            "final": c.get("final"),
            "worst_3_mean": c.get("worst_3_mean"),
            "weighted": c.get("weighted"),
            "present_count": c.get("present_count"),
        }

    rows = [_row(i + 1, n, c) for i, (n, c) in enumerate(ranked)]

    # Legacy schema bridge: the dashboard's ``/api/leaderboard`` and
    # ``/api/miner/{uid}`` endpoints destructure ``top4.king`` and
    # ``top4.contenders`` (UID-bearing dicts) to populate ``is_king``
    # / ``in_top5``. Without these fields ``is_king`` was ``False`` for
    # the actual king and ``in_top5`` was ``False`` for every top-4
    # contender — confirmed in #distil 2026-05-16 ("UID 47 shows
    # is_king:false on the API"). We pick ``king`` from ``h2h_latest``
    # (the canonical seat-of-the-throne record) and ``contenders``
    # from the ranked composites minus the king.
    king_row: dict | None = None
    if king_uid_canonical is not None:
        for r in rows:
            if r.get("uid") == king_uid_canonical:
                king_row = r
                break
        if king_row is None and isinstance(cs, dict):
            kc = cs.get(str(king_uid_canonical)) or {}
            if kc:
                king_row = {
                    "rank": None,
                    "uid": int(king_uid_canonical),
                    "name": (
                        f"{kc.get('model')}@{kc.get('revision')}"
                        if kc.get("model") and kc.get("revision")
                        else kc.get("model")
                    ),
                    "model": kc.get("model"),
                    "final": kc.get("final"),
                    "worst_3_mean": kc.get("worst_3_mean"),
                    "weighted": kc.get("weighted"),
                    "present_count": kc.get("present_count"),
                }
    contenders = [r for r in rows if r is not king_row]

    state.top4_leaderboard = {
        "updated_at": time.time(),
        "rows": rows,
        "king": king_row or {},
        "contenders": contenders,
    }


# ── Composite dethrone floor (single-axis worst-floor gate) ─────────────


def can_dethrone(
    challenger_composite: dict,
    king_composite: dict | None,
    *,
    margin: float | None = None,
) -> tuple[bool, str]:
    """``(allowed, reason)`` — ports prod's ``_composite_dethrone_veto``.

    Fail criteria (in order):
    1. challenger has no ``final`` → no dethrone
    2. no seated king → dethrone (cold start)
    3. < ``composite_dethrone_min_axes`` present on either side → fail open
    4. challenger's single-axis worst < ``composite_dethrone_floor`` → veto
    5. challenger final ≥ king final * (1 + margin) → dethrone

    Otherwise: no dethrone (margin not met).
    """
    m = float(margin if margin is not None else settings.composite_dethrone_margin)
    cf = challenger_composite.get("final") if isinstance(challenger_composite, dict) else None
    if cf is None:
        return False, "challenger_no_final"
    if not isinstance(king_composite, dict) or king_composite.get("final") is None:
        return True, "no_king"
    kf = float(king_composite["final"])
    cf_val = float(cf)
    if (challenger_composite.get("present_count") or 0) < settings.composite_dethrone_min_axes:
        return False, "challenger_too_sparse"
    if (king_composite.get("present_count") or 0) < settings.composite_dethrone_min_axes:
        return True, "king_too_sparse"
    worst = challenger_composite.get("worst")
    if worst is not None and float(worst) < settings.composite_dethrone_floor:
        return False, f"worst_below_floor ({worst:.3f} < {settings.composite_dethrone_floor:.2f})"
    if cf_val >= kf * (1.0 + m):
        return True, f"final_gain {cf_val:.4f} >= king {kf:.4f} * (1+{m:.3f})"
    return False, f"margin_not_met (challenger={cf_val:.4f} king={kf:.4f})"


__all__: Iterable[str] = (
    "process_round",
    "can_dethrone",
    "_check_long_form_derail",
    "_check_activation_copy",
)
