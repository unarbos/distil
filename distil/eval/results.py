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
    """Pick the anchor for relative axes (king if seated, else round-min)."""
    if king_name and (kr := rows.get(king_name)) is not None:
        v = kr.get(key)
        try:
            f = float(v)
            if f > 0 and f == f and f not in (float("inf"), float("-inf")):
                return f
        except (TypeError, ValueError):
            pass
    best: float | None = None
    for row in rows.values():
        if row.get("is_teacher"):
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
        opr = (row or {}).get("on_policy_rkl") or {}
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
    students: dict[str, dict] = dict(pod_results)
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
        if state.is_disqualified(hotkey, uid=int(uid) if uid is not None else None):
            comp["disqualified"] = True
            comp["dq_reason"] = state.dq_reason(hotkey, uid=int(uid) if uid is not None else None)
        composites[name] = comp
        state.composite_scores[name] = comp

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

    record = {
        "block": block,
        "block_hash": block_hash,
        "ts": time.time(),
        "king_name": king_name,
        "reference_name": reference_name,
        "teacher_name": teacher_name,
        "broken_axes": sorted(broken),
        "dq_events": dq_events,
        "students": [
            {
                "name": name,
                "uid": (uid_index.get(name) or {}).get("uid") or students[name].get("uid"),
                "hotkey": (uid_index.get(name) or {}).get("hotkey") or students[name].get("hotkey"),
                "is_king": (name == king_name),
                "is_reference": (name == reference_name),
                "composite": composites.get(name),
                "axes_summary": composites.get(name, {}).get("axes"),
            }
            for name in students
            if not students[name].get("is_teacher")
        ],
        "per_bench_timing": timings or [],
    }
    state.append_round(record)
    state.save()
    _refresh_top4(state, composites)
    return record


# ── Top-4 leaderboard ────────────────────────────────────────────────────


def _refresh_top4(state: ValidatorState, composites: dict[str, dict]) -> None:
    ranked = sorted(
        (
            (name, c) for name, c in composites.items()
            if c.get("final") is not None and not c.get("disqualified")
        ),
        key=lambda kv: kv[1]["final"],
        reverse=True,
    )[:4]
    state.top4_leaderboard = {
        "updated_at": time.time(),
        "rows": [
            {
                "rank": i + 1,
                "name": name,
                "final": c.get("final"),
                "worst_3_mean": c.get("worst_3_mean"),
                "weighted": c.get("weighted"),
                "present_count": c.get("present_count"),
            }
            for i, (name, c) in enumerate(ranked)
        ],
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
