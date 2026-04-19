import logging
import math

from eval.private_pool import dp_noise_for
from eval.scoring import disqualify, is_disqualified, record_failure, reset_failures
from eval.state import ValidatorState, log_event
from scripts.validator.composite import (
    _resolve_king_rkl,
    annotate_h2h_with_composite,
    compute_composite,
)
from scripts.validator.config import ACTIVATION_COPY_THRESHOLD, EPSILON, MAX_KL_THRESHOLD, PAIRED_TEST_ALPHA
from scripts.validator.precheck import check_activation_fingerprint

logger = logging.getLogger("distillation.remote_validator")

MIN_PROMPTS_DETHRONE = 100


def _apply_dp_noise_to_per_prompt(per_prompt, prompt_texts, private_start_idx):
    """Axis A7: inject DP-Laplace noise into per-prompt KL values for prompts
    drawn from the private (reusable-holdout) subset. Public prompt scores are
    untouched.

    per_prompt is a list of floats aligned with prompt_texts. Returns a noised
    copy without mutating the input.
    """
    if not per_prompt or not prompt_texts or private_start_idx is None:
        return per_prompt
    n = min(len(per_prompt), len(prompt_texts))
    if private_start_idx >= n:
        return per_prompt
    out = list(per_prompt)
    for i in range(private_start_idx, n):
        try:
            noise = dp_noise_for(prompt_texts[i])
        except Exception:
            noise = 0.0
        out[i] = max(0.0, float(out[i]) + noise)
    return out


def _paired_t_stats(deltas: list[float]):
    n = len(deltas)
    if n < 2:
        return 0.0, 1.0, 1.0
    mean_delta = sum(deltas) / n
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0, 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0, 0.0
        return 0.0, 1.0, 1.0
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    if se <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0, 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0, 0.0
        return 0.0, 1.0, 1.0
    t_stat = mean_delta / se
    cdf = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    p_one_sided = max(0.0, min(1.0, 1.0 - cdf))
    p_two_sided = max(0.0, min(1.0, 2.0 * min(cdf, 1.0 - cdf)))
    return t_stat, p_one_sided, p_two_sided


def process_results(results, models_to_eval, king_uid, state: ValidatorState, uid_to_hotkey, commitments, n_prompts, current_block, king_kl, epoch_count, is_full_eval, epoch_start_time=None):
    uid_to_model = {uid: model["model"] for uid, model in models_to_eval.items()}
    model_to_uid = {model: uid for uid, model in uid_to_model.items()}
    king_h2h_kl = None
    this_round_uids = set()
    for model_name, student_result in results.get("students", {}).items():
        uid = model_to_uid.get(model_name)
        if uid is None:
            continue
        if models_to_eval.get(uid, {}).get("is_reference", False):
            ref_kl = student_result.get("kl_global_avg", "error")
            logger.info(f"REFERENCE ({model_name}): KL={ref_kl} (baseline — not scored)")
            continue
        if "error" in student_result:
            logger.warning(f"UID {uid} ({model_name}): eval error — {student_result['error']}")
            record_failure(uid, state.failures, state.failure_models, model_name)
            continue
        if student_result.get("functional_copy"):
            copy_of = student_result.get("copy_of", "unknown")
            copy_uid = next((u for u, info in models_to_eval.items() if info["model"] == copy_of), None)
            reason = f"copy: functional copy of {copy_of}" + (f" (UID {copy_uid})" if copy_uid else "") + " — identical logit distribution"
            logger.info(f"UID {uid} ({model_name}): FUNCTIONAL COPY — {reason}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.evaluated_uids.add(str(uid))
            continue
        fingerprint = student_result.get("activation_fingerprint")
        if fingerprint and fingerprint.get("layer_fingerprints"):
            uid_to_commit_block = {
                u: info.get("commit_block")
                for u, info in models_to_eval.items()
                if info.get("commit_block") is not None
            }
            this_commit_block = models_to_eval.get(uid, {}).get("commit_block")
            is_copy, copy_uid, copy_model, orig_uid, orig_model, sim = check_activation_fingerprint(
                model_name, uid, fingerprint, state.state_dir,
                commit_block=this_commit_block,
                uid_to_commit_block=uid_to_commit_block,
            )
            if is_copy:
                if copy_uid == uid:
                    reason = (
                        f"copy: activation-space duplicate of UID {orig_uid} ({orig_model}) — "
                        f"cosine similarity {sim:.6f} > {ACTIVATION_COPY_THRESHOLD}, committed later"
                    )
                    logger.info(f"UID {uid} ({model_name}): ACTIVATION COPY — {reason}")
                    log_event(
                        f"Activation copy detected: UID {uid} is later-committed copy of UID {orig_uid} (sim={sim:.6f})",
                        level="warning", state_dir=str(state.state_dir),
                    )
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    disqualify(hotkey, reason, state.dq_reasons, commit_block=this_commit_block)
                    state.evaluated_uids.add(str(uid))
                    continue
                logger.info(
                    f"UID {uid} ({model_name}): activation match with UID {copy_uid} ({copy_model}) "
                    f"(sim={sim:.6f}) — UID {uid} committed first, NOT disqualifying. UID {copy_uid} "
                    f"will be flagged as the copy when its turn is processed."
                )
        if student_result.get("status") == "fraud_vram":
            reason = student_result.get("reason", "VRAM fraud detected")
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        if student_result.get("status") == "anti_finetune":
            probe = student_result.get("finetune_probe", {}) or {}
            raw_reason = student_result.get("reason") or probe.get("reason") or "anti_finetune"
            detail = raw_reason.split("anti_finetune:", 1)[-1] if "anti_finetune:" in raw_reason else raw_reason
            reason = (
                f"anti-finetune: {detail} "
                f"(loss={probe.get('loss','?')}, "
                f"global_grad={probe.get('global_grad_norm','?')}, "
                f"worst={probe.get('worst_param_type','?')}={probe.get('worst_param_norm','?')}, "
                f"norm_w_max={probe.get('worst_norm_weight','?')}). "
                f"Model cannot be continued-pretrained — see "
                f"https://distil.arbos.life/docs#anti-finetune"
            )
            logger.info(f"UID {uid} ({model_name}): {reason}")
            log_event(
                f"UID {uid} ({model_name}) DQ: anti-finetune ({detail})",
                level="warning", state_dir=str(state.state_dir),
            )
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        speed_flag = student_result.get("speed_flag")
        if speed_flag:
            logger.warning(f"UID {uid} ({model_name}): ⚠️ {speed_flag}")
        kl = student_result.get("kl_global_avg", float("inf"))
        if kl <= 1e-6:
            reason = f"FRAUD: KL={kl:.10f} — model produces identical outputs to teacher"
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        if kl == float("inf") or kl < 0:
            logger.warning(f"UID {uid}: invalid KL={kl}")
            record_failure(uid, state.failures, state.failure_models, model_name)
            continue
        this_round_uids.add(uid)
        if uid == king_uid:
            king_h2h_kl = kl
            state.scores[str(uid)] = kl
            state.evaluated_uids.add(str(uid))
            logger.info(f"UID {uid} ({model_name}): H2H KL={kl:.6f} (king — global score UPDATED)")
            log_event(f"UID {uid}: KL={kl:.6f} (king)", state_dir=str(state.state_dir))
        else:
            per_prompt = student_result.get("per_prompt_kl", [])
            scored_prompts = len(per_prompt) if isinstance(per_prompt, list) and per_prompt else student_result.get("prompts_scored", n_prompts)
            early_stopped = bool(student_result.get("early_stopped", False))
            state.scores[str(uid)] = kl
            if early_stopped and (scored_prompts or 0) < MIN_PROMPTS_DETHRONE:
                logger.info(f"UID {uid} ({model_name}): KL={kl:.6f} (early-stopped, {scored_prompts}/{n_prompts} prompts — NOT marking as evaluated, will retry)")
            else:
                state.evaluated_uids.add(str(uid))
            reset_failures(uid, state.failures)
            logger.info(f"UID {uid} ({model_name}): KL={kl:.6f}")
            vs_info = ""
            if king_h2h_kl is not None and king_h2h_kl > 0:
                pct = (king_h2h_kl - kl) / king_h2h_kl * 100
                vs_info = f", {pct:+.2f}% vs king"
            log_event(f"UID {uid}: KL={kl:.6f}{vs_info}", state_dir=str(state.state_dir))
    if king_uid is not None and king_h2h_kl is None:
        logger.warning(f"King UID {king_uid} did not produce a score — will lose crown to best challenger")
        this_round_scored = set()
        for model_name, student_data in results.get("students", {}).items():
            if "error" not in student_data and student_data.get("kl_global_avg") is not None:
                for uid, info in models_to_eval.items():
                    if info.get("model") == model_name:
                        this_round_scored.add(uid)
                        break
        best_challenger_uid = None
        best_challenger_kl = float("inf")
        for uid in (uid for uid in models_to_eval if uid != king_uid and uid in this_round_scored):
            uid_str = str(uid)
            if uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD and state.scores[uid_str] < best_challenger_kl:
                best_challenger_kl = state.scores[uid_str]
                best_challenger_uid = uid
        if best_challenger_uid is not None:
            logger.info(f"King failed eval — promoting best challenger UID {best_challenger_uid} (KL={best_challenger_kl:.6f}) [fresh score this round]")
            log_event(f"King UID {king_uid} failed to produce score — promoting UID {best_challenger_uid}", level="warning", state_dir=str(state.state_dir))
            king_fail_results = []
            for uid in this_round_scored:
                uid_str = str(uid)
                model_name = uid_to_model.get(uid, "")
                kl = state.scores.get(uid_str)
                if kl and kl > 0:
                    king_fail_results.append({"uid": uid, "model": model_name, "kl": round(kl, 6), "is_king": False, "vs_king": "king_failed"})
            king_fail_results.sort(key=lambda item: item["kl"])
            return best_challenger_uid, best_challenger_kl, king_fail_results, None, None, set(models_to_eval.keys())
        logger.error("King failed eval and no valid challengers produced fresh scores — king retains crown by default")
        log_event(f"King UID {king_uid} failed and no valid challengers with fresh scores", level="error", state_dir=str(state.state_dir))
        return king_uid, king_kl, [], king_h2h_kl, None, set(models_to_eval.keys())
    king_new_kl = king_h2h_kl if king_h2h_kl is not None else state.scores.get(str(king_uid), king_kl) if king_uid else float("inf")
    epsilon_threshold = king_new_kl * (1.0 - EPSILON) if king_uid else float("inf")
    epsilon_dethroned_by = None
    king_model_name = uid_to_model.get(king_uid)
    king_per_prompt = results["students"][king_model_name].get("kl_per_prompt") if king_model_name and king_model_name in results.get("students", {}) else None

    round_info = getattr(state, "current_round", {}) or {}
    prompt_texts_for_dp = round_info.get("prompts") or []
    n_private = int(((round_info.get("private_pool") or {}).get("n") or 0))
    private_start = (len(prompt_texts_for_dp) - n_private) if n_private > 0 else None
    if king_per_prompt is not None and private_start is not None:
        king_per_prompt = _apply_dp_noise_to_per_prompt(king_per_prompt, prompt_texts_for_dp, private_start)
    challengers = {uid: info for uid, info in models_to_eval.items() if uid != king_uid}
    if king_uid is not None and challengers:
        for uid in challengers:
            uid_str = str(uid)
            if uid_str not in state.scores or state.scores[uid_str] <= 0 or state.scores[uid_str] > MAX_KL_THRESHOLD:
                continue
            challenger_kl = state.scores[uid_str]
            challenger_model = uid_to_model.get(uid)
            challenger_per_prompt = results["students"][challenger_model].get("kl_per_prompt") if challenger_model and challenger_model in results.get("students", {}) else None
            if challenger_per_prompt is not None and private_start is not None:
                challenger_per_prompt = _apply_dp_noise_to_per_prompt(challenger_per_prompt, prompt_texts_for_dp, private_start)
            if king_per_prompt and challenger_per_prompt:
                n_paired = min(len(king_per_prompt), len(challenger_per_prompt))
                if n_paired >= MIN_PROMPTS_DETHRONE:
                    deltas = [king_per_prompt[i] - challenger_per_prompt[i] for i in range(n_paired)]
                    mean_delta = sum(deltas) / len(deltas)
                    t_stat, p_value, _ = _paired_t_stats(deltas)
                    pct_better = (mean_delta / king_new_kl * 100) if king_new_kl > 0 else 0
                    if p_value < PAIRED_TEST_ALPHA and mean_delta > 0:
                        logger.info(f"UID {uid} DETHRONED king UID {king_uid}! p={p_value:.6f}, delta={mean_delta:.6f} ({pct_better:.2f}%), t={t_stat:.3f}, n={len(deltas)}")
                        if epsilon_dethroned_by is None or challenger_kl < state.scores.get(str(epsilon_dethroned_by), float("inf")):
                            epsilon_dethroned_by = uid
                    elif mean_delta > 0:
                        logger.info(f"UID {uid}: better but not significant (p={p_value:.4f}, delta={mean_delta:.6f}, n={len(deltas)})")
                    else:
                        logger.info(f"UID {uid}: worse than king (delta={mean_delta:.6f}, p={p_value:.4f}, n={len(deltas)})")
                else:
                    logger.info(f"UID {uid}: insufficient prompts for dethronement ({n_paired} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
            else:
                challenger_n = len(challenger_per_prompt) if challenger_per_prompt else 0
                if challenger_n < MIN_PROMPTS_DETHRONE:
                    logger.info(f"UID {uid}: insufficient prompts for legacy epsilon ({challenger_n} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
                elif challenger_kl < epsilon_threshold:
                    logger.info(f"UID {uid} DETHRONED king UID {king_uid}! KL={challenger_kl:.6f} < {epsilon_threshold:.6f} [legacy epsilon, n={challenger_n}]")
                    if epsilon_dethroned_by is None or challenger_kl < state.scores.get(str(epsilon_dethroned_by), float("inf")):
                        epsilon_dethroned_by = uid
    h2h_candidates = []
    all_round_uids = set([king_uid] + list(challengers.keys())) if king_uid is not None else set(challengers.keys())
    for uid in all_round_uids:
        uid_str = str(uid)
        hotkey = uid_to_hotkey.get(uid, "")
        commit_block = commitments.get(uid, {}).get("block")
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=commit_block):
            continue
        if uid in this_round_uids and uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD:
            h2h_candidates.append((uid, state.scores[uid_str]))

    # ── T2.1: composite-worst as the ranking key ────────────────────────
    # We compute composite up-front for every h2h candidate (plus the king
    # if scored this round) so the canonical "best" is decided by the
    # minimum-axis rule rather than raw KL. The paired t-test gate
    # (``epsilon_dethroned_by``) is unchanged — it still enforces
    # statistical significance before a crown changes hands — but which
    # challenger is considered the canonical winner, and what we display
    # as #1, is now driven by composite.worst.
    students_data = results.get("students", {}) or {}
    try:
        _tmp_h2h = [{"uid": king_uid, "model": uid_to_model.get(king_uid), "is_king": True}] if king_uid else []
        king_rkl_ref = _resolve_king_rkl(king_h2h_kl, students_data, _tmp_h2h)
    except Exception:
        king_rkl_ref = None

    def _composite_for(uid):
        model = uid_to_model.get(uid)
        data = students_data.get(model) or {}
        try:
            return compute_composite(data, king_h2h_kl, king_rkl_ref)
        except Exception:
            return {"worst": None, "weighted": None, "axes": {}, "present_count": 0}

    winner_uid, winner_kl = None, float("inf")
    if h2h_candidates:
        # Primary sort: composite.worst descending (higher-is-better).
        # Ties broken by composite.weighted, then by KL ascending so
        # behaviour degrades gracefully to KL-only when composite is
        # missing (e.g. full-vocab KL still computed but probes errored).
        def _rank_key(item):
            uid_i, kl_i = item
            comp = _composite_for(uid_i)
            worst = comp.get("worst")
            weighted = comp.get("weighted")
            present = comp.get("present_count") or 0
            # Sentinel: composite missing → fall back to KL-only rank.
            if worst is None or present < 2:
                return (0, float("-inf"), float("-inf"), kl_i)
            return (1, worst, weighted if weighted is not None else 0.0, -kl_i)

        h2h_candidates.sort(key=_rank_key, reverse=True)
        best_uid, best_kl = h2h_candidates[0]
        if king_uid is not None and best_uid != king_uid and epsilon_dethroned_by is None:
            winner_uid = king_uid
            winner_kl = state.scores.get(str(king_uid), king_kl)
            logger.info(f"King UID {king_uid} retains crown (no challenger passed paired t-test)")
        elif epsilon_dethroned_by is not None:
            challenger_model = uid_to_model.get(epsilon_dethroned_by, "")
            try:
                from huggingface_hub import HfApi

                info = HfApi().model_info(challenger_model)
                if info.private:
                    logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} is now private!")
                    winner_uid = king_uid
                    winner_kl = state.scores.get(str(king_uid), king_kl)
                    logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                    state.dq_reasons[str(epsilon_dethroned_by)] = "Model went private after scoring"
                    epsilon_dethroned_by = None
                else:
                    winner_uid = epsilon_dethroned_by
                    winner_kl = state.scores.get(str(epsilon_dethroned_by), best_kl)
                    logger.info(f"UID {winner_uid} is new king (paired t-test p<{PAIRED_TEST_ALPHA}), integrity check passed")
            except Exception as exc:
                logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} integrity check failed: {exc}")
                winner_uid = king_uid
                winner_kl = state.scores.get(str(king_uid), king_kl)
                logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                state.dq_reasons[str(epsilon_dethroned_by)] = "Model not accessible on HuggingFace"
                epsilon_dethroned_by = None
        else:
            winner_uid, winner_kl = best_uid, best_kl
    h2h_results = _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl, king_per_prompt, uid_to_model, state.dq_reasons, uid_to_hotkey, commitments)
    try:
        annotate_h2h_with_composite(h2h_results, king_h2h_kl, students_data)
        # Re-sort h2h_results by composite.worst (desc) so the leaderboard
        # endpoint and h2h_latest display rank order matches the ranking
        # key used for crown decisions. KL stays as an informational field
        # on each row.
        def _h2h_sort_key(row):
            comp = row.get("composite") or {}
            worst = comp.get("worst")
            if worst is None:
                return (0, float("-inf"), -(row.get("kl") or float("inf")))
            return (1, worst, -(row.get("kl") or float("inf")))
        h2h_results.sort(key=_h2h_sort_key, reverse=True)

        for entry in h2h_results:
            comp = entry.get("composite") or {}
            worst = comp.get("worst")
            if worst is not None:
                axes = comp.get("axes", {})
                axes_str = " ".join(f"{k}={v:.2f}" if v is not None else f"{k}=–"
                                    for k, v in axes.items())
                logger.info(
                    f"  composite UID {entry.get('uid')}: "
                    f"worst={worst:.3f} weighted={comp.get('weighted')} [{axes_str}]"
                )
    except Exception as exc:
        logger.warning(f"composite annotation failed: {exc}")
    logger.info(f"H2H ROUND RESULTS (block {current_block}):")
    for rank, (uid, kl) in enumerate(h2h_candidates, 1):
        marker = " ← WINNER" if uid == winner_uid else ""
        is_king = " (king)" if uid == king_uid else ""
        logger.info(f"  #{rank}  UID {uid}: KL={kl:.6f}{marker}{is_king}")
    logger.info("GLOBAL LEADERBOARD:")
    sorted_scores = sorted(state.scores.items(), key=lambda item: item[1])
    for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
        uid = int(uid_str)
        hotkey = uid_to_hotkey.get(uid, "")
        commit_block = commitments.get(uid, {}).get("block")
        dq = " ⛔ DQ" if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=commit_block) else ""
        marker = " ← H2H WINNER" if uid == winner_uid else ""
        in_round = " (in round)" if uid in all_round_uids else ""
        logger.info(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{in_round}{dq}")
    return winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, this_round_uids


def _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl, king_per_prompt, uid_to_model,
                       dq_reasons=None, uid_to_hotkey=None, commitments=None):
    h2h_results = []
    prompts_total = results.get("n_prompts")
    dq_reasons = dq_reasons or {}
    uid_to_hotkey = uid_to_hotkey or {}
    commitments = commitments or {}
    for uid, info in models_to_eval.items():
        model_name = info["model"]
        student_data = results.get("students", {}).get(model_name, {})
        kl = student_data.get("kl_global_avg")
        if kl is None or "error" in student_data:
            continue
        is_king = uid == king_uid
        hotkey = info.get("hotkey") or uid_to_hotkey.get(uid, "")
        commit_block = info.get("commit_block") or (commitments.get(uid, {}) or {}).get("block")
        dq_key = f"{hotkey}:{commit_block}" if hotkey and commit_block is not None else hotkey
        dq_reason = dq_reasons.get(dq_key) or (dq_reasons.get(hotkey) if hotkey else None) or dq_reasons.get(str(uid))
        is_dq = bool(dq_reason) and not is_king
        vs_king = ""
        t_test_info = None
        challenger_per_prompt = student_data.get("kl_per_prompt")
        prompts_scored = len(challenger_per_prompt) if isinstance(challenger_per_prompt, list) else student_data.get("prompts_scored")
        paired_prompts = min(len(king_per_prompt), len(challenger_per_prompt)) if king_per_prompt and challenger_per_prompt else prompts_scored
        dethrone_eligible = bool(is_king or (paired_prompts is not None and paired_prompts >= MIN_PROMPTS_DETHRONE))
        if king_h2h_kl is not None and not is_king and king_h2h_kl > 0:
            pct = (king_h2h_kl - kl) / king_h2h_kl * 100
            if king_per_prompt and challenger_per_prompt:
                n_paired = min(len(king_per_prompt), len(challenger_per_prompt))
                deltas = [king_per_prompt[i] - challenger_per_prompt[i] for i in range(n_paired)]
                mean_d = sum(deltas) / len(deltas) if deltas else 0.0
                if n_paired > 1:
                    t_s, p_val, _ = _paired_t_stats(deltas)
                    t_test_info = {"p": round(p_val, 6), "t": round(t_s, 3), "n": n_paired, "mean_delta": round(mean_d, 6)}
                else:
                    t_s, p_val = 0.0, 1.0
                if n_paired < MIN_PROMPTS_DETHRONE:
                    vs_king = f"-{pct:.3f}% ({n_paired}p, need {MIN_PROMPTS_DETHRONE}p)" if mean_d > 0 else "worse"
                elif p_val < PAIRED_TEST_ALPHA and mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f} dethroned)"
                elif mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f}, not significant)"
                else:
                    vs_king = "worse"
            else:
                epsilon_threshold_h2h = king_h2h_kl * (1.0 - EPSILON)
                challenger_n = prompts_scored or 0
                if challenger_n < MIN_PROMPTS_DETHRONE and kl < king_h2h_kl:
                    vs_king = f"-{pct:.3f}% ({challenger_n}p, need {MIN_PROMPTS_DETHRONE}p)"
                elif kl < epsilon_threshold_h2h:
                    vs_king = f"-{pct:.3f}% (dethroned)"
                elif kl < king_h2h_kl:
                    vs_king = f"-{pct:.3f}% (not enough, need >{EPSILON * 100:.0f}%)"
                else:
                    vs_king = "worse"
        if is_dq:
            short = (dq_reason or "").strip()[:140]
            vs_king = f"DQ — not crowned ({short})"
            dethrone_eligible = False
        entry = {
            "uid": uid,
            "model": model_name,
            "kl": round(kl, 6),
            "is_king": is_king,
            "vs_king": vs_king,
            "prompts_scored": prompts_scored,
            "prompts_total": prompts_total,
            "paired_prompts": paired_prompts,
            "dethrone_eligible": dethrone_eligible,
            "early_stopped": bool(student_data.get("early_stopped", False)),
        }
        if is_dq:
            entry["disqualified"] = True
            entry["dq_reason"] = dq_reason
        if t_test_info:
            entry["t_test"] = t_test_info
        if info.get("is_reference"):
            entry["is_reference"] = True
            entry["vs_king"] = "baseline (undistilled)"
        h2h_results.append(entry)
    h2h_results.sort(key=lambda item: item["kl"])
    return h2h_results
