"""Pure helpers for miner eval-status classification."""

from eval_queue import current_model_from_progress


def failure_matches_commitment(fail_entry: str, commitment: dict) -> bool:
    repo = (commitment or {}).get("model")
    rev = (commitment or {}).get("revision")
    if not repo or not fail_entry:
        return False
    if "@" in fail_entry:
        failed_repo, failed_rev = fail_entry.split("@", 1)
        return failed_repo == repo and (failed_rev == rev or not rev)
    return fail_entry == repo


def active_slots_by_uid(progress: dict) -> dict:
    return {
        str(entry.get("uid")): {"position": idx, **entry}
        for idx, entry in enumerate(progress.get("eval_order") or [], start=1)
        if entry.get("uid") is not None
    }


def backlog_rows_by_uid(backlog: dict) -> dict:
    return {
        str(row.get("uid")): row
        for row in (backlog.get("pending") or [])
        if isinstance(row, dict) and row.get("uid") is not None
    }


def build_eval_statuses(
    *,
    scores_data: dict,
    dq_data: dict,
    failures_map: dict,
    failure_models_map: dict,
    evaluated_uids,
    uid_map: dict,
    commitments: dict,
    h2h_tracker: dict,
    latest: dict,
    composite_scores: dict,
    progress: dict,
    backlog: dict,
    epoch_blocks: int,
    dq_reason_for_commitment,
) -> tuple[int | None, int, dict]:
    current_king_uid = latest.get("king_uid")
    current_block = latest.get("block", 0)
    evaluated = {str(uid) for uid in (evaluated_uids or [])}
    composite_scores = composite_scores if isinstance(composite_scores, dict) else {}
    active_slots = active_slots_by_uid(progress)
    backlog_rows = backlog_rows_by_uid(backlog)
    current_model = current_model_from_progress(progress)

    result = {}
    uid_keys = set(scores_data) | set(uid_map) | set(composite_scores) | set(active_slots) | set(backlog_rows)
    for uid_str in sorted(uid_keys, key=lambda value: int(value) if str(value).lstrip("-").isdigit() else 10**9):
        if not str(uid_str).lstrip("-").isdigit():
            continue
        uid = int(uid_str)
        hotkey = uid_map.get(uid_str)
        commitment = commitments.get(hotkey) if hotkey else None
        dq_reason = dq_reason_for_commitment(uid, hotkey, commitment, dq_data)
        fail_count = int(failures_map.get(uid_str, 0) or 0)
        fail_model = failure_models_map.get(uid_str)

        if dq_reason is not None:
            result[uid_str] = {"status": "disqualified", "reason": dq_reason}
        elif current_king_uid is not None and uid == current_king_uid:
            result[uid_str] = {"status": "king"}
        elif fail_count >= 3 and fail_model and failure_matches_commitment(fail_model, commitment or {}):
            result[uid_str] = {
                "status": "skipped_stale",
                "failure_count": fail_count,
                "failure_model": fail_model,
            }
        elif uid_str in active_slots:
            result[uid_str] = {
                "status": "running" if active_slots[uid_str].get("model") == current_model else "queued_active_round",
                "position": active_slots[uid_str].get("position"),
                "phase": progress.get("phase"),
            }
        elif (backlog_rows.get(uid_str) or {}).get("status") == "deferred":
            result[uid_str] = {
                "status": "deferred",
                "round_cap": backlog.get("round_cap"),
                "commit_block": backlog_rows[uid_str].get("commit_block"),
            }
        elif uid >= 0 and not commitment:
            result[uid_str] = {"status": "no_commitment"}
        elif uid_str in composite_scores:
            comp = composite_scores.get(uid_str) if isinstance(composite_scores, dict) else {}
            result[uid_str] = {
                "status": "scored",
                "composite_final": comp.get("final") if isinstance(comp, dict) else None,
                "composite_version": comp.get("version") if isinstance(comp, dict) else None,
                "scored_at": comp.get("ts") if isinstance(comp, dict) else None,
            }
        elif uid_str in evaluated:
            result[uid_str] = {"status": "evaluated_no_composite"}
        else:
            tracker_entry = h2h_tracker.get(uid_str, {})
            if tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
                last_block = tracker_entry["block"]
                epochs_since = (current_block - last_block) // epoch_blocks if current_block > last_block else 0
                result[uid_str] = {"status": "tested", "epochs_ago": epochs_since}
            elif uid_str not in scores_data:
                result[uid_str] = {"status": "queued"}
            else:
                result[uid_str] = {"status": "untested"}
    return current_king_uid, current_block, result
