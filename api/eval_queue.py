"""Pure helpers for constructing eval queue API payloads."""


def _int_uid(value):
    if isinstance(value, (int, str)) and str(value).lstrip("-").isdigit():
        return int(value)
    return value


def current_model_from_progress(progress: dict):
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    return (
        progress.get("current_model")
        or progress.get("current_student")
        or current.get("student_name")
    )


def completed_sets(progress: dict) -> tuple[set, set]:
    completed_uids = set()
    completed_models = set()
    completed = progress.get("completed")
    if not isinstance(completed, list):
        return completed_uids, completed_models
    for row in completed:
        if isinstance(row, (int, str)) and str(row).lstrip("-").isdigit():
            completed_uids.add(int(row))
        elif isinstance(row, dict):
            student_name = row.get("student_name") or row.get("model")
            if student_name:
                completed_models.add(student_name)
            uid = row.get("uid")
            if isinstance(uid, (int, str)) and str(uid).lstrip("-").isdigit():
                completed_uids.add(int(uid))
    return completed_uids, completed_models


def backlog_pending_by_uid(backlog: dict) -> dict[int, dict]:
    return {
        int(row.get("uid")): row
        for row in (backlog.get("pending") or [])
        if isinstance(row, dict) and str(row.get("uid", "")).lstrip("-").isdigit()
    }


def slot_status(uid, model, *, current_model, completed_uids, completed_models, models_done):
    normalized_uid = _int_uid(uid)
    if normalized_uid in completed_uids or (model and model in completed_models):
        return "done"
    if model and model == current_model:
        return "running"
    if model and model in models_done and (models_done.get(model) or {}).get("status") in ("done", "ok"):
        return "done"
    return "pending"


def build_queue_slots(progress: dict, round_state: dict, backlog: dict) -> list[dict]:
    eval_order = progress.get("eval_order") or []
    models_done = progress.get("models") if isinstance(progress.get("models"), dict) else {}
    models_to_eval = (
        round_state.get("models_to_eval")
        if isinstance(round_state.get("models_to_eval"), dict)
        else {}
    )
    current_model = current_model_from_progress(progress)
    completed_uids, completed_models = completed_sets(progress)
    backlog_pending = backlog_pending_by_uid(backlog)

    slots = []
    slot_uids = set()
    for idx, entry in enumerate(eval_order, start=1):
        uid = entry.get("uid")
        model = entry.get("model")
        role = entry.get("role")
        normalized_uid = _int_uid(uid)
        slot_uids.add(normalized_uid)
        info = models_to_eval.get(str(uid)) or models_to_eval.get(uid) or {}
        backlog_row = (
            backlog_pending.get(normalized_uid)
            if isinstance(normalized_uid, int)
            else {}
        ) or {}
        slots.append({
            "position": idx,
            "uid": uid,
            "model": model,
            "role": role,
            # External API consumers (and the Discord audit bot reading
            # raw ``/api/queue.slots[*].is_king``) expected a boolean
            # field — without it ``is_king: undefined`` rendered as
            # ``false`` for the actual king. Surfacing it explicitly
            # matches both ``role`` (legacy frontend) and ``is_king``
            # (legacy bots) without changing the wire shape otherwise.
            "is_king": role == "king",
            "status": slot_status(
                uid,
                model,
                current_model=current_model,
                completed_uids=completed_uids,
                completed_models=completed_models,
                models_done=models_done,
            ),
            "commit_block": info.get("commit_block") or backlog_row.get("commit_block"),
            "revision": info.get("revision") or backlog_row.get("revision"),
        })

    for row in backlog_pending.values():
        uid = row.get("uid")
        if _int_uid(uid) in slot_uids or row.get("status") != "deferred":
            continue
        slots.append({
            "position": None,
            "uid": uid,
            "model": row.get("model"),
            "role": "challenger",
            "is_king": False,
            "status": "deferred",
            "reason": "Deferred by SINGLE_EVAL_MAX_PER_ROUND FIFO cap; will be retried next round.",
            "commit_block": row.get("commit_block"),
            "revision": row.get("revision"),
        })
    return slots
