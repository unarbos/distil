"""Pure progress normalization helpers for API routes."""

import logging
import time

logger = logging.getLogger(__name__)


def normalize_eval_progress(progress):
    if not isinstance(progress, dict):
        return {"active": False}
    normalized = dict(progress)
    current = normalized.get("current") if isinstance(normalized.get("current"), dict) else {}
    current = dict(current)
    fields = {
        "current_student": "student_name",
        "current_prompt": "prompts_done",
        "current_kl": "kl_running_mean",
        "current_best": "best_kl_so_far",
        "current_se": "kl_running_se",
        "current_ci": "ci_95",
        "current_stage": "stage",
        "bench_axis_idx": "bench_axis_idx",
        "bench_axis_total": "bench_axis_total",
    }
    for flat_key, nested_key in fields.items():
        flat_value = normalized.get(flat_key)
        nested_value = current.get(nested_key)
        if flat_value is not None and nested_value is None:
            current[nested_key] = flat_value
        elif flat_value is None and nested_value is not None:
            normalized[flat_key] = nested_value
    if current:
        normalized["current"] = current
    if normalized.get("students_done") is None:
        completed = normalized.get("completed")
        normalized["students_done"] = len(completed) if isinstance(completed, list) else 0
    for key in (
        "phase_detail",
        "progress_fraction",
        "phase_eta_s",
        "teacher_prompts_per_min",
        "elapsed_s",
    ):
        normalized.pop(key, None)
    try:
        now = time.time()
        started_at = float(normalized.get("started_at") or normalized.get("run_started_at") or 0.0)
        elapsed_s = max(0.0, now - started_at) if started_at > 0 else None
        if elapsed_s is not None:
            normalized["elapsed_s"] = round(elapsed_s, 1)

        phase = str(normalized.get("phase") or "")
        teacher_done = normalized.get("teacher_prompts_done")
        prompts_total = normalized.get("prompts_total")
        pod = normalized.get("pod") if isinstance(normalized.get("pod"), dict) else {}
        if pod:
            teacher_done = teacher_done if teacher_done is not None else pod.get("teacher_prompts_done")
            pod_prompts_total = pod.get("prompts_total")
            prompts_total = prompts_total if prompts_total is not None else pod_prompts_total
            pod_effective_total = pod.get("effective_prompts_total") or pod_prompts_total
            if isinstance(pod_effective_total, (int, float)) and pod_effective_total > 0:
                normalized["effective_prompts_total"] = int(pod_effective_total)
            for key in (
                "teacher_started_at",
                "teacher_finished_at",
                "current_student_started_at",
                "original_prompts_total",
                "n_teacher_prompts_total",
                "n_teacher_prompts_with_logprobs",
                "n_teacher_prompts_dropped_missing_logprobs",
            ):
                if normalized.get(key) is None and pod.get(key) is not None:
                    normalized[key] = pod.get(key)

        effective_total = normalized.get("effective_prompts_total")
        if effective_total is None and isinstance(prompts_total, (int, float)) and prompts_total > 0:
            effective_total = int(prompts_total)
        if effective_total is not None:
            normalized["effective_prompts_total"] = int(effective_total)

        if phase in {"api_generating", "teacher_generation", "vllm_generating", "teacher_logits"} and teacher_done is not None and prompts_total:
            done = max(0, int(teacher_done or 0))
            total = max(1, int(prompts_total or 0))
            label = "teacher API generation" if phase == "api_generating" else "teacher generation"
            normalized["phase_detail"] = f"{label} {done}/{total}"
            normalized["progress_fraction"] = min(1.0, done / total)
            teacher_started_at = float(normalized.get("teacher_started_at") or started_at or 0.0)
            phase_elapsed_s = max(0.0, now - teacher_started_at) if teacher_started_at > 0 else None
            if phase_elapsed_s and done > 0:
                rate = done / phase_elapsed_s
                remaining_s = max(0.0, (total - done) / rate) if rate > 0 else None
                normalized["teacher_prompts_per_min"] = round(rate * 60.0, 2)
                if remaining_s is not None:
                    normalized["phase_eta_s"] = round(remaining_s, 1)
        elif normalized.get("students_total"):
            done = int(normalized.get("students_done") or 0)
            total = max(1, int(normalized.get("students_total") or 0))
            normalized["progress_fraction"] = min(1.0, done / total)
            stage = normalized.get("current_stage")
            current_student = normalized.get("current_student")
            detail = f"student scoring {done}/{total}"
            if current_student:
                detail += f" ({current_student}"
                if stage:
                    detail += f": {stage}"
                detail += ")"
            elif stage:
                detail += f" ({stage})"
            normalized["phase_detail"] = detail

            completed = normalized.get("completed")
            durations = []
            if isinstance(completed, list):
                for row in completed:
                    if not isinstance(row, dict):
                        continue
                    elapsed = row.get("elapsed_s")
                    if isinstance(elapsed, (int, float)) and elapsed > 0:
                        durations.append(float(elapsed))
                    else:
                        started = row.get("started_at")
                        finished = row.get("finished_at")
                        if isinstance(started, (int, float)) and isinstance(finished, (int, float)) and finished > started:
                            durations.append(float(finished - started))
            if not durations:
                teacher_finished_at = normalized.get("teacher_finished_at")
                if isinstance(teacher_finished_at, (int, float)) and teacher_finished_at > 0 and done > 0:
                    durations.append(max(0.0, (now - float(teacher_finished_at)) / done))
            if durations and done < total:
                avg_student_s = sum(durations) / len(durations)
                normalized["phase_eta_s"] = round(avg_student_s * (total - done), 1)
    except Exception as exc:
        logger.debug("normalize_eval_progress derived fields failed: %s", exc)
    return normalized


def progress_value(progress, flat_key, nested_key):
    if progress.get(flat_key) is not None:
        return progress.get(flat_key)
    current = progress.get("current")
    if isinstance(current, dict):
        return current.get(nested_key)
    return None
