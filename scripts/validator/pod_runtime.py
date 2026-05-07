"""Small pod-runtime manifests shared by validator pod sessions."""

import os


POD_AUX_MODULES: tuple[tuple[str, str], ...] = (
    ("scripts/ifeval_vendor.py", "ifeval_vendor.py"),
    ("scripts/humaneval_sandbox.py", "humaneval_sandbox.py"),
    ("scripts/api_teacher.py", "api_teacher.py"),
    ("scripts/eval_prompt_accounting.py", "eval_prompt_accounting.py"),
    ("scripts/eval_progress_io.py", "eval_progress_io.py"),
    ("scripts/eval_policy.py", "eval_policy.py"),
    ("scripts/eval_benchmarks.py", "eval_benchmarks.py"),
    ("scripts/eval_items.py", "eval_items.py"),
)


CURRENT_FIELD_MAP: tuple[tuple[str, str, object], ...] = (
    ("current_student", "student_name", None),
    ("current_prompt", "prompts_done", 0),
    ("current_kl", "kl_running_mean", None),
    ("current_se", "kl_running_se", None),
    ("current_ci", "ci_95", None),
    ("current_best", "best_kl_so_far", None),
    ("current_stage", "stage", None),
    ("bench_axis_idx", "bench_axis_idx", None),
    ("bench_axis_total", "bench_axis_total", None),
    ("current_student_started_at", "student_started_at", None),
)


POD_PROGRESS_METADATA_KEYS: tuple[str, ...] = (
    "run_started_at",
    "teacher_started_at",
    "teacher_finished_at",
    "original_prompts_total",
    "effective_prompts_total",
    "effective_prompts_hash",
    "teacher_mode",
    "teacher_api",
    "policy",
    "script_revision",
    "n_teacher_prompts_total",
    "n_teacher_prompts_with_logprobs",
    "n_teacher_prompts_dropped_missing_logprobs",
    "teacher_logprob_coverage",
)


def upload_aux_modules(pod, run_dir: str, logger) -> None:
    """Upload pod-side helper modules if they exist locally."""
    for local_aux, remote_name in POD_AUX_MODULES:
        if not os.path.isfile(local_aux):
            continue
        try:
            pod.upload(local_aux, f"{run_dir}/{remote_name}", max_attempts=3)
        except Exception as exc:
            logger.warning("Failed to upload %s (pod features may skip): %s", local_aux, exc)
