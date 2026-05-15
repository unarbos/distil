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
    # 2026-05-15: ``student_vllm`` powers the vLLM-backed student
    # generation path used by the bench battery (--use-vllm-students /
    # DISTIL_STUDENT_USE_VLLM=1). Without this entry the pod's
    # `import student_vllm` raises and pod_eval.py logs "No module
    # named 'student_vllm'" → silently falls back to HF transformers,
    # turning a ~1 h round into a 4-5 h round. The legacy pod_runtime.py
    # has had this entry since 2026-05-14; reintroducing it here so the
    # active validator tree uploads it on every fresh pod run. Includes
    # the max_position_embeddings clamp added 2026-05-15.
    ("scripts/student_vllm.py", "student_vllm.py"),
    # 2026-05-15: parallel orchestrator + watchdog. Optional companion
    # to pod_eval.py used by validators that want multi-GPU fan-out
    # (8xB200 etc.). pod_session.py invokes it instead of pod_eval.py
    # when DISTIL_USE_PARALLEL_ORCH=1 — see the conditional at the
    # eval-launch site. Even when unused, uploading it is cheap (~25 KB)
    # and lets operators flip the flag without redeploying the pod.
    ("scripts/parallel_orchestrator.py", "parallel_orchestrator.py"),
    # v31 procedural axis package (2026-05-09). The v31 generators
    # are imported on the pod via ``from scripts.v31.<axis> import
    # generate_items`` so they need to land at
    # ``<run_dir>/scripts/v31/<axis>.py`` with the corresponding
    # ``__init__.py`` shells. Each entry is a (local_src, remote_path)
    # pair; ``upload_aux_modules`` creates intermediate dirs as
    # needed and falls back gracefully if an entry is missing.
    ("scripts/v31/__init__.py", "scripts/v31/__init__.py"),
    ("scripts/v31/math_gsm_symbolic.py", "scripts/v31/math_gsm_symbolic.py"),
    ("scripts/v31/ifeval_verifiable.py", "scripts/v31/ifeval_verifiable.py"),
    ("scripts/v31/math_competition.py", "scripts/v31/math_competition.py"),
    ("scripts/v31/math_robustness.py", "scripts/v31/math_robustness.py"),
    ("scripts/v31/code_humaneval_plus.py", "scripts/v31/code_humaneval_plus.py"),
    ("scripts/v31/reasoning_logic_grid.py", "scripts/v31/reasoning_logic_grid.py"),
    ("scripts/v31/reasoning_dyval_arith.py", "scripts/v31/reasoning_dyval_arith.py"),
    ("scripts/v31/long_context_ruler.py", "scripts/v31/long_context_ruler.py"),
    ("scripts/v31/knowledge_multi_hop_kg.py", "scripts/v31/knowledge_multi_hop_kg.py"),
    ("scripts/v31/truthfulness_calibration.py", "scripts/v31/truthfulness_calibration.py"),
    ("scripts/v31/consistency_paraphrase.py", "scripts/v31/consistency_paraphrase.py"),
)


# Empty package marker for the synthetic ``scripts/`` package that
# wraps the v31 modules on the pod. We can't reuse the local
# ``scripts/__init__.py`` (the real one bundles validator-only
# imports) so we synthesise an empty one. The path must match the
# remote-side prefix produced by ``upload_aux_modules`` for the
# ``scripts.v31.<axis>`` import to resolve.
POD_PACKAGE_INIT_FILES: tuple[str, ...] = (
    "scripts/__init__.py",
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
    """Upload pod-side helper modules if they exist locally.

    Handles both flat-file aux modules (uploaded directly under
    ``run_dir``) and packaged paths like ``scripts/v31/foo.py``.
    The Lium SFTP backend uses raw ``sftp.put`` which does NOT
    auto-create intermediate directories, so we ``mkdir -p`` the
    parent of every nested path before uploading. We also
    synthesise empty ``__init__.py`` files for any
    ``POD_PACKAGE_INIT_FILES`` entry so ``import scripts.v31.*``
    resolves on the pod side.
    """
    nested_dirs: set[str] = set()
    for _, remote_name in POD_AUX_MODULES:
        parent = os.path.dirname(remote_name)
        if parent:
            nested_dirs.add(parent)
    for synth_path in POD_PACKAGE_INIT_FILES:
        parent = os.path.dirname(synth_path)
        if parent:
            nested_dirs.add(parent)
    if nested_dirs:
        mkdir_cmd = " && ".join(
            f"mkdir -p {run_dir}/{d}" for d in sorted(nested_dirs)
        )
        try:
            pod.exec(mkdir_cmd, timeout=60)
        except Exception as exc:
            logger.warning("Failed to mkdir nested aux dirs %s: %s", nested_dirs, exc)

    for local_aux, remote_name in POD_AUX_MODULES:
        if not os.path.isfile(local_aux):
            continue
        try:
            pod.upload(local_aux, f"{run_dir}/{remote_name}", max_attempts=3)
        except Exception as exc:
            logger.warning("Failed to upload %s (pod features may skip): %s", local_aux, exc)

    import tempfile
    for synth_path in POD_PACKAGE_INIT_FILES:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as fh:
                fh.write(
                    "# Synthesised by scripts.validator.pod_runtime so the\n"
                    "# pod can import ``scripts.v31.<axis>`` without dragging\n"
                    "# in the validator-only top-level scripts package.\n"
                )
                tmp_path = fh.name
            pod.upload(tmp_path, f"{run_dir}/{synth_path}", max_attempts=3)
        except Exception as exc:
            logger.warning("Failed to write pod package marker %s: %s", synth_path, exc)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
