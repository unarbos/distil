from scripts.validator import pod_runtime


def test_pod_aux_manifest_includes_required_eval_helpers():
    local_paths = {local for local, _remote in pod_runtime.POD_AUX_MODULES}

    assert "scripts/api_teacher.py" in local_paths
    assert "scripts/eval_prompt_accounting.py" in local_paths
    assert "scripts/eval_progress_io.py" in local_paths
    assert "scripts/eval_policy.py" in local_paths
    assert "scripts/eval_benchmarks.py" in local_paths
    assert "scripts/eval_items.py" in local_paths


def test_progress_metadata_manifest_includes_audit_fields():
    keys = set(pod_runtime.POD_PROGRESS_METADATA_KEYS)

    assert "teacher_started_at" in keys
    assert "teacher_finished_at" in keys
    assert "policy" in keys
    assert "script_revision" in keys
    assert "n_teacher_prompts_dropped_missing_logprobs" in keys
    assert "teacher_logprob_coverage" in keys
