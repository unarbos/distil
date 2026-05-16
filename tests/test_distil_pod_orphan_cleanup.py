"""Regression: orphan vLLM EngineCores from prior rounds must be swept
before launching a new orchestrator, and student shards must not fail
on miners whose ``max_position_embeddings`` is smaller than the round's
``max_model_len``.

Two bugs we are guarding against, both surfaced on round 1778905724
(2026-05-16):

1. **Orphan vLLM blocks new round.** When a previous Phase-2 shard
   died ungracefully (OOM, GPU init crash, watchdog SIGKILL), its
   child ``VLLM::EngineCore`` re-parented to ``init`` and survived
   with the model weights still resident in GPU memory. The next
   round's shard couldn't allocate (GPU full), so the king's vLLM
   ``Engine core initialization failed`` and the whole round was
   poisoned. ``run_eval_on_pod`` now sweeps any ``VLLM::EngineCore``
   whose parent is not a live ``distil.pod`` process before launching
   the new orchestrator.

2. **Short-context miner crashes shard.** vLLM 0.21+ raises a Pydantic
   ``ValidationError`` when ``max_model_len`` (32k) exceeds the model's
   declared ``max_position_embeddings`` (e.g. 8k for nimbus-V64).
   That error killed the WHOLE shard, taking out every subsequent
   student assigned to that GPU. ``start_student`` now sets
   ``VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`` around the constructor so vLLM
   clamps to the model's native max instead of raising — short-context
   models just score on whatever positions fit.
"""

from __future__ import annotations

from pathlib import Path

import distil.eval.pod as eval_pod
import distil.pod.student as student


def test_run_eval_on_pod_sweeps_orphan_vllm_before_launch():
    """The pre-launch cleanup must kill stale ``VLLM::EngineCore``
    processes whose parent is no longer a live ``distil.pod`` shard.
    Plain ``pkill -9 -f VLLM::EngineCore`` would be too aggressive (it
    would also kill in-flight shards from a parallel orchestrator).
    """
    src = Path(eval_pod.__file__).read_text()
    # Locate the fresh-start branch (NOT the resume branch).
    fresh_branch = src[src.index("mkdir -p {remote_run}"):src.index("setsid -f python3")]

    assert "VLLM::EngineCore" in fresh_branch, (
        "fresh-start branch must sweep stale VLLM::EngineCore "
        "processes BEFORE launching the orchestrator"
    )
    assert "distil.pod" in fresh_branch, (
        "sweep must distinguish 'parent is distil.pod' (keep) from "
        "'parent is init / something else' (kill) — otherwise we'd "
        "kill the in-flight shards we just spawned"
    )
    assert "pkill -9 -f 'distil.pod.orchestrator'" in fresh_branch, (
        "pre-launch must also kill any stale orchestrator process — "
        "a previous orchestrator that was SIGKILLed by systemd "
        "(unit timeout) can leave a stale process holding the lock"
    )


def test_start_student_allows_long_max_model_len():
    """The vLLM constructor must run with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    so that miners shipping checkpoints with short native context
    (8k, 16k) don't crash the shard with a ValidationError.

    We restore the env to its prior state to keep the host clean.
    """
    src = Path(student.__file__).read_text()
    func = src[src.index("def start_student"):src.index("def score_against_teacher_trace")]

    assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in func, (
        "start_student must set VLLM_ALLOW_LONG_MAX_MODEL_LEN before "
        "constructing LLM() so short-context miners are clamped, not "
        "rejected with a ValidationError that kills the entire shard"
    )
    # The env override must be scoped: prior value restored in finally.
    assert "try:" in func and "finally:" in func, (
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be scoped via try/finally "
        "— leaving it set globally could mask real config bugs in "
        "Phase 3 (judge) or later rounds"
    )
    assert "prev_allow" in func, (
        "the finally branch must restore the prior env value to keep "
        "the pod environment clean across rounds"
    )


def test_start_student_does_not_leak_env_on_success():
    """End-to-end env restoration check (source-level)."""
    src = Path(student.__file__).read_text()
    func = src[src.index("def start_student"):src.index("def score_against_teacher_trace")]
    # Both branches of the finally must be present (None vs preserved).
    assert "os.environ.pop" in func, "must pop env when prev was unset"
    assert "os.environ[\"VLLM_ALLOW_LONG_MAX_MODEL_LEN\"] = prev_allow" in func, (
        "must restore env to prior non-None value"
    )
