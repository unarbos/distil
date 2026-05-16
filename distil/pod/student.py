"""Phase 3 — per-student vLLM engine + ``prompt_logprobs`` scoring.

Implements improvement #1: instead of the legacy HF
``model(input_ids).logits`` per prompt, we batch all (prompt+teacher
continuation) strings through ``LLM.generate(...)`` with
``SamplingParams(max_tokens=0, prompt_logprobs=K)``. vLLM returns the
per-position top-K logprobs the student would have assigned to those
exact tokens, which is exactly what the sparse top-K KL contract needs.

The student engine also runs the warm-up generate (improvement #3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("distil.pod.student")


@dataclass
class StudentScores:
    """Per-prompt sparse student logprobs + (optional) teacher trace NLL."""

    prompt: str
    prefix_len: int
    student_logprobs: list[dict[int, float]]
    teacher_trace_nll: float | None


def start_student(model_repo: str, vllm_cfg: dict[str, Any]):
    """Spin up a vLLM ``LLM`` for ``model_repo`` using the round's
    ``vllm_cfg`` knobs. If the configured ``max_model_len`` exceeds the
    student model's declared ``max_position_embeddings``, vLLM 0.21+
    raises a Pydantic ValidationError that kills the whole shard
    before we even get to score the student. Honest miners sometimes
    ship checkpoints with shorter native context (8 k, 16 k) than the
    teacher's 32 k window — we should NOT disqualify them for that,
    we should fall back to the model's own max so the round produces
    a valid composite. Setting ``VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`` for
    the duration of this constructor lets vLLM clamp instead of raise;
    if the resulting context can't actually hold our prompts the
    prefix-len guard in :func:`score_against_teacher_trace` will drop
    those positions cleanly.
    """
    import os

    from vllm import LLM, SamplingParams

    prev_allow = os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN")
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    try:
        llm = LLM(
            model=model_repo,
            max_model_len=vllm_cfg.get("max_model_len", 32768),
            enable_chunked_prefill=vllm_cfg.get("enable_chunked_prefill", True),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
            dtype=vllm_cfg.get("dtype", "bfloat16"),
            max_logprobs=vllm_cfg.get("max_logprobs", 128),
            trust_remote_code=True,
        )
    finally:
        if prev_allow is None:
            os.environ.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
        else:
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = prev_allow
    try:
        llm.generate(["hi"], SamplingParams(max_tokens=1, temperature=0.0))
        logger.info("student warm-up generate ok")
    except Exception as exc:
        logger.warning(f"student warm-up generate failed (non-fatal): {exc}")
    return llm


def score_against_teacher_trace(
    llm,
    *,
    prompts: list[str],
    teacher_continuations: list[str],
    teacher_token_ids: list[list[int]],
    prompt_logprobs: int,
) -> list[StudentScores]:
    """Compute per-position student logprobs + NLL on the teacher's exact trace."""
    from vllm import SamplingParams

    full_strings = [p + c for p, c in zip(prompts, teacher_continuations, strict=False)]
    # vLLM >= 0.20 rejects ``max_tokens=0`` with a VLLMValidationError
    # ("max_tokens must be at least 1, got 0"). We only care about
    # prompt_logprobs here — the generated continuation is discarded —
    # so we ask for the smallest legal value (1) and let vLLM emit one
    # throwaway token. This is what the legacy
    # ``scripts/pod_eval_vllm.py`` does for the same reason.
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=prompt_logprobs,
    )
    outs = llm.generate(full_strings, params)
    out: list[StudentScores] = []

    for prompt, _full, tok_ids, o in zip(
        prompts, full_strings, teacher_token_ids, outs, strict=False
    ):
        prefix_len = len(o.prompt_token_ids) - len(tok_ids)
        if prefix_len < 0:
            prefix_len = max(0, len(o.prompt_token_ids) - len(tok_ids))
        per_pos: list[dict[int, float]] = []
        nll_total = 0.0
        nll_n = 0
        for i, pos in enumerate(o.prompt_logprobs or []):
            if i < prefix_len:
                continue
            if pos is None:
                per_pos.append({})
                continue
            mapping = {int(tid): float(lp.logprob) for tid, lp in pos.items()}
            per_pos.append(mapping)
            t_idx = i - prefix_len
            if 0 <= t_idx < len(tok_ids):
                lp = mapping.get(int(tok_ids[t_idx]))
                if lp is not None and lp == lp:
                    nll_total -= lp
                    nll_n += 1
        out.append(
            StudentScores(
                prompt=prompt,
                prefix_len=prefix_len,
                student_logprobs=per_pos,
                teacher_trace_nll=(nll_total / nll_n) if nll_n else None,
            )
        )
    return out
