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
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_repo,
        max_model_len=vllm_cfg.get("max_model_len", 32768),
        enable_chunked_prefill=vllm_cfg.get("enable_chunked_prefill", True),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
        dtype=vllm_cfg.get("dtype", "bfloat16"),
        max_logprobs=vllm_cfg.get("max_logprobs", 128),
        trust_remote_code=True,
    )
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
    params = SamplingParams(
        max_tokens=0,
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
