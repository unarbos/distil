"""Phase 1 — vLLM teacher engine.

Hard-coded ``max_model_len=32768`` + ``enable_chunked_prefill=True`` per
improvement #4. Issues a tiny warm-up ``generate(["hi"], max_tokens=1)``
right after construction (improvement #3) so the first real bench isn't
polluted by ~15-30s of FlashInfer JIT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("distil.pod.teacher")


@dataclass
class TeacherOutput:
    """One prompt's teacher continuation + sparse top-K logprobs per position."""

    prompt: str
    continuation: str
    completion_token_ids: list[int]
    completion_logprobs: list[dict[int, float]]


def start_teacher(model_repo: str, vllm_cfg: dict[str, Any]):
    """Construct a vLLM ``LLM`` for the teacher and run the warm-up generate."""
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
        logger.info("teacher warm-up generate ok")
    except Exception as exc:
        logger.warning(f"teacher warm-up generate failed (non-fatal): {exc}")
    return llm


def generate_continuations(
    llm,
    prompts: list[str],
    *,
    max_new_tokens: int,
    top_k: int,
) -> list[TeacherOutput]:
    """Greedy-decode + sparse top-K logprobs per generated position."""
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=top_k,
    )
    outs = llm.generate(prompts, params)
    results: list[TeacherOutput] = []
    for prompt, o in zip(prompts, outs, strict=False):
        gen = o.outputs[0]
        token_ids = list(gen.token_ids)
        per_pos: list[dict[int, float]] = []
        for pos in gen.logprobs or []:
            if pos is None:
                per_pos.append({})
                continue
            per_pos.append({int(tid): float(lp.logprob) for tid, lp in pos.items()})
        results.append(
            TeacherOutput(
                prompt=prompt,
                continuation=gen.text,
                completion_token_ids=token_ids,
                completion_logprobs=per_pos,
            )
        )
    return results
