"""``distil verify-kl`` — one-shot HF↔vLLM scorer parity check (improvement #1).

Compares vLLM's ``prompt_logprobs`` (the new fast scorer path) against an
HF reference computation on the same prompts. Asserts ``max delta < 1e-4``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("distil.cli.verify_kl")

_FIXED_PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "Write a one-line haiku about autumn.",
    "def factorial(n):",
    "The Pythagorean theorem states that",
)


def _hf_logprobs(model_repo: str, prompts: list[str], top_k: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_repo, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = mdl(ids).logits[0]
        logp = torch.log_softmax(logits, dim=-1)
        per_pos = []
        for t in range(logp.shape[0]):
            top = torch.topk(logp[t], k=top_k)
            per_pos.append(
                {
                    int(i): float(v)
                    for i, v in zip(top.indices.tolist(), top.values.tolist(), strict=False)
                }
            )
        out.append(per_pos)
    return out


def _vllm_logprobs(model_repo: str, prompts: list[str], top_k: int):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_repo, max_model_len=4096, dtype="bfloat16")
    params = SamplingParams(max_tokens=0, temperature=0.0, prompt_logprobs=top_k)
    outs = llm.generate(prompts, params)
    out = []
    for o in outs:
        per_pos = []
        for pos in o.prompt_logprobs or []:
            if pos is None:
                per_pos.append({})
                continue
            per_pos.append({int(tid): float(lp.logprob) for tid, lp in pos.items()})
        out.append(per_pos)
    return out


def _max_delta(a, b) -> float:
    import math

    worst = 0.0
    for ra, rb in zip(a, b, strict=False):
        for pa, pb in zip(ra, rb, strict=False):
            for tid in set(pa) & set(pb):
                worst = max(worst, abs(math.exp(pa[tid]) - math.exp(pb[tid])))
    return worst


def run(model_repo: str, n_prompts: int = 100, max_delta: float = 1e-4) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    n = max(1, n_prompts)
    prompts = [_FIXED_PROMPTS[i % len(_FIXED_PROMPTS)] + f" [{i}]" for i in range(n)]

    logger.info(f"HF reference logprobs on {n} prompts …")
    hf = _hf_logprobs(model_repo, prompts, top_k=10)
    logger.info(f"vLLM prompt_logprobs on {n} prompts …")
    vl = _vllm_logprobs(model_repo, prompts, top_k=10)
    delta = _max_delta(hf, vl)
    logger.info(f"max per-position prob delta: {delta:.2e}")
    if delta > max_delta:
        print(f"FAIL — delta {delta:.2e} > threshold {max_delta:.2e}")
        return 1
    print(f"OK — delta {delta:.2e} <= threshold {max_delta:.2e}")
    return 0
