"""
Full-distribution KL-divergence computation on GPU tensors.

Production approach:
1. Forward pass prompt through both models → KL on prompt positions
2. Generate teacher continuation (greedy, 512 tokens)
3. Forward pass full sequence (prompt + continuation) through both models
4. Compute KL only on continuation positions (after prompt)
5. Return weighted average across all evaluated positions
"""
import torch
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger("distillation.kl")


def compute_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    start_pos: int = 0,
) -> dict:
    """
    Exact KL(teacher || student) from full logit tensors.

    Args:
        teacher_logits: [1, seq_len, vocab_size] or [seq_len, vocab_size]
        student_logits: same shape
        start_pos: compute KL only from this position onward (0 = all positions)

    Returns:
        dict with kl_mean, kl_std, kl_max, kl_min, n_positions
    """
    if teacher_logits.dim() == 3:
        teacher_logits = teacher_logits.squeeze(0)
        student_logits = student_logits.squeeze(0)

    # Slice to requested positions
    if start_pos > 0:
        teacher_logits = teacher_logits[start_pos:]
        student_logits = student_logits[start_pos:]

    # Compute in float32 for numerical stability
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    t_p = t_log_p.exp()

    # KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))
    kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1)

    return {
        "kl_mean": kl_per_pos.mean().item(),
        "kl_std": kl_per_pos.std().item(),
        "kl_max": kl_per_pos.max().item(),
        "kl_min": kl_per_pos.min().item(),
        "n_positions": int(kl_per_pos.shape[0]),
    }


@torch.no_grad()
def evaluate_kl_with_continuation(
    teacher_model,
    student_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Production KL evaluation with teacher continuation.

    Steps:
    1. Generate greedy continuation from teacher
    2. Forward pass full sequence through both models
    3. Compute KL on continuation positions only

    Args:
        teacher_model: loaded teacher model
        student_model: loaded student model
        input_ids: [1, prompt_len] tokenized prompt
        max_new_tokens: teacher continuation length
        device: cuda/cpu

    Returns:
        dict with kl_mean, kl_std, kl_max, kl_min, n_positions, prompt_len, gen_len
    """
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # 1. Generate teacher continuation (greedy)
    teacher_output = teacher_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    gen_len = teacher_output.shape[1] - prompt_len

    if gen_len == 0:
        return {
            "kl_mean": float("inf"),
            "kl_std": 0.0,
            "kl_max": float("inf"),
            "kl_min": float("inf"),
            "n_positions": 0,
            "prompt_len": prompt_len,
            "gen_len": 0,
        }

    # 2. Forward pass both models on full sequence (prompt + continuation)
    full_ids = teacher_output  # [1, prompt_len + gen_len]
    teacher_logits = teacher_model(full_ids).logits.float()  # [1, seq, vocab]
    student_logits = student_model(full_ids).logits.float()  # [1, seq, vocab]

    # 3. KL on continuation positions only
    # logits[i] predicts token at position i+1
    # We want predictions for positions prompt_len..end
    # So we use logits at positions (prompt_len-1)..(end-1)
    t_logits = teacher_logits[:, prompt_len - 1:-1, :]
    s_logits = student_logits[:, prompt_len - 1:-1, :]

    result = compute_kl_from_logits(t_logits, s_logits)
    result["prompt_len"] = prompt_len
    result["gen_len"] = gen_len
    return result


def compute_kl_divergence(
    teacher_logprobs: list[dict[str, float]],
    student_logprobs: list[dict[str, float]],
    epsilon: float = 1e-10,
) -> float:
    """
    Approximate KL(teacher || student) from top-k logprob dicts (CPU fallback).
    Used only in simulation/testing. Production uses full-distribution GPU KL.
    """
    import math

    n = min(len(teacher_logprobs), len(student_logprobs))
    if n == 0:
        return float("inf")

    kl_sum = 0.0
    for i in range(n):
        t_lp = teacher_logprobs[i]
        s_lp = student_logprobs[i]
        all_tokens = set(t_lp.keys()) | set(s_lp.keys())

        t_probs = {t: math.exp(t_lp.get(t, math.log(epsilon))) for t in all_tokens}
        s_probs = {t: math.exp(s_lp.get(t, math.log(epsilon))) for t in all_tokens}

        t_total = sum(t_probs.values())
        s_total = sum(s_probs.values())

        kl = 0.0
        for t in all_tokens:
            p = t_probs[t] / t_total
            q = s_probs[t] / s_total
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
        kl_sum += max(kl, 0.0)

    return kl_sum / n
