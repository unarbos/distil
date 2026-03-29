"""
KL-Divergence computation for model distillation evaluation.

Two modes:
1. Full-distribution (GPU): compute_kl_from_logits() — exact KL from raw logits
2. Top-k (CPU fallback): compute_kl_divergence() — approximate from top-k logprobs dicts
"""
import math


def compute_kl_from_logits(teacher_logits, student_logits):
    """
    Exact KL(teacher || student) from full logit tensors on GPU.

    Args:
        teacher_logits: torch.Tensor [seq_len, vocab_size] or [1, seq_len, vocab_size]
        student_logits: same shape

    Returns:
        dict with kl_mean, kl_std, kl_max, n_positions
    """
    import torch

    if teacher_logits.dim() == 3:
        teacher_logits = teacher_logits.squeeze(0)
        student_logits = student_logits.squeeze(0)

    t_log_p = torch.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = torch.log_softmax(student_logits.float(), dim=-1)
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


def compute_kl_divergence(
    teacher_logprobs: list[dict[str, float]],
    student_logprobs: list[dict[str, float]],
    epsilon: float = 1e-10,
) -> float:
    """
    Approximate KL(teacher || student) from top-k logprob dicts (CPU fallback).

    Each element is {token: log_probability} for the top-k tokens at that position.
    Missing tokens get epsilon probability. Averaged across positions.
    """
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
