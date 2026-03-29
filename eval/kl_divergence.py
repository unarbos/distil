"""
Full-distribution KL-divergence computation on GPU tensors.

Production approach:
1. Pre-generate teacher continuations ONCE per epoch (cached on CPU)
2. For each student: forward pass full sequence, compute KL on continuation positions
3. Only continuation logits are kept (memory efficient)

Key optimization: teacher continuations are generated once and reused for all
students, reducing teacher generation from O(students × prompts) to O(prompts).
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
def generate_teacher_continuations(
    teacher_model,
    input_ids_list: list[torch.Tensor],
    max_new_tokens: int = 512,
    block_seed: Optional[int] = None,
    device: str = "cuda",
) -> list[dict]:
    """
    Pre-generate teacher continuations for all prompts in an epoch.

    Called ONCE per epoch, results cached and reused for all student evaluations.
    This reduces teacher generation from O(students × prompts) to O(prompts).

    Args:
        teacher_model: loaded teacher model (on GPU)
        input_ids_list: list of [1, prompt_len] tokenized prompts
        max_new_tokens: continuation length
        block_seed: if provided, use seeded sampling (temperature=0.7, top_p=0.9)
        device: cuda/cpu

    Returns:
        List of dicts, each with:
            - full_ids: [1, prompt_len + gen_len] tensor (on device)
            - teacher_logits: [1, gen_len, vocab_size] continuation logits (on CPU)
            - prompt_len: int
            - gen_len: int
    """
    cache = []
    for i, input_ids in enumerate(input_ids_list):
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # Generate teacher continuation
        gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True)
        if block_seed is not None:
            torch.manual_seed(block_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(block_seed)
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        teacher_output = teacher_model.generate(input_ids, **gen_kwargs)
        gen_len = teacher_output.shape[1] - prompt_len

        if gen_len == 0:
            cache.append({
                "full_ids": teacher_output,
                "teacher_logits": None,
                "prompt_len": prompt_len,
                "gen_len": 0,
            })
            continue

        # Forward pass for teacher logits
        full_ids = teacher_output  # [1, prompt_len + gen_len]
        teacher_logits_full = teacher_model(full_ids).logits

        # Only keep continuation logits (memory efficient) — slice BEFORE .cpu()
        # logits[i] predicts token i+1, so logits[prompt_len-1:-1] predicts continuation
        teacher_cont_logits = teacher_logits_full[:, prompt_len - 1:-1, :].float().cpu()

        cache.append({
            "full_ids": full_ids,  # keep on device for student forward pass
            "teacher_logits": teacher_cont_logits,  # CPU to save GPU memory
            "prompt_len": prompt_len,
            "gen_len": gen_len,
        })

        logger.debug(
            f"  Teacher continuation {i}: {prompt_len} prompt + {gen_len} gen tokens"
        )

    return cache


@torch.no_grad()
def evaluate_student_kl(
    student_model,
    teacher_cache_entry: dict,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a student model against cached teacher continuation data.

    Uses pre-generated teacher continuations to avoid redundant teacher inference.

    Args:
        student_model: loaded student model
        teacher_cache_entry: dict from generate_teacher_continuations()
        device: cuda/cpu

    Returns:
        dict with kl_mean, kl_std, kl_max, kl_min, n_positions, prompt_len, gen_len
    """
    prompt_len = teacher_cache_entry["prompt_len"]
    gen_len = teacher_cache_entry["gen_len"]

    if gen_len == 0 or teacher_cache_entry["teacher_logits"] is None:
        return {
            "kl_mean": float("inf"),
            "kl_std": 0.0,
            "kl_max": float("inf"),
            "kl_min": float("inf"),
            "n_positions": 0,
            "prompt_len": prompt_len,
            "gen_len": 0,
        }

    full_ids = teacher_cache_entry["full_ids"].to(device)

    # Student forward pass on the full sequence
    student_logits_full = student_model(full_ids).logits

    # Only keep continuation logits — slice BEFORE moving to avoid full-seq on CPU
    student_cont_logits = student_logits_full[:, prompt_len - 1:-1, :].float()

    # Move teacher logits to device for KL computation
    teacher_cont_logits = teacher_cache_entry["teacher_logits"].to(device)

    # Compute KL on continuation positions
    result = compute_kl_from_logits(teacher_cont_logits, student_cont_logits)
    result["prompt_len"] = prompt_len
    result["gen_len"] = gen_len
    return result


@torch.no_grad()
def evaluate_kl_with_continuation(
    teacher_model,
    student_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    device: str = "cuda",
    block_seed: Optional[int] = None,
) -> dict:
    """
    Legacy single-call KL evaluation with teacher continuation.
    Kept for backward compatibility. For production, use
    generate_teacher_continuations() + evaluate_student_kl() instead.

    Steps:
    1. Generate continuation from teacher (block-seeded sampling for anti-gaming)
    2. Forward pass full sequence through both models
    3. Compute KL on continuation positions only
    """
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # 1. Generate teacher continuation
    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True)
    if block_seed is not None:
        torch.manual_seed(block_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(block_seed)
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    teacher_output = teacher_model.generate(input_ids, **gen_kwargs)
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
    full_ids = teacher_output
    teacher_logits = teacher_model(full_ids).logits
    student_logits = student_model(full_ids).logits

    # 3. KL on continuation positions only — slice before .float() for memory
    t_logits = teacher_logits[:, prompt_len - 1:-1, :].float()
    s_logits = student_logits[:, prompt_len - 1:-1, :].float()

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
