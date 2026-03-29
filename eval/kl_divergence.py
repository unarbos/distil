"""KL-divergence computation from per-token logprobs."""

import math


def compute_kl_divergence(
    teacher_logprobs: list[dict[str, float]],
    student_logprobs: list[dict[str, float]],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute KL(teacher || student) averaged across token positions.

    Each element is a dict mapping token_string -> log_probability for
    the top-k tokens returned by the model at that generation position.

    For each position:
      1. Take the union of tokens from teacher and student.
      2. Convert logprobs to probs via exp().
      3. Missing tokens receive epsilon probability.
      4. Renormalise both distributions over the union.
      5. KL = Σ p_teacher · log(p_teacher / p_student).
      6. Average across all positions.

    Returns:
        Float KL-divergence (lower ⇒ better distillation).
        Returns inf when no valid positions exist.
    """
    n_positions = min(len(teacher_logprobs), len(student_logprobs))
    if n_positions == 0:
        return float("inf")

    kl_sum = 0.0
    for i in range(n_positions):
        t_lp = teacher_logprobs[i]
        s_lp = student_logprobs[i]

        all_tokens = set(t_lp.keys()) | set(s_lp.keys())
        if not all_tokens:
            continue

        log_eps = math.log(epsilon)

        # Build unnormalised probability vectors
        t_probs: dict[str, float] = {}
        s_probs: dict[str, float] = {}
        for token in all_tokens:
            t_probs[token] = math.exp(t_lp.get(token, log_eps))
            s_probs[token] = math.exp(s_lp.get(token, log_eps))

        # Renormalise
        t_total = sum(t_probs.values())
        s_total = sum(s_probs.values())

        kl = 0.0
        for token in all_tokens:
            p = t_probs[token] / t_total
            q = s_probs[token] / s_total
            if p > 0 and q > 0:
                kl += p * math.log(p / q)

        kl_sum += max(kl, 0.0)  # KL is non-negative by definition

    return kl_sum / n_positions
