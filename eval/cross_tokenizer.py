"""Cross-tokenizer round-trip helper for the Stage-2 Kimi K2.6 teacher swap.

Path A from the [Kimi K2.6 runbook](../reports/2026-04-29-kimi-k2.6-stage2-runbook.md):
re-tokenize Kimi K2.6 generations through the Qwen tokenizer so the
existing student-side eval pipeline (KL, RKL, top-K overlap, IS-KL,
EOPD, on-policy RKL, capability) operates entirely in the Qwen 248,320
vocab. Lossy where token alignment is imperfect (different BPE
schemes); telemetry surfaces a per-prompt drift score so we can
quantify the loss.

This module does NOT do cross-tokenizer logit distillation (Path B,
Universal Logit Distillation / ALM, NeurIPS 2025). Path B is
implemented in a separate ``eval/cross_tokenizer_alm.py`` shipped
when the Stage-2 experiment passes Path A but fails capacity-gap
gates.

Usage flow during the experiment:

  1. Kimi K2.6 vLLM server returns its native-tokenizer continuations
     for each prompt (via ``--max-logprobs 128``).
  2. ``decode_with_kimi_tokenizer`` converts Kimi token IDs → text.
  3. ``retokenize_to_qwen`` converts text → Qwen token IDs.
  4. ``align_logprobs_qwen_to_kimi`` maps Kimi's top-128 logprobs onto
     the closest Qwen-vocab tokens, with a drift score per position
     for telemetry.
  5. Returned record has the same shape as a Qwen-teacher cache, so
     downstream eval code is unchanged.

Engineering status: scaffolding (this module). Real Kimi tokenizer
integration lands when we provision a multi-GPU pod and pull the K2.6
weights. Until then this module's smoke tests verify the round-trip
shape with a Qwen-tokenizer mock.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("distillation.cross_tokenizer")


def decode_with_kimi_tokenizer(token_ids: list[int], kimi_tokenizer) -> str:
    """Decode Kimi K2.6 token IDs to natural-language text.

    Args:
        token_ids: list of Kimi-vocab token IDs.
        kimi_tokenizer: HF AutoTokenizer instance for Kimi K2.6.

    Returns:
        Decoded UTF-8 text. Special tokens are kept (the round-trip
        through Qwen will re-encode them or fall back to a similar
        Qwen control token).
    """
    if not token_ids or kimi_tokenizer is None:
        return ""
    try:
        return kimi_tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        logger.warning(
            f"Kimi tokenizer decode failed on {len(token_ids)} tokens: {exc}"
        )
        return ""


def retokenize_to_qwen(text: str, qwen_tokenizer) -> list[int]:
    """Re-tokenize Kimi-decoded text using the Qwen tokenizer.

    The result is what we'd have gotten if Qwen had emitted that
    text directly. Because the two tokenizers segment text differently,
    the token count typically changes by ±5-15% on natural language and
    ±20-30% on code. We don't try to align tokens 1-to-1 — the
    student-side eval grades on text-level outputs (math answer match,
    code sandbox pass, etc.) so the boundary semantics are preserved.

    Args:
        text: decoded text from Kimi.
        qwen_tokenizer: HF AutoTokenizer for Qwen3.5/3.6 (same vocab
            as the SN97 student tokenizer, 248,320).

    Returns:
        Qwen token IDs covering the same text. Empty list on
        tokenizer failure.
    """
    if not text or qwen_tokenizer is None:
        return []
    try:
        ids = qwen_tokenizer(
            text, add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        return list(ids)
    except Exception as exc:
        logger.warning(
            f"Qwen retokenization failed on {len(text)} chars: {exc}"
        )
        return []


def round_trip_drift(original_kimi_text: str, retokenized_qwen_ids: list[int],
                     qwen_tokenizer) -> float:
    """Quantify the loss from re-tokenizing through Qwen.

    Computes the character-level Levenshtein-edit-distance ratio
    between the original Kimi-decoded text and what Qwen produces
    when it re-decodes the retokenized IDs. 0.0 = byte-identical
    (no drift); 1.0 = completely different.

    Used as a telemetry signal in the Stage-2 experiment to verify
    Path A is acceptable before promoting the teacher swap.
    """
    if not original_kimi_text or not retokenized_qwen_ids or qwen_tokenizer is None:
        return 1.0
    try:
        recovered = qwen_tokenizer.decode(
            retokenized_qwen_ids, skip_special_tokens=False,
        )
    except Exception:
        return 1.0
    return _normalised_edit_distance(original_kimi_text, recovered)


def _normalised_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance / max(len(a), len(b)). Returns 0.0 on
    byte-identical, 1.0 on completely-disjoint."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Standard DP for Levenshtein. O(|a| × |b|) — acceptable for
    # ~1000-token outputs (~5000 chars), but we cap for safety.
    if len(a) > 10000 or len(b) > 10000:
        # Long-output fallback: use a sample ratio.
        return 1.0 - (sum(1 for x, y in zip(a, b) if x == y) / max(len(a), len(b)))
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, cur = cur, prev
    return prev[n] / max(m, n)


def align_logprobs_kimi_to_qwen(
    kimi_top_k_indices: list[list[int]],
    kimi_top_k_logprobs: list[list[float]],
    kimi_tokenizer,
    qwen_tokenizer,
    k: int = 128,
) -> dict[str, Any]:
    """Map Kimi's per-position top-K logprobs onto the closest Qwen-vocab tokens.

    Strategy:
      - For each position, decode Kimi's top-K tokens individually to get
        their text (e.g., " the", " quick", "ization", ...).
      - Re-encode each piece with the Qwen tokenizer. Most pieces are
        byte-identical text and the Qwen ID is unique; for some pieces
        the segmentation differs and we get a multi-token sequence —
        in that case the FIRST Qwen token is used as the alignment.
      - Combine logprobs: when two Kimi pieces map to the same Qwen
        token, sum their probabilities (renormalise top-K afterward
        if needed).

    Returns a dict matching the existing teacher-cache schema:
      ``{"indices": [[qwen_id]*K]*seq_len, "values": [[logprob]*K]*seq_len}``
    that the student-side eval (compute_kl_from_sparse,
    compute_kl_is_from_sparse, compute_eopd_metrics_from_sparse) can
    consume unchanged.

    Engineering status: SCAFFOLDING. The full implementation requires a
    Kimi tokenizer instance which only loads on a multi-GPU pod with
    K2.6 weights. The shape this returns is contractually compatible
    with the existing pipeline; the alignment quality will need
    empirical validation per the Stage-2 runbook (Tier 3 criteria).
    """
    if (
        not kimi_top_k_indices or not kimi_top_k_logprobs
        or kimi_tokenizer is None or qwen_tokenizer is None
    ):
        return {"indices": [], "values": []}
    out_indices: list[list[int]] = []
    out_values: list[list[float]] = []
    for pos_indices, pos_logprobs in zip(kimi_top_k_indices, kimi_top_k_logprobs):
        if len(pos_indices) != len(pos_logprobs):
            continue
        # Map each Kimi top-K id → Qwen first-id.
        qwen_id_to_logprob: dict[int, float] = {}
        for kimi_id, kimi_lp in zip(pos_indices, pos_logprobs):
            try:
                piece_text = kimi_tokenizer.decode([int(kimi_id)],
                                                   skip_special_tokens=False)
            except Exception:
                continue
            if not piece_text:
                continue
            try:
                qwen_ids = qwen_tokenizer(
                    piece_text, add_special_tokens=False,
                )["input_ids"]
            except Exception:
                continue
            if not qwen_ids:
                continue
            q_id = int(qwen_ids[0])
            # If two Kimi tokens map to the same Qwen token, sum probs.
            existing = qwen_id_to_logprob.get(q_id)
            if existing is None:
                qwen_id_to_logprob[q_id] = float(kimi_lp)
            else:
                # Sum probabilities: log(exp(a) + exp(b)) = log_add_exp.
                import math as _m
                qwen_id_to_logprob[q_id] = _m.log(
                    _m.exp(existing) + _m.exp(float(kimi_lp))
                )
        # Take top-K by logprob.
        sorted_pairs = sorted(
            qwen_id_to_logprob.items(), key=lambda kv: kv[1], reverse=True,
        )[:k]
        ids = [p[0] for p in sorted_pairs]
        lps = [p[1] for p in sorted_pairs]
        out_indices.append(ids)
        out_values.append(lps)
    return {"indices": out_indices, "values": out_values}


def stage2_drift_summary(
    drift_per_prompt: list[float],
) -> dict[str, float]:
    """Aggregate per-prompt round-trip drift into a round-level summary.

    Used by the Stage-2 experiment runner to surface the drift signal
    on the dashboard. Tier 1 disqualifying gate: mean drift > 0.10
    means Path A is too lossy to use; bail to Path B before promoting
    the teacher swap.
    """
    if not drift_per_prompt:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    sorted_d = sorted(drift_per_prompt)
    n = len(sorted_d)
    return {
        "n": n,
        "mean": round(sum(sorted_d) / n, 4),
        "p50": round(sorted_d[n // 2], 4),
        "p95": round(sorted_d[min(n - 1, int(n * 0.95))], 4),
        "max": round(sorted_d[-1], 4),
    }
