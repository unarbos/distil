#!/usr/bin/env python3
"""
vLLM-accelerated GPU evaluation script for SN97 validation (v3.0.0).

Architecture & VRAM timeline (single B200 = 192GB):
  Phase 1 — Teacher generation via vLLM:
    [vLLM teacher ~70GB] → generate continuations → kill server
    Time: ~3-5 min (vs 25 min with HF)

  Phase 2 — Teacher logit extraction via HF:
    [HF teacher ~67GB] → forward passes (no autoregressive) → cache logits → unload
    Time: ~8-10 min (forward-only, ~3x faster than generate)

  Phase 3 — Student scoring:
    [teacher logits on CPU ~2GB] + [king ~8GB stays loaded] + [challenger ~8GB rotates]
    Total VRAM: ~18GB (king + challenger + overhead)
    Time: ~2-3 min per student

Optimizations:
  1. vLLM teacher generation: 5-10x faster than HF generate()
  2. King stays in VRAM: no download/load/cleanup between rounds (~3-5 min saved)
  3. Prefetch next student: download while current student scores
  4. Teacher unloaded after logits cached: frees ~67GB for student scoring
  5. Graceful fallback: if vLLM fails, falls back to pure HF path

Usage:
    python3 pod_eval_vllm.py \\
        --teacher Qwen/Qwen3.5-35B-A3B \\
        --students user/king,user/challenger1,user/challenger2 \\
        --prompts prompts.json \\
        --output results.json \\
        --king user/king
"""
import math
import torch

# Chunk size for HF forward pass with KV cache on long sequences.
# Sequences longer than this are processed in chunks to avoid truncation gaming.
HF_CHUNK_SIZE = 4096

# Minimum completion tokens for teacher generation.
# Prompts producing fewer tokens are skipped to prevent near-empty
# completions from polluting KL scores.
MIN_COMPLETION_TOKENS = 10
import torch.nn.functional as F
import json
import time
import argparse
import gc
import hashlib
import os
import sys
import shutil
import subprocess
import signal
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str():
    """Return a human-readable string of current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"

def free_gpu():
    """Free GPU memory: garbage collect, empty cache, synchronize."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def ensure_disk_space(teacher_name, threshold=85):
    """Check disk usage, clean non-teacher caches and stale files if above threshold.

    Called at every phase boundary (before vLLM start, before HF load, before each student)
    to prevent disk exhaustion — the #1 crash cause historically.

    Returns current disk usage percentage.
    """
    try:
        st = os.statvfs("/")
        pct = int(100 * (1 - st.f_bavail / st.f_blocks))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        print(f"  [disk] {pct}% used, {free_gb:.1f}GB free", flush=True)

        if pct > threshold:
            print(f"  [disk] >{threshold}% — cleaning non-teacher caches", flush=True)
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for d in cache_dir.iterdir():
                    if d.is_dir() and d.name.startswith("models--") and d.name != teacher_cache:
                        shutil.rmtree(d)
            # Clean stale /tmp files >1GB
            import glob
            for pattern in ["/tmp/vllm_*", "/tmp/tmp*", "/tmp/teacher_*"]:
                for f in glob.glob(pattern):
                    try:
                        fsize = os.path.getsize(f)
                        if fsize > 1024**3:
                            os.remove(f)
                            print(f"  [disk] Removed stale {f} ({fsize/1024**3:.1f}GB)", flush=True)
                    except Exception:
                        pass
            # Clean stale teacher cache if hash doesn't match
            teacher_logits_path = "/home/teacher_cache.pt"
            if os.path.exists(teacher_logits_path):
                cache_size = os.path.getsize(teacher_logits_path) / (1024**3)
                if cache_size > 0 and pct > 90:
                    os.remove(teacher_logits_path)
                    print(f"  [disk] Removed stale teacher cache ({cache_size:.1f}GB)", flush=True)

            st2 = os.statvfs("/")
            pct2 = int(100 * (1 - st2.f_bavail / st2.f_blocks))
            free_gb2 = (st2.f_bavail * st2.f_frsize) / (1024**3)
            print(f"  [disk] After cleanup: {pct2}% used, {free_gb2:.1f}GB free", flush=True)
            return pct2
        return pct
    except Exception as e:
        print(f"  [disk] Check failed: {e}", flush=True)
        return 0


# ── KL computation ──
# Uses F.kl_div(log_target=True) which is mathematically identical to
# sum(P * (log P - log Q)) but lets PyTorch fuse the kernel internally.
# Chunked over positions to reduce peak memory (~4x less for 512-pos sequences).
# Credit: caseus (github.com/winglian) for the optimization.
KL_CHUNK_SIZE = 128


def _kl_chunk_fn(t_log_p_chunk, s_log_p_chunk):
    """KL per position for a chunk. Uses F.kl_div for better kernel fusion."""
    return F.kl_div(s_log_p_chunk, t_log_p_chunk, log_target=True, reduction="none").sum(dim=-1)


# Try to compile the inner kernel for ~2x speedup
try:
    _kl_chunk_compiled = torch.compile(_kl_chunk_fn, fullgraph=True)
    _KL_USE_COMPILED = True
except Exception:
    _kl_chunk_compiled = _kl_chunk_fn
    _KL_USE_COMPILED = False


# ── Activation fingerprinting for functional copy detection ──
ACTIVATION_FP_SEED = 42
ACTIVATION_FP_N_INPUTS = 5
ACTIVATION_FP_SEQ_LEN = 64
ACTIVATION_FP_VOCAB_SIZE = 248320  # Qwen tokenizer vocab


def compute_activation_fingerprint(model, device="cuda"):
    """
    Compute an activation-space fingerprint for functional copy detection.

    Runs a fixed set of random token sequences through the model and collects
    hidden states at evenly-spaced layers. The fingerprint is a list of mean
    activation vectors (one per layer checkpoint), which is invariant to
    intra-layer weight reparameterization (e.g. scaling V by c, O by 1/c).

    Returns dict with:
      - "layer_fingerprints": {layer_idx: [float, ...]}  (mean hidden state per layer)
      - "n_layers": int
      - "hidden_size": int
    """
    try:
        # Generate deterministic random inputs
        rng = torch.Generator(device=device)
        rng.manual_seed(ACTIVATION_FP_SEED)
        input_ids = torch.randint(
            0, ACTIVATION_FP_VOCAB_SIZE,
            (ACTIVATION_FP_N_INPUTS, ACTIVATION_FP_SEQ_LEN),
            generator=rng, device=device
        )

        # Determine layer checkpoints (4 evenly spaced)
        n_layers = model.config.num_hidden_layers
        checkpoints = sorted(set([
            0,
            n_layers // 3,
            2 * n_layers // 3,
            n_layers - 1,
        ]))

        # Collect activations via hooks
        activations = {idx: [] for idx in checkpoints}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Mean over batch and sequence dimensions → single vector
                activations[layer_idx].append(h.float().mean(dim=(0, 1)).detach().cpu())
            return hook_fn

        # Find the layers module
        layers_module = None
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers_module = obj
                break
            except AttributeError:
                continue

        if layers_module is None:
            print("[fingerprint] Could not find layers module", flush=True)
            return None

        for idx in checkpoints:
            if idx < len(layers_module):
                hooks.append(layers_module[idx].register_forward_hook(make_hook(idx)))

        # Forward pass (no grad)
        with torch.no_grad():
            for i in range(ACTIVATION_FP_N_INPUTS):
                _ = model(input_ids[i:i+1])

        # Remove hooks
        for h in hooks:
            h.remove()

        # Average across all inputs per layer
        layer_fingerprints = {}
        for idx in checkpoints:
            if activations[idx]:
                avg = torch.stack(activations[idx]).mean(dim=0)
                # Truncate to first 128 dims to keep payload small
                fp = avg[:128].tolist()
                layer_fingerprints[str(idx)] = [round(v, 6) for v in fp]

        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0

        # Cleanup
        del input_ids, activations
        torch.cuda.empty_cache()

        print(f"[fingerprint] Computed: {len(layer_fingerprints)} layers, "
              f"n_layers={n_layers}, hidden_size={hidden_size}", flush=True)
        return {
            "layer_fingerprints": layer_fingerprints,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
        }
    except Exception as e:
        print(f"[fingerprint] Error: {e}", flush=True)
        return None


def compute_kl(teacher_logits, student_logits):
    """KL(teacher || student) per position. For one-off use."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    n_pos = t_log_p.shape[-2] if t_log_p.dim() >= 2 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    kl_per_pos = torch.empty(t_log_p.shape[:-1], device=t_log_p.device)
    for i in range(0, n_pos, KL_CHUNK_SIZE):
        j = min(i + KL_CHUNK_SIZE, n_pos)
        if t_log_p.dim() >= 3:
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
        else:
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def compute_kl_from_precomputed(t_log_p, t_p, student_logits):
    """KL using precomputed teacher log_softmax + probs. Saves ~50% compute.

    Uses F.kl_div(log_target=True) + chunking for better memory and speed.
    The t_p argument is kept for API compat but not used (F.kl_div computes
    exp(t_log_p) internally when log_target=True).
    """
    s_logits = student_logits.float()
    # Handle vocab size mismatch (student vs teacher)
    t_vocab = t_log_p.shape[-1]
    s_vocab = s_logits.shape[-1]
    if s_vocab < t_vocab:
        pad = torch.full((*s_logits.shape[:-1], t_vocab - s_vocab), -1e10,
                         device=s_logits.device, dtype=s_logits.dtype)
        s_logits = torch.cat([s_logits, pad], dim=-1)
    elif s_vocab > t_vocab:
        s_logits = s_logits[..., :t_vocab]
    s_log_p = F.log_softmax(s_logits, dim=-1)
    # Chunked KL over positions
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
    else:
        kl_per_pos = torch.empty(n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def _build_token_to_id_map(tokenizer):
    """Build mapping from token text to token ID for vLLM logprobs decoding."""
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    text_to_id = {}
    for tok_str, tok_id in vocab.items():
        # Store raw vocab entry
        text_to_id[tok_str] = tok_id
        # Store decoded form as fallback
        decoded = tokenizer.decode([tok_id])
        if decoded not in text_to_id:
            text_to_id[decoded] = tok_id
    return text_to_id


def vllm_logprobs_to_sparse(top_logprobs_list, token_to_id, tokenizer, k=128):
    """Convert vLLM top_logprobs response to sparse tensor format.

    Args:
        top_logprobs_list: list of dicts (one per generated token),
            each mapping token_str -> logprob for top-k tokens.
        token_to_id: pre-built mapping from token text to token ID.
        tokenizer: tokenizer for fallback encoding.
        k: number of top logprobs to keep per position.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are logprobs (from vLLM log-softmax).
    """
    seq_len = len(top_logprobs_list)
    indices = torch.zeros(1, seq_len, k, dtype=torch.long)
    values = torch.full((1, seq_len, k), -100.0, dtype=torch.float32)

    for pos, top_lp in enumerate(top_logprobs_list):
        sorted_items = sorted(top_lp.items(), key=lambda x: x[1], reverse=True)[:k]
        for j, (token_str, logprob) in enumerate(sorted_items):
            token_id = token_to_id.get(token_str)
            if token_id is None:
                try:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    token_id = encoded[0] if encoded else 0
                except Exception:
                    token_id = 0
            indices[0, pos, j] = token_id
            values[0, pos, j] = logprob

    return {"indices": indices, "values": values}


def dense_to_sparse_topk(logits, k=128):
    """Convert dense logits tensor to sparse top-k format.

    Args:
        logits: [1, seq_len, vocab_size] or [seq_len, vocab_size] tensor.
        k: number of top logits to keep.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are raw logits (not log-softmax).
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    topk_values, topk_indices = logits.float().topk(k, dim=-1)
    return {"indices": topk_indices.cpu(), "values": topk_values.cpu()}


def _is_sparse_logits(entry):
    """Check whether a teacher_logits_list entry is sparse (dict) or dense (tensor)."""
    return isinstance(entry, dict) and "indices" in entry and "values" in entry


def compute_kl_from_sparse(teacher_indices, teacher_values, student_logits,
                           values_are_logprobs=False):
    """KL divergence using sparse top-k teacher logits/logprobs.

    For teacher: renormalize the top-k values into a proper distribution.
    For student: compute full-vocab log_softmax, gather the same k positions,
    then renormalize over those k positions.
    KL = sum_k P_teacher_k * (log P_teacher_k - log Q_student_k)

    Args:
        teacher_indices: [1, seq_len, k] token IDs.
        teacher_values:  [1, seq_len, k] logits or logprobs.
        student_logits:  [1, seq_len, vocab_size] raw student logits.
        values_are_logprobs: if True, teacher_values are already log-probs.

    Returns:
        KL per position, shape [1, seq_len] or [seq_len].
    """
    device = student_logits.device
    t_idx = teacher_indices.to(device)
    t_vals = teacher_values.to(device).float()

    # Teacher: renormalized log-probs over top-k tokens
    if values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)

    # Student: full-vocab log_softmax, then gather the k positions
    s_log_p_full = F.log_softmax(student_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_idx)
    # Renormalize student over the same k tokens for proper KL
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)

    del s_log_p_full

    # Chunked KL over positions for memory efficiency
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = F.kl_div(
                s_log_p_k_norm[:, i:j, :], t_log_p[:, i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)
    else:
        kl_per_pos = torch.empty(n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = F.kl_div(
                s_log_p_k_norm[i:j, :], t_log_p[i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)

    return kl_per_pos


def compute_kl_sparse_vs_sparse(teacher_indices, teacher_values, student_indices, student_values,
                                teacher_values_are_logprobs=False):
    """KL divergence between two sparse top-k representations.

    Used when student logprobs come from vLLM prompt_logprobs (sparse top-k)
    instead of full HF logits (dense).

    For each position, renormalizes both teacher and student over the teacher's
    top-k support. For teacher tokens not found in the student's top-k, uses a
    floor logprob value.

    Args:
        teacher_indices: [1, seq_len, k_t] token IDs.
        teacher_values:  [1, seq_len, k_t] logits or logprobs.
        student_indices: [1, seq_len, k_s] token IDs.
        student_values:  [1, seq_len, k_s] logprobs (from vLLM).
        teacher_values_are_logprobs: if True, teacher_values are already log-probs.

    Returns:
        KL per position, shape [1, seq_len].
    """
    FLOOR_LOGPROB = -23.0  # ~log(1e-10)
    KL_POS_CHUNK = 256  # chunk over positions to limit memory

    device = teacher_indices.device
    B, seq_len, k_t = teacher_indices.shape
    _, _, k_s = student_indices.shape

    t_idx = teacher_indices
    t_vals = teacher_values.float()
    s_idx = student_indices.to(device)
    s_vals = student_values.to(device).float()

    # Teacher: renormalize over top-k
    if teacher_values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)
    t_p = t_log_p.exp()

    kl_per_pos = torch.empty(B, seq_len, device=device)

    for ci in range(0, seq_len, KL_POS_CHUNK):
        cj = min(ci + KL_POS_CHUNK, seq_len)

        t_chunk_idx = t_idx[:, ci:cj, :]     # [B, C, k_t]
        s_chunk_idx = s_idx[:, ci:cj, :]     # [B, C, k_s]
        s_chunk_vals = s_vals[:, ci:cj, :]   # [B, C, k_s]

        # For each teacher token, find matching student token
        # [B, C, k_t, 1] vs [B, C, 1, k_s] → [B, C, k_t, k_s]
        match = (t_chunk_idx.unsqueeze(-1) == s_chunk_idx.unsqueeze(-2))
        has_match = match.any(dim=-1)  # [B, C, k_t]

        # Gather student logprobs at matching positions
        s_expanded = s_chunk_vals.unsqueeze(-2).expand(-1, -1, k_t, -1)  # [B, C, k_t, k_s]
        s_at_match = torch.where(match, s_expanded, torch.tensor(float('-inf'), device=device))
        s_at_teacher, _ = s_at_match.max(dim=-1)  # [B, C, k_t]
        s_at_teacher = torch.where(has_match, s_at_teacher,
                                    torch.tensor(FLOOR_LOGPROB, device=device))

        # Renormalize student over teacher's support
        s_at_teacher_norm = s_at_teacher - s_at_teacher.logsumexp(dim=-1, keepdim=True)

        # KL for this chunk
        t_log_p_chunk = t_log_p[:, ci:cj, :]
        t_p_chunk = t_p[:, ci:cj, :]
        kl_per_pos[:, ci:cj] = (t_p_chunk * (t_log_p_chunk - s_at_teacher_norm)).sum(dim=-1)

        del match, has_match, s_expanded, s_at_match, s_at_teacher, s_at_teacher_norm

    return kl_per_pos


def hf_batched_forward(teacher, sequences_data, device, batch_size=2, logprobs_k=128,
                       progress_cb=None):
    """Batched HF forward pass for teacher logit extraction.

    Groups prompts by similar sequence length, pads within each batch,
    runs the forward pass, then unpads and extracts continuation logits.

    For sequences longer than HF_CHUNK_SIZE, falls back to per-sequence
    chunked forward pass (batching long-context sequences is wasteful due
    to extreme padding).

    Args:
        teacher: HF model (already on device, eval mode).
        sequences_data: list of dicts with 'full_ids' [1, seq_len], 'prompt_len'.
        device: torch device.
        batch_size: number of sequences per batch.
        logprobs_k: if >0, store only top-k logits (sparse). 0 = full vocab.
        progress_cb: optional callback(i) called after each prompt is processed.

    Returns:
        (teacher_logits_list, prompt_lens, full_sequences, n_chunked)
        teacher_logits_list: list of sparse dicts or dense tensors.
    """
    # Separate long sequences (chunked path) from short ones (batched path)
    short_items = []  # (original_idx, data)
    long_items = []   # (original_idx, data)
    for idx, data in enumerate(sequences_data):
        seq_len = data["full_ids"].shape[1]
        if seq_len > HF_CHUNK_SIZE:
            long_items.append((idx, data))
        else:
            short_items.append((idx, data))

    # Sort short items by sequence length for efficient batching
    short_items.sort(key=lambda x: x[1]["full_ids"].shape[1])

    # Pre-allocate result arrays
    n_total = len(sequences_data)
    teacher_logits_list = [None] * n_total
    prompt_lens = [0] * n_total
    full_sequences = [None] * n_total
    n_chunked = 0
    processed = 0

    with torch.no_grad():
        # ── Batched path for short sequences ──
        for batch_start in range(0, len(short_items), batch_size):
            batch = short_items[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                # Single item — no padding needed
                orig_idx, data = batch[0]
                full_ids = data["full_ids"].to(device)
                prompt_len = data["prompt_len"]
                prompt_lens[orig_idx] = prompt_len
                full_sequences[orig_idx] = full_ids

                logits = teacher(full_ids).logits.float()
                cont_logits = logits[:, prompt_len - 1:-1, :]
                if logprobs_k > 0:
                    teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                else:
                    teacher_logits_list[orig_idx] = cont_logits.cpu()
                del logits, cont_logits
                processed += 1
                if progress_cb:
                    progress_cb(processed)
            else:
                # Pad batch to max length
                max_len = max(d["full_ids"].shape[1] for _, d in batch)
                batch_ids = []
                attention_masks = []
                for _, data in batch:
                    ids = data["full_ids"]  # [1, seq_len]
                    seq_len = ids.shape[1]
                    if seq_len < max_len:
                        # Left-pad (standard for causal LMs)
                        pad_len = max_len - seq_len
                        pad = torch.zeros(1, pad_len, dtype=ids.dtype, device=ids.device)
                        ids_padded = torch.cat([pad, ids], dim=1)
                        mask = torch.cat([
                            torch.zeros(1, pad_len, dtype=torch.long, device=ids.device),
                            torch.ones(1, seq_len, dtype=torch.long, device=ids.device),
                        ], dim=1)
                    else:
                        ids_padded = ids
                        mask = torch.ones(1, seq_len, dtype=torch.long, device=ids.device)
                    batch_ids.append(ids_padded)
                    attention_masks.append(mask)

                batch_tensor = torch.cat(batch_ids, dim=0).to(device)  # [B, max_len]
                mask_tensor = torch.cat(attention_masks, dim=0).to(device)  # [B, max_len]

                outputs = teacher(batch_tensor, attention_mask=mask_tensor)
                batch_logits = outputs.logits.float()  # [B, max_len, vocab]
                del outputs

                # Unpad and extract continuation logits per item
                for b_idx, (orig_idx, data) in enumerate(batch):
                    full_ids = data["full_ids"].to(device)
                    prompt_len = data["prompt_len"]
                    seq_len = full_ids.shape[1]
                    pad_len = max_len - seq_len
                    prompt_lens[orig_idx] = prompt_len
                    full_sequences[orig_idx] = full_ids

                    # Extract this item's logits (remove padding offset)
                    item_logits = batch_logits[b_idx:b_idx+1, pad_len:, :]  # [1, seq_len, vocab]
                    cont_logits = item_logits[:, prompt_len - 1:-1, :]
                    if logprobs_k > 0:
                        teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                    else:
                        teacher_logits_list[orig_idx] = cont_logits.cpu()
                    del item_logits, cont_logits
                    processed += 1
                    if progress_cb:
                        progress_cb(processed)

                del batch_tensor, mask_tensor, batch_logits
                torch.cuda.empty_cache()

        # ── Chunked path for long sequences ──
        for orig_idx, data in long_items:
            n_chunked += 1
            full_ids = data["full_ids"].to(device)
            prompt_len = data["prompt_len"]
            seq_len = full_ids.shape[1]
            prompt_lens[orig_idx] = prompt_len
            full_sequences[orig_idx] = full_ids

            all_logit_chunks = []
            past_key_values = None
            for chunk_start in range(0, seq_len, HF_CHUNK_SIZE):
                chunk_end = min(chunk_start + HF_CHUNK_SIZE, seq_len)
                chunk_ids = full_ids[:, chunk_start:chunk_end]
                outputs = teacher(
                    chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                all_logit_chunks.append(outputs.logits.float().cpu())
                past_key_values = outputs.past_key_values
                del outputs
            all_logits = torch.cat(all_logit_chunks, dim=1)
            cont_logits = all_logits[:, prompt_len - 1:-1, :]
            if logprobs_k > 0:
                teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
            else:
                teacher_logits_list[orig_idx] = cont_logits.cpu()
            del all_logits, all_logit_chunks, past_key_values, cont_logits
            torch.cuda.empty_cache()
            processed += 1
            if progress_cb:
                progress_cb(processed)

    return teacher_logits_list, prompt_lens, full_sequences, n_chunked


def load_model(name, device="cuda", dtype=torch.bfloat16, revision=None):
    """Load a HuggingFace model for inference.

    Uses flash_attention_2 when available, falls back to default attention.
    Teacher models get trust_remote_code=True; students don't (security).
    When revision is set, pins to that exact HF commit hash.
    """
    from transformers import AutoModelForCausalLM
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    if revision and revision != "main":
        kwargs["revision"] = revision
        print(f"  [model] Pinning to revision {revision[:12]}", flush=True)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            try:
                m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
                print(f"  [model] Loaded with flash_attention_2", flush=True)
                return m
            except (ValueError, ImportError):
                m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                print(f"  [model] Loaded with default attention", flush=True)
                return m
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout", "HTTPSConnection"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [model] Transient error loading {name} (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                raise


def prefetch_model(name, revision=None, max_retries=3):
    """Download model files to HF cache without loading to GPU. Runs in background.
    
    Retries on transient HF errors (429, 503, connection errors).
    """
    from huggingface_hub import snapshot_download
    dl_kwargs = dict(ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"])
    if revision and revision != "main":
        dl_kwargs["revision"] = revision
    for attempt in range(max_retries):
        try:
            snapshot_download(name, **dl_kwargs)
            print(f"  [prefetch] {name} cached (rev={revision or 'main'})", flush=True)
            return
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [prefetch] {name} transient error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                print(f"  [prefetch] {name} failed: {e}", flush=True)
                return


def clean_model_cache(name, teacher_name=None):
    """Remove HF cache for a specific model, preserving teacher cache."""
    try:
        cache_name = f"models--{name.replace('/', '--')}"
        if teacher_name:
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            if cache_name == teacher_cache:
                return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  [cleanup] Removed {cache_name}", flush=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Server Management
# ═══════════════════════════════════════════════════════════════════════════════

VLLM_PORT = 9100
VLLM_URL = f"http://localhost:{VLLM_PORT}"
VLLM_STARTUP_TIMEOUT = 900  # 15 min


def is_vllm_running():
    """Check if vLLM server is running and healthy."""
    import requests
    try:
        r = requests.get(f"{VLLM_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=16384, revision=None):
    """Start vLLM server via subprocess. Returns True on success."""
    ensure_disk_space(model_name, threshold=80)

    print(f"\n[vllm] Starting server for {model_name}...", flush=True)
    stop_vllm_server()

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(VLLM_PORT),
        "--served-model-name", "teacher",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-prefix-caching",
        "--no-enable-log-requests",
        "--reasoning-parser", "qwen3",
        "--max-logprobs", "128",
    ]
    if revision and revision != "main":
        cmd.extend(["--revision", revision])

    log_f = open("/tmp/vllm_teacher.log", "w")
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    Path("/tmp/vllm_teacher.pid").write_text(str(proc.pid))
    print(f"[vllm] PID: {proc.pid}", flush=True)

    import requests
    start_time = time.time()
    while time.time() - start_time < VLLM_STARTUP_TIMEOUT:
        elapsed = int(time.time() - start_time)
        try:
            if requests.get(f"{VLLM_URL}/health", timeout=3).status_code == 200:
                print(f"[vllm] Ready in {elapsed}s", flush=True)
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"[vllm] Died with code {proc.returncode}", flush=True)
            try:
                print(Path("/tmp/vllm_teacher.log").read_text()[-1500:], flush=True)
            except Exception:
                pass
            return False
        if elapsed % 60 == 0 and elapsed > 0:
            print(f"[vllm] Still starting... ({elapsed}s)", flush=True)
        time.sleep(5)

    print(f"[vllm] Timeout after {VLLM_STARTUP_TIMEOUT}s", flush=True)
    stop_vllm_server()
    return False


def stop_vllm_server():
    """Kill vLLM server and free VRAM."""
    pid_file = Path("/tmp/vllm_teacher.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except ProcessLookupError:
                    break
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)
    try:
        subprocess.run(["fuser", "-k", f"{VLLM_PORT}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass
    free_gpu()
    time.sleep(2)


def _generate_single_prompt(idx, prompt_text, max_new_tokens, block_seed,
                            logprobs_k, tokenizer, token_to_id):
    """Generate a single prompt via vLLM. Used by both sequential and concurrent paths."""
    import requests

    payload = {
        "model": "teacher",
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if block_seed is not None else 0.0,
        "top_p": 0.9 if block_seed is not None else 1.0,
    }
    if block_seed is not None:
        payload["seed"] = block_seed + idx
    if logprobs_k > 0:
        payload["logprobs"] = logprobs_k
        payload["prompt_logprobs"] = 0

    for attempt in range(3):
        try:
            resp = requests.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            cont_text = choice["text"]
            full_text = prompt_text + cont_text
            full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids
            result = {
                "full_ids": full_ids,
                "prompt_len": prompt_ids.shape[1],
                "gen_len": full_ids.shape[1] - prompt_ids.shape[1],
            }
            if logprobs_k > 0 and "logprobs" in choice and choice["logprobs"]:
                lp_data = choice["logprobs"]
                top_lp_list = lp_data.get("top_logprobs")
                if top_lp_list and token_to_id is not None:
                    result["sparse_logprobs"] = vllm_logprobs_to_sparse(
                        top_lp_list, token_to_id, tokenizer, k=logprobs_k
                    )
            return idx, result
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise RuntimeError(f"vLLM generation failed for prompt {idx}: {e}")


def generate_via_vllm(prompts, tokenizer, max_new_tokens, block_seed=None,
                      logprobs_k=128, token_to_id=None, progress_cb=None,
                      concurrency=4):
    """Generate teacher continuations via vLLM API.

    When logprobs_k > 0, requests top-k logprobs from vLLM per generated token
    and converts them to sparse tensor format, eliminating the need for a
    separate HF forward pass (Phase 1b).

    Uses concurrent requests to vLLM for significantly faster teacher generation.
    vLLM's continuous batching handles the internal scheduling — we just need to
    send multiple requests in parallel.

    Args:
        concurrency: Number of parallel requests to vLLM (default 4).
            vLLM batches these internally. Higher values use more VRAM for KV cache
            but significantly speed up total generation time.

    Returns list of dicts with full_ids, prompt_len, gen_len, and optionally
    'sparse_logprobs' (dict with 'indices' and 'values' tensors).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if concurrency <= 1:
        # Sequential fallback
        results = []
        for idx, prompt_text in enumerate(prompts):
            _, result = _generate_single_prompt(
                idx, prompt_text, max_new_tokens, block_seed,
                logprobs_k, tokenizer, token_to_id
            )
            results.append(result)
            if idx % 10 == 0 or idx == len(prompts) - 1:
                has_lp = "sparse_logprobs" in result
                print(f"  [{idx+1}/{len(prompts)}] {result['prompt_len']}+{result['gen_len']} tokens"
                      f"{' (logprobs✓)' if has_lp else ''}", flush=True)
            if progress_cb:
                progress_cb(idx + 1, len(prompts))
        return results

    # Concurrent generation
    print(f"  [vllm] Concurrent generation: {concurrency} parallel requests", flush=True)
    result_slots = [None] * len(prompts)
    completed = 0
    last_log = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for idx, prompt_text in enumerate(prompts):
            fut = executor.submit(
                _generate_single_prompt, idx, prompt_text, max_new_tokens,
                block_seed, logprobs_k, tokenizer, token_to_id
            )
            futures[fut] = idx

        failed = []
        for fut in as_completed(futures):
            orig_idx = futures[fut]
            try:
                _, result = fut.result()
                result_slots[orig_idx] = result
                completed += 1

                # Log every 10 completions or at the end
                if completed - last_log >= 10 or completed == len(prompts):
                    has_lp = "sparse_logprobs" in result
                    print(f"  [{completed}/{len(prompts)}] latest: {result['prompt_len']}+{result['gen_len']} tokens"
                          f"{' (logprobs✓)' if has_lp else ''}", flush=True)
                    last_log = completed

                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e:
                failed.append((orig_idx, str(e)))
                print(f"  [vllm] Prompt {orig_idx} failed: {e}", flush=True)

    # Retry failed prompts sequentially
    if failed:
        print(f"  [vllm] Retrying {len(failed)} failed prompts sequentially...", flush=True)
        for idx, err in failed:
            try:
                _, result = _generate_single_prompt(
                    idx, prompts[idx], max_new_tokens, block_seed,
                    logprobs_k, tokenizer, token_to_id
                )
                result_slots[idx] = result
                completed += 1
                print(f"  [vllm] Retry prompt {idx}: OK", flush=True)
                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e2:
                raise RuntimeError(f"vLLM generation failed for prompt {idx} after retry: {e2}")

    return result_slots


def _parse_vllm_prompt_logprobs(prompt_logprobs_raw, token_to_id, tokenizer, k=128):
    """Parse vLLM prompt_logprobs response into sparse format.

    Handles multiple vLLM response formats:
    1. List of dicts where keys are token strings, values are logprob floats
    2. List of dicts where keys are token IDs (as strings), values are dicts
       with 'logprob', 'rank', 'decoded_token'
    3. List of lists in OpenAI top_logprobs format [{token, logprob, bytes}, ...]

    Args:
        prompt_logprobs_raw: list of (None | dict | list) from vLLM response.
        token_to_id: pre-built mapping from token text to token ID.
        tokenizer: tokenizer for fallback encoding.
        k: number of top logprobs to keep per position.

    Returns:
        Sparse dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are logprobs.
    """
    seq_len = len(prompt_logprobs_raw)
    indices = torch.zeros(1, seq_len, k, dtype=torch.long)
    values = torch.full((1, seq_len, k), -100.0, dtype=torch.float32)

    for pos, entry in enumerate(prompt_logprobs_raw):
        if entry is None:
            continue

        # Parse entry into list of (token_id, logprob) tuples
        items = []

        if isinstance(entry, list):
            # Format 3: OpenAI-style list of {token, logprob, bytes}
            for item in entry:
                if isinstance(item, dict):
                    tok_str = item.get("token", "")
                    lp = item.get("logprob", -100.0)
                    token_id = token_to_id.get(tok_str)
                    if token_id is None:
                        try:
                            encoded = tokenizer.encode(tok_str, add_special_tokens=False)
                            token_id = encoded[0] if encoded else 0
                        except Exception:
                            token_id = 0
                    items.append((token_id, lp))
        elif isinstance(entry, dict):
            for key, val in entry.items():
                if isinstance(val, (int, float)):
                    # Format 1: {token_str: logprob_float}
                    token_id = token_to_id.get(key)
                    if token_id is None:
                        try:
                            token_id = int(key)
                        except (ValueError, TypeError):
                            try:
                                encoded = tokenizer.encode(key, add_special_tokens=False)
                                token_id = encoded[0] if encoded else 0
                            except Exception:
                                token_id = 0
                    items.append((token_id, float(val)))
                elif isinstance(val, dict):
                    # Format 2: {token_id_str: {logprob, rank, decoded_token}}
                    lp = val.get("logprob", -100.0)
                    try:
                        token_id = int(key)
                    except (ValueError, TypeError):
                        token_id = token_to_id.get(key, 0)
                    items.append((token_id, lp))

        # Sort by logprob descending, take top-k
        items.sort(key=lambda x: x[1], reverse=True)
        for j, (tid, lp) in enumerate(items[:k]):
            indices[0, pos, j] = tid
            values[0, pos, j] = lp

    return {"indices": indices, "values": values}


def score_student_via_vllm(student_name, student_rev, full_sequences, prompt_lens,
                            teacher_logits_list, tokenizer, token_to_id,
                            gpu_memory_utilization=0.15, max_model_len=16384,
                            batch_size=20, early_stop_fn=None, progress_cb=None,
                            timeout=600):
    """Score a student model using vLLM prompt_logprobs mode.

    Starts a vLLM server for the student, sends full sequences as token IDs
    with max_tokens=0 and prompt_logprobs=128 to get the student's log-probability
    distribution at each position. Computes KL divergence against teacher's
    sparse logprobs.

    Args:
        student_name: HF model name for the student.
        student_rev: HF revision (commit hash).
        full_sequences: list of [1, seq_len] tensors (full prompt+completion).
        prompt_lens: list of int prompt lengths.
        teacher_logits_list: list of sparse dicts with 'indices' and 'values'.
        tokenizer: tokenizer (for fallback token mapping).
        token_to_id: pre-built {token_str: token_id} mapping.
        gpu_memory_utilization: vLLM GPU memory fraction (default 0.15 for small students).
        max_model_len: max sequence length for vLLM.
        batch_size: number of prompts per vLLM request.
        early_stop_fn: callable(kl_means_so_far) -> bool.
        progress_cb: callable(prompts_done, total, running_kl_mean).
        timeout: per-model timeout in seconds.

    Returns:
        dict with:
          'status': 'scored' | 'early_stopped' | 'error'
          'kl_per_prompt': list of dicts with 'mean' key
          'load_time': vLLM startup time in seconds
          'scoring_time': inference + KL computation time in seconds
          'error': error message (if status == 'error')
    """
    import requests as req_lib

    result = {
        "status": "error",
        "kl_per_prompt": [],
        "load_time": 0.0,
        "scoring_time": 0.0,
        "error": None,
    }

    # Step 1: Start vLLM for student
    t0_load = time.time()
    vllm_ok = start_vllm_server(student_name, gpu_memory_utilization, max_model_len,
                                 revision=student_rev)
    result["load_time"] = time.time() - t0_load

    if not vllm_ok:
        result["error"] = "vLLM failed to start for student"
        stop_vllm_server()
        return result

    t0_score = time.time()
    model_start = time.time()
    kl_per_prompt = []
    kl_means = []
    early_stopped = False

    try:
        # Step 2: Score in batches
        n_prompts = len(full_sequences)
        for batch_start in range(0, n_prompts, batch_size):
            if time.time() - model_start > timeout:
                print(f"  [vllm-scoring] Timeout after {timeout}s", flush=True)
                early_stopped = True
                break

            batch_end = min(batch_start + batch_size, n_prompts)
            batch_token_ids = []
            for idx in range(batch_start, batch_end):
                seq = full_sequences[idx]
                token_ids = seq.squeeze(0).tolist()
                batch_token_ids.append(token_ids)

            # Send batch request to vLLM
            payload = {
                "model": "teacher",  # served-model-name from start_vllm_server
                "prompt": batch_token_ids,
                "max_tokens": 0,
                "prompt_logprobs": 128,
                "temperature": 1.0,
            }

            resp = None
            for attempt in range(3):
                try:
                    resp = req_lib.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=300)
                    resp.raise_for_status()
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise RuntimeError(f"vLLM request failed after 3 attempts: {e}")

            data = resp.json()
            choices = data.get("choices", [])

            if len(choices) != batch_end - batch_start:
                raise RuntimeError(
                    f"Expected {batch_end - batch_start} choices, got {len(choices)}")

            # Step 3: Extract logprobs and compute KL per prompt in this batch
            for bi, choice in enumerate(choices):
                prompt_idx = batch_start + bi
                prompt_len = prompt_lens[prompt_idx]
                tl_entry = teacher_logits_list[prompt_idx]

                if not _is_sparse_logits(tl_entry):
                    raise RuntimeError("vLLM student scoring requires sparse teacher logits")

                # Extract prompt_logprobs — try multiple response locations
                raw_prompt_lp = choice.get("prompt_logprobs")
                if raw_prompt_lp is None:
                    lp_obj = choice.get("logprobs")
                    if isinstance(lp_obj, dict):
                        raw_prompt_lp = lp_obj.get("prompt_logprobs") or lp_obj.get("top_logprobs")
                    elif isinstance(lp_obj, list):
                        raw_prompt_lp = lp_obj

                if not raw_prompt_lp:
                    raise RuntimeError(f"No prompt_logprobs in response for prompt {prompt_idx}")

                # Parse into sparse format (full sequence)
                student_sparse = _parse_vllm_prompt_logprobs(
                    raw_prompt_lp, token_to_id, tokenizer, k=128)

                # Align continuation positions:
                # Teacher cont_logits[j] = logits at position prompt_len-1+j, predicting token prompt_len+j
                # Student prompt_logprobs[prompt_len+j] predicts token prompt_len+j
                # So student positions [prompt_len .. end) align with teacher positions [0 .. cont_len)
                t_indices = tl_entry["indices"]  # [1, cont_len, k]
                t_values = tl_entry["values"]    # [1, cont_len, k]
                cont_len = t_indices.shape[1]

                s_start = prompt_len
                s_end = min(prompt_len + cont_len, student_sparse["indices"].shape[1])
                actual_cont_len = s_end - s_start

                if actual_cont_len == 0:
                    kl_per_prompt.append({"mean": 0.0})
                    kl_means.append(0.0)
                    continue

                s_cont_indices = student_sparse["indices"][:, s_start:s_end, :]
                s_cont_values = student_sparse["values"][:, s_start:s_end, :]
                t_cont_indices = t_indices[:, :actual_cont_len, :]
                t_cont_values = t_values[:, :actual_cont_len, :]

                # Detect if teacher values are logprobs (from vLLM) or raw logits (from HF top-k)
                max_val = t_cont_values.max().item()
                are_logprobs = (max_val <= 0.0)

                kl_per_pos = compute_kl_sparse_vs_sparse(
                    t_cont_indices, t_cont_values,
                    s_cont_indices, s_cont_values,
                    teacher_values_are_logprobs=are_logprobs,
                ).squeeze(0)

                kl_mean = kl_per_pos.mean().item()

                if math.isnan(kl_mean) or math.isinf(kl_mean):
                    raise RuntimeError(f"NaN/Inf KL at prompt {prompt_idx}")

                kl_per_prompt.append({"mean": round(kl_mean, 6)})
                kl_means.append(kl_mean)

                del student_sparse, s_cont_indices, s_cont_values, kl_per_pos

            # Progress reporting
            n_done = len(kl_per_prompt)
            running_mean = sum(kl_means) / len(kl_means) if kl_means else 0
            if progress_cb:
                progress_cb(n_done, n_prompts, running_mean)
            print(f"  [vllm-scoring] [{n_done}/{n_prompts}] avg KL={running_mean:.6f}", flush=True)

            # Early stopping check between batches
            if early_stop_fn and early_stop_fn(kl_means):
                print(f"  [vllm-scoring] Early stop at {n_done} prompts", flush=True)
                early_stopped = True
                break

    except Exception as e:
        result["error"] = str(e)[:500]
        print(f"  [vllm-scoring] Error: {e}", flush=True)
    finally:
        stop_vllm_server()

    result["scoring_time"] = time.time() - t0_score

    if kl_per_prompt:
        result["kl_per_prompt"] = kl_per_prompt
        result["status"] = "early_stopped" if early_stopped else "scored"
    elif not result["error"]:
        result["error"] = "No prompts scored"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Progress reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _write_phase(progress_path, students, phase, teacher_done=None, **extra):
    """Write a phase update to the progress file for the dashboard."""
    try:
        data = {
            "phase": phase,
            "students": students,
            "students_total": len(students),
            "prompts_total": extra.get("prompts_total", 0),
            "teacher_prompts_done": teacher_done,
            "completed": extra.get("completed", []),
            "current": extra.get("current", None),
        }
        with open(progress_path, "w") as pf:
            json.dump(data, pf)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated SN97 evaluation v3")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--revisions", default=None, help="Comma-separated revisions matching --students order")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt texts")
    parser.add_argument("--output", default="/home/eval_results.json")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-params-b", type=float, default=36.0)
    parser.add_argument("--block-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-logits", default="/home/teacher_cache.pt")
    parser.add_argument("--save-teacher-logits", default=None)
    parser.add_argument("--king", default=None, help="King model name — stays in VRAM between rounds")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use pure HF")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization (default 0.90)")
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--logprobs-k", type=int, default=128,
                        help="Top-k logprobs to store (128=sparse, 0=full vocab). Default 128.")
    parser.add_argument("--hf-batch-size", type=int, default=2,
                        help="Batch size for HF forward pass (default 2). Helps on H200 with 140GB VRAM.")
    parser.add_argument("--vllm-student-scoring", action="store_true", default=False,
                        help="Use vLLM prompt_logprobs for student scoring instead of HF forward pass (experimental)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of parallel requests to vLLM for teacher generation (default 4). "
                             "vLLM batches internally — higher values speed up generation but use more KV cache VRAM.")
    # Backward-compatible args (ignored)
    parser.add_argument("--max-logit-len", type=int, default=None,
                        help="DEPRECATED: ignored. Chunked forward pass replaces truncation.")
    parser.add_argument("--gpu", type=int, default=None, help="Ignored — kept for backward compat")
    parser.add_argument("--sequential", action="store_true", help="Ignored — kept for backward compat")
    parser.add_argument("--persistent-vllm", action="store_true", help="Ignored — kept for backward compat")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    # Parse revision pins (prevents weight-swap attacks between precheck and eval)
    if args.revisions:
        revisions = [r.strip() for r in args.revisions.split(",")]
        if len(revisions) != len(students):
            print(f"[eval] WARNING: {len(revisions)} revisions for {len(students)} students, ignoring revisions", flush=True)
            student_revisions = {s: "main" for s in students}
        else:
            student_revisions = dict(zip(students, revisions))
    else:
        student_revisions = {s: "main" for s in students}
    timings = {}

    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]

    # Progress file path
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")

    print(f"[eval] {len(prompts)} prompts (hash={prompts_hash}), {len(students)} students", flush=True)
    print(f"[eval] Teacher: {args.teacher}", flush=True)
    print(f"[eval] King: {args.king or 'none'}", flush=True)
    print(f"[eval] vLLM: {'disabled' if args.no_vllm else 'enabled'}", flush=True)
    print(f"[eval] VRAM: {gpu_mem_str()}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=False).input_ids.to(device)
        input_ids_list.append(ids)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher logits
    # ═══════════════════════════════════════════════════════════════════

    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    teacher_cache_loaded = False

    # Try loading from cache
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        try:
            t0 = time.time()
            cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
            if (len(cache.get("full_sequences", [])) == len(prompts)
                and cache.get("prompts_hash") == prompts_hash):
                full_sequences = [s.to(device) for s in cache["full_sequences"]]
                teacher_logits_list = cache["teacher_logits"]  # keep on CPU
                prompt_lens = cache["prompt_lens"]
                timings["teacher_cache_load"] = time.time() - t0
                timings["teacher_generation"] = 0.0
                timings["teacher_logits_pass"] = 0.0
                print(f"[eval] ✓ Cached logits ({timings['teacher_cache_load']:.1f}s, "
                      f"method={cache.get('generation_method', '?')})", flush=True)
                teacher_cache_loaded = True
            else:
                print(f"[eval] ✗ Cache stale — regenerating", flush=True)
        except Exception as e:
            print(f"[eval] ✗ Cache failed: {e}", flush=True)

    if not teacher_cache_loaded and not args.no_vllm:
        # ── vLLM generation path ──
        ensure_disk_space(args.teacher, threshold=70)

        # Clean stale teacher cache if hash doesn't match
        if args.teacher_logits and os.path.exists(args.teacher_logits):
            try:
                cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
                if cache.get("prompts_hash") != prompts_hash:
                    cache_size = os.path.getsize(args.teacher_logits) / (1024**3)
                    print(f"[disk] Stale teacher cache — removing ({cache_size:.1f}GB)", flush=True)
                    os.remove(args.teacher_logits)
                del cache
            except Exception:
                pass

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1a: vLLM teacher generation", flush=True)
        print(f"{'='*60}", flush=True)

        _write_phase(progress_path, students, "vllm_starting", prompts_total=len(prompts))
        t0 = time.time()
        vllm_ok = start_vllm_server(args.teacher, args.vllm_gpu_util, args.vllm_max_model_len)
        timings["vllm_startup"] = time.time() - t0

        # Build token-to-id mapping for vLLM logprobs decoding
        token_to_id = _build_token_to_id_map(tokenizer) if args.logprobs_k > 0 else None

        sequences_data = None
        if vllm_ok:
            _write_phase(progress_path, students, "vllm_generating", prompts_total=len(prompts))
            t0 = time.time()
            def _vllm_progress(done, total):
                _write_phase(progress_path, students, "vllm_generating",
                             teacher_done=done, prompts_total=total)
            try:
                sequences_data = generate_via_vllm(
                    prompts, tokenizer, args.max_new_tokens, args.block_seed,
                    logprobs_k=args.logprobs_k, token_to_id=token_to_id,
                    progress_cb=_vllm_progress, concurrency=args.concurrency,
                )
                timings["vllm_generation"] = time.time() - t0
                print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] vLLM generation failed: {e} — falling back to HF", flush=True)
            stop_vllm_server()
        else:
            print(f"[eval] vLLM failed to start — falling back to HF", flush=True)

        if sequences_data:
            # Check if vLLM returned logprobs for all prompts
            has_vllm_logprobs = all("sparse_logprobs" in d for d in sequences_data)

            if has_vllm_logprobs:
                # ── vLLM logprobs path: skip Phase 1b entirely ──
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: SKIPPED — vLLM logprobs available (top-{args.logprobs_k})", flush=True)
                print(f"{'='*60}", flush=True)

                for i, data in enumerate(sequences_data):
                    full_ids = data["full_ids"].to(device)
                    prompt_lens.append(data["prompt_len"])
                    full_sequences.append(full_ids)
                    teacher_logits_list.append(data["sparse_logprobs"])  # sparse dict on CPU

                timings["teacher_logits_pass"] = 0.0  # included in vllm_generation
                timings["teacher_hf_load"] = 0.0
                print(f"[eval] ✓ {len(sequences_data)} prompts with top-{args.logprobs_k} logprobs from vLLM", flush=True)
            else:
                # ── HF forward pass fallback for logits (batched) ──
                n_with_lp = sum(1 for d in sequences_data if "sparse_logprobs" in d)
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: HF teacher logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
                if n_with_lp > 0:
                    print(f"  ({n_with_lp}/{len(sequences_data)} had vLLM logprobs — using HF for all for consistency)", flush=True)
                print(f"{'='*60}", flush=True)

                ensure_disk_space(args.teacher, threshold=70)
                t0 = time.time()
                teacher = load_model(args.teacher, device)
                teacher.eval()
                timings["teacher_hf_load"] = time.time() - t0
                print(f"[eval] HF teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

                _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

                def _hf_progress(n_done):
                    if n_done % 10 == 0 or n_done == len(sequences_data):
                        print(f"  Logits [{n_done}/{len(sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
                    _write_phase(progress_path, students, "teacher_logits",
                                 teacher_done=n_done, prompts_total=len(prompts))

                t0 = time.time()
                teacher_logits_list, prompt_lens_out, full_seqs_out, n_chunked = hf_batched_forward(
                    teacher, sequences_data, device,
                    batch_size=args.hf_batch_size,
                    logprobs_k=args.logprobs_k,
                    progress_cb=_hf_progress,
                )
                prompt_lens = prompt_lens_out
                full_sequences = full_seqs_out

                if n_chunked:
                    print(f"[eval] Chunked forward pass used for {n_chunked}/{len(sequences_data)} sequences (chunk_size={HF_CHUNK_SIZE})", flush=True)

                timings["teacher_logits_pass"] = time.time() - t0
                print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)

                # Unload teacher
                del teacher
                free_gpu()
                print(f"[eval] Teacher unloaded. VRAM: {gpu_mem_str()}", flush=True)

            del sequences_data

            # Save cache if requested and enough disk
            if args.save_teacher_logits:
                st = os.statvfs(os.path.dirname(args.save_teacher_logits) or '/')
                free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
                # Sparse caches are tiny (~tens of MB) — lower threshold
                min_free = 5 if args.logprobs_k > 0 else 50
                if free_gb > min_free:
                    cache_tmp = args.save_teacher_logits + ".tmp"
                    gen_method = "vllm_logprobs" if has_vllm_logprobs else "vllm+hf"
                    torch.save({
                        "full_sequences": [s.cpu() for s in full_sequences],
                        "teacher_logits": teacher_logits_list,
                        "prompt_lens": prompt_lens,
                        "block_seed": args.block_seed,
                        "prompts_hash": prompts_hash,
                        "generation_method": gen_method,
                        "logprobs_k": args.logprobs_k,
                        "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
                    }, cache_tmp)
                    os.replace(cache_tmp, args.save_teacher_logits)
                    cache_size = os.path.getsize(args.save_teacher_logits) / (1024**2)
                    print(f"[eval] Cache saved ({cache_size:.1f}MB, method={gen_method})", flush=True)
                else:
                    print(f"[eval] Skipped cache save ({free_gb:.0f}GB free, need {min_free}GB)", flush=True)

            teacher_cache_loaded = True

    if not teacher_cache_loaded:
        # ── Pure HF fallback ──
        # Safety: ensure vLLM is killed before loading HF teacher
        stop_vllm_server()
        time.sleep(3)
        free_gpu()

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1 FALLBACK: HF teacher generation + logit extraction", flush=True)
        print(f"{'='*60}", flush=True)

        ensure_disk_space(args.teacher, threshold=70)

        t0 = time.time()
        teacher = load_model(args.teacher, device)
        teacher.eval()
        timings["teacher_hf_load"] = time.time() - t0
        print(f"[eval] Teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        _write_phase(progress_path, students, "teacher_generation", teacher_done=0, prompts_total=len(prompts))

        # Step 1: Generate continuations
        t0 = time.time()
        hf_sequences_data = []
        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                prompt_len = ids.shape[1]
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True)
                if args.block_seed is not None:
                    torch.manual_seed(args.block_seed + i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(args.block_seed + i)
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    gen_kwargs.update(do_sample=False)
                output_ids = teacher.generate(ids, **gen_kwargs)
                gen_len = output_ids.shape[1] - prompt_len
                hf_sequences_data.append({
                    "full_ids": output_ids,
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                })
                print(f"  Gen [{i+1}/{len(input_ids_list)}]: {prompt_len}+{gen_len} tokens, VRAM: {gpu_mem_str()}", flush=True)
                _write_phase(progress_path, students, "teacher_generation", teacher_done=i + 1, prompts_total=len(prompts))
        timings["teacher_generation"] = time.time() - t0

        # Step 2: Batched forward pass for logit extraction
        print(f"\n[eval] HF logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
        _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

        def _hf_fb_progress(n_done):
            if n_done % 10 == 0 or n_done == len(hf_sequences_data):
                print(f"  Logits [{n_done}/{len(hf_sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
            _write_phase(progress_path, students, "teacher_logits",
                         teacher_done=n_done, prompts_total=len(prompts))

        t0_logits = time.time()
        teacher_logits_list, prompt_lens, full_sequences, n_chunked = hf_batched_forward(
            teacher, hf_sequences_data, device,
            batch_size=args.hf_batch_size,
            logprobs_k=args.logprobs_k,
            progress_cb=_hf_fb_progress,
        )
        timings["teacher_logits_pass"] = time.time() - t0_logits
        if n_chunked:
            print(f"[eval] Chunked forward pass used for {n_chunked}/{len(hf_sequences_data)} sequences", flush=True)
        del hf_sequences_data

        cache_path = args.save_teacher_logits or os.path.join(os.path.dirname(args.output), "teacher_cache.pt")
        torch.save({
            "full_sequences": [s.cpu() for s in full_sequences],
            "teacher_logits": teacher_logits_list,
            "prompt_lens": prompt_lens,
            "block_seed": args.block_seed,
            "prompts_hash": prompts_hash,
            "generation_method": "hf",
            "logprobs_k": args.logprobs_k,
            "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
        }, cache_path)
        del teacher
        free_gpu()
        print(f"[eval] HF generation done in {timings['teacher_generation']:.1f}s, logits in {timings['teacher_logits_pass']:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1c: Chunked GPU processing setup
    # ═══════════════════════════════════════════════════════════════════
    _write_phase(progress_path, students, "gpu_precompute", teacher_done=len(prompts), prompts_total=len(prompts))
    n_sparse = sum(1 for tl in teacher_logits_list if _is_sparse_logits(tl))
    storage_mode = "sparse" if n_sparse == len(teacher_logits_list) else ("mixed" if n_sparse > 0 else "dense")
    print(f"\n[eval] Teacher logits: {len(teacher_logits_list)} prompts on CPU ({storage_mode}, {n_sparse}/{len(teacher_logits_list)} sparse)", flush=True)
    # Keep teacher logits on CPU, compute softmax per-chunk during student scoring
    teacher_log_probs = None
    teacher_probs = None

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1d: Filter short completions & log completion lengths
    # ═══════════════════════════════════════════════════════════════════
    completion_lens = [full_sequences[i].shape[1] - prompt_lens[i] for i in range(len(prompts))]
    min_cl, max_cl = min(completion_lens), max(completion_lens)
    avg_cl = sum(completion_lens) / len(completion_lens)
    print(f"[eval] Completion tokens: min={min_cl} max={max_cl} avg={avg_cl:.0f} across {len(prompts)} prompts", flush=True)

    # Filter out prompts with fewer than MIN_COMPLETION_TOKENS
    n_filtered = 0
    if MIN_COMPLETION_TOKENS > 0:
        keep_indices = [i for i, cl in enumerate(completion_lens) if cl >= MIN_COMPLETION_TOKENS]
        n_filtered = len(prompts) - len(keep_indices)
        if n_filtered > 0:
            skipped_lens = [completion_lens[i] for i in range(len(prompts)) if i not in set(keep_indices)]
            print(f"[eval] Filtered {n_filtered}/{len(prompts)} prompts with <{MIN_COMPLETION_TOKENS} completion tokens "
                  f"(skipped lens: {skipped_lens})", flush=True)
            full_sequences = [full_sequences[i] for i in keep_indices]
            teacher_logits_list = [teacher_logits_list[i] for i in keep_indices]
            prompt_lens = [prompt_lens[i] for i in keep_indices]
            prompts = [prompts[i] for i in keep_indices]
            completion_lens = [completion_lens[i] for i in keep_indices]
            print(f"[eval] Remaining: {len(prompts)} prompts after filtering", flush=True)
        else:
            print(f"[eval] All {len(prompts)} prompts have >={MIN_COMPLETION_TOKENS} completion tokens — no filtering needed", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1e: Save eval data for reproducibility
    # ═══════════════════════════════════════════════════════════════════
    try:
        eval_data_path = os.path.join(os.path.dirname(args.output), "eval_data.json")
        eval_data = []
        for i, prompt_text in enumerate(prompts):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens[i]
            continuation_ids = full_seq[0, prompt_len:].tolist()
            continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            eval_data.append({
                "prompt": prompt_text,
                "continuation": continuation_text,
                "prompt_tokens": prompt_len,
                "continuation_tokens": len(continuation_ids),
            })
        with open(eval_data_path, "w") as f:
            json.dump({
                "teacher": args.teacher,
                "max_new_tokens": args.max_new_tokens,
                "block_seed": args.block_seed,
                "n_prompts": len(prompts),
                "data": eval_data,
            }, f, indent=2)
        print(f"[eval] Saved eval data ({len(prompts)} prompts) to {eval_data_path}", flush=True)
    except Exception as e:
        print(f"[eval] Failed to save eval data (non-fatal): {e}", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Student scoring
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: Student scoring ({len(students)} models)", flush=True)
    print(f"{'='*60}", flush=True)

    # Build token_to_id map for vLLM student scoring (if enabled)
    vllm_token_to_id = None
    if args.vllm_student_scoring:
        vllm_token_to_id = _build_token_to_id_map(tokenizer)
        print(f"[eval] vLLM student scoring: ENABLED (token map: {len(vllm_token_to_id)} entries)", flush=True)

    # Resume support
    prior_results = {}
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prior = json.load(f)
            prior_results = prior.get("students", {})
            scored = [n for n, d in prior_results.items()
                      if d.get("status") != "load_failed" and d.get("kl_global_avg") is not None]
            if scored:
                print(f"[eval] Resuming: {len(scored)} already scored", flush=True)
        except Exception:
            pass

    results = {
        "teacher": args.teacher,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_len": args.max_prompt_len,
        "block_seed": args.block_seed,
        "n_prompts": len(prompts),
        "n_prompts_filtered": n_filtered,
        "min_completion_tokens": MIN_COMPLETION_TOKENS,
        "students": {},
    }
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Live progress
    progress_lock = threading.Lock()
    live_progress = {
        "phase": "scoring", "students": students,
        "students_total": len(students), "prompts_total": len(prompts),
        "completed": [], "current": None,
    }
    def _write_progress():
        """Write current live progress to disk for dashboard consumption."""
        try:
            with progress_lock:
                with open(progress_path, "w") as pf:
                    json.dump(live_progress, pf)
        except Exception:
            pass
    _write_progress()

    # Early stopping state
    best_kl_so_far = None
    best_kl_per_prompt_cumulative = None
    MIN_PROMPTS_EARLY_STOP = 7
    PER_MODEL_TIMEOUT = 600

    # King stays in VRAM
    king_model = None
    king_name = args.king

    # Prefetch executor
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None

    vram_before_students = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for student_idx, student_name in enumerate(students):
        student_rev = student_revisions.get(student_name, "main")
        # Skip already scored
        if student_name in results["students"]:
            prior = results["students"][student_name]
            kl = prior.get("kl_global_avg")
            print(f"\n[eval] {student_name}: SKIP (already scored, KL={kl})", flush=True)
            if kl and kl > 0.001 and kl < float('inf'):
                if best_kl_so_far is None or kl < best_kl_so_far:
                    best_kl_so_far = kl
                    kl_per_prompt = prior.get("kl_per_prompt", [])
                    if kl_per_prompt:
                        best_kl_per_prompt_cumulative = []
                        s = 0.0
                        for j, v in enumerate(kl_per_prompt):
                            s += v
                            best_kl_per_prompt_cumulative.append(s / (j + 1))
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Student: {student_name}" +
              (" (KING — stays in VRAM)" if student_name == king_name else ""), flush=True)

        model_start = time.time()
        ensure_disk_space(args.teacher)

        # Prefetch next student
        if student_idx + 1 < len(students):
            next_name = students[student_idx + 1]
            next_rev = student_revisions.get(next_name, "main")
            if next_name not in results["students"] and next_name != king_name:
                prefetch_future = prefetch_executor.submit(prefetch_model, next_name, next_rev)

        # Load student (or reuse king)
        live_progress["phase"] = "loading_student"
        live_progress["current"] = {"student_name": student_name, "student_idx": student_idx, "prompts_done": 0}
        _write_progress()

        is_king = (student_name == king_name)

        # ── vLLM student scoring path (if enabled, skip for king) ──
        if args.vllm_student_scoring and not is_king:
            can_early_stop_v = (student_idx > 0) and (best_kl_so_far is not None)

            def _vllm_early_stop_check(means_so_far):
                """Early stopping check for vLLM batch scoring."""
                n = len(means_so_far)
                if not can_early_stop_v or n < MIN_PROMPTS_EARLY_STOP:
                    return False
                running_mean = sum(means_so_far) / n
                running_var = sum((x - running_mean) ** 2 for x in means_so_far) / (n - 1)
                running_se = math.sqrt(running_var / n)
                student_lower = running_mean - 1.96 * running_se
                if best_kl_per_prompt_cumulative and n <= len(best_kl_per_prompt_cumulative):
                    best_at_n = best_kl_per_prompt_cumulative[n - 1]
                else:
                    best_at_n = best_kl_so_far
                if best_at_n and best_at_n <= 0.001:
                    best_at_n = best_kl_so_far if best_kl_so_far and best_kl_so_far > 0.001 else float('inf')
                return student_lower > best_at_n

            def _vllm_progress(done, total, running_mean):
                live_progress["phase"] = "scoring"
                live_progress["current"] = {
                    "student_name": student_name, "student_idx": student_idx,
                    "prompts_done": done, "prompts_total": total,
                    "kl_running_mean": round(running_mean, 6) if running_mean else None,
                    "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    "method": "vllm",
                }
                _write_progress()

            vllm_result = score_student_via_vllm(
                student_name=student_name,
                student_rev=student_rev,
                full_sequences=full_sequences,
                prompt_lens=prompt_lens,
                teacher_logits_list=teacher_logits_list,
                tokenizer=tokenizer,
                token_to_id=vllm_token_to_id,
                gpu_memory_utilization=0.15,
                max_model_len=args.vllm_max_model_len,
                batch_size=20,
                early_stop_fn=_vllm_early_stop_check,
                progress_cb=_vllm_progress,
                timeout=PER_MODEL_TIMEOUT,
            )

            if vllm_result["status"] in ("scored", "early_stopped"):
                # Success — record results and continue to next student
                kl_per_prompt_v = vllm_result["kl_per_prompt"]
                kl_means_v = [d["mean"] for d in kl_per_prompt_v]
                kl_avg_v = sum(kl_means_v) / len(kl_means_v)
                n_scored_v = len(kl_per_prompt_v)
                status_v = vllm_result["status"]
                early_stopped_v = (status_v == "early_stopped")

                print(f"  \u2192 KL={kl_avg_v:.6f} ({n_scored_v}/{len(prompts)} prompts, {status_v}, vLLM)", flush=True)

                student_result = {
                    "status": status_v,
                    "kl_global_avg": round(kl_avg_v, 6),
                    "kl_per_prompt": [d["mean"] for d in kl_per_prompt_v],
                    "prompts_scored": n_scored_v,
                    "scoring_time": round(vllm_result.get("scoring_time", 0), 1),
                    "load_time": round(vllm_result.get("load_time", 0), 1),
                    "early_stopped": early_stopped_v,
                    "scoring_method": "vllm",
                }
                results["students"][student_name] = student_result

                # Update best KL tracking
                if kl_avg_v > 0.001 and not early_stopped_v:
                    if best_kl_so_far is None or kl_avg_v < best_kl_so_far:
                        best_kl_so_far = kl_avg_v
                        best_kl_per_prompt_cumulative = []
                        s = 0.0
                        for j, d in enumerate(kl_per_prompt_v):
                            s += d["mean"]
                            best_kl_per_prompt_cumulative.append(s / (j + 1))
                        print(f"  \u2192 New best: KL={kl_avg_v:.6f}", flush=True)

                # Save incremental results
                results["timings"] = {k: round(v, 1) for k, v in timings.items()}
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

                live_progress["completed"].append({
                    "student_name": student_name,
                    "status": status_v,
                    "kl": round(kl_avg_v, 6),
                    "prompts_scored": n_scored_v,
                })
                live_progress["current"] = None
                _write_progress()

                # Cleanup
                clean_model_cache(student_name, args.teacher)
                if prefetch_future:
                    try:
                        prefetch_future.result(timeout=1)
                    except Exception:
                        pass
                    prefetch_future = None

                continue  # Skip HF path for this student
            else:
                print(f"  [vllm-scoring] Failed: {vllm_result.get('error', '?')} \u2014 falling back to HF", flush=True)

        # ── HF scoring path (original) ──
        student = None
        load_time = 0.0

        if is_king and king_model is not None:
            student = king_model
            print(f"[eval] King reused from VRAM", flush=True)
        else:
            try:
                t0 = time.time()
                student = load_model(student_name, device, revision=student_rev)
                student.eval()
                load_time = time.time() - t0
                student_vram_gb = (torch.cuda.memory_allocated() - vram_before_students) / 1024**3
                print(f"[eval] Loaded in {load_time:.1f}s, VRAM: {student_vram_gb:.1f}GB, total: {gpu_mem_str()}", flush=True)
            except Exception as e:
                print(f"[eval] FAILED to load: {e}", flush=True)
                results["students"][student_name] = {
                    "status": "load_failed", "error": str(e)[:500], "kl_global_avg": None}
                results["timings"] = {k: round(v, 1) for k, v in timings.items()}
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                live_progress["completed"].append({"student_name": student_name, "status": "load_failed"})
                live_progress["current"] = None
                _write_progress()
                try: del student
                except: pass
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            # VRAM fraud check
            MAX_STUDENT_VRAM_GB = 20.0
            if not is_king and student_vram_gb > MAX_STUDENT_VRAM_GB:
                msg = f"FRAUD: student VRAM delta {student_vram_gb:.1f}GB > {MAX_STUDENT_VRAM_GB}GB"
                print(f"  ⚠️ {msg}", flush=True)
                results["students"][student_name] = {
                    "status": "fraud_vram", "reason": msg,
                    "vram_gb": round(student_vram_gb, 1), "kl_global_avg": float('inf')}
                del student
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            if is_king:
                king_model = student
                print(f"[eval] King loaded — stays in VRAM", flush=True)

        # ── Activation fingerprint (for functional copy detection) ──
        if student is not None:
            try:
                fp_start = time.time()
                fp = compute_activation_fingerprint(student, device)
                fp_time = time.time() - fp_start
                if fp:
                    results["students"].setdefault(student_name, {})["activation_fingerprint"] = fp
                    print(f"[eval] Fingerprint computed in {fp_time:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] Fingerprint failed: {e}", flush=True)

        # ── Score: per-prompt with early stopping ──
        can_early_stop = (student_idx > 0) and (best_kl_so_far is not None)
        kl_per_prompt = []
        prompt_kl_means = []
        scoring_error = None
        early_stopped = False

        t0 = time.time()
        with torch.no_grad():
            for i in range(len(prompts)):
                try:
                    full_seq = full_sequences[i]
                    prompt_len = prompt_lens[i]
                    tl_entry = teacher_logits_list[i]
                    is_sparse = _is_sparse_logits(tl_entry)

                    # Student forward pass
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]

                    topk_shadow = {}

                    if is_sparse:
                        # ── Sparse teacher logits path (top-k from vLLM or HF) ──
                        t_indices = tl_entry["indices"]  # [1, seq_len, k]
                        t_values = tl_entry["values"]    # [1, seq_len, k]
                        min_len = min(cont_s.shape[1], t_indices.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_indices, t_values
                        else:
                            s_cont_slice = cont_s[:, :min_len, :]
                            # Detect if values are logprobs (from vLLM) or raw logits (from HF top-k)
                            # vLLM logprobs are negative and typically < 0; raw logits can be positive
                            # Heuristic: if max value > 0, they're logits; if all <= 0, they're logprobs
                            max_val = t_values[:, :min_len, :].max().item()
                            are_logprobs = (max_val <= 0.0)
                            kl_per_pos = compute_kl_from_sparse(
                                t_indices[:, :min_len, :], t_values[:, :min_len, :],
                                s_cont_slice, values_are_logprobs=are_logprobs
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()
                            del t_indices, t_values, s_cont_slice, kl_per_pos
                    else:
                        # ── Dense teacher logits path (legacy / full-vocab) ──
                        tl = tl_entry.to(device).float()
                        t_log_p = F.log_softmax(tl, dim=-1)
                        t_p = t_log_p.exp()
                        del tl

                        min_len = min(cont_s.shape[1], t_log_p.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_log_p, t_p
                        else:
                            t_lp_slice = t_log_p[:, :min_len, :]
                            t_p_slice = t_p[:, :min_len, :]
                            s_cont_slice = cont_s[:, :min_len, :]
                            kl_per_pos = compute_kl_from_precomputed(
                                t_lp_slice, t_p_slice, s_cont_slice
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()

                            # Shadow top-k KL metrics (logged but not used for scoring)
                            try:
                                s_log_p_full = F.log_softmax(s_cont_slice.float(), dim=-1)
                                for k_val in (100, 1000):
                                    _, topk_idx = t_p_slice.topk(k_val, dim=-1)
                                    t_topk = t_lp_slice.gather(-1, topk_idx)
                                    s_topk = s_log_p_full.gather(-1, topk_idx)
                                    t_topk_norm = F.log_softmax(t_topk, dim=-1)
                                    s_topk_norm = F.log_softmax(s_topk, dim=-1)
                                    topk_kl = F.kl_div(s_topk_norm, t_topk_norm,
                                                       log_target=True, reduction="none").sum(dim=-1).squeeze(0)
                                    topk_shadow[f"kl_top{k_val}"] = round(topk_kl.mean().item(), 6)
                                    del topk_idx, t_topk, s_topk, t_topk_norm, s_topk_norm, topk_kl
                                del s_log_p_full
                            except Exception:
                                pass

                            del kl_per_pos, t_lp_slice, t_p_slice, s_cont_slice
                            del t_log_p, t_p

                    del s_logits, cont_s
                    if i % 20 == 0:
                        torch.cuda.empty_cache()

                    if math.isnan(kl_mean) or math.isinf(kl_mean):
                        print(f"  [prompt {i}] KL={kl_mean} — invalid, stopping", flush=True)
                        scoring_error = f"NaN/Inf KL at prompt {i}"
                        break

                    prompt_result = {"mean": round(kl_mean, 6)}
                    if topk_shadow:
                        prompt_result.update(topk_shadow)
                    kl_per_prompt.append(prompt_result)
                    prompt_kl_means.append(kl_mean)

                    running_mean = sum(prompt_kl_means) / len(prompt_kl_means)
                    live_progress["phase"] = "scoring"
                    live_progress["current"] = {
                        "student_name": student_name, "student_idx": student_idx,
                        "prompts_done": i + 1, "prompts_total": len(prompts),
                        "kl_running_mean": round(running_mean, 6),
                        "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    }
                    _write_progress()

                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{len(prompts)}] KL={kl_mean:.6f} (avg: {running_mean:.6f})", flush=True)

                except RuntimeError as e:
                    scoring_error = str(e)
                    if "out of memory" in str(e).lower():
                        print(f"  [prompt {i}] OOM", flush=True)
                    else:
                        print(f"  [prompt {i}] RuntimeError: {e}", flush=True)
                    free_gpu()
                    break
                except Exception as e:
                    scoring_error = str(e)
                    print(f"  [prompt {i}] Error: {e}", flush=True)
                    free_gpu()
                    break

                # Early stopping (same-point comparison)
                n = len(prompt_kl_means)
                if can_early_stop and n >= MIN_PROMPTS_EARLY_STOP:
                    running_mean = sum(prompt_kl_means) / n
                    running_var = sum((x - running_mean) ** 2 for x in prompt_kl_means) / (n - 1)
                    running_se = math.sqrt(running_var / n)
                    student_lower = running_mean - 1.96 * running_se

                    if best_kl_per_prompt_cumulative and n <= len(best_kl_per_prompt_cumulative):
                        best_at_n = best_kl_per_prompt_cumulative[n - 1]
                    else:
                        best_at_n = best_kl_so_far
                    if best_at_n and best_at_n <= 0.001:
                        best_at_n = best_kl_so_far if best_kl_so_far and best_kl_so_far > 0.001 else float('inf')

                    if student_lower > best_at_n:
                        print(f"  [early stop] prompt {n}: CI lower {student_lower:.6f} > best@{n} {best_at_n:.6f}", flush=True)
                        early_stopped = True
                        break

                if time.time() - model_start > PER_MODEL_TIMEOUT:
                    print(f"  [timeout] {PER_MODEL_TIMEOUT}s", flush=True)
                    early_stopped = True
                    break

        scoring_time = time.time() - t0

        # Record results
        if scoring_error and not kl_per_prompt:
            results["students"][student_name] = {
                "status": "scoring_error", "error": scoring_error[:500], "kl_global_avg": None}
        elif kl_per_prompt:
            kl_avg = sum(d["mean"] for d in kl_per_prompt) / len(kl_per_prompt)
            n_scored = len(kl_per_prompt)
            status = "early_stopped" if early_stopped else ("partial" if scoring_error else "scored")
            print(f"  → KL={kl_avg:.6f} ({n_scored}/{len(prompts)} prompts, {status})", flush=True)

            topk_aggs = {}
            for k_label in ("kl_top100", "kl_top1000"):
                vals = [d.get(k_label) for d in kl_per_prompt if d.get(k_label) is not None]
                if vals:
                    topk_aggs[k_label] = round(sum(vals) / len(vals), 6)

            student_result = {
                "status": status,
                "kl_global_avg": round(kl_avg, 6),
                "kl_per_prompt": [d["mean"] for d in kl_per_prompt],
                "prompts_scored": n_scored,
                "scoring_time": round(scoring_time, 1),
                "load_time": round(load_time, 1),
                "early_stopped": early_stopped,
            }
            if topk_aggs:
                student_result["shadow_topk"] = topk_aggs
                print(f"  → Shadow top-k: {topk_aggs}", flush=True)
            results["students"][student_name] = student_result

            if kl_avg > 0.001 and not early_stopped and not scoring_error:
                if best_kl_so_far is None or kl_avg < best_kl_so_far:
                    best_kl_so_far = kl_avg
                    best_kl_per_prompt_cumulative = []
                    s = 0.0
                    for j, d in enumerate(kl_per_prompt):
                        s += d["mean"]
                        best_kl_per_prompt_cumulative.append(s / (j + 1))
                    print(f"  → New best: KL={kl_avg:.6f}", flush=True)

        # Save incremental
        results["timings"] = {k: round(v, 1) for k, v in timings.items()}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        live_progress["completed"].append({
            "student_name": student_name,
            "status": results["students"].get(student_name, {}).get("status", "unknown"),
            "kl": results["students"].get(student_name, {}).get("kl_global_avg"),
            "prompts_scored": len(kl_per_prompt),
        })
        live_progress["current"] = None
        _write_progress()

        # Cleanup — DON'T unload king
        if not is_king:
            del student
            free_gpu()
            clean_model_cache(student_name, args.teacher)
        else:
            torch.cuda.empty_cache()

        # Wait for prefetch
        if prefetch_future:
            try:
                prefetch_future.result(timeout=1)
            except Exception:
                pass
            prefetch_future = None

    # Final save
    results["timings"] = {k: round(v, 1) for k, v in timings.items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] DONE — {len(results['students'])} students", flush=True)
    for k, v in sorted(timings.items()):
        print(f"  {k}: {v:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    prefetch_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
