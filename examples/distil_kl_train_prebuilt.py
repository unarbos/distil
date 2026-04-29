#!/usr/bin/env python3
"""
KL Distillation Training for Bittensor Subnet 97.

This script keeps the same core training logic as examples/distil_kl_train.py.
It now exposes only:
1) build_eval_cache_api: precompute deterministic eval prompts/targets once
2) train: online-teacher KL training from streaming data

Examples:
    # Build eval cache once from validator API prompts
    python examples/distil_kl_train_prebuilt.py build_eval_cache_api \
      --teacher Qwen/Qwen3.5-35B-A3B \
      --eval_cache_path ./eval_cache_api.pt

    # Train repeatedly (online teacher mode)
    python examples/distil_kl_train_prebuilt.py train \
      --teacher_gpu 0 --student_gpu 1 \
      --output_dir ./distil-checkpoints
"""

import argparse
import gc
import shutil
import json
import logging
import os
import random
import re
import time
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Keep defaults aligned with distil_kl_train.py
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
STUDENT_MODEL = "Qwen/Qwen3.5-4B"
DATASET = "karpathy/climbmix-400b-shuffle"
LR = 1e-4
WARMUP = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 640
KL_START_POS = 128
SAMPLES_PER_STEP = 100
SAVE_EVERY = 500
MIN_CHARS = 2560
DEFAULT_EVAL_DATA_URL = "https://distil.arbos.life/api/eval-data"


def _require_positive(name: str, value: int):
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def _validate_common_seq_args(max_seq_len: int, kl_start_pos: int):
    _require_positive("max_seq_len", max_seq_len)
    if kl_start_pos < 0:
        raise ValueError(f"kl_start_pos must be >= 0 (got {kl_start_pos})")
    # Keep same training safety margin used later in filtering.
    if max_seq_len <= kl_start_pos + 10:
        raise ValueError(
            f"max_seq_len must be > kl_start_pos + 10 "
            f"(got max_seq_len={max_seq_len}, kl_start_pos={kl_start_pos})"
        )


def _gpu_span(start_gpu: int, gpu_count: int, label: str) -> list[int]:
    if gpu_count <= 0:
        raise ValueError(f"{label}_gpu_count must be > 0 (got {gpu_count})")
    n_cuda = torch.cuda.device_count()
    if n_cuda <= 0:
        raise RuntimeError("No CUDA devices available.")
    if start_gpu < 0 or start_gpu >= n_cuda:
        raise ValueError(
            f"{label}_gpu start index out of range: {start_gpu} (available: 0..{n_cuda-1})"
        )
    end_gpu = start_gpu + gpu_count - 1
    if end_gpu >= n_cuda:
        raise ValueError(
            f"{label}_gpu span exceeds available GPUs: start={start_gpu}, count={gpu_count}, "
            f"needs up to {end_gpu}, available max index={n_cuda-1}"
        )
    return list(range(start_gpu, start_gpu + gpu_count))


def _device_map_and_memory(gpus: list[int]):
    if len(gpus) == 1:
        return {"": gpus[0]}, None
    # Constrain auto-sharding to the selected GPU subset.
    max_memory = {}
    for idx in gpus:
        total_bytes = torch.cuda.get_device_properties(idx).total_memory
        gib = max(1, int((total_bytes * 0.90) / (1024**3)))
        max_memory[idx] = f"{gib}GiB"
    max_memory["cpu"] = "256GiB"
    return "auto", max_memory


def _first_param_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def kl_loss(student_logits, teacher_logits, start_pos=KL_START_POS):
    """Forward KL(teacher || student) from start_pos onward."""
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(-1).mean()


def kl_loss_masked(
    student_logits,
    teacher_logits,
    start_pos: int,
    tail_mask: torch.Tensor,
):
    """
    Forward KL(teacher || student) from start_pos onward.

    Averages the same way as the per-sample loop: mean over valid positions within each
    sequence, then mean across batch. tail_mask: [B, L - start_pos], 1 on real tokens.
    """
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    per_pos = (t_p * (t_log_p - s_log_p)).sum(-1)
    m = tail_mask.to(device=per_pos.device, dtype=per_pos.dtype)
    per_seq = (per_pos * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    return per_seq.mean()


def kl_loss_masked_positions(
    student_logits,
    teacher_logits,
    loss_mask: torch.Tensor,
):
    """Forward KL(teacher || student) at positions selected by ``loss_mask`` [B, L]."""
    s = student_logits.contiguous()
    t = teacher_logits.detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    per_pos = (t_p * (t_log_p - s_log_p)).sum(-1)
    m = loss_mask.to(device=per_pos.device, dtype=per_pos.dtype)
    per_seq = (per_pos * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    return per_seq.mean()


def _pad_token_batch(
    tokens: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad 1D token tensors to [B, Lmax] on CPU; return (input_ids, position_mask)."""
    lengths = [int(x.shape[0]) for x in tokens]
    lmax = max(lengths)
    b = len(tokens)
    out = torch.full((b, lmax), pad_id, dtype=torch.long)
    mask = torch.zeros((b, lmax), dtype=torch.float32)
    for i, x in enumerate(tokens):
        li = lengths[i]
        out[i, :li] = x
        mask[i, :li] = 1.0
    return out, mask


def _build_loss_mask(tokens: list[torch.Tensor], loss_starts: list[int]) -> torch.Tensor:
    """Build a [B, L] float mask enabling KL from each sample's response start onward."""
    lengths = [int(x.shape[0]) for x in tokens]
    lmax = max(lengths)
    mask = torch.zeros((len(tokens), lmax), dtype=torch.float32)
    for i, (length, loss_start) in enumerate(zip(lengths, loss_starts)):
        start = max(0, min(int(loss_start), length - 1))
        mask[i, start:length] = 1.0
    return mask


# Patterns that mark the start of an assistant reply in a chat-template transcript.
_CHAT_ASSISTANT_PATTERNS = (
    re.compile(r"(?s)<\|im_start\|>\s*assistant(?:[^\S\r\n]*\r?\n+|\s+)"),
    re.compile(r"(?s)<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>\s*\r?\n+"),
    re.compile(r"(?im)^\s*assistant\s*:\s*"),
)


def _split_chat_transcript_text(text: str) -> tuple[str, str] | None:
    """Return (prompt_prefix, assistant_completion) from a full chat transcript string.

    Finds the *last* assistant turn header and splits there.  Returns ``None`` when
    no recognisable header is found or the completion would be empty.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for pattern in _CHAT_ASSISTANT_PATTERNS:
        matches = list(pattern.finditer(text))
        for match in reversed(matches):
            boundary = match.end()
            completion = text[boundary:]
            stripped = completion.strip()
            if not stripped:
                continue
            # Skip if the completion is only a closing im_end tag.
            if re.fullmatch(r"(?:<\|im_end\|>\s*)+", stripped, flags=re.IGNORECASE):
                continue
            return text[:boundary], completion
    return None


def _first_nonempty_text(item: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_chat_messages(messages: list) -> list[dict[str, str]]:
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        if not isinstance(content, str) or not content.strip():
            continue
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        else:
            role = "system" if role == "system" else "user"
        normalized.append({"role": role, "content": content.strip()})
    return normalized


def _render_chat_messages(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        parts = [f"{msg['role'].title()}: {msg['content']}" for msg in messages]
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)


def _extract_prompt_completion_text(item: dict, tokenizer) -> tuple[str, str] | None:
    """Try to split a JSONL row into (prompt, completion) strings.

    Priority:
    1. ``messages`` / ``conversations`` list → apply_chat_template split.
    2. ``text`` field containing a chat-template transcript → regex split.
    3. Separate prompt / completion fields (question/answer, instruction/output, …).
    """
    messages = item.get("messages") or item.get("conversations")
    if isinstance(messages, list) and messages:
        normalized = _normalize_chat_messages(messages)
        if len(normalized) >= 2 and normalized[-1]["role"] == "assistant":
            prompt = _render_chat_messages(tokenizer, normalized[:-1], add_generation_prompt=True)
            full = _render_chat_messages(tokenizer, normalized, add_generation_prompt=False)
            if full.startswith(prompt):
                completion = full[len(prompt):]
                if completion.strip():
                    return prompt, completion

    text = _first_nonempty_text(item, ("text",))
    if text:
        split = _split_chat_transcript_text(text)
        if split is not None:
            return split

    prompt = _first_nonempty_text(
        item,
        ("question", "query", "instruction", "prompt", "problem", "input", "task"),
    )
    completion = _first_nonempty_text(
        item,
        ("answer", "response", "output", "solution", "completion", "chosen", "target"),
    )
    if prompt and completion and prompt != completion:
        return prompt, completion
    return None


def _chunk_list(items: list, chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize empty metric list")
    vals = sorted(float(v) for v in values)
    n = len(vals)
    p50 = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {
        "mean": float(sum(vals) / n),
        "std": float(statistics.stdev(vals) if n > 1 else 0.0),
        "p50": float(p50),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "n": int(n),
    }


def _paired_t_stats(deltas: list[float]) -> tuple[float, float]:
    def _betacf(a: float, b: float, x: float) -> float:
        # Continued fraction for incomplete beta (Numerical Recipes).
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    def _regularized_beta(x: float, a: float, b: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * _betacf(a, b, x) / a
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

    def _student_t_cdf(t_value: float, dof: int) -> float:
        if dof <= 0:
            return 0.5
        x = dof / (dof + t_value * t_value)
        ib = _regularized_beta(x, 0.5 * dof, 0.5)
        if t_value >= 0.0:
            return 1.0 - 0.5 * ib
        return 0.5 * ib

    n = len(deltas)
    if n < 2:
        return 0.0, 1.0
    mean_delta = sum(deltas) / n
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    if se <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    t_stat = mean_delta / se
    cdf = _student_t_cdf(t_stat, n - 1)
    p_one_sided = max(0.0, min(1.0, 1.0 - cdf))
    return float(t_stat), float(p_one_sided)


def _update_ema(prev: float | None, value: float, beta: float) -> float:
    if prev is None:
        return float(value)
    return float(beta * prev + (1.0 - beta) * value)


def _gate_score(delta_ema: float, p_value: float) -> float:
    """
    Composite ranking score for stable-vs-king checkpoints.

    Higher is better: reward stronger positive margin, penalize weaker significance.
    """
    return float(delta_ema - 0.5 * p_value)


def _json_lists_to_tuples(value):
    if isinstance(value, list):
        return tuple(_json_lists_to_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _json_lists_to_tuples(v) for k, v in value.items()}
    return value


def _prepare_eval_cache(
    teacher_eval,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    teacher_eval.eval()
    teacher_device = next(teacher_eval.parameters()).device
    cache = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(teacher_device)
            prompt_len = prompt_ids.shape[1]
            prompt_seed = int(seed) + i
            try:
                gen = torch.Generator(device=teacher_device)
                gen.manual_seed(prompt_seed)
                full_seq = teacher_eval.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    generator=gen,
                )
            except ValueError as e:
                # Some model wrappers/custom generation stacks reject `generator`.
                # Fallback keeps deterministic sampling by seeding within a forked RNG scope.
                if "not used by the model: ['generator']" not in str(e):
                    raise
                device_idx = teacher_device.index if teacher_device.type == "cuda" else None
                devices = [device_idx] if device_idx is not None else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(prompt_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(prompt_seed)
                    full_seq = teacher_eval.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                    )
            t_logits = teacher_eval(full_seq).logits.float()
            t_cont = t_logits[:, prompt_len - 1 : -1, :]
            t_log_p = F.log_softmax(t_cont, dim=-1).cpu()
            cache.append(
                {
                    "full_seq": full_seq.cpu(),
                    "prompt_len": int(prompt_len),
                    "teacher_log_probs": t_log_p,
                }
            )
            del prompt_ids, full_seq, t_logits, t_cont, t_log_p
    return cache


def _fetch_eval_data_bundle(eval_data_url: str) -> dict:
    req = urllib.request.Request(
        eval_data_url,
        headers={"User-Agent": "distil_kl_train_prebuilt/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {eval_data_url}")
    return payload


def _extract_eval_prompts_from_bundle(bundle: dict, max_prompts: int) -> list[str]:
    from eval.dataset import format_prompt

    rows = bundle.get("data") or []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("prompt", "")
        if not raw:
            continue
        formatted = format_prompt(raw)
        if not formatted:
            continue
        out.append(formatted)
        if max_prompts > 0 and len(out) >= max_prompts:
            break
    return out


def _save_eval_cache_payload(
    path: Path,
    cache: list[dict],
    prompts: list[str],
    meta: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "meta": meta,
            "prompts": prompts,
            "cache": cache,
        },
        path,
    )


def _load_eval_cache_payload(path: Path) -> tuple[list[dict], list[str], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dict payload")
    cache = payload.get("cache")
    prompts = payload.get("prompts") or []
    meta = payload.get("meta") or {}
    if not isinstance(cache, list) or len(cache) < 2:
        raise ValueError(f"{path} has invalid or too-small eval cache")
    return cache, prompts, meta


def _validate_loaded_eval_cache_meta(
    cache_meta: dict,
    args,
    cache_len: int,
):
    cached_teacher = str(cache_meta.get("teacher") or "").strip()
    if cached_teacher and cached_teacher != str(args.teacher):
        raise ValueError(
            f"Eval cache teacher mismatch: cache={cached_teacher!r} current={args.teacher!r}. "
            "Use --rebuild_eval_cache."
        )
    cached_prompts = cache_meta.get("eval_prompts")
    if cached_prompts is not None and int(cached_prompts) != int(cache_len):
        raise ValueError(
            f"Eval cache metadata mismatch: meta.eval_prompts={cached_prompts} but "
            f"cache has {cache_len} entries. Use --rebuild_eval_cache."
        )
    cached_mode = str(cache_meta.get("prompt_source_mode") or "").strip()
    current_mode = "api" if getattr(args, "eval_use_api_prompts", False) else "dataset"
    if cached_mode and cached_mode != current_mode:
        raise ValueError(
            f"Eval cache prompt source mismatch: cache={cached_mode!r} current={current_mode!r}. "
            "Use --rebuild_eval_cache."
        )
    if getattr(args, "eval_use_api_prompts", False):
        cached_url = cache_meta.get("eval_data_url")
        if cached_url and str(cached_url) != str(args.eval_data_url):
            raise ValueError(
                f"Eval cache API URL mismatch: cache={cached_url!r} current={args.eval_data_url!r}. "
                "Use --rebuild_eval_cache."
            )
    else:
        cached_max_new_tokens = cache_meta.get("max_new_tokens")
        if cached_max_new_tokens is not None and int(cached_max_new_tokens) != int(args.eval_max_new_tokens):
            raise ValueError(
                f"Eval cache max_new_tokens mismatch: cache={cached_max_new_tokens} "
                f"current={args.eval_max_new_tokens}. Use --rebuild_eval_cache."
            )
        cached_eval_seed = cache_meta.get("eval_seed")
        if cached_eval_seed is not None and int(cached_eval_seed) != int(args.eval_seed):
            raise ValueError(
                f"Eval cache eval_seed mismatch: cache={cached_eval_seed} current={args.eval_seed}. "
                "Use --rebuild_eval_cache."
            )
        cached_dataset = cache_meta.get("eval_dataset")
        if cached_dataset and str(cached_dataset) != str(args.eval_dataset):
            raise ValueError(
                f"Eval cache dataset mismatch: cache={cached_dataset!r} current={args.eval_dataset!r}. "
                "Use --rebuild_eval_cache."
            )
        cached_block = cache_meta.get("eval_block_number")
        if cached_block is not None and int(cached_block) != int(args.eval_block_number):
            raise ValueError(
                f"Eval cache block mismatch: cache={cached_block} current={args.eval_block_number}. "
                "Use --rebuild_eval_cache."
            )


def _mean_token_kl_forward_chunked(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    vocab_chunk: int = 4096,
) -> float:
    """
    Forward KL(teacher || student) averaged over token positions.

    Same math as ``kl_loss`` / ``(t_p * (t_log_p - s_log_p)).sum(-1).mean()`` but
    sums over vocabulary in chunks so we never materialize full ``[B, T, V]`` tensors
    (``F.kl_div`` on large V still peaks VRAM for huge vocabs).
    """
    b, t, v = student_log_probs.shape
    if t <= 0 or v <= 0:
        return 0.0
    device = student_log_probs.device
    # [B, T] running sum of per-token KL contributions
    acc = torch.zeros((b, t), device=device, dtype=torch.float32)
    for v0 in range(0, v, vocab_chunk):
        v1 = min(v0 + vocab_chunk, v)
        t_lp = teacher_log_probs[:, :, v0:v1].float()
        s_lp = student_log_probs[:, :, v0:v1].float()
        t_p = t_lp.exp()
        acc += (t_p * (t_lp - s_lp)).sum(dim=-1)
    return float(acc.mean().item())


def _evaluate_against_king(
    student,
    king_eval,
    eval_cache: list[dict],
) -> tuple[list[float], list[float], dict]:

    king_eval.eval()
    student.eval()
    king_device = next(king_eval.parameters()).device
    student_device = next(student.parameters()).device

    eval_scores = []
    king_scores = []
    with torch.no_grad():
        for item in eval_cache:
            full_seq = item["full_seq"]
            prompt_len = int(item["prompt_len"])
            t_log_p = item["teacher_log_probs"].to(student_device)

            s_logits = student(full_seq.to(student_device)).logits.float()
            s_cont = s_logits[:, prompt_len - 1 : -1, :]
            k_logits = king_eval(full_seq.to(king_device)).logits.float()
            k_cont = k_logits[:, prompt_len - 1 : -1, :]

            min_len_s = min(s_cont.shape[1], t_log_p.shape[1])
            min_len_k = min(k_cont.shape[1], t_log_p.shape[1])
            s_lp = F.log_softmax(s_cont[:, :min_len_s, :], dim=-1)
            k_lp = F.log_softmax(k_cont[:, :min_len_k, :], dim=-1)
            # Contiguous copies so we can drop the full teacher log-prob tensor and free VRAM.
            t_lp_s = t_log_p[:, :min_len_s, :].contiguous()
            t_lp_k = t_log_p[:, :min_len_k, :].to(device=king_device, dtype=torch.float32).contiguous()
            del t_log_p

            s_kl = _mean_token_kl_forward_chunked(s_lp, t_lp_s)
            del s_logits, s_cont, s_lp, t_lp_s
            if student_device.type == "cuda":
                torch.cuda.empty_cache()
            k_kl = _mean_token_kl_forward_chunked(k_lp, t_lp_k)
            eval_scores.append(float(s_kl))
            king_scores.append(float(k_kl))

            del k_logits, k_cont, k_lp, t_lp_k

    student.train()
    eval_stats = _summary_stats(eval_scores)
    king_stats = _summary_stats(king_scores)
    deltas = [k - e for k, e in zip(king_scores, eval_scores)]
    t_stat, p_val = _paired_t_stats(deltas)
    return eval_scores, king_scores, {
        "p": p_val,
        "t": t_stat,
        "delta": float(sum(deltas) / len(deltas)),
    }


def _looks_like_hf_repo_id(ref: str) -> bool:
    ref = ref.strip()
    if "/" not in ref or ref.startswith(("/", ".", "~")):
        return False
    left, _, right = ref.partition("/")
    return bool(left) and bool(right) and "\\" not in ref


# Files to preserve from the origin checkpoint (Hub or local). `save_pretrained` may rewrite a
# slimmer config.json; we copy these into each step_* after save so validators see repo-identical JSON.
_ORIGIN_ARTIFACT_NAMES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "chat_template.jinja",
    "vocab.json",
    "added_tokens.json",
    "merges.txt",
    "tokenizer.model",
)


def _ensure_origin_model_configs_cached(
    origin_model_ref: str,
    revision: str | None,
    cache_dir: Path,
) -> list[Path]:
    """
    Cache origin model config/tokenizer metadata for later copy into checkpoints.

    - Local directory: copy matching files from disk (byte-identical).
    - Hugging Face repo id (``org/name``): download the same filenames from the Hub so
      ``config.json`` matches the published model, not only the in-memory config from
      ``save_pretrained``.
    """
    src = Path(origin_model_ref)
    if src.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        names = set(_ORIGIN_ARTIFACT_NAMES)
        for p in src.iterdir():
            if not p.is_file():
                continue
            if p.name in names or p.name.endswith(".json"):
                dst = cache_dir / p.name
                shutil.copy2(p, dst)
                copied.append(dst)
        return copied

    if not _looks_like_hf_repo_id(origin_model_ref):
        return []

    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    repo_id = origin_model_ref.strip()
    rev_kw = {} if revision in (None, "") else {"revision": str(revision)}
    for filename in _ORIGIN_ARTIFACT_NAMES:
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
                **rev_kw,
            )
            copied.append(Path(out))
        except Exception as e:
            if filename == "config.json":
                log.warning(
                    "Could not download origin config.json from Hub repo %s%s: %s",
                    repo_id,
                    f"@{revision}" if rev_kw else "",
                    e,
                )
            continue
    if not (cache_dir / "config.json").is_file():
        log.warning(
            "Origin Hub cache for %s has no config.json; checkpoints will only contain "
            "save_pretrained config (may omit Hub-only fields).",
            repo_id,
        )
    else:
        log.info(
            "Cached %s origin Hub artifact(s) from %s%s into %s",
            len(copied),
            repo_id,
            f" (revision={revision})" if rev_kw else "",
            cache_dir,
        )
    return copied


def _copy_cached_origin_configs(cache_dir: Path, target_dir: Path):
    if not cache_dir.exists():
        return
    for p in cache_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, target_dir / p.name)


def _read_json_file(path: Path) -> dict:
    """Load a JSON object from disk; raise with a clear message if empty or corrupt."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Cannot read {path}: {e}") from e
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    raw = raw.strip()
    if not raw:
        raise ValueError(
            f"{path} is empty or whitespace-only. The checkpoint may be incomplete "
            "(e.g. process killed while saving). Use an older step_* directory, or omit "
            "--resume_from / --resume_latest."
        )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{path} is not valid JSON ({e}). The file may be corrupted or partially written."
        ) from e
    if not isinstance(out, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(out).__name__}")
    return out


def _global_step_from_checkpoint_dir(checkpoint_dir: Path) -> int | None:
    """Parse training step from a directory named ``step_<N>``."""
    name = checkpoint_dir.name
    if not name.startswith("step_"):
        return None
    tail = name.split("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _read_train_state_or_repair(checkpoint_dir: Path) -> tuple[dict, bool]:
    """
    Load ``train_state.json`` from a checkpoint directory.

    If the file is missing, empty, or invalid JSON, recover a minimal dict when the
    directory name is ``step_<N>`` (so resume still works after a crash mid-save).

    Returns ``(state, repaired)`` where ``repaired`` is True if recovery was used.
    """
    train_state_path = checkpoint_dir / "train_state.json"
    err: Exception | None = None
    if train_state_path.is_file():
        try:
            return _read_json_file(train_state_path), False
        except (ValueError, OSError) as e:
            err = e
    else:
        err = FileNotFoundError(str(train_state_path))

    repaired = _global_step_from_checkpoint_dir(checkpoint_dir)
    if repaired is None:
        raise ValueError(
            f"Cannot load train state from {checkpoint_dir}: {err}. "
            f"Expected a readable {train_state_path.name} or a directory named step_<N>."
        ) from err

    log.warning(
        "train_state.json in %s is missing or unreadable (%s). "
        "Rebuilding minimal resume state from directory name (global_step=%s). "
        "data_state was not restored; the data stream may replay samples. "
        "beat_streak / best_beating_delta reset.",
        checkpoint_dir,
        err,
        repaired,
    )
    minimal = {
        "global_step": repaired,
        "data_state": None,
        "best_beating_delta": float("-inf"),
        "beat_streak": 0,
    }
    try:
        _atomic_write_json(train_state_path, minimal)
    except OSError as werr:
        log.warning("Could not write repaired train_state.json (%s).", werr)
    return minimal, True


def _atomic_write_json(path: Path, data: dict):
    """Write JSON atomically (tmp + replace) so crashes never leave an empty train_state.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _open_hf_streaming_iterator(dataset: str, split: str, skip_rows: int):
    """
    Open a streaming HF ``datasets`` iterator, optionally skipping the first ``skip_rows``
    raw examples (fast path uses ``IterableDataset.skip`` when available).
    """
    from datasets import load_dataset

    skip_rows = int(max(0, skip_rows))
    dataset_path = Path(str(dataset)).expanduser()
    if dataset_path.exists() and dataset_path.suffix.lower() in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files=str(dataset_path), split=split, streaming=True)
    else:
        ds = load_dataset(dataset, split=split, streaming=True)
    if skip_rows > 0 and hasattr(ds, "skip"):
        ds = ds.skip(skip_rows)
        return iter(ds)
    if skip_rows > 0:
        log.warning(
            "Streaming dataset has no .skip(); advancing %s rows with next() (slow for large skips).",
            skip_rows,
        )
        it = iter(ds)
        for _ in range(skip_rows):
            next(it)
        return it
    return iter(ds)


class StreamingTokenStream:
    """Streams raw dataset text and tokenizes on the fly."""

    def __init__(
        self,
        teacher_model: str,
        dataset: str = DATASET,
        max_seq_len: int = MAX_SEQ_LEN,
        min_chars: int = MIN_CHARS,
        dataset_split: str = "train",
        dataset_skip_rows: int = 0,
    ):
        from transformers import AutoTokenizer

        if min_chars < 0:
            raise ValueError(f"min_chars must be >= 0 (got {min_chars})")
        self._teacher_model = teacher_model
        self.dataset = dataset
        self.dataset_split = str(dataset_split)
        self.max_seq_len = int(max_seq_len)
        self.min_chars = int(min_chars)
        skip = int(max(0, dataset_skip_rows))
        self._ds = _open_hf_streaming_iterator(self.dataset, self.dataset_split, skip)
        self._consumed = skip
        self.manifest = {
            "dataset": self.dataset,
            "max_seq_len": self.max_seq_len,
            "kl_start_pos": None,
            "num_samples": 0,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize_item(self, item: dict) -> tuple[torch.Tensor, int] | None:
        """Tokenize one dataset row; return (input_ids, loss_start_token_idx) or None."""
        encode_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_seq_len,
        }
        split = _extract_prompt_completion_text(item, self.tokenizer)
        if split is not None:
            prompt_text, completion_text = split
            full_text = prompt_text + completion_text
            # Try to get char-level offsets so we can map prompt end → token index precisely.
            try:
                enc = self.tokenizer(full_text, return_offsets_mapping=True, **encode_kwargs)
                ids = enc.input_ids.squeeze(0).to(torch.int32)
                offsets = enc["offset_mapping"][0].tolist()  # list of (char_start, char_end)
                prompt_chars = len(prompt_text)
                loss_start = None
                for tok_idx, (cs, ce) in enumerate(offsets):
                    if ce <= cs:
                        continue
                    if cs >= prompt_chars:
                        loss_start = max(0, tok_idx - 1)
                        break
            except (TypeError, KeyError, Exception):
                enc = self.tokenizer(full_text, **encode_kwargs)
                ids = enc.input_ids.squeeze(0).to(torch.int32)
                prompt_ids = self.tokenizer(prompt_text, **encode_kwargs).input_ids.squeeze(0)
                loss_start = max(0, int(prompt_ids.shape[0]) - 1)

            if loss_start is None:
                # Fallback: count prompt tokens directly.
                prompt_ids = self.tokenizer(prompt_text, **encode_kwargs).input_ids.squeeze(0)
                loss_start = max(0, int(prompt_ids.shape[0]) - 1)

            if ids.shape[0] <= loss_start + 1:
                return None
            return ids, int(loss_start)

        # No split found — treat as plain text with a sentinel loss_start of -1
        # so the caller substitutes the global kl_start_pos.
        text = _first_nonempty_text(item, ("text",))
        if not text or len(text) < self.min_chars:
            return None
        ids = self.tokenizer(text, **encode_kwargs).input_ids.squeeze(0).to(torch.int32)
        return ids, -1

    def get_batch(self, n: int):
        out = []
        scanned = 0
        max_scan = max(n * 20, n + 1)
        while len(out) < n and scanned < max_scan:
            try:
                item = next(self._ds)
            except StopIteration:
                break
            scanned += 1
            self._consumed += 1
            sample = self._tokenize_item(item)
            if sample is None:
                continue
            ids, loss_start = sample
            out.append({"input_ids": ids, "loss_start": int(loss_start)})
        return out

    @property
    def position(self):
        return self._consumed

    def state_dict(self) -> dict:
        return {
            "variant": "streaming_token_stream",
            "consumed": int(self._consumed),
            "dataset": self.dataset,
            "dataset_split": self.dataset_split,
            "max_seq_len": int(self.max_seq_len),
            "min_chars": int(self.min_chars),
            "teacher_model": self._teacher_model,
        }

    def load_state_dict(self, state: dict):
        if state.get("variant") != "streaming_token_stream":
            raise ValueError(
                "Invalid streaming data_state: expected variant 'streaming_token_stream'"
            )
        if state.get("dataset") != self.dataset:
            raise ValueError(
                f"Streaming resume dataset mismatch: checkpoint={state.get('dataset')} "
                f"current={self.dataset}"
            )
        ckpt_split = state.get("dataset_split")
        if ckpt_split is not None and ckpt_split != self.dataset_split:
            raise ValueError(
                f"Streaming resume dataset_split mismatch: checkpoint={ckpt_split!r} "
                f"current={self.dataset_split!r}"
            )
        if int(state.get("max_seq_len", self.max_seq_len)) != int(self.max_seq_len):
            raise ValueError(
                f"Streaming resume max_seq_len mismatch: checkpoint={state.get('max_seq_len')} "
                f"current={self.max_seq_len}"
            )
        if int(state.get("min_chars", self.min_chars)) != int(self.min_chars):
            raise ValueError(
                f"Streaming resume min_chars mismatch: checkpoint={state.get('min_chars')} "
                f"current={self.min_chars}"
            )
        if state.get("teacher_model") != self._teacher_model:
            raise ValueError(
                f"Streaming resume teacher_model mismatch: checkpoint={state.get('teacher_model')} "
                f"current={self._teacher_model}"
            )
        target = int(state.get("consumed", 0))
        if target < 0:
            raise ValueError(f"Invalid consumed in checkpoint: {target}")
        if target > 0:
            log.info(
                "Restoring streaming dataset position (%s raw rows in split %r; using skip when available)...",
                target,
                self.dataset_split,
            )
        self._ds = _open_hf_streaming_iterator(self.dataset, self.dataset_split, target)
        self._consumed = target


def build_eval_cache_api(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher_gpus = _gpu_span(args.teacher_gpu, args.teacher_gpu_count, "teacher")
    teacher_device_map, teacher_max_memory = _device_map_and_memory(teacher_gpus)
    out_path = Path(args.eval_cache_path)
    if out_path.exists() and not args.rebuild_eval_cache:
        raise FileExistsError(
            f"Eval cache already exists: {out_path}. Use --rebuild_eval_cache to overwrite."
        )

    bundle = _fetch_eval_data_bundle(args.eval_data_url)
    prompts = _extract_eval_prompts_from_bundle(bundle, args.eval_prompts)
    if len(prompts) < 2:
        raise RuntimeError(f"Need at least 2 formatted prompts from eval_data; got {len(prompts)}")
    max_new_tokens = int(bundle.get("max_new_tokens") or args.eval_max_new_tokens)
    eval_seed = int(bundle.get("block_seed") or args.eval_seed)

    log.info(
        "Building eval cache from API: prompts=%s max_new_tokens=%s seed=%s",
        len(prompts),
        max_new_tokens,
        eval_seed,
    )
    teacher_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": teacher_device_map,
        "trust_remote_code": True,
    }
    if teacher_max_memory is not None:
        teacher_load_kwargs["max_memory"] = teacher_max_memory
    teacher_eval = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_load_kwargs)
    teacher_eval.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache = _prepare_eval_cache(
        teacher_eval=teacher_eval,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        seed=eval_seed,
    )
    meta = {
        "teacher": args.teacher,
        "prompt_source_mode": "api",
        "eval_data_url": args.eval_data_url,
        "eval_dataset": None,
        "eval_block_number": None,
        "eval_block_hash": None,
        "eval_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "eval_seed": eval_seed,
        "block_seed": bundle.get("block_seed"),
        "source_n_prompts": bundle.get("n_prompts"),
        "saved_at": _utc_now_iso(),
    }
    _save_eval_cache_payload(out_path, cache, prompts, meta)
    log.info("Saved eval cache to %s", out_path)


def train_online(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from eval.dataset import sample_prompts_from_dataset, format_prompt

    _require_positive("samples_per_step", args.samples_per_step)
    _require_positive("save_every", args.save_every)
    _validate_common_seq_args(args.max_seq_len, args.kl_start_pos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.resume_from and args.resume_latest:
        raise ValueError("Use only one of --resume_from or --resume_latest.")
    if args.early_stop_on_beat_consecutive > 0 and args.eval_every_steps <= 0:
        raise ValueError(
            "--early_stop_on_beat_consecutive requires --eval_every_steps > 0."
        )

    def _resolve_resume_dir() -> Path | None:
        if args.resume_from:
            p = Path(args.resume_from)
            if not p.exists():
                raise FileNotFoundError(f"--resume_from path does not exist: {p}")
            return p
        if not args.resume_latest:
            return None
        candidates = []
        for d in output_dir.glob("step_*"):
            if not d.is_dir():
                continue
            stem = d.name.split("_", 1)[-1]
            if stem.isdigit():
                candidates.append((int(stem), d))
        if not candidates:
            raise FileNotFoundError(
                f"--resume_latest requested but no step_* directories found in {output_dir}"
            )
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    resume_dir = _resolve_resume_dir()

    ds_skip = int(getattr(args, "dataset_skip_rows", 0) or 0)
    if ds_skip < 0:
        raise ValueError("--dataset_skip_rows must be >= 0")
    if resume_dir and ds_skip > 0:
        log.warning(
            "--dataset_skip_rows is ignored when resuming from a checkpoint; "
            "streaming position comes from train_state.json data_state (or consumed)."
        )
    stream_skip_rows = 0 if resume_dir else ds_skip
    if stream_skip_rows > 0:
        log.info(
            "Streaming: skipping first %s raw example(s) in split %r of %s.",
            stream_skip_rows,
            args.dataset_split,
            args.dataset,
        )
    data = StreamingTokenStream(
        teacher_model=args.teacher,
        dataset=args.dataset,
        max_seq_len=args.max_seq_len,
        min_chars=args.min_chars,
        dataset_split=args.dataset_split,
        dataset_skip_rows=stream_skip_rows,
    )

    manifest = data.manifest

    if not args.no_wandb:
        import wandb

        _wb_timeout = max(30, int(args.wandb_init_timeout))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or "distil-kl-online",
            config=vars(args),
            settings=wandb.Settings(init_timeout=_wb_timeout),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_gpus = _gpu_span(args.teacher_gpu, args.teacher_gpu_count, "teacher")
    student_gpus = _gpu_span(args.student_gpu, args.student_gpu_count, "student")
    teacher_device_map, teacher_max_memory = _device_map_and_memory(teacher_gpus)
    student_device_map, student_max_memory = _device_map_and_memory(student_gpus)

    teacher = None
    log.info("Loading teacher (%s) on GPUs %s...", args.teacher, teacher_gpus)
    teacher_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": teacher_device_map,
        "trust_remote_code": True,
    }
    if teacher_max_memory is not None:
        teacher_load_kwargs["max_memory"] = teacher_max_memory
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_load_kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_in_dev = _first_param_device(teacher)
    if teacher_in_dev.type == "cuda":
        log.info(
            "  Teacher primary %s: %.1fGB",
            teacher_in_dev,
            torch.cuda.memory_allocated(teacher_in_dev.index) / 1e9,
        )

    student_source = str(resume_dir) if resume_dir else args.student
    log.info("Loading student (%s) on GPUs %s...", student_source, student_gpus)
    student_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": student_device_map,
        "trust_remote_code": True,
    }
    if student_max_memory is not None:
        student_load_kwargs["max_memory"] = student_max_memory
    student = AutoModelForCausalLM.from_pretrained(student_source, **student_load_kwargs)
    student.train()
    student_in_dev = _first_param_device(student)
    log.info(
        f"  Student: {sum(p.numel() for p in student.parameters()):,} params, "
        f"{(torch.cuda.memory_allocated(student_in_dev.index)/1e9) if student_in_dev.type == 'cuda' else 0.0:.1f}GB"
    )

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    resume_global_step = 0
    resume_state: dict | None = None
    scheduler_path: Path | None = None
    if resume_dir:
        optimizer_path = resume_dir / "optimizer.pt"
        scheduler_path = resume_dir / "scheduler.pt"
        if not optimizer_path.exists():
            raise FileNotFoundError(f"Missing optimizer.pt in resume checkpoint: {resume_dir}")
        if not scheduler_path.exists():
            log.warning(
                "scheduler.pt missing in %s. Scheduler will start fresh from step 0.",
                resume_dir,
            )
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        resume_state, _train_state_repaired = _read_train_state_or_repair(resume_dir)
        resume_global_step = int(resume_state.get("global_step", 0))

    if args.max_steps <= 0:
        raise ValueError(
            "Online mode has no finite manifest sample count. "
            "Set --max_steps to a positive value."
        )
    if args.max_steps > 0:
        target_steps = args.max_steps
        stop_reason = f"max_steps={args.max_steps}"

    # Cosine schedule total must match this run length (hardcoding 100k flattens LR for short runs).
    scheduler_total_steps = max(int(target_steps), int(args.warmup_steps) + 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_steps), scheduler_total_steps
    )
    log.info(
        "LR scheduler: cosine over %s steps (warmup=%s).",
        scheduler_total_steps,
        args.warmup_steps,
    )
    if resume_dir and scheduler_path is not None and scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    origin_cfg_cache_dir = output_dir / "_origin_model_configs"
    copied_origin_cfgs = _ensure_origin_model_configs_cached(
        args.student,
        args.student_revision,
        origin_cfg_cache_dir,
    )
    if copied_origin_cfgs:
        log.info(
            "Cached %s origin model config files from %s",
            len(copied_origin_cfgs),
            args.student,
        )

    eval_prompts = []
    teacher_eval = None
    king_eval = None
    eval_cache = []
    if args.eval_every_steps > 0:
        king_start_gpu = args.king_gpu if args.king_gpu is not None else args.teacher_gpu
        king_gpus = _gpu_span(king_start_gpu, args.king_gpu_count, "king")
        king_device_map, king_max_memory = _device_map_and_memory(king_gpus)
        if not args.king_repo:
            raise ValueError("--king_repo is required when --eval_every_steps > 0")
        cache_path = Path(args.eval_cache_path) if args.eval_cache_path else None
        if cache_path and cache_path.exists() and not args.rebuild_eval_cache:
            log.info("Loading eval cache from %s", cache_path)
            eval_cache, eval_prompts, cache_meta = _load_eval_cache_payload(cache_path)
            _validate_loaded_eval_cache_meta(cache_meta, args, len(eval_cache))
            log.info(
                "Loaded eval cache: prompts=%s max_new_tokens=%s seed=%s",
                len(eval_prompts),
                cache_meta.get("max_new_tokens"),
                cache_meta.get("eval_seed"),
            )
        else:
            if args.eval_use_api_prompts:
                bundle = _fetch_eval_data_bundle(args.eval_data_url)
                eval_prompts = _extract_eval_prompts_from_bundle(bundle, args.eval_prompts)
                eval_tokens = int(bundle.get("max_new_tokens") or args.eval_max_new_tokens)
                eval_seed = int(bundle.get("block_seed") or args.eval_seed)
                log.info(
                    "Eval prompts from API: prompts=%s max_new_tokens=%s seed=%s",
                    len(eval_prompts),
                    eval_tokens,
                    eval_seed,
                )
            else:
                raw_prompts = sample_prompts_from_dataset(
                    n=args.eval_prompts,
                    block_number=args.eval_block_number,
                    block_hash=(args.eval_block_hash or None),
                    dataset_name=args.eval_dataset,
                )
                for text in raw_prompts:
                    formatted = format_prompt(text)
                    if formatted:
                        eval_prompts.append(formatted)
                    if len(eval_prompts) >= args.eval_prompts:
                        break
                eval_tokens = args.eval_max_new_tokens
                eval_seed = args.eval_seed
            if len(eval_prompts) < 2:
                raise RuntimeError(
                    f"Need at least 2 eval prompts after filtering; got {len(eval_prompts)}"
                )
            if teacher is None:
                log.info("Eval enabled: loading teacher model for periodic king-comparison metrics...")
                teacher_eval_kwargs = {
                    "dtype": torch.bfloat16,
                    "device_map": teacher_device_map,
                    "trust_remote_code": True,
                }
                if teacher_max_memory is not None:
                    teacher_eval_kwargs["max_memory"] = teacher_max_memory
                teacher_eval = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_eval_kwargs)
            else:
                teacher_eval = teacher
        log.info("Eval enabled: loading king model (%s) on GPUs %s...", args.king_repo, king_gpus)
        king_load_kwargs = {
            "revision": args.king_revision,
            "dtype": torch.bfloat16,
            "device_map": king_device_map,
            "trust_remote_code": True,
        }
        if king_max_memory is not None:
            king_load_kwargs["max_memory"] = king_max_memory
        king_eval = AutoModelForCausalLM.from_pretrained(args.king_repo, **king_load_kwargs)
        king_eval.eval()
        if not eval_cache:
            if teacher_eval is not None:
                teacher_eval.eval()
            log.info("Preparing deterministic eval cache from teacher continuations...")
            eval_cache = _prepare_eval_cache(
                teacher_eval=teacher_eval,
                tokenizer=tokenizer,
                prompts=eval_prompts,
                max_new_tokens=eval_tokens,
                seed=eval_seed,
            )
            if cache_path:
                meta = {
                    "teacher": args.teacher,
                    "prompt_source_mode": "api" if args.eval_use_api_prompts else "dataset",
                    "eval_data_url": args.eval_data_url if args.eval_use_api_prompts else None,
                    "eval_dataset": None if args.eval_use_api_prompts else args.eval_dataset,
                    "eval_block_number": None if args.eval_use_api_prompts else args.eval_block_number,
                    "eval_block_hash": None if args.eval_use_api_prompts else (args.eval_block_hash or None),
                    "eval_prompts": len(eval_prompts),
                    "max_new_tokens": eval_tokens,
                    "eval_seed": eval_seed,
                    "saved_at": _utc_now_iso(),
                }
                _save_eval_cache_payload(cache_path, eval_cache, eval_prompts, meta)
                log.info("Saved eval cache to %s", cache_path)
        log.info("Prepared eval cache for %s prompts", len(eval_cache))
        log.info(
            "Periodic king eval configured: %s prompts every %s steps",
            len(eval_prompts),
            args.eval_every_steps,
        )

    sdev = _first_param_device(student)
    tdev = _first_param_device(teacher) if teacher is not None else None

    log.info("=== Starting training (online mode) ===")
    log.info(
        f"  LR: {args.lr}, Warmup: {args.warmup_steps}, Samples/step: {args.samples_per_step}"
    )
    log.info(f"  Seq len: {args.max_seq_len}, KL from pos {args.kl_start_pos}")
    log.info(f"  Streaming dataset: {args.dataset} [{args.dataset_split}]")
    log.info(
        "  Online teacher: chunked forwards (online_chunk_size=%s). "
        "Larger chunks = faster but more student-GPU memory for logits/activations.",
        getattr(args, "online_chunk_size", 1),
    )

    best_beating_delta = float("-inf")
    beat_streak = 0
    metrics_jsonl = Path(args.metrics_jsonl) if args.metrics_jsonl else (output_dir / "king_eval_metrics.jsonl")
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gate_delta_ema: float | None = None
    gate_streak = 0
    best_gate_score = float("-inf")
    best_gate_step = 0

    global_step = resume_global_step
    if resume_dir:
        assert resume_state is not None
        data_state = resume_state.get("data_state")
        if data_state is not None:
            data.load_state_dict(data_state)
            log.info("Resumed data stream state from checkpoint.")
        else:
            log.warning(
                "No data_state in checkpoint. Data stream starts fresh; this may replay samples."
            )
        best_beating_delta = float(resume_state.get("best_beating_delta", float("-inf")))
        beat_streak = int(resume_state.get("beat_streak", 0))
        gate_delta_ema_raw = resume_state.get("gate_delta_ema")
        gate_delta_ema = (
            None if gate_delta_ema_raw is None else float(gate_delta_ema_raw)
        )
        gate_streak = int(resume_state.get("gate_streak", 0))
        best_gate_score = float(resume_state.get("best_gate_score", float("-inf")))
        best_gate_step = int(resume_state.get("best_gate_step", 0))
        log.info(
            "Resumed from %s | step=%s | data_pos=%s | best_delta=%.6f | beat_streak=%s | gate_streak=%s",
            resume_dir,
            global_step,
            data.position,
            best_beating_delta if math.isfinite(best_beating_delta) else float("-inf"),
            beat_streak,
            gate_streak,
        )

    while global_step < target_steps:

        t0 = time.time()
        batch = data.get_batch(args.samples_per_step)
        # Resolve per-sample loss starts; fall back to the global kl_start_pos for
        # plain-text rows that couldn't be split (loss_start == -1).
        samples = []
        for row in batch:
            ids = row["input_ids"]
            loss_start = int(row.get("loss_start", -1))
            if loss_start < 0 or loss_start >= ids.shape[0] - 1:
                loss_start = int(args.kl_start_pos)
            if ids.shape[0] > loss_start + 10:
                samples.append({"input_ids": ids, "loss_start": loss_start})
        if not samples:
            log.warning("No valid tokens in batch after filtering.")
            break

        optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        # Batched forwards reduce kernel overhead; chunking caps peak VRAM (full logits [B,L,V]).
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        chunk_sz = max(1, int(getattr(args, "online_chunk_size", 1)))
        n_tokens = len(samples)
        total_loss = 0.0
        _dt = next(student.parameters()).dtype
        for sub in _chunk_list(samples, chunk_sz):
            n_sub = len(sub)
            sub_tokens = [row["input_ids"] for row in sub]
            sub_starts = [int(row["loss_start"]) for row in sub]
            input_ids_cpu, _ = _pad_token_batch(sub_tokens, pad_id)
            loss_mask = _build_loss_mask(sub_tokens, sub_starts)
            attn_cpu = (input_ids_cpu != pad_id).long()
            input_ids_t = input_ids_cpu.to(device=tdev, dtype=torch.long, non_blocking=True)
            attn_t = attn_cpu.to(device=tdev, dtype=torch.long, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(input_ids_t, attention_mask=attn_t).logits.to(
                    device=sdev, dtype=_dt, non_blocking=True
                )
            input_ids_s = input_ids_cpu.to(device=sdev, dtype=torch.long, non_blocking=True)
            attn_s = attn_cpu.to(device=sdev, dtype=torch.long, non_blocking=True)
            s_logits = student(input_ids_s, attention_mask=attn_s).logits
            loss_mask_dev = loss_mask.to(device=sdev, dtype=torch.float32)
            loss_c = kl_loss_masked_positions(s_logits, t_logits, loss_mask_dev)
            (loss_c * (n_sub / n_tokens)).backward()
            total_loss += loss_c.item() * (n_sub / n_tokens)
            del t_logits, s_logits, loss_c
        n = n_tokens

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        global_step += 1

        elapsed = time.time() - t0
        avg_kl = float(total_loss)
        lr = scheduler.get_last_lr()[0]
        log.info(
            f"Step {global_step} | KL: {avg_kl:.4f} | LR: {lr:.2e} | "
            f"{elapsed:.1f}s ({n/max(elapsed, 1e-6):.1f} samp/s) | pos: {data.position:,}"
        )

        if not args.no_wandb:
            import wandb

            wandb.log(
                {"train/kl": avg_kl, "train/lr": lr, "perf/step_time": elapsed},
                step=global_step,
            )

        if args.eval_every_steps > 0 and (
            global_step == 1 or global_step % args.eval_every_steps == 0
        ):
            eval_t0 = time.time()
            eval_scores, king_scores, ttest = _evaluate_against_king(
                student=student,
                king_eval=king_eval,
                eval_cache=eval_cache,
            )
            eval_stats = _summary_stats(eval_scores)
            king_stats = _summary_stats(king_scores)
            lr_scale = lr / max(args.lr, 1e-12)
            gate_delta_ema = _update_ema(
                gate_delta_ema, float(ttest["delta"]), float(args.gate_ema_beta)
            )
            gate_promotable = (
                gate_delta_ema >= float(args.gate_min_delta)
                and float(ttest["p"]) <= float(args.gate_max_p)
            )
            gate_streak = gate_streak + 1 if gate_promotable else 0
            gate_is_stable = gate_streak >= int(args.gate_patience_evals)
            gate_score = _gate_score(gate_delta_ema, float(ttest["p"]))
            entry = {
                "step": int(global_step),
                "lr_scale": float(lr_scale),
                "train_loss": float(avg_kl),
                "eval_stats": eval_stats,
                "king_stats": king_stats,
                "ttest": ttest,
                "gate": {
                    "delta_ema": float(gate_delta_ema),
                    "promotable": bool(gate_promotable),
                    "streak": int(gate_streak),
                    "is_stable": bool(gate_is_stable),
                    "score": float(gate_score),
                },
                "time_s": float(time.time() - eval_t0),
                "ts": _utc_now_iso(),
            }
            print(json.dumps(entry, ensure_ascii=True))
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

            beats_king = (
                eval_stats["mean"] < king_stats["mean"]
                and ttest["delta"] >= args.beat_king_min_delta
                and ttest["p"] <= args.beat_king_p_threshold
            )
            if beats_king:
                beat_streak += 1
                if ttest["delta"] > best_beating_delta:
                    best_beating_delta = ttest["delta"]
                    best_dir = output_dir / "best_beat_king"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    # Restore full origin config/tokenizer files over save_pretrained output when cached.
                    _copy_cached_origin_configs(origin_cfg_cache_dir, best_dir)
                    _atomic_write_json(
                        best_dir / "train_state.json",
                        {
                            "global_step": global_step,
                            "data_position": data.position,
                            "best_beating_delta": best_beating_delta,
                            "ttest": ttest,
                            "eval_stats": eval_stats,
                            "king_stats": king_stats,
                            "saved_at": _utc_now_iso(),
                        },
                    )
                    log.info(
                        "New best king-beating checkpoint saved at %s (delta=%.6f, p=%.6g)",
                        best_dir,
                        ttest["delta"],
                        ttest["p"],
                    )
            else:
                beat_streak = 0

            if gate_is_stable and gate_score > best_gate_score:
                best_gate_score = gate_score
                best_gate_step = int(global_step)
                stable_dir = output_dir / "best_stable_vs_king"
                stable_dir.mkdir(parents=True, exist_ok=True)
                student.save_pretrained(stable_dir)
                tokenizer.save_pretrained(stable_dir)
                _copy_cached_origin_configs(origin_cfg_cache_dir, stable_dir)
                _atomic_write_json(
                    stable_dir / "train_state.json",
                    {
                        "global_step": global_step,
                        "data_position": data.position,
                        "gate": {
                            "delta": float(ttest["delta"]),
                            "p": float(ttest["p"]),
                            "delta_ema": float(gate_delta_ema),
                            "streak": int(gate_streak),
                            "min_delta": float(args.gate_min_delta),
                            "max_p": float(args.gate_max_p),
                            "patience_evals": int(args.gate_patience_evals),
                            "ema_beta": float(args.gate_ema_beta),
                            "score": float(gate_score),
                        },
                        "eval_stats": eval_stats,
                        "king_stats": king_stats,
                        "saved_at": _utc_now_iso(),
                    },
                )
                log.info(
                    "New best stable-vs-king checkpoint saved at %s (score=%.6f, delta_ema=%.6f, p=%.6g)",
                    stable_dir,
                    gate_score,
                    gate_delta_ema,
                    ttest["p"],
                )

            if not args.no_wandb:
                import wandb

                wandb.log(
                    {
                        "eval/mean": eval_stats["mean"],
                        "eval/std": eval_stats["std"],
                        "eval/p50": eval_stats["p50"],
                        "king/mean": king_stats["mean"],
                        "king/std": king_stats["std"],
                        "ttest/p": ttest["p"],
                        "ttest/t": ttest["t"],
                        "ttest/delta": ttest["delta"],
                        "ttest/beats_king": int(beats_king),
                        "train/beat_streak": beat_streak,
                        "gate/delta_ema": gate_delta_ema,
                        "gate/promotable": int(gate_promotable),
                        "gate/streak": gate_streak,
                        "gate/is_stable": int(gate_is_stable),
                        "gate/score": gate_score,
                    },
                    step=global_step,
                )

            if args.early_stop_on_beat_consecutive > 0 and beat_streak >= args.early_stop_on_beat_consecutive:
                log.info(
                    "Early stop: beat-king criteria met for %s consecutive evals.",
                    beat_streak,
                )
                stop_reason = (
                    f"beat-king early stop (streak={beat_streak}, "
                    f"p<={args.beat_king_p_threshold}, delta>={args.beat_king_min_delta})"
                )
                break

        if global_step % args.save_every == 0:
            d = output_dir / f"step_{global_step}"
            d.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(d)
            tokenizer.save_pretrained(d)
            # Overwrite save_pretrained config.json with full origin Hub/local snapshot when cached.
            _copy_cached_origin_configs(origin_cfg_cache_dir, d)
            torch.save(optimizer.state_dict(), d / "optimizer.pt")
            torch.save(scheduler.state_dict(), d / "scheduler.pt")
            _atomic_write_json(
                d / "train_state.json",
                {
                    "global_step": global_step,
                    "data_position": data.position,
                    "data_state": data.state_dict(),
                    "best_beating_delta": best_beating_delta,
                    "beat_streak": beat_streak,
                    "gate_delta_ema": gate_delta_ema,
                    "gate_streak": gate_streak,
                    "best_gate_score": best_gate_score,
                    "best_gate_step": best_gate_step,
                },
            )
            log.info(f"  Saved: {d}")

        if global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    log.info(f"Done. Step {global_step} (stopped by {stop_reason})")
    if best_gate_step > 0:
        log.info(
            "Best stable-vs-king checkpoint selected at step %s (score=%.6f).",
            best_gate_step,
            best_gate_score,
        )
    if not args.no_wandb:
        import wandb

        wandb.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="KL distillation with online teacher + streaming dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval_cache = subparsers.add_parser(
        "build_eval_cache_api",
        help="Build teacher eval cache once from live eval_data API prompts.",
    )
    p_eval_cache.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    p_eval_cache.add_argument("--teacher_gpu", type=int, default=0)
    p_eval_cache.add_argument(
        "--teacher_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to teacher model (starting at --teacher_gpu).",
    )
    p_eval_cache.add_argument(
        "--eval_data_url",
        type=str,
        default=DEFAULT_EVAL_DATA_URL,
        help="URL for validator eval_data JSON bundle.",
    )
    p_eval_cache.add_argument(
        "--eval_prompts",
        type=int,
        default=0,
        help="Max prompts to cache (0 = all prompts from eval_data API).",
    )
    p_eval_cache.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=8192,
        help="Fallback if eval_data has no max_new_tokens.",
    )
    p_eval_cache.add_argument(
        "--eval_seed",
        type=int,
        default=12345,
        help="Fallback if eval_data has no block_seed.",
    )
    p_eval_cache.add_argument(
        "--eval_cache_path",
        type=str,
        default="./eval_cache_api.pt",
        help="Output path for serialized eval cache payload.",
    )
    p_eval_cache.add_argument(
        "--rebuild_eval_cache",
        action="store_true",
        help="Overwrite eval cache if --eval_cache_path already exists.",
    )

    p_train = subparsers.add_parser(
        "train", help="Train with online teacher from streaming dataset."
    )
    p_train.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    p_train.add_argument("--student", type=str, default=STUDENT_MODEL)
    p_train.add_argument(
        "--student_revision",
        type=str,
        default=None,
        help="Optional Hub revision/branch for --student when downloading origin config/tokenizer files.",
    )
    p_train.add_argument("--teacher_gpu", type=int, default=0)
    p_train.add_argument("--student_gpu", type=int, default=1)
    p_train.add_argument(
        "--teacher_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to teacher model (starting at --teacher_gpu).",
    )
    p_train.add_argument(
        "--student_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to student model (starting at --student_gpu).",
    )
    p_train.add_argument(
        "--king_gpu",
        type=int,
        default=None,
        help="Start GPU index for king model (defaults to --teacher_gpu).",
    )
    p_train.add_argument(
        "--king_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to king model (starting at --king_gpu).",
    )
    p_train.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Streaming dataset for training.",
    )
    p_train.add_argument(
        "--min_chars",
        type=int,
        default=MIN_CHARS,
        help="Minimum text length filter for streaming training data.",
    )
    p_train.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Streaming fallback only: Hugging Face ``datasets`` split name (e.g. train, validation).",
    )
    p_train.add_argument(
        "--dataset_skip_rows",
        type=int,
        default=0,
        help=(
            "Streaming fallback only: skip this many raw examples from the start of the split "
            "before training (uses IterableDataset.skip when available). Ignored when resuming from a checkpoint."
        ),
    )
    p_train.add_argument("--lr", type=float, default=LR)
    p_train.add_argument("--warmup_steps", type=int, default=WARMUP)
    p_train.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p_train.add_argument("--samples_per_step", type=int, default=SAMPLES_PER_STEP)
    p_train.add_argument(
        "--online_chunk_size",
        type=int,
        default=1,
        help=(
            "Online teacher only: max sequences per teacher+student forward (memory knob). "
            "1 matches the old per-sample memory; increase (e.g. 2–4) if GPU 1 has headroom for "
            "full logits [chunk,L,V] during backward."
        ),
    )
    p_train.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p_train.add_argument("--kl_start_pos", type=int, default=KL_START_POS)
    p_train.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Training steps (must be > 0 in online mode).",
    )
    p_train.add_argument("--output_dir", type=str, default="./distil-checkpoints")
    p_train.add_argument("--save_every", type=int, default=SAVE_EVERY)
    p_train.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint dir to resume from (e.g. ./distil-checkpoints/step_1400).",
    )
    p_train.add_argument(
        "--resume_latest",
        action="store_true",
        help="Resume from latest step_* checkpoint in --output_dir.",
    )
    p_train.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="If > 0, run periodic eval-vs-king and emit JSON metric lines.",
    )
    p_train.add_argument(
        "--eval_prompts",
        type=int,
        default=500,
        help="Number of prompts used for periodic king comparison.",
    )
    p_train.add_argument(
        "--eval_dataset",
        type=str,
        default=DATASET,
        help="Dataset used to sample eval prompts for king comparison.",
    )
    p_train.add_argument(
        "--eval_block_number",
        type=int,
        default=12345,
        help="Deterministic dataset block for prompt sampling.",
    )
    p_train.add_argument(
        "--eval_block_hash",
        type=str,
        default=None,
        help=(
            "Optional real chain block hash used by eval.dataset shard selection when "
            "sampling dataset prompts (better validator parity than block number alone)."
        ),
    )
    p_train.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=512,
        help="Teacher continuation length for periodic evaluation.",
    )
    p_train.add_argument(
        "--eval_seed",
        type=int,
        default=12345,
        help="Seed for deterministic teacher continuations used in periodic eval.",
    )
    p_train.add_argument(
        "--eval_use_api_prompts",
        action="store_true",
        default=True,
        help="Use prompts from live eval_data API (instead of local dataset sampling).",
    )
    p_train.add_argument(
        "--eval_use_dataset_prompts",
        action="store_false",
        dest="eval_use_api_prompts",
        help="Use local dataset/block sampling instead of eval_data API prompts.",
    )
    p_train.add_argument(
        "--eval_data_url",
        type=str,
        default=DEFAULT_EVAL_DATA_URL,
        help="URL for validator eval_data JSON bundle (used when --eval_use_api_prompts).",
    )
    p_train.add_argument(
        "--eval_cache_path",
        type=str,
        default=None,
        help="Path to eval cache payload (.pt). If present, reuse without teacher generation.",
    )
    p_train.add_argument(
        "--rebuild_eval_cache",
        action="store_true",
        help="Force rebuild eval cache even when --eval_cache_path exists.",
    )
    p_train.add_argument(
        "--king_repo",
        type=str,
        default=None,
        help="King model repo for periodic eval-vs-king metrics.",
    )
    p_train.add_argument(
        "--king_revision",
        type=str,
        default=None,
        help="Optional king model revision.",
    )
    p_train.add_argument(
        "--beat_king_p_threshold",
        type=float,
        default=0.05,
        help="Beat-king criterion: require ttest.p <= this value.",
    )
    p_train.add_argument(
        "--beat_king_min_delta",
        type=float,
        default=0.0,
        help="Beat-king criterion: require ttest.delta >= this value.",
    )
    p_train.add_argument(
        "--early_stop_on_beat_consecutive",
        type=int,
        default=0,
        help="If > 0, stop when beat-king criteria hold this many evals in a row.",
    )
    p_train.add_argument(
        "--metrics_jsonl",
        type=str,
        default=None,
        help="Path to append eval-vs-king JSON lines (default: output_dir/king_eval_metrics.jsonl).",
    )
    p_train.add_argument(
        "--gate_min_delta",
        type=float,
        default=0.0025,
        help="Stable-gate minimum EMA delta (king - student) required for promotion.",
    )
    p_train.add_argument(
        "--gate_max_p",
        type=float,
        default=0.03,
        help="Stable-gate maximum one-sided p-value required for promotion.",
    )
    p_train.add_argument(
        "--gate_patience_evals",
        type=int,
        default=3,
        help="Stable-gate required consecutive promotable evals before checkpoint promotion.",
    )
    p_train.add_argument(
        "--gate_ema_beta",
        type=float,
        default=0.6,
        help="EMA smoothing factor for stable-gate delta.",
    )
    p_train.add_argument("--wandb_project", type=str, default="distil-subnet97")
    p_train.add_argument("--wandb_run", type=str, default=None)
    p_train.add_argument(
        "--wandb_init_timeout",
        type=int,
        default=300,
        help="Seconds to wait for wandb.run init (slow networks / api.wandb.ai).",
    )
    p_train.add_argument("--no_wandb", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "build_eval_cache_api":
        build_eval_cache_api(args)
    elif args.command == "train":
        train_online(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
