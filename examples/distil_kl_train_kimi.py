#!/usr/bin/env python3
"""
KL Distillation Training for Bittensor Subnet 97.

This script keeps the same core training logic as examples/distil_kl_train.py.
It now exposes only:
1) build_eval_cache_api: precompute deterministic eval prompts/targets once
2) train: online-teacher KL training from streaming data

Examples:
    # Build eval cache once from validator API prompts (Kimi teacher + Qwen student tokenizer)
    python examples/distil_kl_train_kimi.py build_eval_cache_api \
      --teacher moonshotai/Kimi-K2.6 \
      --student Qwen/Qwen3.6-4B \
      --eval_cache_path ./eval_cache_api.pt

    # Train repeatedly (online teacher mode)
    python examples/distil_kl_train_kimi.py train \
      --teacher moonshotai/Kimi-K2.6 \
      --student Qwen/Qwen3.6-4B \
      --sequential_gpu_pipeline \
      --teacher_gpu 0 --teacher_gpu_count 8 \
      --student_gpu 0 --student_gpu_count 1 \
      --output_dir ./distil-checkpoints

    # Moonshot teacher + smaller Moonshot student (same tokenizer as Kimi; no Qwen bridge)
    python examples/distil_kl_train_kimi.py train \
      --assume_same_tokenizer_as_teacher \
      --teacher moonshotai/Kimi-K2.6 \
      --student moonshotai/<SmallerMoonshot-Instruct> \
      --teacher_gpu 0 --teacher_gpu_count 8 \
      --student_gpu 0 --student_gpu_count 1 \
      --output_dir ./distil-checkpoints-moonshot
"""

import argparse
import gc
import inspect
import shutil
import json
import logging
import os
import random
import re
import time
import math
import statistics
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW


def _patch_transformers_hub_remote_imports() -> None:
    """Transformers v5 removed ``is_torch_fx_available``; Hub DeepSeek/Moonlight code still imports it."""
    import transformers.utils.import_utils as _import_utils

    if not hasattr(_import_utils, "is_torch_fx_available"):

        def is_torch_fx_available() -> bool:
            return True

        _import_utils.is_torch_fx_available = is_torch_fx_available  # type: ignore[attr-defined]


_PRETRAINED_TIE_WEIGHTS_ORIG = None


def _hub_compat_dispatch_tie_weights(model, missing_keys=None, recompute_mapping=True, **kwargs):
    """
    Call the defining class's ``tie_weights`` with only kwargs it accepts (Hub Kimi defines ``tie_weights(self)``).
    Must not replace ``PreTrainedModel.tie_weights``: subclasses resolve ``self.tie_weights`` to their own method,
    so Transformers' ``init_weights`` / ``_finalize_model_loading`` are patched to call this instead.
    """
    import transformers.modeling_utils as _modeling_utils

    ptm = _modeling_utils.PreTrainedModel
    orig = _PRETRAINED_TIE_WEIGHTS_ORIG
    assert orig is not None
    for cls in type(model).__mro__:
        if cls is ptm:
            return orig(
                model,
                missing_keys=missing_keys,
                recompute_mapping=recompute_mapping,
                **kwargs,
            )
        if "tie_weights" not in cls.__dict__:
            continue
        fn = cls.__dict__["tie_weights"]
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return fn(
                model,
                missing_keys=missing_keys,
                recompute_mapping=recompute_mapping,
                **kwargs,
            )
        call_kw: dict = {}
        if "missing_keys" in params:
            call_kw["missing_keys"] = missing_keys
        if "recompute_mapping" in params:
            call_kw["recompute_mapping"] = recompute_mapping
        for key, val in kwargs.items():
            if key in params:
                call_kw[key] = val
        return fn(model, **call_kw)

    return orig(
        model,
        missing_keys=missing_keys,
        recompute_mapping=recompute_mapping,
        **kwargs,
    )


def _patch_transformers_hub_tie_weights_compat() -> None:
    """
    Transformers v5 calls ``tie_weights(recompute_mapping=...)`` from ``init_weights`` and
    ``tie_weights(missing_keys=..., recompute_mapping=...)`` from ``_finalize_model_loading``.
    Hub Kimi overrides ``tie_weights`` with a no-arg signature; patching ``PreTrainedModel.tie_weights`` does not
    intercept ``self.tie_weights`` on instances, so we patch those call sites and use ``_hub_compat_dispatch_tie_weights``.
    """
    import transformers.modeling_utils as _modeling_utils

    global _PRETRAINED_TIE_WEIGHTS_ORIG

    ptm = _modeling_utils.PreTrainedModel
    if getattr(ptm, "_distil_kl_hub_tie_dispatch_patched", False):
        return

    # Transformers v4.x: ``init_weights`` calls ``self.tie_weights()`` with no kwargs and there is no
    # ``_finalize_model_loading`` hook. Hub Kimi ``tie_weights(self)`` is compatible; skip patching so we do not
    # replace stock ``init_weights`` (which would drop ``prune_heads`` / ``_init_weights`` guards).
    finalize_raw = ptm.__dict__.get("_finalize_model_loading")
    if finalize_raw is None:
        ptm._distil_kl_hub_tie_dispatch_patched = True  # type: ignore[attr-defined]
        return

    import types

    _PRETRAINED_TIE_WEIGHTS_ORIG = ptm.__dict__["tie_weights"]

    def init_weights(self) -> None:
        if _modeling_utils.get_torch_context_manager_or_global_device() != torch.device("meta"):
            self.initialize_weights()
        _hub_compat_dispatch_tie_weights(self, recompute_mapping=False)

    _orig_finalize = finalize_raw.__func__ if hasattr(finalize_raw, "__func__") else finalize_raw

    def _finalize_model_loading(model, load_config, loading_info):
        def _instance_tie_weights(self, missing_keys=None, recompute_mapping=False, **kw):
            return _hub_compat_dispatch_tie_weights(
                self,
                missing_keys=missing_keys,
                recompute_mapping=recompute_mapping,
                **kw,
            )

        model.tie_weights = types.MethodType(_instance_tie_weights, model)
        try:
            return _orig_finalize(model, load_config, loading_info)
        finally:
            if hasattr(model, "tie_weights"):
                delattr(model, "tie_weights")

    ptm.init_weights = init_weights  # type: ignore[method-assign]
    ptm._finalize_model_loading = staticmethod(_finalize_model_loading)  # type: ignore[method-assign]
    ptm._distil_kl_hub_tie_dispatch_patched = True  # type: ignore[attr-defined]


_patch_transformers_hub_remote_imports()
_patch_transformers_hub_tie_weights_compat()


def _normalize_deepseek_hub_rope_config(config) -> None:
    """
    Transformers v5 normalizes RoPE into ``rope_parameters`` (often ``{}`` or ``rope_type`` only).
    Moonlight / Hub ``modeling_deepseek`` still uses ``config.rope_scaling['type']`` and treats any
    non-``None`` dict as an active scaling spec.
    """
    if config is None or getattr(config, "model_type", None) != "deepseek_v3":
        return
    rp = getattr(config, "rope_parameters", None)
    if rp is None:
        return
    if not isinstance(rp, dict):
        return
    if not rp:
        config.rope_parameters = None
        return
    rope_type = rp.get("rope_type") or rp.get("type")
    if rope_type in (None, "default"):
        config.rope_parameters = None
        return
    if "type" not in rp and "rope_type" in rp:
        rp["type"] = rp["rope_type"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Defaults: Kimi K2.6 teacher with Qwen3.6-compatible student vocab (subnet SN97 rollout).
TEACHER_MODEL = "moonshotai/Kimi-K2.6"
STUDENT_MODEL = "Qwen/Qwen3.6-4B"
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

# moonshotai/Kimi-K2.6: Hub vision tower rejects Flash Attention 2; Transformers may enable FA2 by default.
TEACHER_ATTN_IMPLEMENTATION = "eager"


def _patch_kimi_hub_vision_attn(config) -> None:
    """
    KimiK25VisionConfig defaults ``_attn_implementation`` to ``flash_attention_2``; that value is copied
    into ``VisionTowerConfig`` and breaks ``MoonViT3dPretrainedModel`` init. Top-level ``attn_implementation=``
    on ``from_pretrained`` does not override this nested field.
    """
    if config is None or getattr(config, "model_type", None) != "kimi_k25":
        return
    vc = getattr(config, "vision_config", None)
    if vc is not None and hasattr(vc, "_attn_implementation"):
        vc._attn_implementation = TEACHER_ATTN_IMPLEMENTATION


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


def _purge_cuda_model_hold(reason: str, model: torch.nn.Module | None) -> None:
    """Drop refs to a CUDA-resident HF model shard and try to reclaim VRAM."""
    if model is None:
        return
    log.info("Releasing CUDA memory (%s)...", reason)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _instantiate_causal_lm(
    model_ref: str,
    load_kwargs: dict,
    *,
    train: bool,
    freeze_all: bool,
) -> torch.nn.Module:
    """Load HF causal LM once (online train / sequential swap). Caller owns lifetime."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kw = dict(load_kwargs)
    if kw.get("trust_remote_code"):
        cfg = kw.get("config")
        if cfg is None:
            cfg_kw: dict = {"trust_remote_code": True}
            rev = kw.get("revision")
            if rev is not None:
                cfg_kw["revision"] = rev
            cfg = AutoConfig.from_pretrained(model_ref, **cfg_kw)
        _normalize_deepseek_hub_rope_config(cfg)
        _patch_kimi_hub_vision_attn(cfg)
        kw["config"] = cfg

    lm = AutoModelForCausalLM.from_pretrained(model_ref, **kw)
    if freeze_all:
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)
    elif train:
        lm.train()
    else:
        lm.eval()
    return lm


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


def _dual_training_full_strings(
    item: dict, teacher_tokenizer, student_tokenizer
) -> tuple[str, str, int, int] | None:
    """
    Build canonical full training strings rendered with teacher vs student chat templates.

    When ``messages`` is present, prompts differ by tokenizer templates; completions share
    assistant content but occupy different substring ranges.

    Returns (full_teacher, full_student, prompt_char_len_teacher, prompt_char_len_student).
    For plain qa / transcript splits both strings coincide and prompt lengths match.
    """
    messages = item.get("messages") or item.get("conversations")
    if isinstance(messages, list) and messages:
        normalized = _normalize_chat_messages(messages)
        if len(normalized) >= 2 and normalized[-1]["role"] == "assistant":
            full_teacher = _render_chat_messages(
                teacher_tokenizer, normalized, add_generation_prompt=False
            )
            full_student = _render_chat_messages(
                student_tokenizer, normalized, add_generation_prompt=False
            )
            prompt_teacher = _render_chat_messages(
                teacher_tokenizer, normalized[:-1], add_generation_prompt=True
            )
            prompt_student = _render_chat_messages(
                student_tokenizer, normalized[:-1], add_generation_prompt=True
            )
            if (
                full_teacher.startswith(prompt_teacher)
                and full_student.startswith(prompt_student)
                and (
                    len(full_teacher[len(prompt_teacher) :].strip()) > 0
                    or len(full_student[len(prompt_student) :].strip()) > 0
                )
            ):
                return (
                    full_teacher,
                    full_student,
                    len(prompt_teacher),
                    len(prompt_student),
                )

    prompt_compl = _extract_prompt_completion_text(item, teacher_tokenizer)
    if prompt_compl is not None:
        prompt_text, completion_text = prompt_compl
        text = prompt_text + completion_text
        plc = len(prompt_text)
        return text, text, plc, plc

    return None


def _encode_with_offsets_optional(
    tokenizer,
    full_text: str,
    encode_kwargs_no_offsets: dict,
):
    """Return (input_ids 1d int tensor, offsets list[tuple[int,int]] or None)."""
    try:
        enc = tokenizer(
            full_text, return_tensors="pt", return_offsets_mapping=True, **encode_kwargs_no_offsets
        )
        ids = enc.input_ids.squeeze(0)
        offs = enc["offset_mapping"][0].tolist()
        return ids.to(torch.long), offs
    except (TypeError, KeyError, Exception):
        enc = tokenizer(full_text, return_tensors="pt", **encode_kwargs_no_offsets)
        return enc.input_ids.squeeze(0).to(torch.long), None


def _loss_start_token_from_prompt_chars_prompt_ids(
    full_text: str,
    offsets: list | None,
    prompt_char_len: int,
    tokenizer,
    encode_kwargs_simple: dict,
    seq_length: int,
) -> int:
    """Mirror StreamingTokenStream loss_start logic for a canonical full string."""
    if offsets:
        pc = max(0, int(prompt_char_len))
        loss_start = None
        for tok_idx, pair in enumerate(offsets):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            cs, ce = int(pair[0]), int(pair[1])
            if ce <= cs:
                continue
            if cs >= pc:
                loss_start = max(0, tok_idx - 1)
                break
        if loss_start is not None:
            return int(loss_start)
    prompt_chars = max(0, int(prompt_char_len))
    prefix = full_text[:prompt_chars]
    p_ids = tokenizer(prefix, **encode_kwargs_simple).input_ids.squeeze(0)
    return max(0, int(p_ids.shape[0]) - 1)


def _teacher_logits_index_for_student_logit(
    student_offsets: list, teacher_offsets: list, si: int, teach_len: int
) -> int:
    """Map student timestep ``si`` (logits predicting token si+1) to a causal teacher timestep."""
    if si < 0 or si >= len(student_offsets) or teach_len < 2:
        return -1
    pair = student_offsets[si]
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        return -1
    end_char = int(pair[1])
    if end_char <= 0:
        return -1 if teach_len < 2 else 0
    chosen = -1
    if teacher_offsets:
        for k in range(min(len(teacher_offsets), teach_len)):
            tk = teacher_offsets[k]
            if not isinstance(tk, (tuple, list)) or len(tk) != 2:
                continue
            tcs, tce = int(tk[0]), int(tk[1])
            if tcs < end_char:
                chosen = k
        te = chosen
        if teach_len >= 2:
            te = min(max(te, 0), teach_len - 2)
        return te
    return max(0, min(teach_len // 2, teach_len - 2))


def _build_teacher_pos_map(
    student_offsets: list | None,
    teacher_offsets: list | None,
    stud_len: int,
    teach_len: int,
    stud_loss_start: int,
    device,
) -> torch.Tensor:
    """Per student position: teacher timestep for logits supervision, else -1."""
    out = torch.full((stud_len,), -1, device=device, dtype=torch.long)
    if student_offsets is None or teacher_offsets is None:
        return out
    for si in range(stud_loss_start, stud_len - 1):
        ti = _teacher_logits_index_for_student_logit(
            student_offsets, teacher_offsets, si, teach_len
        )
        out[si] = ti if ti >= 0 else -1
    return out


def _fallback_teacher_pos_indices(
    stud_len: int,
    teach_len: int,
    stud_loss_start: int,
    device,
) -> torch.Tensor:
    """Length-ratio alignment when offset_mapping is unavailable (plain-text rows)."""
    out = torch.full((stud_len,), -1, device=device, dtype=torch.long)
    if teach_len < 2 or stud_len < 2:
        return out
    for si in range(int(stud_loss_start), stud_len - 1):
        frac = float(si + 1) / float(max(stud_len - 1, 1))
        ti = int(round(frac * float(max(teach_len - 1, 1)))) - 1
        ti = min(max(ti, 0), teach_len - 2)
        out[si] = ti
    return out


def _teacher_pos_indices_cpu_dual(
    row: dict, teacher_len: int, student_len: int, student_loss_start: int
) -> torch.Tensor:
    s_off = row.get("student_offsets")
    t_off = row.get("teacher_offsets")
    ssl = max(0, min(int(student_loss_start), student_len - 2))
    pmap = _build_teacher_pos_map(
        s_off, t_off, student_len, teacher_len, ssl, torch.device("cpu")
    )
    if int((pmap[ssl:] >= 0).sum()) > 0:
        return pmap
    return _fallback_teacher_pos_indices(student_len, teacher_len, ssl, torch.device("cpu"))


def _cross_vocab_kl_masked_piecewise(
    student_logits_flat: torch.Tensor,
    teacher_logits_flat: torch.Tensor,
    teacher_tokenizer,
    student_tokenizer,
    top_k: int,
    *,
    teacher_is_log_probs: bool = False,
) -> torch.Tensor:
    """
    Differentiable forward-KL proxy: KL( softmax(teacher logits) piecewise-aligned || student ), ignoring
    the constant teacher-entropy term. Uses top-K masses on teacher side and aligns each Kimi/id piece
    to the first student subtoken (same heuristic as ``eval.cross_tokenizer.align_logprobs_kimi_to_qwen``).

    Args:
        student_logits_flat: [N, Vs]
        teacher_logits_flat: [N, Vt] raw logits unless ``teacher_is_log_probs``.
    Returns:
        Scalar mean over positions (N averages).
    """
    if student_logits_flat.numel() == 0:
        return student_logits_flat.new_zeros(())
    vs = student_logits_flat.shape[-1]
    vt = teacher_logits_flat.shape[-1]
    if vs == vt:
        if teacher_is_log_probs:
            t_lp = teacher_logits_flat.float()
            s_lp = F.log_softmax(student_logits_flat.float(), dim=-1)
            t_p = t_lp.exp().clamp(min=1e-30)
        else:
            t_lp = F.log_softmax(teacher_logits_flat.float(), dim=-1)
            s_lp = F.log_softmax(student_logits_flat.float(), dim=-1)
            t_p = t_lp.exp()
        return (t_p * (t_lp - s_lp)).sum(dim=-1).mean()
    kk = max(1, min(int(top_k), int(vt)))
    n = student_logits_flat.shape[0]
    device = student_logits_flat.device
    acc = torch.zeros((), device=device, dtype=torch.float32)
    s_lp_all = F.log_softmax(student_logits_flat.float(), dim=-1)
    t_tensor = teacher_logits_flat.float()
    vals, top_idx = torch.topk(t_tensor, k=kk, dim=-1)
    probs = torch.softmax(vals, dim=-1)

    tops = top_idx.detach().cpu().tolist()
    # Row-wise decode top teacher ids → map to student token ids → gather log probs.
    for row in range(n):
        pr_row = probs[row].tolist()
        idx_row = tops[row]
        row_acc = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)
        for j in range(min(len(idx_row), len(pr_row))):
            mass = float(pr_row[j])
            if mass < 1e-12:
                continue
            tid = int(idx_row[j])
            try:
                piece = teacher_tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:
                continue
            if not piece:
                continue
            try:
                s_ids = student_tokenizer(
                    piece, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                if hasattr(s_ids, "squeeze"):
                    s_ids_list = [int(x) for x in s_ids.squeeze().tolist()]
                elif isinstance(s_ids, list):
                    s_ids_list = [int(x) for x in s_ids]
                else:
                    s_ids_list = [int(x) for x in s_ids.flatten().tolist()]
            except Exception:
                continue
            if not s_ids_list:
                continue
            sid = int(s_ids_list[0])
            if sid < 0 or sid >= vs:
                continue
            row_acc = row_acc + mass * (-s_lp_all[row, sid].float())
            denom = denom + mass
        if denom.item() > 1e-8:
            acc = acc + (row_acc / denom)
    return acc / max(float(n), 1e-8)


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
    teacher_tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
    *,
    cross_tokenizer: bool = False,
    student_tokenizer=None,
    max_student_seq: int | None = None,
) -> list[dict]:
    teacher_eval.eval()
    teacher_device = next(teacher_eval.parameters()).device
    cache = []
    smax = int(max_student_seq) if max_student_seq is not None else 65536
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prompt_ids = (
                teacher_tokenizer(prompt, return_tensors="pt", truncation=False)
                .input_ids.to(teacher_device)
            )
            prompt_len = prompt_ids.shape[1]
            prompt_seed = int(seed) + i
            # Do not pass ``generator=`` into ``generate``: strict ``model_kwargs`` validation
            # (e.g. Hub Kimi / partial GenerationMixin stacks) rejects it as unused.
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
            entry: dict = {
                "full_seq": full_seq.cpu(),
                "prompt_len": int(prompt_len),
                "teacher_log_probs": t_log_p,
            }
            if cross_tokenizer and student_tokenizer is not None:
                row = full_seq[0].tolist()
                decoded = teacher_tokenizer.decode(row, skip_special_tokens=False)
                student_full = student_tokenizer(
                    decoded,
                    return_tensors="pt",
                    truncation=True,
                    max_length=smax,
                ).input_ids.cpu()
                prompt_ids_s = student_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=smax,
                ).input_ids
                entry["student_full_seq"] = student_full
                entry["prompt_len_student"] = int(prompt_ids_s.shape[1])
                entry["cross_tokenizer_kd"] = True
            cache.append(entry)
            del prompt_ids, full_seq, t_logits, t_cont, t_log_p
    return cache


class EvalDataNotAvailable(RuntimeError):
    """Public ``/api/eval-data`` returned HTTP 404 (validator has not published a bundle yet)."""


def _fetch_eval_data_bundle(eval_data_url: str) -> dict:
    req = urllib.request.Request(
        eval_data_url,
        headers={"User-Agent": "distil_kl_train_prebuilt/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise EvalDataNotAvailable(
                f"No eval_data JSON at {eval_data_url!r} (HTTP 404). "
                "The validator has not published a round to public state yet, or the URL is wrong."
            ) from e
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            pass
        raise RuntimeError(
            f"eval_data request failed: HTTP {e.code} {e.reason!r} for {eval_data_url!r}. {detail}"
        ) from e
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
    cached_student = str(cache_meta.get("student") or "").strip()
    current_student = str(getattr(args, "student", "") or "").strip()
    if cache_meta.get("cross_tokenizer_kd"):
        if (
            cached_student
            and current_student
            and cached_student != current_student
        ):
            raise ValueError(
                "Eval cache student mismatch under cross-tokenizer KD: "
                f"cache={cached_student!r} current={current_student!r}. "
                "Use --rebuild_eval_cache."
            )
    if "assume_same_tokenizer_as_teacher" in cache_meta:
        if bool(cache_meta.get("assume_same_tokenizer_as_teacher")) != bool(
            getattr(args, "assume_same_tokenizer_as_teacher", False)
        ):
            raise ValueError(
                "Eval cache assume_same_tokenizer_as_teacher metadata mismatch vs current training flags. "
                "Use --rebuild_eval_cache."
            )
        if (
            bool(cache_meta.get("assume_same_tokenizer_as_teacher"))
            and bool(getattr(args, "assume_same_tokenizer_as_teacher", False))
            and cached_student
            and current_student
            and cached_student != current_student
        ):
            raise ValueError(
                "Eval cache student mismatch (shared tokenizer / Moonshot family): "
                f"cache={cached_student!r} current={current_student!r}. "
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
    teacher_tokenizer,
    student_tokenizer,
    kd_top_k: int,
) -> tuple[list[float], list[float], dict]:

    king_eval.eval()
    student.eval()
    king_device = next(king_eval.parameters()).device
    student_device = next(student.parameters()).device

    eval_scores = []
    king_scores = []
    with torch.no_grad():
        for item in eval_cache:
            t_log_p_cpu = item["teacher_log_probs"]
            t_log_p = t_log_p_cpu.to(student_device, dtype=torch.float32)
            fs_student = item.get("student_full_seq")
            if fs_student is None:
                fs_student = item["full_seq"]
                pl_student = int(item["prompt_len"])
            else:
                pl_student = int(item["prompt_len_student"])

            s_logits = student(fs_student.to(student_device)).logits.float()
            s_cont = s_logits[:, pl_student - 1 : -1, :]
            k_logits = king_eval(fs_student.to(king_device)).logits.float()
            k_cont = k_logits[:, pl_student - 1 : -1, :]

            n = min(s_cont.shape[1], k_cont.shape[1], t_log_p.shape[1])
            vt = t_log_p.shape[-1]
            vs = s_cont.shape[-1]
            vk = k_cont.shape[-1]

            if n <= 0:
                continue

            t_slice = t_log_p[:, :n, :].contiguous()

            if vt == vs:
                s_lp = F.log_softmax(s_cont[:, :n, :], dim=-1)
                s_kl = _mean_token_kl_forward_chunked(s_lp, t_slice)
            else:
                s_kl = float(
                    _cross_vocab_kl_masked_piecewise(
                        s_cont[0, :n, :],
                        t_slice[0, :n, :],
                        teacher_tokenizer,
                        student_tokenizer,
                        kd_top_k,
                        teacher_is_log_probs=True,
                    ).item()
                )

            del s_logits, s_cont
            if student_device.type == "cuda":
                torch.cuda.empty_cache()

            t_on_king = t_log_p_cpu.to(king_device, dtype=torch.float32)[:, :n, :].contiguous()
            if vt == vk:
                k_lp = F.log_softmax(k_cont[:, :n, :], dim=-1)
                k_kl = _mean_token_kl_forward_chunked(k_lp, t_on_king)
            else:
                k_kl = float(
                    _cross_vocab_kl_masked_piecewise(
                        k_cont[0, :n, :],
                        t_on_king[0, :n, :],
                        teacher_tokenizer,
                        student_tokenizer,
                        kd_top_k,
                        teacher_is_log_probs=True,
                    ).item()
                )

            eval_scores.append(float(s_kl))
            king_scores.append(float(k_kl))

            del k_logits, k_cont, t_slice, t_log_p, t_on_king

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
    """Streams dataset rows and builds separate teacher / student token streams."""

    def __init__(
        self,
        teacher_model: str,
        dataset: str = DATASET,
        max_seq_len: int = MAX_SEQ_LEN,
        min_chars: int = MIN_CHARS,
        dataset_split: str = "train",
        dataset_skip_rows: int = 0,
        student_model: str | None = None,
        assume_same_tokenizer_as_teacher: bool = False,
        tokenizer_model: str | None = None,
    ):
        from transformers import AutoTokenizer

        if min_chars < 0:
            raise ValueError(f"min_chars must be >= 0 (got {min_chars})")
        self._teacher_model = str(teacher_model)
        self._student_model = (
            str(student_model).strip() if student_model else self._teacher_model
        )
        self._assume_same_tokenizer_as_teacher = bool(assume_same_tokenizer_as_teacher)
        if tokenizer_model is not None:
            hm = str(tokenizer_model).strip()
            self._tokenizer_hub_ref = hm if hm else None
        else:
            self._tokenizer_hub_ref = None

        repos_differ = self._student_model != self._teacher_model
        self.dual_tokenizers = bool(repos_differ and not self._assume_same_tokenizer_as_teacher)
        tokenizer_hub = (
            (self._tokenizer_hub_ref if self._tokenizer_hub_ref else self._teacher_model)
            if (self.dual_tokenizers is False)
            else None
        )
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
        if self.dual_tokenizers:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                self._teacher_model, trust_remote_code=True
            )
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                self._student_model, trust_remote_code=True
            )
        else:
            src = tokenizer_hub if tokenizer_hub else self._teacher_model
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                src, trust_remote_code=True
            )
            self.student_tokenizer = self.teacher_tokenizer
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        if self.dual_tokenizers and self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        self.tokenizer = self.student_tokenizer
        if self._assume_same_tokenizer_as_teacher and repos_differ:
            src_h = tokenizer_hub if tokenizer_hub else self._teacher_model
            log.info(
                "Shared tokenizer: LM weights teacher=%s student=%s; tokenization/chat template from Hub %s",
                self._teacher_model,
                self._student_model,
                src_h,
            )

    def _tensorize_dual(self, item: dict) -> dict | None:
        truncate_kw = {"truncation": True, "max_length": self.max_seq_len}
        rt_kw = dict(truncate_kw)
        rt_kw["return_tensors"] = "pt"
        duo = _dual_training_full_strings(
            item, self.teacher_tokenizer, self.student_tokenizer
        )
        if duo is None:
            return None
        full_t, full_s, plc_t, plc_s = duo
        if not self.dual_tokenizers:
            text = full_t
            plc = plc_t
            ids, off = _encode_with_offsets_optional(
                self.teacher_tokenizer, text, truncate_kw
            )
            ek_simple_flat = rt_kw.copy()
            loss_s = _loss_start_token_from_prompt_chars_prompt_ids(
                text,
                off,
                plc,
                self.teacher_tokenizer,
                ek_simple_flat,
                int(ids.shape[0]),
            )
            ids_long = ids.to(dtype=torch.long)
            if ids_long.numel() <= loss_s + 1:
                return None
            return {
                "teacher_input_ids": ids_long.clone(),
                "student_input_ids": ids_long.clone(),
                "teacher_loss_start": int(loss_s),
                "student_loss_start": int(loss_s),
                "teacher_offsets": off,
                "student_offsets": off,
            }

        ids_t, off_t = _encode_with_offsets_optional(
            self.teacher_tokenizer, full_t, truncate_kw
        )
        ids_s, off_s = _encode_with_offsets_optional(
            self.student_tokenizer, full_s, truncate_kw
        )

        ek_simple_flat = rt_kw.copy()
        loss_t = _loss_start_token_from_prompt_chars_prompt_ids(
            full_t, off_t, plc_t, self.teacher_tokenizer, ek_simple_flat, int(ids_t.shape[0])
        )
        loss_s = _loss_start_token_from_prompt_chars_prompt_ids(
            full_s, off_s, plc_s, self.student_tokenizer, ek_simple_flat, int(ids_s.shape[0])
        )
        if ids_t.numel() <= loss_t + 1 or ids_s.numel() <= loss_s + 1:
            return None
        return {
            "teacher_input_ids": ids_t.to(dtype=torch.long),
            "student_input_ids": ids_s.to(dtype=torch.long),
            "teacher_loss_start": int(loss_t),
            "student_loss_start": int(loss_s),
            "teacher_offsets": off_t,
            "student_offsets": off_s,
        }

    def _tokenize_item(self, item: dict) -> dict | None:
        """Encode one dataset row for both models; offsets support cross-tokenizer logit alignment."""
        truncate_kw = {"truncation": True, "max_length": self.max_seq_len}
        rt_kw = dict(truncate_kw)
        rt_kw["return_tensors"] = "pt"

        row = self._tensorize_dual(item)
        if row is not None:
            return row

        text = _first_nonempty_text(item, ("text",))
        if not text or len(text) < self.min_chars:
            return None
        ids_t = self.teacher_tokenizer(text, **rt_kw).input_ids.squeeze(0).to(torch.long)
        ids_s = ids_t.clone() if not self.dual_tokenizers else self.student_tokenizer(
            text, **rt_kw
        ).input_ids.squeeze(0).to(torch.long)
        return {
            "teacher_input_ids": ids_t,
            "student_input_ids": ids_s,
            "teacher_loss_start": -1,
            "student_loss_start": -1,
            "teacher_offsets": None,
            "student_offsets": None,
        }

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
            out.append(sample)
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
            "student_model": self._student_model,
            "dual_tokenizers": bool(self.dual_tokenizers),
            "assume_same_tokenizer_as_teacher": bool(self._assume_same_tokenizer_as_teacher),
            "tokenizer_hub_ref": self._tokenizer_hub_ref or "",
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
        if (
            "student_model" in state
            and str(state.get("student_model")) != str(self._student_model)
        ):
            raise ValueError(
                f"Streaming resume student_model mismatch: checkpoint={state.get('student_model')} "
                f"current={self._student_model}"
            )
        if "assume_same_tokenizer_as_teacher" in state and bool(
            state.get("assume_same_tokenizer_as_teacher")
        ) != bool(self._assume_same_tokenizer_as_teacher):
            raise ValueError(
                "Streaming resume assume_same_tokenizer_as_teacher mismatch: "
                f"checkpoint={state.get('assume_same_tokenizer_as_teacher')} "
                f"current={self._assume_same_tokenizer_as_teacher}"
            )
        ckpt_tokhub = str(state.get("tokenizer_hub_ref") or "")
        cur_tokhub = str(self._tokenizer_hub_ref or "")
        if ckpt_tokhub != cur_tokhub:
            raise ValueError(
                f"Streaming resume tokenizer_hub_ref mismatch: checkpoint={ckpt_tokhub!r} "
                f"current={cur_tokhub!r}"
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
    from transformers import AutoTokenizer

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
        "attn_implementation": TEACHER_ATTN_IMPLEMENTATION,
    }
    if teacher_max_memory is not None:
        teacher_load_kwargs["max_memory"] = teacher_max_memory
    teacher_eval = _instantiate_causal_lm(
        args.teacher, teacher_load_kwargs, train=False, freeze_all=True
    )
    teacher_hf_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if teacher_hf_tok.pad_token is None:
        teacher_hf_tok.pad_token = teacher_hf_tok.eos_token
    student_ref = getattr(args, "student", None) or args.teacher
    assume_same = bool(getattr(args, "assume_same_tokenizer_as_teacher", False))
    repos_differ = str(student_ref).strip() != str(args.teacher).strip()
    cross = bool(repos_differ and not assume_same)
    student_hf_tok = None
    if cross:
        student_hf_tok = AutoTokenizer.from_pretrained(student_ref, trust_remote_code=True)
        if student_hf_tok.pad_token is None:
            student_hf_tok.pad_token = student_hf_tok.eos_token

    cache = _prepare_eval_cache(
        teacher_eval=teacher_eval,
        teacher_tokenizer=teacher_hf_tok,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        seed=eval_seed,
        cross_tokenizer=cross,
        student_tokenizer=student_hf_tok if cross else None,
        max_student_seq=int(args.max_student_seq),
    )
    meta = {
        "teacher": args.teacher,
        "student": str(student_ref),
        "assume_same_tokenizer_as_teacher": bool(assume_same),
        "cross_tokenizer_kd": bool(cross),
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
    if getattr(args, "tokenizer_model", None) and not getattr(
        args, "assume_same_tokenizer_as_teacher", False
    ):
        log.warning(
            "--tokenizer_model is ignored unless --assume_same_tokenizer_as_teacher is set."
        )
    data = StreamingTokenStream(
        teacher_model=args.teacher,
        dataset=args.dataset,
        max_seq_len=args.max_seq_len,
        min_chars=args.min_chars,
        dataset_split=args.dataset_split,
        dataset_skip_rows=stream_skip_rows,
        student_model=args.student,
        assume_same_tokenizer_as_teacher=bool(
            getattr(args, "assume_same_tokenizer_as_teacher", False)
        ),
        tokenizer_model=getattr(args, "tokenizer_model", None),
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

    teacher_hf_tok = data.teacher_tokenizer
    tokenizer = data.student_tokenizer

    sequential = bool(getattr(args, "sequential_gpu_pipeline", False))
    teacher_gpus = _gpu_span(args.teacher_gpu, args.teacher_gpu_count, "teacher")
    student_gpus = _gpu_span(args.student_gpu, args.student_gpu_count, "student")
    teacher_device_map, teacher_max_memory = _device_map_and_memory(teacher_gpus)
    student_device_map, student_max_memory = _device_map_and_memory(student_gpus)
    if sequential and set(teacher_gpus) & set(student_gpus):
        log.info(
            "Sequential GPU overlap on %s — teacher loads first each step/then frees "
            "before student/King forwards (sharing one pool of devices).",
            sorted(set(teacher_gpus) & set(student_gpus)),
        )

    teacher = None
    teacher_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": teacher_device_map,
        "trust_remote_code": True,
        "attn_implementation": TEACHER_ATTN_IMPLEMENTATION,
    }
    if teacher_max_memory is not None:
        teacher_load_kwargs["max_memory"] = teacher_max_memory

    if not sequential:
        log.info("Loading teacher (%s) on GPUs %s...", args.teacher, teacher_gpus)
        teacher = _instantiate_causal_lm(
            args.teacher, teacher_load_kwargs, train=False, freeze_all=True
        )
        teacher_in_dev = _first_param_device(teacher)
        if teacher_in_dev.type == "cuda":
            log.info(
                "  Teacher primary %s: %.1fGB",
                teacher_in_dev,
                torch.cuda.memory_allocated(teacher_in_dev.index) / 1e9,
            )
    else:
        log.info(
            "Sequential pipeline: omitting resident teacher LM (e.g. 8×GPU Kimi shard). "
            "Teacher forwards run in bursts with full VRAM, then unload before student."
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
    student = _instantiate_causal_lm(
        student_source, student_load_kwargs, train=True, freeze_all=False
    )
    student_in_dev = _first_param_device(student)
    log.info(
        f"  Student: {sum(p.numel() for p in student.parameters()):,} params, "
        f"{(torch.cuda.memory_allocated(student_in_dev.index)/1e9) if student_in_dev.type == 'cuda' else 0.0:.1f}GB"
    )
    if getattr(data, "dual_tokenizers", False):
        log.info(
            "  Cross-tokenizer distillation enabled (different teacher/student repos). "
            "KL uses timestep alignment via offset_mapping where available, "
            "with length-ratio fallback; vocab bridge uses kd_top_k=%s piece mapping.",
            int(getattr(args, "kd_top_k", 512)),
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
    _vacuum_student_for_teacher_eval = False
    _student_reload_opt_state: dict | None = None
    _student_reload_sched_state: dict | None = None
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
            eval_prompts_from_api = bool(args.eval_use_api_prompts)
            bundle: dict | None = None
            if eval_prompts_from_api:
                try:
                    bundle = _fetch_eval_data_bundle(args.eval_data_url)
                except EvalDataNotAvailable:
                    hint = ""
                    if cache_path is not None:
                        hint = (
                            f" Put a valid cache at {cache_path.resolve()} or wait until {args.eval_data_url!r} "
                            "returns JSON, then rebuild."
                        )
                    log.warning(
                        "eval_data API has no bundle yet (HTTP 404). Falling back to local dataset prompts "
                        "(--eval_use_dataset_prompts behavior).%s",
                        hint,
                    )
                    eval_prompts_from_api = False

            if eval_prompts_from_api and bundle is not None:
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
            reuse_persistent_teacher = (teacher is not None) and (not sequential)
            if reuse_persistent_teacher:
                teacher_eval = teacher
            else:
                if set(teacher_gpus) & set(student_gpus):
                    log.info(
                        "Unloading student temporarily so teacher eval-cache build can use the full GPU pool %s.",
                        sorted(set(teacher_gpus) & set(student_gpus)),
                    )
                    _student_reload_opt_state = optimizer.state_dict()
                    _student_reload_sched_state = scheduler.state_dict()
                    _purge_cuda_model_hold("student (VRAM for teacher eval cache)", student)
                    student = None
                    _vacuum_student_for_teacher_eval = True
                log.info("Loading teacher LM for eval cache build / metrics (release after cache)...")
                teacher_eval_kwargs = {
                    "dtype": torch.bfloat16,
                    "device_map": teacher_device_map,
                    "trust_remote_code": True,
                    "attn_implementation": TEACHER_ATTN_IMPLEMENTATION,
                }
                if teacher_max_memory is not None:
                    teacher_eval_kwargs["max_memory"] = teacher_max_memory
                try:
                    teacher_eval = _instantiate_causal_lm(
                        args.teacher, teacher_eval_kwargs, train=False, freeze_all=True
                    )
                except Exception:
                    if _vacuum_student_for_teacher_eval and student is None:
                        log.warning(
                            "Teacher load failed; restoring student on GPUs %s before re-raising.",
                            student_gpus,
                        )
                        student = _instantiate_causal_lm(
                            student_source, student_load_kwargs, train=True, freeze_all=False
                        )
                        optimizer = AdamW(
                            [p for p in student.parameters() if p.requires_grad],
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                        )
                        if _student_reload_opt_state is not None:
                            optimizer.load_state_dict(_student_reload_opt_state)
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer, int(args.warmup_steps), scheduler_total_steps
                        )
                        if _student_reload_sched_state is not None:
                            scheduler.load_state_dict(_student_reload_sched_state)
                    raise
        king_load_kwargs = {
            "revision": args.king_revision,
            "dtype": torch.bfloat16,
            "device_map": king_device_map,
            "trust_remote_code": True,
        }
        if king_max_memory is not None:
            king_load_kwargs["max_memory"] = king_max_memory
        if sequential:
            log.info(
                "King (%s): lazy load before each periodic eval so all %s GPUs are free for fused teacher bursts.",
                args.king_repo,
                len(teacher_gpus),
            )
        else:
            log.info("Eval enabled: loading king model (%s) on GPUs %s...", args.king_repo, king_gpus)
            king_eval = _instantiate_causal_lm(
                args.king_repo, king_load_kwargs, train=False, freeze_all=True
            )

        if not eval_cache:
            if teacher_eval is not None:
                teacher_eval.eval()
            log.info("Preparing deterministic eval cache from teacher continuations...")
            cross_tc = bool(getattr(data, "dual_tokenizers", False))
            try:
                eval_cache = _prepare_eval_cache(
                    teacher_eval=teacher_eval,
                    teacher_tokenizer=teacher_hf_tok,
                    prompts=eval_prompts,
                    max_new_tokens=eval_tokens,
                    seed=eval_seed,
                    cross_tokenizer=cross_tc,
                    student_tokenizer=tokenizer if cross_tc else None,
                    max_student_seq=int(args.max_seq_len),
                )
                if cache_path:
                    meta = {
                        "teacher": args.teacher,
                        "student": str(args.student),
                        "assume_same_tokenizer_as_teacher": bool(
                            getattr(args, "assume_same_tokenizer_as_teacher", False)
                        ),
                        "cross_tokenizer_kd": cross_tc,
                        "prompt_source_mode": "api" if eval_prompts_from_api else "dataset",
                        "eval_data_url": args.eval_data_url if eval_prompts_from_api else None,
                        "eval_dataset": None if eval_prompts_from_api else args.eval_dataset,
                        "eval_block_number": None if eval_prompts_from_api else args.eval_block_number,
                        "eval_block_hash": None if eval_prompts_from_api else (args.eval_block_hash or None),
                        "eval_prompts": len(eval_prompts),
                        "max_new_tokens": eval_tokens,
                        "eval_seed": eval_seed,
                        "saved_at": _utc_now_iso(),
                    }
                    _save_eval_cache_payload(cache_path, eval_cache, eval_prompts, meta)
                    log.info("Saved eval cache to %s", cache_path)
            finally:
                # Ephemeral teacher copy (sequential trainer or parallel without resident teacher).
                if teacher_eval is not None and teacher_eval is not teacher:
                    _purge_cuda_model_hold("teacher LM (eval cache build only)", teacher_eval)
                    teacher_eval = None
                if _vacuum_student_for_teacher_eval and student is None:
                    log.info("Reloading student (%s) on GPUs %s...", student_source, student_gpus)
                    student = _instantiate_causal_lm(
                        student_source, student_load_kwargs, train=True, freeze_all=False
                    )
                    optimizer = AdamW(
                        [p for p in student.parameters() if p.requires_grad],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                    if _student_reload_opt_state is not None:
                        optimizer.load_state_dict(_student_reload_opt_state)
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer, int(args.warmup_steps), scheduler_total_steps
                    )
                    if _student_reload_sched_state is not None:
                        scheduler.load_state_dict(_student_reload_sched_state)
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
    if sequential:
        log.info(
            "  Sequential GPU pipeline: one fused Kimi LM load per optimizer step (%s reloads/teach fwd), "
            "then student-only backward. Tune --samples_per_step / online_chunk_size to cap CPU logits.",
            "many" if getattr(args, "sequential_teacher_reload_each_microbatch", False) else "single",
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
        samples = []
        for row in batch:
            ids_s = row["student_input_ids"]
            ids_t = row["teacher_input_ids"]
            ls_s = int(row.get("student_loss_start", -1))
            if ls_s < 0 or ls_s >= ids_s.shape[0] - 1:
                ls_s = int(args.kl_start_pos)
            if ids_s.shape[0] <= ls_s + 10 or ids_t.shape[0] < 8:
                continue
            pmap = _teacher_pos_indices_cpu_dual(
                row, int(ids_t.shape[0]), int(ids_s.shape[0]), ls_s
            )
            samples.append(
                {
                    "teacher_input_ids": ids_t.to(dtype=torch.long),
                    "student_input_ids": ids_s.to(dtype=torch.long),
                    "student_loss_start": ls_s,
                    "pmap_cpu": pmap,
                }
            )
        if not samples:
            log.warning("No valid tokens in batch after filtering.")
            break

        optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        pad_teacher = teacher_hf_tok.pad_token_id
        if pad_teacher is None:
            pad_teacher = teacher_hf_tok.eos_token_id
        pad_student = tokenizer.pad_token_id
        if pad_student is None:
            pad_student = tokenizer.eos_token_id

        chunk_sz = max(1, int(getattr(args, "online_chunk_size", 1)))
        n_tokens = len(samples)
        total_loss = 0.0
        _dt = next(student.parameters()).dtype
        kd_top_k = int(args.kd_top_k)
        chunk_batches = list(_chunk_list(samples, chunk_sz))

        # --- GPU scheduling: resident teacher || sequential fused bursts (8×GPU teacher, then student).
        if sequential:
            if king_eval is not None:
                _purge_cuda_model_hold(
                    "king (sequential: free devices for fused teacher LM pass)", king_eval
                )
                king_eval = None

            blobs_cpu: list[torch.Tensor] = []
            reload_teacher_micro = getattr(
                args, "sequential_teacher_reload_each_microbatch", False
            )

            def _teacher_fwd_sub_to_cpu(sub, lm: torch.nn.Module) -> torch.Tensor:
                """Single micro-batch teacher forward → CPU float logits [B,L,V]."""
                sub_teacher_tokens = [r["teacher_input_ids"] for r in sub]
                input_ids_tc, _ = _pad_token_batch(sub_teacher_tokens, int(pad_teacher))
                attn_t = (input_ids_tc != int(pad_teacher)).long()
                tdev_live = _first_param_device(lm)
                input_ids_tok = input_ids_tc.to(
                    device=tdev_live, dtype=torch.long, non_blocking=True
                )
                attn_t_dev = attn_t.to(device=tdev_live, dtype=torch.long, non_blocking=True)
                with torch.no_grad():
                    return lm(input_ids_tok, attention_mask=attn_t_dev).logits.float().cpu()

            if reload_teacher_micro:
                log.debug("Sequential teacher: reload full LM each micro-batch (max peak CPU logits).")
                for sub_m in chunk_batches:
                    teacher_lm_pulse = _instantiate_causal_lm(
                        args.teacher, teacher_load_kwargs, train=False, freeze_all=True
                    )
                    blobs_cpu.append(_teacher_fwd_sub_to_cpu(sub_m, teacher_lm_pulse))
                    _purge_cuda_model_hold(
                        "teacher LM (micro-batch pulse)", teacher_lm_pulse
                    )
            else:
                teacher_lm_pulse = _instantiate_causal_lm(
                    args.teacher, teacher_load_kwargs, train=False, freeze_all=True
                )
                for sub_m in chunk_batches:
                    blobs_cpu.append(_teacher_fwd_sub_to_cpu(sub_m, teacher_lm_pulse))
                _purge_cuda_model_hold(
                    "teacher LM (one fused pulse / train step)", teacher_lm_pulse
                )

            assert len(blobs_cpu) == len(chunk_batches)

            for sub, t_logits_cpu in zip(chunk_batches, blobs_cpu):
                n_sub = len(sub)
                sub_student_tokens = [r["student_input_ids"] for r in sub]
                sub_student_starts = [int(r["student_loss_start"]) for r in sub]
                pmap_list = [r["pmap_cpu"] for r in sub]

                input_ids_sc, _ = _pad_token_batch(sub_student_tokens, int(pad_student))
                attn_s = (input_ids_sc != int(pad_student)).long()
                input_ids_s = input_ids_sc.to(device=sdev, dtype=torch.long, non_blocking=True)
                attn_s_dev = attn_s.to(device=sdev, dtype=torch.long, non_blocking=True)

                t_logits = t_logits_cpu.to(device=sdev, dtype=_dt, non_blocking=True)
                s_logits = student(input_ids_s, attention_mask=attn_s_dev).logits

                if not getattr(data, "dual_tokenizers", False):
                    loss_mask = _build_loss_mask(sub_student_tokens, sub_student_starts)
                    loss_mask_dev = loss_mask.to(device=sdev, dtype=torch.float32)
                    loss_c = kl_loss_masked_positions(s_logits, t_logits, loss_mask_dev)
                else:
                    micro_l = []
                    for bi in range(n_sub):
                        s_log = s_logits[bi]
                        t_log = t_logits[bi]
                        ssl = sub_student_starts[bi]
                        pmap = pmap_list[bi].to(device=sdev, dtype=torch.long)
                        li_b = int(sub_student_tokens[bi].shape[0])
                        rows_s = []
                        rows_t = []
                        for pos in range(ssl, li_b - 1):
                            t_i = int(pmap[pos].item())
                            if t_i < 0:
                                continue
                            if t_i >= t_log.shape[0]:
                                continue
                            rows_s.append(s_log[pos])
                            rows_t.append(t_log[t_i])
                        if not rows_s:
                            continue
                        stk_s = torch.stack(rows_s, dim=0)
                        stk_t = torch.stack(rows_t, dim=0)
                        micro_l.append(
                            _cross_vocab_kl_masked_piecewise(
                                stk_s,
                                stk_t,
                                teacher_hf_tok,
                                tokenizer,
                                kd_top_k,
                                teacher_is_log_probs=False,
                            )
                        )
                    if not micro_l:
                        log.warning(
                            "Cross-tokenizer step had no valid aligned positions; skipping."
                        )
                        del t_logits, s_logits
                        continue
                    loss_c = torch.stack(micro_l, dim=0).mean()

                (loss_c * (n_sub / n_tokens)).backward()
                total_loss += float(loss_c.item()) * (n_sub / n_tokens)
                del t_logits, s_logits, loss_c

        else:
            for sub in chunk_batches:
                n_sub = len(sub)
                sub_teacher_tokens = [r["teacher_input_ids"] for r in sub]
                sub_student_tokens = [r["student_input_ids"] for r in sub]
                sub_student_starts = [int(r["student_loss_start"]) for r in sub]
                pmap_list = [r["pmap_cpu"] for r in sub]

                input_ids_tc, _ = _pad_token_batch(sub_teacher_tokens, int(pad_teacher))
                input_ids_sc, _ = _pad_token_batch(sub_student_tokens, int(pad_student))
                attn_t = (input_ids_tc != int(pad_teacher)).long()
                attn_s = (input_ids_sc != int(pad_student)).long()
                input_ids_t = input_ids_tc.to(device=tdev, dtype=torch.long, non_blocking=True)
                input_ids_s = input_ids_sc.to(device=sdev, dtype=torch.long, non_blocking=True)
                attn_t_dev = attn_t.to(device=tdev, dtype=torch.long, non_blocking=True)
                attn_s_dev = attn_s.to(device=sdev, dtype=torch.long, non_blocking=True)

                with torch.no_grad():
                    t_logits = teacher(input_ids_t, attention_mask=attn_t_dev).logits.to(
                        device=sdev, dtype=_dt, non_blocking=True
                    )
                s_logits = student(input_ids_s, attention_mask=attn_s_dev).logits

                if not getattr(data, "dual_tokenizers", False):
                    loss_mask = _build_loss_mask(sub_student_tokens, sub_student_starts)
                    loss_mask_dev = loss_mask.to(device=sdev, dtype=torch.float32)
                    loss_c = kl_loss_masked_positions(s_logits, t_logits, loss_mask_dev)
                else:
                    micro = []
                    for bi in range(n_sub):
                        s_log = s_logits[bi]
                        t_log = t_logits[bi]
                        ssl = sub_student_starts[bi]
                        pmap = pmap_list[bi].to(device=sdev, dtype=torch.long)
                        li = int(sub_student_tokens[bi].shape[0])
                        rows_s = []
                        rows_t = []
                        for pos in range(ssl, li - 1):
                            ti = int(pmap[pos].item())
                            if ti < 0:
                                continue
                            if ti >= t_log.shape[0]:
                                continue
                            rows_s.append(s_log[pos])
                            rows_t.append(t_log[ti])
                        if not rows_s:
                            continue
                        stk_s = torch.stack(rows_s, dim=0)
                        stk_t = torch.stack(rows_t, dim=0)
                        micro.append(
                            _cross_vocab_kl_masked_piecewise(
                                stk_s,
                                stk_t,
                                teacher_hf_tok,
                                tokenizer,
                                kd_top_k,
                                teacher_is_log_probs=False,
                            )
                        )
                    if not micro:
                        log.warning(
                            "Cross-tokenizer step had no valid aligned positions; skipping."
                        )
                        del t_logits, s_logits
                        continue
                    loss_c = torch.stack(micro, dim=0).mean()

                (loss_c * (n_sub / n_tokens)).backward()
                total_loss += float(loss_c.item()) * (n_sub / n_tokens)
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
            if sequential:
                log.info(
                    "Periodic eval (%s): loading king LM temporarily on %s",
                    args.king_repo,
                    king_gpus,
                )
                king_eval = _instantiate_causal_lm(
                    args.king_repo, king_load_kwargs, train=False, freeze_all=True
                )
            eval_scores, king_scores, ttest = _evaluate_against_king(
                student=student,
                king_eval=king_eval,
                eval_cache=eval_cache,
                teacher_tokenizer=teacher_hf_tok,
                student_tokenizer=tokenizer,
                kd_top_k=int(args.kd_top_k),
            )
            if sequential and king_eval is not None:
                _purge_cuda_model_hold("king after periodic eval (sequential)", king_eval)
                king_eval = None
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
    p_eval_cache.add_argument(
        "--student",
        type=str,
        default=None,
        help=(
            "Miner / Qwen tokenizer Hub id when it differs from --teacher; enables student_full_seq "
            "in the cached payload for king-eval on Qwen token ids."
        ),
    )
    p_eval_cache.add_argument(
        "--max_student_seq",
        type=int,
        default=MAX_SEQ_LEN,
        help="Truncate retokenized student eval sequences to this length.",
    )
    p_eval_cache.add_argument(
        "--assume_same_tokenizer_as_teacher",
        action="store_true",
        help=(
            "Teacher and student share the Moonshot tokenizer; omit student_full_seq in the cache payload."
        ),
    )
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
    p_train.add_argument(
        "--assume_same_tokenizer_as_teacher",
        action="store_true",
        help=(
            "Load a single tokenizer (from --tokenizer_model if set, else --teacher): use when the student "
            "is another Moonshot repo with identical tokenizer+vocab as Kimi. Disables cross-tokenizer KL."
        ),
    )
    p_train.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help=(
            "HF repo id whose tokenizer/chat template weights we use when --assume_same_tokenizer_as_teacher is "
            "set (defaults to --teacher)."
        ),
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
    p_train.add_argument(
        "--kd_top_k",
        type=int,
        default=512,
        help=(
            "Cross-tokenizer KD only: per aligned timestep, approximate teacher distribution with its "
            "top-K logits and map pieces to student ids (see eval/cross_tokenizer.py heuristic)."
        ),
    )
    p_train.add_argument(
        "--sequential_gpu_pipeline",
        action="store_true",
        help=(
            "Time-multiplex GPUs: Kimi shards on all CUDA devices during teacher bursts, unload the "
            "teacher completely, then train the student (required when teacher consumes all GPUs). "
            "King eval lazily reloads."
        ),
    )
    p_train.add_argument(
        "--sequential_teacher_reload_each_microbatch",
        action="store_true",
        help=(
            "With sequential_gpu_pipeline only: instantiate+purge the teacher LM for every micro-batch "
            "(lower CPU peak logits vs one fused pulse; slower)."
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
