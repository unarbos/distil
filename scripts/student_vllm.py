"""In-process vLLM backend for student bench generation.

The eval driver historically generates one prompt at a time via HF
``model.generate()``. With ~25 benches × 14-18 prompts × up to 16K
tokens per prompt, this is the dominant cost of an eval round. vLLM's
continuous-batching scheduler turns the same workload into a single
batched call and gives a 30-50× wall-time win on autoregressive phases.

We keep the HF student loaded for the KL forward path (which is a
single batched forward, already efficient). vLLM is brought up
alongside the HF model with a low ``gpu_memory_utilization`` so both
fit in VRAM at the same time on B200 (183 GB) / B300 (275 GB) class
cards. For ~33B distilled students in bf16 (~66 GB per copy), the
combined budget is ~140 GB which leaves ample KV-cache headroom.

Public surface kept intentionally tiny: a module-global handle plus a
single ``generate_batch`` entry point. Callers in ``pod_eval_vllm.py``
check ``is_active()`` and dispatch when set; the HF path stays the
default so production is unaffected when the flag is off.
"""

from __future__ import annotations

import gc
import os
from typing import Iterable

# Lazily imported so the module can be imported even if vllm is not
# installed (validator still imports pod_eval_vllm.py for type info).
_STUDENT_VLLM = None
_STUDENT_NAME: str | None = None
_STUDENT_TOKENIZER = None
_VLLM_DTYPE = "bfloat16"


def is_active() -> bool:
    """True iff a vLLM student is currently loaded."""
    return _STUDENT_VLLM is not None


def current_student() -> str | None:
    return _STUDENT_NAME


def start(
    model_name: str,
    revision: str | None = None,
    *,
    tokenizer_name_or_path: str | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    trust_remote_code: bool | None = None,
    enforce_eager: bool | None = None,
):
    """Spin up a per-student vLLM ``LLM`` in-process.

    Defaults are read from ``DISTIL_STUDENT_VLLM_*`` environment
    variables so production validators can tune without code edits.

    Returns the ``vllm.LLM`` instance. Idempotent: calling ``start``
    twice for the same model is a no-op; calling for a different
    model auto-shuts-down the previous one first.
    """
    global _STUDENT_VLLM, _STUDENT_NAME, _STUDENT_TOKENIZER

    if _STUDENT_VLLM is not None and _STUDENT_NAME == model_name:
        return _STUDENT_VLLM
    if _STUDENT_VLLM is not None:
        stop()

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(
            os.environ.get("DISTIL_STUDENT_VLLM_GPU_UTIL", "0.55")
        )
    if max_model_len is None:
        max_model_len = int(os.environ.get("DISTIL_STUDENT_VLLM_MAX_LEN", "16384"))
    # 2026-05-15: clamp max_model_len to the student's actual context window.
    # Some distilled students inherit a short max_position_embeddings (e.g.
    # 8192) and vLLM hard-fails to start when the user-supplied max_model_len
    # exceeds the model config. Falling back to HF blows the round budget
    # (~30 min per student → ~3 h for one shard). Detect the cap and clamp.
    try:
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(
            model_name,
            revision=revision if (revision and revision != "main") else None,
            trust_remote_code=True,
        )
        _model_max = None
        _attr_used = None
        for _attr in ("max_position_embeddings", "model_max_length",
                      "max_sequence_length", "n_positions", "seq_length"):
            _v = getattr(_cfg, _attr, None)
            if _v is not None and int(_v) > 0:
                _model_max = int(_v)
                _attr_used = _attr
                break
        if _model_max and _model_max < int(max_model_len):
            print(
                f"[student_vllm] clamping max_model_len {max_model_len} → "
                f"{_model_max} (model.{_attr_used})",
                flush=True,
            )
            max_model_len = _model_max
    except Exception as _cfg_exc:
        print(
            f"[student_vllm] could not resolve model max context "
            f"(continuing with env value {max_model_len}): {_cfg_exc!r}",
            flush=True,
        )
    if tensor_parallel_size is None:
        tensor_parallel_size = int(
            os.environ.get("DISTIL_STUDENT_VLLM_TP", "1")
        )
    if trust_remote_code is None:
        # Kimi-K2.6 derivatives (the entire current student pool) ship a
        # custom ``TokenizersBackend`` class that vLLM can only load with
        # trust_remote_code on. Default ON; set DISTIL_STUDENT_VLLM_TRC=0
        # to opt out for non-custom tokenizers.
        trust_remote_code = (
            os.environ.get("DISTIL_STUDENT_VLLM_TRC", "1") == "1"
        )
    if enforce_eager is None:
        enforce_eager = (
            os.environ.get("DISTIL_STUDENT_VLLM_EAGER", "0") == "1"
        )

    from vllm import LLM
    kwargs = dict(
        model=model_name,
        dtype=_VLLM_DTYPE,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )
    if revision and revision != "main":
        kwargs["revision"] = revision
    # Most distilled students inherit Kimi-K2.6's custom
    # ``TokenizersBackend`` class via ``auto_map`` but don't ship the
    # python source for it, so vLLM's tokenizer load 500s out unless
    # we steer it to the teacher's snapshot (which IS complete and has
    # already been cached on the pod for Phase 1).
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = (
            os.environ.get("DISTIL_STUDENT_VLLM_TOKENIZER")
            or os.environ.get("TEACHER_NAME")
            or os.environ.get("DISTIL_TEACHER_NAME")
        )
    if tokenizer_name_or_path:
        kwargs["tokenizer"] = tokenizer_name_or_path

    _STUDENT_VLLM = LLM(**kwargs)
    _STUDENT_NAME = model_name
    try:
        _STUDENT_TOKENIZER = _STUDENT_VLLM.get_tokenizer()
    except Exception:
        _STUDENT_TOKENIZER = None
    return _STUDENT_VLLM


def stop() -> None:
    """Free the vLLM engine and its KV cache. Safe to call when not active."""
    global _STUDENT_VLLM, _STUDENT_NAME, _STUDENT_TOKENIZER
    if _STUDENT_VLLM is None:
        return
    try:
        del _STUDENT_VLLM
    except Exception:
        pass
    _STUDENT_VLLM = None
    _STUDENT_NAME = None
    _STUDENT_TOKENIZER = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def generate_batch(
    rendered_prompts: list[str],
    max_tokens_per: list[int] | int,
    *,
    greedy: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seeds: Iterable[int] | None = None,
    stop_strings: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Run a vLLM batched generate.

    ``rendered_prompts`` are full chat-template-applied strings (the
    caller handles ``apply_chat_template`` upstream because each bench
    has its own ``enable_thinking`` and message-shape requirements).

    ``max_tokens_per`` accepts either a per-prompt list (preferred — vLLM
    schedules each request with its own budget) or a single int applied
    to every prompt.

    Returns a list of ``(decoded_text, n_generated_tokens)`` aligned
    with the input order.
    """
    if _STUDENT_VLLM is None:
        raise RuntimeError("student_vllm.start(...) was not called")
    if not rendered_prompts:
        return []
    from vllm import SamplingParams

    n = len(rendered_prompts)
    if isinstance(max_tokens_per, int):
        mt_list = [max_tokens_per] * n
    else:
        mt_list = list(max_tokens_per)
        if len(mt_list) != n:
            raise ValueError(
                f"max_tokens_per length {len(mt_list)} != prompts length {n}"
            )

    seeds_list: list[int | None] = []
    if seeds is not None:
        seeds_list = list(seeds)
        if len(seeds_list) != n:
            raise ValueError(
                f"seeds length {len(seeds_list)} != prompts length {n}"
            )
    else:
        seeds_list = [None] * n

    sampling_params = []
    for i in range(n):
        sp_kwargs = dict(
            temperature=0.0 if greedy else temperature,
            top_p=1.0 if greedy else top_p,
            max_tokens=int(mt_list[i]),
        )
        if seeds_list[i] is not None:
            sp_kwargs["seed"] = int(seeds_list[i]) & 0x7FFFFFFF
        if stop_strings:
            sp_kwargs["stop"] = list(stop_strings)
        sampling_params.append(SamplingParams(**sp_kwargs))

    outs = _STUDENT_VLLM.generate(
        rendered_prompts, sampling_params, use_tqdm=False,
    )
    results: list[tuple[str, int]] = []
    by_id = {o.request_id: o for o in outs} if outs and hasattr(outs[0], "request_id") else None
    if by_id and len(by_id) == n:
        # Best-effort preserve input order: vLLM 0.10+ returns in
        # submission order already; this guard only matters when a
        # future version reorders.
        ordered = outs
    else:
        ordered = outs
    for out in ordered:
        gen = out.outputs[0]
        results.append((gen.text, len(gen.token_ids)))
    return results
