"""
vLLM-based inference wrapper for getting logprobs from models.

Supports both teacher (GLM-5) and student (distilled) models.
The teacher is loaded once and kept resident; students are loaded
one at a time, evaluated, then unloaded.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    device: str = "cuda",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    **kwargs: Any,
) -> LLM:
    """Load a model via vLLM with optional tensor parallelism."""
    logger.info("Loading model %s (tp=%d)", model_name, tensor_parallel_size)
    return LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs,
    )


def generate_with_logprobs(
    model: LLM,
    prompts: list[str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_k_logprobs: int = 50,
) -> list[dict]:
    """
    Generate text and return per-token logprobs for each prompt.

    Returns a list of dicts, one per prompt:
        {
            "text": str,           # generated completion text
            "logprobs": list[      # one dict per generated token position
                dict[str, float]   # token_string → log_probability
            ],
        }
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=top_k_logprobs,
    )

    outputs = model.generate(prompts, sampling_params)

    results: list[dict] = []
    for output in outputs:
        completion = output.outputs[0]
        token_logprobs: list[dict[str, float]] = []
        if completion.logprobs:
            for position_logprobs in completion.logprobs:
                position_dict: dict[str, float] = {}
                for _token_id, logprob_info in position_logprobs.items():
                    position_dict[logprob_info.decoded_token] = logprob_info.logprob
                token_logprobs.append(position_dict)

        results.append(
            {
                "text": completion.text,
                "logprobs": token_logprobs,
            }
        )

    return results


def unload_model(model: LLM) -> None:
    """Release model from GPU memory."""
    logger.info("Unloading model from GPU")
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_model_params_billions(model_name: str) -> float:
    """
    Query HuggingFace Hub for total parameter count (in billions).

    Tries safetensors metadata first (fast, no download required),
    then falls back to loading the model config.
    """
    from huggingface_hub import model_info as hf_model_info

    info = hf_model_info(model_name)

    # Safetensors metadata includes an accurate total
    if info.safetensors and hasattr(info.safetensors, "total"):
        total_params = info.safetensors.total
        return total_params / 1e9

    # Fallback: inspect config (approximate)
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_params = getattr(config, "num_parameters", 0)
        if num_params:
            return num_params / 1e9
    except Exception:
        logger.warning("Could not determine param count for %s", model_name)

    return 0.0
