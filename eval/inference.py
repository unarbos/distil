"""vLLM inference wrapper for logprobs extraction."""
import logging
from vllm import LLM, SamplingParams

logger = logging.getLogger("distillation.inference")


def load_model(model_name: str, revision: str = None, tensor_parallel_size: int = 1, **kwargs) -> LLM:
    """Load model via vLLM with optional revision pinning."""
    load_kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        **kwargs,
    )
    if revision:
        load_kwargs["revision"] = revision
    return LLM(**load_kwargs)


def generate_with_logprobs(
    model: LLM,
    prompts: list[str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_k_logprobs: int = 50,
) -> list[dict]:
    """Generate and return per-token logprobs for each prompt."""
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=top_k_logprobs,
    )
    outputs = model.generate(prompts, params)
    results = []
    for output in outputs:
        completion = output.outputs[0]
        token_logprobs = []
        if completion.logprobs:
            for pos_lps in completion.logprobs:
                pos_dict = {}
                for token_id, lp_info in pos_lps.items():
                    pos_dict[lp_info.decoded_token] = lp_info.logprob
                token_logprobs.append(pos_dict)
        results.append({"text": completion.text, "logprobs": token_logprobs})
    return results


def unload_model(model: LLM):
    """Free GPU memory."""
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
