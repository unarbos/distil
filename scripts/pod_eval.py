#!/usr/bin/env python3
"""
Batch inference on a Lium GPU pod. Evaluates multiple models in a single run.

Usage:
    # Single model:
    python3 pod_eval.py --models Qwen/Qwen3.5-35B-A3B --prompts prompts.json --outdir /home/results

    # Multiple models (batch):
    python3 pod_eval.py --models Qwen/Qwen3.5-35B-A3B,Qwen/Qwen3-8B,Qwen/Qwen3-1.7B \
        --prompts prompts.json --outdir /home/results

Output: /home/results/<sanitized_model_name>.json per model
"""
import json, sys, argparse, time, os, gc


def sanitize_name(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def eval_model(model_name: str, prompts: list, outdir: str,
               max_tokens: int, top_k: int, max_model_len: int):
    """Load model, run batch inference, save results, free GPU."""
    from vllm import LLM, SamplingParams

    out_path = os.path.join(outdir, f"{sanitize_name(model_name)}.json")
    print(f"\n{'='*60}", flush=True)
    print(f"[eval] Model: {model_name}", flush=True)
    print(f"[eval] Prompts: {len(prompts)}, max_tokens: {max_tokens}, top_k: {top_k}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        enforce_eager=True,  # Skip CUDA graph capture — faster startup
    )
    load_time = time.time() - t0
    print(f"[eval] Loaded in {load_time:.1f}s", flush=True)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=top_k,
    )

    # Single batch call — vLLM handles all prompts together
    t1 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t1

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
        results.append({
            "text": completion.text,
            "logprobs": token_logprobs,
            "n_tokens": len(token_logprobs),
        })

    with open(out_path, "w") as f:
        json.dump(results, f)

    total_tokens = sum(r["n_tokens"] for r in results)
    print(f"[eval] Generated {total_tokens} tokens in {gen_time:.1f}s ({total_tokens/gen_time:.0f} tok/s)", flush=True)
    print(f"[eval] Saved to {out_path}", flush=True)

    # Free GPU memory
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    # Force kill any lingering CUDA contexts
    time.sleep(2)
    print(f"[eval] GPU freed", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt strings")
    parser.add_argument("--outdir", default="/home/results", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.prompts) as f:
        prompts = json.load(f)

    models = [m.strip() for m in args.models.split(",")]
    print(f"[batch] Evaluating {len(models)} models on {len(prompts)} prompts", flush=True)

    t_total = time.time()
    for model in models:
        eval_model(model, prompts, args.outdir,
                   args.max_tokens, args.top_k, args.max_model_len)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"[batch] All {len(models)} models evaluated in {elapsed:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
