#!/usr/bin/env python3
"""
Runs on a Lium GPU pod. Loads a model via vLLM, generates logprobs, saves to JSON.

Usage:
    python3 pod_eval.py --model Qwen/Qwen3-32B --prompts prompts.json --output teacher.json --max-tokens 256

Output JSON format:
    [
        {
            "text": "generated text...",
            "logprobs": [
                {"token_a": -0.5, "token_b": -1.2, ...},  # position 0
                {"token_c": -0.3, "token_d": -2.1, ...},  # position 1
                ...
            ],
            "n_tokens": 256
        },
        ...
    ]
"""
import json, sys, argparse, time


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference and extract logprobs")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--prompts", required=True, help="JSON file with list of prompt strings")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    print(f"[pod_eval] Loading model: {args.model}", flush=True)
    t0 = time.time()

    from vllm import LLM, SamplingParams

    with open(args.prompts) as f:
        prompts = json.load(f)

    print(f"[pod_eval] Loaded {len(prompts)} prompts", flush=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    load_time = time.time() - t0
    print(f"[pod_eval] Model loaded in {load_time:.1f}s", flush=True)

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=args.top_k,
    )

    print(f"[pod_eval] Generating for {len(prompts)} prompts (max_tokens={args.max_tokens}, "
          f"top_k={args.top_k})...", flush=True)
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

    with open(args.output, "w") as f:
        json.dump(results, f)

    total_tokens = sum(r["n_tokens"] for r in results)
    print(f"[pod_eval] Saved {len(results)} results to {args.output}", flush=True)
    print(f"[pod_eval] Total tokens: {total_tokens}, generation time: {gen_time:.1f}s", flush=True)
    print(f"[pod_eval] Tokens/sec: {total_tokens/gen_time:.1f}", flush=True)

    # Explicitly delete model to free GPU memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print("[pod_eval] Model unloaded, GPU memory freed", flush=True)


if __name__ == "__main__":
    main()
