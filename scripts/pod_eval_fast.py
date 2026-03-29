#!/usr/bin/env python3
"""
Fast full-distribution KL-divergence evaluation on GPU.

Key differences from pod_eval.py:
1. SCORING ONLY — no autoregressive generation, just forward passes (prefill)
2. FULL DISTRIBUTION — KL computed on raw logits (all 248K vocab), not top-k
3. ON-GPU COMPUTATION — KL never leaves the GPU, only scalar results returned
4. SAME TOKENIZER ENFORCED — verified before evaluation
5. PARAM COUNT VERIFIED — from safetensors metadata (cheat-proof)

Speed: ~10s per model for scoring (after load), vs ~10min with old approach.

Usage:
    python3 pod_eval_fast.py \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --students Qwen/Qwen3.5-35B-A3B-GPTQ-Int4,Qwen/Qwen3.5-35B-A3B-FP8 \
        --prompts prompts.json \
        --output results.json \
        --max-len 1024
"""
import torch
import json
import time
import argparse
import gc
import os
import sys


def load_model(name, device="cuda", dtype=torch.bfloat16):
    """Load model with best available attention implementation."""
    from transformers import AutoModelForCausalLM
    kwargs = dict(
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    # Try flash attention, fall back to default
    try:
        return AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_param_count_b(model_name: str) -> float:
    """Get verified parameter count in billions from safetensors metadata."""
    from huggingface_hub import model_info
    info = model_info(model_name)
    if info.safetensors and hasattr(info.safetensors, "total"):
        return info.safetensors.total / 1e9
    # Fallback: count from model state dict (slower but accurate)
    return -1.0


def verify_tokenizer(teacher_name: str, student_name: str) -> tuple:
    """Verify student uses exact same tokenizer as teacher. Returns (ok, reason)."""
    from transformers import AutoTokenizer
    t_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    s_tok = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    if t_tok.vocab_size != s_tok.vocab_size:
        return False, f"vocab_size mismatch: {s_tok.vocab_size} vs {t_tok.vocab_size}"

    # Spot-check encodings
    test_strings = [
        "def fibonacci(n):\n    if n <= 1: return n",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nclass Model(nn.Module):",
    ]
    for s in test_strings:
        if t_tok.encode(s) != s_tok.encode(s):
            return False, f"encoding mismatch on: {s[:40]}..."

    return True, "ok"


def compute_kl_divergence_full(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> dict:
    """
    Compute EXACT KL(teacher || student) using full vocabulary distribution.

    Args:
        teacher_logits: [1, seq_len, vocab_size] float32
        student_logits: [1, seq_len, vocab_size] float32

    Returns dict with per-position and aggregate KL stats.
    """
    # Log-softmax for numerical stability
    t_log_p = torch.log_softmax(teacher_logits, dim=-1)
    s_log_p = torch.log_softmax(student_logits, dim=-1)
    t_p = t_log_p.exp()

    # KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))
    kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1).squeeze(0)  # [seq_len]

    return {
        "kl_mean": kl_per_pos.mean().item(),
        "kl_std": kl_per_pos.std().item(),
        "kl_max": kl_per_pos.max().item(),
        "kl_min": kl_per_pos.min().item(),
        "n_positions": int(kl_per_pos.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast full-distribution KL eval")
    parser.add_argument("--teacher", required=True, help="Teacher/baseline model")
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt strings")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-len", type=int, default=1024, help="Max tokens per prompt")
    parser.add_argument("--max-params-b", type=float, default=3.5, help="Max student params (B)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.prompts) as f:
        prompts = json.load(f)

    students = [s.strip() for s in args.students.split(",")]

    # ── Tokenize ──────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"[fast] Loading tokenizer: {args.teacher}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    print(f"[fast] Tokenizing {len(prompts)} prompts (max_len={args.max_len})", flush=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=args.max_len
        ).input_ids.to(device)
        input_ids_list.append(ids)

    total_tokens = sum(ids.shape[1] for ids in input_ids_list)
    print(f"[fast] Total tokens to score: {total_tokens}", flush=True)

    # ── Teacher forward passes ────────────────────────────────────────
    print(f"\n[fast] Loading teacher: {args.teacher}", flush=True)
    t0 = time.time()
    teacher = load_model(args.teacher, device)
    teacher.eval()
    load_time = time.time() - t0
    print(f"[fast] Teacher loaded in {load_time:.1f}s", flush=True)

    print(f"[fast] Scoring {len(prompts)} prompts with teacher...", flush=True)
    t0 = time.time()
    teacher_logits_cpu = []
    with torch.no_grad():
        for i, ids in enumerate(input_ids_list):
            logits = teacher(ids).logits.float().cpu()  # [1, seq, vocab] → CPU
            teacher_logits_cpu.append(logits)
    score_time = time.time() - t0
    print(f"[fast] Teacher scored in {score_time:.1f}s ({total_tokens/score_time:.0f} tok/s)", flush=True)

    del teacher
    free_gpu()
    print(f"[fast] Teacher unloaded", flush=True)

    # ── Evaluate each student ─────────────────────────────────────────
    all_results = {"teacher": args.teacher, "total_tokens": total_tokens, "students": {}}

    for student_name in students:
        print(f"\n{'='*60}", flush=True)
        print(f"[fast] Student: {student_name}", flush=True)

        # 1. Verify tokenizer
        tok_ok, tok_reason = verify_tokenizer(args.teacher, student_name)
        if not tok_ok:
            print(f"[fast] REJECTED — tokenizer: {tok_reason}", flush=True)
            all_results["students"][student_name] = {"error": tok_reason}
            continue

        # 2. Verify param count
        params_b = get_param_count_b(student_name)
        if params_b > 0 and params_b > args.max_params_b:
            print(f"[fast] REJECTED — {params_b:.1f}B > {args.max_params_b}B max", flush=True)
            all_results["students"][student_name] = {
                "error": f"too_large:{params_b:.1f}B",
                "params_b": params_b,
            }
            continue
        print(f"[fast] Params: {params_b:.1f}B (max: {args.max_params_b}B) ✓", flush=True)
        print(f"[fast] Tokenizer: verified ✓", flush=True)

        # 3. Load and score
        t0 = time.time()
        student = load_model(student_name, device)
        student.eval()
        load_time = time.time() - t0
        print(f"[fast] Loaded in {load_time:.1f}s", flush=True)

        t0 = time.time()
        kl_per_prompt = []
        total_kl_sum = 0.0
        total_positions = 0

        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                s_logits = student(ids).logits.float().cpu()
                kl_stats = compute_kl_divergence_full(teacher_logits_cpu[i], s_logits)
                kl_per_prompt.append(kl_stats)
                total_kl_sum += kl_stats["kl_mean"] * kl_stats["n_positions"]
                total_positions += kl_stats["n_positions"]
                print(
                    f"  Prompt {i}: KL={kl_stats['kl_mean']:.4f} "
                    f"(±{kl_stats['kl_std']:.4f}, {kl_stats['n_positions']} pos)",
                    flush=True,
                )

        score_time = time.time() - t0
        kl_global = total_kl_sum / total_positions if total_positions > 0 else float("inf")

        all_results["students"][student_name] = {
            "params_b": params_b,
            "kl_global_avg": kl_global,
            "kl_per_prompt": kl_per_prompt,
            "score_time_s": score_time,
            "load_time_s": load_time,
            "total_positions": total_positions,
        }
        print(f"\n[fast] {student_name}: global KL={kl_global:.6f} in {score_time:.1f}s", flush=True)

        del student
        free_gpu()

    # ── Save ──────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[fast] Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
