#!/usr/bin/env python3
"""
Standalone GPU evaluation script for testing on remote pods.

Uses the same KL computation as the validator but runs independently.
Upload to a GPU pod, provide prompts, get KL scores.

Usage:
    python3 pod_eval.py \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --students student1/model,student2/model \
        --prompts prompts.json \
        --output results.json \
        --max-prompt-len 1024 \
        --max-new-tokens 512
"""
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc


def load_model(name, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    kwargs = dict(torch_dtype=dtype, device_map=device, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_kl(teacher_logits, student_logits):
    """KL(teacher || student) from logit tensors. Returns per-position KL."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description="GPU KL evaluation with teacher continuation")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt strings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-params-b", type=float, default=3.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.prompts) as f:
        prompts = json.load(f)
    students = [s.strip() for s in args.students.split(",")]

    from transformers import AutoTokenizer
    print(f"[eval] Loading tokenizer: {args.teacher}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    # Tokenize prompts
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).input_ids.to(device)
        input_ids_list.append(ids)
    total_prompt_tokens = sum(ids.shape[1] for ids in input_ids_list)
    print(f"[eval] {len(prompts)} prompts, {total_prompt_tokens} prompt tokens", flush=True)

    # Load teacher
    print(f"\n[eval] Loading teacher: {args.teacher}", flush=True)
    teacher = load_model(args.teacher, device)
    teacher.eval()

    # Generate teacher continuations + get teacher logits
    print(f"[eval] Generating teacher continuations (max_new_tokens={args.max_new_tokens})...", flush=True)
    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []

    with torch.no_grad():
        for i, ids in enumerate(input_ids_list):
            prompt_len = ids.shape[1]
            prompt_lens.append(prompt_len)

            # Generate continuation
            output_ids = teacher.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            full_sequences.append(output_ids)

            # Forward pass on full sequence for logits
            logits = teacher(output_ids).logits.float()
            # Keep only continuation logits: logits[prompt_len-1:-1] predicts continuation tokens
            cont_logits = logits[:, prompt_len - 1:-1, :]
            teacher_logits_list.append(cont_logits.cpu())

            gen_len = output_ids.shape[1] - prompt_len
            print(f"  Prompt {i}: {prompt_len} prompt + {gen_len} gen tokens", flush=True)

    del teacher
    free_gpu()
    print("[eval] Teacher unloaded", flush=True)

    # Evaluate students
    results = {"teacher": args.teacher, "students": {}}

    for student_name in students:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[eval] Student: {student_name}", flush=True)

        t0 = time.time()
        student = load_model(student_name, device)
        student.eval()
        print(f"[eval] Loaded in {time.time() - t0:.1f}s", flush=True)

        kl_per_prompt = []
        total_kl_sum = 0.0
        total_positions = 0

        with torch.no_grad():
            for i, full_ids in enumerate(full_sequences):
                s_logits = student(full_ids).logits.float()
                cont_s_logits = s_logits[:, prompt_lens[i] - 1:-1, :]
                t_logits = teacher_logits_list[i].to(device)

                kl_per_pos = compute_kl(t_logits, cont_s_logits).squeeze(0)
                n_pos = kl_per_pos.shape[0]
                kl_mean = kl_per_pos.mean().item()

                kl_per_prompt.append({
                    "kl_mean": kl_mean,
                    "kl_std": kl_per_pos.std().item(),
                    "n_positions": n_pos,
                })
                total_kl_sum += kl_mean * n_pos
                total_positions += n_pos
                print(f"  Prompt {i}: KL={kl_mean:.4f} ({n_pos} continuation positions)", flush=True)

        kl_global = total_kl_sum / total_positions if total_positions > 0 else float("inf")
        results["students"][student_name] = {
            "kl_global_avg": kl_global,
            "kl_per_prompt": kl_per_prompt,
            "total_positions": total_positions,
        }
        print(f"\n[eval] {student_name}: global KL={kl_global:.6f}", flush=True)

        del student
        free_gpu()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
