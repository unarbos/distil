#!/usr/bin/env python3
"""
Fast prompt-only KL evaluation (no teacher continuation).

Simpler/faster than pod_eval.py — forward passes only, no generation.
Useful for quick A/B comparisons but less thorough than the continuation approach.

For production evaluation, use pod_eval.py which includes teacher continuation.
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


def main():
    parser = argparse.ArgumentParser(description="Fast prompt-only KL eval")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt strings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.prompts) as f:
        prompts = json.load(f)
    students = [s.strip() for s in args.students.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(device)
        input_ids_list.append(ids)

    # Teacher forward passes
    print(f"[fast] Loading teacher: {args.teacher}", flush=True)
    teacher = load_model(args.teacher, device)
    teacher.eval()

    teacher_logits_cpu = []
    with torch.no_grad():
        for ids in input_ids_list:
            logits = teacher(ids).logits.float().cpu()
            teacher_logits_cpu.append(logits)

    del teacher
    free_gpu()

    results = {"teacher": args.teacher, "students": {}}

    for student_name in students:
        print(f"\n[fast] Student: {student_name}", flush=True)
        student = load_model(student_name, device)
        student.eval()

        kl_per_prompt = []
        total_kl_sum = 0.0
        total_positions = 0

        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                s_logits = student(ids).logits.float().cpu()
                t_log_p = F.log_softmax(teacher_logits_cpu[i], dim=-1)
                s_log_p = F.log_softmax(s_logits, dim=-1)
                t_p = t_log_p.exp()
                kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1).squeeze(0)

                kl_mean = kl_per_pos.mean().item()
                n_pos = kl_per_pos.shape[0]
                kl_per_prompt.append({"kl_mean": kl_mean, "n_positions": n_pos})
                total_kl_sum += kl_mean * n_pos
                total_positions += n_pos

        kl_global = total_kl_sum / total_positions if total_positions > 0 else float("inf")
        results["students"][student_name] = {
            "kl_global_avg": kl_global,
            "kl_per_prompt": kl_per_prompt,
            "total_positions": total_positions,
        }
        print(f"[fast] {student_name}: global KL={kl_global:.6f}", flush=True)

        del student
        free_gpu()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[fast] Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
