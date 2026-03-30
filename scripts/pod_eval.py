#!/usr/bin/env python3
"""
Standalone GPU evaluation script for testing on remote pods (v0.8.0).

Uses the same KL computation as the validator but runs independently.
Supports batch scoring, sequential (memory-efficient) mode, and detailed metrics.

Usage:
    python3 pod_eval.py \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --students Qwen/Qwen3.5-35B-A3B-FP8,Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
        --prompts prompts.json \
        --output results.json \
        --max-prompt-len 1024 \
        --max-new-tokens 512 \
        --max-params-b 36.0
"""
import math
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc
import os


def gpu_mem_mb():
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def gpu_mem_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"


def load_model(name, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    # NEVER trust_remote_code for student models — only teacher needs it (Qwen custom tokenizer)
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(torch_dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    try:
        m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
        print(f"  [model] Loaded with flash_attention_2", flush=True)
        return m
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        print(f"  [model] Loaded with default attention", flush=True)
        return m


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


def verify_tokenizer(teacher_name, student_name):
    """Check student uses same tokenizer as teacher."""
    from transformers import AutoTokenizer
    t_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    # NEVER trust_remote_code for students — blocks custom tokenizer.py exploits
    s_tok = AutoTokenizer.from_pretrained(student_name, trust_remote_code=False)

    if t_tok.vocab_size != s_tok.vocab_size:
        return False, f"vocab_size mismatch: {s_tok.vocab_size} vs {t_tok.vocab_size}"

    test_strings = [
        "def fibonacci(n):\n    if n <= 1: return n",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nclass Model(nn.Module):",
    ]
    for s in test_strings:
        if t_tok.encode(s) != s_tok.encode(s):
            return False, f"encoding mismatch on: {s[:40]}..."
    return True, "ok"


def batch_forward_kl(model, sequences, prompt_lens, teacher_logits_list, device, batch_label=""):
    """
    Batch forward pass for scoring. Pads sequences, runs single forward pass,
    computes per-prompt KL against cached teacher logits.

    Returns list of per-prompt dicts with kl_mean, kl_std, n_positions.
    """
    # Pad sequences to same length
    max_len = max(seq.shape[1] for seq in sequences)
    padded = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype, device=device)
    attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[1]
        padded[i, :seq_len] = seq[0]
        attention_mask[i, :seq_len] = 1

    # Single batched forward pass
    t0 = time.time()
    with torch.no_grad():
        outputs = model(padded, attention_mask=attention_mask)
    s_logits_all = outputs.logits.float()
    fwd_time = time.time() - t0
    if batch_label:
        print(f"  [batch] {batch_label} forward: {fwd_time:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

    # Per-prompt KL computation
    results = []
    for i in range(len(sequences)):
        seq_len = sequences[i].shape[1]
        pl = prompt_lens[i]
        # Extract continuation logits for this sequence
        cont_s = s_logits_all[i, pl - 1:seq_len - 1, :].unsqueeze(0)
        t_logits = teacher_logits_list[i].to(device)

        # Ensure matching dimensions
        min_len = min(cont_s.shape[1], t_logits.shape[1])
        kl_per_pos = compute_kl(t_logits[:, :min_len, :], cont_s[:, :min_len, :]).squeeze(0)
        n_pos = kl_per_pos.shape[0]
        results.append({
            "kl_mean": kl_per_pos.mean().item(),
            "kl_std": kl_per_pos.std().item() if n_pos > 1 else 0.0,
            "kl_max": kl_per_pos.max().item(),
            "kl_min": kl_per_pos.min().item(),
            "n_positions": n_pos,
        })
    return results, fwd_time


def main():
    parser = argparse.ArgumentParser(description="GPU KL evaluation with teacher continuation")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt strings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-params-b", type=float, default=3.5)
    parser.add_argument("--block-seed", type=int, default=None,
                        help="Block seed for teacher sampling (enables temperature=0.7 sampling)")
    parser.add_argument("--sequential", action="store_true",
                        help="Memory-efficient mode: unload teacher before loading students")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Specific GPU index to use (e.g. 0 or 1). Default: auto-select.")
    parser.add_argument("--teacher-logits", type=str, default=None,
                        help="Path to pre-cached teacher logits (.pt file). Skips teacher inference.")
    parser.add_argument("--save-teacher-logits", type=str, default=None,
                        help="Save teacher logits to this path after generation.")
    args = parser.parse_args()

    total_start = time.time()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
        print(f"[eval] Pinned to GPU {args.gpu}", flush=True)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    timings = {}

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

    # Tokenizer verification for each student
    for student_name in students:
        print(f"[eval] Verifying tokenizer: {student_name}...", flush=True)
        tok_ok, tok_reason = verify_tokenizer(args.teacher, student_name)
        if not tok_ok:
            print(f"[FAIL] Tokenizer mismatch for {student_name}: {tok_reason}", flush=True)
            return
        print(f"  ✓ Tokenizer matches teacher", flush=True)

    # Load teacher logits — either from cache or by running inference
    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []

    if args.teacher_logits and os.path.exists(args.teacher_logits):
        # Load pre-cached teacher logits (for parallel GPU eval)
        print(f"\n[eval] Loading cached teacher logits from {args.teacher_logits}", flush=True)
        t0 = time.time()
        cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=True)
        full_sequences = [s.to(device) for s in cache["full_sequences"]]
        teacher_logits_list = cache["teacher_logits"]  # keep on CPU
        prompt_lens = cache["prompt_lens"]
        timings["teacher_load"] = time.time() - t0
        print(f"[eval] Loaded cached logits in {timings['teacher_load']:.1f}s ({len(full_sequences)} prompts)", flush=True)
        timings["teacher_generation"] = 0.0
    else:
        # Run teacher inference
        print(f"\n[eval] Loading teacher: {args.teacher}", flush=True)
        t0 = time.time()
        teacher = load_model(args.teacher, device)
        teacher.eval()
        timings["teacher_load"] = time.time() - t0
        print(f"[eval] Teacher loaded in {timings['teacher_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        # Generate teacher continuations + get teacher logits
        print(f"\n[eval] Generating teacher continuations (max_new_tokens={args.max_new_tokens})...", flush=True)

        t0 = time.time()
        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                prompt_len = ids.shape[1]
                prompt_lens.append(prompt_len)

                # Generate continuation (block-seeded sampling if specified)
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True)
                if args.block_seed is not None:
                    torch.manual_seed(args.block_seed + i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(args.block_seed + i)
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    gen_kwargs.update(do_sample=False)

                output_ids = teacher.generate(ids, **gen_kwargs)
                full_sequences.append(output_ids)

                logits = teacher(output_ids).logits.float()
                cont_logits = logits[:, prompt_len - 1:-1, :]
                teacher_logits_list.append(cont_logits.cpu())

                gen_len = output_ids.shape[1] - prompt_len
                print(f"  Prompt {i}: {prompt_len} prompt + {gen_len} gen tokens, VRAM: {gpu_mem_str()}", flush=True)

        timings["teacher_generation"] = time.time() - t0
        print(f"[eval] Generation complete in {timings['teacher_generation']:.1f}s", flush=True)

        # Save teacher logits if requested (for parallel GPU eval)
        if args.save_teacher_logits:
            print(f"[eval] Saving teacher logits to {args.save_teacher_logits}", flush=True)
            torch.save({
                "full_sequences": [s.cpu() for s in full_sequences],
                "teacher_logits": teacher_logits_list,
                "prompt_lens": prompt_lens,
            }, args.save_teacher_logits)

        del teacher
        free_gpu()
        print(f"[eval] Teacher unloaded, VRAM: {gpu_mem_str()}", flush=True)

    # Evaluate students
    results = {
        "teacher": args.teacher,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_len": args.max_prompt_len,
        "block_seed": args.block_seed,
        "n_prompts": len(prompts),
        "students": {},
    }

    # Live progress file — written after every prompt for real-time dashboard updates
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")
    live_progress = {
        "phase": "scoring",
        "students": students,
        "students_total": len(students),
        "prompts_total": len(prompts),
        "completed": [],  # finished student results for dashboard
        "current": None,  # currently scoring student
    }

    def _write_progress():
        try:
            with open(progress_path, "w") as pf:
                json.dump(live_progress, pf)
        except Exception:
            pass

    _write_progress()

    # Track best KL across students for early stopping
    best_kl_so_far = None  # best (lowest) global KL mean from fully-evaluated students
    MIN_PROMPTS_EARLY_STOP = 3  # minimum for non-degenerate variance estimate

    # Logit fingerprinting: detect functional copies even if hashes differ
    # Store per-position KL vectors from first 2 prompts per student
    FINGERPRINT_PROMPTS = 2
    FINGERPRINT_COSINE_THRESHOLD = 0.9999  # functional copy if above this
    logit_fingerprints: dict[str, torch.Tensor] = {}  # student_name -> flattened KL vector

    for student_idx, student_name in enumerate(students):
        print(f"\n{'=' * 60}", flush=True)
        print(f"[eval] Student: {student_name}", flush=True)

        # Update live progress: loading model
        live_progress["current"] = {
            "student_idx": student_idx,
            "student_name": student_name,
            "phase": "loading",
            "prompts_done": 0,
            "kl_running_mean": None,
            "kl_running_se": None,
            "ci_95": None,
        }
        _write_progress()

        t0 = time.time()
        student = load_model(student_name, device)
        student.eval()
        load_time = time.time() - t0
        print(f"[eval] Loaded in {load_time:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        # --- Per-prompt sequential scoring with early stopping ---
        can_early_stop = (student_idx > 0) and (best_kl_so_far is not None)
        kl_per_prompt = []
        early_stopped = False
        is_functional_copy = False
        copy_of = None
        t0 = time.time()

        # Collect per-prompt KL means for running stats
        prompt_kl_means = []  # list of per-prompt kl_mean values
        fingerprint_parts = []  # collect per-position KL vectors for fingerprinting

        with torch.no_grad():
            for i, full_ids in enumerate(full_sequences):
                try:
                    s_logits = student(full_ids).logits.float()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  [prompt {i}] OOM, clearing cache and retrying...", flush=True)
                        free_gpu()
                        s_logits = student(full_ids).logits.float()
                    else:
                        raise

                cont_s = s_logits[:, prompt_lens[i] - 1:-1, :]
                t_logits = teacher_logits_list[i].to(device)
                min_len = min(cont_s.shape[1], t_logits.shape[1])
                kl_per_pos = compute_kl(t_logits[:, :min_len, :], cont_s[:, :min_len, :]).squeeze(0)
                n_pos = kl_per_pos.shape[0]
                kl_mean_val = kl_per_pos.mean().item()

                kl_per_prompt.append({
                    "kl_mean": kl_mean_val,
                    "kl_std": kl_per_pos.std().item() if n_pos > 1 else 0.0,
                    "kl_max": kl_per_pos.max().item(),
                    "kl_min": kl_per_pos.min().item(),
                    "n_positions": n_pos,
                })
                prompt_kl_means.append(kl_mean_val)

                # Collect fingerprint from first N prompts
                if i < FINGERPRINT_PROMPTS:
                    fingerprint_parts.append(kl_per_pos.detach().cpu())

                # After fingerprint prompts, check for functional copy
                if i == FINGERPRINT_PROMPTS - 1 and logit_fingerprints:
                    fp = torch.cat(fingerprint_parts)
                    for prev_name, prev_fp in logit_fingerprints.items():
                        # Match lengths (use min)
                        fp_len = min(fp.shape[0], prev_fp.shape[0])
                        if fp_len < 10:
                            continue
                        cos = torch.nn.functional.cosine_similarity(
                            fp[:fp_len].unsqueeze(0),
                            prev_fp[:fp_len].unsqueeze(0),
                        ).item()
                        if cos > FINGERPRINT_COSINE_THRESHOLD:
                            is_functional_copy = True
                            copy_of = prev_name
                            print(
                                f"  [FUNCTIONAL COPY] {student_name} is a copy of {prev_name} "
                                f"(cosine={cos:.8f} after {FINGERPRINT_PROMPTS} prompts)",
                                flush=True,
                            )
                            break

                    if is_functional_copy:
                        break

                # Update live progress with running stats
                n = len(prompt_kl_means)
                running_mean = sum(prompt_kl_means) / n if n > 0 else 0
                running_se = 0
                ci_low, ci_high = running_mean, running_mean
                if n >= 2:
                    running_var = sum((x - running_mean) ** 2 for x in prompt_kl_means) / (n - 1)
                    running_se = math.sqrt(running_var / n)
                    ci_low = running_mean - 2 * running_se
                    ci_high = running_mean + 2 * running_se
                live_progress["current"] = {
                    "student_idx": student_idx,
                    "student_name": student_name,
                    "phase": "scoring",
                    "prompts_done": n,
                    "kl_running_mean": round(running_mean, 6),
                    "kl_running_se": round(running_se, 6) if running_se else None,
                    "ci_95": [round(ci_low, 6), round(ci_high, 6)] if n >= 2 else None,
                    "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far is not None else None,
                }
                _write_progress()

                # Free per-prompt GPU tensors
                del s_logits, cont_s, t_logits, kl_per_pos

                # Early stopping check
                n = len(prompt_kl_means)
                if can_early_stop and n >= MIN_PROMPTS_EARLY_STOP:
                    running_mean = sum(prompt_kl_means) / n
                    running_var = sum((x - running_mean) ** 2 for x in prompt_kl_means) / (n - 1)
                    running_se = math.sqrt(running_var / n)

                    # Student's lower 95% CI bound (best case for student)
                    student_lower = running_mean - 2 * running_se
                    # Best student's upper 95% CI bound (worst case for best)
                    best_upper = best_kl_so_far  # best_kl_so_far already stores the mean; use generous comparison

                    if student_lower > best_upper:
                        print(
                            f"  [early-stop] {student_name}: KL={running_mean:.6f} ± {running_se:.6f} "
                            f"after {n} prompts, best so far={best_kl_so_far:.6f}",
                            flush=True,
                        )
                        early_stopped = True
                        break

        scoring_time = time.time() - t0

        # Compute global average from scored prompts
        total_kl_sum = sum(r["kl_mean"] * r["n_positions"] for r in kl_per_prompt)
        total_positions = sum(r["n_positions"] for r in kl_per_prompt)
        kl_global = total_kl_sum / total_positions if total_positions > 0 else float("inf")

        # Per-prompt stats table
        print(f"\n  {'Prompt':>6} | {'KL Mean':>10} | {'KL Std':>10} | {'KL Max':>10} | {'KL Min':>10} | {'Positions':>10}", flush=True)
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}", flush=True)
        for i, r in enumerate(kl_per_prompt):
            print(f"  {i:>6} | {r['kl_mean']:>10.6f} | {r['kl_std']:>10.6f} | {r['kl_max']:>10.6f} | {r['kl_min']:>10.6f} | {r['n_positions']:>10}", flush=True)

        status_str = f" (early-stopped after {len(kl_per_prompt)}/{len(prompts)} prompts)" if early_stopped else ""
        print(f"\n  ═══ {student_name}: global KL = {kl_global:.6f}{status_str} ═══", flush=True)
        print(f"  Timings: load={load_time:.1f}s, scoring={scoring_time:.1f}s", flush=True)
        print(f"  VRAM after scoring: {gpu_mem_str()}", flush=True)

        # Benchmark generation speed (tokens/sec) — short autoregressive generation
        tokens_per_sec = None
        if not is_functional_copy:
            try:
                bench_ids = full_sequences[0][:, :64]  # first 64 tokens as prompt
                torch.cuda.synchronize()
                t_gen = time.time()
                gen_tokens = 128
                with torch.no_grad():
                    out = student.generate(
                        bench_ids,
                        max_new_tokens=gen_tokens,
                        do_sample=False,
                    )
                torch.cuda.synchronize()
                gen_time = time.time() - t_gen
                actual_new = out.shape[1] - bench_ids.shape[1]
                tokens_per_sec = round(actual_new / gen_time, 1)
                print(f"  Generation speed: {tokens_per_sec} tok/s ({actual_new} tokens in {gen_time:.2f}s)", flush=True)
            except Exception as gen_err:
                print(f"  Generation benchmark failed: {gen_err}", flush=True)

        student_result = {
            "kl_global_avg": kl_global,
            "kl_per_prompt": kl_per_prompt,
            "total_positions": total_positions,
            "load_time_s": round(load_time, 1),
            "scoring_time_s": round(scoring_time, 1),
            "tokens_per_sec": tokens_per_sec,
        }
        if is_functional_copy:
            student_result["functional_copy"] = True
            student_result["copy_of"] = copy_of
            student_result["prompts_scored"] = len(kl_per_prompt)
        elif early_stopped:
            student_result["early_stopped"] = True
            student_result["prompts_scored"] = len(kl_per_prompt)
        results["students"][student_name] = student_result

        # Store fingerprint for future comparisons (only if not a copy itself)
        if not is_functional_copy and fingerprint_parts:
            logit_fingerprints[student_name] = torch.cat(fingerprint_parts)

        # Update best KL tracker (only from fully evaluated, non-copy students)
        if not early_stopped and not is_functional_copy:
            if best_kl_so_far is None or kl_global < best_kl_so_far:
                best_kl_so_far = kl_global

        # Record completed student in live progress
        status = "scored"
        status_detail = f"KL={kl_global:.6f}"
        if is_functional_copy:
            status = "functional_copy"
            status_detail = f"copy of {copy_of}"
        elif early_stopped:
            status = "early_stopped"
            status_detail = f"KL={kl_global:.6f} after {len(kl_per_prompt)}/{len(prompts)} prompts"
        live_progress["completed"].append({
            "student_idx": student_idx,
            "student_name": student_name,
            "status": status,
            "status_detail": status_detail,
            "kl": round(kl_global, 6),
            "prompts_scored": len(kl_per_prompt),
            "prompts_total": len(prompts),
            "scoring_time_s": round(scoring_time, 1),
        })
        live_progress["current"] = None
        _write_progress()

        del student
        free_gpu()

    # Summary
    total_time = time.time() - total_start
    timings["total"] = total_time

    print(f"\n{'=' * 60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Teacher: {args.teacher}", flush=True)
    print(f"Prompts: {len(prompts)}, max_new_tokens: {args.max_new_tokens}", flush=True)
    print(f"Block seed: {args.block_seed}", flush=True)
    print(f"\nTimings:", flush=True)
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s", flush=True)

    print(f"\nResults:", flush=True)
    for name, data in results["students"].items():
        print(f"  {name}: KL = {data['kl_global_avg']:.6f}", flush=True)

    results["timings"] = {k: round(v, 1) for k, v in timings.items()}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
