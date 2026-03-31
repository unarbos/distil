#!/usr/bin/env python3
"""
vLLM-accelerated GPU evaluation script for SN97 validation (v2.0.0).

Architecture & VRAM timeline (B200 = 192GB):
  Phase 1 — Teacher generation via vLLM:
    [vLLM teacher ~70GB] → generate 60 continuations → kill server
    Time: ~3-5 min (vs 25 min with HF)

  Phase 2 — Teacher logit extraction via HF:
    [HF teacher ~67GB] → 60 forward passes (no autoregressive) → cache logits → unload
    Time: ~8-10 min (forward-only, ~3x faster than generate)

  Phase 3 — Student scoring:
    [teacher logits on CPU ~2GB] + [king ~8GB stays loaded] + [challenger ~8GB rotates]
    Total VRAM: ~18GB (king + challenger + overhead)
    Time: ~2-3 min per student

Optimizations over pod_eval.py:
  1. vLLM teacher generation: 5-10x faster than HF generate()
  2. King stays in VRAM: no download/load/cleanup between rounds (~3-5 min saved)
  3. Prefetch next student: download while current student scores
  4. Teacher unloaded after logits cached: frees ~67GB for student scoring
  5. Graceful fallback: if vLLM fails, falls back to pure HF path

Usage:
    python3 pod_eval_vllm.py \\
        --teacher Qwen/Qwen3.5-35B-A3B \\
        --students user/king,user/challenger1,user/challenger2 \\
        --prompts prompts.json \\
        --output results.json \\
        --king user/king
"""
import math
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc
import os
import sys
import shutil
import subprocess
import signal
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def compute_kl(teacher_logits, student_logits):
    """KL(teacher || student) per position."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(dim=-1)

def load_model(name, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    try:
        m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
        print(f"  [model] Loaded with flash_attention_2", flush=True)
        return m
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        print(f"  [model] Loaded with default attention", flush=True)
        return m

def prefetch_model(name):
    """Download model files to HF cache without loading to GPU. Runs in background."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(name, ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"])
        print(f"  [prefetch] {name} cached", flush=True)
    except Exception as e:
        print(f"  [prefetch] {name} failed: {e}", flush=True)

def clean_model_cache(name, teacher_name=None):
    """Remove HF cache for a model, preserving teacher cache."""
    try:
        cache_name = f"models--{name.replace('/', '--')}"
        if teacher_name:
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            if cache_name == teacher_cache:
                return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  [cleanup] Removed {cache_name}", flush=True)
    except Exception:
        pass

def disk_check_and_clean(teacher_name, threshold=85):
    """Check disk usage, clean non-teacher caches if above threshold."""
    try:
        st = os.statvfs("/")
        pct = int(100 * (1 - st.f_bavail / st.f_blocks))
        if pct > threshold:
            print(f"  [disk] {pct}% — cleaning non-teacher caches", flush=True)
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for d in cache_dir.iterdir():
                    if d.is_dir() and d.name.startswith("models--") and d.name != teacher_cache:
                        shutil.rmtree(d)
            st2 = os.statvfs("/")
            pct2 = int(100 * (1 - st2.f_bavail / st2.f_blocks))
            print(f"  [disk] After cleanup: {pct2}%", flush=True)
        return pct
    except Exception as e:
        print(f"  [disk] Check failed: {e}", flush=True)
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Server Management
# ═══════════════════════════════════════════════════════════════════════════════

VLLM_PORT = 9100
VLLM_URL = f"http://localhost:{VLLM_PORT}"

def start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=4096):
    """Start vLLM server via subprocess. Returns True on success."""
    print(f"\n[vllm] Starting server for {model_name}...", flush=True)
    stop_vllm_server()

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(VLLM_PORT),
        "--served-model-name", "teacher",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-prefix-caching",
        "--disable-log-requests",
    ]

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        n = torch.cuda.device_count()
        cmd.extend(["--tensor-parallel-size", str(n)])
        print(f"[vllm] Tensor parallelism: {n} GPUs", flush=True)

    log_f = open("/tmp/vllm_teacher.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    Path("/tmp/vllm_teacher.pid").write_text(str(proc.pid))
    print(f"[vllm] PID: {proc.pid}", flush=True)

    import requests
    for elapsed in range(0, 300, 3):
        try:
            if requests.get(f"{VLLM_URL}/health", timeout=3).status_code == 200:
                print(f"[vllm] Ready in {elapsed}s", flush=True)
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"[vllm] Died with code {proc.returncode}", flush=True)
            try:
                print(Path("/tmp/vllm_teacher.log").read_text()[-1500:], flush=True)
            except Exception:
                pass
            return False
        time.sleep(3)

    print(f"[vllm] Timeout after 300s", flush=True)
    stop_vllm_server()
    return False


def stop_vllm_server():
    """Kill vLLM server and free VRAM."""
    pid_file = Path("/tmp/vllm_teacher.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except ProcessLookupError:
                    break
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)

    # Belt and suspenders
    try:
        subprocess.run(["fuser", "-k", f"{VLLM_PORT}/tcp"],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    free_gpu()
    time.sleep(2)


def generate_via_vllm(prompts, tokenizer, max_new_tokens, block_seed=None):
    """Generate teacher continuations via vLLM API. Returns list of dicts."""
    import requests

    results = []
    for idx, prompt_text in enumerate(prompts):
        payload = {
            "model": "teacher",
            "prompt": prompt_text,
            "max_tokens": max_new_tokens,
            "temperature": 0.7 if block_seed is not None else 0.0,
            "top_p": 0.9 if block_seed is not None else 1.0,
        }
        if block_seed is not None:
            payload["seed"] = block_seed + idx

        for attempt in range(3):
            try:
                resp = requests.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                cont_text = data["choices"][0]["text"]
                full_text = prompt_text + cont_text
                full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
                prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids
                results.append({
                    "full_ids": full_ids,
                    "prompt_len": prompt_ids.shape[1],
                    "gen_len": full_ids.shape[1] - prompt_ids.shape[1],
                })
                if idx % 10 == 0 or idx == len(prompts) - 1:
                    print(f"  [{idx+1}/{len(prompts)}] {prompt_ids.shape[1]}+{full_ids.shape[1]-prompt_ids.shape[1]} tokens", flush=True)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"vLLM generation failed for prompt {idx}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated SN97 evaluation v2")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt texts")
    parser.add_argument("--output", default="/home/eval_results.json")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-params-b", type=float, default=36.0)
    parser.add_argument("--block-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-logits", default="/home/teacher_cache.pt")
    parser.add_argument("--save-teacher-logits", default=None)
    parser.add_argument("--king", default=None, help="King model name — stays in VRAM between rounds")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use pure HF")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    timings = {}

    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]

    print(f"[eval] {len(prompts)} prompts (hash={prompts_hash}), {len(students)} students", flush=True)
    print(f"[eval] Teacher: {args.teacher}", flush=True)
    print(f"[eval] King: {args.king or 'none'}", flush=True)
    print(f"[eval] vLLM: {'disabled' if args.no_vllm else 'enabled'}", flush=True)
    print(f"[eval] VRAM: {gpu_mem_str()}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).input_ids.to(device)
        input_ids_list.append(ids)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher logits
    # ═══════════════════════════════════════════════════════════════════

    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    teacher_cache_loaded = False

    # Try cache
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        try:
            t0 = time.time()
            cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
            if (len(cache.get("full_sequences", [])) == len(prompts)
                and cache.get("prompts_hash") == prompts_hash):
                full_sequences = [s.to(device) for s in cache["full_sequences"]]
                teacher_logits_list = cache["teacher_logits"]  # keep on CPU
                prompt_lens = cache["prompt_lens"]
                timings["teacher_cache_load"] = time.time() - t0
                timings["teacher_generation"] = 0.0
                timings["teacher_logits_pass"] = 0.0
                print(f"[eval] ✓ Cached logits ({timings['teacher_cache_load']:.1f}s, "
                      f"method={cache.get('generation_method', '?')})", flush=True)
                teacher_cache_loaded = True
            else:
                print(f"[eval] ✗ Cache stale — regenerating", flush=True)
        except Exception as e:
            print(f"[eval] ✗ Cache failed: {e}", flush=True)

    if not teacher_cache_loaded and not args.no_vllm:
        # ── vLLM generation ──
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1a: vLLM teacher generation", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        vllm_ok = start_vllm_server(args.teacher, args.vllm_gpu_util, args.vllm_max_model_len)
        timings["vllm_startup"] = time.time() - t0

        sequences_data = None
        if vllm_ok:
            t0 = time.time()
            try:
                sequences_data = generate_via_vllm(prompts, tokenizer, args.max_new_tokens, args.block_seed)
                timings["vllm_generation"] = time.time() - t0
                print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] vLLM generation failed: {e} — falling back to HF", flush=True)
            stop_vllm_server()
        else:
            print(f"[eval] vLLM failed to start — falling back to HF", flush=True)

        if sequences_data:
            # ── HF forward pass for logits ──
            print(f"\n{'='*60}", flush=True)
            print(f"PHASE 1b: HF teacher logit extraction", flush=True)
            print(f"{'='*60}", flush=True)

            t0 = time.time()
            teacher = load_model(args.teacher, device)
            teacher.eval()
            timings["teacher_hf_load"] = time.time() - t0
            print(f"[eval] HF teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

            # Progress reporting for teacher logits
            progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")
            def _write_teacher_progress(done, total):
                try:
                    with open(progress_path, "w") as pf:
                        json.dump({
                            "phase": "teacher_logits",
                            "students": students,
                            "students_total": len(students),
                            "prompts_total": total,
                            "teacher_prompts_done": done,
                            "completed": [],
                            "current": None,
                        }, pf)
                except Exception:
                    pass

            _write_teacher_progress(0, len(sequences_data))
            t0 = time.time()
            with torch.no_grad():
                for i, data in enumerate(sequences_data):
                    full_ids = data["full_ids"].to(device)
                    prompt_len = data["prompt_len"]
                    prompt_lens.append(prompt_len)
                    full_sequences.append(full_ids)
                    logits = teacher(full_ids).logits.float()
                    cont_logits = logits[:, prompt_len - 1:-1, :]
                    teacher_logits_list.append(cont_logits.cpu())
                    del logits, cont_logits
                    if (i + 1) % 10 == 0 or i == len(sequences_data) - 1:
                        print(f"  Logits [{i+1}/{len(sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
                    _write_teacher_progress(i + 1, len(sequences_data))

            timings["teacher_logits_pass"] = time.time() - t0
            print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)
            del sequences_data

            # Save cache
            cache_path = args.save_teacher_logits or os.path.join(
                os.path.dirname(args.output), "teacher_cache.pt")
            torch.save({
                "full_sequences": [s.cpu() for s in full_sequences],
                "teacher_logits": teacher_logits_list,
                "prompt_lens": prompt_lens,
                "block_seed": args.block_seed,
                "prompts_hash": prompts_hash,
                "generation_method": "vllm+hf",
            }, cache_path)
            print(f"[eval] Cache saved to {cache_path}", flush=True)

            # Unload teacher — free ~67GB VRAM for students
            del teacher
            free_gpu()
            print(f"[eval] Teacher unloaded. VRAM: {gpu_mem_str()}", flush=True)
            teacher_cache_loaded = True

    if not teacher_cache_loaded:
        # ── Pure HF fallback ──
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1 FALLBACK: HF teacher generation", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        teacher = load_model(args.teacher, device)
        teacher.eval()
        timings["teacher_hf_load"] = time.time() - t0
        print(f"[eval] Teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")
        def _write_teacher_progress(done, total):
            try:
                with open(progress_path, "w") as pf:
                    json.dump({
                        "phase": "teacher_generation",
                        "students": students,
                        "students_total": len(students),
                        "prompts_total": total,
                        "teacher_prompts_done": done,
                        "completed": [],
                        "current": None,
                    }, pf)
            except Exception:
                pass

        _write_teacher_progress(0, len(input_ids_list))
        t0 = time.time()
        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                prompt_len = ids.shape[1]
                prompt_lens.append(prompt_len)
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
                del logits, cont_logits
                gen_len = output_ids.shape[1] - prompt_len
                print(f"  Prompt {i}: {prompt_len}+{gen_len} tokens, VRAM: {gpu_mem_str()}", flush=True)
                _write_teacher_progress(i + 1, len(input_ids_list))

        timings["teacher_generation"] = time.time() - t0
        cache_path = args.save_teacher_logits or os.path.join(
            os.path.dirname(args.output), "teacher_cache.pt")
        torch.save({
            "full_sequences": [s.cpu() for s in full_sequences],
            "teacher_logits": teacher_logits_list,
            "prompt_lens": prompt_lens,
            "block_seed": args.block_seed,
            "prompts_hash": prompts_hash,
            "generation_method": "hf",
        }, cache_path)
        del teacher
        free_gpu()
        print(f"[eval] HF generation done in {timings['teacher_generation']:.1f}s, teacher unloaded", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Student scoring
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: Student scoring ({len(students)} models)", flush=True)
    print(f"{'='*60}", flush=True)

    # Resume support
    prior_results = {}
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prior = json.load(f)
            prior_results = prior.get("students", {})
            scored = [n for n, d in prior_results.items()
                      if d.get("status") != "load_failed" and d.get("kl_global_avg") is not None]
            if scored:
                print(f"[eval] Resuming: {len(scored)} already scored", flush=True)
        except Exception:
            pass

    results = {
        "teacher": args.teacher,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_len": args.max_prompt_len,
        "block_seed": args.block_seed,
        "n_prompts": len(prompts),
        "students": {},
    }
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Progress
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")
    progress_lock = threading.Lock()
    live_progress = {
        "phase": "scoring",
        "students": students,
        "students_total": len(students),
        "prompts_total": len(prompts),
        "completed": [],
        "current": None,
    }
    def _write_progress():
        try:
            with progress_lock:
                with open(progress_path, "w") as pf:
                    json.dump(live_progress, pf)
        except Exception:
            pass
    _write_progress()

    # Early stopping
    best_kl_so_far = None
    best_kl_per_prompt_cumulative = None
    MIN_PROMPTS_EARLY_STOP = 7
    PER_MODEL_TIMEOUT = 600

    # ── King stays in VRAM ──
    # If --king is set, load it once and keep it loaded for all rounds.
    # The king is scored first (sets best_kl_so_far for early stopping),
    # then stays loaded while challengers rotate through.
    king_model = None
    king_name = args.king

    # Prefetch executor for downloading next student while scoring current
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None

    # VRAM baseline (no student loaded yet)
    vram_before_students = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for student_idx, student_name in enumerate(students):
        # Skip already scored
        if student_name in results["students"]:
            prior = results["students"][student_name]
            kl = prior.get("kl_global_avg")
            print(f"\n[eval] {student_name}: SKIP (already scored, KL={kl})", flush=True)
            # Update early stopping from resumed
            if kl and kl > 0.001 and kl < float('inf'):
                if best_kl_so_far is None or kl < best_kl_so_far:
                    best_kl_so_far = kl
                    kl_per_prompt = prior.get("kl_per_prompt", [])
                    if kl_per_prompt:
                        best_kl_per_prompt_cumulative = []
                        s = 0.0
                        for j, v in enumerate(kl_per_prompt):
                            s += v
                            best_kl_per_prompt_cumulative.append(s / (j + 1))
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Student: {student_name}" +
              (" (KING — stays in VRAM)" if student_name == king_name else ""), flush=True)

        model_start = time.time()
        disk_check_and_clean(args.teacher)

        # ── Prefetch next student while we score this one ──
        if student_idx + 1 < len(students):
            next_name = students[student_idx + 1]
            if next_name not in results["students"] and next_name != king_name:
                prefetch_future = prefetch_executor.submit(prefetch_model, next_name)

        # ── Load student (or reuse king) ──
        live_progress["current"] = {"student_name": student_name, "prompts_done": 0}
        _write_progress()

        is_king = (student_name == king_name)
        student = None

        if is_king and king_model is not None:
            # Reuse king already in VRAM
            student = king_model
            load_time = 0.0
            student_vram_gb = 0.0  # already accounted for
            print(f"[eval] King reused from VRAM", flush=True)
        else:
            try:
                t0 = time.time()
                student = load_model(student_name, device)
                student.eval()
                load_time = time.time() - t0
                student_vram_gb = (torch.cuda.memory_allocated() - vram_before_students) / 1024**3
                print(f"[eval] Loaded in {load_time:.1f}s, student VRAM: {student_vram_gb:.1f}GB, total: {gpu_mem_str()}", flush=True)
            except Exception as e:
                print(f"[eval] FAILED to load: {e}", flush=True)
                results["students"][student_name] = {
                    "status": "load_failed", "error": str(e)[:500], "kl_global_avg": None}
                results["timings"] = {k: round(v, 1) for k, v in timings.items()}
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                live_progress["completed"].append({"student_name": student_name, "status": "load_failed"})
                live_progress["current"] = None
                _write_progress()
                try: del student
                except: pass
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            # VRAM fraud check (only for non-king, since king was checked on prior load)
            MAX_STUDENT_VRAM_GB = 20.0
            if not is_king and student_vram_gb > MAX_STUDENT_VRAM_GB:
                msg = f"FRAUD: student VRAM delta {student_vram_gb:.1f}GB > {MAX_STUDENT_VRAM_GB}GB"
                print(f"  ⚠️ {msg}", flush=True)
                results["students"][student_name] = {
                    "status": "fraud_vram", "reason": msg,
                    "vram_gb": round(student_vram_gb, 1), "kl_global_avg": float('inf')}
                del student
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            # If this is king, keep reference
            if is_king:
                king_model = student
                print(f"[eval] King loaded — will stay in VRAM", flush=True)

        # ── Score: per-prompt sequential with early stopping ──
        can_early_stop = (student_idx > 0) and (best_kl_so_far is not None)
        kl_per_prompt = []
        prompt_kl_means = []
        scoring_error = None
        early_stopped = False

        t0 = time.time()
        with torch.no_grad():
            for i in range(len(prompts)):
                try:
                    full_seq = full_sequences[i]
                    prompt_len = prompt_lens[i]
                    t_logits = teacher_logits_list[i].to(device)
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]
                    min_len = min(cont_s.shape[1], t_logits.shape[1])
                    kl_per_pos = compute_kl(t_logits[:, :min_len, :], cont_s[:, :min_len, :]).squeeze(0)
                    kl_mean = kl_per_pos.mean().item()
                    del s_logits, cont_s, t_logits, kl_per_pos

                    if math.isnan(kl_mean) or math.isinf(kl_mean):
                        print(f"  [prompt {i}] KL={kl_mean} — invalid, stopping", flush=True)
                        scoring_error = f"NaN/Inf KL at prompt {i}"
                        break

                    kl_per_prompt.append({"mean": round(kl_mean, 6)})
                    prompt_kl_means.append(kl_mean)

                    running_mean = sum(prompt_kl_means) / len(prompt_kl_means)
                    live_progress["current"] = {
                        "student_name": student_name,
                        "prompts_done": i + 1,
                        "kl_running_mean": round(running_mean, 6),
                        "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    }
                    _write_progress()

                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{len(prompts)}] KL={kl_mean:.6f} (avg: {running_mean:.6f})", flush=True)

                except RuntimeError as e:
                    scoring_error = str(e)
                    if "out of memory" not in str(e).lower():
                        print(f"  [prompt {i}] RuntimeError: {e}", flush=True)
                    else:
                        print(f"  [prompt {i}] OOM", flush=True)
                    free_gpu()
                    break
                except Exception as e:
                    scoring_error = str(e)
                    print(f"  [prompt {i}] Error: {e}", flush=True)
                    free_gpu()
                    break

                # Early stopping (same-point comparison)
                n = len(prompt_kl_means)
                if can_early_stop and n >= MIN_PROMPTS_EARLY_STOP:
                    running_mean = sum(prompt_kl_means) / n
                    running_var = sum((x - running_mean) ** 2 for x in prompt_kl_means) / (n - 1)
                    running_se = math.sqrt(running_var / n)
                    student_lower = running_mean - 1.96 * running_se

                    if best_kl_per_prompt_cumulative and n <= len(best_kl_per_prompt_cumulative):
                        best_at_n = best_kl_per_prompt_cumulative[n - 1]
                    else:
                        best_at_n = best_kl_so_far
                    if best_at_n and best_at_n <= 0.001:
                        best_at_n = best_kl_so_far if best_kl_so_far and best_kl_so_far > 0.001 else float('inf')

                    if student_lower > best_at_n:
                        print(f"  [early stop] prompt {n}: CI lower {student_lower:.6f} > best@{n} {best_at_n:.6f}", flush=True)
                        early_stopped = True
                        break

                if time.time() - model_start > PER_MODEL_TIMEOUT:
                    print(f"  [timeout] {PER_MODEL_TIMEOUT}s", flush=True)
                    early_stopped = True
                    break

        scoring_time = time.time() - t0

        # Record results
        if scoring_error and not kl_per_prompt:
            results["students"][student_name] = {
                "status": "scoring_error", "error": scoring_error[:500], "kl_global_avg": None}
        elif kl_per_prompt:
            kl_avg = sum(d["mean"] for d in kl_per_prompt) / len(kl_per_prompt)
            n_scored = len(kl_per_prompt)
            status = "early_stopped" if early_stopped else ("partial" if scoring_error else "scored")
            print(f"  → KL={kl_avg:.6f} ({n_scored}/{len(prompts)} prompts, {status})", flush=True)
            results["students"][student_name] = {
                "status": status,
                "kl_global_avg": round(kl_avg, 6),
                "kl_per_prompt": [d["mean"] for d in kl_per_prompt],
                "prompts_scored": n_scored,
                "scoring_time": round(scoring_time, 1),
                "load_time": round(load_time, 1),
                "early_stopped": early_stopped,
            }
            # Update early stopping baseline
            if kl_avg > 0.001 and not early_stopped and not scoring_error:
                if best_kl_so_far is None or kl_avg < best_kl_so_far:
                    best_kl_so_far = kl_avg
                    best_kl_per_prompt_cumulative = []
                    s = 0.0
                    for j, d in enumerate(kl_per_prompt):
                        s += d["mean"]
                        best_kl_per_prompt_cumulative.append(s / (j + 1))
                    print(f"  → New best: KL={kl_avg:.6f}", flush=True)

        # Save incremental
        results["timings"] = {k: round(v, 1) for k, v in timings.items()}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        live_progress["completed"].append({
            "student_name": student_name,
            "status": results["students"].get(student_name, {}).get("status", "unknown"),
            "kl": results["students"].get(student_name, {}).get("kl_global_avg"),
            "prompts_scored": len(kl_per_prompt),
        })
        live_progress["current"] = None
        _write_progress()

        # Cleanup — DON'T unload king
        if not is_king:
            del student
            free_gpu()
            clean_model_cache(student_name, args.teacher)
        else:
            # King stays loaded — just clear KV cache
            torch.cuda.empty_cache()

        # Wait for prefetch if needed
        if prefetch_future:
            try:
                prefetch_future.result(timeout=1)
            except Exception:
                pass
            prefetch_future = None

    # Final save
    results["timings"] = {k: round(v, 1) for k, v in timings.items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] DONE — {len(results['students'])} students", flush=True)
    for k, v in sorted(timings.items()):
        print(f"  {k}: {v:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    prefetch_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
