#!/usr/bin/env python3
"""
vLLM-accelerated GPU evaluation script for SN97 validation (v1.0.0).

Architecture:
  1. Start vLLM server (nohup) for teacher generation — 5-10x faster than HF generate()
  2. Generate teacher continuations via vLLM OpenAI-compatible API
  3. Stop vLLM server (free VRAM)
  4. Load HF teacher for logit forward passes (one per prompt, cached)
  5. Score students against cached teacher logits (same as current flow)

The HF forward pass is unavoidable — we need full-vocabulary logit distributions
for accurate KL divergence, and vLLM's API only returns top-k logprobs.
But the forward pass is 3-5x faster than generate() because it's a single pass
with no autoregressive decoding.

Usage:
    python3 pod_eval_vllm.py \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --students user/model1,user/model2 \
        --prompts prompts.json \
        --output results.json
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


# ── Shared utilities (identical to pod_eval.py) ──

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

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
    """KL(teacher || student) from logit tensors. Returns per-position KL."""
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


# ── vLLM Server Management ──

VLLM_PORT = 9100
VLLM_URL = f"http://localhost:{VLLM_PORT}"
VLLM_PID_FILE = "/tmp/vllm_teacher.pid"
VLLM_LOG_FILE = "/tmp/vllm_teacher.log"

def start_vllm_server(model_name, gpu_memory_utilization=0.85, max_model_len=4096):
    """Start vLLM server via nohup. Returns True if server starts successfully."""
    print(f"[vllm] Starting server for {model_name} on port {VLLM_PORT}...", flush=True)
    
    # Kill any existing vLLM server
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
        # MoE-specific optimizations for Qwen3.5-35B-A3B
        "--enable-prefix-caching",
    ]
    
    # Check if we have multiple GPUs — use tensor parallelism
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            cmd.extend(["--tensor-parallel-size", str(n_gpus)])
            print(f"[vllm] Using tensor parallelism across {n_gpus} GPUs", flush=True)
    
    # Start with nohup
    with open(VLLM_LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group so we can kill cleanly
        )
    
    # Save PID
    with open(VLLM_PID_FILE, "w") as f:
        f.write(str(proc.pid))
    
    print(f"[vllm] Server PID: {proc.pid}", flush=True)
    
    # Wait for server to be ready (poll health endpoint)
    import requests
    max_wait = 300  # 5 minutes max for model loading
    poll_interval = 5
    start = time.time()
    
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{VLLM_URL}/health", timeout=5)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"[vllm] Server ready in {elapsed:.0f}s", flush=True)
                return True
        except requests.ConnectionError:
            pass
        except Exception as e:
            print(f"[vllm] Health check error: {e}", flush=True)
        
        # Check if process died
        if proc.poll() is not None:
            print(f"[vllm] Server exited with code {proc.returncode}", flush=True)
            try:
                with open(VLLM_LOG_FILE) as f:
                    tail = f.read()[-2000:]
                print(f"[vllm] Last log output:\n{tail}", flush=True)
            except Exception:
                pass
            return False
        
        time.sleep(poll_interval)
    
    print(f"[vllm] Server did not become ready within {max_wait}s", flush=True)
    stop_vllm_server()
    return False


def stop_vllm_server():
    """Stop the vLLM server and free VRAM."""
    try:
        if os.path.exists(VLLM_PID_FILE):
            with open(VLLM_PID_FILE) as f:
                pid = int(f.read().strip())
            # Kill the entire process group
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"[vllm] Sent SIGTERM to process group of PID {pid}", flush=True)
                # Wait up to 10s for graceful shutdown
                for _ in range(20):
                    try:
                        os.kill(pid, 0)  # check if alive
                        time.sleep(0.5)
                    except ProcessLookupError:
                        break
                else:
                    # Force kill
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        print(f"[vllm] Force killed PID {pid}", flush=True)
                    except Exception:
                        pass
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[vllm] Kill error: {e}", flush=True)
            os.unlink(VLLM_PID_FILE)
    except Exception as e:
        print(f"[vllm] Stop error: {e}", flush=True)
    
    # Also kill by port (belt and suspenders)
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{VLLM_PORT}/tcp"],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        pass
    
    # Free GPU memory
    free_gpu()
    time.sleep(2)  # let VRAM settle


def generate_via_vllm(prompts, tokenizer, max_new_tokens, block_seed=None):
    """
    Generate teacher continuations via the vLLM OpenAI-compatible API.
    
    Returns list of (full_text, prompt_len_tokens, gen_len_tokens) tuples.
    
    Key advantage: vLLM uses continuous batching + paged attention,
    making generation 5-10x faster than HF generate() for large models.
    """
    import requests
    
    results = []
    batch_size = 8  # Send multiple prompts at once for better throughput
    
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]
        
        # Use batch completions endpoint for maximum throughput
        for i, prompt_text in enumerate(batch):
            idx = batch_start + i
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
                    resp = requests.post(
                        f"{VLLM_URL}/v1/completions",
                        json=payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    continuation_text = result["choices"][0]["text"]
                    full_text = prompt_text + continuation_text
                    
                    # Token counts
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids
                    full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
                    prompt_len = prompt_ids.shape[1]
                    gen_len = full_ids.shape[1] - prompt_len
                    
                    results.append({
                        "full_text": full_text,
                        "full_ids": full_ids,
                        "prompt_len": prompt_len,
                        "gen_len": gen_len,
                    })
                    
                    if idx % 10 == 0 or idx == len(prompts) - 1:
                        print(f"  [{idx+1}/{len(prompts)}] {prompt_len} prompt + {gen_len} gen tokens", flush=True)
                    break
                    
                except Exception as e:
                    if attempt < 2:
                        print(f"  [prompt {idx}] Retry {attempt+1}: {e}", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  [prompt {idx}] FAILED after 3 attempts: {e}", flush=True)
                        raise
    
    return results


def teacher_logits_forward_pass(model, sequences_data, device):
    """
    Run HF forward pass on pre-generated sequences to extract full logit distributions.
    
    This is ~3x faster than generate() because:
    - No autoregressive decoding (single forward pass per sequence)
    - No KV cache management
    - Can use larger batch sizes
    
    Returns: (full_sequences, teacher_logits_list, prompt_lens)
    """
    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    
    with torch.no_grad():
        for i, data in enumerate(sequences_data):
            full_ids = data["full_ids"].to(device)
            prompt_len = data["prompt_len"]
            
            prompt_lens.append(prompt_len)
            full_sequences.append(full_ids)
            
            # Single forward pass — no generation, just logits
            logits = model(full_ids).logits.float()
            # Extract continuation logits (offset by 1 for next-token prediction)
            cont_logits = logits[:, prompt_len - 1:-1, :]
            teacher_logits_list.append(cont_logits.cpu())
            
            # Free per-prompt GPU tensors
            del logits, cont_logits
            
            if (i + 1) % 10 == 0 or i == len(sequences_data) - 1:
                print(f"  Logits [{i+1}/{len(sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
    
    return full_sequences, teacher_logits_list, prompt_lens


def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated SN97 evaluation")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--students", required=True, help="Comma-separated student model names")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt texts")
    parser.add_argument("--output", default="/home/eval_results.json")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-params-b", type=float, default=36.0)
    parser.add_argument("--block-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-logits", default="/home/teacher_cache.pt",
                       help="Path to cached teacher logits")
    parser.add_argument("--save-teacher-logits", default=None)
    parser.add_argument("--vllm-gpu-util", type=float, default=0.85,
                       help="GPU memory utilization for vLLM server (0-1)")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096,
                       help="Max sequence length for vLLM server")
    parser.add_argument("--no-vllm", action="store_true",
                       help="Disable vLLM, fall back to pure HF (for debugging)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    timings = {}

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]
    print(f"[eval] {len(prompts)} prompts (hash={prompts_hash}), {len(students)} students", flush=True)
    print(f"[eval] Teacher: {args.teacher}", flush=True)
    print(f"[eval] Device: {device}, VRAM: {gpu_mem_str()}", flush=True)
    print(f"[eval] vLLM: {'disabled (--no-vllm)' if args.no_vllm else 'enabled'}", flush=True)

    # Tokenize prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).input_ids.to(device)
        input_ids_list.append(ids)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher logits (cached or regenerated)
    # ═══════════════════════════════════════════════════════════════════
    
    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    teacher_cache_loaded = False

    # Try loading cached teacher logits
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        try:
            t0 = time.time()
            cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
            cached_n_prompts = len(cache.get("full_sequences", []))
            cached_prompts_hash = cache.get("prompts_hash")
            
            if cached_n_prompts == len(prompts) and cached_prompts_hash == prompts_hash:
                full_sequences = [s.to(device) for s in cache["full_sequences"]]
                teacher_logits_list = cache["teacher_logits"]
                prompt_lens = cache["prompt_lens"]
                timings["teacher_load"] = time.time() - t0
                timings["teacher_generation"] = 0.0
                timings["teacher_logits_pass"] = 0.0
                gen_method = cache.get("generation_method", "hf")
                print(f"[eval] ✓ Cached logits loaded in {timings['teacher_load']:.1f}s "
                      f"({len(full_sequences)} prompts, hash={cached_prompts_hash}, method={gen_method})", flush=True)
                teacher_cache_loaded = True
            else:
                print(f"[eval] ✗ Cache stale (cached: {cached_n_prompts}/{cached_prompts_hash}, "
                      f"need: {len(prompts)}/{prompts_hash}). Regenerating.", flush=True)
        except Exception as e:
            print(f"[eval] ✗ Cache load failed: {e}. Regenerating.", flush=True)

    if not teacher_cache_loaded:
        use_vllm = not args.no_vllm
        
        if use_vllm:
            # ── PHASE 1a: vLLM generation ──
            print(f"\n{'='*60}", flush=True)
            print(f"[eval] PHASE 1a: vLLM teacher generation", flush=True)
            print(f"{'='*60}", flush=True)
            
            t0 = time.time()
            vllm_ok = start_vllm_server(
                args.teacher,
                gpu_memory_utilization=args.vllm_gpu_util,
                max_model_len=args.vllm_max_model_len,
            )
            timings["vllm_startup"] = time.time() - t0
            
            if vllm_ok:
                print(f"[eval] vLLM server ready (startup: {timings['vllm_startup']:.0f}s)", flush=True)
                
                t0 = time.time()
                try:
                    sequences_data = generate_via_vllm(
                        prompts, tokenizer, args.max_new_tokens, args.block_seed
                    )
                    timings["vllm_generation"] = time.time() - t0
                    print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s "
                          f"({len(sequences_data)} sequences)", flush=True)
                except Exception as e:
                    print(f"[eval] vLLM generation failed: {e}", flush=True)
                    print(f"[eval] Falling back to HF generate()", flush=True)
                    sequences_data = None
                    use_vllm = False
                
                # Stop vLLM server — free VRAM for HF teacher + students
                print(f"[eval] Stopping vLLM server (freeing VRAM for logit pass)...", flush=True)
                stop_vllm_server()
                
                if sequences_data:
                    # ── PHASE 1b: HF forward pass for full logits ──
                    print(f"\n{'='*60}", flush=True)
                    print(f"[eval] PHASE 1b: HF teacher logit extraction", flush=True)
                    print(f"{'='*60}", flush=True)
                    
                    t0 = time.time()
                    teacher = load_model(args.teacher, device)
                    teacher.eval()
                    timings["teacher_load"] = time.time() - t0
                    print(f"[eval] Teacher loaded (HF) in {timings['teacher_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)
                    
                    t0 = time.time()
                    full_sequences, teacher_logits_list, prompt_lens = teacher_logits_forward_pass(
                        teacher, sequences_data, device
                    )
                    timings["teacher_logits_pass"] = time.time() - t0
                    print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)
                    
                    # Free sequences_data (we have full_sequences now)
                    del sequences_data
                    
                    # Save cache
                    teacher_cache_path = args.save_teacher_logits or os.path.join(
                        os.path.dirname(args.output), "teacher_cache.pt"
                    )
                    print(f"[eval] Saving teacher cache to {teacher_cache_path}...", flush=True)
                    torch.save({
                        "full_sequences": [s.cpu() for s in full_sequences],
                        "teacher_logits": teacher_logits_list,
                        "prompt_lens": prompt_lens,
                        "block_seed": args.block_seed,
                        "prompts_hash": prompts_hash,
                        "generation_method": "vllm+hf",
                    }, teacher_cache_path)
                    
                    # Keep teacher in VRAM for potential re-use
                    torch.cuda.empty_cache()
                    print(f"[eval] Teacher cache saved. VRAM: {gpu_mem_str()}", flush=True)
                    teacher_cache_loaded = True
            else:
                print(f"[eval] vLLM server failed to start — falling back to HF generate()", flush=True)
                use_vllm = False
        
        if not teacher_cache_loaded:
            # ── FALLBACK: Pure HF path (identical to current pod_eval.py) ──
            print(f"\n{'='*60}", flush=True)
            print(f"[eval] FALLBACK: HF teacher generation (no vLLM)", flush=True)
            print(f"{'='*60}", flush=True)
            
            t0 = time.time()
            teacher = load_model(args.teacher, device)
            teacher.eval()
            timings["teacher_load"] = time.time() - t0
            print(f"[eval] Teacher loaded in {timings['teacher_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)
            
            print(f"[eval] Generating teacher continuations (max_new_tokens={args.max_new_tokens})...", flush=True)
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
                    
                    gen_len = output_ids.shape[1] - prompt_len
                    print(f"  Prompt {i}: {prompt_len} prompt + {gen_len} gen tokens, VRAM: {gpu_mem_str()}", flush=True)
            
            timings["teacher_generation"] = time.time() - t0
            print(f"[eval] HF generation complete in {timings['teacher_generation']:.1f}s", flush=True)
            
            teacher_cache_path = args.save_teacher_logits or os.path.join(
                os.path.dirname(args.output), "teacher_cache.pt"
            )
            torch.save({
                "full_sequences": [s.cpu() for s in full_sequences],
                "teacher_logits": teacher_logits_list,
                "prompt_lens": prompt_lens,
                "block_seed": args.block_seed,
                "prompts_hash": prompts_hash,
                "generation_method": "hf",
            }, teacher_cache_path)
            
            torch.cuda.empty_cache()
            print(f"[eval] Teacher cache saved. VRAM: {gpu_mem_str()}", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Student scoring (identical to pod_eval.py from here)
    # ═══════════════════════════════════════════════════════════════════
    
    # Load prior results for --resume
    prior_results = {}
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prior = json.load(f)
            prior_results = prior.get("students", {})
            scored = [name for name, data in prior_results.items()
                      if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None]
            if scored:
                print(f"[eval] Resuming: {len(scored)} students already scored", flush=True)
        except Exception as e:
            print(f"[eval] Resume load failed: {e}", flush=True)

    results = {
        "teacher": args.teacher,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_len": args.max_prompt_len,
        "block_seed": args.block_seed,
        "n_prompts": len(prompts),
        "generation_method": timings.get("vllm_generation") and "vllm+hf" or "hf",
        "students": {},
    }
    # Carry forward prior scored students
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Live progress
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

    # Early stopping state
    best_kl_so_far = None
    best_kl_per_prompt_cumulative = None
    MIN_PROMPTS_EARLY_STOP = 7
    PER_MODEL_TIMEOUT = 600  # 10 min per model

    # Logit fingerprinting for functional copy detection
    teacher_fingerprint = None
    logit_fingerprints = {}

    # Record VRAM before students (teacher model may or may not be loaded)
    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] PHASE 2: Student scoring ({len(students)} models, {len(prompts)} prompts)", flush=True)
    print(f"{'='*60}", flush=True)

    for student_idx, student_name in enumerate(students):
        # Skip if already scored (resume)
        if student_name in results["students"]:
            prior = results["students"][student_name]
            kl = prior.get("kl_global_avg")
            status = prior.get("status", "scored")
            print(f"\n[eval] {student_name}: SKIPPED (already {status}, KL={kl})", flush=True)
            
            # Update early stopping with resumed student
            if kl is not None and kl > 0.001 and kl < float('inf'):
                if best_kl_so_far is None or kl < best_kl_so_far:
                    best_kl_so_far = kl
                    # Reconstruct cumulative means from saved kl_per_prompt
                    kl_per_prompt = prior.get("kl_per_prompt", [])
                    if kl_per_prompt:
                        best_kl_per_prompt_cumulative = []
                        running_sum = 0.0
                        for j, val in enumerate(kl_per_prompt):
                            running_sum += val
                            best_kl_per_prompt_cumulative.append(running_sum / (j + 1))
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Student: {student_name}", flush=True)
        
        model_start_time = time.time()
        
        # Disk check
        try:
            st = os.statvfs("/")
            pct = int(100 * (1 - st.f_bavail / st.f_blocks))
            if pct > 85:
                print(f"  [disk] {pct}% full — emergency cleanup before loading {student_name}", flush=True)
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                if cache_dir.exists():
                    teacher_cache_name = f"models--{args.teacher.replace('/', '--')}"
                    for d in cache_dir.iterdir():
                        if d.is_dir() and d.name.startswith("models--") and d.name != teacher_cache_name:
                            shutil.rmtree(d)
                            print(f"  [disk] Removed {d.name}", flush=True)
                st2 = os.statvfs("/")
                pct2 = int(100 * (1 - st2.f_bavail / st2.f_blocks))
                print(f"  [disk] After cleanup: {pct2}%", flush=True)
        except Exception as e:
            print(f"  [disk] Check failed: {e}", flush=True)

        # Load student model
        live_progress["current"] = {"student_name": student_name, "prompts_done": 0}
        _write_progress()

        try:
            t0 = time.time()
            student = load_model(student_name, device)
            student.eval()
        except Exception as e:
            print(f"[eval] FAILED to load {student_name}: {e}", flush=True)
            results["students"][student_name] = {
                "status": "load_failed",
                "error": str(e)[:500],
                "kl_global_avg": None,
            }
            results["timings"] = {k: round(v, 1) for k, v in timings.items()}
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            live_progress["completed"].append({
                "student_name": student_name,
                "status": "load_failed",
            })
            live_progress["current"] = None
            _write_progress()
            try:
                del student
            except NameError:
                pass
            free_gpu()
            # Clean disk cache for failed model
            try:
                cache_name = f"models--{student_name.replace('/', '--')}"
                cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
                if cache_path.exists():
                    shutil.rmtree(cache_path)
            except Exception:
                pass
            continue
        
        load_time = time.time() - t0
        student_vram_gb = (torch.cuda.memory_allocated() - vram_before) / 1024**3
        print(f"[eval] Loaded in {load_time:.1f}s, student VRAM: {student_vram_gb:.1f}GB, total: {gpu_mem_str()}", flush=True)

        # VRAM fraud check (delta-based)
        MAX_STUDENT_VRAM_GB = 20.0
        if student_vram_gb > MAX_STUDENT_VRAM_GB:
            vram_msg = (f"FRAUD: student VRAM delta {student_vram_gb:.1f}GB > {MAX_STUDENT_VRAM_GB}GB max")
            print(f"  ⚠️ {vram_msg}", flush=True)
            results["students"][student_name] = {
                "status": "fraud_vram",
                "reason": vram_msg,
                "vram_gb": round(student_vram_gb, 1),
                "kl_global_avg": float('inf'),
            }
            del student
            free_gpu()
            try:
                cache_name = f"models--{student_name.replace('/', '--')}"
                cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
                if cache_path.exists():
                    shutil.rmtree(cache_path)
            except Exception:
                pass
            continue

        # ── Per-prompt sequential scoring with early stopping ──
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

                    # Student forward pass
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]
                    
                    min_len = min(cont_s.shape[1], t_logits.shape[1])
                    kl_per_pos = compute_kl(t_logits[:, :min_len, :], cont_s[:, :min_len, :]).squeeze(0)

                    kl_mean_val = kl_per_pos.mean().item()
                    
                    # NaN/Inf check
                    if math.isnan(kl_mean_val) or math.isinf(kl_mean_val):
                        print(f"  [prompt {i}] KL={kl_mean_val} — invalid output, stopping", flush=True)
                        scoring_error = f"NaN/Inf KL at prompt {i}"
                        break

                    kl_per_prompt.append({
                        "mean": round(kl_mean_val, 6),
                        "std": round(kl_per_pos.std().item(), 6) if kl_per_pos.shape[0] > 1 else 0.0,
                        "n_positions": kl_per_pos.shape[0],
                    })
                    prompt_kl_means.append(kl_mean_val)

                    # Update live progress
                    running_mean = sum(prompt_kl_means) / len(prompt_kl_means)
                    live_progress["current"] = {
                        "student_name": student_name,
                        "prompts_done": i + 1,
                        "kl_running_mean": round(running_mean, 6),
                        "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    }
                    _write_progress()

                    if (i + 1) % 5 == 0:
                        print(f"  [{i+1}/{len(prompts)}] KL={kl_mean_val:.6f} (running avg: {running_mean:.6f})", flush=True)

                    # Free per-prompt tensors
                    del s_logits, cont_s, t_logits, kl_per_pos

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  [prompt {i}] OOM — stopping", flush=True)
                    else:
                        print(f"  [prompt {i}] RuntimeError: {e}", flush=True)
                    scoring_error = str(e)
                    free_gpu()
                    break
                except Exception as e:
                    print(f"  [prompt {i}] Unexpected error: {e}", flush=True)
                    scoring_error = str(e)
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

                    # Reject KL ≤ 0.001 from best comparison (fraud prevention)
                    if best_at_n <= 0.001:
                        best_at_n = best_kl_so_far if best_kl_so_far and best_kl_so_far > 0.001 else float('inf')

                    if student_lower > best_at_n:
                        print(
                            f"  [early stop] at prompt {n}: lower CI {student_lower:.6f} > "
                            f"best@{n} {best_at_n:.6f} (running avg: {running_mean:.6f})",
                            flush=True,
                        )
                        early_stopped = True
                        break

                # Per-model timeout
                if time.time() - model_start_time > PER_MODEL_TIMEOUT:
                    print(f"  [timeout] exceeded {PER_MODEL_TIMEOUT}s after {len(prompt_kl_means)} prompts", flush=True)
                    early_stopped = True
                    break

        scoring_time = time.time() - t0

        # Handle scoring results
        if scoring_error and not kl_per_prompt:
            print(f"  ‼️ Scoring failed completely: {scoring_error}", flush=True)
            results["students"][student_name] = {
                "status": "scoring_error",
                "error": scoring_error[:500],
                "kl_global_avg": None,
            }
        elif kl_per_prompt:
            kl_global_avg = sum(d["mean"] for d in kl_per_prompt) / len(kl_per_prompt)
            n_scored = len(kl_per_prompt)
            
            print(f"  → KL={kl_global_avg:.6f} ({n_scored}/{len(prompts)} prompts"
                  f"{', early-stopped' if early_stopped else ''}"
                  f"{', partial (error)' if scoring_error else ''})", flush=True)

            results["students"][student_name] = {
                "status": "early_stopped" if early_stopped else ("partial" if scoring_error else "scored"),
                "kl_global_avg": round(kl_global_avg, 6),
                "kl_per_prompt": [d["mean"] for d in kl_per_prompt],
                "prompts_scored": n_scored,
                "scoring_time": round(scoring_time, 1),
                "load_time": round(load_time, 1),
                "early_stopped": early_stopped,
                "vram_gb": round(student_vram_gb, 1),
            }

            # Update early stopping baseline
            if kl_global_avg > 0.001:  # Reject near-zero (fraud)
                if not early_stopped and not scoring_error:
                    if best_kl_so_far is None or kl_global_avg < best_kl_so_far:
                        best_kl_so_far = kl_global_avg
                        best_kl_per_prompt_cumulative = []
                        running_sum = 0.0
                        for j, d in enumerate(kl_per_prompt):
                            running_sum += d["mean"]
                            best_kl_per_prompt_cumulative.append(running_sum / (j + 1))
                        print(f"  → New best student: KL={kl_global_avg:.6f}", flush=True)

        # Save incremental results
        results["timings"] = {k: round(v, 1) for k, v in timings.items()}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        # Update live progress
        live_progress["completed"].append({
            "student_name": student_name,
            "status": results["students"].get(student_name, {}).get("status", "unknown"),
            "kl": results["students"].get(student_name, {}).get("kl_global_avg"),
            "prompts_scored": len(kl_per_prompt),
        })
        live_progress["current"] = None
        _write_progress()

        # Cleanup student
        del student
        free_gpu()
        try:
            cache_name = f"models--{student_name.replace('/', '--')}"
            cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"  [cleanup] Removed {cache_name}", flush=True)
        except Exception:
            pass

    # Final save
    results["timings"] = {k: round(v, 1) for k, v in timings.items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] ALL DONE — {len(results['students'])} students scored", flush=True)
    print(f"[eval] Timings: {json.dumps(timings, indent=2)}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
