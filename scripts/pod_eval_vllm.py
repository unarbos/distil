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

# ── KL computation ──
# Uses F.kl_div(log_target=True) which is mathematically identical to
# sum(P * (log P - log Q)) but lets PyTorch fuse the kernel internally.
# Chunked over positions to reduce peak memory (~4x less for 512-pos sequences).
# Credit: caseus (github.com/winglian) for the optimization.
KL_CHUNK_SIZE = 128


def _kl_chunk_fn(t_log_p_chunk, s_log_p_chunk):
    """KL per position for a chunk. Uses F.kl_div for better kernel fusion."""
    return F.kl_div(s_log_p_chunk, t_log_p_chunk, log_target=True, reduction="none").sum(dim=-1)


# Try to compile the inner kernel for ~2x speedup
try:
    _kl_chunk_compiled = torch.compile(_kl_chunk_fn, fullgraph=True)
    _KL_USE_COMPILED = True
except Exception:
    _kl_chunk_compiled = _kl_chunk_fn
    _KL_USE_COMPILED = False


def compute_kl(teacher_logits, student_logits):
    """KL(teacher || student) per position. For one-off use."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    n_pos = t_log_p.shape[-2] if t_log_p.dim() >= 2 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    kl_per_pos = torch.empty(t_log_p.shape[:-1], device=t_log_p.device)
    for i in range(0, n_pos, KL_CHUNK_SIZE):
        j = min(i + KL_CHUNK_SIZE, n_pos)
        if t_log_p.dim() >= 3:
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
        else:
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def compute_kl_from_precomputed(t_log_p, t_p, student_logits):
    """KL using precomputed teacher log_softmax + probs. Saves ~50% compute.

    Uses F.kl_div(log_target=True) + chunking for better memory and speed.
    The t_p argument is kept for API compat but not used (F.kl_div computes
    exp(t_log_p) internally when log_target=True).
    """
    s_logits = student_logits.float()
    # Handle vocab size mismatch (student vs teacher)
    t_vocab = t_log_p.shape[-1]
    s_vocab = s_logits.shape[-1]
    if s_vocab < t_vocab:
        # Pad student logits with -inf (zero probability)
        pad = torch.full((*s_logits.shape[:-1], t_vocab - s_vocab), -1e10,
                         device=s_logits.device, dtype=s_logits.dtype)
        s_logits = torch.cat([s_logits, pad], dim=-1)
    elif s_vocab > t_vocab:
        s_logits = s_logits[..., :t_vocab]
    s_log_p = F.log_softmax(s_logits, dim=-1)
    # Chunked KL over positions to reduce peak memory
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
    else:
        kl_per_pos = torch.empty(n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos

def load_model(name, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    # Use device_map="auto" for teacher on multi-GPU to spread weights
    dm = "auto" if is_teacher and torch.cuda.device_count() > 1 else device
    kwargs = dict(dtype=dtype, device_map=dm, trust_remote_code=is_teacher)
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

def is_vllm_running():
    """Check if vLLM server is already running and healthy."""
    import requests
    try:
        r = requests.get(f"{VLLM_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=4096, persistent=False):
    """Start vLLM server via subprocess. Returns True on success.
    If persistent=True, reuses an already-running server."""
    if persistent and is_vllm_running():
        print(f"\n[vllm] Server already running — reusing (persistent mode)", flush=True)
        return True
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
        "--no-enable-log-requests",
    ]

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        n = torch.cuda.device_count()
        # TP size must be a power of 2 that divides model dims
        tp = 1
        for candidate in [8, 4, 2]:
            if n >= candidate:
                tp = candidate
                break
        cmd.extend(["--tensor-parallel-size", str(tp)])
        print(f"[vllm] Tensor parallelism: {n} GPUs", flush=True)

    log_f = open("/tmp/vllm_teacher.log", "w")
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    Path("/tmp/vllm_teacher.pid").write_text(str(proc.pid))
    print(f"[vllm] PID: {proc.pid}", flush=True)

    import requests
    for elapsed in range(0, 900, 5):
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
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index (for compatibility)")
    parser.add_argument("--sequential", action="store_true", help="Ignored — for compatibility")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use pure HF")
    parser.add_argument("--persistent-vllm", action="store_true",
                        help="Keep vLLM running between rounds — reuse if already up, don't kill after generation")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.45,
                        help="vLLM GPU memory utilization (default 0.45 to leave room for students)")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    args = parser.parse_args()

    # Set CUDA device if specified
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    timings = {}

    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]

    # Progress file — written throughout for live dashboard updates
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")
    def _write_phase(phase, teacher_done=None, **extra):
        """Write a phase update to the progress file for the dashboard."""
        try:
            data = {
                "phase": phase,
                "students": students,
                "students_total": len(students),
                "prompts_total": len(prompts),
                "teacher_prompts_done": teacher_done,
                "completed": extra.get("completed", []),
                "current": extra.get("current", None),
            }
            with open(progress_path, "w") as pf:
                json.dump(data, pf)
        except Exception:
            pass

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

        _write_phase("vllm_starting")
        t0 = time.time()
        vllm_ok = start_vllm_server(args.teacher, args.vllm_gpu_util, args.vllm_max_model_len,
                                     persistent=args.persistent_vllm)
        timings["vllm_startup"] = time.time() - t0

        sequences_data = None
        if vllm_ok:
            _write_phase("vllm_generating")
            t0 = time.time()
            try:
                sequences_data = generate_via_vllm(prompts, tokenizer, args.max_new_tokens, args.block_seed)
                timings["vllm_generation"] = time.time() - t0
                print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] vLLM generation failed: {e} — falling back to HF", flush=True)
            if not args.persistent_vllm:
                stop_vllm_server()
            else:
                print(f"[eval] vLLM kept alive (persistent mode)", flush=True)
        else:
            print(f"[eval] vLLM failed to start — falling back to HF", flush=True)

        if sequences_data:
            # ── Free vLLM VRAM before loading HF teacher ──
            if args.persistent_vllm:
                print(f"[eval] Stopping vLLM to free VRAM for HF logit extraction...", flush=True)
                stop_vllm_server()
                import gc; gc.collect(); torch.cuda.empty_cache()
                time.sleep(3)
                print(f"[eval] VRAM after vLLM stop: {gpu_mem_str()}", flush=True)

            # ── HF forward pass for logits ──
            print(f"\n{'='*60}", flush=True)
            print(f"PHASE 1b: HF teacher logit extraction", flush=True)
            print(f"{'='*60}", flush=True)

            t0 = time.time()
            teacher = load_model(args.teacher, device)
            teacher.eval()
            timings["teacher_hf_load"] = time.time() - t0
            print(f"[eval] HF teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

            _write_phase("teacher_logits", teacher_done=0)
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
                    _write_phase("teacher_logits", teacher_done=i + 1)

            timings["teacher_logits_pass"] = time.time() - t0
            print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)
            del sequences_data

            # Save cache only if explicitly requested AND enough disk
            if args.save_teacher_logits:
                cache_path = args.save_teacher_logits
                st = os.statvfs(os.path.dirname(cache_path) or '/')
                free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
                if free_gb > 50:  # need ~45GB for cache
                    cache_tmp = cache_path + ".tmp"
                    torch.save({
                        "full_sequences": [s.cpu() for s in full_sequences],
                        "teacher_logits": teacher_logits_list,
                        "prompt_lens": prompt_lens,
                        "block_seed": args.block_seed,
                        "prompts_hash": prompts_hash,
                        "generation_method": "vllm+hf",
                    }, cache_tmp)
                    os.replace(cache_tmp, cache_path)
                    print(f"[eval] Cache saved to {cache_path}", flush=True)
                else:
                    print(f"[eval] Skipped cache save ({free_gb:.0f}GB free, need 50GB)", flush=True)
            else:
                print(f"[eval] No cache save requested", flush=True)

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

        _write_phase("teacher_generation", teacher_done=0)
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
                _write_phase("teacher_generation", teacher_done=i + 1)

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
    # PHASE 1c: Move teacher logits to GPU + precompute softmax
    # ═══════════════════════════════════════════════════════════════════
    # Teacher logits are ~17GB for 60 prompts × 512 tokens × 152K vocab.
    # B200 has 192GB — we use ~40GB total (king 8GB + challenger 8GB + logits 17GB).
    # Keeping them on GPU eliminates ~93GB of PCIe transfers per round
    # (18.7GB × 5 students). Precomputing log_softmax + probs saves ~50%
    # of KL computation (teacher side computed once, not per-student).
    _write_phase("gpu_precompute", teacher_done=len(prompts))
    print(f"\n[eval] Keeping teacher logits on CPU, will process in chunks during scoring...", flush=True)
    t0 = time.time()
    # CHANGED: Don't precompute all teacher softmax on GPU at once.
    # With 120 prompts, the full tensor is ~68GB which OOMs on B200 (178GB - 82GB teacher).
    # Instead, keep raw logits on CPU and compute softmax per-chunk during student scoring.
    # This trades ~10% speed for fitting in VRAM.
    teacher_log_probs = None  # signal to use chunked processing
    teacher_probs = None
    # Keep teacher_logits_list on CPU for chunked access
    timings["teacher_gpu_precompute"] = time.time() - t0
    print(f"[eval] Teacher logits: {len(teacher_logits_list)} prompts on CPU, chunked GPU processing enabled, VRAM: {gpu_mem_str()}", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1d: Filter short-completion prompts
    # ═══════════════════════════════════════════════════════════════════
    # Prompts where the teacher generated <MIN_COMPLETION_TOKENS tokens are
    # degenerate (teacher hit EOS early). KL on 1-23 tokens is pure noise
    # and inflates variance. Remove for ALL models equally so paired
    # comparisons remain valid.
    MIN_COMPLETION_TOKENS = 64
    completion_lens = [full_sequences[i].shape[1] - prompt_lens[i] for i in range(len(prompts))]
    keep_mask = [cl >= MIN_COMPLETION_TOKENS for cl in completion_lens]
    n_filtered = sum(1 for k in keep_mask if not k)
    if n_filtered > 0:
        filtered_indices = [i for i, k in enumerate(keep_mask) if not k]
        print(f"[eval] Filtering {n_filtered} prompts with <{MIN_COMPLETION_TOKENS} completion tokens: {filtered_indices}", flush=True)
        for idx in filtered_indices:
            print(f"  prompt {idx}: {completion_lens[idx]} tokens", flush=True)
        # Rebuild arrays keeping only valid prompts
        full_sequences = [full_sequences[i] for i in range(len(prompts)) if keep_mask[i]]
        teacher_logits_list = [teacher_logits_list[i] for i in range(len(prompts)) if keep_mask[i]]
        prompt_lens = [prompt_lens[i] for i in range(len(prompts)) if keep_mask[i]]
        prompts = [prompts[i] for i in range(len(prompts)) if keep_mask[i]]
        print(f"[eval] {len(prompts)} prompts remaining after filter", flush=True)
    else:
        print(f"[eval] All {len(prompts)} prompts have >={MIN_COMPLETION_TOKENS} completion tokens — no filtering needed", flush=True)

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
        "n_prompts_filtered": n_filtered,
        "min_completion_tokens": MIN_COMPLETION_TOKENS,
        "students": {},
    }
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Progress (progress_path already defined at top of main)
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
        live_progress["phase"] = "loading_student"
        live_progress["current"] = {
            "student_name": student_name,
            "student_idx": student_idx,
            "prompts_done": 0,
        }
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

        # ── Score: per-prompt with precomputed teacher, early stopping ──
        # Teacher log_probs and probs are already on GPU (precomputed in Phase 1c).
        # No CPU→GPU transfers needed. Student does forward pass, we compute KL
        # using precomputed teacher side (saves ~50% of KL compute).
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
                    # Teacher side: compute softmax on-the-fly per prompt (chunked to avoid OOM)
                    if teacher_log_probs is not None:
                        # Legacy path: precomputed (only if VRAM was sufficient)
                        t_log_p = teacher_log_probs[i]
                        t_p = teacher_probs[i]
                    else:
                        # Chunked path: move single prompt to GPU, compute, use, free
                        tl = teacher_logits_list[i].to(device).float()
                        t_log_p = F.log_softmax(tl, dim=-1)
                        t_p = t_log_p.exp()
                        del tl
                    # Student forward pass
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]
                    min_len = min(cont_s.shape[1], t_log_p.shape[1])
                    # KL with precomputed teacher (skip teacher softmax)
                    t_lp_slice = t_log_p[:, :min_len, :]
                    t_p_slice = t_p[:, :min_len, :]
                    s_cont_slice = cont_s[:, :min_len, :]
                    kl_per_pos = compute_kl_from_precomputed(
                        t_lp_slice, t_p_slice, s_cont_slice
                    ).squeeze(0)
                    kl_mean = kl_per_pos.mean().item()

                    # ── Shadow top-k KL metrics (logged but not used for scoring) ──
                    topk_shadow = {}
                    try:
                        s_log_p_full = F.log_softmax(s_cont_slice.float(), dim=-1)
                        for k_val in (100, 1000):
                            # Get teacher's top-k token indices per position
                            # t_p_slice: [1, min_len, vocab]
                            _, topk_idx = t_p_slice.topk(k_val, dim=-1)  # [1, min_len, k]
                            # Gather teacher and student log-probs at those indices
                            t_topk = t_lp_slice.gather(-1, topk_idx)  # [1, min_len, k]
                            s_topk = s_log_p_full.gather(-1, topk_idx)  # [1, min_len, k]
                            # Renormalize (log_softmax over the k dimension)
                            t_topk_norm = F.log_softmax(t_topk, dim=-1)
                            s_topk_norm = F.log_softmax(s_topk, dim=-1)
                            # KL on renormalized top-k
                            topk_kl = F.kl_div(s_topk_norm, t_topk_norm,
                                               log_target=True, reduction="none").sum(dim=-1).squeeze(0)
                            topk_shadow[f"kl_top{k_val}"] = round(topk_kl.mean().item(), 6)
                            del topk_idx, t_topk, s_topk, t_topk_norm, s_topk_norm, topk_kl
                        del s_log_p_full
                    except Exception:
                        pass  # Shadow metrics are best-effort

                    del s_logits, cont_s, kl_per_pos, t_lp_slice, t_p_slice, s_cont_slice
                    if teacher_log_probs is None:
                        # Chunked path: free per-prompt teacher tensors
                        del t_log_p, t_p
                        if i % 20 == 0:
                            torch.cuda.empty_cache()

                    if math.isnan(kl_mean) or math.isinf(kl_mean):
                        print(f"  [prompt {i}] KL={kl_mean} — invalid, stopping", flush=True)
                        scoring_error = f"NaN/Inf KL at prompt {i}"
                        break

                    prompt_result = {"mean": round(kl_mean, 6)}
                    if topk_shadow:
                        prompt_result.update(topk_shadow)
                    kl_per_prompt.append(prompt_result)
                    prompt_kl_means.append(kl_mean)

                    running_mean = sum(prompt_kl_means) / len(prompt_kl_means)
                    live_progress["phase"] = "scoring"
                    live_progress["current"] = {
                        "student_name": student_name,
                        "student_idx": student_idx,
                        "prompts_done": i + 1,
                        "prompts_total": len(prompts),
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
            # Aggregate shadow top-k metrics
            topk_aggs = {}
            for k_label in ("kl_top100", "kl_top1000"):
                vals = [d.get(k_label) for d in kl_per_prompt if d.get(k_label) is not None]
                if vals:
                    topk_aggs[k_label] = round(sum(vals) / len(vals), 6)

            student_result = {
                "status": status,
                "kl_global_avg": round(kl_avg, 6),
                "kl_per_prompt": [d["mean"] for d in kl_per_prompt],
                "prompts_scored": n_scored,
                "scoring_time": round(scoring_time, 1),
                "load_time": round(load_time, 1),
                "early_stopped": early_stopped,
            }
            # Shadow top-k KL (logged for analysis, not used in scoring)
            if topk_aggs:
                student_result["shadow_topk"] = topk_aggs
                print(f"  → Shadow top-k: {topk_aggs}", flush=True)
            results["students"][student_name] = student_result
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
