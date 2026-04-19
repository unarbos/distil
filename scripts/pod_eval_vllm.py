#!/usr/bin/env python3
"""
vLLM-accelerated GPU evaluation script for SN97 validation (v3.0.0).

Architecture & VRAM timeline (single B200 = 192GB):
  Phase 1 — Teacher generation via vLLM:
    [vLLM teacher ~70GB] → generate continuations → kill server
    Time: ~3-5 min (vs 25 min with HF)

  Phase 2 — Teacher logit extraction via HF:
    [HF teacher ~67GB] → forward passes (no autoregressive) → cache logits → unload
    Time: ~8-10 min (forward-only, ~3x faster than generate)

  Phase 3 — Student scoring:
    [teacher logits on CPU ~2GB] + [king ~8GB stays loaded] + [challenger ~8GB rotates]
    Total VRAM: ~18GB (king + challenger + overhead)
    Time: ~2-3 min per student

Optimizations:
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

File layout (single-file — uploaded to remote GPU pod via SCP):
  1. Imports & Constants
  2. GPU & Disk Utilities
  3. Model Utilities (load, prefetch, cache, fingerprint)
  4. KL Computation (core, sparse, precomputed)
  5. vLLM Server Management (start, stop, health)
  6. vLLM Generation (teacher generation, logprobs parsing)
  7. vLLM Student Scoring
  8. HF Batched Forward Pass
  9. Progress Reporting
  10. Main
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Imports
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# §2  Constants
# ═══════════════════════════════════════════════════════════════════════════════

# -- HF forward pass --
HF_CHUNK_SIZE = 4096        # Chunk size for KV-cached forward on long sequences
MIN_COMPLETION_TOKENS = 10  # Skip prompts producing fewer continuation tokens

# -- KL computation --
# Uses F.kl_div(log_target=True) which is mathematically identical to
# sum(P * (log P - log Q)) but lets PyTorch fuse the kernel internally.
# Chunked over positions to reduce peak memory (~4x less for 512-pos sequences).
# Credit: caseus (github.com/winglian) for the optimization.
KL_CHUNK_SIZE = 128

# -- Activation fingerprinting (functional copy detection) --
ACTIVATION_FP_SEED = 42
ACTIVATION_FP_N_INPUTS = 5
ACTIVATION_FP_SEQ_LEN = 64
ACTIVATION_FP_VOCAB_SIZE = 248320  # Qwen tokenizer vocab

# -- vLLM server --
VLLM_PORT = 9100
VLLM_URL = f"http://localhost:{VLLM_PORT}"
VLLM_STARTUP_TIMEOUT = 900  # 15 min
VLLM_REQUEST_TIMEOUT = 300

# ═══════════════════════════════════════════════════════════════════════════════
# §3  GPU & Disk Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str():
    """Return a human-readable string of current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"


def free_gpu():
    """Free GPU memory: garbage collect, empty cache, synchronize."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


def ensure_disk_space(teacher_name, threshold=85):
    """Check disk usage, clean non-teacher caches and stale files if above threshold.

    Called at every phase boundary (before vLLM start, before HF load, before each student)
    to prevent disk exhaustion — the #1 crash cause historically.

    Returns current disk usage percentage.
    """
    try:
        st = os.statvfs("/")
        pct = int(100 * (1 - st.f_bavail / st.f_blocks))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        print(f"  [disk] {pct}% used, {free_gb:.1f}GB free", flush=True)

        if pct > threshold:
            print(f"  [disk] >{threshold}% — cleaning non-teacher caches", flush=True)
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for d in cache_dir.iterdir():
                    if d.is_dir() and d.name.startswith("models--") and d.name != teacher_cache:
                        shutil.rmtree(d)
            # Clean stale /tmp files >1GB
            for pattern in ["/tmp/vllm_*", "/tmp/tmp*", "/tmp/teacher_*"]:
                for f in glob.glob(pattern):
                    try:
                        fsize = os.path.getsize(f)
                        if fsize > 1024**3:
                            os.remove(f)
                            print(f"  [disk] Removed stale {f} ({fsize/1024**3:.1f}GB)", flush=True)
                    except Exception:
                        pass
            # Clean stale teacher cache if hash doesn't match
            teacher_logits_path = "/home/teacher_cache.pt"
            if os.path.exists(teacher_logits_path):
                cache_size = os.path.getsize(teacher_logits_path) / (1024**3)
                if cache_size > 0 and pct > 90:
                    os.remove(teacher_logits_path)
                    print(f"  [disk] Removed stale teacher cache ({cache_size:.1f}GB)", flush=True)

            st2 = os.statvfs("/")
            pct2 = int(100 * (1 - st2.f_bavail / st2.f_blocks))
            free_gb2 = (st2.f_bavail * st2.f_frsize) / (1024**3)
            print(f"  [disk] After cleanup: {pct2}% used, {free_gb2:.1f}GB free", flush=True)
            return pct2
        return pct
    except Exception as e:
        print(f"  [disk] Check failed: {e}", flush=True)
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Model Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(name, device="cuda", dtype=torch.bfloat16, revision=None):
    """Load a HuggingFace model for inference.

    Uses flash_attention_2 when available, falls back to default attention.
    Teacher models get trust_remote_code=True; students don't (security).
    When revision is set, pins to that exact HF commit hash.
    """
    from transformers import AutoModelForCausalLM
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    if revision and revision != "main":
        kwargs["revision"] = revision
        print(f"  [model] Pinning to revision {revision[:12]}", flush=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            try:
                m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
                print(f"  [model] Loaded with flash_attention_2", flush=True)
                return m
            except (ValueError, ImportError):
                m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                print(f"  [model] Loaded with default attention", flush=True)
                return m
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout", "HTTPSConnection"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [model] Transient error loading {name} (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                raise


def prefetch_model(name, revision=None, max_retries=3):
    """Download model files to HF cache without loading to GPU. Runs in background.

    Retries on transient HF errors (429, 503, connection errors).
    """
    from huggingface_hub import snapshot_download
    dl_kwargs = dict(ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"])
    if revision and revision != "main":
        dl_kwargs["revision"] = revision
    for attempt in range(max_retries):
        try:
            snapshot_download(name, **dl_kwargs)
            print(f"  [prefetch] {name} cached (rev={revision or 'main'})", flush=True)
            return
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [prefetch] {name} transient error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                print(f"  [prefetch] {name} failed: {e}", flush=True)
                return


def clean_model_cache(name, teacher_name=None):
    """Remove HF cache for a specific model, preserving teacher cache."""
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


FINETUNE_PROBE_TEXT = "The capital of France is Paris. The capital of Germany is Berlin."
FINETUNE_GRAD_NORM_MAX = float(os.environ.get("FINETUNE_GRAD_NORM_MAX", "500"))
FINETUNE_NORM_WEIGHT_MAX = float(os.environ.get("FINETUNE_NORM_WEIGHT_MAX", "30"))


def _classify_probe_param(name: str) -> str:
    n = name.lower()
    if "norm" in n.rsplit(".", 1)[-1] or "layernorm" in n or "rmsnorm" in n:
        return "norm"
    if "embed" in n or "wte" in n or "tok_embeddings" in n:
        return "embed"
    if "lm_head" in n or "output_proj" in n:
        return "lm_head"
    if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attention"]):
        return "attn"
    if any(k in n for k in ["mlp", "gate_proj", "up_proj", "down_proj", "ffn", "feed_forward", "experts"]):
        return "ffn"
    if n.endswith(".bias") or ".bias" in n:
        return "bias"
    return "other"


def finetunability_probe(model, tokenizer, device="cuda"):
    """Fine-tunability diagnostic inspired by mantaLLM / const / caseus (SN97 Discord).

    Rejects models that can't be continued-pretrained over:
      - LayerNorm/RMSNorm weights scaled beyond sane bounds (anti-finetune watermark)
      - Gradient explosion on a trivial next-token CE loss
      - NaN/Inf in loss or gradients
      - Per-param-type norm imbalance (one group >> the rest)

    Returns dict with pass, reason, stats. Never raises — errors return pass=True with note.
    """
    stats = {
        "pass": True, "reason": "",
        "global_grad_norm": 0.0,
        "worst_param_type": "",
        "worst_param_norm": 0.0,
        "worst_norm_weight": 0.0,
        "worst_norm_name": "",
        "loss": 0.0,
    }
    try:
        worst_name = ""
        worst_val = 0.0
        for nm, mod in model.named_modules():
            cls = type(mod).__name__.lower()
            if ("norm" in cls or cls.endswith("ln")) and hasattr(mod, "weight") and mod.weight is not None:
                w = mod.weight.detach()
                if not torch.isfinite(w).all():
                    stats.update({"pass": False, "reason": f"norm_weight_nan_inf:{nm}", "worst_norm_name": nm})
                    return stats
                mx = float(w.float().abs().max().item())
                if mx > worst_val:
                    worst_val = mx
                    worst_name = nm
        stats["worst_norm_weight"] = round(worst_val, 4)
        stats["worst_norm_name"] = worst_name
        if worst_val > FINETUNE_NORM_WEIGHT_MAX:
            stats["pass"] = False
            stats["reason"] = f"norm_weight_scaled:{worst_name}={worst_val:.1f}>{FINETUNE_NORM_WEIGHT_MAX:.0f}"
            return stats

        was_training = model.training
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None

        ids = tokenizer(FINETUNE_PROBE_TEXT, return_tensors="pt").input_ids.to(device)
        try:
            with torch.enable_grad():
                out = model(input_ids=ids, labels=ids)
                loss = out.loss
        except Exception as fwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"forward_failed:{str(fwd_err)[:120]}"})
            return stats

        loss_val = float(loss.detach().float().item()) if loss is not None else float("nan")
        stats["loss"] = round(loss_val, 4)
        if loss is None or not math.isfinite(loss_val):
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"loss_nan_inf:{loss_val}"})
            return stats

        try:
            loss.backward()
        except Exception as bwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"backward_failed:{str(bwd_err)[:120]}"})
            return stats

        global_sq = 0.0
        per_type_sq: dict = {}
        for nm, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                if not was_training:
                    model.eval()
                for pp in model.parameters():
                    pp.grad = None
                stats.update({"pass": False, "reason": f"grad_nan_inf:{nm}"})
                return stats
            n_sq = float((g.float() ** 2).sum().item())
            global_sq += n_sq
            ptype = _classify_probe_param(nm)
            per_type_sq[ptype] = per_type_sq.get(ptype, 0.0) + n_sq

        for p in model.parameters():
            p.grad = None
        if not was_training:
            model.eval()

        global_norm = global_sq ** 0.5
        stats["global_grad_norm"] = round(global_norm, 2)
        if global_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:global={global_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        worst_type = ""
        worst_norm = 0.0
        for ptype, sq in per_type_sq.items():
            n = sq ** 0.5
            if n > worst_norm:
                worst_norm = n
                worst_type = ptype
        stats["worst_param_type"] = worst_type
        stats["worst_param_norm"] = round(worst_norm, 2)
        if worst_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:{worst_type}={worst_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        return stats
    except Exception as e:
        try:
            for p in model.parameters():
                p.grad = None
        except Exception:
            pass
        stats["reason"] = f"probe_error:{str(e)[:120]}"
        return stats


CHAT_PROBE_PROMPTS = [
    "hi",
    "What is 2+2?",
    "Say hello in one word.",
    "Reply with just the word: yes",
]
CHAT_PROBE_MAX_TOKENS = int(os.environ.get("CHAT_PROBE_MAX_TOKENS", "768"))
CHAT_PROBE_MIN_ANSWER_CHARS = int(os.environ.get("CHAT_PROBE_MIN_ANSWER_CHARS", "1"))
CHAT_PROBE_TERMINATE_THRESHOLD = float(os.environ.get("CHAT_PROBE_TERMINATE_THRESHOLD", "0.5"))

THINK_PROBE_PROMPTS = [
    # ── Termination battery (16): trivial prompts where any sane model stops fast.
    # A model that fails to stop here is broken, full stop.
    "Hi",
    "What is the largest planet? Answer in one word.",
    "Say the word: done",
    "Reply with just the number 7.",
    "Output only: OK",
    "Greet me.",
    "Which is larger: 2 or 3? Answer with just the digit.",
    "Complete the word: appl_",
    "What color is the sky usually? One word.",
    "Name a primary color. One word.",
    "Answer yes or no: is water wet?",
    "What is 2+2? Reply with just the number.",
    "What is the capital of France? One word.",
    "Say the letter A.",
    "Output the digit 5.",
    "Respond with the single word: hello",
    # ── Reasoning battery (16): prompts that legitimately warrant a chain of
    # thought. A healthy distilled student should *use* CoT and still terminate
    # within the budget. A model that loops, falls into a degenerate pattern,
    # or empties the 2048-token budget without resolving is the failure we
    # want to catch — this is the pathology we’ve observed in KL-saturated
    # kings that “pass” on one-word probes but melt under actual thinking.
    "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step and give the final answer.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain briefly then answer.",
    "I have 3 apples today. Yesterday I ate 2. How many do I have now? Work it out.",
    "What is 23 * 7? Show your steps then give the answer.",
    "If today is Wednesday, what day of the week is it 10 days from now? Reason then answer.",
    "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Reason then answer.",
    "Sort these numbers ascending: 9, 3, 14, 1, 7. List the sorted order.",
    "Is 51 prime? Give your reasoning then answer yes or no.",
    "A rectangle is 5 by 8. What is its area and perimeter?",
    "Explain why ice floats on water in one or two sentences.",
    "List three differences between mitosis and meiosis.",
    "In one short paragraph, explain what a neural network is.",
    "Give three distinct English words that rhyme with 'orange' or say 'none' if impossible.",
    "Translate to Spanish: 'The cat sat on the mat.'",
    "What is the next number in the sequence 1, 1, 2, 3, 5, 8, __? Explain.",
    "Write a single limerick about a programmer who forgets their password.",
]
THINK_PROBE_MAX_TOKENS = int(os.environ.get("THINK_PROBE_MAX_TOKENS", "2048"))
THINK_PROBE_TERMINATE_THRESHOLD = float(os.environ.get("THINK_PROBE_TERMINATE_THRESHOLD", "0.66"))
THINK_PROBE_DEGEN_SIGMA = float(os.environ.get("THINK_PROBE_DEGEN_SIGMA", "4.0"))
THINK_PROBE_GZIP_FLOOR = float(os.environ.get("THINK_PROBE_GZIP_FLOOR", "0.25"))
THINK_PROBE_SELFBLEU_MAX = float(os.environ.get("THINK_PROBE_SELFBLEU_MAX", "0.85"))
# Wilson lower-bound confidence for pass/fail decisions (T0.2). Replaces the
# old hand-picked 7/10 threshold with a statistical test anchored on the
# teacher’s own termination rate on the same prompts. A student fails only
# when its Wilson 95% lower bound on termination rate is strictly below the
# teacher’s Wilson 95% lower bound minus ``THINK_PROBE_WILSON_MARGIN``.
THINK_PROBE_WILSON_Z = float(os.environ.get("THINK_PROBE_WILSON_Z", "1.96"))
THINK_PROBE_WILSON_MARGIN = float(os.environ.get("THINK_PROBE_WILSON_MARGIN", "0.10"))

# ── On-policy reverse-KL probe (T1.1/T1.2) ────────────────────────────
# Generates rollouts from the *student’s* own policy, scores them under
# both student and teacher logits, and reports the reverse KL
# D_KL(student || teacher) and a skew-KL D_KL(teacher || α·teacher + (1-α)·student).
# Reverse KL is the mode-seeking objective that distillation literature
# converges to (Gu et al. 2024 “MiniLLM”, Ko et al. 2024 “DistiLLM”), and
# skew-KL (Ko et al. 2024) is the theoretically-preferred upper bound that
# stays finite when student support drifts outside teacher support — which
# is exactly the failure case off-policy KL silently misses.
ON_POLICY_RKL_N_PROMPTS = int(os.environ.get("ON_POLICY_RKL_N_PROMPTS", "12"))
ON_POLICY_RKL_MAX_NEW = int(os.environ.get("ON_POLICY_RKL_MAX_NEW", "128"))
ON_POLICY_RKL_TEMPERATURE = float(os.environ.get("ON_POLICY_RKL_TEMPERATURE", "0.7"))
ON_POLICY_RKL_TOP_P = float(os.environ.get("ON_POLICY_RKL_TOP_P", "0.9"))
ON_POLICY_RKL_TOP_K_LOGITS = int(os.environ.get("ON_POLICY_RKL_TOP_K_LOGITS", "128"))
ON_POLICY_RKL_SKEW_ALPHA = float(os.environ.get("ON_POLICY_RKL_SKEW_ALPHA", "0.1"))
ON_POLICY_RKL_SEED = int(os.environ.get("ON_POLICY_RKL_SEED", "42"))
ON_POLICY_RKL_ENABLED = os.environ.get("ON_POLICY_RKL", "1") != "0"
ON_POLICY_RKL_PROMPTS = [
    "Explain how a transformer attention layer works, in one paragraph.",
    "What is 13 * 17? Show your reasoning.",
    "Write a haiku about autumn.",
    "List three causes of the French Revolution.",
    "Translate to French: The cat sat on the mat.",
    "What is the capital of Japan?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Is 97 prime? Answer with reasoning.",
    "Define machine learning in one sentence.",
    "Complete the sentence: The sky is blue because",
    "What is the derivative of x^2?",
    "In one sentence, explain why photosynthesis matters.",
    "Name a famous work by Mozart.",
    "What is the square root of 144?",
    "Who wrote Hamlet?",
    "Give a one-line summary of gradient descent.",
]

_CAPABILITY_STATIC_POOL = [
    {"q": "What is the capital of France? One word.", "a": "paris", "kind": "word"},
    {"q": "What is the capital of Japan? One word.", "a": "tokyo", "kind": "word"},
    {"q": "What is the capital of Italy? One word.", "a": "rome", "kind": "word"},
    {"q": "What is the capital of Egypt? One word.", "a": "cairo", "kind": "word"},
    {"q": "What is the capital of Brazil? One word.", "a": "brasilia", "kind": "word_alt", "alts": ["brasília"]},
    {"q": "What is the capital of Australia? One word.", "a": "canberra", "kind": "word"},
    {"q": "What is the capital of Canada? One word.", "a": "ottawa", "kind": "word"},
    {"q": "What is the capital of Kenya? One word.", "a": "nairobi", "kind": "word"},
    {"q": "Which continent is Egypt on? One word.", "a": "africa", "kind": "word"},
    {"q": "Which planet is closest to the sun? One word.", "a": "mercury", "kind": "word"},
    {"q": "Which planet is the largest in our solar system? One word.", "a": "jupiter", "kind": "word"},
    {"q": "Which planet is known for its rings? One word.", "a": "saturn", "kind": "word"},
    {"q": "What color do you get by mixing red and blue? One word.", "a": "purple", "kind": "word_alt", "alts": ["violet", "magenta"]},
    {"q": "What color do you get by mixing blue and yellow? One word.", "a": "green", "kind": "word"},
    {"q": "What color do you get by mixing red and yellow? One word.", "a": "orange", "kind": "word"},
    {"q": "Which month has exactly 28 or 29 days? One word.", "a": "february", "kind": "word"},
    {"q": "How many days are in a leap year? Answer with only the number.", "a": "366", "kind": "int"},
    {"q": "How many sides does a hexagon have? Answer with only the number.", "a": "6", "kind": "int"},
    {"q": "How many sides does a pentagon have? Answer with only the number.", "a": "5", "kind": "int"},
    {"q": "How many sides does an octagon have? Answer with only the number.", "a": "8", "kind": "int"},
    {"q": "How many hours are in 3 days? Answer with only the number.", "a": "72", "kind": "int"},
    {"q": "How many minutes are in 4 hours? Answer with only the number.", "a": "240", "kind": "int"},
    {"q": "How many centimeters are in 2 meters? Answer with only the number.", "a": "200", "kind": "int"},
    {"q": "Is 17 a prime number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 21 a prime number? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is 29 a prime number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 91 a prime number? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is 100 divisible by 4? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 50 divisible by 7? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is water a liquid at 20 degrees Celsius? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Does the sun rise in the west? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is a whale a mammal? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is a spider an insect? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "What is the largest ocean on Earth? One word.", "a": "pacific", "kind": "word"},
    {"q": "What gas do plants primarily absorb from the air? Two words.", "a": "carbon dioxide", "kind": "phrase", "accept_re": r"\bcarbon\s+dioxide\b|\bco2\b"},
    {"q": "What is the chemical symbol for gold? One word.", "a": "au", "kind": "word"},
    {"q": "What is the chemical symbol for water? One word.", "a": "h2o", "kind": "word"},
    {"q": "What is the name of the closest star to Earth? One word.", "a": "sun", "kind": "word"},
    {"q": "What is the freezing point of water in Celsius? Answer with only the number.", "a": "0", "kind": "int"},
    {"q": "What is the boiling point of water in Celsius at sea level? Answer with only the number.", "a": "100", "kind": "int"},
    # Instruction-following / format compliance (IFEval-style)
    {"q": "Reply with exactly the single word 'OK'.", "a": "ok", "kind": "word"},
    {"q": "Answer only 'YES' in all capital letters.", "a": "YES", "kind": "format_re", "accept_re": r"^YES\b"},
    {"q": "List three primary colors separated by commas, nothing else.", "kind": "format_re",
     "accept_re": r"^\s*(red|blue|yellow)\s*,\s*(red|blue|yellow)\s*,\s*(red|blue|yellow)\s*\.?\s*$"},
    {"q": "Answer with exactly three words (no more, no less) describing a cat.",
     "kind": "word_count", "count": 3},
    {"q": "Respond with a single lowercase word that rhymes with 'cat'.",
     "kind": "rhyme", "rhyme": "at", "lowercase": True},
    {"q": "Write the word 'hello' reversed (letter-by-letter). One word only.",
     "a": "olleh", "kind": "word"},
    {"q": "How many letters are in the word 'banana'? Answer with only the number.", "a": "6", "kind": "int"},
    {"q": "How many vowels are in the word 'education'? Answer with only the number.", "a": "5", "kind": "int"},
    # Multi-choice (answer letter only)
    {"q": "Which one is a mammal? A) shark B) eagle C) dolphin D) lizard. Respond with only the letter.",
     "a": "c", "kind": "mc"},
    {"q": "Which one is a noble gas? A) oxygen B) helium C) nitrogen D) hydrogen. Respond with only the letter.",
     "a": "b", "kind": "mc"},
    {"q": "Which one is a prime number? A) 15 B) 21 C) 25 D) 23. Respond with only the letter.",
     "a": "d", "kind": "mc"},
    {"q": "Which one was a Roman emperor? A) Napoleon B) Einstein C) Augustus D) Shakespeare. Respond with only the letter.",
     "a": "c", "kind": "mc"},
]

CAPABILITY_PROBE_MAX_TOKENS = int(os.environ.get("CAPABILITY_PROBE_MAX_TOKENS", "48"))
CAPABILITY_PROBE_N = int(os.environ.get("CAPABILITY_PROBE_N", "24"))
CAPABILITY_PROBE_N_PROC_MATH = int(os.environ.get("CAPABILITY_PROBE_N_PROC_MATH", "12"))
LENGTH_PENALTY_RATIO = float(os.environ.get("LENGTH_PENALTY_RATIO", "2.0"))


def _procedural_math_prompts(rng, n):
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div", "mod"])
        if kind == "add":
            a, b = rng.randint(17, 499), rng.randint(17, 499)
            out.append({"q": f"What is {a} + {b}? Answer with only the number.", "a": str(a + b), "kind": "int"})
        elif kind == "sub":
            a, b = rng.randint(100, 900), rng.randint(10, 99)
            out.append({"q": f"What is {a} - {b}? Answer with only the number.", "a": str(a - b), "kind": "int"})
        elif kind == "mul":
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            out.append({"q": f"What is {a} * {b}? Answer with only the number.", "a": str(a * b), "kind": "int"})
        elif kind == "div":
            b = rng.randint(2, 12)
            q = rng.randint(2, 25)
            a = b * q
            out.append({"q": f"What is {a} / {b}? Answer with only the number.", "a": str(q), "kind": "int"})
        else:
            a, b = rng.randint(20, 200), rng.randint(3, 9)
            out.append({"q": f"What is {a} mod {b}? Answer with only the number.", "a": str(a % b), "kind": "int"})
    return out


def build_capability_prompts(block_seed=None):
    """Return the per-round capability prompt list (static sample + procedural math).

    Determinism: seeded by ``block_seed`` so all validators compute the same
    prompts for a given round, yet the set rotates every round preventing
    memorization. If ``block_seed`` is None (local dev), fall back to a fixed
    seed for repeatability.
    """
    import random
    rng = random.Random(int(block_seed) if block_seed is not None else 20260418)
    pool = list(_CAPABILITY_STATIC_POOL)
    rng.shuffle(pool)
    k = min(CAPABILITY_PROBE_N, len(pool))
    sampled = pool[:k]
    sampled.extend(_procedural_math_prompts(rng, CAPABILITY_PROBE_N_PROC_MATH))
    return sampled


_CAPABILITY_BLOCK_SEED = None
CAPABILITY_PROBE_PROMPTS = build_capability_prompts(None)


def set_capability_block_seed(block_seed):
    """Regenerate CAPABILITY_PROBE_PROMPTS deterministically for this round."""
    global _CAPABILITY_BLOCK_SEED, CAPABILITY_PROBE_PROMPTS
    if block_seed is None or block_seed == _CAPABILITY_BLOCK_SEED:
        return
    _CAPABILITY_BLOCK_SEED = block_seed
    CAPABILITY_PROBE_PROMPTS = build_capability_prompts(block_seed)

_CHAT_PROBE_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_THINK_TRAIL = re.compile(r"^.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_NARRATIVE = re.compile(r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)", re.DOTALL)


def _strip_thinking_probe(text: str) -> str:
    if "<think>" in text:
        text = _CHAT_PROBE_THINK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = _CHAT_PROBE_THINK_TRAIL.sub("", text, count=1)
    if text.lstrip().startswith("Thinking Process:"):
        text = _CHAT_PROBE_NARRATIVE.sub("", text, count=1)
    return text.strip()


def chat_response_probe(model, tokenizer, device="cuda"):
    """Chat-collapse probe: detect students that can't terminate simple chat prompts.

    Off-policy distillation pathology flagged by Allan (SN97 Discord) + literature
    (arxiv 2502.07266 on CoT complexity mismatch, thinkingmachines.ai/blog/on-policy-distillation).
    Miners train students on long teacher reasoning rollouts; the 4B student learns
    to always think, but has no reliable stop — leaving the chat endpoint in a
    "Thinking Process:..." loop that never emits a user-facing answer.

    For a short list of trivial prompts, run the student's chat template with
    enable_thinking=False and require:
      - at least CHAT_PROBE_TERMINATE_THRESHOLD of prompts emit EOS inside
        CHAT_PROBE_MAX_TOKENS
      - at least CHAT_PROBE_TERMINATE_THRESHOLD produce non-empty content after
        stripping <think>...</think> and "Thinking Process:..." narration
    """
    stats = {
        "pass": True, "reason": "",
        "prompts_tested": 0, "prompts_terminated": 0,
        "prompts_non_empty": 0,
        "mean_gen_tokens": 0.0,
        "mean_content_chars": 0.0,
        "mean_reasoning_fraction": 0.0,
        "samples": [],
    }
    try:
        if tokenizer is None or model is None:
            return stats
        tpl_ok = getattr(tokenizer, "chat_template", None)
        if not tpl_ok:
            stats["reason"] = "probe_skip:no_chat_template"
            return stats
        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0

        was_training = model.training
        model.eval()
        terminated = 0
        non_empty = 0
        gen_tokens_acc = 0
        content_chars_acc = 0
        reasoning_frac_acc = 0.0

        with torch.no_grad():
            for prompt in CHAT_PROBE_PROMPTS:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids,
                        max_new_tokens=CHAT_PROBE_MAX_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=pad_id,
                        eos_token_id=eos_ids,
                        use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    did_terminate = (gen_len < CHAT_PROBE_MAX_TOKENS) or (
                        eos_ids is not None and int(new_ids[-1].item()) in eos_ids
                    )
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    stripped = _strip_thinking_probe(raw_text)
                    raw_len = max(1, len(raw_text))
                    reasoning_frac = 1.0 - (len(stripped) / raw_len)
                    if did_terminate:
                        terminated += 1
                    if len(stripped) >= CHAT_PROBE_MIN_ANSWER_CHARS:
                        non_empty += 1
                    gen_tokens_acc += gen_len
                    content_chars_acc += len(stripped)
                    reasoning_frac_acc += max(0.0, min(1.0, reasoning_frac))
                    stats["samples"].append({
                        "prompt": prompt, "gen_tokens": gen_len,
                        "terminated": did_terminate,
                        "content_chars": len(stripped),
                        "reasoning_frac": round(reasoning_frac, 3),
                        "content_preview": stripped[:80],
                    })
                except Exception as e:
                    stats["samples"].append({"prompt": prompt, "error": str(e)[:120]})

        n = max(1, len(CHAT_PROBE_PROMPTS))
        stats["prompts_tested"] = n
        stats["prompts_terminated"] = terminated
        stats["prompts_non_empty"] = non_empty
        stats["mean_gen_tokens"] = round(gen_tokens_acc / n, 1)
        stats["mean_content_chars"] = round(content_chars_acc / n, 1)
        stats["mean_reasoning_fraction"] = round(reasoning_frac_acc / n, 3)

        min_ok = CHAT_PROBE_TERMINATE_THRESHOLD * n - 1e-9
        if terminated < min_ok and non_empty < min_ok:
            stats["pass"] = False
            stats["reason"] = (
                f"chat_collapse:terminated={terminated}/{n} "
                f"non_empty={non_empty}/{n} "
                f"mean_gen={stats['mean_gen_tokens']:.0f} "
                f"mean_think_frac={stats['mean_reasoning_fraction']:.2f}"
            )

        if was_training:
            model.train()
        return stats
    except Exception as e:
        stats["reason"] = f"probe_error:{str(e)[:120]}"
        return stats


def _degeneracy_metrics(text: str) -> dict:
    """Principled per-sample degeneracy metrics. No magic thresholds.

    Refs:
      Holtzman 2019 (arXiv:1904.09751) — the three canonical degeneracy axes.
      Pillutla MAUVE (arXiv:2102.01454) — distributional-gap formalism.

    Metric choices (all model-agnostic, no extra models required):

    * ``gzip_ratio``     = len(gzip(text)) / len(text). Directly measures
                           Kolmogorov-style redundancy; looped "Wait wait wait"
                           compresses to a tiny fraction of its length.
    * ``distinct_k``     = |unique k-grams| / |total k-grams| for k=1,2,4.
                           Low-distinct-n is the classic Li & Jurafsky /
                           Vijayakumar diversity signal.
    * ``top_kgram_rate`` = (count of most-frequent 6-gram) / (#6-grams).
                           Normalized -- scale-free.
    * ``shannon_entropy`` of the byte distribution (bits/byte). Collapsed text
                           has sub-Zipfian byte entropy.

    All are cheap and deterministic. Compared against the teacher's own
    empirical distribution on identical prompts (§thinking_collapse_probe
    uses teacher's mu/sigma instead of a hand-picked threshold).
    """
    import gzip
    from collections import Counter
    if not text:
        return {"len": 0, "gzip_ratio": 1.0, "distinct_1": 0.0,
                "distinct_2": 0.0, "distinct_4": 0.0, "top_kgram_rate": 0.0,
                "byte_entropy": 0.0}
    raw = text.encode("utf-8", errors="replace")
    comp = gzip.compress(raw, compresslevel=6)
    gzip_ratio = len(comp) / max(1, len(raw))
    byte_counts = Counter(raw)
    n_bytes = sum(byte_counts.values())
    h = 0.0
    for c in byte_counts.values():
        p = c / n_bytes
        if p > 0:
            h -= p * math.log2(p)
    tokens = text.split()
    out = {"len": len(raw), "gzip_ratio": gzip_ratio, "byte_entropy": h}
    for k in (1, 2, 4):
        if len(tokens) < k:
            out[f"distinct_{k}"] = 0.0
            continue
        grams = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
        out[f"distinct_{k}"] = len(set(grams)) / max(1, len(grams))
    if len(tokens) >= 6:
        grams6 = [" ".join(tokens[i:i + 6]) for i in range(len(tokens) - 6 + 1)]
        top = max(Counter(grams6).values()) if grams6 else 0
        out["top_kgram_rate"] = top / max(1, len(grams6))
    else:
        out["top_kgram_rate"] = 0.0
    return out


def _robust_zscore(x: float, values: list[float]) -> float:
    """Modified z-score using median/MAD (Iglewicz & Hoaglin 1993).

    Robust to the teacher itself occasionally producing an odd sample; a single
    bad teacher rollout won't shift the threshold enough to let a degenerate
    student through.
    """
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    deviations = sorted(abs(v - med) for v in vals)
    mad = deviations[n // 2] if n % 2 else 0.5 * (deviations[n // 2 - 1] + deviations[n // 2])
    if mad < 1e-9:
        return 0.0
    return 0.6745 * (x - med) / mad


def _wilson_bounds(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score-interval for a binomial proportion (Wilson 1927).

    Returns (lower, upper) 95% CI (default z=1.96). More accurate than the
    normal approximation at small n and near-boundary rates (0 or 1), which
    is exactly the regime the think-probe operates in (n=32 prompts,
    termination rates ≳ 0.9 for sane models). Used to replace the fragile
    hand-picked 7/10 threshold with a teacher-anchored statistical test:
    a student passes iff its Wilson LB is within ``margin`` of the teacher's
    Wilson LB on the same prompt set.
    """
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _self_bleu_pairwise(texts: list[str], n: int = 4) -> float:
    """Cross-rollout 4-gram overlap: 1.0 = identical, 0.0 = disjoint.

    Catches "student memorized one answer and emits it regardless of prompt"
    — a real attack that a per-sample gzip/distinct-k test cannot see,
    because each individual answer looks fine in isolation. Pimentel et al.
    *Standardizing the Measurement of Text Diversity* (arXiv:2403.00553,
    IJCNLP 2025 demo) recommends Self-BLEU alongside compression for exactly
    this reason.

    Pairwise Jaccard-on-4-grams rather than corpus BLEU — 20× faster and
    equally informative at this scale. Returns mean over all pairs.
    """
    if len(texts) < 2:
        return 0.0
    grams = []
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            grams.append(set())
            continue
        grams.append({tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)})
    pairs = 0
    s = 0.0
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            if not grams[i] or not grams[j]:
                continue
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            if union > 0:
                s += inter / union
                pairs += 1
    return s / pairs if pairs else 0.0


def _extract_capability_answer(text: str, kind: str) -> str:
    """Pull the answer token(s) from a model generation.

    We aim for extraction tolerance: students sometimes emit <think> blocks,
    leading newlines, markdown, or verbose wrappers. Strip and search rather
    than demand exact formatting. For ``format_re`` and ``word_count`` we keep
    the full stripped text — the verifier handles whitespace/punctuation.
    """
    raw = _strip_thinking_probe(text or "").strip()
    if not raw:
        return ""
    if kind == "int":
        m = re.search(r"-?\d+", raw)
        return m.group(0) if m else ""
    if kind == "yesno":
        low = raw.lower()
        if re.search(r"\byes\b", low):
            return "yes"
        if re.search(r"\bno\b", low):
            return "no"
        return ""
    if kind == "mc":
        m = re.search(r"\b([A-Da-d])\b", raw)
        return m.group(1).lower() if m else ""
    if kind in ("format_re", "word_count", "rhyme", "phrase"):
        return raw
    m = re.search(r"[a-zA-Z]+", raw.lower())
    return m.group(0) if m else ""


def _capability_score_one(pred: str, item: dict) -> int:
    if not pred:
        return 0
    kind = item.get("kind", "word")
    if kind == "word_alt":
        accepted = {item["a"].lower()} | {a.lower() for a in item.get("alts", [])}
        return 1 if pred.lower() in accepted else 0
    if kind == "phrase":
        pat = item.get("accept_re")
        return 1 if pat and re.search(pat, pred.lower()) else 0
    if kind == "format_re":
        pat = item.get("accept_re")
        return 1 if pat and re.search(pat, pred) else 0
    if kind == "word_count":
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", pred)
        return 1 if len(words) == int(item.get("count", 0)) else 0
    if kind == "rhyme":
        suffix = item.get("rhyme", "")
        if not suffix:
            return 0
        words = re.findall(r"[A-Za-z]+", pred)
        if len(words) != 1:
            return 0
        w = words[0]
        if item.get("lowercase") and w != w.lower():
            return 0
        return 1 if w.lower().endswith(suffix.lower()) else 0
    if kind == "mc":
        return 1 if pred.lower() == item["a"].lower() else 0
    return 1 if pred.lower() == item["a"].lower() else 0


def capability_probe(model, tokenizer, device="cuda"):
    """Verifiable-rewards mini-battery.

    Grounded answers (arithmetic, yes/no, one-word facts) cannot be gamed by
    matching teacher logit shape — the answer is the answer. Zhou et al.
    *IFEval* (arXiv:2311.07911) and the Tülu 3 RLVR line (Lambert et al.
    arXiv:2411.15124) established this axis as the single most reward-hack-
    resistant signal for small models.

    Runs greedy generation at short max_new_tokens so cost is bounded: 10
    prompts × ~32 tokens ≈ 1–3 GPU-seconds, negligible next to the KL pass.
    Returns pass fraction and per-item breakdown for the UI.
    """
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    try:
        if tokenizer is None or model is None:
            return out
        if not getattr(tokenizer, "chat_template", None):
            return out

        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for item in CAPABILITY_PROBE_PROMPTS:
                msgs = [{"role": "user", "content": item["q"]}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids, max_new_tokens=CAPABILITY_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    pred = _extract_capability_answer(raw_text, item["kind"])
                    ok = _capability_score_one(pred, item)
                    out["items"].append({
                        "q": item["q"], "expected": item.get("a", item.get("kind", "")),
                        "kind": item["kind"],
                        "pred": pred, "ok": bool(ok),
                        "tail": raw_text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"q": item["q"], "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def _render_chat_prompt(tokenizer, user_text: str, enable_thinking: bool = False):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def prepare_teacher_probe_refs_hf(teacher, tokenizer, device="cuda"):
    """Run teacher on think-probe + capability-probe prompts while HF-loaded.

    Populates the two globals the student-side probes read. Doing this once
    per round amortizes teacher cost across all students and lets us run
    statistical comparisons (teacher_self_bleu, per-prompt correctness
    delta) that the single-student probe cannot do on its own.
    """
    think_samples = []
    cap_answers = []
    cap_gen_lens = []
    if tokenizer is None or teacher is None:
        return think_samples, cap_answers, cap_gen_lens
    try:
        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)

        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            for prompt in THINK_PROBE_PROMPTS:
                try:
                    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=True)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=THINK_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    think_samples.append(tokenizer.decode(new_ids, skip_special_tokens=True))
                except Exception as e:
                    print(f"[eval] Teacher think-probe prompt failed: {e}", flush=True)
            for item in CAPABILITY_PROBE_PROMPTS:
                try:
                    rendered = _render_chat_prompt(tokenizer, item["q"], enable_thinking=False)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=CAPABILITY_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    cap_answers.append(_extract_capability_answer(raw_text, item["kind"]))
                    cap_gen_lens.append(gen_len)
                except Exception as e:
                    print(f"[eval] Teacher capability prompt failed: {e}", flush=True)
                    cap_answers.append("")
                    cap_gen_lens.append(0)
        if was_training:
            teacher.train()
    except Exception as e:
        print(f"[eval] prepare_teacher_probe_refs_hf error: {e}", flush=True)
    return think_samples, cap_answers, cap_gen_lens


def prepare_teacher_probe_refs_vllm(tokenizer):
    """Same as the HF variant but using the live vLLM server. Greedy only."""
    import requests
    think_samples = []
    cap_answers = []
    cap_gen_lens = []
    if tokenizer is None:
        return think_samples, cap_answers, cap_gen_lens
    try:
        for prompt in THINK_PROBE_PROMPTS:
            try:
                rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=True)
                resp = requests.post(
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": rendered,
                        "max_tokens": THINK_PROBE_MAX_TOKENS,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    timeout=VLLM_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                think_samples.append(resp.json()["choices"][0]["text"])
            except Exception as e:
                print(f"[eval] vLLM teacher think-probe failed: {e}", flush=True)
        for item in CAPABILITY_PROBE_PROMPTS:
            try:
                rendered = _render_chat_prompt(tokenizer, item["q"], enable_thinking=False)
                resp = requests.post(
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": rendered,
                        "max_tokens": CAPABILITY_PROBE_MAX_TOKENS,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    timeout=VLLM_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["text"]
                cap_answers.append(_extract_capability_answer(txt, item["kind"]))
                try:
                    cap_gen_lens.append(len(tokenizer(txt, return_tensors="pt").input_ids[0]))
                except Exception:
                    cap_gen_lens.append(0)
            except Exception as e:
                print(f"[eval] vLLM teacher capability failed: {e}", flush=True)
                cap_answers.append("")
                cap_gen_lens.append(0)
    except Exception as e:
        print(f"[eval] prepare_teacher_probe_refs_vllm error: {e}", flush=True)
    return think_samples, cap_answers, cap_gen_lens


def thinking_collapse_probe(model, tokenizer, device="cuda", teacher_samples=None):
    """Degeneracy probe for off-policy CoT collapse — threshold-free design.

    Replaces the previous hand-picked 6-gram-repeat-15 threshold with a
    distributional comparison against the teacher's own output on the same
    prompts. A student fails only when its compression/distinct-k/top-kgram
    statistics lie >= THINK_PROBE_DEGEN_SIGMA robust MAD-z-scores outside the
    teacher's distribution, OR when it fails to terminate on trivial prompts.

    This is Holtzman et al. 2019 (arXiv:1904.09751) three-axis degeneracy
    (repetition / diversity / entropy) operationalized as a statistical test:
    the teacher defines what "natural" generation looks like at this length,
    and the student must stay within that regime. There is no magic number.

    Args:
      teacher_samples: optional list[str] of teacher's own continuations on
        the probe prompts. If None, falls back to a single-axis gzip floor
        (``THINK_PROBE_GZIP_FLOOR``) — looser but still principled since
        gzip ratio < 0.25 on plain text is statistically impossible without
        pathological repetition.
    """
    stats = {
        "pass": True, "reason": "",
        "prompts_tested": 0, "prompts_terminated": 0, "prompts_degenerate": 0,
        "mean_gen_tokens": 0.0,
        "student_metrics": [], "teacher_metrics": [], "samples": [],
    }
    try:
        if tokenizer is None or model is None:
            return stats
        if not getattr(tokenizer, "chat_template", None):
            stats["reason"] = "think_probe_skip:no_chat_template"
            return stats

        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0

        was_training = model.training
        model.eval()
        terminated = 0
        gen_tokens_acc = 0
        student_m = []
        samples = []

        with torch.no_grad():
            for prompt in THINK_PROBE_PROMPTS:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids, max_new_tokens=THINK_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    did_terminate = (gen_len < THINK_PROBE_MAX_TOKENS) or (
                        eos_ids is not None and int(new_ids[-1].item()) in eos_ids
                    )
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    m = _degeneracy_metrics(raw_text)
                    student_m.append(m)
                    if did_terminate:
                        terminated += 1
                    gen_tokens_acc += gen_len
                    samples.append({
                        "prompt": prompt, "gen_tokens": gen_len,
                        "terminated": did_terminate,
                        "gzip_ratio": round(m["gzip_ratio"], 3),
                        "distinct_4": round(m.get("distinct_4", 0.0), 3),
                        "top_6gram_rate": round(m.get("top_kgram_rate", 0.0), 3),
                        "tail": raw_text[-200:],
                        "_full_text": raw_text,
                    })
                    stats["prompts_tested"] += 1
                except Exception as e:
                    samples.append({"prompt": prompt, "error": str(e)[:120]})
                    continue

        stats["prompts_terminated"] = terminated
        n = max(1, stats["prompts_tested"])
        stats["mean_gen_tokens"] = gen_tokens_acc / n
        stats["student_metrics"] = student_m
        if teacher_samples:
            stats["teacher_metrics"] = [_degeneracy_metrics(t) for t in teacher_samples]
        stats["samples"] = samples

        student_texts = [s.get("_full_text", "") for s in samples if s.get("_full_text")]
        self_bleu = _self_bleu_pairwise(student_texts, n=4) if len(student_texts) >= 2 else 0.0
        stats["self_bleu_across_prompts"] = round(self_bleu, 3)
        for s in samples:
            s.pop("_full_text", None)

        teacher_self_bleu = 0.0
        if teacher_samples and len(teacher_samples) >= 2:
            teacher_self_bleu = _self_bleu_pairwise(teacher_samples, n=4)
        stats["teacher_self_bleu"] = round(teacher_self_bleu, 3)

        degenerate = 0
        for m in student_m:
            if teacher_samples and stats["teacher_metrics"]:
                gzip_z = _robust_zscore(m["gzip_ratio"],
                                        [t["gzip_ratio"] for t in stats["teacher_metrics"]])
                top_z = _robust_zscore(m.get("top_kgram_rate", 0.0),
                                       [t.get("top_kgram_rate", 0.0) for t in stats["teacher_metrics"]])
                is_degen = (gzip_z < -THINK_PROBE_DEGEN_SIGMA) or (top_z > THINK_PROBE_DEGEN_SIGMA)
            else:
                is_degen = m["gzip_ratio"] < THINK_PROBE_GZIP_FLOOR
            if is_degen:
                degenerate += 1
        stats["prompts_degenerate"] = degenerate

        # ── Teacher-anchored Wilson lower-bound test (T0.2) ──────────────
        # Pass/fail is decided by comparing the student's Wilson LB against
        # the teacher's Wilson LB on the *same* prompt set. This replaces
        # the old hand-picked 7/10 threshold that tripped mrchen's model
        # for a legitimate one-prompt difference.
        z = THINK_PROBE_WILSON_Z
        margin = THINK_PROBE_WILSON_MARGIN
        n_teach = len(stats.get("teacher_metrics") or [])
        teacher_term_rate = None
        if teacher_samples and n_teach > 0:
            # The teacher is assumed to terminate on all trivial probes. We
            # approximate teacher termination as n_teach successes / n_teach
            # trials (the teacher refs are pre-filtered for truncation
            # upstream; if we ever let truncated teacher refs through, they
            # would appear as extra tokens in teacher_samples which we do
            # not have directly, so we use n_teach as the optimistic prior
            # and subtract ``margin`` to stay conservative).
            teacher_term_rate = 1.0
            teach_term_lb, _ = _wilson_bounds(n_teach, n_teach, z=z)
        else:
            teach_term_lb = 1.0  # fall back to absolute threshold
        stud_term_lb, stud_term_ub = _wilson_bounds(terminated, n, z=z)
        # Floor ensures we still catch catastrophic non-termination even when
        # the teacher anchor is unavailable; the Wilson margin is the primary
        # gate when teacher stats are present.
        term_floor = THINK_PROBE_TERMINATE_THRESHOLD
        term_ok = stud_term_lb >= max(term_floor, teach_term_lb - margin)

        # Degeneracy uses the symmetric test: student degenerate-rate Wilson
        # UB must not exceed teacher's Wilson UB by more than ``margin``.
        teacher_degen = 0
        if teacher_samples and stats.get("teacher_metrics"):
            for m_t in stats["teacher_metrics"]:
                if m_t.get("gzip_ratio", 1.0) < THINK_PROBE_GZIP_FLOOR:
                    teacher_degen += 1
        teach_degen_lb, teach_degen_ub = _wilson_bounds(teacher_degen, max(1, n_teach), z=z)
        stud_degen_lb, stud_degen_ub = _wilson_bounds(degenerate, n, z=z)
        degen_ok = stud_degen_ub <= teach_degen_ub + margin if n_teach else (degenerate / n < 0.34)

        sb_threshold = max(THINK_PROBE_SELFBLEU_MAX, teacher_self_bleu + 0.10)
        self_bleu_ok = self_bleu < sb_threshold

        stats["wilson"] = {
            "z": z, "margin": margin, "n": n, "n_teacher": n_teach,
            "student_term_lb": round(stud_term_lb, 3),
            "student_term_ub": round(stud_term_ub, 3),
            "teacher_term_lb": round(teach_term_lb, 3),
            "student_degen_lb": round(stud_degen_lb, 3),
            "student_degen_ub": round(stud_degen_ub, 3),
            "teacher_degen_ub": round(teach_degen_ub, 3),
            "term_ok": term_ok, "degen_ok": degen_ok,
        }

        if not term_ok or not degen_ok or not self_bleu_ok:
            stats["pass"] = False
            reasons = []
            if not term_ok:
                reasons.append(
                    f"term_lb={stud_term_lb:.2f}<teach_lb-m={teach_term_lb - margin:.2f} "
                    f"(terminated={terminated}/{n})"
                )
            if not degen_ok:
                reasons.append(
                    f"degen_ub={stud_degen_ub:.2f}>teach_ub+m={teach_degen_ub + margin:.2f} "
                    f"(degenerate={degenerate}/{n})"
                )
            if not self_bleu_ok:
                reasons.append(
                    f"self_bleu={self_bleu:.2f}>{sb_threshold:.2f}"
                )
            stats["reason"] = (
                f"thinking_collapse:{','.join(reasons)} "
                f"mean_gen={stats['mean_gen_tokens']:.0f}"
            )

        if was_training:
            model.train()
        return stats
    except Exception as e:
        stats["reason"] = f"think_probe_error:{str(e)[:120]}"
        return stats


def on_policy_rollouts(student, tokenizer, device="cuda",
                       prompts: list | None = None,
                       n_prompts: int | None = None,
                       max_new: int | None = None,
                       top_k_logits: int | None = None,
                       temperature: float | None = None,
                       top_p: float | None = None,
                       seed: int | None = None) -> list[dict]:
    """Sample rollouts from student, capture its top-K logprobs per token.

    This is **Phase A** of the on-policy RKL probe. We run it while the
    student is still on-GPU (cheap — no second model load) and store:

    - ``full_ids``: prompt + sampled continuation (CPU tensor)
    - ``prompt_len``: length of the prompt portion
    - ``gen_len``: number of student-sampled tokens
    - ``sampled_logprobs``: log-prob the student assigned to the token it
      actually sampled, per generated position (shape [gen_len])
    - ``topk_idx``: student top-K vocab indices per generated position
      (shape [gen_len, K]) — used to gather teacher logprobs in Phase B
    - ``topk_logprobs``: student log-probs on ``topk_idx`` (shape [gen_len, K])
    - ``gen_tail``: decoded tail (for debugging / Discord-facing transparency)

    The teacher is re-loaded once after all students finish and each
    saved-rollout is scored by ``on_policy_rkl_score`` below to compute the
    actual RKL / SKL metrics. This two-phase design is how we avoid the
    VRAM cliff of holding teacher + student concurrently.
    """
    if prompts is None:
        prompts = ON_POLICY_RKL_PROMPTS[:n_prompts or ON_POLICY_RKL_N_PROMPTS]
    max_new = max_new or ON_POLICY_RKL_MAX_NEW
    top_k_logits = top_k_logits or ON_POLICY_RKL_TOP_K_LOGITS
    temperature = temperature or ON_POLICY_RKL_TEMPERATURE
    top_p = top_p or ON_POLICY_RKL_TOP_P
    seed = seed or ON_POLICY_RKL_SEED

    rollouts: list[dict] = []
    if student is None or tokenizer is None:
        return rollouts
    if not getattr(tokenizer, "chat_template", None):
        return rollouts

    eos_ids = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos_ids.append(i)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0

    was_training = student.training
    student.eval()
    try:
        for p_idx, prompt in enumerate(prompts):
            try:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    rendered = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    rendered = prompt
                ids = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=1024).input_ids.to(device)
                torch.manual_seed(seed + p_idx)
                if device == "cuda":
                    torch.cuda.manual_seed(seed + p_idx)
                with torch.no_grad():
                    out = student.generate(
                        ids, max_new_tokens=max_new,
                        do_sample=True, temperature=temperature, top_p=top_p,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                prompt_len = int(ids.shape[1])
                full_ids = out  # [1, prompt_len + gen_len]
                gen_len = int(full_ids.shape[1] - prompt_len)
                if gen_len <= 0:
                    continue
                with torch.no_grad():
                    s_logits = student(full_ids).logits.float()
                # Positions prompt_len-1 .. end-1 predict the generated tokens
                cont = s_logits[0, prompt_len - 1:-1, :]  # [gen_len, V]
                if cont.shape[0] == 0:
                    del s_logits
                    continue
                s_logp = F.log_softmax(cont, dim=-1)  # [gen_len, V]
                k = min(top_k_logits, s_logp.shape[-1])
                top_lp, top_idx = s_logp.topk(k, dim=-1)  # [gen_len, K]
                sampled_tokens = full_ids[0, prompt_len:prompt_len + gen_len].unsqueeze(-1)
                sampled_lp = s_logp.gather(-1, sampled_tokens).squeeze(-1)  # [gen_len]
                gen_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
                rollouts.append({
                    "prompt": prompt,
                    "full_ids": full_ids.detach().cpu(),
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    "sampled_logprobs": sampled_lp.detach().cpu(),
                    "topk_idx": top_idx.detach().cpu(),
                    "topk_logprobs": top_lp.detach().cpu(),
                    "gen_tail": gen_text[-200:],
                })
                del s_logits, s_logp, cont, top_lp, top_idx, sampled_tokens, sampled_lp
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[on-policy-rkl] rollout {p_idx} error: {str(e)[:120]}", flush=True)
                continue
    finally:
        if was_training:
            student.train()
    return rollouts


def on_policy_rkl_score(teacher, rollouts: list[dict], device: str = "cuda",
                        skew_alpha: float | None = None) -> dict:
    """Phase B: score stored student rollouts under teacher logits.

    For each rollout we compute per-position:
      - ``rkl = E_s[log s - log t]`` — reverse KL (mode-seeking, the
        objective on-policy distillation converges to).
      - ``fkl = E_t[log t - log s]`` — forward KL (what we currently
        ship as ``kl_global_avg``, kept for reference).
      - ``skl = D_KL(t || α·t + (1-α)·s)`` — Ko et al. 2024 skew-KL,
        the bounded proxy that fixes teacher-hacking pathologies.
      - ``sampled_gap = log p_s(y|x) - log p_t(y|x)`` on the student's
        sampled trajectory (a cheap single-sample RKL estimator).

    Because we stored only the student's top-K logits to keep the
    rollout cache compact, we compute RKL on the **top-K support** under
    student, approximating the full-vocab RKL. This is a conservative
    proxy: if the student puts mass outside its own top-K, that mass is
    already ≤ 1/K of the total and teacher-hacking cases we care about
    concentrate mass on a few tokens anyway.
    """
    if skew_alpha is None:
        skew_alpha = ON_POLICY_RKL_SKEW_ALPHA
    agg = {
        "n_rollouts": 0, "tokens": 0,
        "mean_rkl": float("nan"), "mean_fkl": float("nan"),
        "mean_skl": float("nan"), "mean_sampled_gap": float("nan"),
        "mean_gen_len": 0.0, "per_rollout": [],
    }
    if teacher is None or not rollouts:
        return agg

    was_training = teacher.training
    teacher.eval()
    try:
        per_rollout = []
        total_rkl = 0.0
        total_fkl = 0.0
        total_skl = 0.0
        total_gap = 0.0
        total_tokens = 0
        total_gen = 0
        for r_idx, r in enumerate(rollouts):
            try:
                full_ids = r["full_ids"].to(device)
                prompt_len = int(r["prompt_len"])
                topk_idx = r["topk_idx"].to(device)          # [gen_len, K]
                s_topk_lp = r["topk_logprobs"].to(device)    # [gen_len, K]
                sampled_lp_s = r["sampled_logprobs"].to(device)  # [gen_len]
                with torch.no_grad():
                    t_logits = teacher(full_ids).logits.float()
                t_cont = t_logits[0, prompt_len - 1:-1, :]   # [gen_len, V]
                if t_cont.shape[0] == 0:
                    continue
                gen_len = min(t_cont.shape[0], topk_idx.shape[0])
                t_cont = t_cont[:gen_len]
                topk_idx = topk_idx[:gen_len]
                s_topk_lp = s_topk_lp[:gen_len]
                sampled_lp_s = sampled_lp_s[:gen_len]

                t_logp_full = F.log_softmax(t_cont, dim=-1)   # [gen_len, V]
                t_topk_lp = t_logp_full.gather(-1, topk_idx)  # [gen_len, K]

                # Renormalize both distributions on the student top-K
                # support and compute the two KLs there. Bounded and
                # avoids the full-vocab sum (which dominates runtime
                # when V≈150k).
                s_lp_norm = F.log_softmax(s_topk_lp, dim=-1)
                t_lp_norm = F.log_softmax(t_topk_lp, dim=-1)
                s_p_norm = s_lp_norm.exp()
                t_p_norm = t_lp_norm.exp()

                rkl_t = (s_p_norm * (s_lp_norm - t_lp_norm)).sum(-1)   # [gen_len]
                fkl_t = (t_p_norm * (t_lp_norm - s_lp_norm)).sum(-1)   # [gen_len]
                # Skew KL: D_KL(t || α t + (1-α) s)
                mix = skew_alpha * t_p_norm + (1.0 - skew_alpha) * s_p_norm
                mix_logp = torch.log(mix.clamp(min=1e-20))
                skl_t = (t_p_norm * (t_lp_norm - mix_logp)).sum(-1)    # [gen_len]

                # Sampled-trajectory gap (single-sample RKL). Needs the
                # teacher's logprob at the token the student actually
                # emitted — which lives in the full vocab, not just the
                # student's top-K, so compute from t_logp_full.
                sampled_tokens = full_ids[0, prompt_len:prompt_len + gen_len].unsqueeze(-1)
                sampled_lp_t = t_logp_full.gather(-1, sampled_tokens).squeeze(-1)
                gap = sampled_lp_s - sampled_lp_t  # [gen_len]

                mean_rkl = float(rkl_t.mean().item())
                mean_fkl = float(fkl_t.mean().item())
                mean_skl = float(skl_t.mean().item())
                mean_gap = float(gap.mean().item())
                tokens = int(rkl_t.shape[0])

                per_rollout.append({
                    "prompt": r.get("prompt", "")[:160],
                    "gen_len": r.get("gen_len", tokens),
                    "tokens_scored": tokens,
                    "rkl": round(mean_rkl, 6),
                    "fkl": round(mean_fkl, 6),
                    "skl": round(mean_skl, 6),
                    "sampled_gap": round(mean_gap, 6),
                    "tail": r.get("gen_tail", ""),
                })
                total_rkl += mean_rkl * tokens
                total_fkl += mean_fkl * tokens
                total_skl += mean_skl * tokens
                total_gap += mean_gap * tokens
                total_tokens += tokens
                total_gen += int(r.get("gen_len", tokens))
                del t_logits, t_cont, t_logp_full, t_topk_lp
                del s_lp_norm, t_lp_norm, s_p_norm, t_p_norm, mix, mix_logp
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[on-policy-rkl] score rollout {r_idx} error: {str(e)[:120]}", flush=True)
                continue

        if total_tokens > 0:
            agg["n_rollouts"] = len(per_rollout)
            agg["tokens"] = total_tokens
            agg["mean_rkl"] = total_rkl / total_tokens
            agg["mean_fkl"] = total_fkl / total_tokens
            agg["mean_skl"] = total_skl / total_tokens
            agg["mean_sampled_gap"] = total_gap / total_tokens
            agg["mean_gen_len"] = total_gen / max(1, len(per_rollout))
        agg["per_rollout"] = per_rollout
        agg["skew_alpha"] = skew_alpha
        agg["top_k"] = int(rollouts[0]["topk_idx"].shape[-1]) if rollouts else None
    finally:
        if was_training:
            teacher.train()
    return agg


def compute_activation_fingerprint(model, device="cuda"):
    """
    Compute an activation-space fingerprint for functional copy detection.

    Runs a fixed set of random token sequences through the model and collects
    hidden states at evenly-spaced layers. The fingerprint is a list of mean
    activation vectors (one per layer checkpoint), which is invariant to
    intra-layer weight reparameterization (e.g. scaling V by c, O by 1/c).

    Returns dict with:
      - "layer_fingerprints": {layer_idx: [float, ...]}  (mean hidden state per layer)
      - "n_layers": int
      - "hidden_size": int
    """
    try:
        # Generate deterministic random inputs
        rng = torch.Generator(device=device)
        rng.manual_seed(ACTIVATION_FP_SEED)
        input_ids = torch.randint(
            0, ACTIVATION_FP_VOCAB_SIZE,
            (ACTIVATION_FP_N_INPUTS, ACTIVATION_FP_SEQ_LEN),
            generator=rng, device=device
        )

        # Determine layer checkpoints (4 evenly spaced)
        n_layers = model.config.num_hidden_layers
        checkpoints = sorted(set([
            0,
            n_layers // 3,
            2 * n_layers // 3,
            n_layers - 1,
        ]))

        # Collect activations via hooks
        activations = {idx: [] for idx in checkpoints}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Mean over batch and sequence dimensions → single vector
                activations[layer_idx].append(h.float().mean(dim=(0, 1)).detach().cpu())
            return hook_fn

        # Find the layers module
        layers_module = None
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers_module = obj
                break
            except AttributeError:
                continue

        if layers_module is None:
            print("[fingerprint] Could not find layers module", flush=True)
            return None

        for idx in checkpoints:
            if idx < len(layers_module):
                hooks.append(layers_module[idx].register_forward_hook(make_hook(idx)))

        # Forward pass (no grad)
        with torch.no_grad():
            for i in range(ACTIVATION_FP_N_INPUTS):
                _ = model(input_ids[i:i+1])

        # Remove hooks
        for h in hooks:
            h.remove()

        # Average across all inputs per layer
        layer_fingerprints = {}
        for idx in checkpoints:
            if activations[idx]:
                avg = torch.stack(activations[idx]).mean(dim=0)
                # Truncate to first 128 dims to keep payload small
                fp = avg[:128].tolist()
                layer_fingerprints[str(idx)] = [round(v, 6) for v in fp]

        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0

        # Cleanup
        del input_ids, activations
        torch.cuda.empty_cache()

        print(f"[fingerprint] Computed: {len(layer_fingerprints)} layers, "
              f"n_layers={n_layers}, hidden_size={hidden_size}", flush=True)
        return {
            "layer_fingerprints": layer_fingerprints,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
        }
    except Exception as e:
        print(f"[fingerprint] Error: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §5  KL Computation
# ═══════════════════════════════════════════════════════════════════════════════

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
        pad = torch.full((*s_logits.shape[:-1], t_vocab - s_vocab), -1e10,
                         device=s_logits.device, dtype=s_logits.dtype)
        s_logits = torch.cat([s_logits, pad], dim=-1)
    elif s_vocab > t_vocab:
        s_logits = s_logits[..., :t_vocab]
    s_log_p = F.log_softmax(s_logits, dim=-1)
    # Chunked KL over positions
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


# ── Sparse KL helpers ──

def _build_token_to_id_map(tokenizer):
    """Build mapping from token text to token ID for vLLM logprobs decoding."""
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    text_to_id = {}
    for tok_str, tok_id in vocab.items():
        # Store raw vocab entry
        text_to_id[tok_str] = tok_id
        # Store decoded form as fallback
        decoded = tokenizer.decode([tok_id])
        if decoded not in text_to_id:
            text_to_id[decoded] = tok_id
    return text_to_id


def _is_sparse_logits(entry):
    """Check whether a teacher_logits_list entry is sparse (dict) or dense (tensor)."""
    return isinstance(entry, dict) and "indices" in entry and "values" in entry


def vllm_logprobs_to_sparse(top_logprobs_list, token_to_id, tokenizer, k=128):
    """Convert vLLM top_logprobs response to sparse tensor format.

    Args:
        top_logprobs_list: list of dicts (one per generated token),
            each mapping token_str -> logprob for top-k tokens.
        token_to_id: pre-built mapping from token text to token ID.
        tokenizer: tokenizer for fallback encoding.
        k: number of top logprobs to keep per position.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are logprobs (from vLLM log-softmax).
    """
    seq_len = len(top_logprobs_list)
    indices = torch.zeros(1, seq_len, k, dtype=torch.long)
    values = torch.full((1, seq_len, k), -100.0, dtype=torch.float32)

    for pos, top_lp in enumerate(top_logprobs_list):
        sorted_items = sorted(top_lp.items(), key=lambda x: x[1], reverse=True)[:k]
        for j, (token_str, logprob) in enumerate(sorted_items):
            token_id = token_to_id.get(token_str)
            if token_id is None:
                try:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    token_id = encoded[0] if encoded else 0
                except Exception:
                    token_id = 0
            indices[0, pos, j] = token_id
            values[0, pos, j] = logprob

    return {"indices": indices, "values": values}


def dense_to_sparse_topk(logits, k=128):
    """Convert dense logits tensor to sparse top-k format.

    Args:
        logits: [1, seq_len, vocab_size] or [seq_len, vocab_size] tensor.
        k: number of top logits to keep.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are raw logits (not log-softmax).
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    topk_values, topk_indices = logits.float().topk(k, dim=-1)
    return {"indices": topk_indices.cpu(), "values": topk_values.cpu()}


def compute_kl_from_sparse(teacher_indices, teacher_values, student_logits,
                           values_are_logprobs=False):
    """KL divergence using sparse top-k teacher logits/logprobs.

    For teacher: renormalize the top-k values into a proper distribution.
    For student: compute full-vocab log_softmax, gather the same k positions,
    then renormalize over those k positions.
    KL = sum_k P_teacher_k * (log P_teacher_k - log Q_student_k)

    Args:
        teacher_indices: [1, seq_len, k] token IDs.
        teacher_values:  [1, seq_len, k] logits or logprobs.
        student_logits:  [1, seq_len, vocab_size] raw student logits.
        values_are_logprobs: if True, teacher_values are already log-probs.

    Returns:
        KL per position, shape [1, seq_len] or [seq_len].
    """
    device = student_logits.device
    t_idx = teacher_indices.to(device)
    t_vals = teacher_values.to(device).float()

    # Teacher: renormalized log-probs over top-k tokens
    if values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)

    # Student: full-vocab log_softmax, then gather the k positions
    s_log_p_full = F.log_softmax(student_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_idx)
    # Renormalize student over the same k tokens for proper KL
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)

    del s_log_p_full

    # Chunked KL over positions for memory efficiency
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = F.kl_div(
                s_log_p_k_norm[:, i:j, :], t_log_p[:, i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)
    else:
        kl_per_pos = torch.empty(n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = F.kl_div(
                s_log_p_k_norm[i:j, :], t_log_p[i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)

    return kl_per_pos


# ═══════════════════════════════════════════════════════════════════════════════
# §6  vLLM Server Management
# ═══════════════════════════════════════════════════════════════════════════════

def start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=16384, revision=None, tensor_parallel_size=1, _attempt=1):
    """Start vLLM server via subprocess. Returns True on success. Retries once on crash."""
    ensure_disk_space(model_name, threshold=80)

    tp_note = f" TP={tensor_parallel_size}" if tensor_parallel_size and tensor_parallel_size > 1 else ""
    print(f"\n[vllm] Starting server for {model_name}{tp_note}..." + (f" (attempt {_attempt})" if _attempt > 1 else ""), flush=True)
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
        "--reasoning-parser", "qwen3",
        "--max-logprobs", "128",
    ]
    if tensor_parallel_size and tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if revision and revision != "main":
        cmd.extend(["--revision", revision])

    log_f = open("/tmp/vllm_teacher.log", "w")
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    Path("/tmp/vllm_teacher.pid").write_text(str(proc.pid))
    print(f"[vllm] PID: {proc.pid}", flush=True)

    import requests
    start_time = time.time()
    while time.time() - start_time < VLLM_STARTUP_TIMEOUT:
        elapsed = int(time.time() - start_time)
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
            if _attempt < 2:
                print("[vllm] Retrying after cleanup...", flush=True)
                stop_vllm_server()
                time.sleep(5)
                return start_vllm_server(model_name, gpu_memory_utilization, max_model_len, revision, tensor_parallel_size, _attempt=2)
            return False
        if elapsed % 60 == 0 and elapsed > 0:
            print(f"[vllm] Still starting... ({elapsed}s)", flush=True)
        time.sleep(5)

    print(f"[vllm] Timeout after {VLLM_STARTUP_TIMEOUT}s", flush=True)
    stop_vllm_server()
    return False


def stop_vllm_server():
    """Kill vLLM server, orphaned engine procs, and free VRAM."""
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
    try:
        subprocess.run(["fuser", "-k", f"{VLLM_PORT}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True, timeout=5)
    except Exception:
        pass
    my_pid = os.getpid()
    for _ in range(3):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            candidates = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip().isdigit() and int(p.strip()) != my_pid]
            if not candidates:
                break
            killed_any = False
            for pid in candidates:
                try:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
                except Exception:
                    cmdline = ""
                if any(tag in cmdline for tag in ("vllm.entrypoints", "vllm/engine", "VllmWorker")):
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_any = True
                    except Exception:
                        pass
            if not killed_any:
                break
            time.sleep(2)
        except Exception:
            break
    try:
        for shm in Path("/dev/shm").glob("vllm*"):
            shm.unlink(missing_ok=True)
    except Exception:
        pass
    free_gpu()
    time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  vLLM Generation (teacher continuations + logprobs)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_single_prompt(idx, prompt_text, max_new_tokens, block_seed,
                            logprobs_k, tokenizer, token_to_id):
    """Generate a single prompt via vLLM. Used by both sequential and concurrent paths."""
    import requests

    payload = {
        "model": "teacher",
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if block_seed is not None else 0.0,
        "top_p": 0.9 if block_seed is not None else 1.0,
        # repetition_penalty removed — wrecks KL scores without stopping loops
    }
    if block_seed is not None:
        payload["seed"] = block_seed + idx
    if logprobs_k > 0:
        payload["logprobs"] = logprobs_k
        payload["prompt_logprobs"] = 0

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{VLLM_URL}/v1/completions",
                json=payload,
                timeout=VLLM_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            cont_text = choice["text"]
            full_text = prompt_text + cont_text
            full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids
            result = {
                "full_ids": full_ids,
                "prompt_len": prompt_ids.shape[1],
                "gen_len": full_ids.shape[1] - prompt_ids.shape[1],
            }
            if logprobs_k > 0 and "logprobs" in choice and choice["logprobs"]:
                lp_data = choice["logprobs"]
                top_lp_list = lp_data.get("top_logprobs")
                if top_lp_list and token_to_id is not None:
                    result["sparse_logprobs"] = vllm_logprobs_to_sparse(
                        top_lp_list, token_to_id, tokenizer, k=logprobs_k
                    )
            return idx, result
        except Exception as e:
            if attempt < 2:
                print(
                    f"  [vllm] Prompt {idx} attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(2)
            else:
                raise RuntimeError(f"vLLM generation failed for prompt {idx}: {e}")


def generate_via_vllm(prompts, tokenizer, max_new_tokens, block_seed=None,
                      logprobs_k=128, token_to_id=None, progress_cb=None,
                      concurrency=4):
    """Generate teacher continuations via vLLM API.

    When logprobs_k > 0, requests top-k logprobs from vLLM per generated token
    and converts them to sparse tensor format, eliminating the need for a
    separate HF forward pass (Phase 1b).

    Uses concurrent requests to vLLM for significantly faster teacher generation.
    vLLM's continuous batching handles the internal scheduling — we just need to
    send multiple requests in parallel.

    Args:
        concurrency: Number of parallel requests to vLLM (default 4).
            vLLM batches these internally. Higher values use more VRAM for KV cache
            but significantly speed up total generation time.

    Returns list of dicts with full_ids, prompt_len, gen_len, and optionally
    'sparse_logprobs' (dict with 'indices' and 'values' tensors).
    """
    if concurrency <= 1:
        # Sequential fallback
        results = []
        for idx, prompt_text in enumerate(prompts):
            _, result = _generate_single_prompt(
                idx, prompt_text, max_new_tokens, block_seed,
                logprobs_k, tokenizer, token_to_id
            )
            results.append(result)
            if idx % 10 == 0 or idx == len(prompts) - 1:
                has_lp = "sparse_logprobs" in result
                print(f"  [{idx+1}/{len(prompts)}] {result['prompt_len']}+{result['gen_len']} tokens"
                      f"{' (logprobs✓)' if has_lp else ''}", flush=True)
            if progress_cb:
                progress_cb(idx + 1, len(prompts))
        return results

    # Concurrent generation
    print(f"  [vllm] Concurrent generation: {concurrency} parallel requests", flush=True)
    result_slots = [None] * len(prompts)
    completed = 0
    last_log = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for idx, prompt_text in enumerate(prompts):
            fut = executor.submit(
                _generate_single_prompt, idx, prompt_text, max_new_tokens,
                block_seed, logprobs_k, tokenizer, token_to_id
            )
            futures[fut] = idx

        failed = []
        for fut in as_completed(futures):
            orig_idx = futures[fut]
            try:
                _, result = fut.result()
                result_slots[orig_idx] = result
                completed += 1

                # Log every 10 completions or at the end
                if completed - last_log >= 10 or completed == len(prompts):
                    has_lp = "sparse_logprobs" in result
                    print(f"  [{completed}/{len(prompts)}] latest: {result['prompt_len']}+{result['gen_len']} tokens"
                          f"{' (logprobs✓)' if has_lp else ''}", flush=True)
                    last_log = completed

                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e:
                failed.append((orig_idx, str(e)))
                print(f"  [vllm] Prompt {orig_idx} failed: {e}", flush=True)

    # Retry failed prompts sequentially
    if failed:
        print(f"  [vllm] Retrying {len(failed)} failed prompts sequentially...", flush=True)
        for idx, err in failed:
            try:
                _, result = _generate_single_prompt(
                    idx, prompts[idx], max_new_tokens, block_seed,
                    logprobs_k, tokenizer, token_to_id
                )
                result_slots[idx] = result
                completed += 1
                print(f"  [vllm] Retry prompt {idx}: OK", flush=True)
                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e2:
                raise RuntimeError(f"vLLM generation failed for prompt {idx} after retry: {e2}")

    return result_slots


# ═══════════════════════════════════════════════════════════════════════════════
# §8  HF Batched Forward Pass
# ═══════════════════════════════════════════════════════════════════════════════

def hf_batched_forward(teacher, sequences_data, device, batch_size=2, logprobs_k=128,
                       progress_cb=None):
    """Batched HF forward pass for teacher logit extraction.

    Groups prompts by similar sequence length, pads within each batch,
    runs the forward pass, then unpads and extracts continuation logits.

    For sequences longer than HF_CHUNK_SIZE, falls back to per-sequence
    chunked forward pass (batching long-context sequences is wasteful due
    to extreme padding).

    Args:
        teacher: HF model (already on device, eval mode).
        sequences_data: list of dicts with 'full_ids' [1, seq_len], 'prompt_len'.
        device: torch device.
        batch_size: number of sequences per batch.
        logprobs_k: if >0, store only top-k logits (sparse). 0 = full vocab.
        progress_cb: optional callback(i) called after each prompt is processed.

    Returns:
        (teacher_logits_list, prompt_lens, full_sequences, n_chunked)
        teacher_logits_list: list of sparse dicts or dense tensors.
    """
    # Separate long sequences (chunked path) from short ones (batched path)
    short_items = []  # (original_idx, data)
    long_items = []   # (original_idx, data)
    for idx, data in enumerate(sequences_data):
        seq_len = data["full_ids"].shape[1]
        if seq_len > HF_CHUNK_SIZE:
            long_items.append((idx, data))
        else:
            short_items.append((idx, data))

    # Sort short items by sequence length for efficient batching
    short_items.sort(key=lambda x: x[1]["full_ids"].shape[1])

    # Pre-allocate result arrays
    n_total = len(sequences_data)
    teacher_logits_list = [None] * n_total
    prompt_lens = [0] * n_total
    full_sequences = [None] * n_total
    n_chunked = 0
    processed = 0

    with torch.no_grad():
        # ── Batched path for short sequences ──
        for batch_start in range(0, len(short_items), batch_size):
            batch = short_items[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                # Single item — no padding needed
                orig_idx, data = batch[0]
                full_ids = data["full_ids"].to(device)
                prompt_len = data["prompt_len"]
                prompt_lens[orig_idx] = prompt_len
                full_sequences[orig_idx] = full_ids

                logits = teacher(full_ids).logits.float()
                cont_logits = logits[:, prompt_len - 1:-1, :]
                if logprobs_k > 0:
                    teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                else:
                    teacher_logits_list[orig_idx] = cont_logits.cpu()
                del logits, cont_logits
                processed += 1
                if progress_cb:
                    progress_cb(processed)
            else:
                # Pad batch to max length
                max_len = max(d["full_ids"].shape[1] for _, d in batch)
                batch_ids = []
                attention_masks = []
                for _, data in batch:
                    ids = data["full_ids"]  # [1, seq_len]
                    seq_len = ids.shape[1]
                    if seq_len < max_len:
                        # Left-pad (standard for causal LMs)
                        pad_len = max_len - seq_len
                        pad = torch.zeros(1, pad_len, dtype=ids.dtype, device=ids.device)
                        ids_padded = torch.cat([pad, ids], dim=1)
                        mask = torch.cat([
                            torch.zeros(1, pad_len, dtype=torch.long, device=ids.device),
                            torch.ones(1, seq_len, dtype=torch.long, device=ids.device),
                        ], dim=1)
                    else:
                        ids_padded = ids
                        mask = torch.ones(1, seq_len, dtype=torch.long, device=ids.device)
                    batch_ids.append(ids_padded)
                    attention_masks.append(mask)

                batch_tensor = torch.cat(batch_ids, dim=0).to(device)  # [B, max_len]
                mask_tensor = torch.cat(attention_masks, dim=0).to(device)  # [B, max_len]

                outputs = teacher(batch_tensor, attention_mask=mask_tensor)
                batch_logits = outputs.logits.float()  # [B, max_len, vocab]
                del outputs

                # Unpad and extract continuation logits per item
                for b_idx, (orig_idx, data) in enumerate(batch):
                    full_ids = data["full_ids"].to(device)
                    prompt_len = data["prompt_len"]
                    seq_len = full_ids.shape[1]
                    pad_len = max_len - seq_len
                    prompt_lens[orig_idx] = prompt_len
                    full_sequences[orig_idx] = full_ids

                    # Extract this item's logits (remove padding offset)
                    item_logits = batch_logits[b_idx:b_idx+1, pad_len:, :]  # [1, seq_len, vocab]
                    cont_logits = item_logits[:, prompt_len - 1:-1, :]
                    if logprobs_k > 0:
                        teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                    else:
                        teacher_logits_list[orig_idx] = cont_logits.cpu()
                    del item_logits, cont_logits
                    processed += 1
                    if progress_cb:
                        progress_cb(processed)

                del batch_tensor, mask_tensor, batch_logits
                torch.cuda.empty_cache()

        # ── Chunked path for long sequences ──
        for orig_idx, data in long_items:
            n_chunked += 1
            full_ids = data["full_ids"].to(device)
            prompt_len = data["prompt_len"]
            seq_len = full_ids.shape[1]
            prompt_lens[orig_idx] = prompt_len
            full_sequences[orig_idx] = full_ids

            all_logit_chunks = []
            past_key_values = None
            for chunk_start in range(0, seq_len, HF_CHUNK_SIZE):
                chunk_end = min(chunk_start + HF_CHUNK_SIZE, seq_len)
                chunk_ids = full_ids[:, chunk_start:chunk_end]
                outputs = teacher(
                    chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                all_logit_chunks.append(outputs.logits.float().cpu())
                past_key_values = outputs.past_key_values
                del outputs
            all_logits = torch.cat(all_logit_chunks, dim=1)
            cont_logits = all_logits[:, prompt_len - 1:-1, :]
            if logprobs_k > 0:
                teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
            else:
                teacher_logits_list[orig_idx] = cont_logits.cpu()
            del all_logits, all_logit_chunks, past_key_values, cont_logits
            torch.cuda.empty_cache()
            processed += 1
            if progress_cb:
                progress_cb(processed)

    return teacher_logits_list, prompt_lens, full_sequences, n_chunked


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Progress Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _write_phase(progress_path, students, phase, teacher_done=None, **extra):
    """Write a phase update to the progress file for the dashboard."""
    try:
        data = {
            "phase": phase,
            "students": students,
            "students_total": len(students),
            "prompts_total": extra.get("prompts_total", 0),
            "teacher_prompts_done": teacher_done,
            "completed": extra.get("completed", []),
            "current": extra.get("current", None),
        }
        with open(progress_path, "w") as pf:
            json.dump(data, pf)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated SN97 evaluation v3")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--revisions", default=None, help="Comma-separated revisions matching --students order")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt texts")
    parser.add_argument("--output", default="/home/eval_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-logits", default="/home/teacher_cache.pt")
    parser.add_argument("--save-teacher-logits", default=None)
    parser.add_argument("--king", default=None, help="King model name — stays in VRAM between rounds")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use pure HF")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization (default 0.90)")
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--logprobs-k", type=int, default=128,
                        help="Top-k logprobs to store (128=sparse, 0=full vocab). Default 128.")
    parser.add_argument("--hf-batch-size", type=int, default=2,
                        help="Batch size for HF forward pass (default 2).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel teacher-gen requests to vLLM. Auto-bumps to 16 when --tensor-parallel-size > 1 unless overridden.")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="vLLM tensor parallel size for teacher. 0 = auto (use all visible GPUs). 1 = single GPU.")
    parser.add_argument("--early-stop-min", type=int, default=0,
                        help="Enable same-point CI early stopping after N prompts (0 = disabled).")
    # Accepted for backward compat with older in-memory validators. Ignored.
    parser.add_argument("--max-prompt-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-params-b", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_capability_block_seed(args.block_seed)

    # Auto-detect tensor-parallel size when unset (0 = all visible GPUs).
    # Allow override via DISTIL_TP_SIZE env var even when caller forgot the flag.
    env_tp = os.environ.get("DISTIL_TP_SIZE")
    if args.tensor_parallel_size == 0 and env_tp:
        try:
            args.tensor_parallel_size = max(1, int(env_tp))
        except ValueError:
            args.tensor_parallel_size = 0
    if args.tensor_parallel_size == 0:
        try:
            args.tensor_parallel_size = max(1, torch.cuda.device_count())
        except Exception:
            args.tensor_parallel_size = 1
    # Higher concurrency pays off only when the teacher is sharded.
    user_set_concurrency = any(a.startswith("--concurrency") for a in sys.argv[1:])
    if args.tensor_parallel_size > 1 and not user_set_concurrency:
        args.concurrency = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    # Parse revision pins (prevents weight-swap attacks between precheck and eval)
    if args.revisions:
        revisions = [r.strip() for r in args.revisions.split(",")]
        if len(revisions) != len(students):
            print(f"[eval] WARNING: {len(revisions)} revisions for {len(students)} students, ignoring revisions", flush=True)
            student_revisions = {s: "main" for s in students}
        else:
            student_revisions = dict(zip(students, revisions))
    else:
        student_revisions = {s: "main" for s in students}
    timings = {}

    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]

    # Progress file path
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")

    print(f"[eval] {len(prompts)} prompts (hash={prompts_hash}), {len(students)} students", flush=True)
    print(f"[eval] Teacher: {args.teacher}", flush=True)
    print(f"[eval] King: {args.king or 'none'}", flush=True)
    print(f"[eval] vLLM: {'disabled' if args.no_vllm else 'enabled'}", flush=True)
    print(f"[eval] VRAM: {gpu_mem_str()}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=False).input_ids.to(device)
        input_ids_list.append(ids)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher logits
    # ═══════════════════════════════════════════════════════════════════

    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    teacher_cache_loaded = False

    # Try loading from cache
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        try:
            t0 = time.time()
            cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
            if (len(cache.get("full_sequences", [])) == len(prompts)
                and cache.get("prompts_hash") == prompts_hash):
                full_sequences = [s.to(device) for s in cache["full_sequences"]]
                teacher_logits_list = cache["teacher_logits"]  # keep on CPU
                prompt_lens = cache["prompt_lens"]
                if cache.get("teacher_probe_samples"):
                    globals()["_TEACHER_PROBE_SAMPLES"] = cache["teacher_probe_samples"]
                cached_refs = cache.get("teacher_capability_refs")
                cached_seed = cache.get("teacher_capability_block_seed")
                if cached_refs and cached_seed == args.block_seed:
                    globals()["_TEACHER_CAPABILITY_REFS"] = cached_refs
                elif cached_refs:
                    print(f"[eval] Capability refs cache stale "
                          f"(seed {cached_seed} != {args.block_seed}); will regenerate", flush=True)
                timings["teacher_cache_load"] = time.time() - t0
                timings["teacher_generation"] = 0.0
                timings["teacher_logits_pass"] = 0.0
                print(f"[eval] ✓ Cached logits ({timings['teacher_cache_load']:.1f}s, "
                      f"method={cache.get('generation_method', '?')}, "
                      f"probe_refs={'yes' if cache.get('teacher_probe_samples') else 'no'})", flush=True)
                teacher_cache_loaded = True
            else:
                print(f"[eval] ✗ Cache stale — regenerating", flush=True)
        except Exception as e:
            print(f"[eval] ✗ Cache failed: {e}", flush=True)

    if not teacher_cache_loaded and not args.no_vllm:
        # ── vLLM generation path ──
        ensure_disk_space(args.teacher, threshold=70)

        # Clean stale teacher cache if hash doesn't match
        if args.teacher_logits and os.path.exists(args.teacher_logits):
            try:
                cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
                if cache.get("prompts_hash") != prompts_hash:
                    cache_size = os.path.getsize(args.teacher_logits) / (1024**3)
                    print(f"[disk] Stale teacher cache — removing ({cache_size:.1f}GB)", flush=True)
                    os.remove(args.teacher_logits)
                del cache
            except Exception:
                pass

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1a: vLLM teacher generation", flush=True)
        print(f"{'='*60}", flush=True)

        _write_phase(progress_path, students, "vllm_starting", prompts_total=len(prompts))
        t0 = time.time()
        vllm_ok = start_vllm_server(
            args.teacher, args.vllm_gpu_util, args.vllm_max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        timings["vllm_startup"] = time.time() - t0

        # Build token-to-id mapping for vLLM logprobs decoding
        token_to_id = _build_token_to_id_map(tokenizer) if args.logprobs_k > 0 else None

        sequences_data = None
        if vllm_ok:
            _write_phase(progress_path, students, "vllm_generating", prompts_total=len(prompts))
            t0 = time.time()
            def _vllm_progress(done, total):
                _write_phase(progress_path, students, "vllm_generating",
                             teacher_done=done, prompts_total=total)
            try:
                sequences_data = generate_via_vllm(
                    prompts, tokenizer, args.max_new_tokens, args.block_seed,
                    logprobs_k=args.logprobs_k, token_to_id=token_to_id,
                    progress_cb=_vllm_progress, concurrency=args.concurrency,
                )
                timings["vllm_generation"] = time.time() - t0
                print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] vLLM generation failed: {e} — falling back to HF", flush=True)

            # Probe-ref collection while vLLM is still hot (cheap: ~15 prompts).
            # Populates _TEACHER_PROBE_SAMPLES so the MAD-z degeneracy branch
            # activates, and _TEACHER_CAPABILITY_REFS so the shadow capability
            # axis has a reference.
            try:
                _tpr_t0 = time.time()
                think_refs, cap_answers, cap_gen_lens = prepare_teacher_probe_refs_vllm(tokenizer)
                globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
                globals()["_TEACHER_CAPABILITY_REFS"] = {
                    "answers": cap_answers, "gen_lens": cap_gen_lens,
                }
                timings["teacher_probe_refs"] = time.time() - _tpr_t0
                print(f"[eval] Teacher probe refs via vLLM: "
                      f"{len(think_refs)} think + {len(cap_answers)} cap "
                      f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[eval] Teacher probe refs (vLLM) failed: {e}", flush=True)
            stop_vllm_server()
        else:
            print(f"[eval] vLLM failed to start — falling back to HF", flush=True)

        if sequences_data:
            # Check if vLLM returned logprobs for all prompts
            has_vllm_logprobs = all("sparse_logprobs" in d for d in sequences_data)

            if has_vllm_logprobs:
                # ── vLLM logprobs path: skip Phase 1b entirely ──
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: SKIPPED — vLLM logprobs available (top-{args.logprobs_k})", flush=True)
                print(f"{'='*60}", flush=True)

                for i, data in enumerate(sequences_data):
                    full_ids = data["full_ids"].to(device)
                    prompt_lens.append(data["prompt_len"])
                    full_sequences.append(full_ids)
                    teacher_logits_list.append(data["sparse_logprobs"])  # sparse dict on CPU

                timings["teacher_logits_pass"] = 0.0  # included in vllm_generation
                timings["teacher_hf_load"] = 0.0
                print(f"[eval] ✓ {len(sequences_data)} prompts with top-{args.logprobs_k} logprobs from vLLM", flush=True)
            else:
                # ── HF forward pass fallback for logits (batched) ──
                n_with_lp = sum(1 for d in sequences_data if "sparse_logprobs" in d)
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: HF teacher logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
                if n_with_lp > 0:
                    print(f"  ({n_with_lp}/{len(sequences_data)} had vLLM logprobs — using HF for all for consistency)", flush=True)
                print(f"{'='*60}", flush=True)

                ensure_disk_space(args.teacher, threshold=70)
                t0 = time.time()
                teacher = load_model(args.teacher, device)
                teacher.eval()
                timings["teacher_hf_load"] = time.time() - t0
                print(f"[eval] HF teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

                _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

                def _hf_progress(n_done):
                    if n_done % 10 == 0 or n_done == len(sequences_data):
                        print(f"  Logits [{n_done}/{len(sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
                    _write_phase(progress_path, students, "teacher_logits",
                                 teacher_done=n_done, prompts_total=len(prompts))

                t0 = time.time()
                teacher_logits_list, prompt_lens_out, full_seqs_out, n_chunked = hf_batched_forward(
                    teacher, sequences_data, device,
                    batch_size=args.hf_batch_size,
                    logprobs_k=args.logprobs_k,
                    progress_cb=_hf_progress,
                )
                prompt_lens = prompt_lens_out
                full_sequences = full_seqs_out

                if n_chunked:
                    print(f"[eval] Chunked forward pass used for {n_chunked}/{len(sequences_data)} sequences (chunk_size={HF_CHUNK_SIZE})", flush=True)

                timings["teacher_logits_pass"] = time.time() - t0
                print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)

                # Probe-ref collection while HF teacher is still loaded.
                try:
                    _tpr_t0 = time.time()
                    think_refs, cap_answers, cap_gen_lens = prepare_teacher_probe_refs_hf(
                        teacher, tokenizer, device,
                    )
                    globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
                    globals()["_TEACHER_CAPABILITY_REFS"] = {
                        "answers": cap_answers, "gen_lens": cap_gen_lens,
                    }
                    timings["teacher_probe_refs"] = time.time() - _tpr_t0
                    print(f"[eval] Teacher probe refs via HF: "
                          f"{len(think_refs)} think + {len(cap_answers)} cap "
                          f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
                except Exception as e:
                    print(f"[eval] Teacher probe refs (HF in vLLM branch) failed: {e}", flush=True)

                # Unload teacher
                del teacher
                free_gpu()
                print(f"[eval] Teacher unloaded. VRAM: {gpu_mem_str()}", flush=True)

            del sequences_data

            # Save cache if requested and enough disk
            if args.save_teacher_logits:
                st = os.statvfs(os.path.dirname(args.save_teacher_logits) or '/')
                free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
                # Sparse caches are tiny (~tens of MB) — lower threshold
                min_free = 5 if args.logprobs_k > 0 else 50
                if free_gb > min_free:
                    cache_tmp = args.save_teacher_logits + ".tmp"
                    gen_method = "vllm_logprobs" if has_vllm_logprobs else "vllm+hf"
                    torch.save({
                        "full_sequences": [s.cpu() for s in full_sequences],
                        "teacher_logits": teacher_logits_list,
                        "prompt_lens": prompt_lens,
                        "block_seed": args.block_seed,
                        "prompts_hash": prompts_hash,
                        "generation_method": gen_method,
                        "logprobs_k": args.logprobs_k,
                        "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
                        "teacher_probe_samples": globals().get("_TEACHER_PROBE_SAMPLES", []),
                        "teacher_capability_refs": globals().get("_TEACHER_CAPABILITY_REFS", {}),
                        "teacher_capability_block_seed": args.block_seed,
                    }, cache_tmp)
                    os.replace(cache_tmp, args.save_teacher_logits)
                    cache_size = os.path.getsize(args.save_teacher_logits) / (1024**2)
                    print(f"[eval] Cache saved ({cache_size:.1f}MB, method={gen_method})", flush=True)
                else:
                    print(f"[eval] Skipped cache save ({free_gb:.0f}GB free, need {min_free}GB)", flush=True)

            teacher_cache_loaded = True

    if not teacher_cache_loaded:
        # ── Pure HF fallback ──
        # Safety: ensure vLLM is killed before loading HF teacher
        stop_vllm_server()
        time.sleep(3)
        free_gpu()

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1 FALLBACK: HF teacher generation + logit extraction", flush=True)
        print(f"{'='*60}", flush=True)

        ensure_disk_space(args.teacher, threshold=70)

        t0 = time.time()
        teacher = load_model(args.teacher, device)
        teacher.eval()
        timings["teacher_hf_load"] = time.time() - t0
        print(f"[eval] Teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        _write_phase(progress_path, students, "teacher_generation", teacher_done=0, prompts_total=len(prompts))

        # Step 1: Generate continuations
        t0 = time.time()
        hf_sequences_data = []
        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                prompt_len = ids.shape[1]
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True)
                if args.block_seed is not None:
                    torch.manual_seed(args.block_seed + i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(args.block_seed + i)
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    gen_kwargs.update(do_sample=False)
                output_ids = teacher.generate(ids, **gen_kwargs)
                gen_len = output_ids.shape[1] - prompt_len
                hf_sequences_data.append({
                    "full_ids": output_ids,
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                })
                print(f"  Gen [{i+1}/{len(input_ids_list)}]: {prompt_len}+{gen_len} tokens, VRAM: {gpu_mem_str()}", flush=True)
                _write_phase(progress_path, students, "teacher_generation", teacher_done=i + 1, prompts_total=len(prompts))
        timings["teacher_generation"] = time.time() - t0

        # Step 2: Batched forward pass for logit extraction
        print(f"\n[eval] HF logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
        _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

        def _hf_fb_progress(n_done):
            if n_done % 10 == 0 or n_done == len(hf_sequences_data):
                print(f"  Logits [{n_done}/{len(hf_sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
            _write_phase(progress_path, students, "teacher_logits",
                         teacher_done=n_done, prompts_total=len(prompts))

        t0_logits = time.time()
        teacher_logits_list, prompt_lens, full_sequences, n_chunked = hf_batched_forward(
            teacher, hf_sequences_data, device,
            batch_size=args.hf_batch_size,
            logprobs_k=args.logprobs_k,
            progress_cb=_hf_fb_progress,
        )
        timings["teacher_logits_pass"] = time.time() - t0_logits
        if n_chunked:
            print(f"[eval] Chunked forward pass used for {n_chunked}/{len(hf_sequences_data)} sequences", flush=True)
        del hf_sequences_data

        try:
            _tpr_t0 = time.time()
            think_refs, cap_answers, cap_gen_lens = prepare_teacher_probe_refs_hf(
                teacher, tokenizer, device,
            )
            globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
            globals()["_TEACHER_CAPABILITY_REFS"] = {
                "answers": cap_answers, "gen_lens": cap_gen_lens,
            }
            timings["teacher_probe_refs"] = time.time() - _tpr_t0
            print(f"[eval] Teacher probe refs via HF: "
                  f"{len(think_refs)} think + {len(cap_answers)} cap "
                  f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
        except Exception as e:
            print(f"[eval] Teacher probe refs (pure HF) failed: {e}", flush=True)

        cache_path = args.save_teacher_logits or os.path.join(os.path.dirname(args.output), "teacher_cache.pt")
        torch.save({
            "full_sequences": [s.cpu() for s in full_sequences],
            "teacher_logits": teacher_logits_list,
            "prompt_lens": prompt_lens,
            "block_seed": args.block_seed,
            "prompts_hash": prompts_hash,
            "generation_method": "hf",
            "logprobs_k": args.logprobs_k,
            "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
            "teacher_probe_samples": globals().get("_TEACHER_PROBE_SAMPLES", []),
            "teacher_capability_refs": globals().get("_TEACHER_CAPABILITY_REFS", {}),
            "teacher_capability_block_seed": args.block_seed,
        }, cache_path)

        del teacher
        free_gpu()
        print(f"[eval] HF generation done in {timings['teacher_generation']:.1f}s, logits in {timings['teacher_logits_pass']:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1c: Chunked GPU processing setup
    # ═══════════════════════════════════════════════════════════════════
    _write_phase(progress_path, students, "gpu_precompute", teacher_done=len(prompts), prompts_total=len(prompts))
    n_sparse = sum(1 for tl in teacher_logits_list if _is_sparse_logits(tl))
    storage_mode = "sparse" if n_sparse == len(teacher_logits_list) else ("mixed" if n_sparse > 0 else "dense")
    print(f"\n[eval] Teacher logits: {len(teacher_logits_list)} prompts on CPU ({storage_mode}, {n_sparse}/{len(teacher_logits_list)} sparse)", flush=True)
    # Keep teacher logits on CPU, compute softmax per-chunk during student scoring
    teacher_log_probs = None
    teacher_probs = None

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1d: Filter short completions & log completion lengths
    # ═══════════════════════════════════════════════════════════════════
    completion_lens = [full_sequences[i].shape[1] - prompt_lens[i] for i in range(len(prompts))]
    min_cl, max_cl = min(completion_lens), max(completion_lens)
    avg_cl = sum(completion_lens) / len(completion_lens)
    print(f"[eval] Completion tokens: min={min_cl} max={max_cl} avg={avg_cl:.0f} across {len(prompts)} prompts", flush=True)

    # Filter out prompts with fewer than MIN_COMPLETION_TOKENS
    n_filtered = 0
    if MIN_COMPLETION_TOKENS > 0:
        keep_indices = [i for i, cl in enumerate(completion_lens) if cl >= MIN_COMPLETION_TOKENS]
        n_filtered = len(prompts) - len(keep_indices)
        if n_filtered > 0:
            skipped_lens = [completion_lens[i] for i in range(len(prompts)) if i not in set(keep_indices)]
            print(f"[eval] Filtered {n_filtered}/{len(prompts)} prompts with <{MIN_COMPLETION_TOKENS} completion tokens "
                  f"(skipped lens: {skipped_lens})", flush=True)
            full_sequences = [full_sequences[i] for i in keep_indices]
            teacher_logits_list = [teacher_logits_list[i] for i in keep_indices]
            prompt_lens = [prompt_lens[i] for i in keep_indices]
            prompts = [prompts[i] for i in keep_indices]
            completion_lens = [completion_lens[i] for i in keep_indices]
            print(f"[eval] Remaining: {len(prompts)} prompts after filtering", flush=True)
        else:
            print(f"[eval] All {len(prompts)} prompts have >={MIN_COMPLETION_TOKENS} completion tokens — no filtering needed", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1e: Save eval data for reproducibility
    # ═══════════════════════════════════════════════════════════════════
    try:
        eval_data_path = os.path.join(os.path.dirname(args.output), "eval_data.json")
        eval_data = []
        for i, prompt_text in enumerate(prompts):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens[i]
            continuation_ids = full_seq[0, prompt_len:].tolist()
            continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            eval_data.append({
                "prompt": prompt_text,
                "continuation": continuation_text,
                "prompt_tokens": prompt_len,
                "continuation_tokens": len(continuation_ids),
            })
        with open(eval_data_path, "w") as f:
            json.dump({
                "teacher": args.teacher,
                "max_new_tokens": args.max_new_tokens,
                "block_seed": args.block_seed,
                "n_prompts": len(prompts),
                "data": eval_data,
            }, f, indent=2)
        print(f"[eval] Saved eval data ({len(prompts)} prompts) to {eval_data_path}", flush=True)
    except Exception as e:
        print(f"[eval] Failed to save eval data (non-fatal): {e}", flush=True)

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
        "block_seed": args.block_seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "n_prompts": len(prompts),
        "n_prompts_filtered": n_filtered,
        "min_completion_tokens": MIN_COMPLETION_TOKENS,
        "students": {},
    }
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Live progress
    progress_lock = threading.Lock()
    live_progress = {
        "phase": "scoring", "students": students,
        "students_total": len(students), "prompts_total": len(prompts),
        "completed": [], "current": None,
    }
    def _write_progress():
        """Write current live progress to disk for dashboard consumption."""
        try:
            with progress_lock:
                with open(progress_path, "w") as pf:
                    json.dump(live_progress, pf)
        except Exception:
            pass
    _write_progress()

    # Early stopping state. args.early_stop_min <= 0 disables it outright.
    best_kl_so_far = None
    best_kl_per_prompt_cumulative = None
    MIN_PROMPTS_EARLY_STOP = args.early_stop_min if args.early_stop_min > 0 else len(prompts) + 1
    PER_MODEL_TIMEOUT = 600

    # King stays in VRAM
    king_model = None
    king_name = args.king

    # Prefetch executor
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None

    vram_before_students = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for student_idx, student_name in enumerate(students):
        student_rev = student_revisions.get(student_name, "main")
        # Skip already scored
        if student_name in results["students"]:
            prior = results["students"][student_name]
            kl = prior.get("kl_global_avg")
            print(f"\n[eval] {student_name}: SKIP (already scored, KL={kl})", flush=True)
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
        ensure_disk_space(args.teacher)

        # Prefetch next student
        if student_idx + 1 < len(students):
            next_name = students[student_idx + 1]
            next_rev = student_revisions.get(next_name, "main")
            if next_name not in results["students"] and next_name != king_name:
                prefetch_future = prefetch_executor.submit(prefetch_model, next_name, next_rev)

        # Load student (or reuse king)
        live_progress["phase"] = "loading_student"
        live_progress["current"] = {"student_name": student_name, "student_idx": student_idx, "prompts_done": 0}
        _write_progress()

        is_king = (student_name == king_name)

        # ── HF scoring path ──
        student = None
        load_time = 0.0

        if is_king and king_model is not None:
            student = king_model
            print(f"[eval] King reused from VRAM", flush=True)
        else:
            try:
                t0 = time.time()
                student = load_model(student_name, device, revision=student_rev)
                student.eval()
                load_time = time.time() - t0
                student_vram_gb = (torch.cuda.memory_allocated() - vram_before_students) / 1024**3
                print(f"[eval] Loaded in {load_time:.1f}s, VRAM: {student_vram_gb:.1f}GB, total: {gpu_mem_str()}", flush=True)
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

            # VRAM fraud check
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

            if is_king:
                king_model = student
                print(f"[eval] King loaded — stays in VRAM", flush=True)

        # ── Fine-tunability probe (anti-finetune defense) ──
        # Based on mantaLLM's Discord diagnostic: reject models that can't be
        # continued-pretrained over due to grad-norm explosion or scaled layer norms.
        # King is probed too — pete147 walk-through: king fails → kl=inf →
        # results.py promotes best clean challenger (no new code needed).
        # Skip the probe when king is reused from VRAM (already probed on first load).
        probe_this = student is not None and os.environ.get("FINETUNE_PROBE", "1") != "0"
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            probe_this = False
        if probe_this:
            try:
                _fp_start = time.time()
                probe = finetunability_probe(student, tokenizer, device)
                _fp_dur = time.time() - _fp_start
                mark = "✓" if probe["pass"] else f"✗ DQ: {probe['reason']}"
                print(
                    f"[eval] Finetune probe: loss={probe['loss']:.3f} "
                    f"global_grad={probe['global_grad_norm']:.1f} "
                    f"worst={probe['worst_param_type']}={probe['worst_param_norm']:.1f} "
                    f"norm_w_max={probe['worst_norm_weight']:.2f} "
                    f"({_fp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["finetune_probe"] = {
                    "pass": probe["pass"],
                    "reason": probe.get("reason", ""),
                    "global_grad_norm": probe["global_grad_norm"],
                    "worst_param_type": probe["worst_param_type"],
                    "worst_param_norm": probe["worst_param_norm"],
                    "worst_norm_weight": probe["worst_norm_weight"],
                    "worst_norm_name": probe.get("worst_norm_name", ""),
                    "loss": probe["loss"],
                }
                if not probe["pass"]:
                    results["students"][student_name].update({
                        "status": "anti_finetune",
                        "reason": f"anti_finetune:{probe['reason']}",
                        "kl_global_avg": float("inf"),
                    })
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)
                    live_progress["completed"].append({"student_name": student_name, "status": "anti_finetune"})
                    live_progress["current"] = None
                    _write_progress()
                    if is_king:
                        king_model = None
                    try:
                        del student
                    except Exception:
                        pass
                    free_gpu()
                    clean_model_cache(student_name, args.teacher)
                    continue
            except Exception as e:
                print(f"[eval] Finetune probe error (non-fatal, allowing): {e}", flush=True)

        # ── Chat-collapse probe (CoT-collapse defense) ──
        # Allan's Discord observation + arxiv 2502.07266: off-policy CoT distillation
        # teaches small students to always think but never stop. Validator scores
        # token-level KL on raw web text so collapsed students slip through even
        # as the chat endpoint hangs forever on "hi". Probe rejects models that
        # can't terminate or produce content on trivial chat prompts.
        chat_probe_this = student is not None and os.environ.get("CHAT_COLLAPSE_PROBE", "1") != "0"
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            chat_probe_this = False
        if chat_probe_this:
            try:
                _cp_start = time.time()
                cprobe = chat_response_probe(student, tokenizer, device)
                _cp_dur = time.time() - _cp_start
                mark = "✓" if cprobe["pass"] else f"✗ DQ: {cprobe['reason']}"
                print(
                    f"[eval] Chat probe: term={cprobe['prompts_terminated']}/{cprobe['prompts_tested']} "
                    f"nonempty={cprobe['prompts_non_empty']}/{cprobe['prompts_tested']} "
                    f"mean_gen={cprobe['mean_gen_tokens']:.0f} "
                    f"think_frac={cprobe['mean_reasoning_fraction']:.2f} "
                    f"({_cp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["chat_probe"] = {
                    "pass": cprobe["pass"],
                    "reason": cprobe.get("reason", ""),
                    "prompts_tested": cprobe["prompts_tested"],
                    "prompts_terminated": cprobe["prompts_terminated"],
                    "prompts_non_empty": cprobe["prompts_non_empty"],
                    "mean_gen_tokens": cprobe["mean_gen_tokens"],
                    "mean_content_chars": cprobe["mean_content_chars"],
                    "mean_reasoning_fraction": cprobe["mean_reasoning_fraction"],
                    "samples": cprobe.get("samples", []),
                }
                if not cprobe["pass"]:
                    results["students"][student_name].update({
                        "status": "chat_collapse",
                        "reason": f"chat_collapse:{cprobe['reason']}",
                        "kl_global_avg": float("inf"),
                    })
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)
                    live_progress["completed"].append({"student_name": student_name, "status": "chat_collapse"})
                    live_progress["current"] = None
                    _write_progress()
                    if is_king:
                        king_model = None
                    try:
                        del student
                    except Exception:
                        pass
                    free_gpu()
                    clean_model_cache(student_name, args.teacher)
                    continue
            except Exception as e:
                print(f"[eval] Chat probe error (non-fatal, allowing): {e}", flush=True)

        # Thinking-collapse probe DISABLED as of 2026-04-19. See reports/
        # 2026-04-19-think-probe-disabled.md — the teacher-anchored Wilson
        # variant of the probe (commit 8eec9a2) DQ'd every student including
        # the teacher itself and the reigning king for three consecutive
        # rounds (blocks 7999728 / 8000338 / 8001xxx). Miners confirmed
        # the breakage empirically; the probe's termination threshold is
        # fundamentally miscalibrated for Qwen3.5 reasoning models that
        # legitimately emit long chain-of-thought before EOS.
        #
        # The entire probe pipeline (generation, Wilson bounds, per-student
        # storage) is retained but gated behind an opt-in env var so we can
        # re-enable it offline once it is properly calibrated against a
        # real teacher baseline. Default is OFF: the probe does NOT run,
        # does NOT consume GPU time, and CANNOT DQ a model.
        think_probe_this = (
            student is not None
            and os.environ.get("THINK_COLLAPSE_PROBE", "0") == "1"
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            think_probe_this = False
        if think_probe_this:
            try:
                _tp_start = time.time()
                tprobe = thinking_collapse_probe(
                    student, tokenizer, device,
                    teacher_samples=globals().get("_TEACHER_PROBE_SAMPLES"),
                )
                _tp_dur = time.time() - _tp_start
                mark = "✓" if tprobe["pass"] else f"✗ DQ: {tprobe['reason']}"
                sb = tprobe.get("self_bleu_across_prompts", 0.0)
                print(
                    f"[eval] Think probe: term={tprobe['prompts_terminated']}/{tprobe['prompts_tested']} "
                    f"degen={tprobe['prompts_degenerate']}/{tprobe['prompts_tested']} "
                    f"mean_gen={tprobe['mean_gen_tokens']:.0f} "
                    f"self_bleu={sb:.2f} "
                    f"({_tp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["think_probe"] = {
                    "pass": tprobe["pass"],
                    "reason": tprobe.get("reason", ""),
                    "prompts_tested": tprobe["prompts_tested"],
                    "prompts_terminated": tprobe["prompts_terminated"],
                    "prompts_degenerate": tprobe["prompts_degenerate"],
                    "mean_gen_tokens": tprobe["mean_gen_tokens"],
                    "self_bleu_across_prompts": sb,
                    "teacher_self_bleu": tprobe.get("teacher_self_bleu", 0.0),
                    "wilson": tprobe.get("wilson"),
                    "samples": tprobe.get("samples", []),
                }
                if not tprobe["pass"]:
                    # Telemetry-only: record the failure on the student's
                    # result entry but DO NOT DQ. See disable rationale in
                    # the opt-in env var gate above.
                    results["students"][student_name]["think_probe"]["would_have_dq"] = True
                    print(
                        f"[eval] Think probe FAILED (telemetry-only, not DQ'ing): "
                        f"{tprobe['reason']}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[eval] Think probe error (non-fatal, allowing): {e}", flush=True)

        # ── Capability probe (SHADOW axis, not gating) ─────────────────
        # Verifiable-rewards micro-battery. Scored against teacher answers
        # on the same prompts. Purpose: expose a capability axis that KL
        # memorization cannot win, feeding the composite score preview.
        cap_probe_this = (
            student is not None
            and os.environ.get("CAPABILITY_PROBE", "1") != "0"
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            cap_probe_this = False
        if cap_probe_this:
            try:
                _cp_start = time.time()
                cap = capability_probe(student, tokenizer, device)
                _cp_dur = time.time() - _cp_start
                refs = globals().get("_TEACHER_CAPABILITY_REFS") or {}
                teacher_ans = refs.get("answers") or []
                teacher_correct = 0
                paired = 0
                for item_idx, item in enumerate(CAPABILITY_PROBE_PROMPTS):
                    if item_idx < len(teacher_ans):
                        paired += 1
                        if _capability_score_one(teacher_ans[item_idx], item):
                            teacher_correct += 1
                teacher_frac = teacher_correct / paired if paired else None
                teach_str = f" teacher={teacher_frac*100:.0f}%" if teacher_frac is not None else ""
                print(
                    f"[eval] Capability probe: "
                    f"{cap['correct']}/{cap['n']} ({cap['pass_frac']*100:.0f}%)"
                    f"{teach_str} ({_cp_dur:.1f}s)",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["capability"] = {
                    "n": cap["n"], "correct": cap["correct"],
                    "pass_frac": round(cap["pass_frac"], 3),
                    "teacher_pass_frac": round(teacher_frac, 3) if teacher_frac is not None else None,
                    "items": cap.get("items", []),
                }
            except Exception as e:
                print(f"[eval] Capability probe error (non-fatal): {e}", flush=True)

        # ── Activation fingerprint (for functional copy detection) ──
        if student is not None:
            try:
                fp_start = time.time()
                fp = compute_activation_fingerprint(student, device)
                fp_time = time.time() - fp_start
                if fp:
                    results["students"].setdefault(student_name, {})["activation_fingerprint"] = fp
                    print(f"[eval] Fingerprint computed in {fp_time:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] Fingerprint failed: {e}", flush=True)

        # ── Score: per-prompt with early stopping ──
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
                    tl_entry = teacher_logits_list[i]
                    is_sparse = _is_sparse_logits(tl_entry)

                    # Student forward pass
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]

                    topk_shadow = {}

                    if is_sparse:
                        # ── Sparse teacher logits path (top-k from vLLM or HF) ──
                        t_indices = tl_entry["indices"]  # [1, seq_len, k]
                        t_values = tl_entry["values"]    # [1, seq_len, k]
                        min_len = min(cont_s.shape[1], t_indices.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_indices, t_values
                        else:
                            s_cont_slice = cont_s[:, :min_len, :]
                            # Detect if values are logprobs (from vLLM) or raw logits (from HF top-k)
                            # vLLM logprobs are negative and typically < 0; raw logits can be positive
                            # Heuristic: if max value > 0, they're logits; if all <= 0, they're logprobs
                            max_val = t_values[:, :min_len, :].max().item()
                            are_logprobs = (max_val <= 0.0)
                            kl_per_pos = compute_kl_from_sparse(
                                t_indices[:, :min_len, :], t_values[:, :min_len, :],
                                s_cont_slice, values_are_logprobs=are_logprobs
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()
                            del t_indices, t_values, s_cont_slice, kl_per_pos
                    else:
                        # ── Dense teacher logits path (legacy / full-vocab) ──
                        tl = tl_entry.to(device).float()
                        t_log_p = F.log_softmax(tl, dim=-1)
                        t_p = t_log_p.exp()
                        del tl

                        min_len = min(cont_s.shape[1], t_log_p.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_log_p, t_p
                        else:
                            t_lp_slice = t_log_p[:, :min_len, :]
                            t_p_slice = t_p[:, :min_len, :]
                            s_cont_slice = cont_s[:, :min_len, :]
                            kl_per_pos = compute_kl_from_precomputed(
                                t_lp_slice, t_p_slice, s_cont_slice
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()

                            # Shadow top-k KL metrics (logged but not used for scoring)
                            try:
                                s_log_p_full = F.log_softmax(s_cont_slice.float(), dim=-1)
                                for k_val in (100, 1000):
                                    _, topk_idx = t_p_slice.topk(k_val, dim=-1)
                                    t_topk = t_lp_slice.gather(-1, topk_idx)
                                    s_topk = s_log_p_full.gather(-1, topk_idx)
                                    t_topk_norm = F.log_softmax(t_topk, dim=-1)
                                    s_topk_norm = F.log_softmax(s_topk, dim=-1)
                                    topk_kl = F.kl_div(s_topk_norm, t_topk_norm,
                                                       log_target=True, reduction="none").sum(dim=-1).squeeze(0)
                                    topk_shadow[f"kl_top{k_val}"] = round(topk_kl.mean().item(), 6)
                                    del topk_idx, t_topk, s_topk, t_topk_norm, s_topk_norm, topk_kl
                                del s_log_p_full
                            except Exception:
                                pass

                            del kl_per_pos, t_lp_slice, t_p_slice, s_cont_slice
                            del t_log_p, t_p

                    del s_logits, cont_s
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
                        "student_name": student_name, "student_idx": student_idx,
                        "prompts_done": i + 1, "prompts_total": len(prompts),
                        "kl_running_mean": round(running_mean, 6),
                        "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    }
                    _write_progress()

                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{len(prompts)}] KL={kl_mean:.6f} (avg: {running_mean:.6f})", flush=True)

                except RuntimeError as e:
                    scoring_error = str(e)
                    if "out of memory" in str(e).lower():
                        print(f"  [prompt {i}] OOM", flush=True)
                    else:
                        print(f"  [prompt {i}] RuntimeError: {e}", flush=True)
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
            preserved = {
                k: v for k, v in results["students"].get(student_name, {}).items()
                if k in ("think_probe", "capability", "activation_fingerprint")
            }
            results["students"][student_name] = {
                "status": "scoring_error", "error": scoring_error[:500],
                "kl_global_avg": None, **preserved}
        elif kl_per_prompt:
            kl_avg = sum(d["mean"] for d in kl_per_prompt) / len(kl_per_prompt)
            n_scored = len(kl_per_prompt)
            status = "early_stopped" if early_stopped else ("partial" if scoring_error else "scored")
            print(f"  → KL={kl_avg:.6f} ({n_scored}/{len(prompts)} prompts, {status})", flush=True)

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
            if topk_aggs:
                student_result["shadow_topk"] = topk_aggs
                print(f"  → Shadow top-k: {topk_aggs}", flush=True)

            # Length penalty axis (shadow). Computes student-mean-gen vs
            # teacher-mean-gen on the think probe prompts. Ratio > 1 means
            # student rambles; 1.0 means matched; ~0 means student hard-stops.
            think_prev = results["students"].get(student_name, {}).get("think_probe") or {}
            stud_mean_gen = float(think_prev.get("mean_gen_tokens") or 0.0)
            teacher_think_refs = globals().get("_TEACHER_PROBE_SAMPLES") or []
            teach_mean_gen = 0.0
            if teacher_think_refs and tokenizer is not None:
                try:
                    lens = []
                    for txt in teacher_think_refs:
                        if not txt:
                            continue
                        lens.append(len(tokenizer(txt, return_tensors="pt",
                                                 truncation=False).input_ids[0]))
                    if lens:
                        teach_mean_gen = sum(lens) / len(lens)
                except Exception:
                    pass
            if stud_mean_gen > 0 and teach_mean_gen > 0:
                ratio = stud_mean_gen / teach_mean_gen
                length_penalty = min(1.0, LENGTH_PENALTY_RATIO / max(ratio, 1e-6))
                student_result["length_axis"] = {
                    "student_mean_gen": round(stud_mean_gen, 1),
                    "teacher_mean_gen": round(teach_mean_gen, 1),
                    "ratio": round(ratio, 3),
                    "penalty": round(length_penalty, 3),
                }
                print(f"  → Length axis: student={stud_mean_gen:.0f} "
                      f"teacher={teach_mean_gen:.0f} ratio={ratio:.2f} "
                      f"penalty={length_penalty:.2f}", flush=True)

            # ── On-policy RKL rollouts (Phase A) ─────────────────────────
            # Sample rollouts from the student's own policy while it is
            # still on-GPU. We save student top-K logprobs alongside each
            # token so the later teacher pass (Phase B, runs once for all
            # students after this loop) is a single forward per rollout.
            if (
                ON_POLICY_RKL_ENABLED
                and student is not None
                and student_result.get("status") in ("scored", "early_stopped", "partial")
            ):
                try:
                    _op_start = time.time()
                    rollouts = on_policy_rollouts(student, tokenizer, device)
                    _op_dur = time.time() - _op_start
                    if rollouts:
                        # Stash raw rollouts on a module global; only the
                        # aggregates get persisted to JSON. The rollouts
                        # themselves live for the duration of this eval
                        # and are consumed in Phase B.
                        _store = globals().setdefault("_ON_POLICY_ROLLOUTS", {})
                        _store[student_name] = rollouts
                        gen_avg = sum(r["gen_len"] for r in rollouts) / len(rollouts)
                        student_result["on_policy_rollouts_meta"] = {
                            "n_rollouts": len(rollouts),
                            "mean_gen_len": round(gen_avg, 1),
                            "collected_at": round(_op_dur, 1),
                        }
                        print(
                            f"  → On-policy rollouts: {len(rollouts)} "
                            f"(avg_gen={gen_avg:.0f}, {_op_dur:.1f}s)",
                            flush=True,
                        )
                except Exception as e:
                    print(f"  → On-policy rollouts skipped: {str(e)[:140]}", flush=True)

            # Preserve probes and fingerprint already written into this
            # student's dict — overwriting it blanks them.
            preserved = {
                k: v for k, v in results["students"].get(student_name, {}).items()
                if k in ("think_probe", "capability", "activation_fingerprint", "on_policy_rkl")
            }
            results["students"][student_name] = {**student_result, **preserved}

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
            torch.cuda.empty_cache()

        # Wait for prefetch
        if prefetch_future:
            try:
                prefetch_future.result(timeout=1)
            except Exception:
                pass
            prefetch_future = None

    # ── On-policy RKL scoring (Phase B) ─────────────────────────────
    # After the student loop, load the teacher once and score all
    # collected rollouts. This amortizes the teacher load across
    # students and keeps total wall time manageable (~30s load + ~0.5s
    # per rollout). Results get merged back into each student's dict.
    _store = globals().get("_ON_POLICY_ROLLOUTS") or {}
    if ON_POLICY_RKL_ENABLED and _store:
        try:
            print(f"\n[eval] On-policy RKL Phase B: scoring {len(_store)} "
                  f"students' rollouts against teacher", flush=True)
            # Free the king if it's still resident — teacher forward pass
            # wants all the VRAM it can get.
            try:
                if king_model is not None:
                    del king_model
                    king_model = None
            except Exception:
                pass
            free_gpu()
            _rkl_t0 = time.time()
            teacher_b = load_model(args.teacher, device)
            teacher_b.eval()
            print(f"[eval] Teacher reloaded for RKL ({time.time() - _rkl_t0:.0f}s), "
                  f"VRAM: {gpu_mem_str()}", flush=True)
            n_scored = 0
            for sn, rolls in _store.items():
                try:
                    _rkl_s_t0 = time.time()
                    rkl = on_policy_rkl_score(teacher_b, rolls, device=device)
                    dur = time.time() - _rkl_s_t0
                    slim = {
                        "n_rollouts": rkl["n_rollouts"],
                        "tokens": rkl["tokens"],
                        "mean_rkl": round(rkl["mean_rkl"], 6) if rkl["mean_rkl"] == rkl["mean_rkl"] else None,
                        "mean_fkl": round(rkl["mean_fkl"], 6) if rkl["mean_fkl"] == rkl["mean_fkl"] else None,
                        "mean_skl": round(rkl["mean_skl"], 6) if rkl["mean_skl"] == rkl["mean_skl"] else None,
                        "mean_sampled_gap": round(rkl["mean_sampled_gap"], 6) if rkl["mean_sampled_gap"] == rkl["mean_sampled_gap"] else None,
                        "mean_gen_len": round(rkl["mean_gen_len"], 1),
                        "skew_alpha": rkl.get("skew_alpha"),
                        "top_k": rkl.get("top_k"),
                        "per_rollout": rkl.get("per_rollout", []),
                        "scoring_time": round(dur, 1),
                    }
                    if sn in results["students"]:
                        results["students"][sn]["on_policy_rkl"] = slim
                    n_scored += 1
                    print(
                        f"  [{sn}] rkl={slim['mean_rkl']} fkl={slim['mean_fkl']} "
                        f"skl={slim['mean_skl']} gap={slim['mean_sampled_gap']} "
                        f"({dur:.1f}s, {rkl['n_rollouts']} rollouts)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [{sn}] RKL scoring error: {str(e)[:160]}", flush=True)
            try:
                del teacher_b
            except Exception:
                pass
            free_gpu()
            timings["on_policy_rkl"] = time.time() - _rkl_t0
            print(f"[eval] On-policy RKL: scored {n_scored}/{len(_store)} students "
                  f"in {timings['on_policy_rkl']:.1f}s", flush=True)
        except Exception as e:
            print(f"[eval] On-policy RKL Phase B failed (non-fatal): {e}", flush=True)
        finally:
            globals()["_ON_POLICY_ROLLOUTS"] = {}

    # Final save
    results["timings"] = {k: round(v, 1) for k, v in timings.items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] DONE — {len(results['students'])} students", flush=True)
    for k, v in sorted(timings.items()):
        print(f"  {k}: {v:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    Path(args.output).with_name("eval_done.marker").write_text(
        f"done {len(results['students'])} students\n")

    prefetch_executor.shutdown(wait=False)


def _write_abort_marker(out_path, reason):
    try:
        Path(out_path).with_name("eval_done.marker").write_text(f"aborted {reason}\n")
    except Exception:
        pass


if __name__ == "__main__":
    _out_guess = None
    for i, a in enumerate(sys.argv):
        if a == "--output" and i + 1 < len(sys.argv):
            _out_guess = sys.argv[i + 1]
            break

    def _sig(signum, _frame):
        reason = f"signal_{signum}"
        if _out_guess:
            _write_abort_marker(_out_guess, reason)
        stop_vllm_server()
        sys.exit(128 + (signum or 0))

    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _sig)
        except Exception:
            pass

    try:
        main()
    except SystemExit:
        raise
    except BaseException as _exc:
        if _out_guess:
            _write_abort_marker(_out_guess, f"exception:{type(_exc).__name__}")
        raise
