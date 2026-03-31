#!/usr/bin/env python3
"""
Distil SN97 — Pre-Submission Model Checker

Run this BEFORE committing your model to avoid wasting registration fees.
Performs ALL the same checks the validator runs, including anti-cheat detection.

Usage:
    # Basic pre-submission check (all validator checks):
    python check_model.py --model-repo user/my-distilled-model

    # With specific revision:
    python check_model.py --model-repo user/my-distilled-model --revision abc123

    # Full eval against current king (optional, requires GPU):
    python check_model.py --model-repo user/my-distilled-model --eval

    # Eval with custom prompt count:
    python check_model.py --model-repo user/my-distilled-model --eval --prompts 20
"""
import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("check_model")

# ── Constants (must match validator) ────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15  # ~5.25B max
BASELINE_VOCAB_SIZE = 248320
MIN_MODEL_BYTES = 500_000_000     # 500MB minimum
MAX_STUDENT_VRAM_GB = 20.0        # Real 4B ≈ 8-10GB
MIN_TOKENS_PER_SEC = 50           # Real 4B on B200 does 100+ tok/s
KL_FRAUD_THRESHOLD = 1e-6         # KL ≤ this = identical to teacher = fraud
FINGERPRINT_COSINE_THRESHOLD = 0.9999  # functional copy detection


def banner(text: str, char: str = "═", width: int = 60):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def check_pass(name: str, detail: str = ""):
    print(f"  ✅ {name}{f' — {detail}' if detail else ''}")


def check_fail(name: str, detail: str = ""):
    print(f"  ❌ {name}{f' — {detail}' if detail else ''}")


def check_warn(name: str, detail: str = ""):
    print(f"  ⚠️  {name}{f' — {detail}' if detail else ''}")


def check_info(name: str, detail: str = ""):
    print(f"  ℹ️  {name}{f' — {detail}' if detail else ''}")


@click.command()
@click.option("--model-repo", required=True, help="HuggingFace repo (e.g. 'user/my-model')")
@click.option("--revision", default=None, help="Specific HF revision/commit SHA")
@click.option("--eval", "run_eval", is_flag=True, default=False,
              help="Run a realistic eval against the current king (requires GPU)")
@click.option("--prompts", type=int, default=20,
              help="Number of prompts for --eval mode (default: 20)")
@click.option("--teacher-cache", default=None, type=click.Path(),
              help="Path to teacher_cache.pt (skips teacher inference if provided)")
@click.option("--dataset", default="karpathy/climbmix-400b-shuffle",
              help="Dataset for eval prompts")
@click.option("--king-repo", default=None,
              help="King model repo for eval comparison (auto-detected if omitted)")
@click.option("--king-revision", default=None,
              help="King model revision")
def main(model_repo, revision, run_eval, prompts, teacher_cache, dataset, king_repo, king_revision):
    """
    Comprehensive pre-submission checker for Distil SN97.

    Runs every check the validator performs so you know BEFORE committing
    whether your model will be accepted or rejected.
    """
    from huggingface_hub import model_info as hf_model_info, hf_hub_download, repo_info

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO
    max_model_bytes = max_params_b * 2.2e9

    failures = []
    warnings = []

    banner("DISTIL SN97 — PRE-SUBMISSION MODEL CHECKER")
    print(f"  Model: {model_repo}")
    print(f"  Revision: {revision or '(latest)'}")
    print(f"  Max params: {max_params_b:.2f}B")

    # ── Resolve revision ────────────────────────────────────────────────
    if not revision:
        try:
            info = repo_info(model_repo, repo_type="model")
            revision = info.sha
            print(f"  Pinned revision: {revision[:12]}...")
        except Exception as e:
            check_fail("Resolve revision", str(e))
            failures.append(("revision", str(e)))
            _print_summary(failures, warnings)
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Repository accessibility
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 1: Repository Accessibility")
    try:
        info = hf_model_info(model_repo, revision=revision, files_metadata=True)
        if info.private:
            check_fail("Public access", "Model is PRIVATE — must be public")
            failures.append(("accessibility", "Model is private"))
        elif info.disabled:
            check_fail("Public access", "Model is DISABLED on HuggingFace")
            failures.append(("accessibility", "Model is disabled"))
        else:
            check_pass("Public access", "Model is publicly accessible")
    except Exception as e:
        err = str(e)
        if "404" in err:
            check_fail("Public access", "Model not found (404)")
            failures.append(("accessibility", "Model not found"))
        elif "403" in err:
            check_fail("Public access", "Model is restricted/gated (403)")
            failures.append(("accessibility", "Model is restricted"))
        else:
            check_fail("Public access", f"Error: {err}")
            failures.append(("accessibility", err))
        _print_summary(failures, warnings)
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: No custom code (security)
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 2: Security — No Custom Code")
    dangerous_files = []
    all_files = []
    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        all_files.append(fname)
        if fname.endswith('.py') and fname != '__init__.py':
            dangerous_files.append(fname)

    if dangerous_files:
        check_fail("No custom code",
                    f"Found Python files: {', '.join(dangerous_files)}. "
                    f"Custom code is NOT allowed — students must use standard architectures only.")
        failures.append(("custom_code", f"Files: {', '.join(dangerous_files)}"))
    else:
        check_pass("No custom code", "No .py files found in repo")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Weight file format (safetensors required)
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 3: Weight File Format & Sizes")
    total_st_bytes = 0
    total_pt_bytes = 0
    st_files = []
    pt_files = []

    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        fsize = 0
        if hasattr(sibling, 'size') and sibling.size is not None:
            fsize = sibling.size
        elif hasattr(sibling, 'lfs') and sibling.lfs:
            fsize = sibling.lfs.get('size', 0)

        if fname.endswith('.safetensors'):
            total_st_bytes += fsize
            st_files.append((fname, fsize))
        elif fname.endswith('.bin') and 'pytorch_model' in fname:
            total_pt_bytes += fsize
            pt_files.append((fname, fsize))

    check_info("Safetensors files", f"{len(st_files)} files, {total_st_bytes / 1e9:.2f} GB")
    if pt_files:
        check_info("PyTorch .bin files", f"{len(pt_files)} files, {total_pt_bytes / 1e9:.2f} GB")

    # RULE: pytorch_model.bin only → rejected
    if pt_files and not st_files:
        check_fail("Safetensors required",
                    f"Only pytorch_model.bin found ({len(pt_files)} files, {total_pt_bytes / 1e9:.1f}GB). "
                    f"Convert with: model.save_pretrained('output', safe_serialization=True)")
        failures.append(("format", "Safetensors required, only .bin found"))
    elif st_files:
        check_pass("Safetensors present")

    # RULE: Tiny safetensors + large .bin = fraud attempt
    if st_files and pt_files:
        if total_st_bytes < MIN_MODEL_BYTES and total_pt_bytes > MIN_MODEL_BYTES:
            check_fail("Weight file integrity",
                       f"FRAUD PATTERN: Tiny safetensors ({total_st_bytes:,}B) alongside large "
                       f"pytorch_model.bin ({total_pt_bytes:,}B). Real model hidden in .bin files.")
            failures.append(("fraud_hidden_weights", "Tiny ST + large .bin"))

    # RULE: Minimum file size
    total_weight_bytes = max(total_st_bytes, total_pt_bytes)
    if 0 < total_weight_bytes < MIN_MODEL_BYTES:
        check_fail("Minimum model size",
                    f"Weight files total {total_weight_bytes:,} bytes — too small for a real model "
                    f"(minimum: {MIN_MODEL_BYTES:,} bytes)")
        failures.append(("min_size", f"Only {total_weight_bytes:,} bytes"))
    elif total_weight_bytes >= MIN_MODEL_BYTES:
        check_pass("Minimum model size", f"{total_weight_bytes / 1e9:.2f} GB")

    # RULE: Maximum file size
    if total_weight_bytes > max_model_bytes:
        check_fail("Maximum model size",
                    f"Weight files total {total_weight_bytes / 1e9:.1f}GB — too large for "
                    f"{max_params_b:.1f}B params (max ~{max_model_bytes / 1e9:.1f}GB in bf16)")
        failures.append(("max_size", f"{total_weight_bytes / 1e9:.1f}GB exceeds limit"))
    elif total_weight_bytes > 0:
        check_pass("Maximum model size", f"Under {max_model_bytes / 1e9:.1f}GB limit")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4: Config analysis (param count, MoE, vocab size)
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 4: Model Configuration")
    try:
        config_path = hf_hub_download(
            repo_id=model_repo, filename="config.json", revision=revision
        )
        with open(config_path) as f:
            config = json.load(f)

        # Import MoE param counter
        sys.path.insert(0, str(Path(__file__).parent))
        from eval.model_checker import compute_moe_params, get_safetensors_param_count

        moe_info = compute_moe_params(config)
        config_total_b = moe_info["total_params"] / 1e9
        config_active_b = moe_info["active_params"] / 1e9

        safetensors_params_b = get_safetensors_param_count(model_repo, revision)
        total_params_b = safetensors_params_b if safetensors_params_b > 0 else config_total_b

        check_info("Config total params", f"{config_total_b:.2f}B (from config)")
        if safetensors_params_b > 0:
            check_info("Safetensors params", f"{safetensors_params_b:.2f}B (verified)")
        check_info("Active params", f"{config_active_b:.2f}B")

        if moe_info["is_moe"]:
            check_info("MoE detected",
                       f"{moe_info['num_experts']} experts, "
                       f"{moe_info['num_active_experts']} active/token")

        # RULE: Total params ≤ max
        if total_params_b > max_params_b:
            check_fail("Parameter count",
                       f"{total_params_b:.2f}B > {max_params_b:.1f}B max (total params, not active)")
            failures.append(("params", f"{total_params_b:.2f}B > {max_params_b:.1f}B"))
        elif total_params_b > 0:
            check_pass("Parameter count", f"{total_params_b:.2f}B ≤ {max_params_b:.1f}B")

        # RULE: Cross-validate config vs file size
        if total_weight_bytes > 0 and total_params_b > 0:
            estimated_params_from_size = total_weight_bytes / 2e9  # bf16 estimate
            if estimated_params_from_size > total_params_b * 2.5:
                check_fail("Config vs file size",
                           f"Config claims {total_params_b:.2f}B but files suggest "
                           f"~{estimated_params_from_size:.1f}B (bf16). Possible teacher in disguise.")
                failures.append(("cross_validate", "Config/file size mismatch"))
            else:
                check_pass("Config vs file size", "Consistent")

        # RULE: No quantization
        quant_config = config.get("quantization_config", {})
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown")
            check_fail("No quantization",
                       f"Quantized model detected ({quant_method}). "
                       f"Subnet requires bf16/fp16 architecture distillation.")
            failures.append(("quantized", quant_method))
        else:
            check_pass("No quantization")

        # RULE: Vocab size matches teacher
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)

        if vocab_size != BASELINE_VOCAB_SIZE:
            check_fail("Vocab size",
                       f"{vocab_size} ≠ {BASELINE_VOCAB_SIZE} (teacher). "
                       f"Must use same tokenizer as Qwen3.5-35B-A3B.")
            failures.append(("vocab_size", f"{vocab_size} ≠ {BASELINE_VOCAB_SIZE}"))
        else:
            check_pass("Vocab size", f"{vocab_size} matches teacher")

        # RULE: Nested MoE detection (text_config with hidden experts)
        text_cfg = config.get("text_config", {})
        nested_experts = text_cfg.get("num_local_experts", 0) or text_cfg.get("num_experts", 0)
        top_experts = config.get("num_local_experts", 0) or config.get("num_experts", 0)
        if nested_experts > 1 and not top_experts:
            check_warn("Nested MoE config",
                       f"text_config has {nested_experts} experts but top-level config doesn't. "
                       f"This pattern is flagged as suspicious.")
            warnings.append(("nested_moe", f"text_config.num_experts={nested_experts}"))
        else:
            check_pass("No nested MoE config")

    except Exception as e:
        check_fail("Config analysis", str(e))
        failures.append(("config", str(e)))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Tokenizer compatibility
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 5: Tokenizer Compatibility")
    try:
        from transformers import AutoTokenizer

        teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
        student_tok = AutoTokenizer.from_pretrained(model_repo, revision=revision, trust_remote_code=False)

        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "日本語のテスト文字列です。Unicode handling matters.",
            "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) for all x in vocabulary",
        ]

        mismatch = False
        for s in test_strings:
            t_ids = teacher_tok.encode(s)
            s_ids = student_tok.encode(s)
            if t_ids != s_ids:
                check_fail("Tokenizer encoding",
                           f"Mismatch on: '{s[:40]}...' "
                           f"(teacher: {len(t_ids)} tokens, student: {len(s_ids)} tokens)")
                failures.append(("tokenizer", f"Encoding mismatch"))
                mismatch = True
                break

        if not mismatch:
            check_pass("Tokenizer encoding", "All test strings match teacher")

    except Exception as e:
        check_warn("Tokenizer check", f"Could not verify: {e}")
        warnings.append(("tokenizer", str(e)))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Duplicate hash detection
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 6: Model Identity (Duplicate Detection)")
    try:
        from eval.model_checker import compute_model_hash

        model_hash = compute_model_hash(model_repo, revision)
        if model_hash:
            check_info("Model hash", f"{model_hash[:16]}...")
            # Check against known hashes if state dir exists
            hash_file = Path("state/model_hashes.json")
            if hash_file.exists():
                known = json.loads(hash_file.read_text())
                for uid_str, known_hash in known.items():
                    if known_hash == model_hash:
                        check_warn("Duplicate check",
                                   f"Same hash as UID {uid_str} already on-chain. "
                                   f"Submitting a copy will be auto-rejected (earlier commit wins).")
                        warnings.append(("duplicate", f"Matches UID {uid_str}"))
                        break
                else:
                    check_pass("Duplicate check", "No known duplicates")
            else:
                check_info("Duplicate check",
                           "Cannot check (no state/model_hashes.json). "
                           "Validator will check on submission.")
        else:
            check_warn("Model hash", "Could not compute hash — no safetensors found?")
            warnings.append(("hash", "Could not compute"))

    except Exception as e:
        check_warn("Duplicate check", f"Error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7: Model integrity (weights unchanged)
    # ══════════════════════════════════════════════════════════════════════
    banner("CHECK 7: Model Integrity")
    try:
        from eval.model_checker import verify_model_integrity

        integrity = verify_model_integrity(model_repo, revision)
        if integrity["pass"]:
            check_pass("Integrity", "Model accessible and weights verifiable")
        else:
            check_fail("Integrity", integrity["reason"])
            failures.append(("integrity", integrity["reason"]))
    except Exception as e:
        check_warn("Integrity", f"Error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY (pre-GPU checks)
    # ══════════════════════════════════════════════════════════════════════
    _print_summary(failures, warnings)

    if failures:
        print("\n⛔ Your model will be REJECTED by the validator.")
        print("   Fix the issues above before committing to avoid wasting registration fees.")
        sys.exit(1)

    if not run_eval:
        print("\n✅ All pre-submission checks passed!")
        print("   Your model should be accepted by the validator.")
        print()
        print("   TIP: Run with --eval to test against the current king on GPU:")
        print(f"   python check_model.py --model-repo {model_repo} --eval")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════
    # OPTIONAL: GPU-based evaluation
    # ══════════════════════════════════════════════════════════════════════
    banner("GPU EVALUATION", char="█")
    print(f"  Running {prompts}-prompt eval against teacher")
    if king_repo:
        print(f"  King comparison: {king_repo}")
    print()

    try:
        import torch
        if not torch.cuda.is_available():
            check_fail("GPU check", "No CUDA GPU available. --eval requires a GPU.")
            sys.exit(1)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        check_info("GPU", f"{gpu_name} ({gpu_mem:.0f}GB)")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        # ── Load teacher ──────────────────────────────────────────────
        banner("Loading Teacher Model")
        teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

        teacher_logits = None
        if teacher_cache and Path(teacher_cache).exists():
            print(f"  Loading cached teacher logits from {teacher_cache}...")
            cache_data = torch.load(teacher_cache, map_location="cpu", weights_only=True)
            teacher_logits = cache_data.get("logits")
            cached_prompts = cache_data.get("prompts", [])
            print(f"  Loaded {len(cached_prompts)} cached prompts")
        
        if teacher_logits is None:
            print(f"  Loading {TEACHER_MODEL}...")
            t0 = time.time()
            teacher = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            teacher.eval()
            print(f"  Teacher loaded in {time.time() - t0:.1f}s")
            teacher_vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  Teacher VRAM: {teacher_vram:.1f}GB")

        # ── Load dataset & sample prompts ─────────────────────────────
        banner("Sampling Eval Prompts")
        ds = load_dataset(dataset, split="train", streaming=True)
        eval_prompts = []
        for i, row in enumerate(ds):
            text = row.get("text", "")
            if len(text) > 100:
                eval_prompts.append(text[:2048])
            if len(eval_prompts) >= prompts:
                break
        print(f"  Sampled {len(eval_prompts)} prompts from {dataset}")

        # ── Generate teacher logits if not cached ─────────────────────
        if teacher_logits is None:
            banner("Generating Teacher Logits")
            teacher_logits = []
            for i, prompt_text in enumerate(eval_prompts):
                ids = teacher_tok(prompt_text, return_tensors="pt", truncation=True,
                                  max_length=512).input_ids.to(teacher.device)
                with torch.no_grad():
                    out = teacher(ids)
                teacher_logits.append(out.logits.cpu())
                if (i + 1) % 5 == 0:
                    print(f"  Teacher: {i + 1}/{len(eval_prompts)} prompts", flush=True)
            print(f"  Teacher logits generated for {len(eval_prompts)} prompts")

            # Unload teacher to free VRAM for student
            del teacher
            torch.cuda.empty_cache()

        # ── Load student ──────────────────────────────────────────────
        banner("Loading Student Model")
        t0 = time.time()
        student = AutoModelForCausalLM.from_pretrained(
            model_repo,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        student.eval()
        load_time = time.time() - t0
        student_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  Student loaded in {load_time:.1f}s, VRAM: {student_vram:.1f}GB")

        # ANTI-CHEAT: VRAM check
        banner("CHECK 8: Runtime Anti-Cheat (VRAM)")
        if student_vram > MAX_STUDENT_VRAM_GB:
            check_fail("VRAM usage",
                       f"Student uses {student_vram:.1f}GB (max {MAX_STUDENT_VRAM_GB}GB). "
                       f"A real ≤5B model uses ~8-10GB. Likely a larger model in disguise.")
            failures.append(("vram_fraud", f"{student_vram:.1f}GB"))
        else:
            check_pass("VRAM usage", f"{student_vram:.1f}GB (max {MAX_STUDENT_VRAM_GB}GB)")

        # ANTI-CHEAT: Generation speed
        banner("CHECK 9: Runtime Anti-Cheat (Speed)")
        try:
            bench_text = "The quick brown fox"
            bench_ids = teacher_tok(bench_text, return_tensors="pt").input_ids.to(student.device)
            with torch.no_grad():
                t0 = time.time()
                out = student.generate(bench_ids, max_new_tokens=128, do_sample=False)
                gen_time = time.time() - t0
            actual_new = out.shape[1] - bench_ids.shape[1]
            tokens_per_sec = round(actual_new / gen_time, 1)
            print(f"  Generation speed: {tokens_per_sec} tok/s ({actual_new} tokens in {gen_time:.2f}s)")

            if tokens_per_sec < MIN_TOKENS_PER_SEC:
                check_warn("Generation speed",
                           f"{tokens_per_sec} tok/s < {MIN_TOKENS_PER_SEC} minimum. "
                           f"Validator will FLAG this as suspicious.")
                warnings.append(("speed", f"{tokens_per_sec} tok/s"))
            else:
                check_pass("Generation speed", f"{tokens_per_sec} tok/s")
        except Exception as e:
            check_warn("Generation speed", f"Benchmark failed: {e}")

        # ── Score KL divergence ───────────────────────────────────────
        banner("CHECK 10: KL Divergence Scoring")
        import torch.nn.functional as F

        kl_scores = []
        for i, prompt_text in enumerate(eval_prompts):
            ids = teacher_tok(prompt_text, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(student.device)
            with torch.no_grad():
                student_out = student(ids)

            t_logits = teacher_logits[i].to(student_out.logits.device)

            # Align lengths
            min_len = min(t_logits.shape[1], student_out.logits.shape[1])
            t_log_probs = F.log_softmax(t_logits[:, :min_len, :], dim=-1)
            s_log_probs = F.log_softmax(student_out.logits[:, :min_len, :], dim=-1)

            # KL(teacher || student) per position, then mean
            kl_per_pos = F.kl_div(s_log_probs, t_log_probs.exp(), reduction='none').sum(dim=-1)
            kl_mean = kl_per_pos.mean().item()
            kl_scores.append(kl_mean)

            if (i + 1) % 5 == 0:
                running_avg = sum(kl_scores) / len(kl_scores)
                print(f"  Prompt {i + 1}/{len(eval_prompts)}: "
                      f"KL={kl_mean:.6f} (running avg: {running_avg:.6f})", flush=True)

        kl_global = sum(kl_scores) / len(kl_scores)
        import statistics
        kl_std = statistics.stdev(kl_scores) if len(kl_scores) > 1 else 0
        kl_ci_low = kl_global - 1.96 * kl_std / (len(kl_scores) ** 0.5)
        kl_ci_high = kl_global + 1.96 * kl_std / (len(kl_scores) ** 0.5)

        print(f"\n  KL Divergence: {kl_global:.6f}")
        print(f"  95% CI: [{kl_ci_low:.6f}, {kl_ci_high:.6f}]")
        print(f"  Std dev: {kl_std:.6f} over {len(kl_scores)} prompts")

        # ANTI-CHEAT: KL too low = teacher copy
        banner("CHECK 11: KL Fraud Detection")
        if kl_global <= KL_FRAUD_THRESHOLD:
            check_fail("KL fraud check",
                       f"KL={kl_global:.10f} ≤ {KL_FRAUD_THRESHOLD}. "
                       f"Model is identical to teacher — automatic DQ.")
            failures.append(("kl_fraud", f"KL={kl_global}"))
        elif kl_global < 0.001:
            check_warn("KL suspiciously low",
                       f"KL={kl_global:.6f} is extremely low. "
                       f"Validator may flag for manual review.")
            warnings.append(("kl_low", f"KL={kl_global:.6f}"))
        else:
            check_pass("KL fraud check", f"KL={kl_global:.6f} (legitimate)")

        # ── Compare against king ──────────────────────────────────────
        if king_repo:
            banner("KING COMPARISON")
            print(f"  Loading king: {king_repo}...")
            # Unload student
            del student
            torch.cuda.empty_cache()

            king = AutoModelForCausalLM.from_pretrained(
                king_repo,
                revision=king_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
            )
            king.eval()

            king_kl_scores = []
            for i, prompt_text in enumerate(eval_prompts):
                ids = teacher_tok(prompt_text, return_tensors="pt", truncation=True,
                                  max_length=512).input_ids.to(king.device)
                with torch.no_grad():
                    king_out = king(ids)

                t_logits = teacher_logits[i].to(king_out.logits.device)
                min_len = min(t_logits.shape[1], king_out.logits.shape[1])
                t_log_probs = F.log_softmax(t_logits[:, :min_len, :], dim=-1)
                k_log_probs = F.log_softmax(king_out.logits[:, :min_len, :], dim=-1)
                kl_per_pos = F.kl_div(k_log_probs, t_log_probs.exp(), reduction='none').sum(dim=-1)
                king_kl_scores.append(kl_per_pos.mean().item())

            king_kl = sum(king_kl_scores) / len(king_kl_scores)

            del king
            torch.cuda.empty_cache()

            print(f"\n  Your model:  KL = {kl_global:.6f}")
            print(f"  Current king: KL = {king_kl:.6f}")
            diff_pct = (kl_global - king_kl) / king_kl * 100
            if kl_global < king_kl:
                print(f"  🏆 Your model BEATS the king by {abs(diff_pct):.2f}%!")
            else:
                print(f"  👑 King is still better by {abs(diff_pct):.2f}%")
                print(f"     You need KL < {king_kl:.6f} to dethrone.")
        elif not king_repo:
            # Auto-detect king from state
            try:
                h2h_file = Path("state/h2h_latest.json")
                if h2h_file.exists():
                    h2h = json.loads(h2h_file.read_text())
                    king_uid = h2h.get("king_uid")
                    # Find king model in results
                    for r in h2h.get("results", []):
                        if r.get("uid") == king_uid:
                            king_kl_est = r.get("kl", 0)
                            king_model = r.get("model", "?")
                            banner("KING COMPARISON (estimated)")
                            print(f"  Current king: UID {king_uid} ({king_model})")
                            print(f"  King KL (last eval): {king_kl_est:.6f}")
                            print(f"  Your model KL:       {kl_global:.6f}")
                            diff_pct = (kl_global - king_kl_est) / king_kl_est * 100
                            if kl_global < king_kl_est:
                                print(f"  🏆 Your model appears to BEAT the king by {abs(diff_pct):.2f}%!")
                                print(f"     (Note: final eval uses {60} prompts with different sampling)")
                            else:
                                print(f"  👑 King is still better by {abs(diff_pct):.2f}%")
                            break
            except Exception:
                pass

        _print_summary(failures, warnings, kl=kl_global)

    except Exception as e:
        import traceback
        traceback.print_exc()
        check_fail("GPU evaluation", str(e))
        failures.append(("eval", str(e)))
        _print_summary(failures, warnings)
        sys.exit(1)

    sys.exit(1 if failures else 0)


def _print_summary(failures, warnings, kl=None):
    banner("SUMMARY")
    if failures:
        print(f"  ❌ {len(failures)} FAILURE(S) — model will be REJECTED:")
        for name, detail in failures:
            print(f"     • {name}: {detail}")
    if warnings:
        print(f"  ⚠️  {len(warnings)} WARNING(S) — may cause issues:")
        for name, detail in warnings:
            print(f"     • {name}: {detail}")
    if not failures and not warnings:
        print(f"  ✅ All checks passed!")
    if kl is not None:
        print(f"\n  📊 KL Divergence: {kl:.6f}")
    print()


if __name__ == "__main__":
    main()
