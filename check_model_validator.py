#!/usr/bin/env python3
"""
Distil SN97 — Pre-Submission Model Checker (validator-aligned eval)

Run this BEFORE committing your model to avoid wasting registration fees.
Performs ALL the same checks the validator runs, including anti-cheat detection.

GPU eval (--eval) by default uses the **live validator** prompt bundle and king:
  - Prompts + max_new_tokens: https://distil.arbos.life/api/eval-data
  - Current king model id:      https://distil.arbos.life/api/h2h-latest

Use --offline-eval to use the local climbmix sampler (legacy) instead.

Requirements:
    pip install click huggingface_hub transformers safetensors

Private models: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), or pass --hf-token, or use
``huggingface-cli login`` so the hub can resolve credentials (token=True).

For --eval mode (optional):
    pip install torch datasets  # + CUDA GPU

Usage:
    # Basic pre-submission check (no GPU needed):
    python check_model_validator.py --model-repo user/my-distilled-model

    # Full eval aligned with on-chain validator (requires GPU + network):
    python check_model_validator.py --model-repo user/my-distilled-model --eval

    # First N prompts only (faster smoke test):
    python check_model_validator.py --model-repo user/my-distilled-model --eval --prompts 50

    # Legacy local prompts + optional explicit king:
    python check_model_validator.py --model-repo user/my-distilled-model --eval --offline-eval --king-repo org/king
"""
import os
import sys
import json
import time
import math
import logging
import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Union

import click


def _resolve_hf_hub_token(explicit: Optional[str]) -> Union[bool, str]:
    """Token for huggingface_hub / transformers: env, --hf-token, else True (CLI cache)."""
    if explicit:
        return explicit.strip()
    env = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if env:
        return env
    return True

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
FINGERPRINT_COSINE_THRESHOLD = 0.99999  # functional copy detection (bumped 2026-04-19, see commit history)

# Live validator APIs (same sources as on-chain evaluation)
DEFAULT_VALIDATOR_EVAL_DATA_URL = "https://distil.arbos.life/api/eval-data"
DEFAULT_VALIDATOR_H2H_LATEST_URL = "https://distil.arbos.life/api/h2h-latest"


def _fetch_json_url(url: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """GET JSON from URL (stdlib only)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "distil-check-model-validator/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(out).__name__}")
    return out


def _paired_t_stats_one_sided(deltas: list[float]) -> tuple[float, float]:
    """
    One-sided paired t-test p-value for mean(delta) > 0 (student better when
    delta = king_kl - student_kl per prompt). Mirrors distil_kl_train_prebuilt._paired_t_stats.
    """
    n = len(deltas)
    if n < 2:
        return 0.0, 1.0
    mean_delta = sum(deltas) / n
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    if se <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    t_stat = mean_delta / se
    cdf = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    p_one_sided = max(0.0, min(1.0, 1.0 - cdf))
    return float(t_stat), float(p_one_sided)


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
@click.option("--prompts", type=int, default=0,
              help="Max prompts for --eval (0 = use full validator bundle when not --offline-eval)")
@click.option("--teacher-cache", default=None, type=click.Path(),
              help="Path to teacher_cache.pt (skips teacher inference if provided)")
@click.option("--dataset", default="karpathy/climbmix-400b-shuffle",
              help="Dataset for eval prompts (--offline-eval only)")
@click.option("--offline-eval", is_flag=True, default=False,
              help="Use local climbmix prompt sampling instead of live /api/eval-data")
@click.option("--validator-eval-data-url", default=DEFAULT_VALIDATOR_EVAL_DATA_URL,
              help="URL for validator prompt bundle JSON")
@click.option("--validator-h2h-url", default=DEFAULT_VALIDATOR_H2H_LATEST_URL,
              help="URL for validator h2h-latest JSON (king model id)")
@click.option("--king-repo", default=None,
              help="King model repo (default: king_model from --validator-h2h-url)")
@click.option("--king-revision", default=None,
              help="King model revision")
@click.option(
    "--hf-token",
    default=None,
    help="Hugging Face access token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN env)",
)
def main(
    model_repo,
    revision,
    run_eval,
    prompts,
    teacher_cache,
    dataset,
    offline_eval,
    validator_eval_data_url,
    validator_h2h_url,
    king_repo,
    king_revision,
    hf_token,
):
    """
    Comprehensive pre-submission checker for Distil SN97.

    Runs every check the validator performs so you know BEFORE committing
    whether your model will be accepted or rejected.
    """
    from huggingface_hub import model_info as hf_model_info, hf_hub_download, repo_info

    hub_token = _resolve_hf_hub_token(hf_token)
    if isinstance(hub_token, str) and hub_token:
        os.environ["HF_TOKEN"] = hub_token

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
            info = repo_info(model_repo, repo_type="model", token=hub_token)
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
        info = hf_model_info(
            model_repo, revision=revision, files_metadata=True, token=hub_token
        )
        if info.private:
            check_warn(
                "Private repository",
                "Authenticated access OK for this pre-check; on-chain submission "
                "usually requires the model repo to be public.",
            )
            warnings.append(
                ("accessibility", "Private repo — make public before submitting if required"),
            )
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
            repo_id=model_repo,
            filename="config.json",
            revision=revision,
            token=hub_token,
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

        # RULE: Must use Qwen3_5ForConditionalGeneration (vLLM-native architecture)
        archs = config.get("architectures", [])
        model_type = config.get("model_type", "")
        has_preproc = any(
            getattr(s, "rfilename", "") == "preprocessor_config.json"
            for s in (info.siblings or [])
        ) if info else False
        if model_type == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in archs and has_preproc:
            check_pass("Architecture", f"Qwen3_5ForConditionalGeneration (vLLM-native)")
        elif model_type == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in archs:
            check_warn("Architecture",
                       f"Qwen3_5ForConditionalGeneration found but missing preprocessor_config.json. "
                       f"Copy it from Qwen/Qwen3.5-4B.")
        else:
            check_fail("Architecture",
                       f"Must use Qwen3_5ForConditionalGeneration (model_type=qwen3_5). "
                       f"Found: {','.join(archs)} (model_type={model_type}). "
                       f"See Discord announcement for conversion instructions.")
            failures.append(("architecture", f"{','.join(archs)} / {model_type}"))

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

        teacher_tok = AutoTokenizer.from_pretrained(
            TEACHER_MODEL, trust_remote_code=True, token=hub_token
        )
        try:
            student_tok = AutoTokenizer.from_pretrained(
                model_repo,
                revision=revision,
                trust_remote_code=False,
                token=hub_token,
            )
        except Exception:
            # Some tokenizers need trust_remote_code or have custom backends
            # The validator also allows this with a warning, so we do the same
            student_tok = AutoTokenizer.from_pretrained(
                model_repo,
                revision=revision,
                trust_remote_code=True,
                token=hub_token,
            )

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

        integrity = verify_model_integrity(
            model_repo, revision, allow_private=True,
        )
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
        print(f"   python check_model_validator.py --model-repo {model_repo} --eval")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════
    # OPTIONAL: GPU-based evaluation
    # ══════════════════════════════════════════════════════════════════════
    # Aligns with live validator: prompts from /api/eval-data, king from /api/h2h-latest.
    # Scoring: continuation-only KL(teacher || student), fp32 log_softmax, F.kl_div(log_target=True).
    banner("GPU EVALUATION", char="█")
    if offline_eval:
        print("  Mode: offline (climbmix prompts, max_new_tokens=512)")
    else:
        print(
            f"  Mode: validator-aligned (eval-data + h2h-latest; "
            f"max_new_tokens from API)"
        )
    print()

    try:
        import torch
        import torch.nn.functional as F
        if not torch.cuda.is_available():
            check_fail("GPU check", "No CUDA GPU available. --eval requires a GPU.")
            sys.exit(1)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check_info("GPU", f"{gpu_name} ({gpu_mem:.0f}GB)")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── Live validator: king model + eval metadata ─────────────────
        h2h_meta: dict[str, Any] = {}
        try:
            h2h_meta = _fetch_json_url(validator_h2h_url)
        except Exception as e:
            check_warn("h2h-latest fetch", f"{e} — set --king-repo explicitly if needed")
            warnings.append(("h2h_fetch", str(e)))

        if not king_repo:
            king_repo = h2h_meta.get("king_model")
            if king_repo:
                check_info("King (from h2h-latest)", f"{king_repo}")
            else:
                check_warn(
                    "King repo",
                    "Could not resolve king_model from API; pass --king-repo",
                )

        # ── Load tokenizer (teacher) ─────────────────────────────────
        banner("Loading Teacher Model")
        teacher_tok = AutoTokenizer.from_pretrained(
            TEACHER_MODEL, trust_remote_code=True, token=hub_token
        )

        # ── Prompts: live eval-data OR offline climbmix ──────────────
        banner("Sampling Eval Prompts")
        from eval.dataset import sample_prompts_from_dataset, format_prompt

        block_seed_for_gen = 12345
        eval_bundle_meta: dict[str, Any] = {}
        eval_items: list[dict[str, Any]] = []

        if offline_eval:
            n_local = prompts if prompts > 0 else 20
            raw_prompts = sample_prompts_from_dataset(
                n=n_local,
                block_number=12345,
                block_hash=None,
                dataset_name=dataset,
            )
            eval_prompts = []
            for text in raw_prompts:
                formatted = format_prompt(text)
                if formatted:
                    eval_prompts.append(formatted)
                    eval_items.append({
                        "prompt": formatted,
                        "row_index": len(eval_items),
                        "continuation": None,
                    })
                if len(eval_prompts) >= n_local:
                    break
            print(
                f"  Offline mode: {len(eval_prompts)} prompts from dataset "
                f"(format_prompt filtered from {len(raw_prompts)} raw)"
            )
            MAX_NEW_TOKENS = 512
        else:
            try:
                eval_bundle = _fetch_json_url(validator_eval_data_url)
            except Exception as e:
                check_fail("eval-data fetch", str(e))
                print(f"  Hint: use --offline-eval for local climbmix prompts, or fix network/DNS.")
                sys.exit(1)

            eval_bundle_meta = {
                "teacher": eval_bundle.get("teacher"),
                "max_new_tokens": eval_bundle.get("max_new_tokens"),
                "block_seed": eval_bundle.get("block_seed"),
                "n_prompts": eval_bundle.get("n_prompts"),
                "private_redacted": eval_bundle.get("private_redacted"),
            }
            rows = eval_bundle.get("data") or []
            if not rows:
                check_fail("eval-data", "Empty data[] in eval-data response")
                sys.exit(1)

            skipped_private = 0
            skipped_invalid = 0
            for row_idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    skipped_invalid += 1
                    continue
                if row.get("is_private"):
                    skipped_private += 1
                    continue
                raw = row.get("prompt", "")
                continuation = row.get("continuation")
                if raw == "[PRIVATE]" or continuation == "[PRIVATE]":
                    skipped_private += 1
                    continue
                formatted = format_prompt(raw) if raw else ""
                if formatted:
                    eval_items.append({
                        "prompt": formatted,
                        "row_index": row_idx,
                        "continuation": continuation if isinstance(continuation, str) and continuation else None,
                        "continuation_tokens": row.get("continuation_tokens"),
                    })
                else:
                    skipped_invalid += 1

            if prompts and prompts > 0:
                eval_items = eval_items[: prompts]

            eval_prompts = [item["prompt"] for item in eval_items]

            block_seed_for_gen = int(eval_bundle.get("block_seed") or 12345)
            MAX_NEW_TOKENS = int(eval_bundle.get("max_new_tokens") or 8192)

            print(
                f"  Validator eval-data: {len(eval_prompts)} public prompts, "
                f"max_new_tokens={MAX_NEW_TOKENS}, block_seed={block_seed_for_gen}"
            )
            if eval_bundle_meta.get("private_redacted") or skipped_private:
                print(
                    f"  Skipped {skipped_private} private/redacted rows "
                    f"(private_redacted={eval_bundle_meta.get('private_redacted')})"
                )
            if skipped_invalid:
                print(f"  Skipped {skipped_invalid} invalid/non-dict rows")
            if eval_bundle_meta.get("teacher") and eval_bundle_meta["teacher"] != TEACHER_MODEL:
                check_warn(
                    "Teacher id",
                    f"eval-data teacher={eval_bundle_meta['teacher']} (local checker uses {TEACHER_MODEL})",
                )

        if len(eval_prompts) == 0:
            check_fail("Prompt sampling", "No valid prompts after filtering")
            sys.exit(1)

        eval_prompt_indices = [int(item.get("row_index", i)) for i, item in enumerate(eval_items)]
        use_published_continuations = (
            not offline_eval
            and len(eval_items) == len(eval_prompts)
            and all(isinstance(item.get("continuation"), str) and item.get("continuation") for item in eval_items)
        )
        if use_published_continuations:
            check_info("Online continuations", "Using public continuations from eval-data")

        print(f"  King comparison target: {king_repo or '(none)'}")
        if h2h_meta.get("king_kl") is not None:
            if use_published_continuations:
                check_info(
                    "h2h-latest reference king_kl",
                    f"{h2h_meta.get('king_kl')} (full private+public set; not used for comparison)",
                )
            else:
                check_info("h2h-latest reference king_kl", f"{h2h_meta.get('king_kl')}")
        if h2h_meta.get("paired_test_alpha") is not None:
            check_info("h2h-latest paired_test_alpha", f"{h2h_meta.get('paired_test_alpha')}")

        teacher_loaded = False
        teacher_logits_list = []  # continuation-only logits per prompt
        full_sequences = []       # full token sequences (prompt + continuation)
        prompt_lens_list = []     # prompt token length per prompt

        if teacher_cache and use_published_continuations:
            print("  Ignoring teacher cache for online eval-data continuations")
        elif teacher_cache and Path(teacher_cache).exists():
            print(f"  Loading cached teacher data from {teacher_cache}...")
            try:
                cache_data = torch.load(teacher_cache, map_location="cpu", weights_only=False)
                if (len(cache_data.get("full_sequences", [])) >= len(eval_prompts)
                    and cache_data.get("teacher_logits")
                    and cache_data.get("prompt_lens")):
                    full_sequences = [s.to("cuda") for s in cache_data["full_sequences"][:len(eval_prompts)]]
                    teacher_logits_list = cache_data["teacher_logits"][:len(eval_prompts)]
                    prompt_lens_list = cache_data["prompt_lens"][:len(eval_prompts)]
                    teacher_loaded = True
                    print(f"  Loaded {len(full_sequences)} cached prompt sequences")
                else:
                    print(f"  Cache incompatible, will regenerate")
            except Exception as e:
                print(f"  Cache load failed: {e}, will regenerate")

        if not teacher_loaded:
            print(f"  Loading {TEACHER_MODEL}...")
            t0 = time.time()
            teacher = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hub_token,
            )
            teacher.eval()
            print(f"  Teacher loaded in {time.time() - t0:.1f}s")
            teacher_vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  Teacher VRAM: {teacher_vram:.1f}GB")

            # ── Generate teacher continuations & extract logits ────────
            # Matches production: generate continuation, then forward pass
            # to get logits, extract continuation-only positions.
            if use_published_continuations:
                banner("Preparing Published Continuations + Teacher Logits")
            else:
                banner("Generating Teacher Continuations + Logits")
            with torch.no_grad():
                for i, prompt_text in enumerate(eval_prompts):
                    prompt_ids = teacher_tok(prompt_text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
                    prompt_len = prompt_ids.shape[1]
                    if use_published_continuations:
                        continuation_text = eval_items[i]["continuation"]
                        continuation_ids = teacher_tok(
                            continuation_text,
                            return_tensors="pt",
                            truncation=False,
                            add_special_tokens=False,
                        ).input_ids.to(teacher.device)
                        output_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
                        gen_len = continuation_ids.shape[1]
                        expected_gen_len = eval_items[i].get("continuation_tokens")
                        if expected_gen_len is not None and int(expected_gen_len) != gen_len:
                            check_warn(
                                "Continuation token count",
                                f"row {eval_prompt_indices[i]} API={expected_gen_len}, tokenizer={gen_len}",
                            )
                    else:
                        prompt_seed = int(block_seed_for_gen) + eval_prompt_indices[i]
                        teacher_device = teacher.device

                        # Generate continuation (validator: seeded sampling per original prompt row)
                        try:
                            gen = torch.Generator(device=teacher_device)
                            gen.manual_seed(prompt_seed)
                            output_ids = teacher.generate(
                                prompt_ids,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                use_cache=True,
                                generator=gen,
                            )
                        except ValueError as e:
                            if "not used by the model: ['generator']" not in str(e):
                                raise
                            device_idx = teacher_device.index if teacher_device.type == "cuda" else None
                            devices = [device_idx] if device_idx is not None else []
                            with torch.random.fork_rng(devices=devices):
                                torch.manual_seed(prompt_seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(prompt_seed)
                                output_ids = teacher.generate(
                                    prompt_ids,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    use_cache=True,
                                )
                        gen_len = output_ids.shape[1] - prompt_len

                    # Forward pass to get logits for the full sequence
                    logits = teacher(output_ids).logits.float()
                    # Extract continuation-only logits: positions prompt_len-1 to -1
                    # (shifted by 1 because logits[t] predicts token[t+1])
                    cont_logits = logits[:, prompt_len - 1:-1, :]

                    full_sequences.append(output_ids.cpu())
                    teacher_logits_list.append(cont_logits.cpu())
                    prompt_lens_list.append(prompt_len)

                    del logits, cont_logits
                    if (i + 1) % 5 == 0 or i == len(eval_prompts) - 1:
                        print(f"  Teacher: {i + 1}/{len(eval_prompts)} prompts "
                              f"({prompt_len}+{gen_len} tokens)", flush=True)

            print(f"  Teacher logits generated for {len(eval_prompts)} prompts")

            # Move full_sequences to cuda
            full_sequences = [s.to("cuda") for s in full_sequences]

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
            token=hub_token,
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

        # ── Score KL divergence (continuation-only, matches production) ──
        # This matches the production eval pipeline in pod_eval_vllm.py:
        # - Forward pass on full sequence (prompt + teacher continuation)
        # - Extract student logits at continuation positions only
        # - Cast to fp32 before log_softmax
        # - Use F.kl_div with log_target=True
        banner("CHECK 10: KL Divergence Scoring")

        kl_scores = []
        for i in range(len(eval_prompts)):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens_list[i]

            # Teacher: compute log_softmax on-the-fly in fp32 (matches production)
            t_logits = teacher_logits_list[i].to(student.device).float()
            t_log_p = F.log_softmax(t_logits, dim=-1)

            with torch.no_grad():
                # Student forward pass on full sequence
                s_logits = student(full_seq).logits.float()
                # Extract continuation-only positions (same slice as production)
                cont_s = s_logits[:, prompt_len - 1:-1, :]

            # Align lengths (in case of minor mismatch)
            min_len = min(cont_s.shape[1], t_log_p.shape[1])
            t_lp_slice = t_log_p[:, :min_len, :]
            s_lp_slice = F.log_softmax(cont_s[:, :min_len, :], dim=-1)

            # KL(teacher || student) using log_target=True (matches production)
            kl_per_pos = F.kl_div(
                s_lp_slice, t_lp_slice, log_target=True, reduction='none'
            ).sum(dim=-1)
            kl_mean = kl_per_pos.mean().item()
            kl_scores.append(kl_mean)

            del s_logits, cont_s, t_logits, t_log_p, t_lp_slice, s_lp_slice, kl_per_pos

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
            if use_published_continuations:
                print(
                    f"  Evaluating king on the same {len(eval_prompts)} public "
                    "eval-data prompt/continuation pairs."
                )
                print("  h2h-latest king_kl is not used because it includes private prompts.")
            print(f"  Loading king: {king_repo}...")
            del student
            torch.cuda.empty_cache()

            king = AutoModelForCausalLM.from_pretrained(
                king_repo,
                revision=king_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
                token=hub_token,
            )
            king.eval()

            king_kl_scores = []
            with torch.no_grad():
                for i in range(len(eval_prompts)):
                    full_seq = full_sequences[i]
                    prompt_len = prompt_lens_list[i]
                    t_logits = teacher_logits_list[i].to(king.device).float()
                    t_log_p = F.log_softmax(t_logits, dim=-1)
                    k_logits = king(full_seq).logits.float()
                    cont_k = k_logits[:, prompt_len - 1:-1, :]
                    min_len = min(cont_k.shape[1], t_log_p.shape[1])
                    t_lp_slice = t_log_p[:, :min_len, :]
                    k_lp_slice = F.log_softmax(cont_k[:, :min_len, :], dim=-1)
                    kl_per_pos = F.kl_div(
                        k_lp_slice, t_lp_slice, log_target=True, reduction='none'
                    ).sum(dim=-1)
                    king_kl_scores.append(kl_per_pos.mean().item())
                    del t_logits, t_log_p, k_logits, cont_k, t_lp_slice, k_lp_slice, kl_per_pos

            king_kl = sum(king_kl_scores) / len(king_kl_scores)

            del king
            torch.cuda.empty_cache()

            # Paired comparison (validator-style): delta_i = king_kl_i - student_kl_i (positive => you win)
            paired_deltas = [k - s for k, s in zip(king_kl_scores, kl_scores)]
            mean_delta = sum(paired_deltas) / len(paired_deltas)
            t_stat, p_paired = _paired_t_stats_one_sided(paired_deltas)
            alpha = h2h_meta.get("paired_test_alpha")
            eps_thr = h2h_meta.get("epsilon_threshold")

            print(f"\n  Your model:  mean KL = {kl_global:.6f}")
            print(f"  King model:  mean KL = {king_kl:.6f}  ({king_repo})")
            print(f"  Paired mean_delta (king - student): {mean_delta:.6f}")
            print(f"  Paired t-test (one-sided H1: delta>0): t={t_stat:.4f}, p={p_paired:.6g}, n={len(paired_deltas)}")
            if alpha is not None:
                print(f"  Validator paired_test_alpha (reference): {alpha}")
            if eps_thr is not None:
                print(f"  Validator epsilon_threshold (reference): {eps_thr}")

            diff_pct = (kl_global - king_kl) / king_kl * 100 if king_kl else 0.0
            if kl_global < king_kl:
                print(f"  🏆 Mean KL: you beat king by {abs(diff_pct):.2f}% (lower is better).")
            else:
                print(f"  👑 Mean KL: king ahead by {abs(diff_pct):.2f}% (lower is better).")
                print(f"     Target: mean KL < {king_kl:.6f} on this prompt set.")

            if mean_delta > 0 and alpha is not None and p_paired <= float(alpha):
                print("  ✅ Paired test: significant improvement vs king (p <= ref alpha).")
            elif mean_delta > 0:
                print(
                    "  ⚠️  Paired mean favors you, but p may not meet validator alpha — "
                    "use full prompt set or reduce variance."
                )
        elif not king_repo and use_published_continuations:
            check_warn(
                "King comparison",
                "No king model resolved; pass --king-repo to evaluate a fair public-only comparison",
            )
            warnings.append(("king_comparison", "No king model resolved for public-only comparison"))
        elif not king_repo:
            try:
                h2h_file = Path("state/h2h_latest.json")
                if h2h_file.exists():
                    h2h = json.loads(h2h_file.read_text())
                    king_uid = h2h.get("king_uid")
                    for r in h2h.get("results", []):
                        if r.get("uid") == king_uid:
                            king_kl_est = r.get("kl", 0)
                            king_model = r.get("model", "?")
                            banner("KING COMPARISON (offline fallback)")
                            print(f"  Current king: UID {king_uid} ({king_model})")
                            print(f"  King KL (cached h2h): {king_kl_est:.6f}")
                            print(f"  Your model KL:      {kl_global:.6f}")
                            diff_pct = (kl_global - king_kl_est) / king_kl_est * 100 if king_kl_est else 0.0
                            if kl_global < king_kl_est:
                                print(f"  🏆 Your model appears to BEAT the king by {abs(diff_pct):.2f}% (mean KL).")
                            else:
                                print(f"  👑 King is still better by {abs(diff_pct):.2f}% (mean KL).")
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
