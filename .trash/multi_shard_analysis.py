#!/usr/bin/env python3
"""
Multi-Shard Analysis Tool — Paired Statistical Comparison of Models

Evaluates multiple models across the same N shards from karpathy/climbmix-400b-shuffle,
computes per-shard KL divergence, and runs paired statistical tests.

Requested by BC on Discord for benchmarking all past/current kings.

Usage:
    python scripts/multi_shard_analysis.py \
        --models "user/model-a,user/model-b,user/model-c" \
        --num-shards 50 \
        --prompts-per-shard 40 \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --output results/

Can be run locally with a GPU or on a Lium pod via SSH.
"""
import os
import sys
import json
import time
import argparse
import hashlib
import random
import math
import logging
from pathlib import Path
from itertools import combinations
from datetime import datetime

import numpy as np

logger = logging.getLogger("multi_shard_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add project root to path so we can import eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset constants (same as eval/dataset.py)
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
CLIMBMIX_TEXT_FIELD = "text"

DEFAULT_TEACHER = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_NUM_SHARDS = 50
DEFAULT_PROMPTS_PER_SHARD = 40
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_PROMPT_TOKENS = 1024


def parse_args():
    p = argparse.ArgumentParser(description="Multi-shard paired KL analysis across models")
    p.add_argument("--models", required=True, help="Comma-separated HuggingFace model repos")
    p.add_argument("--teacher", default=DEFAULT_TEACHER, help=f"Teacher model (default: {DEFAULT_TEACHER})")
    p.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="Number of shards to evaluate")
    p.add_argument("--prompts-per-shard", type=int, default=DEFAULT_PROMPTS_PER_SHARD, help="Prompts per shard")
    p.add_argument("--output", default="results", help="Output directory for results")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max continuation tokens")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shard selection")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap resamples for CI")
    p.add_argument("--resume", action="store_true", help="Resume from partial results in output dir")
    return p.parse_args()


def select_shards(num_shards: int, seed: int) -> list[int]:
    """Select num_shards shard indices deterministically from seed."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(CLIMBMIX_NUM_SHARDS), num_shards))


def load_shard_prompts(shard_idx: int, n_prompts: int, seed: int,
                       min_chars: int = 200, max_chars: int = 4000) -> list[str]:
    """Load and sample prompts from a single climbmix shard."""
    from datasets import load_dataset
    from eval.dataset import format_prompt

    shard_file = f"shard_{shard_idx:05d}.parquet"
    logger.info(f"Loading shard {shard_idx} ({shard_file})...")

    ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")

    # Deterministic shuffle per shard+seed
    rng = random.Random(f"{seed}_{shard_idx}")
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    for idx in indices:
        text = ds[idx].get(CLIMBMIX_TEXT_FIELD, "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        formatted = format_prompt(text)
        if formatted:
            prompts.append(formatted)
        if len(prompts) >= n_prompts:
            break

    logger.info(f"  Shard {shard_idx}: got {len(prompts)} prompts")
    return prompts


def evaluate_model_on_prompts(
    student_model,
    teacher_cache: list[dict],
    device: str = "cuda",
) -> list[float]:
    """Evaluate a student model on pre-cached teacher continuations, return per-prompt KL."""
    from eval.kl_divergence import evaluate_student_kl

    kl_values = []
    for entry in teacher_cache:
        if entry["teacher_logits"] is None or entry["gen_len"] == 0:
            continue
        result = evaluate_student_kl(student_model, entry, device=device)
        kl_values.append(result["kl_mean"])

    return kl_values


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test: H0 is mean(a) == mean(b). Returns t-statistic and p-value."""
    from scipy import stats
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    if std_diff == 0:
        return {"t_stat": float("inf") if mean_diff != 0 else 0.0, "p_value": 0.0 if mean_diff != 0 else 1.0, "n": n}
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n": n, "mean_diff": float(mean_diff)}


def generate_markdown_table(results: dict) -> str:
    """Generate a human-readable markdown report from results."""
    lines = [
        "# Multi-Shard KL Analysis Results",
        "",
        f"**Date:** {results['metadata']['timestamp']}",
        f"**Teacher:** `{results['metadata']['teacher']}`",
        f"**Shards:** {results['metadata']['num_shards']}",
        f"**Prompts/shard:** {results['metadata']['prompts_per_shard']}",
        f"**Total prompts per model:** {results['metadata']['num_shards'] * results['metadata']['prompts_per_shard']}",
        "",
        "## Model Summary",
        "",
        "| Model | Mean KL | Std Dev | 95% CI Lower | 95% CI Upper | Shards |",
        "|-------|---------|---------|--------------|--------------|--------|",
    ]

    # Sort by mean KL (best first)
    summaries = sorted(results["model_summaries"], key=lambda x: x["mean_kl"])
    for s in summaries:
        lines.append(
            f"| `{s['model']}` | {s['mean_kl']:.6f} | {s['std_kl']:.6f} | "
            f"{s['ci_lower']:.6f} | {s['ci_upper']:.6f} | {s['n_shards']} |"
        )

    lines.extend(["", "## Paired Comparisons (t-test)", ""])

    if results.get("paired_tests"):
        lines.extend([
            "| Model A | Model B | Mean Diff (A-B) | t-stat | p-value | Significant? |",
            "|---------|---------|-----------------|--------|---------|-------------|",
        ])
        for test in results["paired_tests"]:
            sig = "✅ Yes" if test["p_value"] < 0.05 else "❌ No"
            lines.append(
                f"| `{test['model_a']}` | `{test['model_b']}` | "
                f"{test['mean_diff']:+.6f} | {test['t_stat']:.3f} | {test['p_value']:.4f} | {sig} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Lower KL = better** (closer to teacher distribution)",
        "- **Paired t-test** compares models on the *same* shards (fair comparison)",
        "- **Significant?** indicates p < 0.05 (95% confidence the models differ)",
        "- **Mean Diff < 0** means Model A is better than Model B",
        "",
        "---",
        f"*Generated by multi_shard_analysis.py*",
    ])

    return "\n".join(lines)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 1:
        print("ERROR: Need at least 1 model", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for partial results to resume
    partial_file = output_dir / "partial_results.json"
    partial_results = {}
    if args.resume and partial_file.exists():
        try:
            partial_results = json.loads(partial_file.read_text())
            logger.info(f"Resuming from {len(partial_results)} cached shard results")
        except Exception:
            pass

    logger.info(f"Models: {models}")
    logger.info(f"Teacher: {args.teacher}")
    logger.info(f"Shards: {args.num_shards}, Prompts/shard: {args.prompts_per_shard}")

    # Select shards deterministically
    shards = select_shards(args.num_shards, args.seed)
    logger.info(f"Selected shards: {shards[:10]}... (showing first 10)")

    # Import heavy deps only after arg parsing
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval.kl_divergence import generate_teacher_continuations

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load teacher model
    logger.info(f"Loading teacher: {args.teacher}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    teacher_model.eval()

    # Per-model, per-shard KL scores: {model: {shard_idx: mean_kl}}
    all_results: dict[str, dict[int, float]] = {m: {} for m in models}

    # Load partial results
    for model_name, shard_data in partial_results.items():
        if model_name in all_results:
            all_results[model_name] = {int(k): v for k, v in shard_data.items()}

    for shard_i, shard_idx in enumerate(shards):
        logger.info(f"\n{'='*60}")
        logger.info(f"SHARD {shard_i+1}/{args.num_shards} (index={shard_idx})")
        logger.info(f"{'='*60}")

        # Skip if all models already have this shard
        if all(shard_idx in all_results[m] for m in models):
            logger.info(f"  All models already evaluated on shard {shard_idx}, skipping")
            continue

        # Load prompts for this shard
        prompts = load_shard_prompts(shard_idx, args.prompts_per_shard, args.seed)
        if len(prompts) < args.prompts_per_shard // 2:
            logger.warning(f"  Only got {len(prompts)} prompts from shard {shard_idx}, skipping")
            continue

        # Tokenize prompts
        input_ids_list = []
        for prompt_text in prompts:
            tokens = teacher_tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                       max_length=DEFAULT_MAX_PROMPT_TOKENS)
            input_ids_list.append(tokens["input_ids"])

        # Generate teacher continuations (once per shard, reused for all students)
        logger.info(f"  Generating teacher continuations ({len(input_ids_list)} prompts)...")
        shard_seed = int(hashlib.sha256(f"{args.seed}_{shard_idx}".encode()).hexdigest()[:8], 16)
        teacher_cache = generate_teacher_continuations(
            teacher_model, input_ids_list,
            max_new_tokens=args.max_new_tokens,
            block_seed=shard_seed,
            device=device,
        )

        # Evaluate each student model on this shard
        for model_name in models:
            if shard_idx in all_results[model_name]:
                logger.info(f"  {model_name}: already evaluated on shard {shard_idx}")
                continue

            logger.info(f"  Evaluating: {model_name}")
            try:
                student_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
                )
                student_model.eval()

                kl_values = evaluate_model_on_prompts(student_model, teacher_cache, device=device)

                if kl_values:
                    shard_mean_kl = float(np.mean(kl_values))
                    all_results[model_name][shard_idx] = shard_mean_kl
                    logger.info(f"    KL={shard_mean_kl:.6f} ({len(kl_values)} prompts)")
                else:
                    logger.warning(f"    No valid KL values for {model_name} on shard {shard_idx}")

                # Free GPU memory
                del student_model
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"    FAILED: {model_name} on shard {shard_idx}: {e}")

        # Save partial results after each shard
        partial_save = {m: {str(k): v for k, v in shard_data.items()} for m, shard_data in all_results.items()}
        partial_file.write_text(json.dumps(partial_save, indent=2))

    # ── Compute final statistics ──
    logger.info(f"\n{'='*60}")
    logger.info("COMPUTING FINAL STATISTICS")
    logger.info(f"{'='*60}")

    model_summaries = []
    for model_name in models:
        shard_kls = all_results[model_name]
        if not shard_kls:
            logger.warning(f"No results for {model_name}")
            continue

        values = np.array(list(shard_kls.values()))
        ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=args.bootstrap_samples)

        summary = {
            "model": model_name,
            "mean_kl": float(values.mean()),
            "std_kl": float(values.std()),
            "median_kl": float(np.median(values)),
            "min_kl": float(values.min()),
            "max_kl": float(values.max()),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_shards": len(values),
            "per_shard_kl": {str(k): v for k, v in shard_kls.items()},
        }
        model_summaries.append(summary)
        logger.info(f"  {model_name}: mean={summary['mean_kl']:.6f} ± {summary['std_kl']:.6f}  "
                     f"CI=[{ci_lo:.6f}, {ci_hi:.6f}]  n={len(values)}")

    # Paired t-tests between all model pairs
    paired_tests = []
    for (model_a, model_b) in combinations(models, 2):
        # Only compare on shards both models have
        common_shards = sorted(set(all_results[model_a].keys()) & set(all_results[model_b].keys()))
        if len(common_shards) < 3:
            logger.warning(f"  Too few common shards ({len(common_shards)}) for {model_a} vs {model_b}")
            continue

        a_vals = np.array([all_results[model_a][s] for s in common_shards])
        b_vals = np.array([all_results[model_b][s] for s in common_shards])

        test_result = paired_ttest(a_vals, b_vals)
        test_result["model_a"] = model_a
        test_result["model_b"] = model_b
        paired_tests.append(test_result)
        logger.info(f"  {model_a} vs {model_b}: diff={test_result['mean_diff']:+.6f}  "
                     f"t={test_result['t_stat']:.3f}  p={test_result['p_value']:.4f}")

    # Assemble final results
    results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "teacher": args.teacher,
            "num_shards": args.num_shards,
            "prompts_per_shard": args.prompts_per_shard,
            "seed": args.seed,
            "models": models,
        },
        "model_summaries": model_summaries,
        "paired_tests": paired_tests,
    }

    # Write JSON
    json_path = output_dir / "analysis_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nJSON results: {json_path}")

    # Write Markdown
    md_path = output_dir / "analysis_results.md"
    md_path.write_text(generate_markdown_table(results))
    logger.info(f"Markdown report: {md_path}")

    # Clean up partial file on success
    if partial_file.exists():
        partial_file.unlink()

    logger.info("\nDone! ✅")


if __name__ == "__main__":
    main()
