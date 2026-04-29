#!/usr/bin/env python3
"""Stage-2 Kimi K2.6 teacher-swap A/B experiment runner.

See reports/2026-04-29-kimi-k2.6-stage2-runbook.md for the full
experiment design. This script orchestrates a 5-student × 3-variant ×
6-round comparison on a SEPARATE evaluation pod from production. The
production validator continues running on Qwen3.6-35B-A3B during the
experiment.

Variants (per the runbook §3.1):
  - ``qwen36_baseline``  Qwen3.6-35B-A3B teacher (production reference).
  - ``kimi26_pathA``     Kimi K2.6 with Qwen-tokenizer round-trip
                         (lossy but cheap; uses eval/cross_tokenizer).
  - ``kimi26_pathB``     Kimi K2.6 native via Universal Logit
                         Distillation. Only run if Path A passes Tier 1
                         but fails Tier 2 (still need an honest
                         A/B comparison without the round-trip drift).

Output: per-variant per-round per-student composite_scores under
``/opt/distil/experiments/kimi26-stage2/round_{seed}/{variant}/``.
The companion ``evaluate_kimi26_a_b.py`` ingests these and applies
the tiered decision rule.

Status: SCAFFOLDING. The actual pod orchestration requires a
multi-GPU pod with Kimi K2.6 weights pre-staged. The sequence below
is correct; running it on a real pod is the operator step.

Usage:
    # On the experiment pod, with K2.6 weights present:
    /opt/distil/venv/bin/python scripts/experiments/run_kimi26_a_b.py \\
        --students UID156,UID149,UID123,UID217,UID-1 \\
        --rounds 6 \\
        --output-dir /opt/distil/experiments/kimi26-stage2 \\
        --variants qwen36_baseline,kimi26_pathA
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("experiments.kimi26_ab")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


VARIANT_TEACHER_CONFIG: dict[str, dict] = {
    "qwen36_baseline": {
        "teacher_model": "Qwen/Qwen3.6-35B-A3B",
        "teacher_tp_size": 0,  # auto = all visible GPUs
        "tokenizer_path_extra": None,
        "use_cross_tokenizer": False,
        "vllm_concurrency": 16,
    },
    "kimi26_pathA": {
        "teacher_model": "moonshot/Kimi-K2.6-Instruct",
        "teacher_tp_size": 0,
        "tokenizer_path_extra": None,
        "use_cross_tokenizer": True,
        "vllm_concurrency": 16,
    },
    "kimi26_pathB": {
        "teacher_model": "moonshot/Kimi-K2.6-Instruct",
        "teacher_tp_size": 0,
        "tokenizer_path_extra": None,
        "use_cross_tokenizer": False,  # native via ALM (separate path)
        "vllm_concurrency": 16,
    },
}


def _run_one_eval(variant: str, students: list[str], block_seed: int,
                  output_dir: Path) -> Path:
    """Run pod_eval_vllm.py for one variant × one block_seed.

    The student set is identical across variants so the comparison
    isolates the teacher swap. Returns the path to the per-variant
    eval JSON.
    """
    cfg = VARIANT_TEACHER_CONFIG[variant]
    out_path = output_dir / f"round_{block_seed}" / variant / "eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).parents[1] / "pod_eval_vllm.py"),
        "--teacher", cfg["teacher_model"],
        "--students", ",".join(students),
        "--block-seed", str(block_seed),
        "--output", str(out_path),
        "--use-vllm",
        "--concurrency", str(cfg["vllm_concurrency"]),
        "--tensor-parallel-size", str(cfg["teacher_tp_size"]),
    ]
    if cfg["use_cross_tokenizer"]:
        cmd.append("--cross-tokenizer-roundtrip")

    logger.info(
        f"Running variant={variant} block_seed={block_seed} "
        f"students={len(students)} → {out_path}"
    )
    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        logger.info(
            f"  ✓ {variant}/{block_seed} completed in {elapsed:.0f}s "
            f"({len(result.stdout.splitlines())} lines stdout)"
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        logger.error(
            f"  ✗ {variant}/{block_seed} failed after {elapsed:.0f}s: "
            f"{exc.stderr[:500] if exc.stderr else exc}"
        )
        # Write a sentinel error file so the evaluator can flag the
        # missing data point rather than silently skipping.
        out_path.write_text(json.dumps({
            "error": str(exc)[:1000],
            "stderr_tail": (exc.stderr or "")[-1000:],
            "variant": variant,
            "block_seed": block_seed,
        }))
    return out_path


def main():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--students", required=True,
                        help="Comma-separated student model paths or UIDs")
    parser.add_argument("--rounds", type=int, default=6,
                        help="Number of fresh block_seeds to evaluate "
                             "(default 6 per the runbook).")
    parser.add_argument("--seed-base", type=int, default=int(time.time()),
                        help="Base block_seed; rounds use seed_base + round_idx.")
    parser.add_argument("--variants", default="qwen36_baseline,kimi26_pathA",
                        help="Comma-separated variants to run.")
    parser.add_argument("--output-dir",
                        default="/opt/distil/experiments/kimi26-stage2",
                        help="Where to write per-variant per-round results.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command sequence without running.")
    args = parser.parse_args()

    students = [s.strip() for s in args.students.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check variants are known.
    for v in variants:
        if v not in VARIANT_TEACHER_CONFIG:
            logger.error(f"Unknown variant: {v}. "
                         f"Known: {list(VARIANT_TEACHER_CONFIG)}")
            sys.exit(2)

    logger.info(
        f"Stage-2 K2.6 A/B: variants={variants} rounds={args.rounds} "
        f"students={len(students)}"
    )
    if args.dry_run:
        for r in range(args.rounds):
            for v in variants:
                logger.info(
                    f"DRY RUN: variant={v} block_seed={args.seed_base + r}"
                )
        return

    # Run each variant × each round serially (can be parallelised on
    # multi-GPU pods by spawning one subprocess per GPU group, but
    # the per-round wall time dominates so serial is fine for a
    # 6-round experiment).
    manifest_entries: list[dict] = []
    for round_idx in range(args.rounds):
        block_seed = args.seed_base + round_idx
        for variant in variants:
            out_path = _run_one_eval(variant, students, block_seed, output_dir)
            manifest_entries.append({
                "round": round_idx,
                "block_seed": block_seed,
                "variant": variant,
                "output": str(out_path),
            })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2))
    logger.info(f"Manifest: {manifest_path}")
    logger.info(
        "Run scripts/experiments/evaluate_kimi26_a_b.py to apply the "
        "tiered decision rule and produce a recommendation."
    )


if __name__ == "__main__":
    main()
