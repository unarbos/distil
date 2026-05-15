"""v31 procedural axis runners.

Each module exports ``run(student_engine, *, prompts_seed, n_items, ...) -> dict``
returning the standard payload ``{n, correct, pass_frac, mean_gen_tokens_correct,
items, ...}``. A ``run_all_axes`` orchestrator records per-bench wall-time +
tokens/sec into ``eval_progress.json`` (improvement #5).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from distil.pod.axes import (
    calibration_bench,
    code_humaneval,
    consistency,
    ifeval,
    knowledge_kg,
    long_context_ruler,
    math_competition,
    math_gsm,
    math_robustness,
    reasoning_dyval,
    reasoning_logic_grid,
    truthfulness,
)
from distil.pod.progress import record_bench_timing

logger = logging.getLogger("distil.pod.axes")


AXIS_RUNNERS: dict[str, Callable] = {
    "v31_math_gsm_symbolic": math_gsm.run,
    "v31_math_competition": math_competition.run,
    "v31_math_robustness": math_robustness.run,
    "v31_code_humaneval_plus": code_humaneval.run,
    "v31_ifeval_verifiable": ifeval.run,
    "v31_reasoning_logic_grid": reasoning_logic_grid.run,
    "v31_reasoning_dyval_arith": reasoning_dyval.run,
    "v31_long_context_ruler": long_context_ruler.run,
    "v31_knowledge_multi_hop_kg": knowledge_kg.run,
    "v31_truthfulness_calibration": truthfulness.run,
    "v31_consistency_paraphrase": consistency.run,
    "calibration_bench": calibration_bench.run,
}


def run_all_axes(
    student_engine,
    *,
    block_seed: int,
    n_items: int,
    progress_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Run every v31 axis on ``student_engine``; record per-bench timing."""
    results: dict[str, dict[str, Any]] = {}
    for name, runner in AXIS_RUNNERS.items():
        t0 = time.time()
        try:
            payload = runner(student_engine, block_seed=block_seed, n_items=n_items)
        except Exception as exc:
            logger.exception(f"axis {name} crashed: {exc}")
            payload = {"error": f"{type(exc).__name__}: {exc}", "n": 0}
        results[name] = payload
        if progress_path is not None:
            record_bench_timing(
                progress_path,
                name=name,
                wall_s=time.time() - t0,
                n_prompts=int(payload.get("n") or 0),
                completion_tokens=int(payload.get("completion_tokens") or 0),
            )
    return results


__all__ = ["AXIS_RUNNERS", "run_all_axes"]
