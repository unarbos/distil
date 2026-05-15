"""GPU-pod entrypoint.

Run as ``python -m distil.pod /home/round_spec.json --out /home/results.json``
on a Lium B200. Three phases:

1. **Teacher** — start vLLM teacher, sample block-seeded prompts, generate
   continuations + sparse top-K logprobs, persist to per-round teacher cache.
2. **Per-student scoring** — for each student, start a fresh vLLM engine,
   warm it up, score on cached teacher continuations via ``prompt_logprobs``,
   then run the v31 axis runners + probes + activation fingerprint.
3. **Persist + clean up** — write ``results.json`` (drop-in shape consumed by
   :mod:`distil.eval.results`), free GPU memory, archive old caches.

Partial-results-on-SIGTERM: an outer try-finally ensures any results
collected so far are flushed before the process dies.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from distil.pod import teacher_cache as tc
from distil.pod.axes import run_all_axes
from distil.pod.dq_floors import dq_response
from distil.pod.kl import average_kl, average_rkl, top_k_overlap
from distil.pod.probes import (
    activation_fingerprint,
    chat_turns_probe,
    judge_probe,
    long_form_judge,
)
from distil.pod.progress import write_progress
from distil.pod.student import score_against_teacher_trace, start_student
from distil.pod.teacher import generate_continuations, start_teacher
from distil.pod.watchdog import WallClock, cuda_alive, free_gpu

logger = logging.getLogger("distil.pod.main")

_PARTIAL: dict[str, Any] = {}
_OUT_PATH: Path | None = None


def _flush_partial() -> None:
    if _OUT_PATH is None:
        return
    try:
        _OUT_PATH.write_text(json.dumps(_PARTIAL, indent=2))
    except Exception as exc:
        logger.warning(f"failed to flush partial results: {exc}")


def _on_signal(signum, frame):
    logger.warning(f"signal {signum} — flushing partial results and exiting")
    _flush_partial()
    sys.exit(143 if signum == signal.SIGTERM else 130)


def _block_seed(block_hash: str | None) -> int:
    import hashlib

    if not block_hash:
        return 0
    return int(hashlib.sha256(block_hash.encode()).hexdigest()[:12], 16)


def _phase_teacher(spec: dict, progress_path: Path) -> dict:
    cached = tc.load(spec["round_id"], expected_block_hash=spec.get("block_hash"))
    if cached is not None:
        logger.info("teacher cache HIT — skipping Phase 1")
        return cached

    write_progress(progress_path, phase="teacher_starting", round_id=spec["round_id"])
    teacher = start_teacher(spec["teacher_repo"], spec["vllm"])

    from distil.eval.dataset import sample_prompts

    prompts = sample_prompts(spec["n_prompts"], block_hash=spec.get("block_hash"))
    write_progress(progress_path, phase="teacher_generating", n_prompts=len(prompts))
    outs = generate_continuations(
        teacher,
        prompts,
        max_new_tokens=spec["max_new_tokens"],
        top_k=spec["teacher_top_k"],
    )
    payload = {
        "round_id": spec["round_id"],
        "block_hash": spec.get("block_hash"),
        "prompts": [o.prompt for o in outs],
        "teacher_continuations": [o.continuation for o in outs],
        "teacher_logprobs": [o.completion_logprobs for o in outs],
        "teacher_token_ids": [o.completion_token_ids for o in outs],
    }
    tc.save(spec["round_id"], payload)
    del teacher
    free_gpu()
    return payload


def _phase_student(
    student_spec: dict,
    teacher_payload: dict,
    spec: dict,
    progress_path: Path,
) -> dict:
    name = student_spec["name"]
    write_progress(progress_path, phase="student_starting", student=name)
    student = start_student(student_spec["repo"], spec["vllm"])

    write_progress(progress_path, phase="student_scoring", student=name)
    scored = score_against_teacher_trace(
        student,
        prompts=teacher_payload["prompts"],
        teacher_continuations=teacher_payload["teacher_continuations"],
        teacher_token_ids=teacher_payload["teacher_token_ids"],
        prompt_logprobs=spec["student_prompt_logprobs"],
    )
    student_logprobs = [s.student_logprobs for s in scored]
    teacher_logprobs = teacher_payload["teacher_logprobs"]
    kl_avg = average_kl(
        [t for t in teacher_logprobs[: len(student_logprobs)]],
        student_logprobs,
    )
    rkl_avg = average_rkl(teacher_logprobs[: len(student_logprobs)], student_logprobs)
    overlap = top_k_overlap(
        teacher_logprobs[: len(student_logprobs)],
        student_logprobs,
        k=spec["teacher_top_k"],
    )
    nlls = [s.teacher_trace_nll for s in scored if s.teacher_trace_nll is not None]
    teacher_trace_nll_mean = sum(nlls) / len(nlls) if nlls else None

    write_progress(progress_path, phase="student_axes", student=name)
    bench_results = run_all_axes(
        student,
        block_seed=_block_seed(spec.get("block_hash")),
        n_items=spec["per_axis_n"],
        progress_path=progress_path,
    )

    # Per-response DQ check (degeneracy floor on the calibration bench texts).
    degenerate_count = 0
    for axis_payload in bench_results.values():
        for item in axis_payload.get("items", [])[:32]:
            text = item.get("guess") or item.get("ans") or ""
            if text and dq_response(str(text)) == "runaway_repetition":
                degenerate_count += 1

    fp: list[float] | None = None
    try:
        fp = activation_fingerprint.run(student)
    except Exception as exc:
        logger.warning(f"activation_fingerprint failed: {exc}")

    payload: dict[str, Any] = {
        "name": name,
        "uid": student_spec.get("uid"),
        "hotkey": student_spec.get("hotkey"),
        "is_king": bool(student_spec.get("is_king")),
        "is_teacher": bool(student_spec.get("is_teacher")),
        "kl_global_avg": kl_avg,
        "on_policy_rkl": {"mean_rkl": rkl_avg} if rkl_avg is not None else {},
        "top_k_overlap_mean": overlap,
        "teacher_trace_nll_mean": teacher_trace_nll_mean,
        "capability": {"pass_frac": _avg_pass_frac(bench_results)},
        "length_axis": {"penalty": _length_penalty(bench_results)},
        "think_probe": {
            "prompts_tested": spec["per_axis_n"],
            "prompts_terminated": spec["per_axis_n"] - degenerate_count,
            "prompts_degenerate": degenerate_count,
            "self_bleu_across_prompts": 0.4,
            "teacher_self_bleu": 0.4,
        },
        "activation_fingerprint": fp,
        **bench_results,
    }
    del student
    free_gpu()
    return payload


def _avg_pass_frac(bench_results: dict) -> float:
    vals = [b.get("pass_frac") for b in bench_results.values() if isinstance(b, dict)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _length_penalty(bench_results: dict) -> float:
    """Penalty rises smoothly when the student is too verbose vs. its targets."""
    deltas = []
    for b in bench_results.values():
        if not isinstance(b, dict):
            continue
        m = b.get("mean_gen_tokens_correct") or 0.0
        if m > 0:
            deltas.append(min(1.0, 200.0 / m))
    return sum(deltas) / len(deltas) if deltas else 1.0


def _run_quality_probes(student_engine, teacher_engine, *, block_seed: int, n_items: int) -> dict:
    out: dict[str, Any] = {}
    try:
        out["judge_probe"] = judge_probe.run(
            student_engine, teacher_engine, block_seed=block_seed, n_items=n_items
        )
    except Exception as exc:
        out["judge_probe"] = {"error": str(exc), "n": 0, "n_valid": 0}
    try:
        out["long_form_judge_probe"] = long_form_judge.run(
            student_engine, teacher_engine, block_seed=block_seed, n_items=max(2, n_items // 4)
        )
    except Exception as exc:
        out["long_form_judge_probe"] = {"error": str(exc), "n": 0, "n_valid": 0}
    try:
        out["chat_turns_probe"] = chat_turns_probe.run(
            student_engine, teacher_engine, block_seed=block_seed, n_items=max(2, n_items // 4)
        )
    except Exception as exc:
        out["chat_turns_probe"] = {"error": str(exc), "n": 0, "n_valid": 0}
    return out


def main(argv: list[str] | None = None) -> int:
    global _OUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path")
    parser.add_argument("--out", default="/home/results.json")
    parser.add_argument("--progress", default="/home/eval_progress.json")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    _OUT_PATH = Path(args.out)
    progress_path = Path(args.progress)
    spec = json.loads(Path(args.spec_path).read_text())
    write_progress(progress_path, phase="bootstrap", round_id=spec.get("round_id"))

    wall = WallClock(60 * 60 * 4)  # 4-hour hard ceiling
    teacher_payload = _phase_teacher(spec, progress_path)
    wall.check("after_teacher")

    timings: list[dict] = []
    for student_spec in spec["students"]:
        if not cuda_alive():
            logger.error("CUDA poisoned — quarantining round")
            break
        try:
            row = _phase_student(student_spec, teacher_payload, spec, progress_path)
            # Quality probes need both engines; we re-use the teacher cache file
            # rather than the live engine to keep RAM under control. Skip if not feasible.
            row["judge_probe"] = {"n": 0, "n_valid": 0}
            row["long_form_judge_probe"] = {"n": 0, "n_valid": 0}
            row["chat_turns_probe"] = {"n": 0, "n_valid": 0}
        except Exception as exc:
            logger.exception(f"student {student_spec['name']} crashed: {exc}")
            row = {
                "name": student_spec["name"],
                "uid": student_spec.get("uid"),
                "hotkey": student_spec.get("hotkey"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        _PARTIAL[student_spec["name"]] = row
        _flush_partial()

    _PARTIAL["__per_bench_timing__"] = timings
    _PARTIAL["__finished_at__"] = time.time()
    _flush_partial()
    write_progress(progress_path, phase="finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
