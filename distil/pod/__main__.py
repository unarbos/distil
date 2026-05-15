"""GPU-pod entrypoint.

Run as ``python -m distil.pod /home/round_spec.json --out /home/results.json``
on a Lium B200 (or persistent 8xB200 pod). Three phases:

1. **Teacher continuations** — start vLLM teacher, generate block-seeded
   prompt continuations with sparse top-K logprobs, persist to the
   per-round teacher cache, then UNLOAD the teacher to free memory.
2. **Per-student scoring** — for every student:
   * Load via vLLM (own engine), warm-up.
   * Score teacher-trace KL / RKL / top-K overlap from cached logprobs.
   * Run the 12 axis runners (11 v31 procedural + calibration_bench)
     via :func:`distil.pod.axes.run_all_axes`.
   * Collect raw responses for judge / long-form / chat-turns probes
     (graded in phase 3 — teacher isn't loaded yet).
   * Record activation fingerprint, unload student.
3. **Judge phase** — re-load the teacher one final time and grade all
   collected judge / long-form / chat-turns responses. Results merged
   into each student's payload.

Partial-results-on-SIGTERM: an outer try-finally flushes any results
collected so far before the process dies.
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

from distil.pod import cache as hf_cache
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
        logger.info("teacher cache HIT — skipping Phase 1 generation")
        return cached

    from distil.eval.dataset import sample_prompts

    prompts = sample_prompts(spec.get("n_prompts", 300), block_hash=spec.get("block_hash"))
    max_new = spec.get("max_new_tokens", 512)
    top_k = spec.get("teacher_top_k", 5)

    # Production path: Kimi-K2.6 is 1T params and does NOT fit in vLLM
    # on the 8xB200 pod, so we route teacher inference through an
    # OpenAI-compatible cloud API (default: OpenRouter / Inceptron).
    # The same env var the legacy validator uses controls this:
    # DISTIL_TEACHER_MODE=api  →  distil.pod.teacher_api
    # DISTIL_TEACHER_MODE=vllm →  distil.pod.teacher (local vLLM)
    teacher_mode = os.environ.get("DISTIL_TEACHER_MODE", "vllm").lower().strip()
    if teacher_mode == "api":
        from distil.pod.teacher_api import generate_continuations_api

        write_progress(
            progress_path,
            phase="teacher_generating",
            round_id=spec["round_id"],
            n_prompts=len(prompts),
            mode="api",
        )
        outs = generate_continuations_api(prompts, max_new_tokens=max_new, top_k=top_k)
    else:
        write_progress(progress_path, phase="teacher_starting", round_id=spec["round_id"])
        teacher = start_teacher(spec["teacher_repo"], spec.get("vllm") or {})
        write_progress(
            progress_path,
            phase="teacher_generating",
            n_prompts=len(prompts),
            mode="vllm",
        )
        outs = generate_continuations(teacher, prompts, max_new_tokens=max_new, top_k=top_k)

    payload = {
        "round_id": spec["round_id"],
        "block_hash": spec.get("block_hash"),
        "prompts": [o.prompt for o in outs],
        "teacher_continuations": [o.continuation for o in outs],
        "teacher_logprobs": [o.completion_logprobs for o in outs],
        "teacher_token_ids": [o.completion_token_ids for o in outs],
    }
    tc.save(spec["round_id"], payload)
    if teacher_mode != "api":
        del teacher  # noqa: F821 — only bound on the vllm branch
        free_gpu()
    return payload


def _phase_student(
    student_spec: dict,
    teacher_payload: dict,
    spec: dict,
    progress_path: Path,
) -> tuple[dict, dict]:
    """Returns ``(student_payload, raw_responses)``. Raw responses are
    graded in :func:`_phase_judge` after the teacher reloads."""
    name = student_spec["name"]
    write_progress(progress_path, phase="student_starting", student=name)
    student = start_student(student_spec["repo"], spec.get("vllm") or {})

    write_progress(progress_path, phase="student_scoring", student=name)
    scored = score_against_teacher_trace(
        student,
        prompts=teacher_payload["prompts"],
        teacher_continuations=teacher_payload["teacher_continuations"],
        teacher_token_ids=teacher_payload["teacher_token_ids"],
        prompt_logprobs=spec.get("student_prompt_logprobs", True),
    )
    student_logprobs = [s.student_logprobs for s in scored]
    teacher_logprobs = teacher_payload["teacher_logprobs"]
    kl_avg = average_kl(teacher_logprobs[: len(student_logprobs)], student_logprobs)
    rkl_avg = average_rkl(teacher_logprobs[: len(student_logprobs)], student_logprobs)
    overlap = top_k_overlap(
        teacher_logprobs[: len(student_logprobs)],
        student_logprobs,
        k=spec.get("teacher_top_k", 5),
    )
    nlls = [s.teacher_trace_nll for s in scored if s.teacher_trace_nll is not None]
    teacher_trace_nll_mean = sum(nlls) / len(nlls) if nlls else None

    write_progress(progress_path, phase="student_axes", student=name)
    bench_results = run_all_axes(
        student,
        block_seed=_block_seed(spec.get("block_hash")),
        n_items=spec.get("per_axis_n", 16),
        progress_path=progress_path,
    )

    degenerate_count = 0
    for axis_payload in bench_results.values():
        for item in axis_payload.get("items", [])[:32]:
            text = item.get("tail") or item.get("guess") or item.get("ans") or ""
            if text and dq_response(str(text)) == "runaway_repetition":
                degenerate_count += 1

    write_progress(progress_path, phase="student_probe_collect", student=name)
    n_judge = spec.get("judge_n_items", 8)
    n_chat = spec.get("chat_n_items", 4)
    raw = {
        "judge_responses": _collect_judge_responses(student, n_items=n_judge),
        "long_form_responses": _collect_long_form_responses(student, n_items=n_judge),
        "chat_dialogues": _collect_chat_dialogues(student, n_items=n_chat),
    }

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
        "degenerate_count": degenerate_count,
        "activation_fingerprint": fp,
        **bench_results,
    }
    del student
    free_gpu()
    return payload, raw


def _phase_judge(
    spec: dict,
    raw_by_student: dict[str, dict],
    progress_path: Path,
) -> dict[str, dict[str, Any]]:
    """Re-load the teacher and grade collected responses for every student."""
    if not raw_by_student:
        return {}
    write_progress(progress_path, phase="judge_loading")
    teacher = start_teacher(spec["teacher_repo"], spec.get("vllm") or {})
    out: dict[str, dict[str, Any]] = {}
    try:
        for name, raw in raw_by_student.items():
            write_progress(progress_path, phase="judge_grading", student=name)
            out[name] = {
                "judge_probe": judge_probe.grade_responses(
                    teacher, raw["judge_responses"]
                ),
                "long_form_judge_probe": long_form_judge.grade_responses(
                    teacher, raw["long_form_responses"]
                ),
                "chat_turns_probe": chat_turns_probe.grade_dialogues(
                    teacher, raw["chat_dialogues"]
                ),
            }
    finally:
        del teacher
        free_gpu()
    return out


def _collect_judge_responses(engine, *, n_items: int) -> list[dict]:
    return judge_probe.collect_responses(engine, n_items=n_items)


def _collect_long_form_responses(engine, *, n_items: int) -> list[dict]:
    return long_form_judge.collect_responses(engine, n_items=n_items)


def _collect_chat_dialogues(engine, *, n_items: int) -> list[dict]:
    return chat_turns_probe.collect_dialogues(engine, n_items=n_items)


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


def _filter_students_for_shard(students: list[dict], shard_idx: int, n_shards: int) -> list[dict]:
    """Round-robin shard slice (preserves king position 0 on shard 0)."""
    return students[shard_idx::n_shards]


def main(argv: list[str] | None = None) -> int:
    global _OUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path")
    parser.add_argument("--out", default="/home/results.json")
    parser.add_argument("--progress", default="/home/eval_progress.json")
    parser.add_argument(
        "--phase",
        choices=("all", "teacher", "students", "judge"),
        default="all",
        help="Run a single phase (used by the multi-GPU orchestrator).",
    )
    parser.add_argument(
        "--shard",
        default="0/1",
        help="Shard ``idx/n`` for --phase=students (e.g. ``3/8``).",
    )
    parser.add_argument(
        "--raw-in",
        default=None,
        help="For --phase=judge: path to a merged raw_responses.json from shards.",
    )
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

    # ── Phase 0: pre-round HF cache broom (any phase except judge) ──
    if args.phase in ("all", "teacher", "students"):
        keep_repos = [spec["teacher_repo"]] + [s["repo"] for s in spec.get("students", [])]
        try:
            hf_cache.sweep(keep_repos)
        except Exception as exc:
            logger.warning(f"pre-round cache sweep failed: {exc}")

    write_progress(progress_path, phase="bootstrap", round_id=spec.get("round_id"))
    wall = WallClock(60 * 60 * 4)

    # ── Phase 1: teacher continuations ──────────────────────────────
    if args.phase in ("all", "teacher"):
        _phase_teacher(spec, progress_path)
        wall.check("after_teacher")
    if args.phase == "teacher":
        write_progress(progress_path, phase="finished")
        return 0

    # ── Phase 2: per-student scoring + raw probe collection ─────────
    if args.phase in ("all", "students"):
        teacher_payload = _phase_teacher(spec, progress_path)  # cache hit on shards
        shard_idx, n_shards = (int(x) for x in args.shard.split("/"))
        my_students = _filter_students_for_shard(spec["students"], shard_idx, n_shards)
        logger.info(f"shard {shard_idx}/{n_shards}: {len(my_students)} students")
        raw_by_student: dict[str, dict] = {}
        for student_spec in my_students:
            if not cuda_alive():
                logger.error("CUDA poisoned — quarantining round")
                break
            try:
                row, raw = _phase_student(student_spec, teacher_payload, spec, progress_path)
            except Exception as exc:
                logger.exception(f"student {student_spec['name']} crashed: {exc}")
                row = {
                    "name": student_spec["name"],
                    "uid": student_spec.get("uid"),
                    "hotkey": student_spec.get("hotkey"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raw = None
            _PARTIAL[student_spec["name"]] = row
            if raw is not None and "error" not in row:
                raw_by_student[student_spec["name"]] = raw
            _flush_partial()
            try:
                hf_cache.clean_model(
                    student_spec["repo"], keep_repos=(spec["teacher_repo"],)
                )
            except Exception as exc:
                logger.warning(f"post-student cache clean failed: {exc}")
            wall.check(f"after_student_{student_spec['name']}")

        if args.phase == "students":
            # Side-car the raw responses so the orchestrator can merge them.
            raw_path = _OUT_PATH.with_suffix(".raw.json")
            raw_path.write_text(json.dumps(raw_by_student, indent=2))
            write_progress(progress_path, phase="finished")
            return 0

    # ── Phase 3: teacher reload + judge grading ─────────────────────
    if args.phase in ("all", "judge"):
        if args.phase == "judge":
            raw_path = Path(args.raw_in) if args.raw_in else _OUT_PATH.with_suffix(".raw.json")
            raw_by_student = json.loads(raw_path.read_text()) if raw_path.exists() else {}
            # In --phase=judge mode, the _PARTIAL we'll merge into comes from --out (the merged shards).
            if _OUT_PATH.exists():
                _PARTIAL.update(json.loads(_OUT_PATH.read_text()))
        try:
            graded = _phase_judge(spec, raw_by_student, progress_path)
        except Exception as exc:
            logger.exception(f"judge phase crashed: {exc}")
            graded = {}
        for name, scores in graded.items():
            if name in _PARTIAL and isinstance(_PARTIAL[name], dict):
                _PARTIAL[name].update(scores)

    _PARTIAL["__finished_at__"] = time.time()
    _flush_partial()
    write_progress(progress_path, phase="finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
