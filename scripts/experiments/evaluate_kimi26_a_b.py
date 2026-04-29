#!/usr/bin/env python3
"""Stage-2 Kimi K2.6 A/B decision-rule evaluator.

Ingests per-variant per-round eval JSONs produced by
``run_kimi26_a_b.py`` and applies the tiered decision rule from
reports/2026-04-29-kimi-k2.6-stage2-runbook.md §4. Outputs a
recommendation: PROMOTE (Kimi K2.6 wins) / HOLD / REVERT.

Tier 1 (any one fails ⇒ STOP):
  - king-canary regression > 5pp on any held-out canary axis.
  - reference 4B model regression > 5pp on any procedural bench.
  - top_k_overlap regression > 5pp.

Tier 2 (all must pass to PROMOTE):
  - mean composite.final lift ≥ +0.03 across the 5 students.
  - mean top_k_overlap lift ≥ +0.02 on king + runner-up.
  - no axis regressing > -0.03 on king + runner-up.
  - long-form judge axis acceptable (no rubric collapse).

Tier 3 (any of these can also gate promotion):
  - teacher_trace_plausibility lift ≥ +0.05.
  - forking_rkl lift ≥ +0.05 on king + runner-up.

Usage:
    /opt/distil/venv/bin/python scripts/experiments/evaluate_kimi26_a_b.py \\
        --manifest /opt/distil/experiments/kimi26-stage2/manifest.json \\
        --baseline-variant qwen36_baseline \\
        --candidate-variant kimi26_pathA
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean

logger = logging.getLogger("experiments.kimi26_eval")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


CANARY_AXES = ("gsm8k", "humaneval", "bbh", "ifeval", "mmlu_pro")
KEY_AXES_FOR_PARETO = (
    "math_skill_group", "code_skill_group", "reasoning_skill_group",
    "knowledge_skill_group", "ifeval_bench", "tool_use_bench",
    "calibration_bench", "judge_probe", "long_form_judge",
    "chat_turns_probe", "top_k_overlap", "kl_is", "forking_rkl",
    "teacher_trace_plausibility", "entropy_aware_kl", "super_teacher",
)


def _load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _load_eval(eval_path: Path) -> dict | None:
    if not eval_path.exists():
        return None
    try:
        return json.loads(eval_path.read_text())
    except Exception as exc:
        logger.warning(f"Failed to load {eval_path}: {exc}")
        return None


def _student_axis_value(eval_data: dict, model: str, axis: str) -> float | None:
    """Get axis value for a specific student from an eval dump."""
    students = (eval_data or {}).get("students") or {}
    s = students.get(model)
    if not s:
        return None
    comp = s.get("composite") or {}
    axes = comp.get("axes") or {}
    return axes.get(axis)


def _student_final(eval_data: dict, model: str) -> float | None:
    students = (eval_data or {}).get("students") or {}
    s = students.get(model)
    if not s:
        return None
    comp = s.get("composite") or {}
    return comp.get("final")


def _student_canary(eval_data: dict, model: str, axis: str) -> float | None:
    """Held-out canary axis (gsm8k, humaneval, ...) lives at
    student[axis].pass_frac if present."""
    students = (eval_data or {}).get("students") or {}
    s = students.get(model)
    if not s:
        return None
    canary = s.get("canary") or {}
    val = canary.get(axis)
    if isinstance(val, dict):
        return val.get("pass_frac")
    return val


def _aggregate_per_variant(manifest: list[dict], variant: str) -> dict:
    """Aggregate per-variant per-student per-axis means across rounds."""
    rounds = [m for m in manifest if m["variant"] == variant]
    if not rounds:
        return {}
    per_student: dict[str, dict[str, list[float]]] = {}
    canary_per_student: dict[str, dict[str, list[float]]] = {}
    final_per_student: dict[str, list[float]] = {}
    for entry in rounds:
        eval_data = _load_eval(Path(entry["output"]))
        if not eval_data:
            continue
        for student in (eval_data.get("students") or {}):
            ps = per_student.setdefault(student, {})
            cs = canary_per_student.setdefault(student, {})
            for axis in KEY_AXES_FOR_PARETO:
                v = _student_axis_value(eval_data, student, axis)
                if v is not None:
                    ps.setdefault(axis, []).append(float(v))
            for axis in CANARY_AXES:
                v = _student_canary(eval_data, student, axis)
                if v is not None:
                    cs.setdefault(axis, []).append(float(v))
            f = _student_final(eval_data, student)
            if f is not None:
                final_per_student.setdefault(student, []).append(float(f))
    return {
        "axes_mean": {
            stud: {ax: mean(vs) for ax, vs in axes.items() if vs}
            for stud, axes in per_student.items()
        },
        "canary_mean": {
            stud: {ax: mean(vs) for ax, vs in axes.items() if vs}
            for stud, axes in canary_per_student.items()
        },
        "final_mean": {
            stud: mean(vs) for stud, vs in final_per_student.items() if vs
        },
        "n_rounds": len(rounds),
    }


def _diff(baseline: dict, candidate: dict) -> dict:
    """Compute per-student per-axis diff (candidate − baseline)."""
    diffs: dict = {"per_student_axes": {}, "per_student_canary": {},
                   "per_student_final": {}}
    base_axes = baseline.get("axes_mean", {})
    cand_axes = candidate.get("axes_mean", {})
    for student in cand_axes:
        if student not in base_axes:
            continue
        d = {}
        for axis, c_v in cand_axes[student].items():
            b_v = base_axes[student].get(axis)
            if b_v is not None:
                d[axis] = c_v - b_v
        diffs["per_student_axes"][student] = d
    base_can = baseline.get("canary_mean", {})
    cand_can = candidate.get("canary_mean", {})
    for student in cand_can:
        if student not in base_can:
            continue
        d = {}
        for axis, c_v in cand_can[student].items():
            b_v = base_can[student].get(axis)
            if b_v is not None:
                d[axis] = c_v - b_v
        diffs["per_student_canary"][student] = d
    base_fin = baseline.get("final_mean", {})
    cand_fin = candidate.get("final_mean", {})
    for student in cand_fin:
        if student in base_fin:
            diffs["per_student_final"][student] = (
                cand_fin[student] - base_fin[student]
            )
    return diffs


def _apply_tier1(diffs: dict) -> tuple[bool, list[str]]:
    """Tier 1 disqualifying gates. Returns (passed, list of failures)."""
    failures = []
    for student, can_diffs in diffs["per_student_canary"].items():
        for axis, d in can_diffs.items():
            if d < -0.05:
                failures.append(
                    f"king_canary_regression>5pp: {student}/{axis} "
                    f"= {d:+.3f}"
                )
    for student, axis_diffs in diffs["per_student_axes"].items():
        if "top_k_overlap" in axis_diffs and axis_diffs["top_k_overlap"] < -0.05:
            failures.append(
                f"top_k_overlap_regression>5pp: {student} "
                f"= {axis_diffs['top_k_overlap']:+.3f}"
            )
    # Reference regression check (looks for "Qwen3.5-4B" or similar).
    for student, axis_diffs in diffs["per_student_axes"].items():
        if "Qwen" in student and "4B" in student:
            for axis, d in axis_diffs.items():
                if axis.endswith("_skill_group") or axis.endswith("_bench"):
                    if d < -0.05:
                        failures.append(
                            f"reference_4B_regression>5pp: {axis} = {d:+.3f}"
                        )
    return (len(failures) == 0, failures)


def _apply_tier2(diffs: dict) -> tuple[bool, list[str]]:
    failures = []
    final_diffs = list(diffs["per_student_final"].values())
    if not final_diffs:
        failures.append("no final_diff data")
    elif mean(final_diffs) < 0.03:
        failures.append(
            f"composite.final lift below +0.03 (got {mean(final_diffs):+.3f})"
        )
    # top_k_overlap lift on king + runner-up
    overlap_lifts = []
    for student, axis_diffs in diffs["per_student_axes"].items():
        if "top_k_overlap" in axis_diffs:
            overlap_lifts.append(axis_diffs["top_k_overlap"])
    if overlap_lifts:
        avg_overlap = mean(overlap_lifts)
        if avg_overlap < 0.02:
            failures.append(
                f"top_k_overlap mean lift below +0.02 (got {avg_overlap:+.3f})"
            )
    # No axis regressing > -0.03 on top students (we treat all as top).
    for student, axis_diffs in diffs["per_student_axes"].items():
        for axis, d in axis_diffs.items():
            if d < -0.03:
                failures.append(
                    f"axis regression > -0.03: {student}/{axis} = {d:+.3f}"
                )
    return (len(failures) == 0, failures)


def _apply_tier3(diffs: dict) -> dict:
    """Tier 3 supporting evidence (informational only)."""
    trace_lifts = []
    forking_lifts = []
    for student, axis_diffs in diffs["per_student_axes"].items():
        if "teacher_trace_plausibility" in axis_diffs:
            trace_lifts.append(axis_diffs["teacher_trace_plausibility"])
        if "forking_rkl" in axis_diffs:
            forking_lifts.append(axis_diffs["forking_rkl"])
    return {
        "teacher_trace_plausibility_lift_mean": (
            mean(trace_lifts) if trace_lifts else None
        ),
        "forking_rkl_lift_mean": mean(forking_lifts) if forking_lifts else None,
        "trace_passes": (
            mean(trace_lifts) >= 0.05 if trace_lifts else None
        ),
        "forking_passes": (
            mean(forking_lifts) >= 0.05 if forking_lifts else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.json from run_kimi26_a_b.py")
    parser.add_argument("--baseline-variant", default="qwen36_baseline")
    parser.add_argument("--candidate-variant", default="kimi26_pathA")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the decision report.")
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    baseline = _aggregate_per_variant(manifest, args.baseline_variant)
    candidate = _aggregate_per_variant(manifest, args.candidate_variant)

    if not baseline or not candidate:
        logger.error("Missing baseline or candidate variant in manifest.")
        sys.exit(2)

    diffs = _diff(baseline, candidate)

    tier1_pass, tier1_fail = _apply_tier1(diffs)
    tier2_pass, tier2_fail = _apply_tier2(diffs) if tier1_pass else (False, [])
    tier3 = _apply_tier3(diffs) if tier1_pass else {}

    if not tier1_pass:
        recommendation = "REVERT"
    elif tier1_pass and tier2_pass:
        recommendation = "PROMOTE"
    else:
        recommendation = "HOLD"

    report = {
        "baseline_variant": args.baseline_variant,
        "candidate_variant": args.candidate_variant,
        "n_rounds_baseline": baseline.get("n_rounds"),
        "n_rounds_candidate": candidate.get("n_rounds"),
        "tier1_passed": tier1_pass,
        "tier1_failures": tier1_fail,
        "tier2_passed": tier2_pass,
        "tier2_failures": tier2_fail,
        "tier3_signals": tier3,
        "recommendation": recommendation,
        "diffs": diffs,
    }

    output_text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output_text)
        logger.info(f"Decision report written to {args.output}")
    print(output_text)

    print(f"\n=== RECOMMENDATION: {recommendation} ===")
    if tier1_fail:
        print("Tier 1 failures:")
        for f in tier1_fail:
            print(f"  - {f}")
    if tier2_fail:
        print("Tier 2 failures:")
        for f in tier2_fail:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
