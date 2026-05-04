"""Multi-axis composite score computation — production ranking + dethrone gate.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible on the 2026-04-22 SN97 king, which rambles
3–10x longer than the teacher on trivial prompts while passing KL). The
fix is to score each student on several independent axes and combine them
with a worst-case rule, so that gaming any single axis penalizes overall
rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: PRODUCTION — ranking + dethrone veto.
  * 2026-04-19 (commit 8eec9a2): promoted from shadow to production
    ranking key. ``composite.worst`` orders the leaderboard and selects
    the canonical challenger for display.
  * 2026-04-22: ``composite.worst`` is now ALSO a dethrone gate. A
    challenger that passes the KL paired t-test + 3% epsilon is still
    blocked from taking the crown if its worst composite axis is below
    ``COMPOSITE_DETHRONE_FLOOR`` (currently 0.20). See
    ``scripts/validator/results.py::_composite_dethrone_veto``.
  * Same commit: the ``length`` axis is now always populated even when
    ``THINK_COLLAPSE_PROBE=0``. It falls back to the always-on
    ``chat_probe`` length vs a teacher anchor captured in
    ``prepare_teacher_probe_refs_*``. This closes the gap that let a
    KL-specialized-but-rambling model keep the crown unopposed.
  * 2026-04-23: judge-probe axis added in SHADOW mode. The teacher
    scores each student's greedy response to 16 rotated realistic
    prompts on a 1-5 rubric, normalized to [0, 1]. Computed + logged
    per round but excluded from ``worst`` / ``weighted`` aggregation
    until the ``JUDGE_AXIS_IN_COMPOSITE`` gate flips (Session 2). See
    ``reports/2026-04-23-goodhart-immune-eval.md``.
  * 2026-04-24: **Arena v3 — comprehensive eval**. Session 2 promoted:
    the five absolute-correctness bench axes (``math_bench``,
    ``code_bench``, ``reasoning_bench``, ``knowledge_bench``,
    ``ifeval_bench``) and ``judge_probe`` are now all IN the composite
    ranking by default. These break the last Goodhart hole — the old
    six axes all scored *relative* to the teacher, so a perfectly-
    distilled model of a non-SOTA teacher ranked #1 but couldn't do
    grade-school math. New axes score against ground truth so
    overfitting them ⇒ SOTA small model.
  * 2026-04-24: **Session 3 axes promoted live**:
    ``aime_bench`` (AIME olympiad math), ``mbpp_bench`` (MBPP+ code),
    ``tool_use_bench`` (agentic Python tool use), and
    ``self_consistency_bench`` (majority-vote over sampled generations).
    These give miners more surface area to optimize against, each
    pointing towards a genuinely valuable capability. See
    ``reports/2026-04-24-arena-v3.md`` for the full Affine-Cortex-
    inspired design.
  * 2026-04-24: **Pareto majority dominance**: in addition
    to the worst-axis floor, a challenger that beats the king on KL
    but loses on a majority of axes is blocked. This is part of the
    dethrone gate by default after the public telemetry window.
  * 2026-04-25: Session 3.1 ``arc_bench`` (AI2 ARC-Challenge
    commonsense science MC), 3.2 ``reasoning_density`` (pass_frac ×
    length_bonus — explicitly penalizes over-think-on-trivia),
    3.3 ``chat_turns_probe`` (teacher-graded 3-turn dialogues),
    3.4 ``truthful_bench`` (TruthfulQA adversarial factuality),
    3.5 ``long_context_bench`` (procedural needle-in-haystack
    over ~1400 tokens — literally uncheatable because items are
    generated fresh every round from the block_seed, no fixed
    dataset exists), and 3.6 ``procedural_bench`` (block-seeded
    synthetic reasoning / instruction following / factual retrieval).
    Each targets a capability that Session 2 + relative axes don't
    already reward, so climbing them requires genuine model improvement. See
    ``reports/2026-04-24-arena-v3.md`` and the ``MINER_FAQ.md``
    playbook.

Axes that are missing for a given round (e.g. ``degeneracy`` while
``THINK_COLLAPSE_PROBE=0``) drop out and the weighted mean renormalizes
over the surviving axes. The veto fails open if fewer than
``COMPOSITE_DETHRONE_MIN_AXES`` axes are populated — we don't want a pod
probe outage to freeze the crown.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Five-axis composite (T1.3). On-policy RKL is the primary distillation
# signal under the new framework — it is the axis that miners cannot game
# by teacher-forced memorization, because the rollouts are the student's
# own policy. KL (off-policy forward-KL) is retained as a transparency
# axis but down-weighted; its existing saturation at the top of the board
# (Δ < 0.0005 nats across the top-5) is exactly what we’re moving away
# from. Capability and the two structural axes round out the signal so a
# model must be competitive on reasoning, length discipline, and
# non-degenerate generation — not just logit-matching.
AXIS_WEIGHTS = {
    # Tier 1: relative (teacher-referenced) axes. Production since
    # 2026-04-19. Kept at the same relative weighting; the weights
    # below are only used by the ``weighted`` aggregation, which is
    # auxiliary — the production ranking key is ``worst``.
    # 2026-04-29 (v29.7): all relative weights are now env-overridable
    # so the v30 audit rebalance (drop saturated kl/capability/length)
    # can land via distil.env without a code change.
    "on_policy_rkl":  float(os.environ.get("ON_POLICY_RKL_WEIGHT", "0.35")),
    "kl":             float(os.environ.get("BENCH_KL_WEIGHT", "0.15")),
    # 2026-04-29 (v30): top-K overlap axis. Per the 2026 'Rethinking OPD'
    # paper, this is the most predictive single signal of downstream
    # OPD success. Conservative initial weight 0.10 — same magnitude
    # as a single bench axis. Already in [0, 1] from per-prompt
    # |top_K_t ∩ top_K_s| / K averaged. Add to the weighted
    # aggregation by default; gated env-overridable while we collect
    # 48h of telemetry (see TOP_K_OVERLAP_AXIS_IN_COMPOSITE below).
    "top_k_overlap":  float(os.environ.get("TOP_K_OVERLAP_AXIS_WEIGHT", "0.10")),
    # 2026-04-29 (v30) — entropy-aware adaptive KL axis. Per the EOPD
    # paper (arXiv 2510.27485), per-token RKL/FKL weighting based on
    # teacher entropy is +1.37 to +5.05 Pass@8 on small-model math
    # benchmarks. Defaults to 0 (SHADOW) — we collect 48h of correlation
    # telemetry against the held-out canary before promoting to ranking.
    # Operators can flip in via ``ENTROPY_AWARE_KL_WEIGHT=0.05`` once
    # the shadow window concludes.
    "entropy_aware_kl": float(os.environ.get("ENTROPY_AWARE_KL_WEIGHT", "0.0")),
    # 2026-04-29 (v30) — three additional research-paper shadow axes,
    # all default weight 0 until 48h of correlation telemetry against
    # the held-out canary validates them. Computed from the existing
    # top-128 sparse cache so wall-time impact is negligible.
    #
    #   * kl_is — Anshumann ACL 2025 importance-sampled KL. Unbiased
    #     full-vocab KL contribution from top-K support, replacing the
    #     biased renormalised KL on shared support. See
    #     ``compute_kl_is_from_sparse``.
    #   * forking_rkl — Wang et al. 2025 forking-token RKL. Average
    #     reverse-KL at positions in the top quartile of teacher
    #     entropy ("decision points" with most informative teacher
    #     feedback).
    #   * teacher_trace_plausibility — average NLL the student
    #     assigns to the teacher's actually-emitted tokens. Distinct
    #     from FKL (full-distribution match) and RKL (student-policy
    #     match); catches LIMO/s1 SFT-only "place mass everywhere
    #     except where the teacher goes" failure modes.
    "kl_is":          float(os.environ.get("KL_IS_AXIS_WEIGHT", "0.0")),
    "forking_rkl":    float(os.environ.get("FORKING_RKL_AXIS_WEIGHT", "0.0")),
    "teacher_trace_plausibility": float(
        os.environ.get("TEACHER_TRACE_PLAUSIBILITY_WEIGHT", "0.0")
    ),
    # 2026-04-29 (v30.3) — tail-decoupled KL shadow axis. Detects
    # "match teacher head but flatten tail" over-confidence pathology
    # documented in the Tail-Aware Distillation paper. Default 0
    # (SHADOW); promote once a 48h round-correlation pass against the
    # canary confirms the signal.
    "tail_decoupled_kl": float(
        os.environ.get("TAIL_DECOUPLED_KL_WEIGHT", "0.0")
    ),
    "capability":     float(os.environ.get("BENCH_CAPABILITY_WEIGHT", "0.25")),
    "length":         float(os.environ.get("BENCH_LENGTH_WEIGHT", "0.10")),
    "degeneracy":     float(os.environ.get("BENCH_DEGENERACY_WEIGHT", "0.15")),
}

# v30 — top-K overlap axis gate. Defaults ON; env-flippable for fast
# rollback if the 48h shadow window flags any issue with the new axis.
TOP_K_OVERLAP_AXIS_IN_COMPOSITE = (
    os.environ.get("TOP_K_OVERLAP_AXIS_IN_COMPOSITE", "1") != "0"
)

# 2026-04-23 — judge axis weight. When promoted, the other axes retain
# their relative weighting and the judge weight is added on top before
# normalization; callers see a per-axis breakdown and the aggregated
# worst/weighted, so the absolute number changes slightly but the
# ordering intent is preserved.
JUDGE_AXIS_WEIGHT = float(os.environ.get("JUDGE_AXIS_WEIGHT", "0.15"))

# Shadow/promote gate. 2026-04-24: PROMOTED to production after the
# original 48h telemetry window. Override with ``JUDGE_AXIS_IN_COMPOSITE=0``
# if a teacher-rubric outage requires a temporary rollback.
JUDGE_AXIS_IN_COMPOSITE = os.environ.get("JUDGE_AXIS_IN_COMPOSITE", "1") != "0"

# ── 2026-04-24 — Arena v3 Session 2 (PRODUCTION) ──────────────────────
# Five absolute-correctness axes drawn from public held-out benchmarks
# (GSM8K+MATH-500 / HumanEval / BBH / MMLU-Pro / IFEval). Each
# normalized to [0, 1] by raw ``pass_frac``. Promoted to composite
# ranking 2026-04-24 after the planned 48h shadow window (see the
# 2026-04-24-pareto-holistic-eval-v2.md report section 5 + the
# Discord 48h announcement).
BENCH_AXIS_WEIGHTS = {
    # v30.2 (2026-04-29) — Skill-group regrouping. The individual bench
    # sub-axes (code_bench, math_bench, ...) are now 0 by default; the
    # composite weight has migrated to the group axes (code_skill_group,
    # math_skill_group, etc.) defined below in BENCH_GROUP_AXIS_WEIGHTS.
    # Sub-axes are still computed (for telemetry/per_src/saturation
    # audit) but don't directly drive ranking. Setting any sub-axis
    # weight back > 0 via env restores its individual ranking
    # contribution (useful for ablations).
    "math_bench":      float(os.environ.get("BENCH_MATH_WEIGHT", "0.0")),
    "code_bench":      float(os.environ.get("BENCH_CODE_WEIGHT", "0.0")),
    "reasoning_bench": float(os.environ.get("BENCH_REASONING_WEIGHT", "0.0")),
    "knowledge_bench": float(os.environ.get("BENCH_KNOWLEDGE_WEIGHT", "0.0")),
    "ifeval_bench":    float(os.environ.get("BENCH_IFEVAL_WEIGHT", "0.07")),
}

# v30.2 — Skill-group axis weights. These pick up the weight previously
# distributed across the individual bench sub-axes.
#
# Net comparison (v30 → v30.2):
#   v30: code 0.14 + mbpp 0.06 + debug 0.06 + correction 0.03 + refactor 0.04
#        = 0.33
#   v30.2: code_skill_group 0.20 (sub-axes 0). Net cut: 0.13 redirected.
#
#   v30: math 0.14 + aime 0.10 + robustness 0.07 = 0.31
#   v30.2: math_skill_group 0.18. Net cut: 0.13.
#
#   v30: reasoning 0.10 + long_context 0.04 + multi_doc 0.05 = 0.19
#   v30.2: reasoning_skill_group 0.12. Net cut: 0.07.
#
#   v30: knowledge 0.05 + pragmatic 0.04 = 0.09
#   v30.2: knowledge_skill_group 0.07. Net cut: 0.02.
#
# Net saved: 0.35 weight redirected. Allocated to:
#   super_teacher 0.10 (incentivize beyond-teacher)
#   on_policy_rkl 0.35 → 0.40 (largest signal stays largest)
#   judge_probe 0.15 (unchanged — short-form quality)
#   long_form_judge 0.05 → 0.08 (essay quality)
#   tool_use_bench 0.06 (unchanged — agentic Python distinct)
#   calibration_bench 0.06 (unchanged — refusal distinct)
#   ifeval_bench 0.07 (kept separate — instruction following distinct)
#   chat_turns_probe 0.08 (unchanged — multi-turn distinct)
#   shadow axes (kl_is etc.) 0.02 → 0.05 each
BENCH_GROUP_AXIS_WEIGHTS = {
    "code_skill_group":      float(os.environ.get("CODE_SKILL_GROUP_WEIGHT", "0.20")),
    "math_skill_group":      float(os.environ.get("MATH_SKILL_GROUP_WEIGHT", "0.20")),
    "reasoning_skill_group": float(os.environ.get("REASONING_SKILL_GROUP_WEIGHT", "0.14")),
    "knowledge_skill_group": float(os.environ.get("KNOWLEDGE_SKILL_GROUP_WEIGHT", "0.10")),
    # 2026-05-02 (v30.5): super_teacher axis REMOVED.
    # Rationale: the axis was conceptually wrong for distillation — by
    # construction, a student matching the teacher's distribution
    # (which is what KL/RKL/top_k_overlap reward) cannot exceed the
    # teacher on the same skill surface. Empirically every trained
    # miner sat at exactly 0.00 on this axis since launch, which
    # (a) ruined the radar-chart visualisation by anchoring one spoke
    # at zero, and (b) wasted 10% composite weight that should be
    # rewarding actual capability.
    # The 0.10 weight is redistributed: math +0.02, reasoning +0.02,
    # knowledge +0.03, leaving room for the bigger student cap (40B)
    # to express more capability on the existing axes.
    # ``_axis_super_teacher`` is kept as a private helper for legacy
    # composite records that still reference it, but the weight is 0
    # so it never contributes to the live composite ranking.
    "super_teacher":         float(os.environ.get("SUPER_TEACHER_WEIGHT", "0.0")),
}

BENCH_AXES_IN_COMPOSITE = os.environ.get("BENCH_AXES_IN_COMPOSITE", "1") != "0"

# ── 2026-04-24 — Arena v3 Session 3 (PRODUCTION) ─────────────────────
# Four capability-extending axes inspired by Affine Cortex's environment
# suite. Each scores absolute correctness against a public gold source;
# the ordering gain is in **coverage** — a model that overfits AIME
# still had to actually learn olympiad math, a model that overfits
# tool_use_bench still had to actually learn when to write Python.
# Tuned weights are conservative (3-6%) so Session 2 + the relative
# axes remain important while hard capability coverage is binding.
ARENA_V3_AXIS_WEIGHTS = {
    # 2026-04-26 (v28) — Quality > Quantity (Directive 2).
    # We dropped six axes that were either redundant or eval-setup-
    # fragile, and redirected their composite weight to harder, more
    # discriminating capability axes. Cuts:
    #   * ``self_consistency_bench`` (was 0.04): same item pool as
    #     math_bench, just sampled K-way and majority-voted. Miners who
    #     beat math_bench beat this; miners who fail math_bench
    #     occasionally pick up free credit when their k=4 sampler hits a
    #     correct answer by chance. No marginal signal.
    #   * ``arc_bench`` (was 0.04): commonsense MC. Reference 4B base
    #     scored 0.50 by random-pick, ceiling for the king is around
    #     0.75 — small dynamic range, and the signal it carries is
    #     dominated by knowledge_bench + reasoning_bench. Conceptually
    #     duplicative.
    #   * ``truthful_bench`` (was 0.03): adversarial trivia, narrow
    #     surface (~50 question categories). Top miner saturated with
    #     refusal-trained heuristics, not real epistemic discipline.
    #   * ``procedural_bench`` (was 0.05): now covered by the post-v27
    #     procedural rewrite of math_bench / capability / reasoning.
    #     Removing avoids triple-counting the same procedural-arithmetic
    #     signal across three weighted axes.
    #   * ``noise_resistance_bench`` (was 0.04): sibling of robustness
    #     that perturbs surface noise (typos / case). After v23/v24
    #     code-paraphrase + BBH-shuffle + math paraphrase landed, the
    #     same signal is captured by robustness_bench at half the
    #     wall-time cost.
    # Net cut: 0.20 weight + ~24 items per round (~9 min wall-time).
    # Redirected:
    #   aime_bench    +0.04  (0.06 → 0.10) — olympiad math, hard, ~zero
    #                                        memorisation surface post-v21.
    #   mbpp_bench    +0.02  (0.06 → 0.08) — programming breadth, complement
    #                                        to code_bench.
    #   tool_use_bench +0.02 (0.04 → 0.06) — agentic Python, no proxy axis.
    #   long_context  +0.01  (0.03 → 0.04) — procedural needle-in-haystack,
    #                                        uniquely tests retrieval.
    #   robustness    +0.03  (0.04 → 0.07) — absorbs the cut noise axis;
    #                                        validator now runs paraphrase +
    #                                        noise perturbations under one
    #                                        umbrella.
    # v30.2 — Most bench sub-axes migrated to the skill groups (see
    # BENCH_GROUP_AXIS_WEIGHTS above). Sub-axes still compute and feed
    # the groups; their direct composite weight is 0 by default. Three
    # axes stay separate because they measure orthogonal capabilities
    # the groups don't cover:
    #   * tool_use_bench — agentic Python (distinct from write/debug code)
    #   * calibration_bench — honest refusal under unsolvable items
    #   * ifeval_bench — instruction following with structural constraints
    #     (in BENCH_AXIS_WEIGHTS above, not here)
    "aime_bench":              float(os.environ.get("BENCH_AIME_WEIGHT", "0.0")),  # in math_skill_group
    "mbpp_bench":              float(os.environ.get("BENCH_MBPP_WEIGHT", "0.0")),  # in code_skill_group
    "tool_use_bench":           float(os.environ.get("BENCH_TOOL_USE_WEIGHT", "0.06")),
    "self_consistency_bench":   float(os.environ.get("BENCH_SC_WEIGHT", "0.0")),
    "arc_bench":                float(os.environ.get("BENCH_ARC_WEIGHT", "0.0")),
    "truthful_bench":           float(os.environ.get("BENCH_TRUTHFUL_WEIGHT", "0.0")),
    "long_context_bench":       float(os.environ.get("BENCH_LC_WEIGHT", "0.0")),  # in reasoning_skill_group
    "procedural_bench":         float(os.environ.get("BENCH_PROCEDURAL_WEIGHT", "0.0")),
    "robustness_bench":         float(os.environ.get("BENCH_ROBUSTNESS_WEIGHT", "0.0")),  # in math_skill_group
    "noise_resistance_bench":   float(os.environ.get("BENCH_NOISE_WEIGHT", "0.0")),
    "debug_bench":              float(os.environ.get("BENCH_DEBUG_WEIGHT", "0.0")),  # in code_skill_group
    "correction_bench":          float(os.environ.get("BENCH_CORRECTION_WEIGHT", "0.0")),  # in code_skill_group
    "multi_doc_synthesis_bench": float(os.environ.get("BENCH_MULTI_DOC_WEIGHT", "0.0")),  # in reasoning_skill_group
    "calibration_bench":         float(os.environ.get("BENCH_CALIBRATION_WEIGHT", "0.06")),
    "refactor_bench":            float(os.environ.get("BENCH_REFACTOR_WEIGHT", "0.0")),  # in code_skill_group
    "pragmatic_bench":           float(os.environ.get("BENCH_PRAGMATIC_WEIGHT", "0.0")),  # in knowledge_skill_group
}

ARENA_V3_AXES_IN_COMPOSITE = os.environ.get("ARENA_V3_AXES_IN_COMPOSITE", "1") != "0"

# ── 2026-04-25 — Session 3.2 reasoning_density axis (PRODUCTION) ─────
# User-reported pathology: "models are too distilled and think for too
# long about simple questions." The existing ``length`` axis addresses
# chat-probe length only. Bench probes give us per-axis mean_gen_tokens
# (see ``_bench_finalize_token_stats`` in pod_eval_vllm.py), so we can
# now score bench-level efficiency: pass_frac × length_bonus per bench,
# averaged across whichever benches emitted valid data.
#
# Target token counts are calibrated to the teacher's typical bench
# output lengths (empirical, April 2026). When a student gets the
# answer right in ≤ target tokens → length_bonus = 1.0. When they use
# 2× target → bonus ≈ 0.5. 4× target → bonus ≈ 0.25. This directly
# penalizes both the "over-think simple questions" failure mode and
# the "memorize answer-only training data" failure mode (a model that
# outputs "42" with 5 tokens still needs to get the answer right on a
# rotating pool; if it does, fine — the axis is neutral).
#
# Live with low weight so it is an auxiliary signal, not a dominant one.
REASONING_DENSITY_TARGET_TOKENS = {
    "math_bench":            float(os.environ.get("RD_MATH_TARGET", "400")),
    "code_bench":            float(os.environ.get("RD_CODE_TARGET", "300")),
    "reasoning_bench":       float(os.environ.get("RD_REASONING_TARGET", "150")),
    "knowledge_bench":       float(os.environ.get("RD_KNOWLEDGE_TARGET", "30")),
    "ifeval_bench":          float(os.environ.get("RD_IFEVAL_TARGET", "250")),
    "aime_bench":            float(os.environ.get("RD_AIME_TARGET", "800")),
    "mbpp_bench":            float(os.environ.get("RD_MBPP_TARGET", "250")),
    "tool_use_bench":        float(os.environ.get("RD_TOOL_USE_TARGET", "300")),
    "self_consistency_bench": float(os.environ.get("RD_SC_TARGET", "300")),
    "arc_bench":             float(os.environ.get("RD_ARC_TARGET", "50")),
    "truthful_bench":        float(os.environ.get("RD_TRUTHFUL_TARGET", "40")),
    "long_context_bench":    float(os.environ.get("RD_LC_TARGET", "30")),
    "procedural_bench":      float(os.environ.get("RD_PROCEDURAL_TARGET", "50")),
    # Session 3.7 — robustness reuses math items so target ≈ math but
    # tighter (the perturbation prefixes inflate input slightly; a
    # 380-token cap keeps the comparison fair across wrappers).
    "robustness_bench":      float(os.environ.get("RD_ROBUSTNESS_TARGET", "400")),
    # Session 3.7 — noise_resistance reuses math items as well; same
    # target token budget so reasoning-density comparisons across the
    # math/robustness/noise triple are apples-to-apples.
    "noise_resistance_bench": float(os.environ.get("RD_NOISE_TARGET", "400")),
    # v29.2 — debug_bench: corrected functions are typically 6-15 lines
    # (~200-400 tokens). 350 catches the median; longer corrections
    # still get partial reasoning-density credit via the soft penalty.
    "debug_bench":           float(os.environ.get("RD_DEBUG_TARGET", "350")),
    # v29.4 — calibrated against the typical answer length of each axis.
    "correction_bench":      float(os.environ.get("RD_CORRECTION_TARGET", "350")),
    "multi_doc_synthesis_bench": float(os.environ.get("RD_MULTI_DOC_TARGET", "60")),
    "calibration_bench":     float(os.environ.get("RD_CALIBRATION_TARGET", "60")),
    "refactor_bench":        float(os.environ.get("RD_REFACTOR_TARGET", "300")),
    # v30 — pragmatic_bench. Items request short answers ("Reply with
    # the container name only" / "Reply with 'yes' or 'no'"). 60-token
    # target catches the median; longer wraps still get partial credit.
    "pragmatic_bench":       float(os.environ.get("RD_PRAGMATIC_TARGET", "60")),
}
REASONING_DENSITY_WEIGHT = float(os.environ.get("REASONING_DENSITY_WEIGHT", "0.05"))
REASONING_DENSITY_IN_COMPOSITE = (
    os.environ.get("REASONING_DENSITY_IN_COMPOSITE", "1") != "0"
)

# ── 2026-04-25 — Session 3.3 chat_turns_probe axis (PRODUCTION) ──────
# Multi-turn coherence probe. Teacher grades a 3-turn transcript on a
# 1-5 rubric (coherence + consistency + helpfulness). Normalized to
# [0, 1]; identical shape to judge_probe so axis values are directly
# comparable in telemetry. A model that aces single-turn KL but can't
# hold context gets flagged by this axis — directly addressing the
# user-reported "models are too distilled, forget context" pathology.
#
# Live: single-turn KL specialists should not dethrone if they cannot
# maintain coherence over a short dialogue.
CHAT_TURNS_AXIS_WEIGHT = float(os.environ.get("CHAT_TURNS_AXIS_WEIGHT", "0.08"))
CHAT_TURNS_AXIS_IN_COMPOSITE = (
    os.environ.get("CHAT_TURNS_AXIS_IN_COMPOSITE", "1") != "0"
)
CHAT_TURNS_MIN_VALID = int(os.environ.get("CHAT_TURNS_MIN_VALID", "2"))
# Judge-probe min-valid threshold. Default 4 lets it work with reduced
# budgets (we currently run JUDGE_PROBE_PER_ROUND=6 for speed) without
# silently dropping the axis. Bug-discovered 2026-04-26: the previous
# hardcoded 8 was higher than the configured budget, so judge_probe
# was always None in production.
JUDGE_PROBE_MIN_VALID = int(os.environ.get("JUDGE_PROBE_MIN_VALID", "4"))

# v30.2 (2026-04-29) — composite.final ranking key.
#
# Replaces the legacy ``worst`` (single-axis min) as the canonical
# dethrone gate with a blend:
#
#     final = α · worst_3_mean + (1 − α) · weighted
#
# where ``worst_3_mean`` is the mean of the 3 lowest non-broken axis
# values (with weight > 0) and ``weighted`` is the existing weighted
# convex combination.
#
# Why blend? The audit at reports/2026-04-29-v30-strategic-audit.md
# §5 showed that ``min(axes)`` is dominated by NOISE on the
# lowest-data axis (35/160 UIDs sat at exactly worst=0 — a 22%
# saturated-floor cluster that worst() can't discriminate). Mean of
# the bottom-3 smooths variance while preserving the anti-Goodhart
# pressure (a model that tanks one axis still gets ~33% pull from the
# tanked axis). Blending with weighted retains all-axis information
# so models that excel broadly are not penalised by a single quirky
# subaxis floor.
#
# Default α = 0.7 (heavy emphasis on the bottom 3, in line with the
# audit's "high (~75%)" recommendation). Tunable via env.
COMPOSITE_FINAL_BOTTOM_WEIGHT = float(
    os.environ.get("COMPOSITE_FINAL_BOTTOM_WEIGHT", "0.7")
)
WORST_3_MEAN_K = int(os.environ.get("WORST_3_MEAN_K", "3"))

# v30 — long-form judge axis. Default per-round budget is 4 prompts so
# the floor is set at 2 (half the budget) — matches the JUDGE_PROBE
# floor convention. Operators bumping ``LONG_FORM_JUDGE_PER_ROUND`` on
# the eval side can raise this floor accordingly.
LONG_FORM_JUDGE_MIN_VALID = int(os.environ.get("LONG_FORM_JUDGE_MIN_VALID", "2"))
# 2026-05-01 (v30.4 patch v3): long-form weights raised to make
# long-generation coherence dominant. The chat.arbos.life screenshots
# show kings still producing pure multilingual word salad past 800
# tokens; the only way that's compatible with composite ≥0.55 is if
# long-form axes are too small a slice of the composite. Bumping
# both the rubric-graded ``long_form_judge`` AND the pure-statistical
# ``long_gen_coherence`` so:
#   • combined long-form weight = 0.25 (was 0.09)
#   • a derailed king (coh ~0.05) loses ~0.20 on weighted
#   • coh = 0.05 lands as a worst-3 axis pulling worst_3_mean toward 0
#   • on composite.final = 0.7·worst_3_mean + 0.3·weighted, that's
#     ~0.15 hit on final — ample to flip the dethrone gate
# The pure-statistical axis can't be cheated by rubric leniency.
LONG_FORM_JUDGE_AXIS_WEIGHT = float(
    os.environ.get("LONG_FORM_JUDGE_WEIGHT", "0.10")
)
LONG_GEN_COHERENCE_AXIS_WEIGHT = float(
    os.environ.get("LONG_GEN_COHERENCE_WEIGHT", "0.15")
)
LONG_GEN_COHERENCE_AXIS_IN_COMPOSITE = bool(
    int(os.environ.get("LONG_GEN_COHERENCE_IN_COMPOSITE", "1") or 1)
)
LONG_FORM_JUDGE_AXIS_IN_COMPOSITE = (
    os.environ.get("LONG_FORM_JUDGE_IN_COMPOSITE", "1") != "0"
)
# 2026-05-01 (v30.4 patch v3): hard-DQ floor on long-form coherence.
# When >LONG_FORM_DERAIL_DQ_RATIO of the round's responses score below
# LONG_FORM_DERAIL_DQ_THRESHOLD coherence, the model is permanently
# DQ'd at this commit-block. Soft-weight degradation alone wasn't
# enough — we kept seeing kings retain at composite ≥0.5 because
# their bench scores compensated for the coherence hit. Hard DQ is
# justified because long-form word salad is not a partial failure;
# the model cannot sustain coherent generation, which is a core
# deployment capability. Re-eval requires a fresh hotkey.
LONG_FORM_DERAIL_DQ_RATIO = float(
    os.environ.get("LONG_FORM_DERAIL_DQ_RATIO", "0.5")
)
LONG_FORM_DERAIL_DQ_THRESHOLD = float(
    os.environ.get("LONG_FORM_DERAIL_DQ_THRESHOLD", "0.30")
)
LONG_FORM_DERAIL_DQ_ENABLED = bool(
    int(os.environ.get("LONG_FORM_DERAIL_DQ_ENABLED", "1") or 1)
)

# Per-axis minimum valid-item count below which the axis drops as
# "insufficient sample". Small pools (code_bench samples only 4 items
# per round by design) get a lower floor so rounding noise doesn't
# exclude them unnecessarily.
BENCH_MIN_VALID = {
    "math_bench": 4,
    "code_bench": 2,
    "reasoning_bench": 4,
    "knowledge_bench": 4,
    "ifeval_bench": 4,
    # Session 3 axes — now live, so require enough items to dampen lucky
    # pass_frac spikes while still failing open on probe outages.
    "aime_bench": 3,
    "mbpp_bench": 3,
    "tool_use_bench": 3,
    "self_consistency_bench": 3,
    # Session 3.1 — ARC larger budget (6 per round), keep floor at 4
    # so one parse failure doesn't drop the axis.
    "arc_bench": 4,
    # Session 3.4 — TruthfulQA 4 per round, tight floor at 2.
    "truthful_bench": 3,
    # Session 3.5 — long-context 3 per round, tight floor at 2 since
    # each item is expensive (~1400 input tokens).
    "long_context_bench": 2,
    "procedural_bench": 4,
    # Session 3.7 — robustness draws K_perturb generations per item, so
    # a 4-item budget yields 8+ generations: hold the min_valid floor
    # at K_perturb so a single item drop doesn't kill the axis.
    "robustness_bench": 2,
    # Session 3.7 — noise_resistance has the same shape as robustness
    # (K perturbations × N items); use the same floor so a single
    # tokenization or grader edge case can't drop the axis.
    "noise_resistance_bench": 2,
    # v29.2 — debug_bench. 6 items per round; floor at 3 so a
    # single sandbox glitch doesn't drop the axis but most parse
    # failures do.
    "debug_bench": 3,
    # v29.4 — same conservative floors as the other coding-style axes
    # so a sandbox outage on a few items doesn't drop the axis entirely.
    "correction_bench": 3,
    "multi_doc_synthesis_bench": 3,
    "calibration_bench": 4,
    "refactor_bench": 2,
    # v30 — pragmatic_bench. 8 items per round; floor at 4 so a few
    # parse failures don't drop the axis but a sandbox outage does.
    "pragmatic_bench": 4,
}

# Schema version of the composite ranker. Bumped any time the composite
# computation, axis weights, or per-axis grading semantics change in a
# way that would let an old record keep an inflated floor under the new
# grading. The king-selection filter (``_KING_SELECTION_MIN_VERSION``)
# quarantines records below this version so the dethrone gate never
# compares stale-grading records to honest current-grading records.
#
# Detailed per-version changelog (Session 3.10 → 3.21, v17 → v30.2):
# see ``reports/2026-04-*-*.md``. The high-level timeline:
#   * v17  — rotate on_policy_rkl rollout seed per block.
#   * v18  — MBPP/HumanEval prose-stripping via ast.parse.
#   * v19  — capability_probe procedural rebalance.
#   * v20  — per-round MC option shuffle (arc/knowledge/truthful).
#   * v21–22 — math/aime/tool_use/self_consistency paraphrase.
#   * v23–24 — code/MBPP and BBH paraphrase + option shuffle.
#   * v25–26 — judge/chat_turns/on_policy_rkl chat paraphrase.
#   * v27   — full procedural switch for every benchmark axis.
#   * v28   — quality > quantity rebalance (mute 6 weak axes).
#   * v30.2 — composite.final = α·worst_3_mean + (1−α)·weighted is the
#             canonical ranking key; skill-group axes (code/math/
#             reasoning/knowledge) added; super_teacher axis added;
#             king re-eval per round; legacy ``worst`` retained as
#             telemetry but no longer the dethrone gate.
COMPOSITE_SHADOW_VERSION = 29

# ── Pareto majority dominance (Session 3 shadow) ──────────────────────
# An extra dethrone consideration: a challenger must beat the king on a
# majority of scorable axes, not just the single worst axis. Inspired
# by Affine Cortex's environment-level Pareto dominance. Starts as an
# informational score logged + surfaced in telemetry; promotion to
# dethrone gate flips via ``PARETO_DOMINANCE_GATE=1`` after the 48h
# public notice.
PARETO_DOMINANCE_MARGIN = float(os.environ.get("PARETO_DOMINANCE_MARGIN", "0.02"))
PARETO_DOMINANCE_MIN_COMPARABLE = int(os.environ.get("PARETO_DOMINANCE_MIN_COMPARABLE", "5"))
PARETO_DOMINANCE_GATE = os.environ.get("PARETO_DOMINANCE_GATE", "1") != "0"

# ── King regression health (2026-04-24, SHADOW) ──────────────────────────
# leeroyjkin (distil-97, 2026-04-24): "Why is the king safe when it scores
# poorly on your axis test, in fact worse than the base model?" The
# composite-as-veto gate blocks a challenger from *taking* the crown, but
# nothing forces the king to *defend* its composite. A king can camp the
# crown while its bench axes regress to the base-model floor because only
# KL is used for re-promotion.
#
# Minimal fix (shadow-first): compute a ``king_health`` summary each round
# and stamp it on the king's composite row. Two flags:
#   * ``below_floor``     — king's worst axis < KING_COMPOSITE_FLOOR
#   * ``worse_than_base`` — king's worst axis < base model's worst axis
# Consecutive at-risk rounds accumulate in ``state.king_regression_streak``
# (per king_uid). Dashboard + /api/miner/{uid} surface the streak so
# miners and spectators can see it; dethronement remains KL-only until we
# have ≥1 week of telemetry validating the floor choice.
#
# When ``KING_REGRESSION_GATE=1`` and streak ≥ ``KING_REGRESSION_MIN_STREAK``,
# the king is force-dethroned in favor of the highest-composite challenger
# in the current round that also passed the structural gates.
KING_COMPOSITE_FLOOR = float(os.environ.get("KING_COMPOSITE_FLOOR", "0.20"))
KING_REGRESSION_MIN_STREAK = int(os.environ.get("KING_REGRESSION_MIN_STREAK", "3"))
KING_REGRESSION_GATE = os.environ.get("KING_REGRESSION_GATE", "1") != "0"

# ── Canary-regression auto-dethrone (2026-04-28) ──────────────────────────
# Sibling of KING_REGRESSION_GATE. The internal at-risk check uses the
# validator's own composite.worst, which is gameable (the whole point
# of the goodhart canary). This canary gate uses HELD-OUT evalscope
# benchmarks that are NEVER inside the validator: when the king's
# average held-out score across {gsm8k, humaneval, bbh, ifeval} drops
# more than ``KING_CANARY_MARGIN`` pp below the Qwen 4B base reference
# for ``KING_CANARY_MIN_STREAK`` consecutive canonical rounds, the
# composite-floor veto is waived (same mechanism as the at-risk gate).
# This means a challenger who would normally be blocked by composite-
# floor veto can dethrone a king whose held-out is regressing — the
# explicit answer to "did the composite eval produce a model that's
# actually better, or just better at the composite?".
KING_CANARY_MARGIN = float(os.environ.get("KING_CANARY_MARGIN", "0.05"))
KING_CANARY_MIN_STREAK = int(os.environ.get("KING_CANARY_MIN_STREAK", "2"))
KING_CANARY_GATE = os.environ.get("KING_CANARY_GATE", "1") != "0"
KING_CANARY_AXES = ("gsm8k", "humaneval", "bbh", "ifeval")
KING_CANARY_BASELINE_FILE = os.environ.get("KING_CANARY_BASELINE_FILE", "baseline_qwen35_4b.json")

# ── Per-axis baseline-relative penalty (2026-04-28, v29.1) ────────────
# The 2026-04-28 audit confirmed every king from 2026-04-17 → today
# regressed below Qwen3.5-4B base on the held-out canary (-7.4pp gsm8k,
# -10pp ifeval, -16pp bbh, -12pp humaneval typical). The pre-existing
# defenses — ``_baseline_floor_dethrone_veto`` (10pp absolute floor) and
# ``king_canary_streak`` (2-round held-out streak) — both fire AT
# crowning / streak time, not during scoring. So a model can climb
# composite.worst and stay there indefinitely while regressing on real
# capability versus the un-distilled control.
#
# The fix is to make the per-axis composite directly reflect "stay
# above Qwen-4B-base". For each enabled bench axis, we compare each
# student's pass_frac to the *same-round* reference (REFERENCE_UID = -1,
# Qwen3.5-4B) score on the SAME block-seeded items. If the student is
# below the reference, the axis value gets docked by
# ``alpha * (ref - student)`` clipped to 0, where ``alpha`` is the
# regression weight.
#
# Why this works:
#   * Same-round paired comparison: both models see identical procedural
#     items (block_seed-deterministic), so the comparison is sample-
#     paired and free of cross-round prompt drift.
#   * Reward parity, punish regression: a student that BEATS base on
#     math gets full credit. A student that ties gets full credit. A
#     student that regresses is docked proportionally.
#   * No artificial ceiling: students who legitimately exceed base on
#     an axis are unaffected — overfitting in the *good* direction
#     (genuine skill > base) is encouraged.
#   * Compatible with worst-axis aggregation: the docked axis flows
#     into worst() so the dethrone gate naturally favors balanced-and-
#     above-base students over below-base specialists.
#
# Calibration:
#   * ``BASELINE_RELATIVE_PENALTY_ALPHA = 1.5`` — a 10pp regression below
#     base docks the axis by 15pp. This makes "stay at parity" the
#     dominant strategy: it costs the same as a 6.7pp axis-specific
#     improvement to drop 10pp on another axis. Aligned with the pre-
#     existing ``BASELINE_FLOOR_MARGIN = 0.10`` veto threshold.
#   * Penalty applied to bench axes only. Relative axes (kl, on_policy_rkl,
#     capability, length, degeneracy) are normalized differently and
#     would double-penalize. The judge_probe / chat_turns_probe axes
#     are absolute correctness but reference-model-flat (small dynamic
#     range), so we exclude them too.
#   * ``BASELINE_RELATIVE_PENALTY_AXES`` is the explicit allow-list of
#     axes that get the penalty.
BASELINE_RELATIVE_PENALTY_ENABLED = (
    os.environ.get("BASELINE_RELATIVE_PENALTY_ENABLED", "1") != "0"
)
BASELINE_RELATIVE_PENALTY_ALPHA = float(
    os.environ.get("BASELINE_RELATIVE_PENALTY_ALPHA", "1.5")
)
# Bench axes where regression below same-round reference docks the axis.
# All Session-2 + Session-3 bench axes that have a real ground truth and
# are scored by absolute pass_frac. Do NOT include relative axes.
BASELINE_RELATIVE_PENALTY_AXES = frozenset({
    "math_bench", "code_bench", "reasoning_bench", "ifeval_bench",
    "aime_bench", "mbpp_bench", "tool_use_bench",
    "long_context_bench", "robustness_bench",
    # v29.2 — debug_bench joins the baseline-relative penalty set: a
    # student that regresses on debugging vs Qwen-4B-base loses ranking
    # on this axis proportionally.
    "debug_bench",
    # v29.4 — four new SOTA-aligned axes, all subject to the
    # baseline-relative penalty so a model regressing below
    # Qwen-4B-base on any of them gets docked.
    "correction_bench", "multi_doc_synthesis_bench",
    "calibration_bench", "refactor_bench",
    # v30 — pragmatic_bench. A model regressing on theory-of-mind /
    # scalar implicature / indirect-request recognition vs the
    # un-distilled Qwen-4B base loses ranking accordingly.
    "pragmatic_bench",
})

# ── Teacher sanity gate (2026-04-23) ──────────────────────────────────────
# For each ranking axis we can optionally compute the axis value for the
# teacher itself (scored as if it were a student). If the teacher's axis
# value falls below this floor, the axis is miscalibrated for the round
# (probe miscoded, prompt pool corrupted, etc.) and must be dropped before
# it can corrupt rankings. This is the structural defense against the
# 2026-04-19 outage class (Wilson-anchor think-probe DQ'd the teacher
# itself, so every student failed). See
# ``reports/2026-04-23-goodhart-immune-eval.md`` section on invariants.
#
# Threshold reasoning: a well-calibrated axis should show the teacher at
# >= 0.85 comfortably (the teacher IS what we distill to, any axis where
# the teacher scores poorly is definitionally measuring the wrong thing).
# We pick 0.70 as the "drop the axis" floor to give some slack for
# stochasticity in the teacher's own generations (temperature=0 helps but
# vLLM sampling can still jitter), while still catching outright bugs.
TEACHER_SANITY_FLOOR = 0.70

# Reference-broken-axes filter (2026-04-26). The reference model is the
# baseline undistilled Qwen base; every round it runs through the same
# bench probes the students do. If the reference scores ``pass_frac == 0``
# on a bench axis, that axis is *not* measuring student skill — it is
# measuring an eval-setup bug (token truncation, malformed prompt,
# unsolvable item set, etc.). Audit 2026-04-26 of last_eval.json showed
# the reference scoring 0 on aime_bench (token truncation), code_bench,
# tool_use_bench, and noise_resistance_bench — locking ``worst() == 0``
# for all 36 current-schema records and making the dethrone gate
# degenerate. By dropping such axes from ``worst()`` we restore signal
# without giving miners a free pass: the axes still appear in the
# ``axes`` dict and contribute to ``weighted`` aggregation.
#
# We're more conservative than ``TEACHER_SANITY_FLOOR`` (0.70) here
# because the reference model is a *small* base model (Qwen3.5-4B) — it
# legitimately fails some hard items. Only the ``pass_frac == 0`` exact
# floor is treated as eval-broken; partial scores 0.25-0.50 are kept so
# students who outperform the reference are properly rewarded.
REFERENCE_BROKEN_BENCH_FLOOR = 0.0


def _king_ratio_axis(
    student: dict,
    student_field: str,
    king_ref: float | None,
) -> float | None:
    """Generic ``king_ref / student_value`` ratio axis, clamped to [0, 1].

    Used by every lower-is-better divergence axis (``kl``, ``kl_is``,
    ``forking_rkl``, ``tail_decoupled_kl``, ``teacher_trace_plausibility``,
    ``entropy_aware_kl``): a student matching the king scores 1.0;
    2× the king's value scores ~0.5; 10× scores ~0.1. Anchoring on the
    king keeps every axis scaled to real, achievable values rather than
    a hard-coded absolute floor.

    Returns ``None`` when:
      * student lacks the field (legacy record / dense-path eval / the
        teacher-as-student row, which has no KL vs itself);
      * king reference is None or non-positive;
      * the student value can't be coerced to a positive finite float.
    """
    val = student.get(student_field)
    if val is None or king_ref is None or king_ref <= 0:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")) or v <= 0:
        return None
    return max(0.0, min(1.0, float(king_ref) / v))


def _axis_kl(student: dict, king_kl: float | None) -> float | None:
    """KL axis — normalises ``kl_global_avg`` against the king's KL."""
    return _king_ratio_axis(student, "kl_global_avg", king_kl)


def _axis_kl_is(student: dict, king_kl_is: float | None) -> float | None:
    """v30 — Importance-sampled KL axis (SHADOW). Reads
    ``student.kl_is_mean`` (full-vocab KL contribution from top-K
    support). Per Anshumann et al. ACL 2025, this estimator is unbiased
    where the renormalised KL on shared support is biased by the
    teacher's top-K mass coverage."""
    return _king_ratio_axis(student, "kl_is_mean", king_kl_is)


def _axis_forking_rkl(student: dict,
                       king_forking_rkl: float | None) -> float | None:
    """v30 — Forking-token RKL axis (SHADOW). Reads
    ``student.forking_rkl_mean``: average reverse-KL at positions in the
    top quartile of teacher entropy. Per Wang et al. 2025, high-entropy
    positions are the "decision points" where the teacher's feedback is
    most informative and a stronger OPD predictor than the mean RKL."""
    return _king_ratio_axis(student, "forking_rkl_mean", king_forking_rkl)


def _axis_tail_decoupled_kl(student: dict,
                             king_kl_tail: float | None) -> float | None:
    """v30.3 — Tail-decoupled KL axis (SHADOW). Reads
    ``student.kl_tail_mean``: per-position mean KL contribution from
    the TAIL of the teacher's top-K cache (positions K_head+1 .. K).
    Catches the "match teacher's head but flatten the tail" pathology —
    a documented SFT-only over-confidence failure mode."""
    return _king_ratio_axis(student, "kl_tail_mean", king_kl_tail)


def _axis_teacher_trace_plausibility(student: dict,
                                      king_trace_nll: float | None) -> float | None:
    """v30 — Teacher-trace plausibility axis (SHADOW). Reads
    ``student.teacher_trace_nll_mean``: average NLL the student assigns
    to the teacher's actually-emitted tokens. Captures support coverage
    on the teacher's chosen path — distinct from FKL/RKL. A model with
    high FKL but low plausibility is a known LIMO/s1 SFT failure mode."""
    return _king_ratio_axis(
        student, "teacher_trace_nll_mean", king_trace_nll,
    )


def _axis_entropy_aware_kl(student: dict,
                            king_eopd: float | None) -> float | None:
    """v30 — Entropy-Aware adaptive KL axis (SHADOW). Reads
    ``student.eopd_adaptive_mean``, the per-prompt mean of the per-token
    quantity ``α(H_t)·RKL + (1−α(H_t))·FKL`` where ``α`` is high when the
    teacher is confident. Per arXiv 2510.27485, strictly more
    discriminating than vanilla per-token KL on small-model math benches
    (+1.37 to +5.05 Pass@8 on Qwen3-{0.6B,1.7B,4B})."""
    return _king_ratio_axis(student, "eopd_adaptive_mean", king_eopd)


def _axis_top_k_overlap(student: dict) -> float | None:
    """Top-K token overlap between teacher and student distributions.

    Definition: at each generated position, count how many of the
    teacher's top-K predicted tokens also appear in the student's top-K
    predicted tokens. Average this fraction across positions, then
    across prompts. Returns the model-level mean, already in [0, 1],
    higher is better.

    Why this axis matters (v30, 2026-04-29):

    Per the 2026 'Rethinking On-Policy Distillation' paper (arXiv
    2604.13016), top-K agreement between teacher and student
    distributions is the single most predictive signal of downstream
    OPD success — more predictive than raw KL or capability pass-rate.
    Successful OPD runs converge to 97-99% top-K overlap with the
    teacher, while failed runs (mode collapse, teacher hacking) sit
    at 60-80% even when KL looks acceptable.

    Dynamic range:
      * Random student: ~K/vocab_size ≈ 128/248320 ≈ 0.05% per slot,
        with 128 slots ≈ 7% expected overlap by chance alone.
      * Decent distillation student: 70-90%.
      * SOTA distilled student: 97-99%.

    So the axis has natural dynamic range from ~7% (random) to ~99%
    (SOTA), which maps cleanly onto the [0, 1] composite scale without
    further normalization.

    Goodhart resistance:
      * Cannot be gamed by per-token probability calibration (only the
        SET of top-K matters, not the probabilities on those K tokens).
      * Cannot be gamed by prompt-pool memorisation (top-K depends on
        the entire generated trajectory; off-policy memorisation
        doesn't help on-policy continuations).
      * Cannot be gamed by length collapse (positions are evaluated
        per-token; a 1-token early stop would only sample 1 position).

    Edge cases:
      * Returns ``None`` if no prompt had a valid overlap (e.g. all
        cache entries were sparse-empty / continuation length 0).
      * Returns ``None`` for rows that lack the field (teacher row,
        legacy records).

    The K used in production is 128 (matches the vLLM
    ``--max-logprobs 128`` cap that produces the teacher's sparse
    cache). Override via ``TOP_K_OVERLAP_K`` in the eval script for
    research replays.
    """
    val = student.get("top_k_overlap_mean")
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf guard
        return None
    return max(0.0, min(1.0, v))


def _axis_capability(student: dict) -> float:
    """Verifiable-rewards pass fraction with absolute-correctness floor.

    Before 2026-04-23 this axis was purely ``frac / teacher_frac``, which
    hit 1.0 whenever a student matched the teacher — *including on the
    teacher's wrong answers*. Empirically this was mild because the
    teacher got ~85-90% on the capability pool, but in principle it
    rewards teacher-hacking: a student that learns to echo teacher's
    mistakes scores identically to one that actually learned to answer.
    The 2026-04-23 Goodhart-immune eval design (see
    ``reports/2026-04-23-goodhart-immune-eval.md``) adds an absolute
    term so matching the teacher at a low absolute accuracy is not
    credited as full marks.

    Shape:
      score = (absolute_accuracy + min(frac / max(teacher, 0.5), 1.0)) / 2

    Properties:
      * Monotonic in both absolute (frac) and relative (frac / teacher)
        correctness.
      * Student at 100%, teacher at 100% → 1.0 (unchanged).
      * Student and teacher both at 30% → 0.65 (was 1.0).
      * Student at 50%, teacher at 100% → 0.50 (was 0.50; unchanged).
      * Student at 80%, teacher at 80% → 0.90 (was 1.0).
      * Student at 100%, teacher at 60% → (1.0 + 1.0) / 2 = 1.0.
      * Floor of 0.5 in the relative denominator prevents a flaky
        round (teacher errored on many items) from saturating the
        axis.

    Returns ``None`` if the probe didn't run.
    """
    cap = student.get("capability") or {}
    frac = cap.get("pass_frac")
    if frac is None:
        return None
    absolute = max(0.0, min(1.0, float(frac)))
    teach = cap.get("teacher_pass_frac")
    if teach and teach > 0:
        relative = max(0.0, min(1.0, float(frac) / max(float(teach), 0.5)))
    else:
        relative = absolute
    return max(0.0, min(1.0, 0.5 * absolute + 0.5 * relative))


def _axis_length(student: dict) -> float:
    """Length penalty as stored by the eval script. Already in [0, 1]."""
    la = student.get("length_axis") or {}
    pen = la.get("penalty")
    return None if pen is None else max(0.0, min(1.0, pen))


def _axis_degeneracy(student: dict) -> float:
    """Think-probe terminates+non-degenerate+self-bleu as a single score.

    Pass on all three components => 1.0. Partial pass linearly interpolated.
    The probe uses MAD-z against teacher statistics when available, so this
    axis is already threshold-free.
    """
    tp = student.get("think_probe") or {}
    if not tp:
        return None
    tested = tp.get("prompts_tested") or 0
    if tested == 0:
        return None
    term = tp.get("prompts_terminated") or 0
    degen = tp.get("prompts_degenerate") or 0
    sb = tp.get("self_bleu_across_prompts") or 0.0
    teach_sb = tp.get("teacher_self_bleu") or 0.0
    term_score = term / tested
    degen_score = max(0.0, 1.0 - degen / tested)
    sb_margin = max(0.0, 0.9 - max(sb - teach_sb, 0.0))
    sb_score = min(1.0, sb_margin / 0.9)
    return 0.4 * term_score + 0.4 * degen_score + 0.2 * sb_score


def _judge_probe_axis(
    student: dict,
    probe_key: str,
    score_field: str,
    min_valid: int,
) -> float | None:
    """Generic ``probe.<score_field>`` extractor with ``n_valid`` floor.

    Used by every teacher-rubric axis (``judge_probe``,
    ``long_form_judge``, ``long_gen_coherence``, ``chat_turns_probe``,
    and skill-group-specific variants): the eval payload always carries
    ``{ <score_field>: float in [0,1], n_valid: int, ... }``; we drop
    the axis when fewer than ``min_valid`` prompts parsed cleanly so a
    rubric/teacher drift signal doesn't silently bleed into composite
    rankings.

    Returns ``None`` when the probe didn't run, the score field is
    absent, or ``n_valid`` is below the floor. Otherwise clamps the
    raw value into [0, 1].
    """
    payload = student.get(probe_key) or {}
    if not payload:
        return None
    raw = payload.get(score_field)
    if raw is None:
        return None
    if (payload.get("n_valid") or 0) < min_valid:
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _axis_judge_probe(student: dict) -> float | None:
    """Teacher-as-judge normalized score in [0, 1] (2026-04-23 shadow axis).

    Teacher scores ``JUDGE_PROBE_PER_ROUND`` rotated prompts on a 1-5
    rubric; valid scores averaged and mapped via ``(mean - 1) / 4``.
    Drops below ``JUDGE_PROBE_MIN_VALID`` parses (rubric/teacher drift
    signal — telemetry is more meaningful than a noisy score)."""
    return _judge_probe_axis(
        student, "judge_probe", "normalized", JUDGE_PROBE_MIN_VALID,
    )


def _axis_long_form_judge(student: dict) -> float | None:
    """v30 — long-form judge axis.

    Teacher rubric grades each student's 300-500 word essay-style
    response on STRUCTURE / DEPTH / COHERENCE / LENGTH. Per-prompt
    score is multiplied by the six-signal statistical coherence factor
    in ``long_form_judge_teacher_score`` BEFORE aggregation, so a
    derailed response cannot earn a high rubric grade even if the
    teacher was lenient (v30.4, 2026-05-01)."""
    return _judge_probe_axis(
        student, "long_form_judge_probe", "normalized",
        LONG_FORM_JUDGE_MIN_VALID,
    )


def _axis_long_gen_coherence(student: dict) -> float | None:
    """v30.4 — long-form-generation coherence axis (zero teacher involvement).

    Returns the mean per-prompt coherence factor from the long-form
    judge probe — exactly the multiplier already applied to the
    ``long_form_judge`` axis grade. Pure statistical signal (six
    surface signals on the 2048-token response): the composite
    directly penalises derail even if a miner gamed the teacher
    rubric to score 5/5 on a derailed response."""
    return _judge_probe_axis(
        student, "long_form_judge_probe", "coherence_factor",
        LONG_FORM_JUDGE_MIN_VALID,
    )


def _axis_chat_turns_probe(student: dict) -> float | None:
    """Multi-turn coherence axis (2026-04-25 Session 3.3, SHADOW).

    Teacher judges 6 rotated 3-turn transcripts on a 1-5 rubric
    (coherence / consistency / helpfulness). Forces miners to keep
    multi-turn coherence in the loss tent — KL distillation only
    optimises single-turn climbmix-style prompts, so a model can ace
    KL yet fall apart on multi-turn dialogue."""
    return _judge_probe_axis(
        student, "chat_turns_probe", "normalized", CHAT_TURNS_MIN_VALID,
    )


def _axis_bench_pass_frac(student: dict, axis_name: str) -> float | None:
    """Generic [0, 1] pass-fraction extractor for Pareto holistic eval v2.

    Each ``*_bench`` key in the student result has the same schema:
    ``{"n": int, "correct": int, "pass_frac": float, "items": [...]}``.
    Returns the raw ``pass_frac`` if there are at least ``BENCH_MIN_VALID``
    items; otherwise ``None`` so the axis drops out for the round.
    Errored probes are also mapped to None (fail-open).
    """
    payload = student.get(axis_name) or {}
    if not payload or payload.get("error"):
        return None
    n = int(payload.get("n") or 0)
    if n < BENCH_MIN_VALID.get(axis_name, 4):
        return None
    frac = payload.get("pass_frac")
    if frac is None:
        return None
    try:
        return max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        return None


# Bench axes are uniform thin wrappers around _axis_bench_pass_frac.
# Each name is exposed as ``_axis_<name>`` for symmetry with hand-written
# axes and so external callers / tests can ``from composite import
# _axis_<name>``. The actual logic lives in _axis_bench_pass_frac.
_BENCH_AXIS_NAMES: tuple[str, ...] = (
    "math_bench", "code_bench", "reasoning_bench", "knowledge_bench",
    "ifeval_bench", "aime_bench", "mbpp_bench", "tool_use_bench",
    "self_consistency_bench", "arc_bench", "truthful_bench",
    "long_context_bench", "procedural_bench", "robustness_bench",
    "noise_resistance_bench", "debug_bench", "correction_bench",
    "multi_doc_synthesis_bench", "calibration_bench", "refactor_bench",
    "pragmatic_bench",
)


def _make_bench_axis(name: str):
    def _axis(student: dict) -> float | None:
        return _axis_bench_pass_frac(student, name)
    _axis.__name__ = f"_axis_{name}"
    _axis.__qualname__ = _axis.__name__
    _axis.__doc__ = f"Bench-axis wrapper for ``{name}`` (see _axis_bench_pass_frac)."
    return _axis


for _bench_name in _BENCH_AXIS_NAMES:
    globals()[f"_axis_{_bench_name}"] = _make_bench_axis(_bench_name)
del _bench_name


# v30.2 (2026-04-29) — Skill-group axes.
#
# Why this exists. The audit at reports/2026-04-29-v30-strategic-audit.md
# §3 found heavy axis sprawl: 5 axes measure code (code_bench,
# mbpp_bench, debug_bench, correction_bench, refactor_bench), 3 measure
# math (math_bench, aime_bench, robustness_bench), and several axes
# overlap on retrieval (long_context_bench, multi_doc_synthesis_bench)
# and knowledge (knowledge_bench v2, pragmatic_bench). Every axis still
# runs (no information loss) but only the GROUP score gates ranking —
# this reduces worst-3 noise without dropping any measurement.
#
# A skill-group axis is the equal-weighted MEAN of its sub-axes (only
# the sub-axes that returned a non-None pass_frac are averaged). When
# all sub-axes drop, the group axis drops too. Sub-axes remain in
# ``axes`` for dashboard / per_src telemetry / saturation audit.
#
# The composite weights below put the WEIGHT on the group axis and
# leave the sub-axes at 0 (they're computed but not in
# ``effective_weights``). To opt out of the grouping for a specific
# axis (e.g., re-promote ``code_bench`` to its own slot during a
# debug session), set its env weight > 0 explicitly.

CODE_SKILL_GROUP_SUB_AXES = (
    "code_bench",
    "mbpp_bench",
    "debug_bench",
    "correction_bench",
    "refactor_bench",
)

MATH_SKILL_GROUP_SUB_AXES = (
    "math_bench",
    "aime_bench",
    "robustness_bench",
)

REASONING_SKILL_GROUP_SUB_AXES = (
    "reasoning_bench",
    "multi_doc_synthesis_bench",
    "long_context_bench",
)

KNOWLEDGE_SKILL_GROUP_SUB_AXES = (
    "knowledge_bench",
    "pragmatic_bench",
)


def _axis_skill_group_mean(
    student: dict,
    sub_axes: tuple[str, ...],
    broken_axes: set[str] | None = None,
) -> float | None:
    """Equal-weighted mean of present (non-None) bench sub-axis
    pass-fracs.

    v30.2 (2026-04-29): preserves the broken-axes invariant from the
    pre-grouping era. When a sub-axis is in ``broken_axes`` (the
    reference 4B scored 0 — eval-setup signal, not skill), it is
    excluded from the group computation. This means a student who
    just happens to score 0 on aime_bench (because the reference
    couldn't either) doesn't get docked in math_skill_group, while
    a student who SOLVED aime items the reference couldn't solve also
    doesn't get an inflated lift via the broken sub-axis. The group
    score reflects only the eval-valid sub-axes.

    When broken_axes is None (legacy callers), all non-None sub-axes
    are included — same as v30.2 release behaviour.

    Returns None when no eval-valid sub-axis has data — graceful
    drop so the composite renormalises over surviving axes.

    We use the RAW pass_frac (``_axis_bench_pass_frac``), not the
    baseline-relative-penalty-adjusted value, because the per-axis
    penalty already lives on each sub-axis when applicable. Stacking
    penalties on a group score would double-penalise.
    """
    vals: list[float] = []
    for ax in sub_axes:
        if broken_axes and ax in broken_axes:
            # Broken sub-axis: drop from group mean to preserve the
            # "broken axes don't penalize" invariant.
            continue
        v = _axis_bench_pass_frac(student, ax)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


# v30.2 — Skill-group registry. Each entry is
# ``(group_axis_name, sub_axes)``. The composite ``axes`` dict is
# populated for every group via a loop in compute_axes; consumers
# (composite weights, telemetry, broken-axis sanity) reference the
# group axis name directly. Per-group wrapper functions
# (``_axis_<group>_skill_group``) are defined right below for
# backwards-compat with the existing test suite — they all delegate
# to ``_axis_skill_group_mean``.
_SKILL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code_skill_group", CODE_SKILL_GROUP_SUB_AXES),
    ("math_skill_group", MATH_SKILL_GROUP_SUB_AXES),
    ("reasoning_skill_group", REASONING_SKILL_GROUP_SUB_AXES),
    ("knowledge_skill_group", KNOWLEDGE_SKILL_GROUP_SUB_AXES),
)


def _make_skill_group_axis(sub_axes: tuple[str, ...]):
    def _axis(student: dict, broken_axes: set[str] | None = None) -> float | None:
        return _axis_skill_group_mean(student, sub_axes, broken_axes)
    return _axis


_axis_code_skill_group = _make_skill_group_axis(CODE_SKILL_GROUP_SUB_AXES)
_axis_math_skill_group = _make_skill_group_axis(MATH_SKILL_GROUP_SUB_AXES)
_axis_reasoning_skill_group = _make_skill_group_axis(REASONING_SKILL_GROUP_SUB_AXES)
_axis_knowledge_skill_group = _make_skill_group_axis(KNOWLEDGE_SKILL_GROUP_SUB_AXES)


# v30.2 — Super-teacher axis: rewards exceeding the teacher on
# verifiable benches.
#
# Why this exists. Pure distillation cannot exceed teacher capability —
# a student that perfectly matches Qwen3.6-35B on every bench axis
# tops out at the teacher's pass rate. To produce SOTA-class small
# models we need miners to mix in (b) RL on verifiable rewards and
# (c) post-distillation SFT on harder data than the teacher saw. The
# super-teacher axis explicitly rewards beating the teacher on any
# verifiable axis, so a student that runs Stage-4 GRPO + curated-data
# SFT (per the Mining Guide v2) earns above-teacher pass rates and
# captures the bonus. See strategic audit §1 #4.
#
# Computation: for each verifiable bench axis the student and teacher
# both reported, the per-axis "lift" is max(0, student_frac −
# teacher_frac). The axis value is the mean of per-axis lifts mapped
# to [0, 1] via a soft tanh: small lifts (~0.05) score ~0.5, lifts
# ~0.20 score ~0.95.
#
# The teacher's pass_frac per axis is exposed via the teacher row in
# ``students_data`` (the same row resolve_teacher_broken_axes uses);
# the caller threads ``teacher_axes`` through compute_axes /
# compute_composite so this axis can read it.

SUPER_TEACHER_AXES = (
    # Verifiable benches where the teacher's pass_frac is meaningful.
    # Excludes pure teacher-similarity axes (kl, rkl, top_k_overlap)
    # and judge-rubric axes (judge_probe, long_form_judge,
    # chat_turns_probe) where "above teacher" isn't well-defined.
    "math_bench", "code_bench", "reasoning_bench", "ifeval_bench",
    "aime_bench", "mbpp_bench", "tool_use_bench", "long_context_bench",
    "robustness_bench",
    "debug_bench", "correction_bench", "multi_doc_synthesis_bench",
    "calibration_bench", "refactor_bench",
    "pragmatic_bench", "knowledge_bench",
)
SUPER_TEACHER_SOFT_SCALE = float(
    os.environ.get("SUPER_TEACHER_SOFT_SCALE", "0.10")
)


def _axis_super_teacher(
    student: dict,
    teacher_axes: dict[str, float | None] | None,
) -> float | None:
    """Reward student for exceeding the teacher on verifiable bench axes.

    Returns ``mean(per_axis_lift)`` mapped through a soft tanh to [0, 1]:
    a student that exactly matches the teacher on every axis scores 0;
    a student that beats the teacher by ~0.20 on average scores ~0.95.

    Returns ``None`` when:
      * teacher_axes is None / empty (we can't compute lift without
        the teacher's score).
      * The student reports no super-teacher-eligible axes.
    """
    if not teacher_axes:
        return None
    lifts: list[float] = []
    for ax in SUPER_TEACHER_AXES:
        s = _axis_bench_pass_frac(student, ax)
        t = teacher_axes.get(ax)
        if s is None or t is None:
            continue
        try:
            lift = max(0.0, float(s) - float(t))
        except (TypeError, ValueError):
            continue
        lifts.append(lift)
    if not lifts:
        return None
    mean_lift = sum(lifts) / len(lifts)
    # Soft tanh-style mapping: 0 lift → 0, scale lift → ~0.76, 2*scale → ~0.96.
    import math as _m
    score = _m.tanh(mean_lift / max(SUPER_TEACHER_SOFT_SCALE, 1e-6))
    return max(0.0, min(1.0, score))


def _axis_reasoning_density(student: dict) -> float | None:
    """Reasoning-density axis (Session 3.2, 2026-04-25).

    For each bench axis the student actually ran, compute
    ``efficiency = pass_frac * length_bonus`` where ``length_bonus`` is a
    soft penalty around the per-bench target token count:

        ratio   = mean_gen_tokens_correct / target
        bonus   = 1.0                             (ratio <= 1)
                  1 / (1 + (ratio-1))              (1 < ratio)

    So at ratio=1 we get 1.0, at ratio=2 we get 0.5, at ratio=4 we get
    0.25. No bonus below ratio=1 so a concise correct model gets the
    same credit as one that matches target exactly; this rewards
    efficiency without penalizing further.

    The axis value is the mean of per-bench efficiencies over whichever
    benches emitted ``mean_gen_tokens_correct`` > 0. Returns None if
    no bench reported correct tokens (e.g. a shadow round where every
    bench was skipped or the student got zero correct — then the axis
    has nothing to say and falls back to other capability axes).

    Rationale for absolute targets (not teacher-relative): the teacher
    currently doesn't run the full bench battery. Absolute targets
    drawn from observed teacher behavior (April 2026) avoid needing a
    teacher-bench pass and are easy to recalibrate as the teacher
    changes. See composite.py REASONING_DENSITY_TARGET_TOKENS.
    """
    per_bench_scores: list[float] = []
    for axis_name, target in REASONING_DENSITY_TARGET_TOKENS.items():
        payload = student.get(axis_name) or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("error"):
            continue
        n = int(payload.get("n") or 0)
        if n < BENCH_MIN_VALID.get(axis_name, 2):
            continue
        correct = int(payload.get("correct") or 0)
        if correct == 0:
            # Zero-correct benches contribute zero to the axis: a model
            # that fails every problem on a bench shouldn't sneak a
            # high reasoning_density score just by using few tokens.
            per_bench_scores.append(0.0)
            continue
        mean_tok = float(payload.get("mean_gen_tokens_correct") or 0.0)
        if mean_tok <= 0 or target <= 0:
            # Missing token data — skip this bench rather than
            # inventing a score. Old-code rounds will have no token
            # stats, which is the fail-open path we want.
            continue
        ratio = mean_tok / target
        bonus = 1.0 if ratio <= 1.0 else 1.0 / (1.0 + (ratio - 1.0))
        pass_frac = correct / n
        per_bench_scores.append(pass_frac * bonus)
    if not per_bench_scores:
        return None
    return max(0.0, min(1.0, sum(per_bench_scores) / len(per_bench_scores)))


def _axis_on_policy_rkl(student: dict, king_rkl: float | None) -> float:
    """Normalize on-policy reverse KL to [0, 1] higher-is-better.

    On-policy RKL is the primary distillation signal under the new
    framework: it is computed on the student's *own* rollouts, so the
    student cannot hide behind teacher-forced memorization. Lower RKL
    means the student's policy is closer to the teacher in the mode-
    seeking direction, which is the actual objective of distillation.

    We normalize against the king's RKL: a student tied with the king
    scores 1.0; RKL at 2× king → ~0.5; at 10× king → ~0.1. If RKL
    numbers are missing (old snapshot, probe disabled, probe errored),
    return None so the axis drops out and the weighted mean
    renormalizes over the remaining axes.
    """
    opr = student.get("on_policy_rkl") or {}
    if not opr:
        return None
    rkl = opr.get("mean_rkl")
    if rkl is None or rkl != rkl:  # NaN
        return None
    if king_rkl is None or king_rkl <= 0:
        # Fall back to an absolute-ish anchor: RKL under ~0.1 nats is
        # typical for well-distilled students on-policy in our smoke tests.
        if rkl <= 0:
            return 1.0
        return max(0.0, min(1.0, 0.1 / max(rkl, 1e-6)))
    if rkl <= 0:
        return 1.0
    return max(0.0, min(1.0, king_rkl / rkl))


def compute_axes(student: dict, king_kl: float | None = None,
                 king_rkl: float | None = None,
                 king_eopd: float | None = None,
                 king_kl_is: float | None = None,
                 king_forking_rkl: float | None = None,
                 king_trace_nll: float | None = None,
                 king_kl_tail: float | None = None,
                 teacher_axes: dict[str, float | None] | None = None,
                 broken_axes: set[str] | None = None) -> dict[str, float | None]:
    """Compute the raw per-axis values for one student dict.

    Pulled out of ``compute_composite`` so that the teacher sanity gate
    can score the teacher itself on the same axes (by passing the teacher
    row from ``results['students']``). Returns a dict keyed by axis name;
    values are floats in [0, 1] or None if the axis couldn't be computed.

    Session 3 axes (``aime_bench``, ``mbpp_bench``, ``tool_use_bench``,
    ``self_consistency_bench``) are always computed when the probe
    reported, but only included in ``worst`` / ``weighted`` aggregation
    by ``compute_composite`` when ``ARENA_V3_AXES_IN_COMPOSITE`` is
    truthy. Same shadow-then-promote pattern as Session 2.
    """
    out: dict[str, float | None] = {
        "on_policy_rkl": _axis_on_policy_rkl(student, king_rkl),
        "kl": _axis_kl(student, king_kl),
        "top_k_overlap": _axis_top_k_overlap(student),
        "entropy_aware_kl": _axis_entropy_aware_kl(student, king_eopd),
        "kl_is": _axis_kl_is(student, king_kl_is),
        "forking_rkl": _axis_forking_rkl(student, king_forking_rkl),
        "teacher_trace_plausibility": _axis_teacher_trace_plausibility(
            student, king_trace_nll
        ),
        # v30.3 — tail-decoupled KL: catches over-confident tail-flatteners.
        "tail_decoupled_kl": _axis_tail_decoupled_kl(student, king_kl_tail),
        "capability": _axis_capability(student),
        "length": _axis_length(student),
        "degeneracy": _axis_degeneracy(student),
        "judge_probe": _axis_judge_probe(student),
        "long_form_judge": _axis_long_form_judge(student),
        "long_gen_coherence": _axis_long_gen_coherence(student),
        "chat_turns_probe": _axis_chat_turns_probe(student),
    }
    for _bench in _BENCH_AXIS_NAMES:
        out[_bench] = _axis_bench_pass_frac(student, _bench)
    # v30.2 — skill-group axes (mean of non-broken sub-axes; sub-axes
    # still populated above for telemetry). Driven by ``_SKILL_GROUPS``
    # so adding a new group is a one-line registry edit.
    for group_name, sub_axes in _SKILL_GROUPS:
        out[group_name] = _axis_skill_group_mean(student, sub_axes, broken_axes)
    out.update({
        # v30.2 — incentivize exceeding the teacher on verifiable
        # benches. Reads the teacher's per-axis scores (None when
        # teacher_axes not threaded through).
        "super_teacher": _axis_super_teacher(student, teacher_axes),
        "reasoning_density": _axis_reasoning_density(student),
    })
    return out


def resolve_reference_broken_axes(reference_student_row: dict | None) -> set[str]:
    """Identify bench axes where the reference base model itself scored 0.

    The reference model (Qwen3.5-4B base, ``REFERENCE_UID = -1``) is the
    undistilled control we run every round. It is a small 4B base model,
    so we expect it to fail some hard items — that's *real* signal a
    distilled student can pick up. But if the reference scores
    ``pass_frac == 0`` on a bench axis, the axis is broken at the
    eval-setup level (token truncation, malformed prompt, unsolvable
    items): the *base* model can't even partially attempt it, so any
    student score on that axis is noise.

    Audit 2026-04-26 found:
      * aime_bench: reference 0/3 — 256-token cap truncates derivations
      * code_bench: reference 0/3 — also token-bound
      * tool_use_bench: reference 0/3 — 192-token cap + tool format
      * noise_resistance_bench: reference 0/6 — perturbation strength

    These four axes alone caused 100 % of current-schema records to sit
    at ``worst == 0``, breaking the dethrone gate. Dropping them from
    ``worst()`` (but keeping them in ``weighted`` and the per-axis
    dashboard) restores signal without giving miners a free pass.

    Returns the set of axis names to exclude. Empty set if the
    reference row is absent (round didn't include reference) or all
    reference scores are non-zero.
    """
    if not reference_student_row:
        return set()
    broken: set[str] = set()
    # Axes we consider eval-setup-fragile. Relative axes (kl, rkl,
    # capability, length, degeneracy) reference at 1.0 by definition,
    # so they're never in this set.
    bench_axes = {
        "aime_bench", "mbpp_bench", "code_bench", "math_bench",
        "knowledge_bench", "reasoning_bench", "tool_use_bench",
        "robustness_bench", "noise_resistance_bench", "ifeval_bench",
        "self_consistency_bench", "arc_bench", "truthful_bench",
        "long_context_bench", "procedural_bench",
        # v29.2 — debug_bench is also an absolute-pass-frac axis
        # eligible to be flagged broken if the reference scores 0/n
        # (e.g. sandbox outage).
        "debug_bench",
        # v29.4 — same broken-axis treatment for the new SOTA axes.
        "correction_bench", "multi_doc_synthesis_bench",
        "calibration_bench", "refactor_bench",
        # v30 — pragmatic_bench is procedural and self-contained so a
        # reference 0/N is a real signal of generator regression
        # rather than skill, justifying broken-axis treatment.
        "pragmatic_bench",
    }
    for axis in bench_axes:
        bench = reference_student_row.get(axis)
        if not isinstance(bench, dict):
            continue
        n = bench.get("n") or 0
        pass_frac = bench.get("pass_frac")
        # Only flag if the reference *attempted* the axis (n>0) and
        # scored exactly 0. ``n == 0`` means the axis didn't run at all,
        # which is a different failure mode.
        if n > 0 and pass_frac is not None and float(pass_frac) <= REFERENCE_BROKEN_BENCH_FLOOR:
            broken.add(axis)
    return broken


def get_effective_axis_weights() -> dict[str, float]:
    """Return the weights of all axes currently active in the composite.

    Single source of truth for "which axes gate ranking this round" — the
    same dict that ``compute_composite`` builds internally and that
    ``_composite_dethrone_veto`` / ``resolve_teacher_broken_axes`` /
    h2h backfill all need to filter axis dicts. Centralising here
    means a new shadow→production promotion only needs to flip the
    relevant ``*_IN_COMPOSITE`` env gate, with no risk of one caller
    forgetting an axis (e.g. results.py forgot ``long_form_judge`` and
    ``long_gen_coherence`` for months, so the worst-axis veto would
    cite the second-worst axis when those derail axes were active).

    Excludes zero-weight axes — a SHADOW axis with weight 0 is by
    definition not gating ranking even if its ``*_IN_COMPOSITE`` flag
    is set.
    """
    weights: dict[str, float] = {k: w for k, w in AXIS_WEIGHTS.items() if w > 0}
    if not TOP_K_OVERLAP_AXIS_IN_COMPOSITE:
        weights.pop("top_k_overlap", None)
    if JUDGE_AXIS_IN_COMPOSITE and JUDGE_AXIS_WEIGHT > 0:
        weights["judge_probe"] = JUDGE_AXIS_WEIGHT
    if BENCH_AXES_IN_COMPOSITE:
        weights.update(
            {k: w for k, w in BENCH_AXIS_WEIGHTS.items() if w > 0},
        )
    if ARENA_V3_AXES_IN_COMPOSITE:
        weights.update(
            {k: w for k, w in ARENA_V3_AXIS_WEIGHTS.items() if w > 0},
        )
    weights.update(
        {k: w for k, w in BENCH_GROUP_AXIS_WEIGHTS.items() if w > 0},
    )
    if REASONING_DENSITY_IN_COMPOSITE and REASONING_DENSITY_WEIGHT > 0:
        weights["reasoning_density"] = REASONING_DENSITY_WEIGHT
    if CHAT_TURNS_AXIS_IN_COMPOSITE and CHAT_TURNS_AXIS_WEIGHT > 0:
        weights["chat_turns_probe"] = CHAT_TURNS_AXIS_WEIGHT
    if LONG_FORM_JUDGE_AXIS_IN_COMPOSITE and LONG_FORM_JUDGE_AXIS_WEIGHT > 0:
        weights["long_form_judge"] = LONG_FORM_JUDGE_AXIS_WEIGHT
    if (
        LONG_GEN_COHERENCE_AXIS_IN_COMPOSITE
        and LONG_GEN_COHERENCE_AXIS_WEIGHT > 0
    ):
        weights["long_gen_coherence"] = LONG_GEN_COHERENCE_AXIS_WEIGHT
    return weights


def resolve_teacher_broken_axes(teacher_student_row: dict | None,
                                king_kl: float | None = None,
                                king_rkl: float | None = None) -> set[str]:
    """Identify axes where the teacher itself fails the sanity floor.

    If ``teacher_student_row`` is None (no teacher-as-student probe this
    round, e.g. the teacher-as-student pass wasn't added yet) returns an
    empty set — fail open. For any axis where the teacher scores a real
    value < ``TEACHER_SANITY_FLOOR`` we return that axis name so the
    caller can drop it from ranking. Axes where the teacher returns
    None are considered uncalibrated and also dropped defensively.

    The applicable set is broader than ``get_effective_axis_weights``:
    we also include zero-weight bench sub-axes when their gates are on
    (``BENCH_AXES_IN_COMPOSITE`` / ``ARENA_V3_AXES_IN_COMPOSITE``).
    Skill-group axes drop teacher-broken sub-axes from their mean even
    when the sub-axis itself doesn't directly gate ranking, so we must
    flag those broken sub-axes here.
    """
    broken: set[str] = set()
    if not teacher_student_row:
        return broken
    teacher_axes = compute_axes(teacher_student_row, king_kl, king_rkl)
    applicable = set(get_effective_axis_weights().keys())
    if BENCH_AXES_IN_COMPOSITE:
        applicable.update(BENCH_AXIS_WEIGHTS.keys())
    if ARENA_V3_AXES_IN_COMPOSITE:
        applicable.update(ARENA_V3_AXIS_WEIGHTS.keys())
    for axis, val in teacher_axes.items():
        if axis not in applicable:
            continue
        if val is None:
            continue
        if val < TEACHER_SANITY_FLOOR:
            broken.add(axis)
    return broken


def _apply_baseline_relative_penalty(
    axis_name: str,
    axis_value: float | None,
    reference_value: float | None,
) -> float | None:
    """Dock a bench axis when the student regresses below the same-round
    reference (Qwen-4B-base) on that axis.

    Returns ``axis_value`` unchanged when:
      * the penalty system is disabled
      * the axis is not in ``BASELINE_RELATIVE_PENALTY_AXES``
      * either value is None
      * the student is at parity or above the reference

    Otherwise returns ``max(0, axis_value - alpha * (ref - axis_value))``.
    The clip-to-zero matches the [0, 1] domain of bench axes; the
    composite ``worst`` aggregation already treats 0 as the lower
    bound, so a heavily-docked axis surfaces immediately as the
    worst-axis penalty.

    Same-round paired semantics are guaranteed by the caller: both
    models see identical block-seeded items, so the regression measure
    isolates real capability gap from prompt-mix drift.
    """
    if not BASELINE_RELATIVE_PENALTY_ENABLED:
        return axis_value
    if axis_name not in BASELINE_RELATIVE_PENALTY_AXES:
        return axis_value
    if axis_value is None or reference_value is None:
        return axis_value
    try:
        a = float(axis_value)
        r = float(reference_value)
    except (TypeError, ValueError):
        return axis_value
    if a >= r:
        return axis_value
    gap = r - a
    docked = a - BASELINE_RELATIVE_PENALTY_ALPHA * gap
    return max(0.0, docked)


def compute_composite(student: dict, king_kl: float | None = None,
                      king_rkl: float | None = None,
                      broken_axes: set[str] | None = None,
                      reference_axes: dict[str, float | None] | None = None,
                      king_eopd: float | None = None,
                      king_kl_is: float | None = None,
                      king_forking_rkl: float | None = None,
                      king_trace_nll: float | None = None,
                      king_kl_tail: float | None = None,
                      teacher_axes: dict[str, float | None] | None = None) -> dict:
    """Return per-axis and composite (worst-case + weighted mean) scores.

    We emit *both* aggregations so the validator can A/B them offline
    before committing to one as the canonical score:

    - ``worst`` (Coste 2024, Pan 2025 min-form): the minimum of present
      axes. This is the anti-gaming rule — you win only if all axes are
      competitive. Robust to axis-specific overfitting.
    - ``weighted`` (standard convex combination with AXIS_WEIGHTS): a
      softer aggregation that still rewards high-KL students somewhat,
      useful during the grace period so we don't suddenly dethrone the
      current king while miners re-tool.

    ``broken_axes`` (2026-04-23 / refined 2026-04-26): axes that should
    not gate the dethrone decision. Two distinct sources:
      * teacher-broken — the teacher itself failed the sanity floor this
        round (``resolve_teacher_broken_axes``).
      * reference-broken — the reference base model scored 0 on the axis
        (``resolve_reference_broken_axes``). This indicates an
        eval-setup signal (token truncation, parsing bug, unsolvable
        items), not student skill.

    Filter semantics (refined 2026-04-26):
      * ``worst`` excludes broken axes — otherwise the worst-axis
        anti-gaming rule degenerates to ``min == 0`` for every student
        whenever an axis hits a setup floor.
      * ``weighted`` KEEPS broken axes when computable — a student who
        scores >0 on a broken axis (e.g. they actually solved
        tool_use_bench while Qwen base couldn't) still gets credit in
        the soft aggregator. Only axes a student didn't even score
        (None) drop out of weighted.

    Caller is responsible for passing the union of teacher-broken and
    reference-broken sets via ``broken_axes``.

    ``reference_axes`` (2026-04-28, v29.1): per-axis values of the
    same-round Qwen-4B-base reference. When provided, each bench axis in
    ``BASELINE_RELATIVE_PENALTY_AXES`` is docked by
    ``BASELINE_RELATIVE_PENALTY_ALPHA × (ref - student)`` if the student
    regresses below the reference. The raw axis values are preserved in
    ``axes_raw`` for telemetry; ``axes`` reflects the docked values that
    flow into ``worst`` / ``weighted`` and downstream gates.
    """
    raw_axes = compute_axes(
        student, king_kl, king_rkl,
        king_eopd=king_eopd,
        king_kl_is=king_kl_is,
        king_forking_rkl=king_forking_rkl,
        king_trace_nll=king_trace_nll,
        king_kl_tail=king_kl_tail,
        teacher_axes=teacher_axes,
        broken_axes=broken_axes,
    )
    if reference_axes:
        axes = {
            k: _apply_baseline_relative_penalty(k, v, reference_axes.get(k))
            for k, v in raw_axes.items()
        }
    else:
        axes = dict(raw_axes)
    # Effective weights = the canonical "active axes" set. Single source
    # of truth in ``get_effective_axis_weights``: a SHADOW→production
    # promotion only needs to flip the relevant ``*_IN_COMPOSITE`` env
    # gate and (if w > 0) increase the axis weight; every gate downstream
    # (this function, the composite-floor veto, the teacher sanity gate,
    # the dashboard backfill) automatically picks up the new axis.
    effective_weights = get_effective_axis_weights()
    # ``ranked`` = axes used by ``worst()``: drops broken axes so the
    # min is not artificially floored by an axis the eval setup itself
    # can't pass (the dethrone gate degenerates to 0=0=0 otherwise).
    ranked = {
        k: v for k, v in axes.items()
        if v is not None
        and k in effective_weights
        and (not broken_axes or k not in broken_axes)
    }
    # ``weighted_axes`` = axes used by ``weighted``: KEEP broken axes
    # so a student who beats the reference on a broken axis still gets
    # credit in the soft aggregator. Only None values drop out.
    weighted_axes = {
        k: v for k, v in axes.items()
        if v is not None and k in effective_weights
    }
    if not ranked:
        return {"version": COMPOSITE_SHADOW_VERSION, "axes": axes,
                "axes_raw": (
                    {k: (round(v, 4) if v is not None else None) for k, v in raw_axes.items()}
                    if reference_axes else None
                ),
                "baseline_penalty": None,
                "worst": None, "worst_3_mean": None, "final": None,
                "weighted": None, "present_count": 0,
                "broken_axes": sorted(broken_axes) if broken_axes else [],
                "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE,
                "bench_in_composite": BENCH_AXES_IN_COMPOSITE,
                "arena_v3_in_composite": ARENA_V3_AXES_IN_COMPOSITE,
                "reasoning_density_in_composite": REASONING_DENSITY_IN_COMPOSITE,
                "chat_turns_in_composite": CHAT_TURNS_AXIS_IN_COMPOSITE,
                "top_k_overlap_in_composite": TOP_K_OVERLAP_AXIS_IN_COMPOSITE,
                "long_form_judge_in_composite": LONG_FORM_JUDGE_AXIS_IN_COMPOSITE}
    worst = min(ranked.values())
    # v30.2 — bottom-K mean (default K=3). Smooths the single-axis-min
    # noise pathology while preserving anti-Goodhart pressure. Drops
    # broken axes the same way ``worst`` does. If fewer than K axes
    # are present (small round / many broken), use whatever's there.
    sorted_ranked = sorted(ranked.values())
    k_eff = min(WORST_3_MEAN_K, len(sorted_ranked))
    worst_k_mean = (
        sum(sorted_ranked[:k_eff]) / k_eff if k_eff > 0 else None
    )
    total_w = sum(effective_weights[k] for k in weighted_axes)
    weighted = (
        sum(effective_weights[k] * v for k, v in weighted_axes.items()) / total_w
        if total_w else None
    )
    # v30.2 — final ranking key. Blends the bottom-K mean with the
    # weighted mean. Default 0.7 / 0.3 split (heavy on the bottom).
    # If either component is None, fall back to the available one.
    if worst_k_mean is not None and weighted is not None:
        final_score = (
            COMPOSITE_FINAL_BOTTOM_WEIGHT * worst_k_mean
            + (1.0 - COMPOSITE_FINAL_BOTTOM_WEIGHT) * weighted
        )
    elif worst_k_mean is not None:
        final_score = worst_k_mean
    elif weighted is not None:
        final_score = weighted
    else:
        final_score = None

    # 2026-04-25 — anti-gaming visibility. Two informational scores that
    # tell operators when a student is unusually narrow:
    #
    #   * ``axis_spread``: stdev of all axis values present in the round
    #     (whether they're in the composite or not). A balanced student
    #     has low spread (~0.05); a specialist who games one axis has
    #     spread > 0.15. Not used for gating — the worst-axis rule
    #     already captures specialist failure, this just makes narrow
    #     profiles visible earlier.
    #
    #   * ``bench_vs_rel_gap``: mean(bench pass-fracs) minus mean(relative
    #     axes). A miner who memorized bench items via rotation
    #     inspection without improving policy-level capability shows up
    #     as a big positive gap. Normal miners: roughly zero. Flagged
    #     in telemetry so we can audit rotation-memorization attempts
    #     before they matter.
    all_values = [v for v in axes.values() if v is not None]
    axis_spread = None
    if len(all_values) >= 2:
        m = sum(all_values) / len(all_values)
        var = sum((v - m) ** 2 for v in all_values) / len(all_values)
        axis_spread = var ** 0.5

    rel_keys = ("kl", "on_policy_rkl", "top_k_overlap", "entropy_aware_kl",
                "kl_is", "forking_rkl", "teacher_trace_plausibility",
                "tail_decoupled_kl",
                "capability", "judge_probe", "long_form_judge",
                "chat_turns_probe", "length", "degeneracy")
    bench_keys = tuple(BENCH_AXIS_WEIGHTS.keys()) + tuple(ARENA_V3_AXIS_WEIGHTS.keys())
    rel_vals = [axes[k] for k in rel_keys if axes.get(k) is not None]
    bench_vals = [axes[k] for k in bench_keys if axes.get(k) is not None]
    bench_vs_rel_gap = None
    if len(rel_vals) >= 2 and len(bench_vals) >= 2:
        bench_vs_rel_gap = (sum(bench_vals) / len(bench_vals)) - (sum(rel_vals) / len(rel_vals))

    # 2026-04-28 (v29.1): when the baseline-relative penalty is in
    # effect, ``axes`` reflects the *post-penalty* values that drive
    # ranking. We also surface ``axes_raw`` (pre-penalty) and a
    # ``baseline_penalty`` summary so the dashboard can show both the
    # raw bench score and the regression dock without losing signal.
    baseline_penalty_summary = None
    if reference_axes:
        deltas: dict[str, dict[str, float]] = {}
        for axis in BASELINE_RELATIVE_PENALTY_AXES:
            raw_v = raw_axes.get(axis)
            adj_v = axes.get(axis)
            ref_v = reference_axes.get(axis)
            if raw_v is None or ref_v is None:
                continue
            if adj_v is None:
                continue
            if abs(raw_v - adj_v) < 1e-6 and raw_v >= ref_v:
                continue
            deltas[axis] = {
                "raw": round(float(raw_v), 4),
                "adjusted": round(float(adj_v), 4),
                "reference": round(float(ref_v), 4),
                "gap": round(float(ref_v - raw_v), 4),
                "dock": round(float(raw_v - adj_v), 4),
            }
        baseline_penalty_summary = {
            "enabled": BASELINE_RELATIVE_PENALTY_ENABLED,
            "alpha": BASELINE_RELATIVE_PENALTY_ALPHA,
            "applied": deltas,
            "n_docked": len(deltas),
        }

    return {
        "version": COMPOSITE_SHADOW_VERSION,
        "axes": {k: (round(v, 4) if v is not None else None) for k, v in axes.items()},
        "axes_raw": (
            {k: (round(v, 4) if v is not None else None) for k, v in raw_axes.items()}
            if reference_axes else None
        ),
        "baseline_penalty": baseline_penalty_summary,
        # v30.2 — ``final`` is the canonical ranking key (was ``worst``).
        # ``worst`` (single-axis min) and ``worst_3_mean`` are kept for
        # telemetry / dashboard / regression analysis.
        "final": round(final_score, 4) if final_score is not None else None,
        "worst": round(worst, 4),
        "worst_3_mean": round(worst_k_mean, 4) if worst_k_mean is not None else None,
        "final_alpha": round(COMPOSITE_FINAL_BOTTOM_WEIGHT, 4),
        "weighted": round(weighted, 4) if weighted is not None else None,
        "axis_spread": round(axis_spread, 4) if axis_spread is not None else None,
        "bench_vs_rel_gap": round(bench_vs_rel_gap, 4) if bench_vs_rel_gap is not None else None,
        "present_count": len(ranked),
        "broken_axes": sorted(broken_axes) if broken_axes else [],
        "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE,
        "bench_in_composite": BENCH_AXES_IN_COMPOSITE,
        "arena_v3_in_composite": ARENA_V3_AXES_IN_COMPOSITE,
        "reasoning_density_in_composite": REASONING_DENSITY_IN_COMPOSITE,
        "chat_turns_in_composite": CHAT_TURNS_AXIS_IN_COMPOSITE,
        "top_k_overlap_in_composite": TOP_K_OVERLAP_AXIS_IN_COMPOSITE,
        "long_form_judge_in_composite": LONG_FORM_JUDGE_AXIS_IN_COMPOSITE,
    }


def compute_pareto_dominance(
    challenger_axes: dict[str, float | None],
    king_axes: dict[str, float | None],
    margin: float | None = None,
    min_comparable: int | None = None,
    include_shadow: bool = True,
) -> dict[str, Any]:
    """Compute pairwise Pareto dominance of challenger vs king across axes.

    Returns a dict with:
      * ``wins``: axes where challenger > king + margin (strictly better).
      * ``losses``: axes where king > challenger + margin (strictly worse).
      * ``ties``: axes where |challenger - king| <= margin (within noise).
      * ``comparable``: count of axes where both have data.
      * ``pareto_wins`` (bool): challenger beats king on a majority of
        comparable axes AND does not lose on more axes than it wins.
        This is the "soft Pareto dominance" the Affine subnet uses:
        rather than requiring strict dominance on every axis (noisy
        and unwinnable), we require the challenger to win a majority
        without losing more than it wins. Returns False on
        insufficient comparable axes (fails open — the existing
        worst-axis gate still applies).

    The ``include_shadow`` flag controls whether axes that are currently
    in SHADOW mode are considered. By default we include them so the
    Pareto score reflects the full eval surface — shadow axes are
    designed to become production, this score is how we judge whether
    they're ready to flip.
    """
    margin = margin if margin is not None else PARETO_DOMINANCE_MARGIN
    min_comparable = (
        min_comparable if min_comparable is not None
        else PARETO_DOMINANCE_MIN_COMPARABLE
    )
    axes_to_consider = set(AXIS_WEIGHTS.keys()) | {"judge_probe"}
    axes_to_consider |= set(BENCH_AXIS_WEIGHTS.keys())
    if include_shadow:
        axes_to_consider |= set(ARENA_V3_AXIS_WEIGHTS.keys())
        axes_to_consider |= {"reasoning_density", "chat_turns_probe"}

    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    comparable = 0
    for axis in axes_to_consider:
        c = challenger_axes.get(axis)
        k = king_axes.get(axis)
        if c is None or k is None:
            continue
        comparable += 1
        if c > k + margin:
            wins.append(axis)
        elif k > c + margin:
            losses.append(axis)
        else:
            ties.append(axis)
    if comparable < min_comparable:
        pareto_wins = False
        reason = f"insufficient_comparable_axes ({comparable} < {min_comparable})"
    else:
        # "Soft" Pareto: majority win AND net wins >= 0.
        majority = (comparable // 2) + 1
        pareto_wins = len(wins) >= majority and len(wins) >= len(losses)
        reason = (
            "dominates"
            if pareto_wins
            else (
                "no_majority"
                if len(wins) < majority
                else "more_losses_than_wins"
            )
        )
    return {
        "wins": sorted(wins),
        "losses": sorted(losses),
        "ties": sorted(ties),
        "comparable": comparable,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_ties": len(ties),
        "margin": margin,
        "min_comparable": min_comparable,
        "pareto_wins": bool(pareto_wins),
        "reason": reason,
    }


def _resolve_king_rkl(king_kl: float | None,
                      students_data: dict[Any, dict],
                      h2h_results: list[dict]) -> float | None:
    """Round-wide reference RKL for axis normalization.

    We anchor the RKL axis on the **best** (lowest) RKL observed in the
    round rather than the king's, because the king might be the model we
    are trying to dethrone for an on-policy pathology — if we anchored
    on the king's RKL the challenger could never score above 1.0 on
    this axis and the pathology would stay invisible in the composite.

    The resolution order is:
      1. Round-wide minimum mean_rkl across all students with a probe
         record, as long as at least one non-king student reported.
      2. Fall back to the king's own RKL if only the king reported
         (shouldn't really happen but keeps the axis defined).
      3. Return ``None`` if nobody reported, letting
         ``compute_composite`` use an absolute fallback anchor.
    """
    best = None
    non_king_best = None
    king_model = None
    king_entry = next((r for r in h2h_results if r.get("is_king")), None)
    if king_entry:
        king_model = king_entry.get("model")
    for model_name, data in students_data.items():
        opr = (data or {}).get("on_policy_rkl") or {}
        v = opr.get("mean_rkl")
        if v is None or v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
        if model_name != king_model and (non_king_best is None or v < non_king_best):
            non_king_best = v
    if non_king_best is not None and best is not None:
        return best
    if best is not None:
        return best
    return None


def _resolve_king_metric_min(students_data: dict[Any, dict],
                              key: str,
                              skip_floor: float = 1e-4) -> float | None:
    """Generic helper: round-wide MINIMUM of a numeric per-student field
    (lower-is-better metric like KL / RKL / NLL).

    Skips values <= ``skip_floor`` to avoid the teacher-vs-itself row
    pinning the reference to ~0 (every challenger would then receive
    None on the axis).

    Returns None if no qualifying row has the field.
    """
    best = None
    for _model_name, data in students_data.items():
        v = (data or {}).get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v or v <= skip_floor:
            continue
        if best is None or v < best:
            best = v
    return best


def _resolve_king_kl(king_kl: float | None,
                     students_data: dict[Any, dict]) -> float | None:
    """Round-wide reference KL for the ``kl`` axis.

    Mirrors :func:`_resolve_king_rkl` for the ``kl_global_avg`` metric.
    When a king is seated for the round we anchor on the king's KL.
    When there is no king (cold-start, post-cutover, or single-eval
    mode where the king isn't a round participant), we fall back to
    the round-wide MINIMUM ``kl_global_avg`` so the kl axis still has
    a meaningful denominator.

    Without this fallback the kl axis was ``None`` for every student
    in cold-start rounds because ``_axis_kl`` returns ``None`` whenever
    its ``king_kl`` argument is ``None``. That left the dashboard
    showing blank for ``Kl`` on the entire post-Kimi-cutover leader-
    board (Sebastian's report 2026-05-04). With the fallback the new
    round winner scores 1.0 on ``kl`` (best-anchored), challengers
    score on the king/winner ratio, and subsequent rounds inherit the
    crowned king's KL as the natural anchor.
    """
    try:
        if king_kl is not None:
            v = float(king_kl)
            if v == v and v > 0 and v not in (float("inf"), float("-inf")):
                return v
    except (TypeError, ValueError):
        pass
    return _resolve_king_metric_min(students_data, "kl_global_avg")


def _resolve_king_eopd(students_data: dict[Any, dict],
                       h2h_results: list[dict]) -> float | None:
    """Round-wide reference adaptive-KL for the entropy-aware (EOPD)
    axis. Min over students' ``eopd_adaptive_mean`` (lower=better),
    dropping the teacher-vs-itself row (~0 nats) so challengers strictly
    better than the king can score >1.

    Returns ``None`` if no student has the field.
    """
    return _resolve_king_metric_min(students_data, "eopd_adaptive_mean")


def compute_king_health(
    king_composite: dict | None,
    base_composite: dict | None,
) -> dict | None:
    """Summarize the king's composite health vs floor + base model.

    Shadow telemetry for the leeroyjkin/distil-97 feedback (2026-04-24).
    Two signals:
      * ``below_floor``     — worst axis < KING_COMPOSITE_FLOOR
      * ``worse_than_base`` — worst axis < base model's worst axis
                              (the base model is always in every round as
                              a reference anchor, so this is the most
                              direct "is the king regressing?" check)

    Returns None when either composite is missing or has no populated
    axes — fail open, never block on noisy probe data.
    """
    if not king_composite or king_composite.get("worst") is None:
        return None
    king_worst = float(king_composite["worst"])
    king_axes = king_composite.get("axes") or {}
    king_worst_axis = min(
        ((k, v) for k, v in king_axes.items() if v is not None),
        key=lambda kv: kv[1],
        default=(None, None),
    )[0]
    base_worst: float | None = None
    base_worst_axis: str | None = None
    if base_composite and base_composite.get("worst") is not None:
        base_worst = float(base_composite["worst"])
        base_axes = base_composite.get("axes") or {}
        base_worst_axis = min(
            ((k, v) for k, v in base_axes.items() if v is not None),
            key=lambda kv: kv[1],
            default=(None, None),
        )[0]
    below_floor = king_worst < KING_COMPOSITE_FLOOR
    worse_than_base = base_worst is not None and king_worst < base_worst
    return {
        "king_worst": king_worst,
        "king_worst_axis": king_worst_axis,
        "base_worst": base_worst,
        "base_worst_axis": base_worst_axis,
        "floor": KING_COMPOSITE_FLOOR,
        "below_floor": below_floor,
        "worse_than_base": worse_than_base,
        "at_risk": bool(below_floor or worse_than_base),
        "gate_active": KING_REGRESSION_GATE,
        "min_streak": KING_REGRESSION_MIN_STREAK,
    }


def _compute_king_canary_regression(king_uid: Any, state_dir: Any) -> dict | None:
    """Detect held-out canary regression of the current king vs Qwen 4B base.

    Reads:
      ``state/benchmarks/uid_<king_uid>.json``   — most recent canary run
                                                    for the current king
      ``state/benchmarks/<KING_CANARY_BASELINE_FILE>`` — Qwen 4B base
                                                    reference (default
                                                    ``baseline_qwen35_4b.json``)

    Computes the king's mean held-out score across ``KING_CANARY_AXES``
    (only axes with positive ``count`` count toward the mean) and the
    baseline's mean over the same axes, then flags at_risk when the
    king is more than ``KING_CANARY_MARGIN`` (default 5 pp) below the
    baseline. Used by ``state_manager.py`` to maintain a separate
    ``king_canary_streak`` and by ``_king_regression_floor_waived`` to
    waive the composite-floor veto for canary-regressing kings.

    Returns None if either file is missing, the uid in the king file
    doesn't match, or there are no comparable axes (fail open — never
    block dethrones on noisy probe data).
    """
    if king_uid is None:
        return None
    try:
        from pathlib import Path
        import json as _json
        bench_dir = Path(state_dir) / "benchmarks"
        if not bench_dir.exists():
            return None
        king_file = bench_dir / f"uid_{int(king_uid)}.json"
        baseline_file = bench_dir / KING_CANARY_BASELINE_FILE
        if not king_file.exists() or not baseline_file.exists():
            return None
        with open(king_file) as fh:
            king = _json.load(fh)
        with open(baseline_file) as fh:
            base = _json.load(fh)
        if king.get("uid") not in (int(king_uid), str(king_uid)):
            return None
        comparable: list[str] = []
        king_vals: list[float] = []
        base_vals: list[float] = []
        for axis in KING_CANARY_AXES:
            kv = king.get("benchmarks", {}).get(axis)
            bv = base.get("benchmarks", {}).get(axis)
            kc = (king.get("counts") or {}).get(axis)
            bc = (base.get("counts") or {}).get(axis)
            if not isinstance(kv, (int, float)) or not isinstance(bv, (int, float)):
                continue
            if kc is not None and (kc == 0):
                continue
            if bc is not None and (bc == 0):
                continue
            comparable.append(axis)
            king_vals.append(float(kv))
            base_vals.append(float(bv))
        if not comparable:
            return None
        king_mean = sum(king_vals) / len(king_vals)
        base_mean = sum(base_vals) / len(base_vals)
        gap = base_mean - king_mean
        at_risk = gap > KING_CANARY_MARGIN
        return {
            "king_canary_mean": round(king_mean, 4),
            "base_canary_mean": round(base_mean, 4),
            "gap_pp": round(gap, 4),
            "axes_compared": comparable,
            "margin": KING_CANARY_MARGIN,
            "at_risk": bool(at_risk),
            "gate_active": KING_CANARY_GATE,
            "min_streak": KING_CANARY_MIN_STREAK,
            "baseline_file": KING_CANARY_BASELINE_FILE,
        }
    except Exception:
        return None


def annotate_h2h_with_composite(h2h_results: list[dict], king_kl: float | None,
                                students_data: dict[Any, dict],
                                teacher_student_row: dict | None = None,
                                reference_model: str | None = None,
                                reference_uid: Any = None) -> None:
    """Mutates h2h_results in place to add ``composite`` per entry.

    ``students_data`` maps model_name -> the raw per-student dict from
    ``pod_eval_vllm.py`` output. We resolve each h2h entry's student by
    model name.

    ``teacher_student_row`` (optional, 2026-04-23) is the teacher's own
    per-student row — when pod_eval_vllm runs the teacher through the
    student probes (think/chat/capability/rkl) and deposits a row under
    the teacher's model name, pass that row here. Axes where the teacher
    falls below ``TEACHER_SANITY_FLOOR`` are dropped from every
    challenger's composite ranking this round with a note, preventing a
    miscalibrated axis from corrupting rankings (2026-04-19 failure
    class). If None / absent, every axis stays in play — fail open.

    2026-04-24 (Arena v3): also attaches a ``pareto`` sub-dict to each
    non-king row describing wins/losses/ties vs the current king across
    every available axis (shadow + production). The ``pareto_wins``
    boolean is informational-only until ``PARETO_DOMINANCE_GATE`` is
    flipped to production; meanwhile it is surfaced in the dashboard
    and telemetry so we can validate the gate's behavior on real data
    before promotion.
    """
    king_rkl = _resolve_king_rkl(king_kl, students_data, h2h_results)
    king_eopd = _resolve_king_eopd(students_data, h2h_results)
    # v30 — three additional research-paper shadow-axis king references.
    # Resolved via the same min-with-floor pattern as king_eopd.
    king_kl_is = _resolve_king_metric_min(students_data, "kl_is_mean")
    king_forking_rkl = _resolve_king_metric_min(students_data, "forking_rkl_mean")
    king_trace_nll = _resolve_king_metric_min(
        students_data, "teacher_trace_nll_mean"
    )
    # v30.3 — tail-decoupled KL king reference.
    king_kl_tail = _resolve_king_metric_min(students_data, "kl_tail_mean")
    broken = resolve_teacher_broken_axes(teacher_student_row, king_kl, king_rkl)

    # v30.2 — Teacher axis values for the super_teacher axis.
    # ``teacher_student_row`` is the teacher-as-student probe; we read
    # the raw bench pass_fracs to compute per-axis lift for each
    # student. We don't pass king/reference args because the teacher
    # row is the reference point itself.
    teacher_axes_for_super: dict[str, float | None] | None = None
    if teacher_student_row:
        try:
            teacher_axes_for_super = {
                ax: _axis_bench_pass_frac(teacher_student_row, ax)
                for ax in SUPER_TEACHER_AXES
            }
        except Exception as exc:
            logger.warning(
                f"composite: failed to extract teacher axes for "
                f"super_teacher computation: {exc} — axis will fail open."
            )
            teacher_axes_for_super = None
    # Layer 2 (2026-04-26): reference-broken axes. The reference base model
    # (REFERENCE_UID = -1, Qwen3.5-4B) runs the same bench probes as
    # students; an axis where the reference scores 0 is testing the eval
    # setup, not student skill. Drop it from worst() / weighted to keep
    # the dethrone gate from degenerating to "everyone is at 0".
    reference_row = None
    if reference_model and reference_model in students_data:
        reference_row = students_data[reference_model]
    reference_broken = resolve_reference_broken_axes(reference_row)
    if reference_broken:
        try:
            logger.info(
                "composite: dropping reference-broken axes this round: "
                f"{sorted(reference_broken)} (reference={reference_model} "
                "scored pass_frac=0 on each — eval-setup signal, not "
                "student skill)"
            )
        except Exception:
            pass
    broken = (broken or set()) | reference_broken

    # 2026-04-28 (v29.1): same-round reference axes for the per-axis
    # baseline-relative penalty. We compute the Qwen-4B-base axis values
    # once here and pass them to every compute_composite call so each
    # student's bench axes get docked when they regress below the same
    # block-seeded reference. Reference NOT being in the round (legacy
    # rounds before INCLUDE_REFERENCE_IN_ROUND=1) ⇒ reference_axes=None
    # ⇒ penalty silently fails open and scoring is unchanged.
    reference_axes_raw: dict[str, float | None] | None = None
    if reference_row is not None:
        try:
            # Reference uses ``broken_axes=None``: it's the row that
            # DEFINES which axes are broken (reference scored 0), so
            # we don't drop them from the reference's own group axes —
            # the reference is the anchor.
            reference_axes_raw = compute_axes(
                reference_row, king_kl, king_rkl,
                king_eopd=king_eopd,
                king_kl_is=king_kl_is,
                king_forking_rkl=king_forking_rkl,
                king_trace_nll=king_trace_nll,
                king_kl_tail=king_kl_tail,
                teacher_axes=teacher_axes_for_super,
                broken_axes=None,
            )
        except Exception as exc:
            logger.warning(
                f"composite: failed to compute reference axes: {exc} "
                "— per-axis baseline penalty will fail open this round."
            )
            reference_axes_raw = None

    king_model = None
    king_entry = next((r for r in h2h_results if r.get("is_king")), None)
    if king_entry:
        king_model = king_entry.get("model")
    king_raw_axes = None
    if king_model and king_model in students_data:
        king_raw_axes = compute_axes(
            students_data[king_model], king_kl, king_rkl,
            king_eopd=king_eopd,
            king_kl_is=king_kl_is,
            king_forking_rkl=king_forking_rkl,
            king_trace_nll=king_trace_nll,
            king_kl_tail=king_kl_tail,
            teacher_axes=teacher_axes_for_super,
            broken_axes=broken,
        )

    # 2026-04-29 (v29.3): which bench axes carry per-template breakdown.
    # We surface ``composite.per_src`` so the saturation audit can attribute
    # pass-rate per procedural template across rounds. Cheap to compute
    # (already populated by ``_bench_finalize_token_stats``); we just
    # forward it onto the h2h entry. Filter to bench axes only — the
    # relative axes (``kl`` etc.) don't have a meaningful template
    # breakdown.
    PER_SRC_AXES = (
        "math_bench", "code_bench", "reasoning_bench", "ifeval_bench",
        "aime_bench", "mbpp_bench", "tool_use_bench", "long_context_bench",
        "robustness_bench", "noise_resistance_bench", "knowledge_bench",
        "self_consistency_bench", "arc_bench", "truthful_bench",
        "procedural_bench", "debug_bench",
        # v29.4 axes also expose per_src for saturation telemetry.
        "correction_bench", "multi_doc_synthesis_bench",
        "calibration_bench", "refactor_bench",
        # v30.3 — pragmatic_bench now also exposes per_src (mirrored
        # alongside its legacy per_subtype) so the saturation audit
        # picks it up. The dashboard can keep reading per_subtype.
        "pragmatic_bench",
    )

    for entry in h2h_results:
        model = entry.get("model")
        if not model or model not in students_data:
            continue
        # The reference itself is NOT penalized against itself — pass
        # ``reference_axes=None`` for that one row so its composite
        # reflects raw axis values (it's the anchor by definition).
        is_reference_row = (reference_model is not None and model == reference_model)
        ref_axes_for_call = None if is_reference_row else reference_axes_raw
        comp = compute_composite(
            students_data[model], king_kl, king_rkl,
            broken, ref_axes_for_call,
            king_eopd=king_eopd,
            king_kl_is=king_kl_is,
            king_forking_rkl=king_forking_rkl,
            king_trace_nll=king_trace_nll,
            king_kl_tail=king_kl_tail,
            teacher_axes=teacher_axes_for_super,
        )
        # Note: compute_composite passes ``broken`` to compute_axes
        # internally so the group axes drop broken sub-axes.
        if entry.get("disqualified") and not entry.get("is_king"):
            comp = {**comp, "worst": 0.0, "weighted": 0.0,
                    "disqualified": True, "dq_reason": entry.get("dq_reason")}
        # Pareto dominance vs king (non-king rows only — a king Pareto
        # score against itself is definitionally the trivial tie case).
        if king_raw_axes is not None and not entry.get("is_king"):
            challenger_raw_axes = compute_axes(
                students_data[model], king_kl, king_rkl,
                king_eopd=king_eopd,
                king_kl_is=king_kl_is,
                king_forking_rkl=king_forking_rkl,
                king_trace_nll=king_trace_nll,
                king_kl_tail=king_kl_tail,
                teacher_axes=teacher_axes_for_super,
                broken_axes=broken,
            )
            comp["pareto"] = compute_pareto_dominance(
                challenger_raw_axes, king_raw_axes, include_shadow=True,
            )
        # v29.3: forward per-template breakdown (`per_src`) for each
        # bench axis that has it. Used by the saturation audit script
        # to identify dead/saturated templates per axis.
        per_src_summary: dict[str, dict] = {}
        for axis in PER_SRC_AXES:
            payload = students_data[model].get(axis)
            if isinstance(payload, dict):
                ps = payload.get("per_src")
                if isinstance(ps, dict) and ps:
                    per_src_summary[axis] = ps
        if per_src_summary:
            comp["per_src"] = per_src_summary
        entry["composite"] = comp

    # ── King health (2026-04-24 shadow) ─────────────────────────────
    # Stamp ``composite.king_health`` on the king's row with a worst-axis
    # summary vs floor + base model. Shadow only — state_manager tracks
    # the consecutive streak; dashboard + /api/miner/{uid} surface it.
    # When the gate is off (default), this is pure telemetry.
    # Callers pass in the reference model/UID so this module stays
    # ML-dep free (see module docstring).
    try:
        king_comp = king_entry["composite"] if king_entry and "composite" in king_entry else None
        base_entry = None
        if reference_uid is not None or reference_model is not None:
            base_entry = next(
                (r for r in h2h_results
                 if (reference_uid is not None and r.get("uid") == reference_uid)
                 or (reference_model is not None and r.get("model") == reference_model)),
                None,
            )
        base_comp = base_entry.get("composite") if base_entry else None
        health = compute_king_health(king_comp, base_comp)
        if health and king_comp is not None:
            king_comp["king_health"] = health
    except Exception:
        pass  # telemetry failure must never break ranking
