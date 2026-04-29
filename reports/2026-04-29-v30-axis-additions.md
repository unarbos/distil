# SN97 v30 — Eight New Axes & Mining Guide v2

**Date:** 2026-04-29
**Eval version:** v30 (composite_shadow_version=28 — schema unchanged; new axes are additive)
**Validator commit:** TBD (this report)

## Summary

This drop ships **eight new composite axes** (four ranking + four shadow) and a **mining guide v2** focused on the 4-stage 2026 SOTA distillation pipeline. The new axes fill capability + research-backed gaps identified in the [v30 strategic rollup](2026-04-29-v30-strategic-rollup.md) and the [SOTA distillation synthesis](2026-04-29-distillation-sota-synthesis.md). All eight are procedural / cache-derived, Goodhart-resistant, and free-or-cheap to compute. Together they raise the eval's faithful-predictor ratio without expanding the validator's wall-time budget by more than ~3%.

| Axis | Weight | What it measures | Why it exists |
| --- | --- | --- | --- |
| `top_k_overlap` | 0.10 | Mean fraction of teacher's top-K tokens that also appear in student's top-K, averaged across generated positions | The 2026 *Rethinking OPD* paper (arXiv 2604.13016) identifies top-K agreement as **the single most predictive signal of OPD success**. Successful runs converge to 97–99% shared mass; failed runs sit at 60–80% even when KL looks fine. Free to compute from the existing top-128 cache. |
| `entropy_aware_kl` | 0 (SHADOW) | Per-token RKL/FKL adaptive blend weighted by teacher entropy: α(H_t) · RKL + (1 − α(H_t)) · FKL with α = sigmoid((H₀ − H_t)/τ) | The EOPD paper (arXiv 2510.27485) shows +1.37 to +5.05 Pass@8 across 6 small-model math benchmarks vs vanilla OPD when this is used as the **training** loss. As an eval signal we collect it in shadow first, then promote once 48h of correlation data validates the H₀ = 1.5 nats / τ = 0.5 defaults. |
| `kl_is` | 0 (SHADOW) | Importance-sampled KL: `Σ p_t_full(k) · (log p_t_full(k) − log p_s_full(k))` over teacher's top-K. Unbiased full-vocab KL contribution. | Anshumann et al. ACL 2025 ("Sparse Logit Sampling: Accelerating KD in LLMs") proves that the existing renormalised top-K KL is **biased** by the teacher's tail-mass coverage. The IS estimator drops the renormalisation and uses raw teacher logprobs as full-vocab probabilities, giving a tight lower bound on the true full-vocab KL. Shadow until correlation telemetry confirms the bias matters in practice. |
| `forking_rkl` | 0 (SHADOW) | Reverse-KL averaged ONLY at positions in the top quartile of teacher entropy ("decision points") | Wang et al. 2025 / synthesis §4.2 #5: high-entropy positions are where teacher feedback is most informative; RKL there is a stronger predictor of OPD success than mean RKL across all positions. Computed from the same EOPD top-K cache pass at zero extra cost. |
| `teacher_trace_plausibility` | 0 (SHADOW) | Average NLL student assigns to teacher's actually-emitted tokens. `−mean(log p_s(teacher_token_t))`. | Synthesis §4.2 #4. Captures "support coverage" — does the student place mass on the teacher's chosen path? A model with high FKL but low plausibility is putting mass everywhere except where the teacher actually goes (a known LIMO/s1 SFT-only failure mode). Distinct signal from FKL (full-distribution match) and RKL (student-rollout match). |
| `knowledge_bench v2` | 0.05 (was 0) | Procedural fact-like reasoning (price tables, transitive ordering, container counting, alphabet/calendar/weekday/unit/roman conventions). Open-ended generation with regex-match grading. | The legacy MMLU-Pro 10-way MC version was muted to weight 0 in v28 because of a random-pick floor. The v2 redesign keeps factual reasoning in the composite without the MC floor problem. |
| `pragmatic_bench` | 0.04 | Procedural theory-of-mind (Sally-Anne false-belief), scalar implicature, epistemic state tracking, and indirect-request recognition | No existing axis measures pragmatic reasoning. SOTA models that ace MMLU + GSM8K can still fail false-belief items, and this is a real deployment-quality gap for assistants. |
| `long_form_judge` | 0.05 | Teacher rubric (1-5 → [0,1]) on 300-500 word essay-style responses, scoring structure / depth / coherence / length | The short-form `judge_probe` only grades 1-2 line answers. Multi-paragraph coherence is a SOTA-distinct capability that pure climbing-KL distillation does not reward. |

Net composite weight added to ranking: 0.24 (the four shadow axes ship at weight 0 and contribute nothing to ranking until promoted). The four ranking axes share the 0.85 bench / 0.15 relative split via the existing renormalisation, so there's no re-tuning required for the existing axes.

Promotion order: shadow axes are promoted to weight > 0 once 48h of round-by-round correlation telemetry against the held-out canary axes (gsm8k / humaneval / bbh / ifeval / mmlu_pro) confirms they discriminate. The expected promotion order based on research priors:

1. `top_k_overlap` (already live, 0.10) — `Rethinking OPD` calls this the strongest single predictor.
2. `kl_is` — drop-in unbiased replacement for the existing `kl` axis (currently 0.05 weight); migrate weight from `kl` → `kl_is` to avoid axis duplication.
3. `forking_rkl` — complement to `on_policy_rkl` (0.35 weight); start at 0.05.
4. `teacher_trace_plausibility` — complement to FKL; start at 0.05.
5. `entropy_aware_kl` — complement to both; start at 0.05.

## Detailed design

### `top_k_overlap`

**Definition.** At each generated position, count how many of the teacher's top-K predicted tokens also appear in the student's top-K predicted tokens. Average per-position over the prompt continuation, average across prompts.

**Implementation.** In `scripts/pod_eval_vllm.py`, the per-prompt KL loop is augmented with a top-K overlap computation in both the sparse (vLLM top-128 cache) and dense (legacy HF top-K) paths. Computed alongside KL, no extra forward pass needed. Aggregate stored as `student.top_k_overlap_mean`.

In `scripts/validator/composite.py`, `_axis_top_k_overlap` reads the field and clamps to [0, 1]. Wired into `compute_axes`, `AXIS_WEIGHTS` (default 0.10), `compute_composite`'s effective weights, and the `bench_vs_rel_gap` rel-axis classification.

**Goodhart resistance.** Cannot be gamed by per-token probability calibration (only the *set* of top-K matters), cannot be gamed by prompt-pool memorisation (top-K depends on the on-policy trajectory), cannot be gamed by length collapse (positions are evaluated per-token).

**Dynamic range.** Random student ≈ 7%, decent ≈ 70-90%, SOTA-distilled ≈ 97-99%. Already in [0, 1] from per-prompt averaging.

**Tests.** `tests/test_arena_v3_composite.py::TestTopKOverlapAxis` — 5 test cases covering field reading, clamping, missing/NaN/Inf handling, gate respect, and the relative-axis classification for telemetry.

### `knowledge_bench v2`

**Subtypes (8, balanced per round):**

1. **price_table_total** — 3-row price table; compute total of N items.
2. **transitive_order** — N people ordered transitively; name the position.
3. **container_count** — N containers with red/blue items; total of one colour.
4. **alphabet_position** — Nth letter / position of X / distance between two letters.
5. **calendar_offset** — month arithmetic with month names.
6. **day_of_week_offset** — weekday arithmetic with weekday names.
7. **unit_convert** — integer-output unit conversions (km↔m, hr↔min, etc).
8. **roman_numeral** — Roman ↔ Arabic for values 2–50.

All subtypes are 100% procedural — the (question, gold) pair is fresh per `block_seed`. Open-ended generation; grading uses per-item compiled regex `accept` patterns with word/digit boundary protection so substring confounders don't false-match (gold "8" inside "180" rejected, gold "R" inside "RR" rejected).

**Implementation.** New `_generate_knowledge_v2_items(block_seed, n_items)` generator in `pod_eval_vllm.py`; updated `knowledge_bench_probe` to detect MC-shape vs open-ended-shape items and grade accordingly (legacy MC pre-cached records still grade correctly during the transition window). Wired into `set_bench_block_seed`.

**Tests.** `tests/test_knowledge_v2_bench.py` — 8 test cases covering subtype coverage, block-seed determinism, gold/wrap matching, boundary-substring rejection, and empty-input handling.

### `pragmatic_bench`

**Subtypes (4, balanced per round):**

1. **false_belief** — Sally-Anne style. Two question types per item: "where will X look first?" (gold = container_1, X's belief) vs "where IS the object now?" (gold = container_2, world state). Forces the model to track WHOSE knowledge state is being asked about.
2. **scalar_implicature** — given a weak-quantifier utterance ("some/several/many/a few"), ask whether the strong interpretation ("all") follows. Gold = no.
3. **epistemic_state_tracking** — three actors A, B, C. A tells B; C is in a different room. Question targets one of them.
4. **indirect_request** — polite-form question that is canonically a request ("Could you pass the salt?"). Two question types: literal (about ability) vs pragmatic (about action).

100% procedural with synthetic names, objects, containers via `_synthetic_name` / `_synthetic_word`. Open-ended grading via the same `_knowledge_v2_grade_one` helper used by `knowledge_bench_v2`.

**Implementation.** New `_generate_pragmatic_items` generator + `pragmatic_bench_probe` probe + `_axis_pragmatic_bench` composite axis. Bench-stream offset `0xB0D1CA10` keeps the pool disjoint from other axes' rotations. Wired into the bench battery (`run_bench_battery`), `compute_axes`, `ARENA_V3_AXIS_WEIGHTS`, `BASELINE_RELATIVE_PENALTY_AXES`, `BENCH_MIN_VALID`, `REASONING_DENSITY_TARGET_TOKENS`, and the broken-axis allowlist.

**Tests.** `tests/test_pragmatic_bench.py` — 7 test cases covering subtype coverage, determinism, gold matching, false-belief Q-type variation, scalar-implicature contract, and epistemic-state-tracking yes/no contract.

### `kl_is` (SHADOW)

**Definition.** `KL_IS = Σ_{k ∈ top_K_t} p_t_full(k) · (log p_t_full(k) − log p_s_full(k))` where `p_t_full(k) = exp(t_logp_raw[k])` is the teacher's true probability for token k under the full vocab (the top-K vLLM logprobs are already log-probs of the full distribution restricted to the K most likely tokens, NOT renormalised over those K).

**Why this matters.** The existing `kl` axis (computed via `compute_kl_from_sparse`) renormalises BOTH teacher and student over the shared top-K support, so it measures `KL(p_t_topK || p_s_topK)` — a real KL but on a different distribution than the full-vocab teacher and student. The bias depends on the teacher's top-K mass coverage (typically 0.95-0.99 for K=128 but varies per token).

The IS estimator drops the renormalisation and uses raw teacher logprobs as full-vocab probabilities. The tail contribution is dropped, but for a typical LM this misses ≤0.05 nats even when full-vocab KL is 0.5+ nats, so the estimator is a tight lower bound on full-vocab KL.

**Implementation.** `compute_kl_is_from_sparse` in `pod_eval_vllm.py`. Computed alongside the existing FKL in the per-prompt sparse loop. Persisted as `kl_is_mean` (mean) and `topk_mass_mean` (teacher's top-K coverage, telemetry signal — positions with `topk_mass < 0.9` are exactly where the bias of the renormalised KL is largest).

**Composite axis.** `_axis_kl_is(student, king_kl_is)` normalises against the round-wide minimum the same way `_axis_kl` does (king/student ratio). Default weight 0 (SHADOW). Resolution helper: `_resolve_king_metric_min(students_data, "kl_is_mean")` with the standard 1e-4 floor to skip teacher-vs-itself rows.

**Tests.** `tests/test_arena_v3_composite.py::TestImportanceSampledKLAxis` — 3 test cases. Plus `TestResolveKingMetricMin` covers the generic king-min resolver shared with the other shadow axes.

### `forking_rkl` (SHADOW)

**Definition.** `forking_rkl = mean(RKL[t] | H_t[t] ≥ Q_3(H_t))` — average reverse-KL over positions in the top quartile of teacher entropy.

**Why this matters.** Wang et al. 2025 (cited in the Thinking Machines OPD blog and our synthesis §4.2 #5): high-entropy teacher positions are the "decision points" where the teacher's feedback is most informative. The RKL at those positions is a stronger predictor of downstream OPD success than the mean RKL across all positions, because confident-teacher positions carry less new information for the student.

**Implementation.** Computed inside the EOPD pass: we already have per-token teacher entropy `H_t` and per-token RKL from `compute_eopd_metrics_from_sparse`; we just compute the 75th-percentile threshold of `H_t` per prompt and average RKL over the masked positions. Free.

**Composite axis.** `_axis_forking_rkl(student, king_forking_rkl)` with the same king/student normalisation shape as the other shadow axes. Default weight 0.

**Tests.** `tests/test_arena_v3_composite.py::TestForkingRKLAxis` — 2 test cases.

### `teacher_trace_plausibility` (SHADOW)

**Definition.** `−mean(log p_s_full(teacher_token_t))` — average negative log-likelihood the student assigns to the teacher's actually-emitted (greedy) continuation tokens.

**Why this matters.** Distinct from FKL (which weights the FULL teacher distribution including tokens the teacher didn't sample) and RKL (which weights the student's own rollout, not the teacher's). Catches "place mass everywhere except where the teacher goes" failure modes — empirically a LIMO/s1 SFT-only pathology. Synthesis §4.2 #4.

**Implementation.** Per-prompt: gather student log-prob at teacher's emitted token id at each continuation position, average. Free (uses the student's full-vocab `log_softmax` already computed for FKL and IS-KL).

**Composite axis.** `_axis_teacher_trace_plausibility(student, king_trace_nll)` with king/student normalisation. Default weight 0.

**Tests.** `tests/test_arena_v3_composite.py::TestTeacherTracePlausibilityAxis` — 2 test cases.

### `entropy_aware_kl` (SHADOW)

**Definition.** Per-token, on the same shared top-K support that the existing FKL is computed on:
- FKL = Σ_k p_t · (log p_t − log p_s)
- RKL = Σ_k p_s · (log p_s − log p_t)
- H_t = −Σ_k p_t · log p_t  (teacher entropy in nats)
- α(H_t) = sigmoid((H₀ − H_t) / τ)
- adaptive[t] = α · RKL[t] + (1 − α) · FKL[t]

When teacher is confident (low H), α → 1 and the metric weights RKL more (mode-seeking). When teacher is uncertain (high H), α → 0 and the metric weights FKL more (mode-covering). This is the **eval mirror** of the EOPD training loss.

**Defaults:** H₀ = 1.5 nats (median ClimbMix-prompt crossover), τ = 0.5 (smooth transition). Both are env-overridable for ablation (`EOPD_ENTROPY_THRESHOLD`, `EOPD_ENTROPY_SCALE`).

**Implementation.** New `compute_eopd_metrics_from_sparse` function in `pod_eval_vllm.py` reuses the same top-K cache as `compute_kl_from_sparse` and adds:
- Per-position teacher entropy (`(p_t * log p_t).sum`).
- Reverse KL on the renormalised shared support.
- Sigmoid-weighted adaptive blend.

The per-prompt loop computes the adaptive value alongside FKL and persists `eopd_adaptive`, `eopd_rkl`, `teacher_entropy` per prompt. Aggregated to model-level `eopd_adaptive_mean`, `eopd_rkl_mean`, `teacher_entropy_mean`.

In `composite.py`:
- `_axis_entropy_aware_kl(student, king_eopd)` normalises the same way `_axis_kl` does (king/student ratio, clipped to [0, 1]).
- `_resolve_king_eopd(students_data, h2h)` picks the round-wide minimum adaptive-KL with a sanity floor of 1e-4 nats so the teacher-vs-itself row (always ≈ 0) doesn't pin king_eopd to 0.
- `compute_axes` and `compute_composite` accept a `king_eopd` parameter; `annotate_h2h_with_composite` resolves it once per round and threads it through every per-student call.

**SHADOW deployment.** Default weight 0 in `AXIS_WEIGHTS`. The new zero-weight filter in `compute_composite` drops zero-weight axes from `effective_weights` so the shadow axis is computed and exposed in `axes` for telemetry but does NOT gate `worst()` or contribute to `weighted`. Operators promote by setting `ENTROPY_AWARE_KL_WEIGHT=0.05` once 48h of held-out canary correlation data validates the threshold defaults.

**Tests.** `tests/test_arena_v3_composite.py::TestEntropyAwareKLAxis` — 8 test cases covering normalisation correctness, missing-data handling, zero-king guard, compute_axes propagation, the zero-weight-doesn't-gate-worst contract, the post-promotion gating contract, king resolution minimum-pick, and teacher-near-zero floor handling.

**Smoke validation:** at peaked-teacher position (H = 0.42 nats, α = 0.90), adaptive ≈ RKL. At flat-teacher position (H = 1.39 nats, α = 0.56), adaptive blends ~50/50. The math matches the EOPD paper's spec.

### `long_form_judge`

**Architecture.** Mirrors the existing `judge_probe` but with:

- 4 prompts per round (vs 16 for short-form).
- 1024 max tokens per response (vs 256).
- Distinct rubric template (`LONG_FORM_JUDGE_RUBRIC_TEMPLATE`) explicitly weighting STRUCTURE / DEPTH / COHERENCE / LENGTH on a 1-5 scale.
- Distinct procedural prompt synthesis: 6 prompt templates × 20 topics, both rotated per round via `block_seed`.

**Phase A** (per-student): `long_form_judge_response_probe` collects greedy responses; rollouts stash in `_LONG_FORM_JUDGE_ROLLOUTS` for Phase B.

**Phase B** (after teacher reload): `long_form_judge_teacher_score` runs the rubric for each (prompt, response) pair, parses the 1-5 digit, normalises via (mean−1)/4. Result stored at `student.long_form_judge_probe`.

**Composite.** `_axis_long_form_judge` reads `long_form_judge_probe.normalized` with min-valid floor `LONG_FORM_JUDGE_MIN_VALID=2` (same convention as judge_probe).

**Tests.** `tests/test_arena_v3_composite.py::TestLongFormJudgeAxis` — 5 test cases covering field reading, missing-data handling, min-valid floor, compute_axes integration, gate-respect.

## Mining Guide v2

[`docs/MINING_GUIDE_V2.md`](../docs/MINING_GUIDE_V2.md) — 350-line miner-facing playbook documenting:

- The **4-stage pipeline** (mid-train SFT → curated SFT → on-policy RKL → optional GRPO).
- The **LIMO/s1 regression warning** (Phi-4-Mini-Reasoning paper data showing 4B students *regress* below base on naive LIMO/s1 SFT).
- Stage-by-stage **success thresholds** at each axis (king-class vs "good enough to commit").
- The new v30 axes and how to maximise them.
- 7 common Goodhart traps with concrete fixes.
- An end-to-end pre-commit checklist.

## Operational notes

- The four ranking axes (`top_k_overlap`, `knowledge_bench v2`, `pragmatic_bench`, `long_form_judge`) default ON (`*_IN_COMPOSITE=1`) and have env-flippable kill switches for rollback. The `entropy_aware_kl` axis ships SHADOW (weight 0) and is excluded from `worst()` until promoted.
- Wall-time impact: top_k_overlap and entropy_aware_kl are essentially free (computed from the existing top-K cache; <0.5s/student amortised). `knowledge_bench` v2 + `pragmatic_bench` + `long_form_judge` together add ~25-30s per student per round (~2-3% of typical round wall time).
- The four ranking axes contribute to the `worst()` ranking key — a model that scores 0 on any of them gets pulled down on the leaderboard regardless of its other scores. This is intentional: the axes were chosen because each measures a real capability gap.

## Rollback

Single-flag rollback for any axis:

- `TOP_K_OVERLAP_AXIS_IN_COMPOSITE=0`
- `BENCH_KNOWLEDGE_PER_ROUND=0` (also retires the v2 pool)
- `BENCH_PRAGMATIC_PER_ROUND=0`
- `LONG_FORM_JUDGE_IN_COMPOSITE=0` and/or `LONG_FORM_JUDGE_PROBE=0`
- `ENTROPY_AWARE_KL_WEIGHT=0` (already 0 by default — shadow axis kill switch)
- `KL_IS_AXIS_WEIGHT=0` (default 0)
- `FORKING_RKL_AXIS_WEIGHT=0` (default 0)
- `TEACHER_TRACE_PLAUSIBILITY_WEIGHT=0` (default 0)

Full rollback to v29.7: set the four ranking gates to 0 and restart the validator. The four shadow axes are already at weight 0 by default and have no production effect.

## Next steps

- 48h shadow-window monitoring on the four shadow axes (`entropy_aware_kl`, `kl_is`, `forking_rkl`, `teacher_trace_plausibility`) correlating against the held-out canary axes (gsm8k, humaneval, bbh, ifeval, mmlu_pro). Promote each to weight ≥ 0.05 once Pearson correlation > 0.4 against the canary mean.
- Per-axis correlation telemetry (already in `scripts/audit/axis_correlation.py`) extended to surface the four new shadow axes alongside the existing ones.
- Stage 2 Kimi K2.6 teacher-swap experiment runbook ([`reports/2026-04-29-kimi-k2.6-stage2-runbook.md`](2026-04-29-kimi-k2.6-stage2-runbook.md)) — separate-pod A/B experiment design with a tiered decision rule. Blocked on tokenizer-compatibility decision (Path A re-tokenize vs Path B Universal Logit Distillation). Not running in production until the experiment passes.
- Mining Guide v2 ([`docs/MINING_GUIDE_V2.md`](../docs/MINING_GUIDE_V2.md)) cited from the Discord welcome thread + the validator dashboard so the next king cycle has clear miner-facing recipes for each of the new axes.
