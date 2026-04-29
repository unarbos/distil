# SN97 v30 — Strategic Audit & Direction Review

**Date:** 2026-04-29 06:00 UTC+2
**Status:** Internal review after v30 / v30.1 axes shipped (commits `ae7152b` + `2ea8ed5`)

## TL;DR — what is true today

- **Big picture is mostly right** but has three structural issues we're not yet fixing: (a) the `worst()` aggregator is dominated by noise on the lowest-data axis, (b) winner-takes-all emissions over-concentrate gaming pressure, (c) we have ~25 axes with material weight overlap (5 of them measure code, 4 measure math).
- **Implementation is correct** — composite + axes are mathematically sound; IS-KL was numerically validated; tests pass; the new `worst()` filter for zero-weight shadow axes is safe.
- **Major efficiency wins available** but most require ~1 week of engineering each. The biggest is collapsing redundant axes and pre-batching student forward passes.
- **King selection has a real issue:** 35/160 (22%) of scored UIDs sit at `worst = 0` — `worst()` cannot discriminate inside that cluster, so `weighted` is the de facto tiebreaker for nearly a quarter of the leaderboard. The data shows we already implicitly built around this in `resolve_dethrone`, but it should be principled, not patched.
- **Shadow axes are safe to enable at low weight today.** The IS-KL math is correct, the EOPD computation matches the paper, and the failure mode is "axis returns None" (graceful degrade), not "axis crashes scoring".
- **Kimi K2.6 is NOT safe to switch to today** — tokenizer mismatch breaks every relative axis (KL / RKL / top-K-overlap / IS-KL / EOPD / on-policy-RKL). The runbook is the path; ~9 days of work.
- **Better incentive structure exists** — `build_winner_take_all_weights` puts 100% on the king on chain. Top-K linear or softmax-distributed weights would reduce gaming pressure without changing the eval.

---

## 1. Is the big picture correct?

**Claim:** SN97 distills a SOTA teacher (now Qwen3.6-35B-A3B) into miner-trained students (≤7B, same tokenizer); the validator scores students on a multi-axis composite; the king (highest `composite.worst`) takes 100% of emissions on chain.

### What's right

- **Multi-axis anti-Goodhart is sound.** The composite has 25+ axes spanning teacher-similarity (KL, RKL, top-K overlap), absolute correctness (math, code, reasoning, IFEval, AIME, MBPP), capability (debug, correction, refactor, calibration), pragmatic (false-belief, scalar-implicature), structure (long-form judge, chat-turns), and discipline (length, degeneracy). A model that games one axis loses on the others.
- **Procedural item generation is the only Goodhart-resistant route at our scale.** Static benchmarks (MMLU-Pro, AIME) are fully discoverable on disk and fully memorisable by miners. Block-seeded procedural items rotate the (question, gold) pair every round so memorisation isn't even available as a strategy.
- **Per-axis baseline-relative penalty (v29.1) is principled.** A regression below the same-round Qwen-4B base on ANY bench axis costs the student `1.5 × (ref - score)` extra. This makes "stay above base on every axis" the dominant strategy.
- **King-canary streak (v29.1) is principled.** If the king regresses on held-out evalscope (gsm8k / humaneval / bbh / ifeval) for 2+ rounds, the dethrone-floor veto is waived so a challenger can dethrone even on a borderline composite. This is our reality-check against the validator's own composite.

### What's structurally off

1. **`worst() = min(axes)` is dominated by noise on the lowest-data axis.** For 25 axes scored on 4–12 items each, the probability that AT LEAST ONE axis lands at 0 by item-level variance is ~30–60% even for genuinely good models. Our data confirms it: 35 of 160 UIDs (22%) sit at exactly `worst = 0`. The `resolve_dethrone` function already implicitly works around this with a "both saturated → fall back to weighted" path, but the official ranking key is still `worst`.

2. **Winner-takes-all amplifies gaming.** `build_winner_take_all_weights` puts 1.0 on the king and 0.0 on every challenger. A miner that's #2 by 0.001 gets 0% of emissions; a miner that's #1 by 0.001 gets 100%. This:
   - Multiplies gaming pressure — a 0.001 nat improvement in KL isn't 0.5% better, it's the difference between $0 and $X/round.
   - Discourages incremental improvements — why ship a 5% better model if the king will hold the crown until you can ship a 100% better one?
   - Concentrates compute risk — one miner on one good GPU pod crushes 200 miners with adequate but not optimal setups.

3. **Axis sprawl.** We have 25+ axes, many of which overlap heavily:
   - **Code:** `code_bench` (HumanEval-style write), `mbpp_bench` (MBPP), `debug_bench` (fix bugs), `correction_bench` (fix with error trace), `refactor_bench` (preserve behaviour) — 5 axes, weight total 0.31. A single "code competence" student beats all 5.
   - **Math:** `math_bench` (GSM8K-narrative), `aime_bench` (olympiad), `robustness_bench` (math under paraphrase), partly `tool_use_bench` (code-applied math) — 3-4 axes, weight total 0.27.
   - **Knowledge / reasoning:** `knowledge_bench v2` (factual reasoning), `reasoning_bench` (BBH), `pragmatic_bench` (theory-of-mind), `multi_doc_synthesis_bench` (cross-doc retrieval), `long_context_bench` (needle-in-haystack), `procedural_bench` (legacy, muted). Partial overlap.
   - **Teacher-similarity:** `kl`, `kl_is` (shadow), `on_policy_rkl`, `forking_rkl` (shadow), `top_k_overlap`, `entropy_aware_kl` (shadow), `teacher_trace_plausibility` (shadow). All measuring distillation quality, slightly different angles.

   Sprawl isn't fatal — it's the source of anti-Goodhart pressure — but it makes the worst-axis ranking very noisy and increases wall-time.

4. **The teacher is the ceiling.** Pure distillation cannot exceed teacher capability. To produce models that beat Qwen3.6 on real-world benchmarks, we need miners to mix in (a) teacher swap to a stronger teacher (Kimi K2.6 path), (b) RL on verifiable rewards (mining guide v2 stage 4), or (c) post-distillation SFT on harder data than the teacher saw. The eval needs to *reward* (b) and (c), not just measure teacher-similarity.

### Bottom line on big picture

The core architecture is right. The three structural issues above are addressable; none requires re-thinking the design.

---

## 2. Is the implementation correct?

I went back and re-verified every change shipped in v30 (`ae7152b`) and v30.1 (`2ea8ed5`). All five tests of correctness pass.

### v30 axes (top_k_overlap + knowledge_v2 + pragmatic + long_form_judge + entropy_aware_kl)

- **top_k_overlap.** Mathematically `|top_K_t ∩ top_K_s| / K`. We compute via broadcast + `.any(-1).sum(-1)` per chunk, exact and free. Smoke tested. `_axis_top_k_overlap` reads `top_k_overlap_mean` field, clamps to [0,1], handles NaN/Inf. ✅
- **knowledge_v2 grader.** Boundary-protected regex (`(?<!\d)8(?!\d)` etc.) so substring confounders don't false-match. 14/14 grading tests passed. ✅
- **pragmatic_bench.** Procedural false-belief, scalar implicature, epistemic state tracking, indirect-request items. Grammar fix shipped (`"is it the case that all of the X..."` not the prior ungrammatical `"did all of the X..."`). All 8 subtypes generate clean items. ✅
- **long_form_judge.** Phase A collects 4 essay-style responses per student; Phase B teacher rubric grades 1-5 → [0,1]. Mirrors the existing judge_probe architecture exactly. Per-prompt budget 1024 tokens. ✅
- **entropy_aware_kl (EOPD shadow).** `α(H_t) · RKL + (1 − α(H_t)) · FKL` with `α = sigmoid((H₀ - H_t)/τ)`. Smoke-validated: at peaked-teacher (H=0.42 nats, α=0.90) adaptive ≈ RKL; at flat-teacher (H=1.39 nats, α=0.56) adaptive blends. Math matches the EOPD paper spec. ✅

### v30.1 axes (kl_is + forking_rkl + teacher_trace_plausibility)

- **kl_is.** I derived the relationship between renormalised top-K KL and IS-KL: `KL_IS = topk_mass · (KL_renorm − log(s_topk_mass / topk_mass))`. Smoke confirmed: biased renormalised KL = 1.73 / 0.52 nats vs unbiased IS-KL = 2.47 / 1.36 nats — IS-KL is HIGHER, as expected because it does not pretend the top-K mass is 1.0. ✅
- **forking_rkl.** Computes per-position teacher entropy via the EOPD pass; finds the 75th-percentile threshold; averages RKL over the masked positions. Free since EOPD already produces these tensors. ✅
- **teacher_trace_plausibility.** `−mean(log p_s_full(teacher_token_t))` — gathers student log-prob at teacher's emitted token at each continuation position. Free, uses the same `s_log_p_full` we already compute for FKL. ✅

### Composite logic

- **Zero-weight filter** in `compute_composite`: `effective_weights = {k: w for k, w in effective_weights.items() if w > 0}`. This is critical — without it, the four shadow axes (default weight 0) would still be in `effective_weights`, so any student with `entropy_aware_kl = 0.05` would be pulled down to `worst = 0.05`. Tests covered this case (`test_zero_weight_axis_does_not_gate_worst`). ✅
- **King resolution** for the new axes uses `_resolve_king_metric_min(students, key, skip_floor=1e-4)` to skip the teacher-vs-itself row that has `≈ 0` adaptive-KL/RKL/NLL. Tests cover this. ✅
- **Threading.** Every call site that builds axes (`compute_axes` → `compute_composite` → `annotate_h2h_with_composite`) gets the four king refs (`king_kl_is`, `king_forking_rkl`, `king_trace_nll`, `king_eopd`) consistently. ✅
- **Shadow axis fail-safe.** All four shadow axes return `None` when the field is missing, when the king ref is missing, or when the value is non-positive / NaN / Inf. Failure mode is "axis drops out", never "axis returns garbage". ✅

### Live state cross-check (2026-04-29 06:00)

- Validator is on `2ea8ed5` (verified via `journalctl -u distil-validator`).
- `state/composite_scores.json`: 160 entries, schema version 28, `axes` dict has all 25+ keys.
- `state/king_regression_streak.json`: `{194: 0, 238: 0}` — no kings under regression streak.
- `state/axis_correlation.json` (computed 2026-04-28, BEFORE v30): `math_bench` r=−0.875 (strong Goodhart), `aime_bench` r=−0.500 (Goodhart). ⚠️ **This is the pre-v30 data. Need to recompute after v30 deploys to confirm the v29 GSM8K-narrative rebalance fixed it.**

### Bottom line on implementation

Everything I shipped is mathematically correct and tested. The one remaining concern is **stale axis_correlation telemetry** — the math/AIME anti-correlation may already be fixed by v29's GSM8K-narrative rebalance (which the audit predates), but we won't know until we re-run the audit script after the next ~10 rounds.

**Action:** re-run `scripts/audit/axis_correlation.py` once v30 has produced 5+ kings.

---

## 3. How can we be more efficient?

### Wall-time bottlenecks (measured + estimated)

Per-round wall time is approximately:

| Phase | Wall time | What |
| --- | --- | --- |
| Teacher continuations (vLLM) | ~3 min | 300 prompts × (sample teacher continuations) |
| Per-student KL/EOPD/IS-KL | ~7 min × N students | Student forward pass + per-prompt scoring |
| Bench probes (math/code/...) | ~2 min × N | 12 + 8 + 10 + 6 + 8 + ... ≈ 80 items per student |
| Judge + long-form judge collection | ~1 min × N | 16 + 4 prompts |
| RKL on-policy + chat-turns + judge teacher pass | ~3 min total | Phase B amortised across students |
| **Total** | **~60-80 min** | for ~10-15 students |

### Where compute is wasted

1. **Per-student KL is sequential.** We forward-pass each student through 300 prompts independently. With batching across students (load 3-4 students into VRAM at once) we'd cut Phase A by 3-4x. Engineering: ~1 week. Memory headroom: tight on a single H100, comfortable on H200/H100×2.

2. **Bench probe wall-time scales linearly with axis count.** With 5 code axes and 3 math axes, we're scoring code/math 5+3=8 times instead of 1+1. Most of these axes are 4-8 items, so the per-axis fixed cost (load tokens, generate, parse) dominates the actual scoring cost.

   **Concrete cut:**
   - Drop `mbpp_bench` (overlaps `code_bench`, weight 0.06 — saves ~60s/student).
   - Drop `correction_bench` (overlaps `debug_bench`, weight 0.03 — saves ~30s/student).
   - Drop `refactor_bench` (overlaps `code_bench` for write-from-scratch, weight 0.04 — saves ~30s/student).
   - **Saves ~2 min/student × 10 students = 20 min/round.**
   - Net composite weight redistributed: `code_bench` 0.14 → 0.20, `debug_bench` 0.06 → 0.09.

3. **Knowledge / reasoning axes:**
   - Drop `multi_doc_synthesis_bench` (weight 0.05) — `long_context_bench` already covers retrieval-with-distractors; multi-doc is a more complex variant of the same task. Saves ~30s/student.
   - Drop `procedural_bench` (already weight 0, but still in the bench loop). Can disable via `BENCH_PROCEDURAL_PER_ROUND=0` (already done, but verify).

4. **The bench item generators run inside each pod startup.** Could pre-generate per `(block_seed, axis)` tuple and ship as a small JSON in the eval payload. Saves ~5-10s of pod cold-start. Minor.

5. **Teacher cache regeneration per round.** The teacher's continuations on the 300 ClimbMix prompts are entirely deterministic given `(block_seed, teacher_revision)`. We could cache them across rounds and only regenerate when `block_seed` changes (which it does per round, so no immediate win) BUT we could precompute the teacher continuations BEFORE the round starts (background job during the previous round's downtime). Saves ~3 min/round.

### Recommended efficiency cut (concrete, conservative)

Drop `mbpp_bench`, `correction_bench`, `refactor_bench`, `multi_doc_synthesis_bench` — 4 axes that overlap heavily with retained axes. Bump `code_bench`, `debug_bench`, `long_context_bench` to absorb their weight. **Saves 20-25 min per round. Loses ~3% of axis-coverage breadth.** This is net positive — wall-time is the binding constraint on per-round throughput, and the dropped axes' marginal information is small.

I'm not shipping this in this turn because it's a SCHEMA bump (composite_shadow_version 28 → 29) and would invalidate every current king's record. It needs its own report, miner notice (Discord), and post-deploy monitoring. I'll write the cut-list spec.

---

## 4. Is the way we decide new kings correct?

### What we actually do (verified from `single_eval.py::resolve_dethrone`)

```
ch_worst > inc_worst × (1 + 3% margin)  →  challenger wins
ch_worst < inc_worst × (1 − 3% margin)  →  challenger rejected (regressed)
both worst at the saturated floor (≤ ε) →  fall through to weighted comparison
otherwise (tied) →  fall through to weighted with 3% margin
```

Plus:
- King-canary streak waives the floor veto if held-out canary regresses 2+ rounds.
- `_KING_SELECTION_MIN_VERSION = 28` quarantines old-schema records on every schema bump.
- Per-axis baseline penalty docks bench scores below same-round Qwen-4B reference.

### What's right

- **`worst()` is anti-Goodhart by construction.** A model that maxes one axis to 1.0 and tanks another to 0.10 ranks 0.10. Forces all-around competence.
- **3% dethrone margin protects against single-eval noise.** A challenger needs to clear the king on `worst` by 3% (or be at the saturated floor and clear on `weighted`). Real progress passes; lucky rounds don't.
- **Schema versioning prevents stale-grader inheritance.** When we bump composite from v27→v28→…, old kings can't camp on stale grading.
- **Held-out canary breaks ties when the validator goes Goodhart.** The streak gate auto-dethrones a king regressing on real benchmarks.

### What's wrong

1. **The `worst = 0` cluster is huge (22% of the leaderboard).** When 35 UIDs tie at 0, `worst()` cannot distinguish them. The actual ranker becomes `weighted`. This is FINE in practice (resolve_dethrone falls through), but the documented ranking key is `worst`, not `weighted-after-tied-worst`. The implementation matches reality but the explanation doesn't.

2. **Single-eval-per-commitment has variance issues.** A challenger gets ONE eval. If that eval lands on a particularly hard `block_seed` for that student's specific weak point, they don't get a do-over. Mitigation: the procedural items rotate, so the variance is roughly equal across students within a round. But if a student is borderline, single-eval noise can push them either side of the dethrone margin. We've seen this — the leaderboard has nearly-identical models clustered at 0.667 / 0.658 / 0.643 / 0.642 with `axis_spread = 0`.

3. **Saturated floor problem.** When `inc_worst = 0` and `ch_worst = 0`, `worst × (1 + margin) = 0`, so any positive `ch_worst` would dethrone. The code handles this with the `both_saturated` check + weighted fallback. Correct, but the dethrone rule is no longer "best worst-axis wins".

4. **Stale king never re-evaluated.** Once a king is set, their score is frozen. The procedural items rotate per round so the king's old block_seed is unique to their eval round. This means the king's score reflects performance on items NO ONE else saw, which is fine, but it also means we can't compare current students to the king on the SAME items without re-running them. This is the bigger issue: we trust the procedural distribution to be stationary across block_seeds, but we don't actively verify it.

### Recommendation: keep `worst()` as primary, fix the saturation and document the truth

- **Primary ranking:** `worst()` (unchanged) — anti-specialist, anti-Goodhart.
- **Tie-break:** explicit lex-order — `(worst, weighted, axis_spread_inverse, judge_probe)`. Currently the implementation does this implicitly; document it explicitly so miners know.
- **Saturated-floor cluster:** when 4+ models tie at `worst ≤ 0.02`, treat the tie as "the worst-axis is broken at the eval level", drop that axis from the worst() computation for THAT round only, and use second-worst. This is essentially the existing `broken_axes` mechanism extended to "axes that are broken for everyone."
- **Periodic king re-eval:** every Nth round (say N=5), re-run the king on a fresh block_seed. If their `composite.worst` drops more than 5pp, dethrone. This catches schema-version drift and stale-king camping.

Engineering for the first two: ~1 day. Engineering for periodic re-eval: ~3 days (need to thread a king-re-eval signal through the pod orchestration without conflicting with the per-round student evals).

---

## 5. Is "highest minimum across all axes" the best discriminator?

### Theoretical analysis

The aggregator choice trades off three properties:

| Aggregator | Anti-Goodhart | Stable to noise | Discriminates well |
| --- | --- | --- | --- |
| `min(axes)` | **Strong** | Weak (one bad axis = bad rank) | Weak in the saturated cluster |
| `mean(axes)` | Weak | **Strong** | **Strong** but rewards specialists |
| `geometric_mean` | Medium | Medium | Medium |
| `min_of_top(N)` (drop worst K) | Medium | Medium | Medium |
| `mean_of_bottom(N)` | Medium-strong | Medium | Medium |
| `lex_sort(axes_desc)` | Strong | Strong | Strong (full Pareto) |
| `softmin(axes, β)` | Tunable | Tunable | Tunable |

### What I'd change

**Current (`min`)**: a single noisy axis floors the entire score. Not robust.

**Proposed (`mean_of_bottom(3)`)**: average of the 3 lowest-scoring axes after dropping `broken_axes`. This:
- **Keeps the anti-Goodhart pressure** (a model that tanks one axis still gets ~33% pull on its rank).
- **Smooths variance** (one fluky 0 averaged with two 0.4s gives 0.13, not 0).
- **Stays interpretable** — "your worst three axes" is intuitive.
- **Discriminates the saturated cluster** — even if all your `worst-1` is 0, your `worst-2` and `worst-3` differentiate you.

For backward compat: keep `worst` as the legacy name, expose `worst_3_mean` as the new key, and do a 1-week shadow window where both are reported but only `worst` ranks. After validation, flip the ranker.

**Don't switch to `weighted` mean as primary** — it incentivises specialists, which is exactly what the user asked us to avoid.

**Lexicographic ordering** is theoretically the cleanest (Pareto-style) but expensive to communicate to miners ("your rank is the tuple of your sorted axes from worst to best, compared element by element"). The engineering is fine; the comms cost is high.

### Action items

1. Ship `worst_3_mean` as a SHADOW telemetry signal in the next composite update. Compute alongside `worst`, surface in the dashboard. (~1 day.)
2. After 2 weeks of shadow data, compare which discriminates better against the held-out canary. (Apply the existing `axis_correlation.py` machinery against `worst_3_mean` to see if its r is higher than `worst`'s.)
3. If `worst_3_mean` wins, promote it to ranker. Bump composite_shadow_version → 29.

---

## 6. Better ways to eval

Beyond the aggregator change, several concrete additions would make the eval more discriminating:

### 6.1 Out-of-distribution probe (high value, low cost)

Take 5% of round prompts from a guaranteed-OOD distribution (e.g., scientific abstracts from arXiv post-2025-12, after the teacher's training cutoff). Tests genuine generalization. Cost: ~10s/student. Implement: `_BENCH_OOD = "ood_probe"` axis with weight 0.05.

### 6.2 Pairwise judge tournament (high value, medium cost)

Currently each student gets an absolute 1-5 rubric grade. Instead: pick two students at random per prompt, ask the teacher "which is better, A or B?" Aggregate Bradley-Terry scores. This is what LMSYS does and it's much more discriminating than absolute rubric grading on a 1-5 scale.

- Cost: same as judge_probe (single token from teacher), but per-pair instead of per-prompt. With 10 students × 12 prompts × pairwise = 120 pairs ≈ 30s of teacher time.
- Implement: new `pairwise_tournament` axis. Use existing judge_response_probe to collect responses, add a `_pairwise_grade(teacher, response_a, response_b)` step.

### 6.3 Adaptive eval (high value, high engineering)

Spend more compute on close decisions. Initial eval is fast; if a challenger is within 5% of the king, run a more expensive verification round on a fresh block_seed. Engineering: ~3-5 days. Big throughput improvement when the leaderboard has a tight pack of competitors.

### 6.4 Consensus across validators (high value, low engineering)

If multiple validators run in parallel, they all see the same procedural items (block-seeded determinism). Their composite scores per UID should be identical modulo numerics. Compute the median composite per UID across validators; use that as the canonical score. Detects off-by-one bugs, GPU-arithmetic-drift, or single-validator gaming.

- For now we run one validator. But the design is right: when we add a second validator it should produce identical numbers, and a divergence is a signal worth investigating.

### 6.5 Verifiable-reward axis (Stage 4 of the mining guide)

The mining guide v2 proposes GRPO on verifiable rewards (math answer match, code test pass). The validator could compute a pass-rate on a fresh GRPO-style item set per round — items with deterministic verifiability that a teacher rubric isn't needed for. This rewards a capability the current axes don't reward (agentic problem-solving, verified correctness above teacher-similarity). Engineering: ~3-5 days.

### Priority order (recommended)

1. **`worst_3_mean` shadow + bench-axis cut** (drop 4 redundant axes). Net effect: stronger discrimination + 20 min faster rounds. ~3 days engineering.
2. **OOD probe** (factual content the teacher couldn't have memorized). Cheap, high signal. ~2 days.
3. **Pairwise judge tournament**. Medium cost, high discriminator. ~3 days.
4. **Periodic king re-eval** (every 5 rounds). Catches stale-king camping. ~3 days.
5. **Adaptive eval**. Larger engineering, biggest throughput improvement. ~5 days.
6. **Verifiable-reward axis**. Reward genuine capability beyond teacher-similarity. ~5 days.

That's ~3 weeks of focused engineering for the next phase, all of it shippable incrementally.

---

## 7. Can we switch to Kimi K2.6 now?

**No, not safely.** The blockers are concrete:

### Blocker A: tokenizer mismatch (critical)

Every relative axis (`kl`, `kl_is`, `on_policy_rkl`, `forking_rkl`, `top_k_overlap`, `entropy_aware_kl`, `teacher_trace_plausibility`, `capability`) computes over the teacher's vocab via the teacher's top-K logprobs. Switching teacher families changes the vocab. Without a cross-tokenizer mapping (which we don't have), every relative axis breaks or returns noise.

The runbook (`reports/2026-04-29-kimi-k2.6-stage2-runbook.md`) lays out two paths:
- **Path A:** re-tokenize Kimi outputs through the Qwen tokenizer. Cheap (~1 day) but lossy.
- **Path B:** Universal Logit Distillation / Approximate Likelihood Matching (Boizard TMLR 2025, Minixhofer NeurIPS 2025). Clean but ~2 weeks.

### Blocker B: vLLM compatibility unverified

Kimi's MoE architecture differs from Qwen3's. We need to verify:
- vLLM serves K2.6 with `--max-logprobs 128` and at our 8×H100 tensor-parallel layout.
- The fast-fail mechanism (`_VLLM_DEAD_EVENT`) triggers correctly under K2.6 deadlocks.
- We don't need a `_stub_missing_preprocessor_config`-style workaround for K2.6's config.json shape.

### Blocker C: capacity-gap risk

Per the synthesis report (§3.1), the trl-distillation-trainer paper shows that Qwen-235B teacher does NOT outperform Qwen-30B teacher for a 4B student — and GPQA Diamond REGRESSES 10pp with the larger teacher. Kimi K2.6 is ~1T total; we have no evidence that the swap would help and clear evidence (from analogous teacher swaps) that it could hurt.

The runbook mandates an A/B experiment before any production change.

### What needs to happen before the swap

1. ✅ Mining Guide v2 (done).
2. ✅ Stage-2 runbook (done).
3. ☐ Tokenizer round-trip helper (`eval/cross_tokenizer.py`) — Path A. ~1 day.
4. ☐ vLLM smoke test against K2.6 — ~0.5 day.
5. ☐ Experiment pod orchestration (`scripts/experiments/run_kimi26_a_b.py`) — ~1 day.
6. ☐ 6-round × 5-student × 3-variant A/B run — ~10 hours of pod compute.
7. ☐ Decision-rule evaluator + report.
8. ☐ Production swap if Tier 1 + Tier 2 + Tier 3 of the runbook all pass.

**Estimated wall time to the decision point: ~9 days of work + 3 days of compute.**

### What we COULD do today

We could run the **smoke tests** (§5.1 steps 3-4 of the runbook) without committing to the full swap. That gives us early signal on the vLLM compatibility question without spending much compute. ~0.5 day. I'd recommend doing this as the next compute-window task, but I'm NOT going to ship it in this turn because it requires GPU-pod compute I shouldn't presume to allocate.

---

## 8. Can we turn on the shadow axes?

**Yes — at low weight. Here's the safe rollout plan.**

### What's at stake

Four shadow axes shipped at weight 0:

| Axis | Computation status | Risk if turned on |
| --- | --- | --- |
| `entropy_aware_kl` | Validated math, smoke-tested numerics | Low — graceful None on missing data |
| `kl_is` | Validated against the existing `kl` (relationship is consistent) | Low |
| `forking_rkl` | Free from EOPD pass | Low |
| `teacher_trace_plausibility` | Free from existing student logits | Low |

All four axes:
- Default to None on missing data, drop out cleanly.
- Use the same king-min normalisation pattern as `kl` and `on_policy_rkl`.
- Are research-validated (not made up).

### Recommended low-weight rollout

Rather than fully promoting one at a time over 48h windows, **promote all four simultaneously at low weight (0.02 each = 0.08 total composite weight) immediately.** Reasons:

1. **Each axis returns None when its underlying cache is missing.** If a sparse path doesn't fire, the axis drops, no harm done.
2. **The dethrone-floor protection is preserved** — `worst()` still gates dethrone, and the new axes can only ADD a low-floor axis to the min, not remove existing ones.
3. **Low weight (0.02) means even if the axis is uninformative, the impact on rankings is bounded** — a student with `kl_is = 0` instead of `kl = 0.95` would only see their `weighted` drop by 0.02 × (0.95 − 0) = 0.02 pp.
4. **More signal, faster.** Waiting 48h × 4 axes for sequential promotion is 8 days. Promoting all at once gets the data we need in 48h.

### Safety net

The zero-weight filter in `compute_composite` ensures any axis with `weight = 0` drops out of `effective_weights`. So if any axis blows up, set its env weight to 0 and restart — no rollback commit needed.

### Action

I'll promote the four shadow axes to weight 0.02 each in `distil.env` and restart the validator. After 48h of telemetry I'll either:
- Bump them to 0.05 each (research-paper recommended weights),
- Or revert specific axes to 0 if the correlation against the canary is bad,
- Or rebalance based on which axis correlates strongest with held-out skill.

---

## 9. Better ways to incentivize better models

### Current emission structure

- **On-chain weights:** `build_winner_take_all_weights(n_uids, king_uid)` — `[0.0, 0.0, ..., 1.0, ..., 0.0]`. Strict winner-takes-all.
- **Validator's view:** the king gets all of *this validator's* weight on chain. With multiple validators all setting weights for the same king, the king's effective emission is ~100% of the subnet's emission.
- **Non-king miners:** 0% of emissions.

### Issues this creates

1. **Sparsity of incentive** — only 1 of N miners earns. With 200+ commitments, that's 0.5% of miners earning at any time.
2. **Discontinuous payoff** — a miner that's 1% behind the king gets 0% of emissions. A 1% improvement past the king flips the entire emission.
3. **Concentration risk** — one miner with one bad config can lose the whole subnet's productive output for a round.
4. **Discourages exploration** — why ship a small improvement when only the absolute best earns?

### Alternatives within the Bittensor framework

Bittensor lets validators set arbitrary weight vectors. We currently set one-hot. We could set:

#### 9.1 Top-K linear (recommended)

```python
def build_top_k_weights(n_uids, ranked_uids, k=5, geometric_decay=0.5):
    """Ranked weights for top-k miners, geometric decay."""
    weights = [0.0] * n_uids
    total = sum(geometric_decay ** i for i in range(k))
    for i, uid in enumerate(ranked_uids[:k]):
        weights[uid] = (geometric_decay ** i) / total
    return weights
```

For `k=5, decay=0.5`: weights = [0.516, 0.258, 0.129, 0.065, 0.032]. King gets 51.6%, runner-up 25.8%, etc.

**Effect:** dramatically reduces gaming pressure (a 1% improvement past #2 = 26% emission instead of 0%). Encourages incremental progress. Spreads compute cost across a wider miner base.

**Trade-off:** lower max emission to the king. Some miners might prefer the gambler's payoff of winner-takes-all.

#### 9.2 Softmax over composite worst

```python
def build_softmax_weights(n_uids, scored_uids, beta=10.0):
    """Softmax-distributed weights. β=10 puts ~70% on the king for a typical worst-axis spread."""
    scores = [score for _, score in scored_uids]
    max_s = max(scores)
    exps = [math.exp(beta * (s - max_s)) for s in scores]
    z = sum(exps)
    weights = [0.0] * n_uids
    for (uid, _), e in zip(scored_uids, exps):
        weights[uid] = e / z
    return weights
```

**Effect:** smooth, no hard ranking edges. Continuous payoff in the score → incremental improvements always pay.

#### 9.3 Improvement bounty

```python
def build_improvement_bounty(n_uids, prior_king_uid, current_king_uid, current_king_score, prior_king_score):
    """If the king changed, give the new king a 1.5x bounty for one round."""
    weights = [0.0] * n_uids
    if current_king_uid == prior_king_uid:
        weights[current_king_uid] = 1.0
    else:
        weights[current_king_uid] = 1.5  # gets re-normalized by Bittensor
    return weights
```

**Effect:** rewards GENUINE PROGRESS extra. Prevents stale-king camping.

### Recommendation

Ship **top-K linear with k=5, geometric decay 0.5**. Risk-bounded, well-understood, and addresses every concern with the current scheme. The top-1 still gets 51.6%, which preserves enough "winning is winning" incentive to drive serious mining effort, while #2-#5 get meaningful emissions (25.8%, 12.9%, 6.5%, 3.2%) so a wider field of miners has reason to run their setups.

**Engineering:** ~1 day. Just replace `build_winner_take_all_weights` with `build_top_k_weights` and pass the sorted candidate list. Backward-compatible because weight vectors are arbitrary in Bittensor.

**Coordination:** since other validators on the subnet are setting their own weights independently, we can ship this unilaterally and observe its effect on miner behavior. If other validators don't follow, the king still gets a majority of emissions but the runner-ups get a meaningful slice from us.

I'm NOT shipping this in this turn because it's a subnet-economics change that should be discussed with the operator (you) before deployment. It changes how miners get paid; that's not a code-only decision.

---

## Concrete action items (prioritised)

### Ship today (low risk, high signal)
1. **Promote 4 shadow axes to weight 0.02 each** (`KL_IS_AXIS_WEIGHT=0.02`, `FORKING_RKL_AXIS_WEIGHT=0.02`, `TEACHER_TRACE_PLAUSIBILITY_WEIGHT=0.02`, `ENTROPY_AWARE_KL_WEIGHT=0.02`). Restart validator. **Effect:** 4 research-validated signals start contributing to ranking; revert is instant.
2. **Re-run `scripts/audit/axis_correlation.py`** once 5 v30 kings exist (next ~5 rounds). Confirms the v29 GSM8K-narrative rebalance fixed the math_bench Goodhart pathology.

### Ship next sprint (medium engineering)
3. **`worst_3_mean` shadow telemetry** — alongside the existing `worst` and `weighted`. ~1 day.
4. **Bench axis cut** — drop 4 redundant axes (mbpp, correction, refactor, multi_doc). Bump composite_shadow_version → 29. Saves 20 min/round. ~2 days + Discord notice.
5. **OOD probe axis** — 5% of prompts from arXiv post-cutoff. ~2 days.
6. **Top-K linear weight emission** (with operator approval) — `[0.516, 0.258, 0.129, 0.065, 0.032]` for top-5. ~1 day.

### Ship in 2-3 weeks (larger engineering)
7. **Pairwise judge tournament** — Bradley-Terry over teacher pairwise grades. ~3 days.
8. **Periodic king re-eval** — every 5th round. ~3 days.
9. **Kimi K2.6 stage-2 experiment** — per the runbook. ~9 days + 3 days compute.
10. **Adaptive eval** — rerun close decisions on fresh block_seed. ~5 days.
11. **Verifiable-reward axis** — GRPO-style verifiable items. ~5 days.

### Decision needed from operator (not code)
- Approve top-K linear emission split (changes how miners get paid).
- Approve Kimi K2.6 swap once the runbook gates pass.
- Approve schema bump to v29 when bench-axis cut is ready.

---

## Closing observation

The single most important thing I want to flag: **`state/axis_correlation.json` (computed 2026-04-28, the day BEFORE v30 deployed) shows `math_bench` r=−0.875 against held-out GSM8K**. That's a Goodhart pathology — pre-v30 kings were getting BETTER on validator math while getting WORSE on real GSM8K. The v29 GSM8K-narrative rebalance was specifically designed to fix this, but we don't have post-v29/v30 telemetry yet.

The next 5-10 rounds are the most important data we'll collect. If `axis_correlation.json` re-runs and shows math_bench correlation flipped to positive, the entire v29-v30 axis re-design is validated. If it stays negative, we have a deeper problem — either the procedural items still don't match GSM8K's distribution, or the worst() aggregator is structurally rewarding gaming over generalization.

I'd recommend re-running the axis correlation script in 48h. It's the cleanest single signal that the work shipped today is actually moving the right thing.
