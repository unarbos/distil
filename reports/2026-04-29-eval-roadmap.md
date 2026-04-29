# 2026-04-29 SN97 eval roadmap — what to ship next

This report enumerates the remaining capability gaps in the SN97
validator and prioritises the next 5-10 changes to keep moving toward
"models that overfit our eval are SOTA-quality 4B-class models." Each
item lists the gap, the proposed change, the leverage / cost balance,
and a rollback strategy.

## Context

After v29.2, we have:

* 9 procedural bench axes covering math / code / reasoning / ifeval /
  mbpp / aime / tool_use / long_context (multi-needle) / robustness
* 1 new debugging axis (debug_bench, fixes existing buggy code)
* 7 relative axes (kl, on_policy_rkl, capability, length, degeneracy,
  judge_probe, chat_turns_probe) and 1 efficiency axis (reasoning_density)
* Per-axis baseline-relative penalty (must beat or tie Qwen-4B-base)
* KL probe diversified with 20 % skill-style prompts (math, code, reasoning, IFEval-shape)
* `worst()` aggregation as primary ranking key; baseline-floor / Pareto / canary streak gates
* Per-axis correlation telemetry (`scripts/audit/axis_correlation.py`) — first run revealed pre-v29 math_bench at r=−0.875 with held-out gsm8k
* Per-template saturation telemetry (`scripts/audit/per_template_saturation.py`, v29.3)

Audit findings still open:

| Axis | Issue | Status |
|------|-------|--------|
| `capability` | 53 % saturated, weight 0.25 | **#1 priority** below |
| `mbpp_bench` | bimodal (45 % saturated AND 36 % dead) | needs medium tier |
| `aime_bench` | 47 % dead at 0 | needs easier tier |
| KL probe | climbmix prompts may still under-test math/code/reasoning | considered, partly addressed in v29.1 |
| `axis_correlation.json` | only n=5 paired data; 4 axes need more data | accumulates over time |

## Priority ranking

Each item gets `[leverage / cost]` where leverage is the expected
impact on validator-as-SOTA-proxy quality and cost is implementation
complexity (S / M / L). Items are roughly ordered by priority.

### 1. `capability` axis tightening — `[H / S]`

**Gap.** Capability is 53 % saturated at 1.0 with weight 0.25 — the
single highest-weight saturated axis. Half the leaderboard scores
perfect, so capability contributes zero discrimination at the top of
the board on a heavy-weighted axis. The procedural arithmetic /
number-theory / string-op kinds are too easy for 4B-class models.

**Change.** Reduce static-trivia portion (`CAPABILITY_PROBE_N` 12 → 4),
add 6-8 harder procedural kinds:
* `multi_step_arithmetic` — chained 3-4 operations, parens
* `seq_next` — pattern-recognition (Fibonacci-like, polynomial)
* `string_transform_chain` — multi-step string ops (reverse + uppercase + count)
* `list_op_chain` — multi-step list ops (filter + map + reduce)
* `counterfactual_arithmetic` — "if X is now Y, what's Z?"
* `nested_logic` — boolean evaluation with negations + parens

**Expected effect.** Mean drops from 0.90 to ~0.65, saturation from
53 % to ~15 %. The 0.25 weight unlocks discrimination among the top
half of the leaderboard.

**Rollback.** Revert `CAPABILITY_PROBE_N` and the new kinds via env.

### 2. `correction_bench` — iterative self-correction — `[M / M]`

**Gap.** SOTA workflow is read → hypothesize → run → see error → fix.
We test write-from-scratch (`code_bench`) and find-and-fix
(`debug_bench`), but not "given an error message, fix the code."
That's a real capability — parsing tracebacks, understanding test
failures, applying targeted edits.

**Change.** New axis. Item structure:
1. Procedural function with a bug (reuse `_generate_debug_items`
   templates as the source).
2. Run the buggy version against the tests; capture the AssertionError
   / TypeError / NameError trace.
3. Show the model: buggy code + the actual error trace + the failing
   test.
4. Model emits the corrected function. Sandbox grades against same
   tests.

**Expected effect.** Tests an SOTA-distinct skill that no current axis
covers. Adds 0.04 weight; complements `debug_bench`.

**Rollback.** `BENCH_CORRECTION_PER_ROUND=0`.

### 3. CoT (chain-of-thought) trace grading — `[H / L]`

**Gap.** Bench axes score final-answer correctness only. A model that
gets the right answer with broken reasoning gets full credit; a model
that gets the wrong answer due to a single arithmetic slip but had
clear reasoning gets 0. Real SOTA models have **coherent reasoning
traces**.

**Change.** Add a sub-axis on `math_bench` and `reasoning_bench`:
teacher grades the model's CoT for validity (1-5 rubric). Run on a
sample (4 items per round to control cost). Combined axis value =
`α × pass_frac + (1-α) × cot_quality_norm`, with `α=0.7` so
final-answer correctness still dominates but reasoning quality
contributes.

**Expected effect.** Bigger correlation with real SOTA capability.
Discourages "answer-only" memorisation and rewards models that
actually think.

**Cost.** ~1 extra teacher pass per CoT-graded item × 8 items × N
students ≈ +30 s/student. Not free.

**Rollback.** `COT_GRADING_ENABLED=0`.

### 4. Drop saturated capability templates surgically — `[M / S]`

**Gap.** Once `per_template_saturation.json` accumulates a few rounds
of v29.3 data, we'll see exactly which procedural templates inside
each axis are saturated / dead. Each saturated template is a free 1.0
contribution to whoever is in the round; each dead template is a free
0.0.

**Change.** After 1-2 weeks of v29.3 telemetry, run
`scripts/audit/per_template_saturation.py`, identify templates with
`saturation: saturated_high` and `n_obs >= 6`, and remove them from
the corresponding `_generate_*_items` kind list. Replace with harder
variants (`coin_change_min` could split into `coin_change_min_2` with
denominations not in `{1, 5, 10, 25}`, etc.).

**Expected effect.** Each axis becomes more discriminating without
adding new axes — surgical refinement based on data.

**Rollback.** Revert specific templates; per-template list is local.

### 5. `multi_doc_synthesis_bench` — knowledge integration — `[M / M]`

**Gap.** `long_context_bench` (post-v29.2) tests retrieval +
combination of needles in ONE document. Real SOTA models also handle
**multi-document synthesis**: given 3-4 short docs, answer a question
requiring info from 2+ of them.

**Change.** New axis. Each item: 3-4 procedurally-generated "fact
cards" (each card has a topic + named entity + one numeric or
short-string fact) + a question requiring info from 2 specific cards.
Distinct from `long_context_bench` because the cards are short
discrete documents rather than one long document.

**Expected effect.** Tests integration across discrete sources, a
real-world capability not covered.

### 6. Calibration / honest-hedging axis — `[H / M]`

**Gap.** SOTA models say "I don't know" when uncertain instead of
confabulating. We don't measure this — a model that hallucinates
fluent BS could currently pass our bench axes (it'd just be wrong).

**Change.** New axis: mix solvable + intentionally unsolvable items.
Reward both correct answers AND correct refusals (e.g., "no
solution exists" / "insufficient information"). Penalize confident
wrong answers. Grading: regex for refusal phrases + correct answer
matching + a "confidence calibration" sub-score.

**Expected effect.** Discourages confabulation, rewards epistemic
honesty.

**Caveats.** Refusal grading is tricky — easy to over-reward "I don't
know" on solvable items. Needs careful template design. Defer until
we have bandwidth for a 1-week design pass.

### 7. `refactor_bench` — preserve behaviour, improve form — `[M / L]`

**Gap.** SOTA coding includes refactoring (restructure without changing
behaviour). `code_bench` / `mbpp_bench` / `debug_bench` don't measure
this.

**Change.** New axis: given working code + a constraint ("no nested
loops", "single-pass", "<10 lines"), model produces a version that
passes the same tests AND meets the constraint. Constraint check via
AST analysis (count nested loops, count lines, etc.).

**Expected effect.** Tests a SOTA-distinct skill.

**Caveats.** AST-based constraint grading is fragile — easy to score 0
on a correct-but-edge-case fix. Defer until we have a clean grader
design.

### 8. Auto-cull saturated items per round — `[M / M]`

**Gap.** Even within a single template, some specific (parameter,
gold) instances are easier than others. We could detect this in real
time: if 100 % of round-N students get `(template=foo, params=P) →
gold=G` correct, that instance's parameter range is too easy and we
should bias future generations away from it.

**Change.** Track per-instance pass-rate during the round; for the
NEXT round, draw parameters from a distribution that down-weights
ranges where pass-rate hit ceiling and up-weights ranges where it was
discriminating. Adaptive procedural calibration.

**Cost.** Persistent state per-template per-parameter-bucket. Some
risk of overfitting the rebalancer to round noise.

### 9. Stronger "stretch baseline" — `[L / S]`

**Gap.** We use Qwen-4B-base as the floor (must-beat). We don't
currently track distance to a **stretch ceiling** like Qwen-4B-instruct
or Phi-4-mini. A miner could score "5 pp above base" without knowing
they're "10 pp below SOTA-4B."

**Change.** Optionally include a second reference model (e.g.,
Phi-4-mini) in every round, score it on every axis, surface the gap
in `composite.king_health.stretch_gap_pp` and dashboard. NOT used in
ranking — purely informational.

**Cost.** ~25 min extra wall-time per round (reference compute).

### 10. Reward distribution beyond king-only — `[L / L]`

**Gap.** Currently weights flow only to the king. Top-N miners get
nothing. This caps the incentive surface — a miner who's #2 forever
has no return.

**Change.** Top-K weight distribution: 50 % to king, 30 % split among
top-3, 20 % split among top-10. Encourages broader competitive
investment.

**Cost.** Architectural — touches subtensor weight setting, registration
flow, and miner FAQ.

## What's already shipped (post-v29.2)

* per-axis baseline-relative penalty (v29.1)
* KL probe skill diversification (v29.1)
* HF upload time tiebreaker for griefing (v29.1)
* `long_context_bench` multi-needle reasoning (v29.2)
* `debug_bench` axis (v29.2)
* per-axis correlation telemetry (v29.2)
* per-template saturation telemetry (v29.3 — wiring landed today,
  populates on next round)

## Recommended next 2-3 weeks

Week 1:
1. **Capability tightening** (#1) — quick win on a 0.25-weight axis.
2. **Wait for per-template saturation data** to accumulate, then surgical template prunes.

Week 2:
3. **`correction_bench`** (#2) — rounds out the coding capability suite.
4. **CoT trace grading** (#3) — lifts reasoning-axis correlation with held-out skill.

Week 3:
5. **Calibration axis** (#6) — needs design pass first.
6. **Multi-doc synthesis** (#5) — extends long-context coverage.

Big bets (defer until other items stabilize):
* CoT grading at scale (cost), `refactor_bench` (grader complexity),
  reward redistribution (architecture).
