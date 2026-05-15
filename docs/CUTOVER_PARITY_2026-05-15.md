# Cutover parity report — `scripts/` → `distil/`

**Date**: 2026-05-15 (revised) · **Sample**: `state/incoming/round_20260514T053302Z/eval_results_merged.json`

## Top-line: **99.84% composite parity, byte-identical weighted axis set**

Apples-to-apples, both engines fed the **same** student row produce composite
finals that agree to **< 0.16% absolute** on every non-erroring student:

| student | prod final | distil final | abs diff |
|---|---:|---:|---:|
| Foremost04/will_king_v2 | 0.5535 | 0.5520 | 0.0015 |
| arboskiller/fakearbos   | 0.5567 | 0.5553 | 0.0014 |
| slowsnake/hydra-7       | 0.5114 | 0.5101 | 0.0013 |
| slowsnake/halcyon-22    | 0.4876 | 0.4874 | 0.0002 |
| allforone111/nimbus-V72 | 0.5364 | 0.5350 | 0.0014 |
| best26/sn97-ls-v6       | 0.5228 | 0.5212 | 0.0016 |
| tom-jerry-603/distil4   | 0.5340 | 0.5346 | 0.0006 |
| const0312/real_king     | 0.5347 | 0.5333 | 0.0014 |

**Max abs diff: 0.0016 (0.16%)** · **Mean: 0.0012 (0.12%)**

All `worst_3_mean` values were **identical**. The residual diff comes from the
`reasoning_density` axis — see "Residual diffs" below.

## What I got wrong in the first draft of this report

The first draft of this doc claimed distil scored "21 axes" vs prod's "49"
and that a cutover would shift the leaderboard. **That framing was wrong.**

Prod records 49 axes per student, but **26 of them have weight = 0**. They
are telemetry-only: written to `eval_results_merged.json` for analysis but
never affect the composite, never appear on the dashboard, and never affect
king selection. The actual weighted-axis set in both stacks is identical:

```
prod weighted axes:   23
distil weighted axes: 23
set difference:       empty
```

| axis | weight |
|---|---:|
| on_policy_rkl | 0.39 |
| long_gen_coherence | 0.25 |
| judge_probe | 0.20 |
| long_form_judge | 0.20 |
| chat_turns_probe | 0.10 |
| top_k_overlap | 0.09 |
| v31_code_humaneval_plus | 0.08 |
| v31_math_gsm_symbolic | 0.06 |
| kl, capability, length, degeneracy, calibration_bench, reasoning_density, v31_math_competition, v31_reasoning_logic_grid, v31_long_context_ruler | 0.05 each |
| v31_reasoning_dyval_arith, v31_knowledge_multi_hop_kg, v31_ifeval_verifiable | 0.04 each |
| v31_math_robustness, v31_truthfulness_calibration, v31_consistency_paraphrase | 0.03 each |

(Sum = 2.03; the composite renormalises by total active weight at runtime.)

After fixing two default-weight drifts in `distil/settings.py`
(`chat_turns_probe` 0.14→0.10, `calibration_bench` 0.06→0.05) the weight
maps are byte-identical.

## The "26 telemetry axes" prod records

| group | axes | why kept |
|---|---|---|
| Legacy "bench" | math_bench, code_bench, reasoning_bench, knowledge_bench, ifeval_bench | Pre-v31 deterministic-item benches; superseded by v31 procedural axes |
| Skill groups | code_skill_group, math_skill_group, reasoning_skill_group, knowledge_skill_group | Aggregates over legacy benches |
| Arena-v3 | aime_bench, mbpp_bench, tool_use_bench, long_context_bench, multi_doc_synthesis_bench, pragmatic_bench, robustness_bench, refactor_bench, correction_bench, debug_bench | v3 expansion that never reached the composite |
| Canaries | canary_gsm8k, canary_humaneval, canary_bbh, canary_mmlu_pro, canary_ifeval | Held-out anchors used by the v31 promotion gate (offline analysis) |
| Alternative KL | kl_is, forking_rkl, entropy_aware_kl, tail_decoupled_kl, teacher_trace_plausibility | Experimental variants; on_policy_rkl won |

These are all `weight=0.0` and have been so for several months. They're
prod's "lab notebook": diagnostic data captured every round and used for
offline correlation analysis (e.g. validating that v31 procedural axes
track held-out canary scores). They never affect:

* the composite score
* king selection
* DQ decisions
* the leaderboard / dashboard
* chain weights

Distil intentionally **does not record them**. That's a feature, not a
gap — fewer side products to maintain.

## Residual diffs (0.16% max)

The one numerical source of diff between the two engines on the same input:

`reasoning_density` mixes pass-rate × token-budget across multiple benches.
Prod's target dict has **24 entries** (10 v31 axes + 14 legacy benches).
Distil's has **10 entries** (v31 axes only).

When feeding distil a row that contains BOTH v31 and legacy bench data
(which only happens for prod-produced rows), distil averages over fewer
points than prod does, hence the 0.01–0.02 axis-level diff that contributes
~0.001 to `final` (weight 0.05 × axis diff 0.02 = 0.001).

**In steady state** (distil end-to-end, no legacy benches in the row),
this drift disappears and the two engines produce the same number for the
same model.

## Recommendation

1. **Validator cutover is functionally safe** w.r.t. composite scoring. The
   max anticipated leaderboard shift on the day of cutover is < 1% across
   all surviving UIDs — within normal between-round noise.

2. **Three pre-cutover items remain** before the operator can flip
   systemd (these are not parity issues, they're feature gaps):
   * Per-axis DQ thresholds (`scripts/validator/results.py` → `distil/eval/results.py`)
   * Resume-on-attach when validator restarts mid-round
   * Activation-fingerprint history dedup across rounds

3. **API cutover is blocked** until the missing 30 prod routes are ported
   into `distil/api/routes.py` (dashboard frontend depends on them).

The "wait for v31 promotion gate" path described in the first revision of
this doc was based on a wrong reading of which axes carry weight. The
v31 promotion gate is real, but its target is only to make `canary_*`
axes redundant — they're already at weight 0 in the composite. The gate's
completion does not block the distil cutover.

## Repeat the test

```bash
python scripts/parity_check.py
```

Includes the same set of student diffs, axis overlap detail, and which
weighted axes each side knows about.
