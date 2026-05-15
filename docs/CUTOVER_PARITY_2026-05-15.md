# Cutover parity report — `scripts/` → `distil/`

**Date**: 2026-05-15 · **Sample**: `state/incoming/round_20260514T053302Z/eval_results_merged.json`

## Top-line: distil composite scores **21 axes**; prod scores **49**

If you flip systemd to `distil` today, every student that earns their composite
on the 28 axes distil does **not** yet score will see a sharp drop. UID 30
(`ClarenceDan/sn97-kimi-r6-uid30-sta`) went from `0.4586` to `0.0000` in this
test because every axis it scored on lives in the prod-only column below.

## Per-student diff (5-student sample, real prod data)

| Student | distil final | prod final | abs diff | shared axes |
|---|---:|---:|---:|---:|
| `Foremost04/will_king_v2` | 0.5512 | 0.5019 | 0.0493 | 19/21 vs 49 |
| `moonshotai/Kimi-K2.6` | 0.8960 | (teacher, no record) | – | – |
| `arboskiller/fakearbos` | 0.5548 | (no record) | – | – |
| `ClarenceDan/sn97-kimi-r6-uid30-sta` | **0.0000** | **0.4586** | **0.4586** | 0/0 vs 41 |
| `slowsnake/hydra-7` | 0.5096 | 0.4783 | 0.0313 | 21/21 vs 45 |

## Axes — distil has but prod does not score this round

```
v31_ifeval_verifiable, v31_math_gsm_symbolic
```

Both axes exist in prod (`scripts/v31/`) but happen to be at weight 0 in the
running composite for this round.

## Axes — prod scores but distil does not

```
aime_bench           canary_bbh           canary_gsm8k         canary_humaneval
canary_ifeval        canary_mmlu_pro      code_bench           code_skill_group
correction_bench     debug_bench          entropy_aware_kl     forking_rkl
ifeval_bench         kl                   kl_is                knowledge_bench
knowledge_skill_group   long_context_bench    math_bench         math_skill_group
mbpp_bench           multi_doc_synthesis_bench  pragmatic_bench
reasoning_bench      reasoning_skill_group    refactor_bench    robustness_bench
tail_decoupled_kl    teacher_trace_plausibility    tool_use_bench
```

## Axes — both score (the safe baseline)

```
calibration_bench          capability                  chat_turns_probe
degeneracy                 judge_probe                 length
long_form_judge            long_gen_coherence          reasoning_density
top_k_overlap              v31_code_humaneval_plus     v31_consistency_paraphrase
v31_knowledge_multi_hop_kg v31_long_context_ruler      v31_math_competition
v31_math_robustness        v31_reasoning_dyval_arith   v31_reasoning_logic_grid
v31_truthfulness_calibration
```

## Implications

The `scripts/v31/` package was explicitly designed to **replace** the v30
axes (`scripts/v31/__init__.py` describes a promotion gate: v31 axes go from
SHADOW → PRODUCTION when their Pearson correlation with the corresponding
held-out canary on ≥ 4 paired UIDs exceeds 0.5). The intent IS for the v31
set to be the only composite axes once all 11 axes have passed the gate.

But the current production composite **still includes** ~30 v30 axes alongside
v31. A clean cutover requires one of:

* **(A) Wait for v31 promotion gate to complete** (per the design doc) and then
  cut over — `distil/` is already correctly weighting the v31 axes.
* **(B) Cut over now, accept the leaderboard shift**. The shift would push
  scoring strictly toward the procedurally-generated (memorisation-resistant)
  axes. UIDs whose score came from saturating v30 axes (e.g. UID 30) lose
  weight; UIDs that score on the procedural v31 axes are unaffected.
* **(C) Port the missing v30 axes into distil/eval/composite.py** so cutover
  is byte-identical. This re-introduces ~2 000 LoC and the legacy axes that
  v31 was designed to deprecate.

## Recommendation

**Hold the systemd flip** until either (A) the v31 promotion gate fires
upstream OR (B) the operator explicitly accepts the shift. Everything except
the `ExecStart` line is already wired so the cutover is a one-line config
change after that decision.

## How to repeat this test

```bash
python /tmp/parity_check.py
```

(Script at `/tmp/parity_check.py` reads the latest
`state/incoming/round_*/eval_results_merged.json` and runs both composite
engines on it.)
