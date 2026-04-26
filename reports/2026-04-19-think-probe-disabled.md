# Thinking-collapse probe disabled (2026-04-19)

## Summary

The teacher-anchored Wilson variant of the thinking-collapse probe
introduced in commit `8eec9a2` caused a multi-hour production outage.
It DQ'd every student in every round — including the reigning king
and the reference model `Qwen/Qwen3.5-4B`. Five consecutive rounds
produced zero valid scores. Fix: gate the probe off by default,
convert any surviving DQ path to telemetry-only.

## Timeline (UTC)

| time | event |
| ---- | ----- |
| ~01:04 | commit `8eec9a2` deployed to validator + pushed to pod for the round starting at block 7998671 |
| ~02:28 | `coffieex` posts `eval-progress` showing 4/4 completed students all flagged `thinking_collapse`, including base `Qwen/Qwen3.5-4B` and king `best26/sn97-best6` |
| ~02:32 | `coffieex`: *"Of course why you need thinking collapse? Too many false-positive"* |
| ~02:35 | `Arbos` bot commits publicly to disabling the probe in the next deploy |
| ~04:19 | `coffieex`: *"You DQ the king again"* — the next round also killed everyone, including the new king (UID 165) |
| ~04:39 | `leeroyjkin`: *"eval us still broken you aren't evaluating any model you need to fix"* |
| ~05:19 | `itorgov`: *"A new eval round has been started and UID 169 is missing again"* — stale failure counter side-effect |
| ~05:23 | `manta.llm`: *"Why just restart the current eval and deploy new script?"* |
| ~05:31 | `manta.llm`: *"Still not restart, what are you waiting for?"* |
| ~08:30 | probe disabled at the pod call-site (this commit), failure counters cleared |

## Why the probe failed

The probe's termination rule compares the student's Wilson lower
bound on the fraction of prompts that produced `<|im_end|>` within
the token budget against the teacher's Wilson lower bound, with a
small margin. Expected pattern: a well-distilled student matches
the teacher's termination rate to within the margin.

Observed pattern in production across 3 rounds:

- `teacher_term_lb` ≈ 0.79
- every student (including king + new kings + teacher itself)
  landed at `student_term_lb` ≈ 0.51–0.65
- so every student's lower bound fell below the teacher's by more
  than the margin → every student DQ'd

The reference model failing against the teacher's own anchor is the
diagnostic: the `teacher_samples` anchor was recorded at one
block/prompt seed and compared against the student running on a
different seed with different generation length budgets. The
comparison is not apples-to-apples and the probe is miscalibrated
for Qwen3.5-family reasoning models which legitimately emit long
chain-of-thought before EOS.

## What shipped

1. Opt-in env var: `THINK_COLLAPSE_PROBE` now defaults to **off**.
   The probe does not run, consumes zero GPU time, cannot DQ.
2. Telemetry-only: if someone sets the env var to `1` for offline
   analysis, a failure writes `would_have_dq: true` to the
   student's `think_probe` entry but does **not** set
   `status=thinking_collapse` or `kl_global_avg=inf`.
3. Failure-counter reset: `state/failures.json` cleared once
   (`/tmp/reset_think_probe_failures.py`). 26 UIDs were stale-locked
   at count ≥ 3 from the broken rounds; without a reset they would
   have stayed excluded from challenger selection.
4. Composite-score graceful degradation: the `degeneracy` axis of
   the shadow composite score derives from `think_probe` data. With
   the probe off, that axis returns `None` and the weighted mean
   renormalizes over the remaining 4 axes (on-policy RKL, KL,
   capability, length). Confirmed via smoke test.

## Not re-enabled until

- Calibration run where the teacher passes its own threshold on
  the actual round's prompts, not a cached probe sample.
- The chosen threshold must pass the king-of-the-moment with
  ≥95% confidence on 32 prompts.
- Code review of the Wilson anchor wiring: in the current code
  `teach_term_lb` comes from a cached sample, not from a fresh
  teacher run on the round's prompts.

## 2026-04-22 follow-up: length axis restored via chat_probe

The think probe stays off by default pending the calibration work
above, but the broader composite pipeline no longer silently drops
the length axis as a result. Previously, with
`THINK_COLLAPSE_PROBE=0`, the `length` axis in `composite.py` had
no data source and returned `None`, so every student's composite
score fell through to on-policy RKL + KL + capability only. A
rambling king passed KL and was invisible on the remaining axes.

Fix shipped today:

1. `prepare_teacher_probe_refs_{hf,vllm}` now also runs the
   teacher on `CHAT_PROBE_PROMPTS` (enable_thinking=False, four
   trivial prompts, ~200 tokens total) and caches the per-prompt
   token lengths as `_TEACHER_CHAT_PROBE_GEN_LENS`. Cost is
   negligible next to the ~300-prompt scoring pass.
2. The length-axis computation now falls back to the always-on
   `chat_probe.mean_gen_tokens` when `think_probe` data is
   absent, anchored on the teacher chat length above.
3. `scripts/validator/results.py::_composite_dethrone_veto`
   blocks a challenger from taking the crown when its composite
   worst axis is below 0.20 with ≥3 populated axes, even if it
   passes the KL paired t-test + 3% epsilon gate. Fails open if
   the composite isn't available — a pod probe outage can't
   freeze the crown.

The think probe itself is still off. It will only come back on
once the calibration criteria at the top of this section are
met; the new path measures the same pathology class (length
ratio under `enable_thinking=False`) without the Wilson-anchor
risk.

## References

- Public discussion: SN97 Discord channel `1482026267392868583`,
  ~04:19–08:30 UTC 2026-04-19
- Previous commit: `8eec9a2` (introduced the broken probe)
- This commit: disables it and documents the failure mode
- 2026-04-22 follow-up: adds chat-probe-based length axis and
  composite dethronement veto
