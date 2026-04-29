# Tweet Drafts — v30.2/v30.3 Eval Overhaul

Six options in @arbos_born style: long-form note tweets, multi-paragraph, hook-first, concrete numbers, declarative.

Pick one to post, or mix and match.

---

## Option 1 — "The Goodhart problem in numbers"

> Pre-v30, SN97's `math_bench` had Pearson r = -0.665 against held-out GSM8K.
>
> Translation: validators got more confident in their math axis as miner models got worse at real math. Kings climbed the leaderboard while regressing on the actual benchmark we were trying to approximate.
>
> Goodhart's law, measured in nats.
>
> v30.2 ships the fix: a new ranking key `composite.final = 0.7 × worst_3_mean + 0.3 × weighted`, four bench-axis groups (code/math/reasoning/knowledge), a `super_teacher` axis that explicitly rewards beating the teacher, and 6 new research-backed signals (top-K overlap, importance-sampled KL, forking-token RKL, teacher-trace plausibility, entropy-aware adaptive KL, tail-decoupled KL).
>
> The miners who were optimizing the metric will see their scores move. The ones who were building good models won't.
>
> Live now: https://distil.arbos.life

**~830 chars.** Lead with the pathology, then the fix, then the implication.

---

## Option 2 — "Winner-take-all keeps. Worst-axis-min goes."

> SN97 just changed how the king is decided.
>
> Old rule: `composite.worst` — the single lowest axis sets your rank. 22% of the leaderboard sat at exactly 0 because one noisy axis was enough to floor everyone. The metric became a coin flip in the saturated cluster.
>
> New rule (v30.2): `composite.final = 0.7 × mean(bottom 3 axes) + 0.3 × weighted`. The bottom 3 still dominate (anti-Goodhart pressure stays), but a single fluky 0 no longer floors your entire score. Real progress wins again.
>
> The legacy `worst` is preserved as telemetry — the dashboard shows both. Only the dethrone gate changes.
>
> Winner-take-all stays. The scoreboard just got harder to game.
>
> Mining Guide v2 in the repo for the new playbook: github.com/unarbos/distil

**~770 chars.** The "stays / changes" rhythm matches Arbos's "old / new" cadence.

---

## Option 3 — "Pure distillation has a ceiling. We just made it crossable."

> Pure distillation has a ceiling: the teacher.
>
> If your student perfectly mimics Qwen3.6-35B, you tap out at Qwen3.6-35B. The real SOTA-class small models — DeepSeek-R1-Distill, Phi-4-Mini-Reasoning — all do something more: rejection-sampled SFT on harder data than the teacher saw, then GRPO on verifiable rewards.
>
> Until today, SN97's eval didn't reward that. Match the teacher and you got full marks.
>
> v30.2 ships the `super_teacher` axis (weight 0.10): rewards student.pass_frac > teacher.pass_frac on 16 verifiable benches via tanh(mean_lift / 0.10). Match teacher = 0. Beat by +0.20 ≈ 0.96. The path to the king now runs through Stage-4 GRPO and curated post-distillation SFT, not pure KL.
>
> Cap is 7B params. Teacher is 35B. There's room to swing above your weight class. The eval finally pays for it.

**~870 chars.** Most aspirational angle.

---

## Option 4 — "Six axes, six 2026 papers"

> SN97 v30.2 ships 6 new distillation signals, every one informed by a 2026 paper:
>
> • `top_k_overlap` — Rethinking OPD (arXiv 2604.13016): top-K agreement is the single most predictive signal of distillation success. Successful runs converge to 97-99% shared mass. Failed runs sit at 60-80%.
>
> • `kl_is` — Anshumann et al., ACL 2025: importance-sampled KL drops the renormalization bias of top-K cached KL. Tighter lower bound on the true full-vocab KL.
>
> • `forking_rkl` — Wang et al. 2025 / Thinking Machines OPD blog: reverse-KL only at top-quartile-teacher-entropy positions. Decision points carry the signal; the rest is filler.
>
> • `entropy_aware_kl` — EOPD (arXiv 2510.27485): adaptive RKL/FKL blend. +1.37 to +5.05 Pass@8 on Qwen3-{0.6B, 1.7B, 4B} math benchmarks vs vanilla OPD.
>
> • `tail_decoupled_kl` — Tail-Aware Distillation: catches "match teacher head, flatten the tail" SFT-only over-confidence.
>
> • `super_teacher` — capacity-gap aware lift over teacher; rewards exceeding the ceiling, not just matching it.
>
> Free to compute (all from the existing top-128 cache). Default weight 0.05 each except super_teacher (0.10) and top_k_overlap (0.10). Live now.

**~1140 chars.** Maximum technical credibility — name-checks every paper.

---

## Option 5 — "The eval is now a moving target — for the right reason"

> Every SN97 round, the validator regenerates ~150 bench items from the round's block hash. Math, code, reasoning, IFEval, AIME, knowledge, pragmatic theory-of-mind — all procedural. There is no static answer key on disk for miners to memorize, because there is no dataset on disk.
>
> The only way to climb is to learn the underlying skill.
>
> v30.2 doubles down: `super_teacher` axis explicitly rewards beating the teacher on those procedural items. `worst_3_mean` smooths single-axis noise so genuine progress is visible round-to-round. Group axes (code, math, reasoning, knowledge) replace per-bench fragmentation so wide-but-shallow doesn't beat narrow-but-deep.
>
> The eval moves every block. The right answer to that, for a miner, is to get better — not faster at memorizing.
>
> distil.arbos.life

**~830 chars.** Frames the procedural-eval design as the centerpiece.

---

## Option 6 — "What we shipped this week" (operational, builder-tone)

> Shipped on SN97 this week:
>
> • New ranking key: `composite.final = 0.7 × worst_3_mean + 0.3 × weighted`. The single-axis `worst()` is retained as telemetry but is no longer the dethrone gate. 22% of the leaderboard previously sat at exactly 0 — the new metric breaks that tie.
>
> • Bench axes collapsed into 4 skill groups (code, math, reasoning, knowledge) — sub-axes still computed for telemetry, group means drive ranking. Reduces ranking noise without dropping any measurement.
>
> • New axis `super_teacher` (weight 0.10): rewards student > teacher on 16 verifiable benches. Match teacher = 0; beat by +0.20 ≈ 0.96.
>
> • 6 research-paper-backed shadow signals at low weight: top-K overlap, IS-KL (unbiased), forking-RKL, teacher-trace plausibility, entropy-aware KL, tail-decoupled KL.
>
> • King is now re-evaluated every round on the same procedural items as challengers (paired-fairness fix). One eval per commitment for everyone else.
>
> • +50% bench items per round. Tighter axis-level standard error.
>
> Mining Guide v2, full release report, and per-axis playbook: github.com/unarbos/distil

**~1100 chars.** Most info-dense; matches the "we shipped X" cadence of his April 18 mechanism-change tweet.

---

## My recommendation

If you only post one: **Option 1** (the Goodhart-by-numbers angle). The `r = -0.665` opener is the kind of concrete-number hook that travels — same template as the "8 kings in 12 days" / "11 kings in 2 weeks" tweets that did well historically.

For maximum technical-audience pull (other ML folks, researchers): **Option 4** (the six-papers tweet).

For the existing miner audience: **Option 6** (the operational shipped-list, mirrors the Apr 18 mechanism-change tweet's structure).

If you want to thread, Options 2 → 3 → 4 → 6 → 1 chain naturally: hook with the saturation problem (2), pivot to the super-teacher incentive (3), back it with research (4), drop the changelog (6), close with the Goodhart pathology + invite (1).

Just say which one (or which thread) and I can post it via the API in the .env.
