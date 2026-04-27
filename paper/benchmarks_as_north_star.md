# Held-out display vs in-composite eval — clarifying the bench surface

**Status.** Ground-truth as of commit `b0e0ed9` / 2026-04-27.

## The confusion

Allan in `#ა・distil・97` (2026-04-27, 14:38 UTC):

> Even though you do some prompt shuffle, it will still pollute the train data of model. We can not know the real ability of models as you encourage model overfit on these benchmark.

Followed by:

> You should not use benchmark as evaluation and north star in the same time. You should use some math data as measurement, and the benchmark as north star.

The framing implicitly assumes **the validator scores miners on GSM8K / HumanEval / MBPP / BBH / IFEval**. That's not true and hasn't been since v27 (Session 3.20). The bot agreed with the literal framing, then constructed a 3-message "real solution" that demoted procedural bench axes — which would have been wrong, because Session 3.20 already did the equivalent of what Allan was reaching for.

This document records the actual surface so we don't get confused again.

## Two completely separate "bench" surfaces

There are **two** systems people call "benchmarks" and they share zero data:

### A. The validator's composite bench axes (`*_bench`)

- Live in `compute_axes()` in `scripts/validator/composite.py`.
- Generated **per round** by `_generate_*_items` in `scripts/pod_eval_vllm.py` from `block_seed`. The (problem, gold) pair is created on the fly each round and exists nowhere on disk.
- Cover: math, code, reasoning, knowledge, ifeval, aime, mbpp, tool_use, robustness, long_context.
- Used to compute `composite.worst` — the king-selection ranking key.
- Goodhart resistance: **no on-disk lookup table**. A miner with the entire public corpus pre-downloaded cannot score a `{problem_text → answer}` lookup against per-round-generated items.

What a miner can still do — and where the residual Goodhart vector sits — is overfit their model on the *distribution* the procedural generator samples from. If `_generate_math_items` produces problems that look like GSM8K (multi-step word problems with integer answers), a miner who fine-tuned on GSM8K will do well even though the specific problems are fresh. That's a softer form of overfitting (the model genuinely learned to solve that *kind* of problem) and is not the dataset-leak version Allan was describing.

### B. The dashboard's Bench tab (held-out auto-bench)

- Lives in `scripts/run_king_benchmark.py`.
- Runs **after** a round, on a separate Lium pod, against the king's served vLLM.
- Uses the **public** evalscope datasets: GSM8K, HumanEval, MBPP, AIME, BBH, MMLU-Pro, IFEval.
- Output goes to `state/benchmarks/uid_<N>.json` and surfaces in the dashboard's Bench tab.
- **Not** consumed by the validator. Has zero effect on `composite.worst`. Has zero effect on king selection. **Not in the composite. Not a metric.**

These numbers are the answer to "did the model the validator picked actually generalise?" — pure post-hoc transfer measurement on data the validator never touched.

## Why we keep both

1. **In-composite procedural bench axes (A)** are how we score miners. They use procedurally generated items so dataset memorisation can't saturate them.

2. **Held-out public-dataset auto-bench (B)** is our Goodhart canary. If `composite.worst` keeps climbing while held-out scores plateau or regress, the procedural axes are no longer measuring real capability — they're being optimised in ways that don't transfer.

The held-out display is the *external check*. The composite is the *ranking key*. Mixing them up is exactly the failure mode Allan was warning about, just inverted: he thought we were ranking on (B) and using (A) as the canary; in reality we're ranking on (A) and using (B) as the canary.

## What Allan's correct version of the concern is

Even with (A) + (B) separated as above, a miner can still:

1. Train on the public datasets in (B) directly. This raises the held-out display.
2. Train on the *kind* of problem (A) generates. This raises composite.worst.

Both are legitimate — a miner who actually makes their model better at math should do well on both. The Goodhart concern is a miner who games (A) without making the model better, which would show up as (B) regressing relative to (A). That's exactly the divergence we now watch for between `composite.worst` and the held-out tab.

The 2026-04-27 UID 160 → UID 123 king flip showed an early warning of this divergence (gsm8k -5pp, bbh -9pp on held-out while composite.worst stayed at 0.667). We flagged it in the `slab 4` Discord post for monitoring, not as cause to demote axes.

## What we are NOT doing (and why)

- ❌ **Demoting `BENCH_AXES_IN_COMPOSITE` / `ARENA_V3_AXES_IN_COMPOSITE`** to 0. Those axes are validator-internal and procedurally generated — they're the core of the Goodhart-resistant signal. Removing them would leave only `on_policy_rkl` + `kl` + `capability` + chat probes, which is a smaller, less binding composite.
- ❌ **Renaming the held-out tab to "north star"** — it's not a north star, it's a transfer-test canary. The model card targets ARE GSM8K / HumanEval / etc, sure, but those are aspirational signals, not what we score.
- ❌ **Building a separate "private benchmark suite"** — we already have one. It's the per-round procedural items in (A). Spending engineering on a third tier when (A) is in place is not the highest-value move.

## What we ARE doing (and why)

1. Documenting the distinction (this file).
2. Re-stating it on the dashboard's Docs panel + the Bench tab footer so miners aren't confused that the held-out scores are the ranking key.
3. Adding a one-pager to the Discord channel pointing here.
4. Updating the bot policy so it doesn't agree with the literal "GSM8K is in math_bench" framing on autopilot. The accurate response is "math_bench items are procedurally generated per round from `_generate_math_items`; GSM8K only shows up in the held-out auto-bench, which is not in the composite". (See `/root/.openclaw/agents/sn97-bot/SOUL.md` rule O5.)

## Credit

Thanks to Allan (`@allan_ww`) for raising the framing question. The literal version of the complaint doesn't hold against the v27/v28 procedural switch, but the *category* of concern is exactly right and worth being explicit about.
