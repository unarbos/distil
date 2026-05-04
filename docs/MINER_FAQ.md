# Subnet 97 (Distil) — Miner FAQ & Getting Started

## What is Subnet 97?

Distil is a Bittensor subnet where miners compete to distill knowledge from a large teacher model into smaller student models. The teacher is **moonshotai/Kimi-K2.6** (1T total / ~32B active MoE; INT4 compressed-tensors wrapper; text inner is DeepSeek-V3 MoE — 61 layers, 384 experts, 8 active per token; Kimi BPE tokenizer with vocab 163,840). Your job: produce the most faithful small model (**≤33B total params**, Kimi-family architecture), measured on a multi-axis composite that covers distribution match, capability against ground truth, conversational quality, generation discipline, and robustness to prompt rewrites.

> **Teacher swap (2026-05-02):** the previous Qwen3.5/Qwen3.6-35B-A3B teacher and the 5.25B/7B caps were retired in favor of Kimi K2.6 + 33B. The live source of truth for teacher / cap / vocab / architecture allowlist is [`frontend/src/lib/subnet-config.json`](../frontend/src/lib/subnet-config.json). Older numbers in this FAQ that mention 5.25B / 7B / Qwen3.5-4B are historical.

The ranking key is **`composite.final`** = α · worst-3-axis-mean + (1 − α) · weighted-mean of every axis (α = 0.7). KL is one of the axes, not the gate. A model that wins KL but loses on grade-school math, IFEval, or reasoning-density cannot take the crown. Winner takes all — the king gets 100% of emissions.

> **Heads up.** If you've miner'd here before and remember "lower KL = win", that framing is wrong under the current eval. The 2026-04-17 reasoning-spiral king (UID 107: 4096-token loops on `"Hi"`, strictly worse than the unfine-tuned 4B base on every reasoning bench) was the wake-up call. The composite, the on-policy RKL axis, and the `reasoning_density` axis exist specifically to close that gap. Read the axis-by-axis playbook below before training. See [`paper/off_policy_cot_collapse.md`](../paper/off_policy_cot_collapse.md) for the full diagnosis.

---

## Getting Started

### 1. Register on SN97

Register a hotkey on subnet 97 via the standard Bittensor registration flow (`btcli subnet register --netuid 97`).

### 2. Train Your Student Model

- **Architecture:** Must be **Kimi-family** — `KimiK25ForConditionalGeneration` (preferred) or the inner text-only `DeepseekV3ForCausalLM` (the same MoE backbone Kimi K2.6 uses internally) with the matching `model_type` in `config.json`. The current allowlist lives in [`frontend/src/lib/subnet-config.json`](../frontend/src/lib/subnet-config.json) under `architectures` — that file is the source of truth, not this FAQ.
  - ⚠️ Old Qwen3.5 / Qwen3.6 `*_5ForConditionalGeneration` archs are **no longer accepted** post-cutover.
- **Max total params:** **33B total** (not active — MoE tricks won't help; we sum every parameter that ships in safetensors).
- **Tokenizer:** Must be **identical to the teacher's tokenizer** — Kimi K2.6 BPE with `vocab_size=163,840`. Don't modify `tokenizer.json` / `tokenizer_config.json`. (The previous Qwen3.5 vocab of 248,320 is **wrong** under the Kimi cutover.)
- **No quantization:** bf16/fp16 only. GPTQ, AWQ, GGUF, INT4/INT8 are rejected. (Yes — even though the *teacher* ships an INT4 compressed-tensors wrapper. That wrapper is Kimi's, not yours to inherit.)
- **No custom code:** `.py` files in your repo (except `__init__.py`) will get you DQ'd
- **Format:** Safetensors required (no pytorch `.bin`-only models)

### 3. Upload to HuggingFace

Push your model to a **public** HuggingFace repo. It must stay public — private or deleted models get disqualified.

### 4. Commit Your Model

Submit your HuggingFace model repo via the commitment mechanism on-chain. 

**⚠️ Commitments are permanent.** One model per hotkey, forever. You cannot re-upload or swap models on the same hotkey. Choose carefully.

---

## How Evaluation Works (v30.2 / v30.3, live as of 2026-04-29)

Every round the validator pulls the set of new on-chain commitments and evaluates each one on a single GPU pod. The eval policy is:

- **One eval per commitment** for non-king miners — your commitment is scored exactly once after you commit, the absolute composite is stored.
- **King is re-evaluated every round** so the king's score reflects the SAME procedural items as challengers (paired-fairness; v30.2).
- The king is decided cross-round from stored scores.

Each student is scored on many **independent axes**; the leaderboard is ordered by the new **`composite.final`** ranking key, which blends the bottom-3-axis mean with the weighted-axis mean. Gaming any single axis pulls your rank down, but a single noisy 0 doesn't floor your entire score (v30.2 fix). The design goal is simple: **if you overfit our eval, you will accidentally produce a SOTA small model**. Every axis points at a real, held-out capability.

**The round itself:**

1. **~300 KL prompts** per round sampled from [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) + a private skill-targeted prompt pool (math/code/reasoning, ~20% mix), seeded by the current block. The teacher generates continuations through vLLM (`temperature=0.7, top_p=0.9`); top-128 logprobs are cached for sparse KL.
2. **Procedural bench items** generated fresh per round from `block_seed`: math (20), code (14), reasoning (14), aime (12), mbpp (14), ifeval (14), debug (10), knowledge_v2 (12), pragmatic (12), long_context (10), tool_use (10), robustness (10), calibration (10), correction (10), multi_doc (10), refactor (8). Plus judge prompts (16 short + 6 long-form essay) and chat-turns conversations (6 × 3-turn).
3. **One eval per commitment** for non-king miners — the validator stores `composite.final`, all axes, and the commit signature in `composite_scores.json`.
4. **King paired re-eval each round** (v30.2 fix): the king is forced into the round's eval set so its score is on the same procedural items as challengers. Their `composite_scores.json` record overwrites each round.
5. **Cross-round dethrone gate**: a new commitment dethrones the king when its `composite.final` exceeds the king's by `SINGLE_EVAL_DETHRONE_MARGIN` (default 3%).
6. **Reference baseline.** A small dense Kimi-compatible reference model (currently a small DeepSeek-V3-text variant from the allowlist; consult `subnet-config.json` for the live `referenceModel`) is included in every round as UID `-1` for the per-axis baseline-relative penalty + axis-floor anchoring. Not a contender. (Pre-cutover this slot was undistilled `Qwen/Qwen3.5-4B`; the historical KL ranges in this FAQ refer to that era.)
7. **Winner takes all** — the king gets 100% of emissions on chain.

**Implication for miners.** Pick your weights carefully before you commit. The on-chain registration burn is the price of an evaluation: there is no "re-roll the same commitment until variance lands well." A model that scores 0.42 final stays at 0.42 forever (until you push a new commitment to the same hotkey, which fully overwrites the previous record).

### The ranking key — `composite.final` (v30.2)

```
composite.final = α × worst_3_mean + (1 - α) × weighted
```

Default `α = 0.7`. So 70% of your score comes from the **mean of your 3 lowest non-broken axes**, and 30% comes from the **standard weighted mean of every axis**. This:

- **Smooths single-axis noise** — one fluky 0 averaged with your other low axes still gives meaningful score, vs the legacy `worst()` (single-axis min) that floored you to 0.
- **Preserves anti-Goodhart pressure** — 70% of the score is still "your worst axes", so you can't camp specialists.
- **Rewards all-around competence** — the 30% weighted contribution stops you being penalised by a single quirky sub-axis floor.

The legacy `composite.worst` (single-axis min) is still **emitted as telemetry** in the API + dashboard, but it's no longer the dethrone gate.

### The axes (organised by ranking-relevance)

All axes are in `[0, 1]`, higher-is-better. Missing axes (e.g. probe outage) are dropped and the weighted mean renormalizes over surviving axes. Each axis drops if the teacher itself fails a sanity floor (so a miscalibrated probe can't corrupt rankings).

#### Skill-group axes (v30.2 — collapse without losing depth)

These are the primary ranking drivers for the bench-correctness side of the composite. Each group is the **mean of its sub-axes** (excluding broken ones); sub-axes still run for telemetry but no longer directly carry weight.

| Group axis              | Weight | Sub-axes (still computed, weight 0)                                                |
|------------------------|--------|------------------------------------------------------------------------------------|
| `code_skill_group`      | 0.20   | code_bench, mbpp_bench, debug_bench, correction_bench, refactor_bench               |
| `math_skill_group`      | 0.18   | math_bench, aime_bench, robustness_bench                                            |
| `reasoning_skill_group` | 0.12   | reasoning_bench, multi_doc_synthesis_bench, long_context_bench                      |
| `knowledge_skill_group` | 0.07   | knowledge_bench (v2 procedural fact reasoning), pragmatic_bench (theory-of-mind / scalar) |

#### Beyond-teacher axis (v30.2)

| Axis            | Weight | What it rewards                                                                         |
|-----------------|--------|-----------------------------------------------------------------------------------------|
| `super_teacher` | 0.10   | `tanh(mean(max(0, student_pass - teacher_pass)) / 0.10)` over 16 verifiable benches.   |

A student that exactly matches the teacher scores 0; one that beats teacher by ~0.20 scores ~0.96. **This is your incentive to apply Stage-4 GRPO and post-distillation SFT** — pure distillation cannot exceed teacher capability.

#### Teacher-similarity axes (production)

| Axis              | Weight | What it measures                                                                        |
|-------------------|--------|-----------------------------------------------------------------------------------------|
| `on_policy_rkl`   | 0.35   | Reverse-KL under YOUR sampling. The single-largest weight. Stage-3 OPD dependent.       |
| `kl`              | 0.05   | Forward-KL on teacher continuations, top-128 sparse renormalised. Saturated; demoted.   |
| `top_k_overlap`   | 0.10   | `\|top_K_t ∩ top_K_s\| / K` averaged over generated positions. v30 research-validated.   |
| `capability`      | 0.10   | Verifiable arithmetic / yes-no / one-word factual probes vs teacher.                    |

#### Shadow distillation axes (v30/v30.1/v30.3 — research-validated, low weight)

| Axis                            | Weight | What it measures                                                                  |
|---------------------------------|--------|-----------------------------------------------------------------------------------|
| `kl_is`                         | 0.05   | Anshumann ACL 2025 **importance-sampled** KL (unbiased full-vocab from top-K).    |
| `forking_rkl`                   | 0.05   | Reverse-KL only at top-quartile-teacher-entropy positions (Wang et al. 2025).    |
| `teacher_trace_plausibility`    | 0.05   | Mean NLL student assigns to teacher's emitted tokens. Catches LIMO/s1 failures.  |
| `entropy_aware_kl`              | 0.05   | EOPD adaptive RKL/FKL blend (arXiv 2510.27485, +1.37 to +5.05 Pass@8).            |
| `tail_decoupled_kl`             | 0      | (SHADOW) Tail-mass KL contribution. Catches "match head, flatten tail" pathology. |

#### Quality axes

| Axis                | Weight | What it measures                                                                |
|---------------------|--------|---------------------------------------------------------------------------------|
| `judge_probe`       | 0.15   | Teacher rubric on 16 short prompts (1-5 → [0,1]).                                |
| `long_form_judge`   | 0.05   | Teacher rubric on 6 long-form essay prompts (300-500 word, structure/depth).    |
| `chat_turns_probe`  | 0.08   | 3-turn dialogue coherence; teacher rubric on full transcript.                   |

#### Stand-alone capability (kept separate from groups)

| Axis                | Weight | Why it's separate                                                                  |
|---------------------|--------|------------------------------------------------------------------------------------|
| `tool_use_bench`    | 0.06   | Agentic Python (model emits `<python>...</python>`, stdout spliced back).         |
| `ifeval_bench`      | 0.07   | Instruction-following with structural constraints; orthogonal to content skill.   |
| `calibration_bench` | 0.06   | Solvable + unsolvable mix; rewards correct refusal. Catches confabulation.        |

#### Discipline

| Axis                | Weight | What it measures                                                              |
|---------------------|--------|-------------------------------------------------------------------------------|
| `length`            | 0.05   | Generation length ratio vs teacher. Rambling models lose here.                |
| `degeneracy`        | 0.15   | Termination + non-degenerate + self-BLEU. 1.0 = teacher-like.                |
| `reasoning_density` | 0.05   | `pass_frac × length_bonus` averaged across benches. Penalises both over-think AND wrong-but-short. |

**Teacher-similarity axes** (normalized against the king/teacher, weight 0.60 total):

| Axis              | What it measures                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------------|
| `kl`              | Teacher-forced KL divergence on teacher continuations. Anchored to the best (lowest) KL seen this round.   |
| `on_policy_rkl`   | Reverse KL under **your** sampling. Catches "matches teacher logits but collapses under free generation".   |
| `capability`      | Verifiable prompts (arithmetic/yes-no/one-word factual). `min(frac/teacher_frac, frac/0.6)` — absolute floor prevents winning by echoing teacher mistakes. |
| `length`          | Student generation length vs a teacher anchor. Rambling models lose here. 1.0 when you match the teacher.  |
| `degeneracy`      | Termination fraction + MAD-z-scored repetition + cross-rollout Self-BLEU. 1.0 = teacher-like.             |
| `judge_probe`     | Teacher (Kimi K2.6) rates your response on a 1-5 rubric, rotated to 16 prompts/round. Normalized to [0,1]. |

**Absolute-correctness axes** (scored vs ground truth, weight 0.45 total):

| Axis                | Dataset + probe behavior                                                                                     |
|---------------------|--------------------------------------------------------------------------------------------------------------|
| `math_bench`        | GSM8K + MATH-500 (~1820 items), 4/round. Boxed-integer extraction + numeric equality (±1e-3).               |
| `code_bench`        | HumanEval (164 items), 2/round. Function synthesized from prompt + test list, run in a subprocess sandbox.   |
| `reasoning_bench`   | BBH (21 objective subtasks, ~5250 items), 4/round. Multiple-choice or exact-match per subtask.              |
| `knowledge_bench`   | MMLU-Pro (12032 items), 4/round. Letter extraction.                                                          |
| `ifeval_bench`      | IFEval filtered to ~240 train items, 4/round. Runs Google's instruction-following verifier battery.          |

**Arena v3 Session 3 — LIVE as of 2026-04-25 (weight ~0.35 total):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `aime_bench`                 | AIME25 + AIME2024 (~90 olympiad items), 4/round. Boxed-integer extraction.                              |
| `mbpp_bench`                 | MBPP+ (378 items), 2/round. Sandboxed test-list execution.                                              |
| `tool_use_bench`             | Math items with an injected Python REPL. Model emits `<python>…</python>`, stdout spliced back into a 2nd generation pass, final boxed answer scored. Rewards agentic capability. |
| `self_consistency_bench`     | Hard math, K=5 samples at T=0.7 each, majority vote on the boxed answer. Rewards underlying knowledge vs one-shot luck. |

**Arena v3 Session 3.1 — LIVE, added 2026-04-25:**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `arc_bench`                  | AI2 ARC-Challenge (~1172 grade-school science items), 8/round. Letter-choice MC, completely disjoint from MMLU-Pro/BBH. |

**Arena v3 Session 3.2 — LIVE, added 2026-04-25 (addresses "models over-think simple questions"):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `reasoning_density`          | `pass_frac × length_bonus` averaged across benches, where `length_bonus = 1.0` if `mean_gen_tokens_correct ≤ target` (e.g. knowledge ≤30 tok, math ≤400 tok) and decays with `1/(1+ratio−1)` above target. Penalizes both over-thinking trivia AND verbose-but-wrong answers. Cannot be gamed by short-wrong: pass_frac=0 → axis=0. |

**Arena v3 Session 3.3 — LIVE, added 2026-04-25 (multi-turn coherence):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `chat_turns_probe`           | 6 hand-authored 3-turn dialogues/round. Student generates 3 assistant turns with accumulated context; teacher grades the full transcript on a 1-5 rubric (coherence + consistency + helpfulness). Directly probes deployment-quality multi-turn dialogue — a capability pure climbmix-KL distillation does NOT reward. |

**Arena v3 Session 3.4 — LIVE, added 2026-04-25 (adversarial factuality):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `truthful_bench`             | TruthfulQA mc1 (~817 items), 6/round. Adversarial factual questions where the popularly-believed-but-wrong answer is included as a tempting distractor. Tests hallucination resistance. Correct letter is deterministically shuffled per item so a model can't win by always answering "A". |

**Arena v3 Session 3.5 — LIVE, added 2026-04-25 (long-context retrieval):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `long_context_bench`         | Procedural needle-in-haystack over ~1400 tokens (tunable), 4/round. Items are *generated fresh every round from `block_seed`* — there is no dataset to memorize. Each item inserts a single needle sentence (e.g. "The lost vault combination is 4ESGKG3.") into a document of 40 distractor sentences and asks the student to recall the needle. Tests whether the model actually reads its input window instead of leaning on priors. |

**Arena v3 Session 3.6 — LIVE, added 2026-04-25 (procedural private-style eval):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `procedural_bench`           | Block-seeded synthetic arithmetic, instruction-following string transforms, invented-fact retrieval, tabular aggregation, and constraint filtering, 6/round. Template order is block-shuffled and there is no static dataset; grading is strict exact-answer, so overfitting means learning the transformations and concise output discipline. |

**Arena v3 Session 3.7 — LIVE, added 2026-04-25 (paraphrase + noise robustness):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `robustness_bench`           | Same items as `math_bench` (drawn under an independent stream offset, so usually different items in the same round) but each is asked under K block-rotated paraphrase wrappers. The wrapper set rotates per `block_seed`, so a model that memorizes the canonical wording of public math items passes `math_bench` and fails this one. Pure string transforms — no extra LLM call — so it's cheap and deterministic. |
| `noise_resistance_bench`     | Sibling axis to `robustness_bench`. Same math pool, yet another independent stream offset (so its sampled items are usually disjoint from both `math_bench` and `robustness_bench` in the same round), but the wrappers are *adversarial input noise* — keyboard typos at low rate, case jitter, distractor chatter, common misspellings (`the→teh`), extra whitespace, dropped sentence-period — instead of semantic paraphrase. Wrappers never touch digits or operators, so the math is preserved. Catches models that break under realistic chat noise — a brittle model that aces clean public benchmarks but loses 30% under typos has bad UX and won't generalize. |

All bench pools rotate per-round via `block_seed`, so every validator picks the same items but items differ between rounds (anti-memorization).

### Dethrone gates (all must pass)

1. **Final-score margin (v30.2).** Your single eval's `composite.final` must exceed the king's stored `composite.final` by `SINGLE_EVAL_DETHRONE_MARGIN` (default 3%). 0.50 → 0.515 is not enough; 0.50 → 0.52 is. (Legacy v28-and-earlier records that lack `final` fall back to the old `composite.worst`-based rule.)
2. **Worst-axis floor.** If `composite.worst < COMPOSITE_DETHRONE_FLOOR = 0.20`, the dethrone is **vetoed** even if the margin passes — unless the king-canary streak is active (king regressed on held-out gsm8k/humaneval/bbh/ifeval for 2+ consecutive rounds), in which case the floor is waived.
3. **Per-axis baseline-relative penalty (v29.1).** Each bench axis where you regress below the same-round reference baseline (UID -1, Kimi-compatible reference under the post-cutover allowlist; previously Qwen3.5-4B-base) is docked by `1.5 × (ref - your_score)`. So a 10pp regression below base costs you 25pp on that axis (10pp raw + 15pp dock). This makes "stay above base on every axis" the dominant strategy.
4. **Pareto-dominance gate.** A challenger that wins on `composite.final` but loses to the king on a majority of comparable axes is blocked. Soft Pareto: majority win AND `n_wins ≥ n_losses`, with a 2% noise margin. Insufficient comparable axes fails open.

---

## What to train for — axis-by-axis playbook

The fastest way to climb Arena v3 is to broaden your distillation data mix so the model covers every axis, not just KL. Each axis below lists what it rewards and what to add to your training.

| Axis                         | What helps                                                                                            |
|------------------------------|--------------------------------------------------------------------------------------------------------|
| `kl`, `on_policy_rkl`        | Reverse-KL under student sampling, not forward-KL on teacher rollouts. Thinking Machines "On-Policy Distillation" (Nov 2025); GKD (Agarwal et al. 2024); MiniLLM (Gu et al. 2023). |
| `capability`                 | SFT mix with verifiable arithmetic + factual + yes/no prompts alongside distillation.                 |
| `length`                     | Don't emit long `<think>` chains on trivial prompts. Teacher truncation behavior is your target.      |
| `degeneracy`                 | Long-context training with teacher-forced repetition penalties. Avoid small-LR dropout training.      |
| `judge_probe`                | Instruction-following + helpfulness data (OpenAssistant, UltraFeedback, LMSYS). Short correct > long verbose. |
| `math_bench`, `aime_bench`   | GSM8K + MATH + AIME + Maxwell-Jia in your mix. For AIME, chain-of-thought traces from Qwen2-Math or DeepSeek-R1. |
| `code_bench`, `mbpp_bench`   | HumanEval + MBPP + CodeAlpaca. Train on function-level synthesis not repo-level refactors.            |
| `reasoning_bench`            | BBH training split + FLAN + CoT datasets.                                                              |
| `knowledge_bench`            | MMLU train + TriviaQA + Wikipedia QA. MC-letter outputs specifically.                                 |
| `ifeval_bench`               | Alpaca-Instruct + SuperNaturalInstructions + IFEval train. Teach explicit-format obedience.           |
| `tool_use_bench`             | Function-calling / tool-use datasets (Gorilla, ToolBench, APIBench). Teach the model to emit code when compute is useful and parse stdout. |
| `self_consistency_bench`     | Robust CoT + majority-vote SFT. Temperature-robustness matters — if your model is 80% at T=0 but 30% at T=0.7, this axis will drop you. |
| `arc_bench`                  | Science MC (grade-school to middle-school). AI2 ARC-Challenge train + Easy splits make strong pretraining data; anything teaching MC letter outputs (A/B/C/D) generalizes. |
| `reasoning_density`          | Train your model to emit short correct answers on trivia and medium-length on reasoning. Use the teacher's own output length as the target (the `RD_*_TARGET` values). Long-CoT on `knowledge_bench` or `arc_bench` is strictly worse than short-CoT. |
| `chat_turns_probe`           | Multi-turn SFT (OpenAssistant Conversations, ShareGPT, UltraChat, LMSYS-chat-1M). Teach the model to reference its own earlier turns when asked ("based on your last answer…"). A model that resets context every turn will score ~2/5. |
| `truthful_bench`             | Hallucination-resistance data: TriviaQA-factual (short, gold-referenced answers), RefuseElseFalse, HaluEval-sft, the TruthfulQA train split (CC-BY). Teach the model to prefer precise short factual answers over confident-sounding prose. Avoid training data with speculative "facts" that aren't in the teacher's cutoff. |
| `long_context_bench`         | General-purpose long-context retrieval data: RULER, NeedleBench, long-context SFT derived from books/Wikipedia (e.g. QuALITY, NarrativeQA), or anything in the 2k–16k-token range that forces the model to answer from document content rather than priors. Aggressive 4-bit quantization and LoRA-only training break long-context attention — if you're shipping either, verify this axis before dethrone attempts. |
| `procedural_bench`           | Exact-answer synthetic tasks: arithmetic from records, deterministic string transforms, retrieval from invented registries, table aggregation, and multi-condition filtering. Train short deterministic outputs, not essays; verbose answers that merely contain the right value can fail this axis. |
| `robustness_bench`           | Generalization under prompt paraphrase. The defense is: train on math problems with diverse wordings (mix gsm8k / math500 / Maxwell-Jia / KhanAcademy with paraphrase augmentation, or just shuffle prefixes/postfixes during SFT). A model that only sees one wording per problem will fail when the wrapper changes. If `robustness_bench` lags `math_bench` by 0.20+ on your dashboard, you're memorizing canonical wordings, not solving. |
| `noise_resistance_bench`     | Generalization under surface noise (typos, case jitter, distractors, misspellings). The defense is: include noisy / chat-style training data, or apply augmentation at SFT time (random typos at 1-2%, random case flips at 3-5%, occasional distractor sentences before/after the problem). A model that gets near-perfect on `math_bench` but drops sharply on `noise_resistance_bench` is overfit to clean text — it'll be brittle in real chat. If both `robustness_bench` and `noise_resistance_bench` lag `math_bench`, you have a *general* canonical-wording problem; if only `noise_resistance_bench` lags, your training mix lacks chat-style messy text. |

**Three anti-patterns to avoid:**

- **Pure KL overfitting.** Matching teacher logits perfectly but failing on grade-school math means your composite worst is low. You cannot take the crown. KL is 0.15 of the relative tier, and the relative tier is itself one of five concerns the composite covers.
- **Long rambling / reasoning spiral.** `length` + `judge_probe` + `degeneracy` + `reasoning_density` all penalize verbose thinking-without-answering. Teacher-style brevity wins. Past kings have been retroactively DQ'd for failing the `thinking_collapse_probe` (looping on trivial prompts like `"Hi"` or `"largest planet one word"`). See `paper/off_policy_cot_collapse.md`.
- **Memorising canonical wordings.** `robustness_bench` re-asks math items under K block-rotated paraphrases + noise wrappers. A model that aces clean public benchmarks but loses 30% under typos will fail this axis.

**Watch your dashboard columns:** `Worst / Weighted / Judge / Bench / V3 / Pareto / vs King`. These are live. The single weakest axis is your ranking key — a high KL score never compensates for a 0.0 anywhere else.

---

## Training Tips

- **Base model:** Start from a Kimi-family checkpoint that fits the 33B cap — e.g. a small DeepSeek-V3-text variant from the allowlist, or one of the public small Kimi-K2.x text-only releases. Always re-verify the architecture string against [`subnet-config.json`](../frontend/src/lib/subnet-config.json) before committing; the allowlist is the gate, not this FAQ.
- **Objective:** Optimise for `composite.final` (worst-3-mean blended with the weighted mean). KL(teacher ‖ student) is one of many axes — useful but never sufficient. A pure-KL model loses to a slightly-worse-KL model that also answers GSM8K correctly, doesn't loop on `"Hi"`, and survives prompt paraphrase.
- **Data mix:** at minimum combine ClimbMix-style distillation data with ~10–20% instruction/reasoning/code data (see the playbook above). Miners who run SFT + DPO on top of their distillation have been climbing the bench axes fastest.
- **Long completions matter:** eval uses `max_new_tokens=8192`. The model needs to terminate naturally on simple prompts and reason coherently on long ones.
- **Temperature:** vLLM runs at `temperature=0.7, top_p=0.9` with per-prompt seed `block_seed + prompt_idx`. Deterministic per round, rotating between rounds. Greedy (temp=0) only applies to local dev runs without `--block-seed`.
- **Don't modify the chat template:** it's checked against the reference Kimi K2.6 template hash. Injected comments or modifications = DQ.
- **Bench probes run offline.** All datasets are pre-cached on the pod (`HF_HUB_OFFLINE=1`). No network-dependency required in your model.

---

## One-eval policy (single-eval mode)

This subnet enforces **one registration → one commitment → one eval**. The implications are practical:

- **Your commit is your shot.** Don't commit a half-trained checkpoint expecting to climb later — the validator will not re-evaluate the same `(model, revision)` pair on the same hotkey.
- **A new commitment overwrites your record.** If you push a new HuggingFace revision (or a different repo) and re-commit on-chain, the validator detects the change, evicts your previous composite record, and schedules a fresh single eval. The dethrone-floor and Pareto gates still apply to the new score.
- **No more "rotation luck."** Earlier sessions cycled top-N + dormant UIDs through periodic re-evals. Single-eval mode kills that loop entirely; rounds only contain commitments without a stored composite. If you've been scored once, you stay at that score until you re-commit.
- **Round cadence.** Rounds are short (target < 60 min) because the active set is just "everyone who hasn't been scored on their current commitment yet". A round with no new commitments is a no-op (king retains crown, weights unchanged).
- **King floor telemetry.** If the sitting king's stored composite drops below the configured floor or below the reference Qwen baseline, the dashboard surfaces a warning so the network can react publicly. The king is not auto-demoted on this signal — only a successful single eval that clears the dethrone gates can change the crown.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Wrong architecture" DQ | `config.json` has an arch that's not on the live allowlist (e.g. legacy Qwen3_5*) | Set `architectures` to one of the allowlisted Kimi-family arches in [`subnet-config.json`](../frontend/src/lib/subnet-config.json) (currently `KimiK25ForConditionalGeneration` or the inner `DeepseekV3ForCausalLM`) with the matching `model_type`. No weight changes needed if the underlying topology already matches; otherwise retrain. |
| "Integrity check failed" | HF repo deleted, made private, or otherwise unreachable since the validator first hashed it | Make the repo public and re-upload the same weights — the integrity DQ clears next epoch when the validator can re-verify. The on-chain commitment doesn't move; only the HF repo state matters. (Permanent DQs from `copy`, `anti_finetune`, or `arch` cannot be cleared this way — those require a new hotkey.) |
| "Copy detected" | Model hash matches another miner's submission | Your weights are identical to another miner's. Train your own model. |
| "Model is now private" DQ | HuggingFace repo set to private or deleted | Keep your model repo public at all times. |
| "Vocab size mismatch" | Modified tokenizer / using legacy Qwen3.5 tokenizer post-cutover | Use the exact same tokenizer as the Kimi K2.6 teacher (`vocab_size=163,840`). |
| "Quantized model detected" | Model has `quantization_config` in config.json | Remove quantization. Use bf16/fp16 weights only. (The teacher's INT4 wrapper does not transfer to your student.) |
| "Custom code files" DQ | `.py` files found in your repo | Remove all Python files from your HuggingFace repo. |
| "Tokenizer encoding mismatch" | Tokenizer produces different token IDs than teacher | Use the unmodified Kimi K2.6 tokenizer files. |
| "Chat template modified" | `chat_template` in tokenizer_config.json differs from reference | Use the original Kimi K2.6 chat template without modifications. |

---

## Useful Links

- **Dashboard:** <https://distil.arbos.life>
- **API Health:** <https://api.arbos.life/api/health>
- **GitHub:** <https://github.com/unarbos/distil>
- **Discord:** Channel `ა・distil・97` in the Bittensor Discord

---

## API Endpoints for Miners

All endpoints are on `api.arbos.life`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/miner/{uid}` | Details for a specific miner |
| `GET /api/scores` | Current scores |
| `GET /api/leaderboard` | Leaderboard (who's king, top contenders) |
| `GET /api/compare?uids=2,34,36` | Head-to-head comparison between miners |
| `GET /api/eval-status` | Current eval round status |
| `GET /api/eval-data` | Raw eval data |
| `GET /api/eval-stats` | Eval statistics |
| `GET /api/pod-logs` | Pod logs (paginated) |

---

## Key Constants

> **Reminder.** The values below are mirrored from [`frontend/src/lib/subnet-config.json`](../frontend/src/lib/subnet-config.json) and the `subnet_config` API endpoint. If they ever drift, **trust the JSON / API**, not this table. The Discord bot and the dashboard read directly from `subnet-config.json`.

| Parameter | Value |
|-----------|-------|
| Subnet UID | 97 |
| Teacher model | `moonshotai/Kimi-K2.6` (post-2026-05-02 cutover) |
| Max student params | **33B (total)** |
| Required architecture | Kimi-family (e.g. `KimiK25ForConditionalGeneration` or inner `DeepseekV3ForCausalLM`); see allowlist in `subnet-config.json` |
| Required model_type | matching the chosen Kimi-family arch (`kimi_k25` / `deepseek_v3`); see `subnet-config.json` |
| Vocab size | **163,840** (Kimi K2.6 BPE) |
| Eval prompts per UID | 300 (block-seeded, single-eval policy) |
| Eval prompts (broad sweep) | 60 |
| Max new tokens | 8,192 |
| Max prompt tokens | 1,024 |
| Eval policy | `SINGLE_EVAL_MODE=1` — one commitment, one eval |
| Challengers per round (cap) | `SINGLE_EVAL_MAX_PER_ROUND=10` (FIFO by `commit_block`) |
| Dethronement gate | `challenger.composite.final > incumbent.composite.final × 1.03` (cross-round, on absolute composite) |
| Saturated-floor tiebreaker | when both `worst ≤ 0.005`, same 3% margin on `composite.weighted` |
| King selection schema floor | `_KING_SELECTION_MIN_AXES = 17` (Arena v3.7) |
| Composite version | Arena v3.7 |
| Live axes | kl, on_policy_rkl, capability, length, degeneracy, judge_probe, math_bench, code_bench, reasoning_bench, knowledge_bench, ifeval_bench, aime_bench, mbpp_bench, tool_use_bench, self_consistency_bench, arc_bench, truthful_bench, long_context_bench, procedural_bench, robustness_bench, noise_resistance_bench, reasoning_density, chat_turns_probe, pareto_dominance |
| Shadow axes | none |
| Top-N always included | n/a in single-eval mode (no re-eval rotation) |
| Dataset (distillation) | `karpathy/climbmix-400b-shuffle` |
| Reference baseline | UID -1 — small Kimi-compatible reference (consult `subnet-config.json` `referenceModel`) |
