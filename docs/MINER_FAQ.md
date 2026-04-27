# Subnet 97 (Distil) — Miner FAQ & Getting Started

## What is Subnet 97?

Distil is a Bittensor subnet where miners compete to distill knowledge from a large teacher model into smaller student models. The teacher is **Qwen/Qwen3.5-35B-A3B** (35B total params, ~3B active — it's a Mixture-of-Experts model). Your job: produce the most faithful small model (≤5.25B params), measured on a **17-axis composite** that covers distribution match, capability against ground truth, conversational quality, generation discipline, and robustness to prompt rewrites.

The ranking key is **`composite.worst`** — the single weakest axis. KL is one of those 17 axes, not the gate. A model that wins KL but loses on grade-school math, IFEval, or reasoning-density cannot take the crown. Winner takes all — the king gets 100% of emissions.

> **Heads up.** If you've miner'd here before and remember "lower KL = win", that framing is wrong under the current eval. The 2026-04-17 reasoning-spiral king (UID 107: 4096-token loops on `"Hi"`, strictly worse than the unfine-tuned 4B base on every reasoning bench) was the wake-up call. The composite, the on-policy RKL axis, and the `reasoning_density` axis exist specifically to close that gap. Read the axis-by-axis playbook below before training. See [`paper/off_policy_cot_collapse.md`](../paper/off_policy_cot_collapse.md) for the full diagnosis.

---

## Getting Started

### 1. Register on SN97

Register a hotkey on subnet 97 via the standard Bittensor registration flow (`btcli subnet register --netuid 97`).

### 2. Train Your Student Model

- **Architecture:** Must be `Qwen3_5ForConditionalGeneration` with `model_type: "qwen3_5"` in `config.json`
  - ⚠️ **NOT** `Qwen3_5ForCausalLM` / `qwen3_5_text` — this will get you disqualified
- **Max total params:** 5.25B (total, not active — MoE tricks won't help)
- **Tokenizer:** Must be identical to the teacher's tokenizer (vocab size 248,320). Don't modify `tokenizer.json` or `tokenizer_config.json`
- **No quantization:** bf16/fp16 only. GPTQ, AWQ, GGUF etc. are rejected
- **No custom code:** `.py` files in your repo (except `__init__.py`) will get you DQ'd
- **Format:** Safetensors required (no pytorch `.bin`-only models)

### 3. Upload to HuggingFace

Push your model to a **public** HuggingFace repo. It must stay public — private or deleted models get disqualified.

### 4. Commit Your Model

Submit your HuggingFace model repo via the commitment mechanism on-chain. 

**⚠️ Commitments are permanent.** One model per hotkey, forever. You cannot re-upload or swap models on the same hotkey. Choose carefully.

---

## How Evaluation Works (Arena v3, live as of 2026-04-24)

Every round the validator pulls the set of new on-chain commitments and evaluates each one on a single GPU pod. The eval policy is **one registration → one commitment → one eval**: your commitment is scored exactly once, the absolute composite is stored, and the king is decided cross-round from those stored scores. There are no re-evals, no king re-runs, no top-N rotations, and no dormant-rotation refreshes. Each student is scored on many **independent axes**; the leaderboard is ordered by the **worst** of those axes, so gaming any single one will pull your overall rank down. The design goal is simple: **if you overfit our eval, you will accidentally produce a SOTA small model**. Every axis points at a real, held-out capability.

**The round itself:**

1. **300 prompts** per round sampled from [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), seeded by the current block. The teacher generates continuations through vLLM (`temperature=0.7, top_p=0.9`, per-prompt seed = `block_seed + prompt_idx`; deterministic per round, rotating across rounds). Teacher logprobs are stored as sparse top-k logprobs when vLLM provides them, with HF teacher forward as fallback.
2. **KL divergence** is computed on those teacher continuations by running each student through an HF forward pass and comparing the student's distribution against the teacher's stored logits/logprobs.
3. **One eval per commitment.** Your model is scored exactly once after you commit. The validator stores `composite.worst`, every axis, and the commit signature in `composite_scores.json`. The eval is not repeated unless you submit a new commitment (different model / revision / block) on the same hotkey.
4. **Cross-round king.** The king is the UID with the highest stored `composite.worst` across all non-DQ scored UIDs. A new commitment can dethrone the king by clearing `worst > king.worst × (1 + SINGLE_EVAL_DETHRONE_MARGIN)` (default 3% margin) on its single eval. There is no paired-t-test re-run because the king is not re-evaluated.
5. **Reference baseline.** Undistilled `Qwen/Qwen3.5-4B` is included in every round as UID `-1` for axis-floor anchoring and dashboard visibility — it is not a contender.
6. **Winner takes all** — the king gets 100% of emissions.

**Implication for miners.** Pick your weights carefully before you commit. The on-chain registration burn is the price of an evaluation: there is no "re-roll the same commitment until variance lands well." A model that scores 0.42 worst-axis stays at 0.42 forever (until you push a new commitment to the same hotkey, which fully overwrites the previous record).

### The axes (ranking key = `composite.worst` = min of every axis below)

All axes are in `[0, 1]`, higher-is-better. Missing axes (e.g. probe outage) are dropped and the weighted mean renormalizes over surviving axes. Each axis drops if the teacher itself fails a sanity floor (so a miscalibrated probe can't corrupt rankings).

**Teacher-similarity axes** (normalized against the king/teacher, weight 0.60 total):

| Axis              | What it measures                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------------|
| `kl`              | Teacher-forced KL divergence on teacher continuations. Anchored to the best (lowest) KL seen this round.   |
| `on_policy_rkl`   | Reverse KL under **your** sampling. Catches "matches teacher logits but collapses under free generation".   |
| `capability`      | Verifiable prompts (arithmetic/yes-no/one-word factual). `min(frac/teacher_frac, frac/0.6)` — absolute floor prevents winning by echoing teacher mistakes. |
| `length`          | Student generation length vs a teacher anchor. Rambling models lose here. 1.0 when you match the teacher.  |
| `degeneracy`      | Termination fraction + MAD-z-scored repetition + cross-rollout Self-BLEU. 1.0 = teacher-like.             |
| `judge_probe`     | Teacher (Qwen3.5-35B) rates your response on a 1-5 rubric, rotated to 16 prompts/round. Normalized to [0,1]. |

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

Single-eval mode replaces the round-local paired-t-test with a cross-round absolute-composite comparison. The composite-floor and Pareto gates remain in place as anti-gaming defenses.

1. **Composite-worst margin.** Your single eval's `composite.worst` must exceed the king's stored `composite.worst` by `SINGLE_EVAL_DETHRONE_MARGIN` (default 3%). 0.50 → 0.515 is not enough; 0.50 → 0.52 is.
2. **Worst-axis floor.** If `composite.worst < COMPOSITE_DETHRONE_FLOOR = 0.20`, the dethrone is **vetoed** even if the margin passes. The axis that triggered the veto is logged and surfaced in telemetry.
3. **Pareto-dominance gate.** A challenger that wins on `composite.worst` but loses to the king on a majority of comparable axes is blocked. Pareto semantics are *soft*: majority win AND `n_wins ≥ n_losses`, with a 2% noise margin. Insufficient comparable axes fails open.

When `SINGLE_EVAL_MODE` is OFF (development / fallback), the legacy KL paired t-test (p < 0.03 + 3% epsilon vs the sitting king) is used instead of #1 above. The other gates are identical.

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

- **Base model:** Start from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or a compatible Qwen3.5 architecture.
- **Objective:** Optimise for `composite.worst`, not KL. KL(teacher ‖ student) is one of 17 axes — useful but never sufficient. A pure-KL model loses to a slightly-worse-KL model that also answers GSM8K correctly, doesn't loop on `"Hi"`, and survives prompt paraphrase.
- **Data mix:** at minimum combine ClimbMix-style distillation data with ~10–20% instruction/reasoning/code data (see the playbook above). Miners who run SFT + DPO on top of their distillation have been climbing the bench axes fastest.
- **Long completions matter:** eval uses `max_new_tokens=8192`. The model needs to terminate naturally on simple prompts and reason coherently on long ones.
- **Temperature:** vLLM runs at `temperature=0.7, top_p=0.9` with per-prompt seed `block_seed + prompt_idx`. Deterministic per round, rotating between rounds. Greedy (temp=0) only applies to local dev runs without `--block-seed`.
- **Don't modify the chat template:** it's checked against the reference Qwen3.5 template hash. Injected comments or modifications = DQ.
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
| "Wrong architecture" DQ | `config.json` has `Qwen3_5ForCausalLM` or wrong `model_type` | Change `architectures` to `["Qwen3_5ForConditionalGeneration"]` and `model_type` to `"qwen3_5"` in config.json. No weight changes needed. |
| "Integrity check failed" | HF repo deleted, made private, or otherwise unreachable since the validator first hashed it | Make the repo public and re-upload the same weights — the integrity DQ clears next epoch when the validator can re-verify. The on-chain commitment doesn't move; only the HF repo state matters. (Permanent DQs from `copy`, `anti_finetune`, or `arch` cannot be cleared this way — those require a new hotkey.) |
| "Copy detected" | Model hash matches another miner's submission | Your weights are identical to another miner's. Train your own model. |
| "Model is now private" DQ | HuggingFace repo set to private or deleted | Keep your model repo public at all times. |
| "Vocab size mismatch" | Modified tokenizer | Use the exact same tokenizer as the teacher (Qwen3.5-35B-A3B). |
| "Quantized model detected" | Model has `quantization_config` in config.json | Remove quantization. Use bf16/fp16 weights only. |
| "Custom code files" DQ | `.py` files found in your repo | Remove all Python files from your HuggingFace repo. |
| "Tokenizer encoding mismatch" | Tokenizer produces different token IDs than teacher | Use the unmodified teacher tokenizer files. |
| "Chat template modified" | `chat_template` in tokenizer_config.json differs from reference | Use the original Qwen3.5 chat template without modifications. |

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

| Parameter | Value |
|-----------|-------|
| Subnet UID | 97 |
| Teacher model | `Qwen/Qwen3.5-35B-A3B` |
| Max student params | 5.25B (total) |
| Required architecture | `Qwen3_5ForConditionalGeneration` |
| Required model_type | `qwen3_5` |
| Vocab size | 248,320 |
| Eval prompts per UID | 300 (block-seeded, single-eval policy) |
| Eval prompts (broad sweep) | 60 |
| Max new tokens | 8,192 |
| Max prompt tokens | 1,024 |
| Eval policy | `SINGLE_EVAL_MODE=1` — one commitment, one eval |
| Challengers per round (cap) | `SINGLE_EVAL_MAX_PER_ROUND=10` (FIFO by `commit_block`) |
| Dethronement gate | `challenger.composite.worst > incumbent.composite.worst × 1.03` (cross-round, on absolute composite) |
| Saturated-floor tiebreaker | when both `worst ≤ 0.005`, same 3% margin on `composite.weighted` |
| King selection schema floor | `_KING_SELECTION_MIN_AXES = 17` (Arena v3.7) |
| Composite version | Arena v3.7 |
| Live axes | kl, on_policy_rkl, capability, length, degeneracy, judge_probe, math_bench, code_bench, reasoning_bench, knowledge_bench, ifeval_bench, aime_bench, mbpp_bench, tool_use_bench, self_consistency_bench, arc_bench, truthful_bench, long_context_bench, procedural_bench, robustness_bench, noise_resistance_bench, reasoning_density, chat_turns_probe, pareto_dominance |
| Shadow axes | none |
| Top-N always included | n/a in single-eval mode (no re-eval rotation) |
| Dataset (distillation) | `karpathy/climbmix-400b-shuffle` |
| Reference baseline | `Qwen/Qwen3.5-4B` (UID -1) |
