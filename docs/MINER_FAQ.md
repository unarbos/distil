# Subnet 97 (Distil) — Miner FAQ & Getting Started

## What is Subnet 97?

Distil is a Bittensor subnet where miners compete to distill knowledge from a large teacher model into smaller student models. The teacher is **Qwen/Qwen3.5-35B-A3B** (35B total params, ~3B active — it's a Mixture-of-Experts model). Your job: produce the most faithful small model (≤5.25B params), measured by KL divergence against the teacher's output distribution.

Lower KL = better. Winner takes all — the king gets 100% of emissions.

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

Every round, the validator evaluates the king + top-5 challengers + a handful of new/changed challengers on a single GPU pod. Each student is scored on many **independent axes**; the leaderboard is ordered by the **worst** of those axes, so gaming any single one will pull your overall rank down. The design goal is simple: **if you overfit our eval, you will accidentally produce a SOTA small model**. Every axis points at a real, held-out capability.

**The round itself:**

1. **300 prompts** per round sampled from [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), seeded by the current block. Both teacher and student are run through vLLM (`temperature=0.7, top_p=0.9`, per-prompt seed = `block_seed + prompt_idx`; deterministic per round, rotating across rounds).
2. **KL divergence** is computed between teacher and student output distributions.
3. **King-of-the-hill:** the current king is re-evaluated alongside every round. Challenger must beat the king with statistical significance (paired t-test, p < 0.03) to take the crown **AND** pass every axis gate.
4. **Top 5 contenders** are always included. A reference baseline (undistilled `Qwen/Qwen3.5-4B`) is evaluated every round as UID `-1`.
5. **Winner takes all** — the king gets 100% of emissions.

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

**Arena v3 Session 3 — SHADOW, promoting 2026-04-26 (weight 0.20 total):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `aime_bench`                 | AIME25 + AIME2024 (~90 olympiad items), 4/round. Boxed-integer extraction.                              |
| `mbpp_bench`                 | MBPP+ (378 items), 2/round. Sandboxed test-list execution.                                              |
| `tool_use_bench`             | Math items with an injected Python REPL. Model emits `<python>…</python>`, stdout spliced back into a 2nd generation pass, final boxed answer scored. Rewards agentic capability. |
| `self_consistency_bench`     | Hard math, K=5 samples at T=0.7 each, majority vote on the boxed answer. Rewards underlying knowledge vs one-shot luck. |

**Arena v3 Session 3.1 — SHADOW, added 2026-04-25:**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `arc_bench`                  | AI2 ARC-Challenge (~1172 grade-school science items), 6/round. Letter-choice MC, completely disjoint from MMLU-Pro/BBH. |

**Arena v3 Session 3.2 — SHADOW, added 2026-04-25 (addresses "models over-think simple questions"):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `reasoning_density`          | `pass_frac × length_bonus` averaged across benches, where `length_bonus = 1.0` if `mean_gen_tokens_correct ≤ target` (e.g. knowledge ≤30 tok, math ≤400 tok) and decays with `1/(1+ratio−1)` above target. Penalizes both over-thinking trivia AND verbose-but-wrong answers. Cannot be gamed by short-wrong: pass_frac=0 → axis=0. |

**Arena v3 Session 3.3 — SHADOW, added 2026-04-25 (multi-turn coherence):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `chat_turns_probe`           | 6 hand-authored 3-turn dialogues/round. Student generates 3 assistant turns with accumulated context; teacher grades the full transcript on a 1-5 rubric (coherence + consistency + helpfulness). Directly probes deployment-quality multi-turn dialogue — a capability pure climbmix-KL distillation does NOT reward. |

**Arena v3 Session 3.4 — SHADOW, added 2026-04-25 (adversarial factuality):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `truthful_bench`             | TruthfulQA mc1 (~817 items), 4/round. Adversarial factual questions where the popularly-believed-but-wrong answer is included as a tempting distractor. Tests hallucination resistance. Correct letter is deterministically shuffled per item so a model can't win by always answering "A". |

**Arena v3 Session 3.5 — SHADOW, added 2026-04-25 (long-context retrieval):**

| Axis                         | What it tests                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| `long_context_bench`         | Procedural needle-in-haystack over ~600 tokens (tunable), 3/round. Items are *generated fresh every round from `block_seed`* — there is no dataset to memorize. Each item inserts a single needle sentence (e.g. "The lost vault combination is 4ESGKG3.") into a document of 40 distractor sentences and asks the student to recall the needle. Tests whether the model actually reads its input window instead of leaning on priors. Probably the most Goodhart-immune axis we have. |

All bench pools rotate per-round via `block_seed`, so every validator picks the same items but items differ between rounds (anti-memorization).

### Dethrone gates (all must pass)

1. **KL gate.** Paired t-test p < 0.03 + 3% epsilon vs the sitting king on the H2H prompt set. (Legacy epsilon path remains as a fallback.)
2. **Worst-axis floor.** If `composite.worst < COMPOSITE_DETHRONE_FLOOR = 0.20`, the dethrone is **vetoed** — even if KL passes. The axis that triggered the veto is logged and surfaced in telemetry.
3. **Pareto-dominance gate** (shadow, activates 2026-04-26). A challenger that beats the king on KL but loses on a majority of comparable axes gets blocked. Pareto semantics are *soft*: majority win AND `n_wins ≥ n_losses`, with a 2% noise margin. Insufficient comparable axes fails open.

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

**Two anti-patterns to avoid:**

- **Pure KL overfitting.** Matching teacher logits perfectly but failing on grade-school math means your composite worst is low. You cannot take the crown.
- **Long rambling.** `length` + `judge_probe` + `degeneracy` all penalize verbose thinking. Teacher-style brevity wins.

**Watch your dashboard columns:** `Judge / Bench / V3* / Pareto* / Worst / vs King`. The `*` means shadow (will go live on 2026-04-26). If your `V3*` worst is below your production worst, that's the next 48 h of work.

---

## Training Tips

- **Base model:** Start from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or a compatible Qwen3.5 architecture.
- **Objective:** KL(teacher ‖ student) is the floor, not the ceiling. A pure-KL model loses to a slightly-worse-KL model that also answers GSM8K correctly.
- **Data mix:** at minimum combine ClimbMix-style distillation data with ~10–20% instruction/reasoning/code data (see the playbook above). Miners who run SFT + DPO on top of their distillation have been climbing the bench axes fastest.
- **Long completions matter:** eval uses `max_new_tokens=8192`. The model needs to terminate naturally on simple prompts and reason coherently on long ones.
- **Temperature:** vLLM runs at `temperature=0.7, top_p=0.9` with per-prompt seed `block_seed + prompt_idx`. Deterministic per round, rotating between rounds. Greedy (temp=0) only applies to local dev runs without `--block-seed`.
- **Don't modify the chat template:** it's checked against the reference Qwen3.5 template hash. Injected comments or modifications = DQ.
- **Bench probes run offline.** All datasets are pre-cached on the pod (`HF_HUB_OFFLINE=1`). No network-dependency required in your model.

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
| Eval prompts (head-to-head) | 300 |
| Eval prompts (broad sweep) | 60 |
| Max new tokens | 8,192 |
| Max prompt tokens | 1,024 |
| Dethronement threshold | paired t-test, p < 0.03 AND worst-axis ≥ 0.20 |
| Composite version | Arena v3 (shadow v9) |
| Live axes | kl, on_policy_rkl, capability, length, degeneracy, judge_probe, math_bench, code_bench, reasoning_bench, knowledge_bench, ifeval_bench |
| Shadow axes (live 2026-04-26) | aime_bench, mbpp_bench, tool_use_bench, self_consistency_bench, arc_bench, truthful_bench, long_context_bench, reasoning_density, chat_turns_probe, pareto_dominance |
| Top-N always included | 5 |
| Dataset (distillation) | `karpathy/climbmix-400b-shuffle` |
| Reference baseline | `Qwen/Qwen3.5-4B` (UID -1) |
