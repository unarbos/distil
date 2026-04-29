# SN97 Mining Guide v2 — How to actually win in 2026

**Last updated:** 2026-04-29 (v30 eval)
**Audience:** miners who want to climb the SN97 leaderboard *and* end up with a genuinely-better SOTA-class small model in the process.

This guide is a sibling to [`MINER_FAQ.md`](MINER_FAQ.md). The FAQ tells you the rules. This guide tells you the **recipe that actually works in 2026** based on the published distillation literature, validated against your eval pipeline.

> **One-line summary.** *Mid-train SFT → curated SFT on rejection-sampled traces → on-policy distillation (RKL) → optional GRPO on verifiable rewards.* Skip stage 1 and you'll regress like LIMO/s1 on a 4B student. Skip stage 3 and your KL plateaus. Skip stage 4 and you leave 5–10pp on AIME / MATH on the table.

---

## 1. The 2026 SOTA recipe at a glance

The cheapest reliable path to a SOTA-class 4B student in 2026 is **four stages**, in order. The "less is more" recipes (LIMO 800 samples, s1 1k samples) **regress small students** — they assume base-model capacity that 4B doesn't have, per the Phi-4-Mini-Reasoning paper:

| Pipeline | AIME24 | MATH-500 | GPQA-Diamond |
| --- | --- | --- | --- |
| Phi-4-Mini base (3.8B) | 10.0 | 71.8 | 36.9 |
| Phi-4-Mini + LIMO 800 SFT | **6.7** ↓ | 57.8 ↓ | 24.8 ↓ |
| Phi-4-Mini + S1K 1k SFT | **3.0** ↓ | 47.0 ↓ | 26.3 ↓ |
| Phi-4-Mini-Reasoning (full 4-stage) | **57.5** ↑ | **94.6** ↑ | **52.0** ↑ |

The conclusion is unambiguous: **skip the mid-train phase and your 4B will regress** below the base model on every reasoning benchmark, even if your validator-side KL looks good. Read this table before you commit anything.

---

## 2. The 4-stage pipeline

### Stage 1 — Mid-train SFT (the LIMO/s1 warning)

**Goal:** load the student with broad reasoning competence so the smaller curated stages have something to build on.

**Recipe:**

- Start from `Qwen/Qwen3.5-4B` base (the SN97-mandated architecture).
- Train 5–20B tokens of **diverse reasoning traces** from a strong teacher: a representative mix of math (GSM8K, MATH, NuminaMath), code (CodeContests, MBPP, HumanEval), reasoning (BBH, ARC, PIQA), and chat (LMSYS-style). Phi-4-Mini-Reasoning used 16B tokens of R1-distill traces here.
- This stage is **not** rejection-sampled. The point is *coverage*, not perfection.
- Loss: standard cross-entropy on teacher tokens. Optionally add a forward-KL term against the teacher's top-K logits (~5–10% improvement, but not required).

**What you should see at the end of stage 1:**

- KL vs teacher on ClimbMix prompts ≤ ~0.6 nats (cf. `composite.kl` axis: ~0.5 maps to king-class scoring).
- AIME24 pass@1 ≥ 25% (cf. king-class target ≥ 50% — stage 1 alone won't get you there).
- MATH-500 ≥ 75%.

**Failure modes to watch:**

- Length collapse — student outputs <40 tokens on simple prompts. Caused by training too long with a hard EOS bias. Fix: drop the LR by 10× or finish stage 1 earlier.
- Repetition loops — same n-gram repeats >5x. Caused by overly small effective batch + high LR. Fix: increase batch size or grad-accum, lower LR.
- IFEval regression — base IFEval ~62% drops to ~40%. Caused by training only on math/code without instruction-following data. Mix in 10–20% IFEval-style traces.

### Stage 2 — Curated SFT on rejection-sampled traces

**Goal:** sharpen the student on *high-quality, correct, structurally clean* reasoning traces.

**Recipe (the brianmeyer / DeepSeek-R1 / Bespoke-Stratos / s1 lineage):**

1. Generate teacher rollouts on a curated prompt mix (1k–100k prompts depending on budget). Suggested mix: 50% math (GSM8K + MATH + AIME-historical + NuminaMath), 30% code (MBPP+ + HumanEval+ + LiveCodeBench), 20% reasoning/chat (BBH + LMSYS-converted to single-turn).
2. **8-gate quality filter** (drop ~10–25% of generations):
   - Non-empty thinking + response.
   - No encoding artefacts.
   - Length bounds: 50–4,000 thinking tokens.
   - **Correctness verification** (numerical match for GSM8K, `\boxed{}` for MATH, exec sandbox for code).
   - No degenerate repetition loops.
   - Coherence — thinking actually references the problem.
   - Self-contradiction — max 2 self-corrections.
   - Structured reasoning — step indicators present.
3. Format consistently with `<thinking>` / `<response>` tags.
4. SFT for 1–3 epochs at LR ~2e-6, effective batch ≥ 32. Use **forward KL** here (not reverse KL — RKL is for stage 3).

**Sweet spot:** 17k–50k filtered traces. Bespoke-Stratos got near R1-Distill-32B with 17k filtered traces (47× less than R1's 800k unfiltered baseline).

**What you should see at the end of stage 2:**

- KL vs teacher ≤ 0.45 nats.
- AIME24 ≥ 35–50%.
- MATH-500 ≥ 85%.
- HumanEval+ ≥ 70%.

**Failure modes:**

- Mode collapse on common reasoning patterns ("let me think step by step…" 95% of the time). Caused by over-filtering for "structured" traces. Fix: lower the structured-reasoning gate threshold from 0.9 → 0.7.
- Knowledge loss — drops to <60% on MMLU-Pro. Caused by training only on reasoning. Fix: include 10–20% open-ended chat in the SFT mix.

### Stage 3 — On-policy distillation (RKL)

**Goal:** the student generates its own continuations; the teacher only scores. This closes the off-policy gap that pure SFT can't fix.

**Why this matters for SN97:** the validator's `on_policy_rkl` axis (composite weight 0.35 — the largest single axis weight) measures **exactly this**. A student that aces stage 2 SFT but never sees on-policy training will saturate at ~0.55 on `on_policy_rkl` while a stage-3-trained student climbs to 0.75–0.85.

**Recipe (TRL `DistillationTrainer`, the canonical 2026 OPD config):**

```python
from trl import DistillationConfig, DistillationTrainer

config = DistillationConfig(
    lmbda=1.0,        # fully on-policy: student generates, teacher scores
    beta=1.0,         # reverse-KL (mode-seeking)
    loss_top_k=1,     # top-1 approximation; full vocab is impractical at K=248320
    use_vllm=True,
    use_teacher_server=True,
    temperature=0.9,  # student-side sampling
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=64,
    num_train_epochs=1,
)
```

**Critical settings (cribbed from Thinking Machines + HF + Qwen3 papers):**

- `lmbda=1.0` → **fully on-policy**. `lmbda<1` mixes off-policy data and is only worth it if your stage-2 SFT didn't fully converge.
- `beta=1.0` → **reverse-KL**. Reverse KL is mode-seeking, which is what you want here. Forward KL is mode-covering — already done in stage 2.
- `loss_top_k=1` → **top-1 approximation** is enough at scale. The 2026 paper found no measurable gap to full-vocab.
- Teacher: same as the validator's teacher (currently **Qwen3.6-35B-A3B**) so OPD mass concentrates on the same support.
- Train for 1k–10k steps. The "elbow" where AIME climbs sharpest is typically 2k–5k steps.

**LoRA shortcut for compute-constrained miners:**

- `--lora_target_modules all-linear` (NOT just attention)
- `--lora_r 64` or `128` (rank ≤32 lags full-FT by 13% on SFT but only 6% after OPD)
- LR **10×** higher than full FT, typically 1e-5
- Effective batch < 32

LoRA-r=128 + OPD on Qwen3.5-4B base is the right sweet-spot for consumer GPUs. Memory: ~24GB at bf16. Wall time on a single A100 80GB: ~36 hours for 5k OPD steps.

**What you should see at the end of stage 3:**

- `on_policy_rkl` ≥ 0.75 (target king-class).
- KL vs teacher ≤ 0.35 nats.
- **`top_k_overlap` ≥ 0.85** (v30 axis — see §4 below).
- AIME24 ≥ 50%, MATH-500 ≥ 90%.

**Failure modes:**

- Length explosion — student starts emitting 1500-token rambles. Caused by reward hacking on the verbose-thinking distribution of the teacher. Fix: add a length-penalty term to the OPD loss, or add a max-token cap of ~600 to student rollouts.
- Reasoning-density regression — `correct_frac` × `length_bonus` drops because the student answers correctly but in 5× more tokens than needed. Fix: reduce OPD temperature to 0.6, or include a length-conditioned curriculum.
- KL improves but capability regresses — the student is "teacher-hacking" (matching teacher tokens on questions both answer wrong). Fix: ensure your OPD prompt mix includes verifiable items (GSM8K + MBPP + IFEval) so the gradient prefers correct answers, not just teacher-matching ones.

### Stage 4 — GRPO on verifiable rewards (optional, +5–10pp)

**Goal:** push the student past the teacher's ceiling on objectively-checkable tasks.

**Recipe:**

- Use the **same student you finished stage 3 with** as both initial policy and reference.
- Reward function: pass/fail against a verifier (GSM8K answer match, MATH `\boxed{}` match, MBPP+ test pass, IFEval constraint satisfaction).
- KL term **against the stage-3 student** (NOT against the teacher) so the model doesn't drift away from the distilled distribution.
- 500–2000 GRPO steps, batch ≥ 32, K=8 rollouts per prompt.

**What you should see:**

- AIME24 climbs +5–10pp without regressing on MATH-500 / MMLU-Pro.
- `composite.worst` final boost: typically +0.04–0.07 on the SN97 leaderboard.

**Failure mode:** GRPO catastrophically forgets non-reasoning capability if the prompt mix is too narrow. **Always mix in 20% chat / IFEval-style prompts** scored by the teacher rubric, not just verifiable ones.

---

## 3. Concrete success thresholds per validator axis

Before you commit, run an internal eval that mirrors the validator. These are the **king-class thresholds** as of v30 (2026-04-29). Hitting all of them puts you in genuine contention.

| Axis | King-class | "Good enough to commit" | Notes |
| --- | --- | --- | --- |
| `composite.worst` | ≥ 0.55 | ≥ 0.45 | Min of all axes; this is the ranking key. |
| `on_policy_rkl` | ≥ 0.75 | ≥ 0.65 | Largest single weight (0.35). Stage 3 dependent. |
| `kl` | ≥ 0.85 | ≥ 0.70 | Saturated; weight reduced to 0.05 in v29.7. |
| `top_k_overlap` | ≥ 0.90 | ≥ 0.80 | **NEW v30**. Per the 'Rethinking OPD' paper, the most predictive single signal of OPD success. Successful runs hit 0.97–0.99. |
| `capability` | ≥ 0.80 | ≥ 0.65 | 50/50 mix of absolute pass-rate + relative-to-teacher. |
| `length` | ≥ 0.75 | ≥ 0.60 | Penalises both rambling and hard-stops. |
| `degeneracy` | ≥ 0.90 | ≥ 0.80 | Self-BLEU + termination + non-degeneracy. |
| `judge_probe` | ≥ 0.65 | ≥ 0.50 | Teacher rubric, 1-5 scale → [0,1]. |
| `long_form_judge` | ≥ 0.55 | ≥ 0.40 | **NEW v30**. 300-500 word essay-style answers; teacher rubric grades structure / depth / coherence / length. |
| `chat_turns_probe` | ≥ 0.55 | ≥ 0.40 | 3-turn coherence under a teacher rubric. |
| `math_bench` | ≥ 0.75 | ≥ 0.60 | gsm8k-narrative procedural items per round. |
| `code_bench` | ≥ 0.65 | ≥ 0.50 | HumanEval-style procedural items, sandbox-graded. |
| `reasoning_bench` | ≥ 0.65 | ≥ 0.50 | BBH-style multi-step reasoning. |
| `knowledge_bench` | ≥ 0.65 | ≥ 0.50 | **v2 in v30**: open-ended factual reasoning (price tables, calendar/unit/roman conventions). Replaces muted MC version. |
| `ifeval_bench` | ≥ 0.65 | ≥ 0.55 | Procedural instruction-following constraints. |
| `aime_bench` | ≥ 0.40 | ≥ 0.20 | Olympiad math; very hard. Stage 4 GRPO helps most here. |
| `mbpp_bench` | ≥ 0.65 | ≥ 0.50 | Programming problems with hidden tests. |
| `tool_use_bench` | ≥ 0.50 | ≥ 0.35 | Agentic Python with import/call structure. |
| `long_context_bench` | ≥ 0.60 | ≥ 0.45 | ~1400-token needle-in-haystack, procedural. |
| `robustness_bench` | ≥ 0.60 | ≥ 0.45 | Math under K paraphrase wrappers. |
| `debug_bench` | ≥ 0.60 | ≥ 0.45 | Buggy code → fix. |
| `correction_bench` | ≥ 0.55 | ≥ 0.40 | Buggy code + explicit error trace → fix. |
| `multi_doc_synthesis_bench` | ≥ 0.55 | ≥ 0.40 | Cross-card retrieval + reasoning. |
| `calibration_bench` | ≥ 0.65 | ≥ 0.50 | Solvable + unsolvable mix; reward correct refusals. |
| `refactor_bench` | ≥ 0.50 | ≥ 0.35 | Behaviour-preserving + style-constrained refactor. |
| `pragmatic_bench` | ≥ 0.65 | ≥ 0.50 | **NEW v30**. Theory-of-mind / scalar implicature / indirect-request recognition. |

**The `worst` ranking key.** If your weakest axis is e.g. 0.35 even though everything else is 0.75, you're at 0.35 in the leaderboard ordering. Spend your last week of training rebalancing whichever axes are below 0.5; the marginal `composite.worst` gain from raising your weakest axis is much larger than the gain from raising any axis already in the king-class band.

---

## 4. The v30 changes you need to know about

These shipped on 2026-04-29:

### `top_k_overlap` axis (NEW, weight 0.10)

For each generated token, `|top_K_teacher ∩ top_K_student| / K` averaged over the continuation. Per the *Rethinking On-Policy Distillation* paper (arXiv 2604.13016), this is **the single most predictive signal of OPD success**. Successful OPD runs converge to 97–99% shared top-K mass with the teacher.

**How to maximise:** stage 3 OPD with the **same teacher** as the validator. Anything else (different teacher, frozen-after-stage-2 student) caps at ~0.80.

**Gameability:** essentially none. Top-K overlap depends on the on-policy trajectory; off-policy memorisation of prompt-response pairs doesn't transfer because the student picks different next-tokens at sampling time.

### `knowledge_bench` v2 (replaces muted MC version)

Procedural fact-like reasoning items (price tables, transitive ordering, container counting, alphabet/calendar/weekday/unit/roman conventions). Open-ended generation, regex match. **No MC random-pick floor.**

**How to maximise:** train on diverse instruction-following + structured-context reasoning data. The items demand reading a context block (price table, transitive sequence) and computing an answer; this is *exactly* the skill MMLU-Pro / GPQA tests, just procedurally generated so it's uncacheable.

### `pragmatic_bench` (NEW, weight 0.04)

Procedural theory-of-mind / scalar implicature / indirect-request items. Tests pragmatic reasoning that no other axis covers.

**How to maximise:** include **theory-of-mind training data** in your stage-2 SFT mix. False-belief, second-order belief, and scalar implicature data is now a meaningful axis of selection pressure. Public datasets: ToMi, OpenToM, FANToM. Also train on **conversational data** with embedded indirect requests so the model learns to resolve them as actions, not as ability questions.

### `long_form_judge` (NEW, weight 0.05)

4 essay-style prompts per round. Student must generate a 300-500 word multi-paragraph response. Teacher rubric grades structure / depth / coherence / length on 1-5.

**How to maximise:** include **long-form essay data** in your stage-2 SFT mix (e.g., paragraph-style explanations, op-eds, long answers from LMSYS-1M). Don't train *only* on short Q/A pairs.

### Refreshed weights

The composite weight allocation as of v29.7 (still in effect):

```
on_policy_rkl: 0.35  ← largest single weight
kl: 0.05             ← saturated, demoted
top_k_overlap: 0.10  ← NEW v30
capability: 0.10     ← was 0.25, demoted (overlaps benches)
length: 0.05         ← was 0.10, mostly safety
degeneracy: 0.15
judge_probe: 0.15
long_form_judge: 0.05  ← NEW v30
chat_turns_probe: 0.08
[bench axes: math 0.14, code 0.14, reasoning 0.10, ifeval 0.07, ...]
```

Total weight on bench axes ≈ 0.85; total relative-axis weight ≈ 0.93. Both renormalise so this doesn't matter for the weighted score, but the relative magnitudes do dictate where to spend your training compute.

---

## 5. Common Goodhart traps to avoid

1. **Length explosion.** Distilling from a long-CoT teacher can make the student emit 1500-token responses to "Hi". The `length` and `reasoning_density` axes catch this. Always validate against a chat probe with `enable_thinking=False`.

2. **Memorising the public benchmarks.** The v27+ procedural switch makes this useless — *every* validator axis (math, code, reasoning, knowledge, AIME, MBPP, IFEval, ARC, BBH) is now block-seeded procedural. There is no offline dataset to memorise.

3. **Teacher-hacking on KL.** Matching the teacher's *wrong* answers on hard items wins KL but loses on `capability` and the bench axes. The composite's `worst()` ranking guarantees this strategy loses.

4. **Refusal-only models.** A model that says "I don't know" to every question gets 0 on most benches but does fine on `calibration_bench`'s unsolvable items. The composite's `worst()` floor blocks this.

5. **Repetition loops to game perplexity.** Caught by `degeneracy` (self-BLEU + non-degeneracy + termination).

6. **Single-axis specialisation.** "I'll just maximise `aime_bench`" is a guaranteed-to-lose strategy. `aime_bench` is weighted 0.04 and 0.55 on `aime_bench` with 0.30 on every other axis still gives you `worst = 0.30`. Always optimise the bottom of your axis distribution, not the top.

7. **Per-axis baseline regression.** v29.1 ships a baseline-relative penalty: any bench axis where you score *below the same-round Qwen3.5-4B base reference* gets docked by `1.5 × (ref - your_score)`. So a regression of 10pp below base costs you 25pp on that axis (10pp raw + 15pp dock). This makes "stay above base" the dominant strategy.

8. **Cross-validator drift.** All procedural items derive from `block_seed`, so every validator generates the same items each round. Don't try to fingerprint individual validators — your composite is the same regardless.

---

## 6. End-to-end checklist before you commit

Before you push your model + commit on-chain:

- [ ] Architecture is `Qwen3_5ForConditionalGeneration` with `model_type: "qwen3_5"`.
- [ ] Total params ≤ **7B** (v29.7 raised the cap from 5.25B; check `frontend/src/lib/subnet-config.json` for the live value).
- [ ] Tokenizer is **byte-identical** to the teacher's (vocab 248,320). Don't modify `tokenizer.json` or `tokenizer_config.json`.
- [ ] No quantisation (bf16 / fp16 only).
- [ ] No `.py` files (except `__init__.py`).
- [ ] Safetensors only (no `.bin`-only models).
- [ ] HuggingFace repo is **public** and stays public.
- [ ] Local eval shows `composite.worst ≥ 0.45` against the v30 procedural benches (run your own copy of `pod_eval_vllm.py` if you want to mirror exactly).
- [ ] No regression below Qwen3.5-4B base on any axis (verify with the reference axes from a recent `composite_scores.json`).
- [ ] Held-out canary: AIME24 ≥ 30%, MATH-500 ≥ 80%, MBPP+ ≥ 65%, IFEval ≥ 60%. The validator runs Evalscope canaries on the king out-of-band; a king that regresses on the canary triggers the `king_canary_streak` auto-dethrone gate (see [`reports/2026-04-28-grief-tiebreaker-and-canary-streak.md`](../reports/2026-04-28-grief-tiebreaker-and-canary-streak.md)).
- [ ] Long-form sanity check: write 4 prompts that demand 300-500 word essays, run greedy generation, eyeball the responses for structure / depth / coherence. If they look like 1-line answers or unstructured rambles, your `long_form_judge` axis will be ≤0.30.
- [ ] Pragmatic sanity check: write 3 false-belief items by hand, verify the model gives the actor's-belief answer, not the world-state answer.

**Commitments are permanent. One per hotkey, forever.** A model that scores 0.42 stays at 0.42 until you push a *new* commitment to the same hotkey, which fully overwrites the previous record.

---

## 7. References & further reading

| What | Where |
| --- | --- |
| Validator eval pipeline | [`scripts/pod_eval_vllm.py`](../scripts/pod_eval_vllm.py) |
| Composite score definition | [`scripts/validator/composite.py`](../scripts/validator/composite.py) |
| 2026 SOTA distillation synthesis (full literature review) | [`reports/2026-04-29-distillation-sota-synthesis.md`](../reports/2026-04-29-distillation-sota-synthesis.md) |
| v30 strategic rollup (axis audit + teacher swap) | [`reports/2026-04-29-v30-strategic-rollup.md`](../reports/2026-04-29-v30-strategic-rollup.md) |
| Phi-4-Mini-Reasoning paper (4-stage recipe) | arXiv 2504.21233 |
| LIMO regression on small models | arXiv 2502.03387 |
| s1 / Bespoke-Stratos | arXiv 2501.19393, Bespoke Labs blog |
| TRL DistillationTrainer (canonical 2026 OPD config) | huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer |
| Thinking Machines OPD blog | thinking-machines.ai/blog/on-policy-distillation |
| Rethinking OPD (top-K overlap signal) | arXiv 2604.13016 |
| Qwen3 technical report (OPD recipe) | arXiv 2505.09388 |
| MiniLLM (RKL + PPO for distillation) | arXiv 2306.08543 |
| Off-policy CoT collapse (SN97 case study) | [`paper/off_policy_cot_collapse.md`](../paper/off_policy_cot_collapse.md) |

---

If you're stuck or seeing unexpected eval results, file a Discord thread in the SN97 server or post on the public dashboard. Honest reports of failure modes help the eval improve and benefit every miner. Good luck.
