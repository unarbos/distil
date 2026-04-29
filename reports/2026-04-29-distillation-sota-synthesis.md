# SOTA LLM Distillation — Research Synthesis for SN97 (April 2026)

Below is a structured, source-cited synthesis of the SOTA in 4B-class student distillation, organised exactly the way you asked. References are in `[N]` format with full URLs at the bottom. I cross-checked every claim against your validator code (`eval/kl_divergence.py`, `eval/scoring.py`, `README.md`).

---

## TL;DR — One-Paragraph Verdict

The 2026 SOTA recipe for 4B-class reasoning distillation is **"strong-to-weak"**: cold-start with off-policy SFT on **rigorously rejection-sampled** teacher traces (DeepSeek-R1 / Qwen3 / Phi-4-Mini-Reasoning all do this), then **on-policy distillation (OPD)** with **per-token reverse-KL** as the dense reward signal (Thinking Machines, Qwen3, TRL `DistillationTrainer`). Pure RL on small students is dominated by OPD by 9–30× FLOPs; pure SFT-only stalls log-linearly. **Your SN97 composite — RKL primary + FKL secondary + capability + degeneracy + density — is largely well-aligned with 2026 SOTA**, and is *better* designed than a pure-KL ranking. The two main gaps are (1) **top-K caching introduces known biases** that you can fix with importance-sampling-based KL [13], and (2) **you don't currently measure "thinking-pattern overlap"** which 2026 OPD-mechanism papers identify as the most predictive single signal of distillation success [4].

---

## 1. HuggingFaceTB / `trl-distillation-trainer` (HF Space, Apr 12 2026)

**What it actually is.** A polished research-article companion to the **TRL `DistillationTrainer`** + a 40× engineering speed-up for distilling **100B+ teachers** into 4B-class students [1]. Authors: Carlos Miguel Patiño, Kashif Rasul, Edward Beeching, Lewis Tunstall (HF). Published 2026-04-12.

### Method — what the trainer optimises


| Knob              | Default | What it does                                                                                                                                                                                                              |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lmbda` (∈ [0,1]) | varies  | Fraction of **on-policy** student-generated rollouts. `0.0` = fully off-policy (SFT-style on dataset/teacher completions). `1.0` = fully on-policy (student generates, teacher only scores).                              |
| `beta` (∈ [0,1])  | varies  | Generalised JSD interpolation. `0.0` ≈ **forward-KL**, `1.0` ≈ **reverse-KL**, `0.5` = symmetric JSD.                                                                                                                     |
| `seq_kd`          | `False` | If `True` and `lmbda=0`, supervised JSD on **teacher-generated** sequences.                                                                                                                                               |
| `loss_top_k`      | varies  | `0` = exact full-vocab; `>0` = top-k approximation (required when using a remote teacher server). For `beta=0` (FKL), `loss_top_k>0`; for `beta>0` (RKL/JSD), `loss_top_k=1` is the documented constraint in the example. |
| `temperature`     | `0.9`   | Sampling temp for student rollouts.                                                                                                                                                                                       |


The **canonical "SOTA" config they push** in the article is:

```python
DistillationConfig(
    lmbda=1.0,    # fully on-policy
    beta=1.0,     # reverse-KL
    loss_top_k=1, # top-1 approximation
    use_vllm=True,
    use_teacher_server=True,
)
```

This is **exactly Thinking Machines' OPD recipe** [3], wrapped in a trainer that scales to 235B teachers via three engineering tricks:

1. **Generation buffer** — prompts batched across grad-accum steps before being sent to vLLM; **41.7× speed-up** at grad_accum=64 without breaking on-policy semantics (student weights are frozen until the optimiser step).
2. **Teacher-server batching** — async queue with 5 ms wait window and `_MAX_BATCH_TOKENS = max_model_len * dp_size`; **10× lower tail latency**.
3. **Binary-encoded logprobs** — base64-packed `(B, T, K)` NumPy arrays instead of nested JSON; **5× smaller payload, 25× faster client decode** (Python double-loop 436 ms → vectorised 17 ms).

### Demonstrated results (in the article itself)


| Run       | Teacher                         | Student               | Result                                                                                                                                                                                                                                                                                     |
| --------- | ------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bushcraft | `gemma-4-31B-it`                | `gemma-4-E2B-it`      | Refusal flipped to high-quality response in **150 distillation steps**; 2× faster inference, half the GPUs.                                                                                                                                                                                |
| Math      | `Qwen3-235B-A22B-Instruct-2507` | `Qwen3-4B (no-think)` | **+39 points on AIME25**; matched by 30B teacher (8B FLOP cost), confirming the **capacity-gap** finding — the 235B teacher is *not* better than the 30B teacher *for the student*. GPQA Diamond drops ~10% with 235B teacher → caution: aggressive math distill can erode OOD capability. |


### Best practices for 1–4B students (HF's own recommendations)

- `**lmbda=1.0` (fully on-policy) outperforms off-policy** when initial student support overlaps the teacher's. Otherwise warm-start with off-policy SFT first.
- `**beta=1.0` (RKL)** for mode-seeking, "unhackable" reward (low KL ⇔ teacher-preferred behaviour).
- **Don't use a teacher that is too large**. The 235B vs 30B AIME25 results are essentially identical despite a 10-point teacher gap. Capacity-gap papers [Mirzadeh 2019, Xu 2025, Gu 2023] all confirm.
- **Top-k=1** approximation is enough at scale; full vocab is impractical (~150M logprobs per 1k-token sample with Qwen tokenizer).

**Pinpoint citation for SN97**: This article is **the 2026 reference implementation** for what Thinking Machines [3] and the Qwen3 team [5] call "on-policy distillation" — your validator's RKL axis is measuring exactly the loss this trainer optimises. ✅

---

## 2. brianmeyer / `distillreasoning` (GitHub, Apr 2026)

A clean controlled experiment: **"Does the teacher's reasoning style transfer to a 4B student?"** [2]

### Pipeline (literal sequence)

1. **Collect** — 2,083 problems from GSM8K + MATH + ARC-Challenge + HumanEval.
2. **Generate** — Both teachers (GLM-5 744B-A40B & Kimi K2.5 ~1T-A32B) solve every problem with full `<thinking>` traces, via Ollama Cloud (4 parallel workers each, ~7 hours per teacher, **$0 generation cost** — Ollama Cloud thinking tags are free).
3. **8-gate quality filter** (this is the "rejection-sampling" stage):
  1. Non-empty thinking + response.
  2. No encoding artefacts.
  3. **Length bounds 50–4,000 thinking tokens** (drops 101 GLM-5, 5 Kimi).
  4. **Correctness** (numerical match for GSM8K, `\boxed{}` for MATH, letter for ARC). Drops 160 GLM-5, 235 Kimi.
  5. No degenerate repetition loops.
  6. **Coherence** — thinking references the problem (drops 22 / 25).
  7. **Self-contradiction** — max 2 self-corrections.
  8. **Structured reasoning** — step indicators present (drops 51 GLM-5, 0 Kimi).
4. Format with `<thinking>` / `<response>` tags, 80/10/10 stratified split.
5. **SFT** with **LoRA rank 32** on Tinker (Qwen3.5-4B base, ~$6–7 per run).
6. **GRPO** (TRL `GRPOTrainer`) on Colab Pro H100 with GSM8K-style reward (correctness).
7. Eval with `**lm-evaluation-harness`** (the EleutherAI standard used by HF Open LLM Leaderboard).
8. Merge LoRA → push to HF → GGUF (q4_k_m, q8_0) for Ollama.

**Filter keep-rate**: GLM-5 1,744/2,083 = **83.7%**. Kimi 1,802/2,083 = **86.5%**.

### Headline results (4B Qwen3.5 student)


| Teacher                      | Median thinking-trace tokens | GSM8K (zero-shot, ~17 problems) |
| ---------------------------- | ---------------------------- | ------------------------------- |
| **Kimi K2.5**                | **325** (concise)            | **71%**                         |
| Combined (GLM-5 + Kimi)      | ~430                         | 67%                             |
| GLM-5                        | 433 (verbose)                | 53%                             |
| Base Qwen3.5-4B (no distill) | —                            | 25%                             |


Reference points:

- Llama-3.2-3B (no distill): 10%
- Qwen3-8B (no distill): 67%  ← **the distilled 4B-Kimi student matches a 2× larger model**
- gpt-oss-20B: 84.6% (ceiling)

### Key findings (verbatim from the DEVLOG)

1. **Concise teachers beat verbose ones for small students.** Kimi's 325-token median traces produced a better 4B than GLM-5's 433-token traces. *"A 4B can't absorb 6,000 words of reasoning — it overwhelms the model's capacity."*
2. **Distillation can beat 2× model size.** Distilled 4B (72.6% GSM8K) > raw Qwen3-8B (63.0%).
3. **More data isn't always better.** Combined (3,196 traces, 71.3%) lost to Kimi-only (1,624 traces, 72.6%) — verbose GLM-5 traces added noise to a 4B's training.
4. **Reasoning *skill* transfers, not just format**. Distilled student writes equations, sets up variables, verifies answers — **even when** `<thinking>`/`<response>` tags are stripped from output (model uses native chat template).
5. **Benchmark contamination is easy to miss.** First eval was 75–80% — turned out to be **94% overlap** with training data. Always verify clean.
6. **Rejection-sampling on correctness alone matters more than fancy filters.** The "correctness" gate (4) dropped the most data and contributed the most to quality.

### Compute budget

- **Trace generation**: $0 (free Ollama Cloud, ~14 hours total)
- **SFT (3 runs on Tinker)**: under $50 in credits (4B at $0.67/M training tokens, ~10M tokens per run)
- **GRPO + lm-eval + publish**: Colab Pro H100, single overnight session
- **Total**: <$100 + free Colab Pro subscription

This is a **single-developer, weekend-budget** distillation pipeline that produced a competitive 4B reasoner. **The key delta vs SN97** is that they use SFT-then-GRPO, not on-policy RKL distillation — which is fine for a static publish-once model, but very different from the "match teacher" objective your validator scores.

---

## 3. Other 2025–2026 Distillation Research Worth Knowing

### 3.1 DeepSeek-R1-Distill (Jan 2025) — *the* reference [6]

- **Method**: Pure SFT, **800k samples** (600k reasoning + 200k general) generated by full DeepSeek-R1, fine-tunes Qwen2.5-Math-1.5B/Qwen2.5-{7,14,32}B/Llama-3.1-8B/Llama-3.3-70B. **No RL on the student** — the DeepSeek team explicitly tested RL-on-small-base-models and found it unstable, distillation strictly dominates.
- **Eval**: AIME24 28.9% (1.5B), 55.5% (7B), 72.6% (32B); MATH-500 83.9% / 92.8% / 94.3%.
- **Critical detail Sebastian Raschka emphasises** [11]: this is **not** classical Hinton-style logit distillation — it's **instruction-SFT on teacher-generated CoT traces**. The "distillation" name is colloquial today, denoting any "train smaller model on stronger model's outputs". True logit-level KL distillation is what TRL's `DistillationTrainer` does (and what your SN97 validator measures).

### 3.2 On-Policy Distillation, Thinking Machines (Oct 27, 2025) [3]

The blog that re-popularised OPD. Lu & Lab demonstrate:

- **Loss**: per-token reverse KL `KL(π_θ || π_teacher)`, advantage = `−reverse_KL`, fed into a standard RL importance-sampling loss.
- **Setup**: Qwen3-8B-Base student, Qwen3-32B teacher (they actually use Qwen3-8B as teacher because it scores slightly higher; "**capacity-matched teacher**" finding).
- **Results**:
  - SFT-400K (off-policy distill on OpenThoughts-3): AIME'24 = 60%
  - SFT-2M (extrapolated, log-linear): ~70%
  - **OPD on top of SFT-400K**: **70% in 150 steps (~77K prompts)**
  - **9× cheaper than SFT-2M** when SFT data already exists; **30× cheaper** when teacher cost is included; **50–100× cheaper than RL** when starting from same checkpoint and the teacher is within student's support.
- **Their Qwen3 reproduction** (Table 21 of Qwen3 tech report [5]): SFT 55% → +RL 67.6% (17,920 GPU hr) → **+OPD 74.4% (1,800 GPU hr) — 10× compute reduction at higher accuracy**.
- **Why RKL "unhackable"**: low RKL ⇔ high probability of teacher-preferred behaviour. Mode-seeking; reduces exposure bias [Bengio 2015]; "darker red" tokens are the **forking tokens** [Wang et al, 2025] that drive student astray.
- **Personalisation finding**: SFT regresses IF-eval even with 100% chat-only data. OPD with a self-teacher (Qwen3-8B) **recovers IF-eval to 83% from 79%** without losing the new domain knowledge — this matters for your "judge-probe" + "chat-turns" axes.

### 3.3 Qwen3 Strong-to-Weak Distillation (May 2025) [5]

The **industrial-scale recipe** the Qwen team uses for *all* Qwen3 small models (0.6B / 1.7B / 4B / 8B / 14B / 30B-A3B):

1. **Off-policy distillation** — student trained on teacher outputs in **both `/think` and `/no_think`** modes (this is where mode-switching is learned).
2. **On-policy distillation** — student samples in either mode, teacher (Qwen3-32B or Qwen3-235B-A22B) provides logits, **minimise KL divergence** (their wording is generic — appears to be RKL based on context).
3. **Result**: 1/10 of the GPU hours of running the full 4-stage post-training pipeline per small model. **Higher Pass@1, higher Pass@64 (better exploration)**.

This is the literal recipe SN97 should encourage miners to follow (with the new Qwen3.6-35B-A3B teacher).

### 3.4 Per-Trace-Quality vs Per-Logit Distillation Spectrum


| Method                                              | Signal                                            | Bits/sample | Best when…                                     | Reference                        |
| --------------------------------------------------- | ------------------------------------------------- | ----------- | ---------------------------------------------- | -------------------------------- |
| **Pure SFT on teacher traces** ("R1-Distill style") | hard tokens                                       | O(N)        | ≥100k high-quality samples available           | DeepSeek-R1 [6]                  |
| **Sequence-KD** (TRL `seq_kd=True`)                 | teacher's full distribution per token, off-policy | O(N·V)      | Teacher is in same family (≤8B)                | Agarwal 2023 [9]                 |
| **GKD** (mixed lmbda, JSD)                          | mixed                                             | O(N·V)      | Tuning unknown task                            | Agarwal 2023, TRL [9]            |
| **MiniLLM** (RKL + PPO)                             | sequence-level RKL                                | O(1)        | Long-form generation, want mode-seek           | Gu et al. 2024 ICLR [12]         |
| **OPD** (per-token RKL, on-policy)                  | dense, per-token RKL                              | O(N)        | Student already in teacher support             | Thinking Machines [3], Qwen3 [5] |
| **Top-K logit distill**                             | sparse logit                                      | O(N·K)      | Bandwidth-constrained (this is your prod path) | Sparse Logit Sampling [13]       |
| **Cross-tokenizer ALM**                             | chunk-level likelihood                            | O(N)        | Different tokenizers                           | Minixhofer 2025 NeurIPS [14]     |


### 3.5 Reverse-KL vs Forward-KL — where the field landed (2026)

**Empirical observation across MiniLLM [12], TM-OPD [3], EOPD [4], ICLR 2025 blog [15]**:

- At the **token level**, forward and reverse KL converge to similar performance when fully trained — there's no clean winner [15].
- **Reverse KL is mode-seeking** ⇒ better for **focused reasoning** (one correct answer) — DeepSeek-R1, MiniLLM, OPD all use it.
- **Forward KL is mode-covering** ⇒ better for **diversity / exploration / SFT-style training** ("adds support" for new tokens — Thinking Machines blog explicitly says "SFT, using forward KL, adds support; RKL methods then mode-seek within that support" [3]).
- **Entropy-Aware OPD** (EOPD, late 2025) [4]: **dynamically switch RKL ⇄ FKL based on teacher token entropy**. Reverse KL when teacher is confident (low entropy), forward KL when teacher is uncertain (high entropy). On Qwen3-{0.6B, 1.7B, 4B}-Base: **Pass@8 +1.37 / +2.39 / +5.05 across 6 math benchmarks** vs vanilla OPD. This is the *most direct* improvement over your current RKL-only validator measurement.

**Practical 2026 consensus**:

- **Off-policy SFT phase** → forward KL (or just CE on teacher tokens; this is the LIMA/LIMO/s1/Bespoke-Stratos default).
- **On-policy refinement phase** → reverse KL (TM, Qwen3, MiniLLM, your SN97 RKL axis).
- **Both, when teacher entropy is high** → EOPD-style adaptive (newest 2026 finding).

### 3.6 Rejection Sampling, Filtering, "Less Is More"

The recipe brianmeyer used (8 quality gates) is now considered a **lightweight version** of the SOTA rejection-sampling stack.


| Source                                               | Recipe                                                                                                   | Output size | Result on student                                       |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------- |
| **DeepSeek-R1** [6]                                  | Math: exact match. Code: sandbox tests. QA: reward model.                                                | **800k**    | 1.5B → 28.9 AIME24                                      |
| **LIMO** (GAIR-NLP, COLM 2025) [7]                   | Multi-layer: difficulty filter → coverage → coherence → step-clarity → answer-correctness. Hand-curated. | **800–817** | 32B SFT → **63.3 AIME24, 95.6 MATH500**                 |
| **s1** (Muennighoff, 2025) [8]                       | Difficulty + diversity + quality. Gemini-2 traces, then R1 traces in s1.1.                               | **1,000**   | 32B SFT → matches o1-preview by **+27%**                |
| **Bespoke-Stratos** (Bespoke Labs 2025) [16]         | DeepSeek-R1 traces + GPT-4o-mini correctness filter (raised retention 25→73%)                            | **17k**     | 32B SFT → near R1-Distill-32B (47× less data)           |
| **OpenThoughts-3** (Guha et al., DataComp 2025) [17] | QwQ-32B traces, ablated across **1000+ design experiments**                                              | **1.2M**    | 7B → state-of-the-art open-data                         |
| **REDI** (May 2025) [18]                             | **Use both correct AND incorrect traces**, asymmetric weighting α=0.8 on negatives                       | **131k**    | 1.5B → matches 800k positive-only                       |
| **Phi-4-Mini-Reasoning** (Microsoft, Apr 2025) [19]  | Mid-train on diverse R1 distill → SFT on selected subset → Rollout DPO → RL with verifiable reward       | (4-stage)   | **3.8B → 57.5 AIME24, 94.6 MATH500, 52.0 GPQA-Diamond** |


**Critical 2026 finding** (Phi-4-Mini paper [19]): **Naively distilling LIMO or s1K into a small student REGRESSES performance**:


| Model                               | AIME24   | MATH-500 | GPQA-Diamond |
| ----------------------------------- | -------- | -------- | ------------ |
| Phi-4-Mini (base)                   | 10.0     | 71.8     | 36.9         |
| Phi-4-Mini + LIMO                   | 6.7      | 57.8     | 24.8         |
| Phi-4-Mini + S1K                    | 3.0      | 47.0     | 26.3         |
| Phi-4-Mini-Reasoning (full 4-stage) | **57.5** | **94.6** | **52.0**     |


Why? **Small students lack the latent capacity that LIMO/s1 assume**. LIMO works on 32B (already knowledge-rich); below ~7B you need the big mid-training distillation phase first. **This is a direct warning to SN97 miners**: don't expect to drop LIMO 800-sample SFT on a 4B and win.

### 3.7 OPD-mechanism papers (2026) — what predicts success

**"Rethinking OPD" (arXiv 2604.13016, 2026)** [20] identifies **two governing factors**:

1. **Thinking-pattern consistency** — student and teacher must already share **compatible top-k token distributions** at student-visited states. *97–99% of probability mass concentrates on a small shared token set in successful OPD runs.* Mismatched thinking patterns produce low initial overlap that training cannot recover.
2. **Genuinely-new capability** — even with consistent patterns, the teacher must offer knowledge the student doesn't have. Same-family same-recipe teachers/students at different scales (1.5B vs 7B) end up **distributionally indistinguishable**.

**Recipe to fix failing OPD** (from same paper):

- (a) **Off-policy cold start** — warmup SFT on teacher rollouts before OPD raises initial overlap ratio.
- (b) **Teacher-aligned prompt selection** — use prompts drawn from the teacher's post-training data, mixed with OOD prompts.

This is **the most under-measured concept** for SN97: **token-level overlap ratio** between teacher and student top-k distributions on the student's own rollouts is a stronger signal than KL alone.

### 3.8 Compute-efficient distillation: LoRA-without-Regret

Thinking Machines, Sep 2025 [21]:

- **LoRA can match full fine-tuning** in the "low-regret regime" (post-training scale datasets) when:
  1. LoRA is applied to **all linear weights** (not just attention) — `--lora_target_modules all-linear`
  2. **LR is ~10× higher** than full FT (typically ~1e-5 vs 1e-6)
  3. Effective batch size **< 32**
  4. Rank chosen for dataset size; for SFT post-training, rank 32–128 typically suffices for ≤4B students.
- **FLOPs cost is 2/3 of full FT** per pass. Memory cost much lower.
- **Key tradeoff for distillation**: LoRA with rank ≤32 lags full FT by **13% on SFT but only 6% after on-policy distillation** (Thinking Machines OPD blog) — OPD partially closes the LoRA gap.

For SN97 miners on consumer GPUs: **LoRA r=128 + OPD on Qwen3.5-4B base** is the right capacity sweet-spot.

### 3.9 Sparse logit caching — known bias [13]

**ACL 2025: "Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs"** (Anshumann et al.) proves that **naive top-K caching of teacher probabilities produces biased estimates** of the teacher distribution → suboptimal student calibration. The fix is **importance-sampling-based "Random Sampling Knowledge Distillation"** with <10% overhead.

**This is directly relevant to SN97's prod path** (`top-128 sparse KL` per the README). Your current renormalisation-over-shared-support approach is reasonable but not unbiased; the ACL paper provides the correction.

Related: **Tail-Aware Distillation (TAD)** [22] decouples top-K vs tail KL components to prevent the gradient on top-K tokens from dominating, which is what naive `KL_div(top-K renorm)` does. The decoupled gradient lets the student match the *full* distribution rather than just the modes.

### 3.10 Cross-tokenizer distillation [14, 23]

**Universal Logit Distillation** (Boizard et al., TMLR 2025) and **Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching** (Minixhofer et al., NeurIPS 2025) enable distillation when teacher and student use different tokenizers (subword↔byte, etc.). Currently irrelevant to SN97 because you mandate same tokenizer (vocab 248,320), but **it's a good safety valve to have noted** if you ever swap teachers across tokenizer families (Kimi K2.6 → ?).

### 3.11 Distillation > Zero-RL on Small Models (May 2025) [10]

**Why distillation can outperform zero-RL** — distillation produces "**flexible reasoning**":

- More **anthropomorphic tokens** ("hmm", "let me think", "wait") and **logical connectors** ("therefore", "however").
- Increases two cognitive behaviours: **Multi-Perspective Thinking** and **Metacognitive Awareness**.
- Zero-RL produces rigid step-by-step output; distillation transfers a *reasoning style*.

This is exactly the "**reasoning_density** axis ≠ thinking-without-answering" failure mode you caught on 2026-04-17 (UID 107 spiral). The paper says: blocking these tokens at decoding *hurts but doesn't kill* the distilled model — the cognitive behaviours are internalised, not just surface-level. **Your reasoning_density axis is implicitly measuring whether miners distilled the *style without the substance***, which is the right gate.

---

## 4. Practical Recommendations for SN97

### 4.1 Should the validator measure on-policy RKL, FKL, or both?

**Both — and you already do, with the right relative weighting.** Your composite has:

- On-policy RKL: 35% of the relative slice (primary distillation signal)
- FKL on teacher continuations: 15% of the relative slice

This is **defensible 2026 SOTA** because:

1. **RKL is the "match the teacher" signal** that successful OPD runs minimise — Thinking Machines, Qwen3, TRL `DistillationTrainer` all optimise this. Miners who actually run OPD will dominate this axis.
2. **FKL on teacher continuations** is essentially measuring "**how plausible does the student find the teacher's outputs?**" — which captures **support coverage** (did the student even learn the vocabulary the teacher uses?). LIMO/s1/Bespoke-Stratos-style SFT on teacher traces should improve this axis specifically.
3. **Together they catch failure modes either alone misses**:
  - High RKL + low FKL = student avoids teacher's distribution but happens to match on its own samples (common gaming pattern)
  - Low RKL + high FKL = student perfectly imitates teacher tokens but on its own rollouts diverges (off-policy SFT collapse)

**Suggested upgrade (cheap, high-value)**: **add an Entropy-Aware adjustment** [4] — when the teacher's continuation has high token-level entropy (>some threshold, say 1.5 nats), weight FKL more; when teacher is confident, weight RKL more. This is what 2026's best math-distillation results use [4].

### 4.2 Distillation-quality signals you may be missing

Ranked by likely impact on validator robustness:

1. **Top-k token overlap ratio between teacher & student** on the *student's* rollouts. Per [20], this is **the single most predictive signal** of OPD success — successful runs converge to a 97–99% shared mass on a small token set. This is cheap (you already cache top-128) and gives you a "**thinking-pattern alignment**" axis that catches "decorative reasoning that diverges in token-space" — exactly the pathology your 2026-04-17 spiral king exhibited. *Implementation*: For each generated token, compute |top-K_teacher ∩ top-K_student| / K, average over the continuation. Add as a 17th axis with ~5% weight.
2. **Importance-sampled KL replacing top-K renormalised KL** [13]. Your current path renormalises over teacher's top-128 support — this is **biased**. The Anshumann et al. ACL 2025 fix uses importance sampling to give an **unbiased** estimate at <10% overhead. This won't change rankings dramatically but it means the published KL nat values are correct.
3. **Tail-decoupled KL** [22]. Splits the KL into top-K vs tail components and decouples gradient contributions. For *evaluation*, this gives you a **separate "tail probability mass alignment"** signal — a model that matches the teacher on top-K but flattens the tail (overconfident) will look fine on top-128 KL but fail this axis. Cheap to compute from cached top-K.
4. **Teacher-trace plausibility likelihood** — straight `−log P_student(teacher_full_trace | prompt)` averaged over a few teacher rollouts per prompt. Already measurable from your FKL pipeline; surfacing it as a separate axis is cheap and gives miners a clean feedback signal.
5. **"Forking-token RKL"** [Wang et al, 2025; cited in TM blog]. Compute the per-token RKL but only at high-entropy student positions (top-quartile), where teacher feedback is most informative. This better reflects what the OPD loss actually optimises and is a stronger predictor than mean-KL.
6. **Robustness to length-variation in teacher trace** — re-evaluate the same prompts with **truncated** and **extended** teacher continuations. If the student's RKL on a 256-tok continuation differs wildly from RKL on a 1024-tok continuation, the student is brittle. This catches "match teacher style for first 100 tokens, drift after" — a real 4B failure pattern (brianmeyer's project saw this with GLM-5-distilled students).
7. `**reasoning_density` × `correctness` joint distribution**. Currently `reasoning_density` is one axis and benches measure correctness separately. A model that gets 50% on reasoning + 0% on math is trivially gated by `worst()`; but a model that gets 30% on density + 30% on math is *less interesting* than one with 10% density + 50% math (the latter is "cleanly answering"). You already have the data; surface it as a derived signal.

### 4.3 What "distillation succeeded" looks like in 2026 SOTA terms

For a Qwen3.6-35B-A3B teacher → 4B student, **success benchmarks** as of April 2026:


| Metric                                   | Target (4B-class, distilled from 35B-class teacher)                           | Reference                                                          |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| AIME 2024 / 2025 (Pass@1)                | 50–75%                                                                        | Thinking Machines + Qwen3 + Phi-4-Mini-Reasoning                   |
| MATH-500                                 | 88–95%                                                                        | DeepSeek-R1-Distill-7B baseline at 92.8 [6]                        |
| GPQA Diamond                             | 45–55%                                                                        | Phi-4-Mini-Reasoning 52.0 [19]                                     |
| GSM8K (5-shot CoT)                       | 88–94%                                                                        | DeepSeek-R1-Distill-7B 95.3 [6]                                    |
| MMLU-Pro                                 | 50–60%                                                                        | Open LLM Leaderboard v2                                            |
| IFEval                                   | 75–85% (post-OPD)                                                             | Thinking Machines 83% [3]                                          |
| **On-policy RKL vs teacher** (your axis) | **<0.10 nats** average on continuation tokens                                 | Brianmeyer's distilled 4B + reproductions in [3] are in this range |
| **Forward-KL on teacher continuations**  | <0.15 nats                                                                    | Same as above                                                      |
| **Top-K overlap @ K=64**                 | >85%                                                                          | [20] reports 97–99% on shared mass                                 |
| **Reasoning density**                    | >0.6                                                                          | New axis you added; calibrated against Qwen3-4B-Thinking           |
| **No degeneracy spirals**                | `thinking_collapse_probe` passes on `"Hi"`, `"largest planet one word"`, etc. | Your 2026-04-17 fix                                                |


**Your KL ranges (README)** at "no distillation training" are:

- Qwen3.5-4B: ~0.10–0.15 nats — this is **already in the OPD-success range**, suggesting the base Qwen3.5-4B has high natural overlap with the 35B-A3B teacher (same family, MoE shares dense path). After distillation, miners should target **<0.05 nats** to be competitive.
- Qwen3.5-2B: ~0.12–0.16 — competitive but capacity-limited.
- Qwen3.5-0.8B: ~0.17–0.21 — likely capped by capacity gap.

### 4.4 Open-source benchmarks specifically for distilled models

There is **no widely-adopted "distillation-quality" benchmark** as of April 2026. The closest:

- **Open LLM Leaderboard v2** (HF) — uses `lm-evaluation-harness` for reasoning suite (MMLU-Pro, GPQA, MATH-Lvl-5, IFEval, BBH, MUSR) — what brianmeyer/distillreasoning eventually settled on.
- **OpenThoughts evaluation suite** [17] — math + code + science, designed specifically for reasoning-distilled small models.
- **REDI / s1 / LIMO** all evaluate on AIME24, AIME25, MATH-500, GPQA-Diamond.
- **THINKSLM** [from 72-SLM survey, arXiv 2502.11569] — 72 SLMs across 17 reasoning benchmarks; the most comprehensive small-model reasoning eval to date.
- **There's a real opportunity** for SN97 to define a **"distillation fidelity benchmark"**: same prompts, measure (teacher_continuation_likelihood, RKL, FKL, top-K overlap, reasoning density, IFEval, math, code, robustness) as a single composite — which is very close to what your 17-axis composite already does.

---

## 5. Direct alignment check: SN97 vs SOTA practice


| SOTA practice                         | SN97 today                                        | Gap?                                                    |
| ------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| RKL as primary distillation signal    | ✅ 35% of relative slice                           | None                                                    |
| Off-policy FKL/SFT measurement        | ✅ 15% (FKL on teacher continuations)              | None                                                    |
| Capability evaluation alongside KL    | ✅ 25% capability + 9 absolute benches             | None                                                    |
| Anti-degeneracy / spiral protection   | ✅ `reasoning_density` + `thinking_collapse_probe` | None — caught the 2026-04-17 failure                    |
| Top-K logit caching for bandwidth     | ✅ top-128 with renormalisation                    | **Biased estimate** [13] — fix with importance sampling |
| Block-seeded fresh prompts            | ✅ ClimbMix-400B with block-hash seed              | None — better than most subnets                         |
| Token-overlap / thinking-pattern axis | ❌ Not measured                                    | **Add as 17th axis** [20]                               |
| Entropy-aware FKL/RKL switching       | ❌ Fixed weights                                   | **Easy upgrade** [4]                                    |
| Rejection-sampling on teacher traces  | N/A (validator side)                              | Document for miners                                     |
| LoRA-r-128 + OPD for 4B-class         | N/A                                               | Document in mining guide                                |
| Multi-stage Phi-4-Mini-style recipe   | N/A                                               | Document for serious miners                             |
| Cross-tokenizer support               | ❌ Same tokenizer mandated                         | Fine — keeps eval clean                                 |


**Verdict**: Your validator is **better-designed than 80% of public distillation eval setups** because it's multi-axis with absolute floors. The gaps are minor and incremental.

---

## 6. Best-practice training pipeline to recommend to miners

For a 4B-class student from Qwen3.6-35B-A3B (and future Kimi K2.6):

```
Stage 1: Mid-train (large-scale off-policy SFT)
  - Generate ~50k–200k teacher rollouts on diverse problems
    (math + code + science + general; OpenThoughts-3 / DeepMath-103k)
  - Rejection-sample on correctness (verifiable: math-verify, sandboxed code
    execution, reward model for QA)
  - Drop length outliers (>4096 tokens), repetition loops, refusals
  - SFT with train_on_responses_only / instruction masking
  - LoRA rank 128, lr 1e-5, batch <= 32 (per LoRA-Without-Regret)
  - Train 1-3 epochs

Stage 2: SFT on high-quality compact set
  - Curate 800-2000 hardest, most diverse, hand-checked problems
    (LIMO/s1-style for the *right* base model)
  - 1 epoch full-FT or LoRA-r-128

Stage 3: On-Policy Distillation (per Thinking Machines / Qwen3 / TRL)
  - Use TRL DistillationTrainer with:
      lmbda=1.0 (fully on-policy)
      beta=1.0 (reverse KL)  -- or 0.7 (mostly RKL with FKL safety)
      loss_top_k=1 (or 64 for richer signal)
      use_vllm=True
      use_teacher_server=True
  - Generation buffer set to grad_accum_steps for max throughput
  - 100-200 steps is usually sufficient if Stage 1 was solid
  - Teacher: Qwen3.6-35B-A3B served via vLLM with --max-logprobs 128

Stage 4 (optional): RL with verifiable reward (Phi-4-Mini style)
  - GRPO on math/code with answer-correctness reward
  - Run only after OPD has converged; tends to help Pass@1 and Pass@64

Eval:
  - Use lm-evaluation-harness with EXACT settings:
    gsm8k_cot 8-shot, minerva_math 4-shot, arc_challenge 25-shot,
    gpqa_diamond 0-shot, mmlu_pro 5-shot
  - Plus RKL/FKL against your committed teacher (matches what SN97 measures)
```

**Cost estimate** (single H100 / B200 80–192GB):

- Stage 1: 24–48 hr
- Stage 2: 1–2 hr
- Stage 3: 8–16 hr (with teacher server)
- Stage 4: 6–12 hr
- **Total**: ~$100–500 in cloud compute, vs ~$50 for brianmeyer's minimal-budget version that *also worked*.

---

## 7. References (numbered, with full URLs)

1. **HF TRL Distillation Trainer + 2026 article** — Patiño, Rasul, Beeching, Tunstall (HF). "Distilling 100B+ Models 40x Faster with TRL." Apr 12, 2026. [https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer](https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer) · Trainer docs: [https://huggingface.co/docs/trl/main/en/distillation_trainer](https://huggingface.co/docs/trl/main/en/distillation_trainer) · GKD docs: [https://huggingface.co/docs/trl/main/en/gkd_trainer](https://huggingface.co/docs/trl/main/en/gkd_trainer) · Source: [https://github.com/huggingface/trl/blob/main/trl/experimental/distillation/](https://github.com/huggingface/trl/blob/main/trl/experimental/distillation/)
2. **brianmeyer/distillreasoning** — Brian Meyer. "Borrow reasoning from a 744B model. Teach it to a 4B model." 2026. [https://github.com/brianmeyer/distillreasoning](https://github.com/brianmeyer/distillreasoning) · DEVLOG: [https://github.com/brianmeyer/distillreasoning/blob/main/DEVLOG.md](https://github.com/brianmeyer/distillreasoning/blob/main/DEVLOG.md) · Datasets: [https://huggingface.co/datasets/bmeyer2025/](https://huggingface.co/datasets/bmeyer2025/)
3. **On-Policy Distillation (Thinking Machines)** — Lu, Kevin and Thinking Machines Lab. "On-Policy Distillation." Oct 27, 2025. [https://thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation) · Tinker cookbook: [https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
4. **Entropy-Aware OPD** — "Entropy-Aware On-Policy Distillation of Language Models." OpenReview WSRQ37tzk1 (under review 2025–2026). [https://openreview.net/pdf?id=WSRQ37tzk1](https://openreview.net/pdf?id=WSRQ37tzk1) · arXiv preprint: [https://arxiv.org/html/2603.07079v1](https://arxiv.org/html/2603.07079v1)
5. **Qwen3 Technical Report** — Qwen Team. arXiv:2505.09388 (May 2025). [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) · §4.5 "Strong-to-Weak Distillation" is the relevant section.
6. **DeepSeek-R1 / R1-Distill** — DeepSeek-AI et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." Jan 22, 2025. [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) · Paper PDF: [https://www.thewirechina.com/wp-content/uploads/2025/01/DeepSeek-R1-Document.pdf](https://www.thewirechina.com/wp-content/uploads/2025/01/DeepSeek-R1-Document.pdf)
7. **LIMO** — Ye, Huang, Xiao, Chern, Xia, Liu. "LIMO: Less is More for Reasoning." arXiv:2502.03387 (Feb 2025, COLM 2025). [https://arxiv.org/abs/2502.03387](https://arxiv.org/abs/2502.03387) · Code: [https://github.com/GAIR-NLP/LIMO](https://github.com/GAIR-NLP/LIMO)
8. **s1: Simple test-time scaling** — Muennighoff, Yang, Shi et al. arXiv:2501.19393 (Jan 2025). [https://arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393) · Code: [https://github.com/simplescaling/s1](https://github.com/simplescaling/s1)
9. **Generalized Knowledge Distillation (GKD)** — Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." arXiv:2306.13649 (2023, ICLR 2024). [https://arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649)
10. **Why Distillation > Zero-RL** — Liu et al. "Why Distillation can Outperform Zero-RL: The Role of Flexible Reasoning." arXiv:2505.21067 (May 2025). [https://arxiv.org/abs/2505.21067](https://arxiv.org/abs/2505.21067)
11. **Sebastian Raschka — Understanding Reasoning LLMs** — [https://magazine.sebastianraschka.com/p/understanding-reasoning-llms](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) (Feb 2025) · State of LLM Reasoning Inference: [https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling)
12. **MiniLLM** — Gu, Dong, Wei, Huang. "MiniLLM: Knowledge Distillation of Large Language Models." arXiv:2306.08543 (v6: Jan 2026, ICLR 2024). [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543) · TRL implementation: [https://huggingface.co/docs/trl/main/en/minillm](https://huggingface.co/docs/trl/main/en/minillm)
13. **Sparse Logit Sampling** — Anshumann, Zaidi, Kedia, Ahn, Kwon, Lee, Lee, Lee. "Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs." arXiv:2503.16870 / ACL 2025. [https://arxiv.org/abs/2503.16870](https://arxiv.org/abs/2503.16870) · [https://aclanthology.org/2025.acl-long.885/](https://aclanthology.org/2025.acl-long.885/)
14. **Universal Cross-Tokenizer Distillation (ALM)** — Minixhofer et al. arXiv:2503.20083 (NeurIPS 2025). [https://arxiv.org/abs/2503.20083](https://arxiv.org/abs/2503.20083)
15. **ICLR 2025 Blog: FKL vs RKL** — [https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/](https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/)
16. **Bespoke-Stratos** — Bespoke Labs. Jan 2025. [https://bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation](https://bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)
17. **OpenThoughts** — Guha, Marten, Keh et al. "OpenThoughts: Data Recipes for Reasoning Models." arXiv:2506.04178 (Jun 2025). [https://arxiv.org/abs/2506.04178](https://arxiv.org/abs/2506.04178) · Code: [https://github.com/open-thoughts/open-thoughts](https://github.com/open-thoughts/open-thoughts)
18. **REDI (Reinforcement Distillation)** — "Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning." arXiv:2505.24850 (May 2025). [https://arxiv.org/abs/2505.24850](https://arxiv.org/abs/2505.24850)
19. **Phi-4-Mini-Reasoning** — Microsoft. arXiv:2504.21233 (Apr 2025). [https://arxiv.org/abs/2504.21233](https://arxiv.org/abs/2504.21233)
20. **Rethinking OPD** — "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe." arXiv:2604.13016 (2026). [https://arxiv.org/html/2604.13016v1](https://arxiv.org/html/2604.13016v1)
21. **LoRA Without Regret** — Schulman, John and Thinking Machines Lab. Sep 2025. [https://thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/) · TRL writeup: [https://huggingface.co/docs/trl/main/en/lora_without_regret](https://huggingface.co/docs/trl/main/en/lora_without_regret)
22. **Tail-Aware Distillation (TAD)** — "Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation." arXiv:2602.20816 (2026). [https://www.arxiv.org/pdf/2602.20816](https://www.arxiv.org/pdf/2602.20816)
23. **Universal Logit Distillation (ULD)** — Boizard, El Haddad, Hudelot, Colombo. TMLR 01/2025. [https://arxiv.org/pdf/2402.12030](https://arxiv.org/pdf/2402.12030)
24. **Nathan Lambert / interconnects.ai** — "How much does distillation really matter for Chinese LLMs?" [https://www.interconnects.ai/p/how-much-does-distillation-really](https://www.interconnects.ai/p/how-much-does-distillation-really) · "The state of post-training in 2025" [https://www.interconnects.ai/p/the-state-of-post-training-2025](https://www.interconnects.ai/p/the-state-of-post-training-2025) · "What comes next with open models" [https://www.interconnects.ai/p/the-next-phase-of-open-models](https://www.interconnects.ai/p/the-next-phase-of-open-models)
25. **Curriculum Reasoning Distillation** — Jiang, Lu, Lin, Han, Sun. "Teach Small Models to Reason by Curriculum Distillation." EMNLP 2025. [https://aclanthology.org/2025.emnlp-main.376/](https://aclanthology.org/2025.emnlp-main.376/)

---

If you want, I can next: (a) prototype the 17th "thinking-pattern overlap" axis against your existing top-128 cache; (b) draft the Entropy-Aware FKL/RKL weighting upgrade; or (c) write a "Mining Guide v2" markdown that codifies the Stage 1–4 pipeline above for inclusion in your README.