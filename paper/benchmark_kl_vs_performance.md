# Does Lower KL Divergence Produce Better Models? Benchmarking SN97 Distillation Quality

**Authors:** Distil SN97  
**Date:** April 12, 2026 (v2 — full-dataset benchmarks, updated methodology)  
**Subnet:** Bittensor SN97 (Distil)  
**Dashboard:** [distil.arbos.life](https://distil.arbos.life) | **Chat:** [chat.arbos.life](https://chat.arbos.life)

---

> **⚠️ Historical document — superseded.** This paper analyses a 2026-04-12 era when SN97 scored miners on **KL alone**. It documented a positive correlation between lower KL and better held-out benchmarks at the time. That correlation **broke down** five days later: on 2026-04-17, the KL-only ranking crowned UID 107 (`gtensorapp/prime-dusk-4260`), whose 4B student looped indefinitely on `"Hi"` (4096-token cap, 102× repetition of a 6-word phrase) and was *strictly worse than the unfine-tuned 4B base on 5/5 reasoning benchmarks*. The diagnosis is in [`off_policy_cot_collapse.md`](off_policy_cot_collapse.md). The fix — a 17-axis composite where KL is 0.15 of the relative tier and `composite.worst` is the gate — is in [`mechanism_hardening.md`](mechanism_hardening.md), [`composite.py`](../scripts/validator/composite.py), and the production v28 schema. Read this paper for the historical record; do **not** read it as a guide to current SN97 mining.

---

## Abstract

Bittensor Subnet 97 (Distil) incentivizes miners to distill Qwen3.5-35B-A3B into compact student models (≤5.25B params), scored purely on KL divergence from the teacher. A fundamental question emerges: **does optimizing for lower KL divergence actually produce a better model, or just a more faithful token-distribution mimic?**

We benchmark distilled "king" models against the official Qwen3.5-4B base model across 8 standard benchmarks using **full evaluation sets** (no subsampling — MMLU alone uses ~14,000 questions). The king model (UID 170, KL=0.049) wins **6 of 8 benchmarks**, with strong gains in reasoning (BBH +13.3%, GSM8K +8.4%), knowledge (MMLU +5.9%), and truthfulness (TruthfulQA +5.8%). The sole regressions are instruction following (IFEval −6.3%) and commonsense (Winogrande −0.4%, within noise). These results provide empirical evidence that KL-divergence-based distillation incentives produce genuinely more capable models — not just closer token distribution approximations.

The king model is freely available for interactive testing at [chat.arbos.life](https://chat.arbos.life).

## 1. Introduction

### 1.1 The Distil Subnet

SN97 runs a winner-take-all competition: miners distill the teacher model (Qwen/Qwen3.5-35B-A3B, a 35B-parameter Mixture-of-Experts with 3B active parameters) into students with ≤5.25B total parameters. Validators score students by computing top-128 sparse KL divergence against the teacher on shared prompts from the ClimbMix-400B dataset — teacher returns top-128 logprobs per position, student softmaxes over the full 248,320-token vocab then gathers + renormalizes to the shared top-128 support. The model with the lowest KL divergence — the "king" — receives 100% of emissions.

Dethronement requires statistical significance: a one-sided paired t-test at p<0.01 on ~300 shared prompts must demonstrate that the challenger has consistently lower KL divergence than the current king.

### 1.2 Research Question

**Does lower KL divergence from the teacher translate to better downstream benchmark performance?**

If yes, SN97's incentive mechanism is well-aligned — miners competing on KL are implicitly competing to build better models. If no, the subnet may be incentivizing distributional mimicry that doesn't transfer to real-world utility.

### 1.3 Why This Matters

- **Distillation research:** Validates whether KL-based training objectives produce models that generalize
- **Bittensor subnet design:** Demonstrates whether simple, measurable metrics can serve as reliable proxies for model quality
- **Practical deployment:** If the king is genuinely better, SN97 produces useful models deployable at a fraction of the teacher's compute cost

## 2. Models Under Evaluation

### 2.1 Baseline: Qwen/Qwen3.5-4B

| Property | Value |
|----------|-------|
| Model | `Qwen/Qwen3.5-4B` |
| Parameters | 4.66B total |
| Architecture | Qwen3.5 (hybrid attention with DeltaNet linear attention layers) |
| KL from teacher | ~0.149 nats |

Qwen's official 4B model from the same family as the teacher. Not distilled for SN97 — represents the natural starting point at this parameter count.

### 2.2 King: UID 170 — QuiteLLM/sn97-test-ver26

| Property | Value |
|----------|-------|
| Model | `QuiteLLM/sn97-test-ver26` |
| Parameters | ~4B |
| KL from teacher | 0.0488 nats |
| SN97 status | Former king (UID 170), dethroned April 8 |

First king to hold the crown through the architecture enforcement transition. Full 10-benchmark suite completed.

### 2.3 King: UID 252 — ncaagcc/sn97-q8rn

| Property | Value |
|----------|-------|
| Model | `ncaagcc/sn97-q8rn` |
| Parameters | ~4B |
| KL from teacher | 0.066381 nats |
| SN97 status | Current king (UID 252) as of April 11 |

Current king, first legitimate holder after the 401-exploit fix. Full 8-benchmark suite in progress.

## 3. Methodology

### 3.1 Benchmark Suite

| Benchmark | Few-shot | Type | Measures |
|-----------|----------|------|----------|
| **MMLU** | 5-shot | Knowledge | Broad academic knowledge across 57 subjects |
| **GSM8K** | 5-shot | Math | Grade-school math word problems |
| **BBH CoT** | 3-shot | Reasoning | BIG-Bench Hard with chain-of-thought |
| **HellaSwag** | 10-shot | Commonsense | Sentence completion |
| **Winogrande** | 5-shot | Commonsense | Pronoun resolution |
| **ARC-Challenge** | 25-shot | Science | Grade-school science (hard subset) |
| **TruthfulQA MC2** | 0-shot | Truthfulness | Resistance to common misconceptions |
| **IFEval** | 0-shot | Instruction Following | Strict format/constraint adherence |

All benchmarks use **full evaluation sets** (no `--limit` subsampling). MMLU alone contains ~14,000 questions. Loglikelihood tasks report `acc_norm` where available.

### 3.2 Evaluation Framework

- **Framework:** lm-evaluation-harness v0.4.11 (EleutherAI)
- **Model backend:** `local-completions` via vLLM API
- **Samples:** Full benchmark datasets (no subsampling)
- **Decoding:** Greedy (temperature=0) for reproducibility
- **Precision:** bfloat16
- **Batch size:** 1 (to avoid OOM on larger benchmarks)

### 3.3 Infrastructure

- **GPU:** NVIDIA H200 (144GB VRAM) on Lium
- **Inference server:** vLLM serving the model via OpenAI-compatible API
- **Memory allocation:** `--gpu-memory-utilization 0.30` (~44GB for 4B model)
- **Max sequence length:** 32,768 tokens
- **Concurrent requests:** 4

All models evaluated on the same GPU with the same vLLM server configuration. The vLLM API backend ensures consistent inference conditions across all benchmarks.

### 3.4 Fairness Controls

1. **Same GPU and server:** All models served via the same vLLM instance on the same H200
2. **Same prompts:** lm-eval uses deterministic prompt ordering; full dataset evaluation eliminates sampling variance
3. **Same decoding:** Greedy decoding (temperature=0)
4. **Same precision:** bfloat16 for all models
5. **Full datasets:** No subsampling — results are statistically robust across thousands of questions
6. **Same evaluation code:** lm-eval v0.4.11 for all runs

## 4. Results

### 4.1 UID 170 vs Baseline (Full Suite)

| Benchmark | Metric | Baseline | King (UID 170) | Δ | Winner |
|-----------|--------|----------|----------------|---|--------|
| **MMLU** | acc | 0.698 | **0.739** | **+5.9%** | King ✅ |
| **GSM8K** | exact_match | 0.751 | **0.814** | **+8.4%** | King ✅ |
| **BBH CoT** | exact_match | 0.693 | **0.785** | **+13.3%** | King ✅ |
| **HellaSwag** | acc_norm | 0.730 | **0.737** | +0.9% | King ✅ |
| **Winogrande** | acc | **0.698** | 0.695 | −0.4% | Baseline ❌ |
| **ARC-Challenge** | acc_norm | 0.540 | **0.551** | +2.1% | King ✅ |
| **TruthfulQA MC2** | acc | 0.489 | **0.518** | **+5.8%** | King ✅ |
| **IFEval** | prompt_strict | **0.314** | 0.294 | −6.3% | Baseline ❌ |

**Score: King 6, Baseline 2.** Average improvement across all 8 benchmarks: **+3.7%**.

### 4.2 Key Findings

**Strong reasoning gains:** The largest improvements come in reasoning tasks — BBH (+13.3%) and GSM8K (+8.4%). This is notable because these gains emerge purely from KL-divergence optimization against a MoE teacher, not from task-specific fine-tuning. The student appears to internalize the teacher's reasoning patterns through distributional alignment alone. BBH CoT in particular requires multi-step logical inference — the +13.3% improvement suggests the distilled model captures the teacher's chain-of-thought structure, not just surface-level token probabilities.

**Knowledge compression:** MMLU (+5.9% across 57 subjects, ~14,000 questions) demonstrates that the distilled model retains *more* factual knowledge than the base model at the same parameter count. This is the core value proposition of SN97: competitive pressure drives miners to find distillation techniques that effectively compress 35B worth of knowledge into 4B parameters — producing models that outperform what Qwen achieved with their own official 4B release.

**Truthfulness improvement:** TruthfulQA MC2 (+5.8%) is perhaps the most surprising result. The distilled model is better at resisting common misconceptions than the base model, suggesting that the teacher's calibrated uncertainty over the full vocabulary helps the student learn more nuanced probability distributions over factual claims.

**IFEval regression:** The −6.3% on IFEval (instruction-following format compliance) is the expected tradeoff of distribution-matching distillation. The base model's post-training instruction alignment gets partially overwritten when optimizing for general KL divergence. Future work could explore hybrid objectives that preserve instruction-following while minimizing KL.

**Winogrande within noise:** The −0.4% difference is within the standard error for this benchmark size and is not meaningful.

### 4.3 KL Divergence vs Performance Correlation

| Model | KL from Teacher | Avg Benchmark Score |
|-------|----------------|-------------------|
| Qwen3.5-4B (baseline) | 0.149 | 0.614 |
| UID 170 (former king) | 0.049 | 0.641 |
| UID 252 (current king) | 0.066 | In progress |

The 67% reduction in KL divergence (UID 170 vs baseline) corresponds to a 4.4% average improvement across all benchmarks. This provides evidence of a positive correlation between distributional fidelity and downstream performance.

## 5. Evaluation Pipeline

### 5.1 Dethronement Mechanism

SN97 uses a statistical testing framework to prevent noise from causing false king changes:

1. **Shared evaluation:** ~300 prompts sampled from the ClimbMix-400B dataset, block-seeded for reproducibility
2. **Per-prompt KL:** Both king and challenger produce full logit distributions; KL divergence computed per prompt
3. **Paired t-test:** One-sided paired t-test on the per-prompt deltas (king KL − challenger KL)
4. **Significance threshold:** p < 0.01 required for dethronement (tightened from p < 0.05 on April 11, 2026)
5. **Minimum samples:** At least 20 paired observations required

This means a challenger must demonstrate consistently lower KL divergence across many diverse prompts, not just get lucky on a few.

### 5.2 Anti-Exploitation Measures

- **Revision pinning:** Models are evaluated at the specific committed revision to prevent post-commit weight swaps
- **Revision-based integrity:** HuggingFace repo git SHA is tracked — any new commit to the repo after evaluation triggers DQ (cheaper and faster than re-hashing weights)
- **Architecture enforcement:** All models must use `Qwen3_5ForConditionalGeneration` for vLLM compatibility
- **Fresh score requirement:** Kings must produce a fresh score every round or lose the crown to the best challenger

### 5.3 Reproducibility

All evaluation data (prompts, teacher completions, student scores) is available via the API:

- **Live status:** `https://api.arbos.life/api/status`
- **Leaderboard:** `https://api.arbos.life/api/leaderboard`
- **Eval data:** `https://api.arbos.life/api/eval-data` (latest and historical rounds)
- **H2H history:** `https://api.arbos.life/api/h2h-history`
- **Dashboard:** `https://distil.arbos.life`

## 6. Discussion

### 6.1 KL as a Proxy for Quality

Our results support the hypothesis that KL divergence is a meaningful proxy for model quality. The distilled king model outperforms the base model on 6 of 8 benchmarks, with the largest gains in the most valuable capabilities (reasoning and knowledge). This suggests that SN97's incentive mechanism — pure KL minimization — is well-aligned with producing genuinely useful models.

### 6.2 The Value of Competition

The competitive dynamics of SN97 drive continuous improvement. The king model has changed hands multiple times, with each new king demonstrating lower KL divergence. Our benchmarks show this competitive pressure produces real quality gains, not just incremental distributional tweaks.

### 6.3 Limitations

- **Single architecture family:** All models are Qwen3.5-based. Results may not generalize to other architectures.
- **No chat/thinking mode evaluation:** Qwen3.5 supports a reasoning mode via `<think>` tokens that dramatically changes behavior. We evaluate base completion mode only.
- **IFEval sensitivity:** The instruction-following regression suggests KL distillation may weaken behaviors that were fine-tuned into the base model post-training. Task-specific distillation could address this.
- **Snapshot evaluation:** Benchmarks represent performance at a single point in time. The competitive nature of SN97 means models continue to improve.

## 7. Conclusion

KL divergence optimization in SN97 produces student models that are genuinely more capable than the base model at the same parameter count. The king model outperforms Qwen's own 4B release on 6 of 8 standard benchmarks, with particularly strong gains in reasoning (+13.3% BBH, +8.4% GSM8K) and knowledge (+5.9% MMLU across ~14,000 questions). These results validate that SN97's simple, measurable incentive mechanism translates to real-world model quality improvements.

The subnet effectively crowdsources distillation research: miners compete to compress a 35B MoE teacher into a 4B student, and the competitive pressure produces models that capture teacher knowledge more effectively than Qwen's official distillation. The current king model can be tested interactively at [chat.arbos.life](https://chat.arbos.life).

As the competition matures — with 11 king changes in the first two weeks and increasingly sophisticated distillation techniques from miners — we expect these gains to continue growing. The combination of economic incentives, statistical rigor (p<0.01 dethronement), and anti-exploitation measures (revision pinning, continuous integrity checks) creates a robust framework for decentralized model improvement.

---

## Appendix A: Raw Benchmark Data

### A.1 Baseline — Qwen/Qwen3.5-4B

```
ARC-Challenge (25-shot):     acc_norm = 0.5401
HellaSwag (10-shot):         acc_norm = 0.7303
Winogrande (5-shot):         acc      = 0.6985
TruthfulQA MC2 (0-shot):    acc      = 0.4893
BBH CoT (3-shot):            exact_match = 0.6928
GSM8K (5-shot):              exact_match = 0.7513
IFEval (0-shot):             prompt_level_strict_acc = 0.3142
MMLU (5-shot):               acc      = 0.6980
```

### A.2 King UID 170 — QuiteLLM/sn97-test-ver26

```
ARC-Challenge (25-shot):     acc_norm = 0.5512
HellaSwag (10-shot):         acc_norm = 0.7366
Winogrande (5-shot):         acc      = 0.6946
TruthfulQA MC2 (0-shot):    acc      = 0.5177
BBH CoT (3-shot):            exact_match = 0.7847
GSM8K (5-shot):              exact_match = 0.8143
IFEval (0-shot):             prompt_level_strict_acc = 0.2939
MMLU (5-shot):               acc      = 0.7390
```

### A.3 King UID 252 — ncaagcc/sn97-q8rn

*Benchmarks in progress — results will be appended when complete.*

## Appendix B: Reproduction

To reproduce these benchmarks:

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model <model_name> \
  --port 8100 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.30

# 2. Run lm-eval
lm_eval --model local-completions \
  --model_args "model=<model_name>,base_url=http://localhost:8100/v1/completions,num_concurrent=4,tokenized_requests=False" \
  --tasks mmlu,gsm8k,bbh_cot_fewshot,hellaswag,winogrande,arc_challenge,truthfulqa_mc2,ifeval \
  --num_fewshot <see table above> \
  --batch_size 1 \
  --output_path ./results/
```

GPU: Any NVIDIA GPU with ≥24GB VRAM (RTX 4090 or better recommended for reasonable runtimes).
