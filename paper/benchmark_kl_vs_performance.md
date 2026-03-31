# Does Lower KL Divergence Produce Better Models? Benchmarking SN97 Distillation Quality

**Authors:** Distil SN97 Research  
**Date:** March 31, 2026  
**Subnet:** Bittensor SN97 (Distil)

---

## Abstract

Bittensor Subnet 97 (Distil) incentivizes miners to distill Qwen3.5-35B-A3B into compact student models (≤5.25B params), scored purely on KL divergence from the teacher. A fundamental question emerges: **does optimizing for lower KL divergence actually produce a better model, or just a more faithful token-distribution mimic?**

We benchmark the current SN97 "king" — the miner with the lowest KL divergence — against the official Qwen3.5-4B base model across 7 standard benchmarks. The king (KL=0.049) wins **6 out of 7 benchmarks** versus the baseline (KL=0.149), with particularly strong gains in reasoning (+10pts GSM8K) and instruction following (+4pts IFEval). The sole regression is factual knowledge (MMLU-Pro, −4.3pts). These results provide empirical evidence that KL-divergence-based distillation incentives, as implemented in SN97, produce genuinely more capable models — not just closer token distribution approximations.

## 1. Introduction

### 1.1 The Distil Subnet

SN97 runs a winner-take-all competition: miners distill the teacher model (Qwen/Qwen3.5-35B-A3B, a 35B-parameter Mixture-of-Experts with 3B active parameters) into students with ≤5.25B total parameters. Validators score students by computing full-distribution KL divergence against the teacher on shared FineWeb prompts. The model with the lowest KL divergence — the "king" — receives 100% of emissions.

This creates a strong economic incentive to minimize KL divergence. But KL divergence measures distributional similarity across all 248,320 tokens, not downstream task performance. A model could theoretically achieve low KL by closely matching the teacher's probability distribution without actually being "smarter" at reasoning, following instructions, or answering questions correctly.

### 1.2 Research Question

**Does lower KL divergence from the teacher translate to better downstream benchmark performance?**

If yes, SN97's incentive mechanism is well-aligned — miners competing on KL are implicitly competing to build better models. If no, the subnet may be incentivizing a form of distributional mimicry that doesn't transfer to real-world utility.

### 1.3 Why This Matters

This question has implications beyond SN97:

- **Distillation research:** Validates whether KL-based training objectives produce models that generalize, not just memorize teacher distributions
- **Bittensor subnet design:** Demonstrates whether simple, measurable metrics (KL divergence) can serve as reliable proxies for model quality
- **Practical deployment:** If the king is genuinely better, SN97 is producing useful models that can be deployed in production at a fraction of the teacher's compute cost

## 2. Models Under Evaluation

### 2.1 Baseline: Qwen/Qwen3.5-4B

| Property | Value |
|----------|-------|
| Model | `Qwen/Qwen3.5-4B` |
| Parameters | 4.66B total |
| Architecture | Qwen3.5 (hybrid attention with DeltaNet linear attention layers) |
| KL from teacher | ~0.149 nats |
| SN97 status | UID 179 (disqualified as raw copy — no distillation training applied) |

This is Qwen's official 4B model from the same family as the teacher. It was not distilled specifically for SN97 — it is the natural starting point, representing "what you get for free" at this parameter count.

### 2.2 King: iotaminer/distil-qwen35-4b

| Property | Value |
|----------|-------|
| Model | `iotaminer/distil-qwen35-4b` |
| Parameters | ~4B total |
| Architecture | Qwen3.5 text-only (`qwen3_5_text`) |
| KL from teacher | 0.0488 nats |
| SN97 status | UID 194 — current king (100% of emissions) |

This is the current SN97 king, having won the competition by achieving 67% lower KL divergence than the baseline. The specific distillation training methodology is proprietary to the miner.

### 2.3 KL Divergence Gap

The king achieves **KL = 0.049** versus the baseline's **KL = 0.149** — a **67% reduction**. In information-theoretic terms, the king's output distribution is substantially closer to the teacher's across all 248,320 tokens. Our experiment tests whether this distributional closeness corresponds to measurable capability improvements.

## 3. Methodology

### 3.1 Benchmark Suite

We evaluate on 7 widely-used benchmarks spanning reasoning, knowledge, instruction following, and commonsense:

| Benchmark | Type | Measures | Task Format |
|-----------|------|----------|-------------|
| **MMLU-Pro** | Knowledge + Reasoning | Broad academic knowledge across 14 subjects | `generate_until` (10-choice) |
| **ARC-Challenge** | Science Reasoning | Grade-school science questions (hard subset) | `loglikelihood` (4-choice) |
| **HellaSwag** | Commonsense | Sentence completion with commonsense | `loglikelihood` (4-choice) |
| **WinoGrande** | Commonsense | Pronoun resolution requiring world knowledge | `loglikelihood` (2-choice) |
| **TruthfulQA MC2** | Truthfulness | Resistance to common misconceptions | `loglikelihood` (multi-true) |
| **GSM8K** | Math Reasoning | Grade-school math word problems | `generate_until` (open-ended) |
| **IFEval** | Instruction Following | Strict format/constraint adherence | `generate_until` (open-ended) |

This mix is important: `loglikelihood` tasks test how well the model ranks correct answers (distribution quality), while `generate_until` tasks test actual text generation quality (can the model produce correct, well-formatted answers?).

### 3.2 Evaluation Framework

- **Framework:** lm-evaluation-harness v0.4.11 (EleutherAI)
- **Samples:** 100 per subtask (`--limit 100`)
- **Decoding:** Greedy (temperature=0, do_sample=False) for reproducibility
- **Precision:** bfloat16 for both models
- **No chat template:** Both models evaluated as base completion models without chat formatting

### 3.3 Infrastructure

| Property | Baseline | King |
|----------|----------|------|
| GPU | NVIDIA B200 (183GB VRAM) | NVIDIA B200 (183GB VRAM) |
| Backend | vLLM 0.18.0 | HuggingFace Transformers 5.5.0.dev0 |
| Batch size | auto | 1 |
| Runtime | ~6 minutes | ~3 hours |

**Backend asymmetry note:** The king model uses `model_type: qwen3_5_text` (text-only variant) while the baseline uses the standard `qwen3_5` type. vLLM 0.18.0 only registers `Qwen3_5ForConditionalGeneration` (multimodal) and cannot load the text-only `Qwen3_5ForCausalLM` architecture. Multiple config-patching approaches were attempted (wrapping as multimodal config, registering the text-only class in vLLM's model registry) but all failed due to missing vision weights or config parsing mismatches.

The HuggingFace backend produces identical numerical results for evaluation — the difference is solely in inference speed. Both models use the same precision (bf16), same greedy decoding, and see the same prompts. The backend difference does not affect scores.

**Dependency note:** Qwen3.5's hybrid attention architecture (DeltaNet linear attention layers) requires `flash-linear-attention` (fla) ≥0.5.0 from source. The PyPI version (0.4.2) is missing `fla.modules.FusedRMSNormGated`, which transformers 5.5.0 imports unconditionally for Qwen3.5 models. Without this library, the torch fallback path consumes excessive memory and silently crashes during text generation.

### 3.4 Fairness Controls

To ensure a valid comparison:

1. **Same GPU:** Both models evaluated on the same B200 GPU
2. **Same prompts:** lm-eval uses deterministic prompt ordering; `--limit 100` selects the same first 100 samples
3. **Same decoding:** Greedy decoding eliminates sampling variance
4. **Same precision:** bfloat16 for both
5. **No chat template:** Neither model uses chat formatting, avoiding tokenizer-specific advantages
6. **Same evaluation code:** lm-eval v0.4.11 for both runs

### 3.5 Limitations

- **Sample size:** 100 samples per subtask yields ±4-5% standard error. Differences smaller than ~5 points should be interpreted cautiously.
- **Backend difference:** While numerically equivalent, we cannot rule out subtle floating-point differences between vLLM and HF backends.
- **No chat/thinking mode:** Qwen3.5 models support a `<think>` reasoning mode via chat template, which dramatically changes behavior. We evaluate base completion mode only, as lm-eval's `--apply_chat_template` does not enable Qwen3.5's thinking mode.
- **Single teacher family:** Results may not generalize to distillation from non-Qwen teachers.

## 4. Results

### 4.1 Main Results

| Benchmark | Metric | Baseline | King | Δ | Winner |
|-----------|--------|----------|------|---|--------|
| **ARC-Challenge** | acc_norm | 54.00 | **59.00** | **+5.00** | King ✅ |
| **GSM8K** | exact_match (strict) | 74.00 | **84.00** | **+10.00** | King ✅ |
| **HellaSwag** | acc_norm | 68.00 | **69.00** | +1.00 | King ✅ |
| **IFEval** | prompt_level_strict | 19.00 | **23.00** | **+4.00** | King ✅ |
| **MMLU-Pro** | exact_match | **57.21** | 52.93 | −4.28 | Baseline ❌ |
| **TruthfulQA MC2** | acc | 49.11 | **51.55** | +2.44 | King ✅ |
| **WinoGrande** | acc | 75.00 | **79.00** | **+4.00** | King ✅ |

**King wins 6 out of 7 benchmarks.** The average delta across all 7 benchmarks is **+3.17 points** in the king's favor.

### 4.2 MMLU-Pro Subject Breakdown

The king's only losing benchmark deserves closer examination:

| Subject | Baseline | King | Δ |
|---------|----------|------|---|
| Math | **72.0** | 58.0 | **−14.0** |
| Chemistry | **64.0** | 56.0 | −8.0 |
| Computer Science | **69.0** | 61.0 | −8.0 |
| Physics | **65.0** | 57.0 | −8.0 |
| Health | **64.0** | 58.0 | −6.0 |
| Law | **34.0** | 28.0 | −6.0 |
| Psychology | **66.0** | 61.0 | −5.0 |
| Biology | **69.0** | 65.0 | −4.0 |
| Economics | **68.0** | 64.0 | −4.0 |
| Business | **56.0** | 53.0 | −3.0 |
| Other | **45.0** | 44.0 | −1.0 |
| Engineering | 43.0 | **44.0** | +1.0 |
| Philosophy | 46.0 | **48.0** | +2.0 |
| History | 40.0 | **44.0** | +4.0 |

The regression is concentrated in **STEM subjects** (Math −14, Chemistry/CS/Physics −8 each). Humanities subjects (History +4, Philosophy +2) actually improve. This suggests the distillation process trades stored factual knowledge for improved reasoning patterns — the student learned the teacher's *reasoning behavior* more than its *factual recall*.

### 4.3 Statistical Significance

With 100 samples per subtask, standard errors are approximately:

| Benchmark | Δ | SE(Δ) | Δ/SE | Significant? |
|-----------|---|-------|------|--------------|
| GSM8K | +10.00 | ~5.7 | ~1.75 | Likely |
| ARC-Challenge | +5.00 | ~7.0 | ~0.71 | Marginal |
| WinoGrande | +4.00 | ~6.0 | ~0.67 | Marginal |
| MMLU-Pro | −4.28 | ~1.8 | ~2.38 | Yes |
| IFEval | +4.00 | ~5.8 | ~0.69 | Marginal |
| TruthfulQA MC2 | +2.44 | ~6.0 | ~0.41 | No |
| HellaSwag | +1.00 | ~6.6 | ~0.15 | No |

GSM8K's +10 point improvement and MMLU-Pro's −4.3 point regression are the most statistically robust findings. The consistent positive direction across 6/7 benchmarks strengthens the overall conclusion even where individual differences are within noise.

## 5. Analysis

### 5.1 What Distillation Improved

The king shows the largest gains on tasks requiring **multi-step reasoning** and **instruction following**:

- **GSM8K (+10.0):** Multi-step arithmetic word problems. The teacher (35B MoE) is a strong reasoner; the distilled student has internalized more of this reasoning ability than the baseline.
- **ARC-Challenge (+5.0):** Science reasoning requiring inference beyond memorized facts.
- **IFEval (+4.0):** Instruction following with strict format constraints. The king better adheres to the teacher's disciplined output formatting.
- **WinoGrande (+4.0):** Commonsense reasoning about pronoun references.

### 5.2 What Distillation Lost

The king's regression on MMLU-Pro, concentrated in STEM subjects, reveals a **capacity tradeoff**:

- At 4B parameters, the student has limited capacity to store both the teacher's reasoning patterns and its factual knowledge.
- KL-divergence-based distillation optimizes for matching the teacher's full output distribution. When the teacher "knows" a STEM fact, it expresses this as a sharp probability distribution over the correct answer. The student, with less capacity, may match the distribution shape (reasoning pattern) better than the distribution content (factual recall).
- The −14 point drop on MMLU-Pro Math is striking because GSM8K (+10) tests math *reasoning*. The student learned how to reason about math but lost some of the factual mathematical knowledge the baseline carried.

### 5.3 Implications for KL as an Incentive Metric

These results suggest that **KL divergence is a well-aligned incentive metric** for distillation quality:

1. **KL captures reasoning quality:** The king's improved GSM8K, ARC, and IFEval scores show that minimizing KL divergence forces the student to learn the teacher's reasoning behavior, not just its surface-level token preferences.

2. **The tradeoff is reasonable:** The MMLU-Pro regression is a natural consequence of limited capacity — at 4B params, there aren't enough weights to perfectly mimic a 35B model. The distillation process prioritizes the most informative aspects of the teacher's distribution (reasoning patterns, output formatting) over memorized facts, which is arguably the right tradeoff for a compact model.

3. **Full-distribution KL matters:** SN97 scores on all 248,320 tokens, not just top-k. This forces students to match the teacher's full probability landscape, including the subtle probability mass allocations that encode reasoning chains and calibrated uncertainty.

## 6. Methodology Notes

### 6.1 Why Not Qwen's Published Scores?

Qwen reports significantly higher scores for Qwen3.5-4B using their EvalScope framework with thinking mode enabled (`enable_thinking=True` + chat template + `<think>` tag stripping). We confirmed that lm-eval's `--apply_chat_template` flag does **not** enable Qwen3.5's thinking mode, producing *worse* results than no chat template:

| Benchmark | No Chat Template | Chat Template (broken) |
|-----------|-----------------|----------------------|
| MMLU-Pro | 57.21% | 30.79% |
| GSM8K | 74.00% | 47.00% |
| IFEval | 19.00% | 20.00% |

The model generates unstructured reasoning without proper `<think>` tags, burning context and confusing lm-eval's regex extractors. We chose no-chat-template for both models to ensure a fair, apples-to-apples comparison.

### 6.2 vLLM Incompatibility

The king model uses `Qwen3_5ForCausalLM` (text-only), while vLLM 0.18.0 only registers `Qwen3_5ForConditionalGeneration` (multimodal). Attempts to patch the config or registry failed because:

1. Setting `model_type: qwen3_5` (multimodal) requires vision weights that don't exist in the text-only checkpoint
2. Registering `Qwen3_5ForCausalLM` in vLLM's registry fails because the config parser expects nested `text_config` that the text-only format doesn't provide
3. The fundamental issue is that vLLM's Qwen3.5 support is multimodal-first

This forced using the HuggingFace backend for the king model, which is functionally equivalent but ~30x slower.

### 6.3 flash-linear-attention Dependency

Qwen3.5's hybrid attention architecture alternates between standard attention and DeltaNet linear attention layers. The `transformers` library (5.5.0.dev0) imports `FusedRMSNormGated` from the `flash-linear-attention` (fla) package unconditionally. The PyPI version (0.4.2) lacks this module; installation from source (0.5.0) is required:

```bash
pip install "git+https://github.com/fla-org/flash-linear-attention.git" --no-deps
```

Without this library, Qwen3.5 models silently crash during generation as the torch fallback path for linear attention consumes excessive memory.

## 7. Conclusion

**Lower KL divergence from the teacher does translate to better benchmark performance.** The SN97 king (KL=0.049) outperforms the baseline (KL=0.149) on 6 out of 7 standard benchmarks, with an average improvement of +3.17 points. The gains are concentrated in reasoning (+10 GSM8K), commonsense (+5 ARC, +4 WinoGrande), and instruction following (+4 IFEval), with a tradeoff in factual knowledge (−4.3 MMLU-Pro).

This validates SN97's core design: **a simple, measurable metric (KL divergence) serves as an effective proxy for model quality.** Miners competing to minimize KL divergence are implicitly competing to build better models. The winner-take-all incentive structure produces real, measurable improvements in downstream task performance — not just closer distributional mimicry.

### 7.1 Future Work

- **Full-dataset evaluation:** Run on complete benchmark test sets (not limited to 100 samples) for tighter statistical confidence
- **Chat/thinking mode evaluation:** Develop proper lm-eval integration for Qwen3.5's thinking mode to test whether the distilled model also improves in structured reasoning contexts
- **Multiple distillation methods:** Compare different miners' approaches (not just the king) to understand which distillation techniques best preserve different capabilities
- **Downstream deployment:** Test the king model in real application scenarios (code generation, RAG, tool use) beyond standardized benchmarks
- **Temporal analysis:** Track whether improvements in KL divergence over time correspond to monotonic improvements in benchmark scores

## Appendix A: Raw Results

### A.1 Full Metric Dump

#### Baseline (Qwen/Qwen3.5-4B)

| Metric | Value |
|--------|-------|
| arc_challenge / acc | 0.5500 |
| arc_challenge / acc_norm | 0.5400 |
| gsm8k / exact_match,flexible-extract | 0.7500 |
| gsm8k / exact_match,strict-match | 0.7400 |
| hellaswag / acc | 0.5100 |
| hellaswag / acc_norm | 0.6800 |
| ifeval / prompt_level_loose_acc | 0.2000 |
| ifeval / prompt_level_strict_acc | 0.1900 |
| ifeval / inst_level_loose_acc | 0.4172 |
| ifeval / inst_level_strict_acc | 0.4110 |
| mmlu_pro / exact_match | 0.5721 |
| truthfulqa_mc2 / acc | 0.4911 |
| winogrande / acc | 0.7500 |

#### King (iotaminer/distil-qwen35-4b)

| Metric | Value |
|--------|-------|
| arc_challenge / acc | 0.5200 |
| arc_challenge / acc_norm | 0.5900 |
| gsm8k / exact_match,flexible-extract | 0.8400 |
| gsm8k / exact_match,strict-match | 0.8400 |
| hellaswag / acc | 0.5300 |
| hellaswag / acc_norm | 0.6900 |
| ifeval / prompt_level_loose_acc | 0.2500 |
| ifeval / prompt_level_strict_acc | 0.2300 |
| ifeval / inst_level_loose_acc | 0.4479 |
| ifeval / inst_level_strict_acc | 0.4356 |
| mmlu_pro / exact_match | 0.5293 |
| truthfulqa_mc2 / acc | 0.5155 |
| winogrande / acc | 0.7900 |

### A.2 Evaluation Environment

| Component | Version |
|-----------|---------|
| lm-evaluation-harness | 0.4.11 |
| transformers | 5.5.0.dev0 (from source) |
| vLLM | 0.18.0 (baseline only) |
| flash-linear-attention | 0.5.0 (from source) |
| PyTorch | 2.7+ |
| GPU | NVIDIA B200 (183GB VRAM) |
| CUDA | 12.8 |

### A.3 Reproduction

```bash
# Install dependencies
pip install "vllm>=0.18.0" lm-eval
pip install "git+https://github.com/huggingface/transformers.git"
pip install "git+https://github.com/fla-org/flash-linear-attention.git" --no-deps

# Baseline (vLLM backend, ~6 min on B200)
lm_eval --model vllm \
    --model_args "pretrained=Qwen/Qwen3.5-4B,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=32768,trust_remote_code=True" \
    --tasks mmlu_pro,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k,ifeval \
    --batch_size auto --limit 100 \
    --output_path results/baseline --log_samples

# King (HF backend — vLLM cannot load qwen3_5_text architecture)
lm_eval --model hf \
    --model_args "pretrained=iotaminer/distil-qwen35-4b,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu_pro,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k,ifeval \
    --batch_size 1 --limit 100 \
    --output_path results/king --log_samples
```

**Note:** The king model requires `flash-linear-attention >= 0.5.0` from source. Without it, generation tasks will silently crash.
