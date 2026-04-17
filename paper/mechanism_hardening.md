# Mechanism Hardening Roadmap — SN97

Goal: structure incentives so that a miner who maximally overfits to the
validator produces a *genuinely good* 4B distillation of the teacher, not a
teacher-token-statistics mimic. Related goal: retire arbitrary magic numbers
(``THINK_PROBE_LOOP_NGRAM_HITS = 15``, ``THINK_PROBE_LOOP_THRESHOLD = 0.5``)
in favor of statistical tests against the teacher's own distribution.

---

## Root cause recap

Current scoring: teacher-forced top-128 sparse KL on teacher-generated
continuations. Pure off-policy distillation. The literature (Gudibande et al.
*False Promise of Imitating Proprietary LLMs*, arXiv:2305.15717; Lu et al.
*On-Policy Distillation*, Thinking Machines 2025; Agarwal et al. *GKD*,
arXiv:2306.13649) shows that this optimum is *reachable by degenerate students
that generate only teacher-token-statistics-matching filler in teacher-forced
contexts and cannot stand up to autoregressive rollout*. We observed this
exact failure on UID 107 — "Hi" produces 4096 tokens of ``*Wait, I'll write:*``
with no answer, while benchmark perplexity still looks fine because
``enable_thinking=False`` hides the loop.

Gao et al. (*Scaling Laws for Reward Model Overoptimization*, PMLR 2023) and
the *Catastrophic Goodhart* result (arXiv:2407.14503) make this explicit: KL
regularization alone cannot prevent Goodhart when the proxy error is
heavy-tailed — which teacher-forced KL obviously is.

---

## Phase 1 — Threshold-free degeneracy gate (shipped in this commit)

Replace the hand-picked `_detect_phrase_loop` thresholds with a distributional
test. For each probe sample we compute:

| metric           | meaning                                          | reference                  |
|------------------|--------------------------------------------------|----------------------------|
| ``gzip_ratio``   | `len(gzip(text))/len(text)` — Kolmogorov proxy   | Holtzman 2019 axis (iii)   |
| ``distinct_{1,2,4}`` | unique k-gram fraction                       | Li & Jurafsky 2016         |
| ``top_kgram_rate`` | most-frequent 6-gram / total 6-grams           | Holtzman axis (i)          |
| ``byte_entropy``   | Shannon entropy of byte distribution           | Holtzman axis (iii)        |

Student metrics are compared against the *teacher's own distribution on the
same prompts* via the robust median/MAD z-score (Iglewicz & Hoaglin 1993).
A student is degenerate when ``gzip_z < -4σ`` or ``top_kgram_z > 4σ``. The
teacher defines normality; the threshold is not a magic constant.

Fallback when no teacher sample is available on the probe prompts: a
scale-free floor of ``gzip_ratio < 0.25`` — statistically impossible on
plain English text of length >128 chars without pathological repetition.

**Why this is principled.** Any 4σ-robust-z-score outlier has (under the
Gaussian-null) p < 0.0001. With 5 probe prompts and one-sided rejection we
get family-wise error ~5 × 10⁻⁴ — one in two thousand false-DQ rate on a
well-behaved student. Tightenable via ``THINK_PROBE_DEGEN_SIGMA``.

---

## Phase 2 — On-policy reverse-KL scoring (spec, not yet shipped)

Primary score stays teacher-forced KL for continuity, but we add an **on-policy
reverse-KL axis**:

```
for each eval prompt x:
    y = student.generate(x)               # greedy, bounded length
    s_lp = student.logprob(y | x)
    t_lp = teacher.logprob(y | x)         # single forward pass
    rkl_token = s_lp - t_lp               # length-normalized
    on_policy_rkl = mean(rkl_token)
```

This is the exact objective used in GKD (arXiv:2306.13649), MiniLLM (Gu et al.,
arXiv:2306.08543), and Thinking Machines' on-policy distillation post. Key
properties:

1. The student must *itself* roll out the sequence — teacher-forcing
   exploit is impossible.
2. Reverse KL is mode-seeking: a student that covers only a subset of the
   teacher's modes scores well *if the covered modes are high-probability
   teacher modes*. Empty filler has low teacher-logprob under autoregressive
   rollout → direct penalty.
3. Compute cost ≈ identical to current path: one teacher forward pass per
   student sample.

Implementation sketch: in ``pod_eval_vllm.py``, before the top-128 sparse KL
stage, sample N=64 student greedy rollouts of length 256 on held-out prompts.
Score their length-normalized teacher NLL. Composite score:

```
composite = α · sparse_kl_teacher_forced + (1-α) · on_policy_rkl
            (smaller is better for both terms)
```

Start with α=0.5, make it configurable. Long-term goal: α → 0 as
on-policy signal proves itself.

See also *DistiLLM* (Ko et al., arXiv:2402.03898) for skew-KL stabilization
and *DistiLLM-2* (arXiv:2503.07067) for the contrastive teacher/student-sampled
decomposition.

---

## Phase 3 — Pareto-frontier multi-axis aggregation

Replace the scalar composite with a rank-aggregated Pareto test. A new
challenger defeats the king when it *Pareto-dominates* on all axes, or wins a
Borda-count aggregation. Rank aggregation (Copeland/Borda) is invariant to
monotone rescaling of any individual axis, so "pump one metric" attacks
cannot win.

Proposed axes (all cheap on a single GPU):

1. ``rkl_on_policy`` — §Phase 2
2. ``sparse_kl_tf`` — current teacher-forced score (kept as a smoothness
   regularizer)
3. ``degen_gate_pass_rate`` — fraction of probe samples inside the
   teacher's band (§Phase 1)
4. ``capability_under_perturbation`` — IFEval + GSM8K-mini + a fixed "chat"
   battery, with prompt paraphrase/few-shot-reorder perturbations (per
   Zhou et al. IFEval, arXiv:2311.07911 and Retro-Holdouts, arXiv:2410.09247)
5. ``diversity_vs_archive`` — 1 − max_similarity(student_outputs, archive)
   where the archive contains output embeddings of the last K scored models
   (Self-BLEU generalization — Zhu et al. *Texygen*, arXiv:1802.01886)
6. ``independent_lm_ppl`` — perplexity of student samples under a sibling
   model (e.g. Qwen3-8B-Base). Breaks the "match teacher statistics" optimum
   because an independent LM gives a parallel truth.

Aggregate via **Copeland**: for each pair of miners (i, j), count how many
axes i beats j on. Sum wins. This is the Chatbot Arena style (Chiang et al.
ICML 2024; Zheng et al. arXiv:2306.05685) applied to model-vs-model instead
of human-vs-human.

---

## Phase 4 — Private rotating prompt pool + commit-reveal

Current: block-seeded public prompts.
Problem: miners can reproduce the exact prompt set from the block hash and
overfit to it.

Fix:

1. Weights commit-reveal: miner posts a hash of weights on-chain; actual
   weights only uploaded to HF after the validator has finalized the eval
   prompts for that round.
2. Validator maintains a private prompt pool of size P=5000, partitioned into
   (round_seed, held_out_twin) pairs (TRUCE, arXiv:2403.00393). Each round
   samples from the pool with per-validator entropy.
3. Held-out twins (Retro-Holdouts, arXiv:2410.09247): a statistically matched
   private twin of every public benchmark. Any large divergence between
   public and twin scores is a sandbagging/contamination signal.

Cost: 1× teacher-generation pass per prompt, amortized over many rounds.
Security: miners learn the distribution of prompts, not the individual
prompts; overfitting to distribution is fine (that's just fitting the task).

---

## Phase 5 — Adversarial distinguishing-prompt adversary

Optional, higher effort. A second validator-side model (e.g. a ~7B LM) is
prompted to generate prompts that *maximize teacher-vs-student output
divergence under autoregressive rollout* — i.e., the prompts where failures
are loudest. See *AutoRed* (arXiv:2510.08329) and *FLRT* (arXiv:2407.17447)
for attack-generation recipes. These become a dedicated axis #7 in the
Pareto aggregation.

Without ML: a manually curated battery of ~200 short prompts known to
crush "always-think" miners (greetings, one-word-answer prompts, format
constraints, refusal probes). Zero-cost to maintain, high signal.

---

## Phase 6 — Incentive smoothing & anti-sniping

Orthogonal but related: several of the *timing* games the current mechanism
is vulnerable to — e.g. submitting a copy of the current king right before
king-change, or coordinating coldkey-splits to game the Pareto axis — can
be dampened with:

* **Age-weighted scoring**: a model's effective score is `score × (1 -
  exp(-age_in_blocks/τ))`, so scores are not final until the model has
  survived at least τ blocks of re-evaluation.
* **Coldkey-diversity penalty**: if two UIDs on the same coldkey are both in
  the Pareto front, only the higher-ranking one counts.
* **Commit-block tiebreaker** (partially already implemented): earlier
  commit wins all shard-level ties, preventing copy-front-runs.

---

## Implementation priority

1. **Phase 1 — degeneracy gate.** Shipped.
2. **Phase 2 — on-policy RKL axis.** 1-2 days of work in
   ``pod_eval_vllm.py``; biggest expected quality win.
3. **Phase 4 — commit-reveal + private pool.** 2-3 days of work, protects
   against prompt-set overfitting and is cheap.
4. **Phase 3 — Pareto aggregation.** 1-2 days of work after Phases 2 & 4
   are live; primarily validator-side scoring change.
5. **Phase 5 — adversarial prompts.** Nice-to-have.
6. **Phase 6 — smoothing.** Incremental, can be done last.

---

## References (bibliography)

1. Holtzman et al. *The Curious Case of Neural Text Degeneration*, arXiv:1904.09751 (2019).
2. Li & Jurafsky. *A Diversity-Promoting Objective Function for Neural Conversation Models*, NAACL 2016.
3. Pillutla et al. *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers*, arXiv:2102.01454 (2021).
4. Gu et al. *MiniLLM: Knowledge Distillation of Large Language Models*, arXiv:2306.08543 (2023).
5. Agarwal et al. *Generalized Knowledge Distillation (GKD)*, arXiv:2306.13649 (2023).
6. Gudibande et al. *The False Promise of Imitating Proprietary LLMs*, arXiv:2305.15717 (2023).
7. Gao et al. *Scaling Laws for Reward Model Overoptimization*, PMLR 2023.
8. Ko et al. *DistiLLM: Towards Streamlined Distillation*, arXiv:2402.03898 (2024).
9. *Catastrophic Goodhart*, arXiv:2407.14503 (2024).
10. Wu et al. *Rethinking KL in Knowledge Distillation*, arXiv:2404.02657 (2024).
11. Zhou et al. *IFEval: Instruction-Following Evaluation*, arXiv:2311.07911 (2023).
12. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685 (2023).
13. *TRUCE: Benchmark Contamination Detection*, arXiv:2403.00393 (2024).
14. *Retro-Holdouts*, arXiv:2410.09247 (2024).
15. *DistiLLM-2*, arXiv:2503.07067 (2025).
16. Zhu et al. *Texygen: Benchmarking Text Generation Platform*, arXiv:1802.01886 (2018).
17. Iglewicz & Hoaglin. *How to Detect and Handle Outliers*, ASQC 1993.
18. Lu et al. *On-Policy Distillation*, Thinking Machines Lab, Oct 2025.
