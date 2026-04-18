# SN97 Mechanism Hardening Roadmap (v2, April 2026)

**Goal.** Design scoring such that a miner's dominant strategy
(maximize-score-given-rules) produces a *genuinely useful* 4B
distillation of the teacher — not a teacher-token-statistics mimic, not
a logit-matched zombie, not a king that infinite-loops under rollout.

**The headline result from the literature.** The exact pathology we
observe on SN97 today ("wins teacher-forced sparse-top-K KL while
collapsing into infinite CoT under autoregressive rollout") is the
*predicted equilibrium* of off-policy KL scoring. Three 2025 papers
describe this failure precisely:

1. **Teacher Hacking** (Tiapkin et al., ICML 2025, arXiv:2502.02671) — a
   student trained on a fixed offline dataset of teacher samples
   provably exploits imperfections in the teacher's probability
   distribution, and the exploitation is invisible to offline
   teacher-forced KL.
2. **Looping in Reasoning Models** (arXiv:2512.12895, 2025) — distilled
   students loop *substantially more* than their teachers, and the
   effect is worst at ~4B parameters. Not a mechanism bug, a known
   distillation pathology at our scale.
3. **Reward Hacking as Equilibrium under Finite Evaluation**
   (arXiv:2603.28063, 2026) — proves under 5 minimal axioms that *any*
   finite scorer is equilibrium-hacked; no static rule achieves
   "dominant strategy = produce a good model." The best you can do is
   rotate dimensions and cover a broad quality frontier.

So the answer to "how do we pick a threshold so miners can't game KL"
is: *you can't*. Any fixed scalar loss is asymptotically gamed. The
subnet must (a) score on a *broad* frontier of axes that are
individually gameable but jointly impractical to game, (b) *rotate*
which axes matter per tempo, (c) make the hardest-to-fake axes —
on-policy rollout quality and outcome-verifiable rewards — primary.

---

## Axes of hardening

Six axes below. Each axis has an independent implementation cost and an
independent bound on what the miner can do. In combination they
implement the §7 "equilibrium hacking" defense: no single axis can be
dominated without making another axis worse.

Priority column: **P0** = ship this tempo; **P1** = next month; **P2** =
next quarter; **P3** = spec only, implementation-deferred.

| # | Axis | Primary attack prevented | Priority | Key ref |
|---|---|---|---|---|
| A1 | On-policy reverse-KL (student rollout + teacher NLL) | Teacher-forced hacking (Tiapkin) | P1 *(prototype)* | Thinking Machines blog 10/2025 |
| A2 | Autoregressive degeneracy gate (gzip + spectral + Self-BLEU) | CoT collapse | P0 *(shipped)* | Pimentel 2403.00553; Holtzman 1904.09751 |
| A3 | Length penalty on rollouts | Infinite-loop emission-farming | P0 *(shipped, shadow)* | Kimi k1.5 2501.12599 |
| A4 | Verifiable outcome rewards (math/format/factoid) | "Miracle-step" overfitting | P0 *(v2 shipped shadow: rotating per-block + procedural math)* | Tülu 3 2411.15124, Rubric Rewards 2510.07774 |
| A5 | Teacher ensemble worst-case (Qwen + DeepSeek + Llama) | Teacher-specific Goodhart | P1 | Coste 2406.01013 |
| A6 | Weight-fingerprint dedup at registration | Copy-miner / fork-king | P1 | AWM 2510.06738, REEF, DuFFin |
| A7 | Rotating prompt pool + reusable-holdout DP noise | Prompt memorization / reverse-engineering | P0 *(v1 shipped shadow: 198-prompt private holdout, commit-reveal, DP noise scaffolding)* | LiveBench 2406.19314, Dwork 2015 |
| A8 | Dethronement ε(Δblocks) margin | Copy-miner passing t-test by chance | P1 | SN37 + Blum&Hardt "Ladder" 1502.04585 |
| A9 | TOPLOC attestations on miner inference | Model-swap after commitment | P2 | Ong 2501.16007 |
| A10 | Commit-reveal weights v3 + Yuma 3 | Validator weight-copying | P1 | BT weight-copier whitepaper 05/2024 |
| A11 | Behavioral dedup via logit fingerprints | Near-copy with noise | P2 | SN25 pattern + TOPLOC |
| A12 | Adversarial prompt generation (Dynabench style) | King camping on fixed set | P3 | Kiela 2021 |

The current (shipped) probe covers one prompt axis of A2; everything
else is future work.

---

## Axis A1 — On-policy reverse-KL (the single biggest win)

**What.** Instead of scoring teacher-forced sparse-top-K KL on
teacher-generated continuations, score the *student's own rollouts*
under the teacher's distribution:

```
for each prompt x in eval batch:
    y = student.generate(x, T=0.7, max_new_tokens=L)
    s_lp = student.logprob(y | x)        # already have (student produced y)
    t_lp = teacher.logprob(y | x)        # one teacher forward pass
    rkl_token[t] = s_lp[t] - t_lp[t]     # per-token reverse KL
score_A1 = -mean_over_tokens(rkl_token)  # smaller is worse
```

**Why this is the primary fix.**

- **Mode-seeking, not mode-covering.** Gu et al. *MiniLLM* (ICLR 2024,
  arXiv:2306.08543) shows forward KL + off-policy is the literal recipe
  for "student spreads mass across all teacher modes → under
  autoregression it samples absurd low-probability continuations →
  infinite CoT loops." Reverse KL on-policy is the textbook fix.

- **Evaluates on the student's own distribution.** GKD (Agarwal et al.,
  ICLR 2024, arXiv:2306.13649) — the student is *evaluated* on states
  *it* visits, not states the teacher visits. The current train/inference
  mismatch grows with sequence length and that is exactly why our king
  can win KL but fail on greedy "Hi".

- **Empirically dominant.** Thinking Machines' *On-Policy Distillation*
  (Oct 2025, thinkingmachines.ai/blog, DOI 10.64434/tml.20251026)
  reproduces Qwen3's result: on-policy reverse KL hits AIME'24 74.4% at
  **1/10 the GPU hours of RL** and 7–10× fewer gradient steps than RL at
  matched rank.

- **Formally unhackable *if* on-policy.** The blog makes the subtle but
  crucial point: reverse KL is "unhackable in the sense that low KL
  always corresponds to a high probability of desirable behavior from
  the teacher's point of view" — **only** under on-policy sampling.
  Off-policy reverse KL is gameable. So it is specifically the
  student-rollout sampling that kills hacking, not the KL direction
  alone.

**Refinements.**

- **Skew KL** (DistiLLM, ICML 2024, arXiv:2402.03898): use
  D_KL(απ_t + (1−α)π_s ‖ π_t) with α ≈ 0.1 — bounded gradients when
  teacher and student are far apart. Drop-in replacement; more stable.
- **Entropy-aware switching (EOPD)** (arXiv:2603.07079, ICLR 2026):
  reverse KL when teacher entropy < θ, forward KL otherwise. Avoids the
  mode-collapse failure at high-entropy "reasoning forks." +1.4 to +5.1
  Pass@8 on Qwen3-0.6B/1.7B/4B over vanilla on-policy distillation.
  Cheap: `H(π_teacher)` is already computed.
- **Speculative KD / interleaved sampling** (SKD, Xu et al., ICLR 2025,
  arXiv:2410.11325): student proposes each token, teacher replaces
  low-rank ones. A cheaper approximation of full rollout; score = (frac
  teacher-accepted tokens) + KL on accepted tokens.
- **Contrastive term** (DistiLLM-2, Ko et al., ICML 2025 oral,
  arXiv:2503.07067): add a negative-pair term `-log π_teacher(student_sample)`
  that rewards "teacher thinks student's output is plausible."
  Catastrophic for nonsense-rollout miners by construction.

**Compute budget.** N rollouts × L tokens × (student forward) + N × L ×
(teacher forward). On a B200 with N=16, L=256, batched: ~2 s per miner
per prompt pair. Budget ≤ 30 s/miner per round if we parallelize across
prompts. Compare to current teacher-forced KL which takes ~60 s/miner —
this is *cheaper* than what we run today.

---

## Axis A2 — Autoregressive degeneracy gate (shipped, keep refining)

What we shipped in this commit cycle covers the gzip / top-k-gram /
byte-entropy + robust-z-score multi-metric test on a small "thinking"
probe. Next iterations:

- **Extend to full eval rollouts**, not just probe prompts. Every prompt
  in the eval batch gets a rollout, every rollout gets a degeneracy
  score, aggregated as `min` (worst-case per rollout, per Pan et al.
  *PURE*, arXiv:2504.15275: min-form credit assignment beats sum-form
  for reward hacking).
- **SpecRA / structural repetition** (OpenReview xVO4BqmzVD, 2025):
  cheap FFT-based repetition detector that catches paraphrase loops a
  literal n-gram test misses. Worth ~30 lines of numpy.
- **Perplexity is actively harmful** (OpenReview
  bdef329fe64d…, 2024): "perplexity rewards repetitive text while
  penalizing well-formed outputs." Note in the doc so no one proposes
  this as an "obvious" score.
- **Self-BLEU over N rollouts from same prompt** (Pimentel et al.
  *Standardizing Text Diversity*, IJCNLP 2025 demo, arXiv:2403.00553):
  hard-zero the score if N rollouts of the same prompt have
  Self-BLEU > 0.7. Catches "student memorized one response and emits it
  regardless."

---

## Axis A3 — Length penalty

Score −= λ · max(0, len − len_95th_teacher) where len_95th_teacher is
the 95th-percentile length of *teacher's own* rollouts on the same
prompt. Non-arbitrary. From Kimi k1.5 (arXiv:2501.12599) — they get
simple length penalties to work where MCTS/PRMs don't.

Trivial to implement on top of A1 since teacher rollouts are already
being generated for the baseline comparison.

---

## Axis A4 — Verifiable outcome rewards (RLVR subset)

The DeepSeek-R1 / Tülu 3 lesson (arXiv:2501.12948, arXiv:2411.15124):
wherever a prompt has a verifiable answer, use the *answer* as the
primary reward, not KL. Categories to add to eval:

- **Math** (GSM8K-mini, MATH-mini): final answer via regex + SymPy.
- **Code** (HumanEval / MBPP-mini): unit-test pass rate. Sandboxed.
- **Instruction-following** (IFEval): format-compliance regex.
- **Multi-choice**: exact match.

Two critical caveats from the 2025 literature:

- **Spurious Rewards** (Shao et al., arXiv:2506.10947) — on Qwen-family
  models, RLVR with *random* or *incorrect* labels still yields +20–24%
  on MATH-500 because GRPO clipping amplifies pretrained high-probability
  tokens. Our teacher is Qwen3.5-4B; if students are also Qwen-based we
  risk rewarding "Qwen-nature surfacing" rather than real improvement.
  **Mitigation:** keep at least one non-Qwen teacher in the ensemble
  (A5); cross-family verification.
- **Miracle Steps** (Rubric Rewards, arXiv:2510.07774): outcome-only
  rewards produce correct finals with unsound reasoning from
  memorization. Use problem-specific rubrics — grade as
  (final correct) × (rubric fraction). Verified Pass@1024 on AIME'24:
  26.7% → 62.6% with rubrics.
- **Imperfect verifiers** (arXiv:2510.00915, 2025): our regex/test
  checks *are* noisy. Use multiple equivalent verifiers per prompt; 2-of-3
  agreement required.
- **Skip PRMs.** ProcessBench (arXiv:2412.06559, NeurIPS 2024) and
  PRMBench (arXiv:2501.03124, 2025) show current open PRMs are brittle
  and fail to generalize beyond GSM8K/MATH. Outcome verification >
  process reward at our scale.

### What shipped (April 2026, v2)

Implementation in `scripts/pod_eval_vllm.py`:

- **Static pool** of ~50 verifiable prompts spanning: capitals/geography,
  basic chemistry, factoids, prime checks, divisibility, IFEval-style
  format compliance (all-caps regex, comma-separated lists, exact word
  count, single-word rhyme, lowercase constraint), multi-choice (A/B/C/D),
  and word/letter manipulation. Per round we deterministically sample
  `CAPABILITY_PROBE_N=24` of these via `random.Random(block_seed)`.
- **Procedurally generated math** (`CAPABILITY_PROBE_N_PROC_MATH=12`):
  add/sub/mul/div/mod with operands drawn from the same seeded RNG.
  *Fresh every round*, so memorization is impossible.
- **Multi-kind verifier** (`_capability_score_one`): exact int / yes-no /
  one-letter-MC / word / `word_alt` synonym set / regex / word-count /
  rhyme-suffix / phrase regex. Each kind has a tolerant extractor
  (`_extract_capability_answer`) that strips `<think>` blocks and
  markdown wrappers before scoring, so partial credit goes to models
  that *could* answer but haven't been RLHF'd for format.
- **Teacher normalization**: `composite._axis_capability` divides
  student `pass_frac` by `teacher_pass_frac` (computed by
  `prepare_teacher_probe_refs_vllm` on the *same* sampled set), so the
  axis is calibrated to teacher difficulty. Cache invalidates when
  `block_seed` changes (`teacher_capability_block_seed` field in the
  on-disk torch cache).

Still to do (deferred to private-pool axis A7 + A1 on-policy roll-outs):
real GSM8K/MATH-mini sourced from a held-out validator-only set, code
sandboxing for HumanEval, multi-verifier 2-of-3 agreement.

---

## Axis A5 — Teacher ensemble, worst-case aggregation

Single-teacher KL is a single reward model with all the known
overoptimization pathologies (Rafailov et al. *Scaling Laws for Reward
Model Overoptimization in DPO*, arXiv:2406.02900, 2024: overoptimization
appears even without PPO, even in DPO, even before one epoch).

**Fix** (Coste et al., arXiv:2406.01013, 2024): score against an
**ensemble** of teachers with worst-case or uncertainty-weighted
aggregation. For SN97:

- Qwen3.5-4B (current) — primary, has exact-architecture match
- DeepSeek-R1-Distill-Qwen-7B — reasoning-focused, different post-training
- Llama-3.1-8B-Instruct — different model family, defeats Qwen-specific
  priors that cause *Spurious Rewards* (A4)

Per-prompt rule: `score = min(rkl_qwen, rkl_deepseek, rkl_llama)` — a
miner must be good against all three to win. Overfitting one teacher
gets punished by the others.

Memory budget on a B200 (80GB): three ~8B teachers in fp16 fit in ~48GB;
inference is batched. Feasible.

---

## Axis A6 — Weight-fingerprint dedup (copy-miner killer)

Today we SHA256 the raw safetensors at registration. Attackers defeat
this with any trivial perturbation (LoRA, noise, norm-scale). Adopt
**AWM** (arXiv:2510.06738, 2025): Linear Assignment Problem + unbiased
CKA on weight matrices, AUC 1.000 on "is this a fine-tune of X?", 30 s
compute, robust to fine-tuning / pruning / RL. Pair with **REEF** (ICLR
2025): feature-space similarity, robust to weight permutations. Dual
check at registration: reject if `AWM(new, any_existing) > τ_AWM` OR
`REEF(new, any_existing) > τ_REEF`.

The 2026 **DuFFin** paper (ACL Findings EACL 2026) combines
trigger-pattern + knowledge-level fingerprints; IP-ROC > 0.99 even on
quantized, safety-aligned variants. Strongest black-box option. If we
can only pick one, use DuFFin.

**Sentient OML** (github.com/sentient-agi/OML-1.0-Fingerprinting) is
an orthogonal tool: the *validator* embeds a secret fingerprint at
training time (by demanding the miner train on a validator-provided
data shard with embedded trigger patterns). A copy-miner that didn't
actually train cannot match the trigger. This requires an A7-style
script-execution regime to be meaningful.

---

## Axis A7 — Rotating prompt pool + reusable-holdout DP noise

Static prompts are memorized. The theory:

- **Reward Hacking as Equilibrium** (arXiv:2603.28063, 2026): any fixed
  scorer is equilibrium-hacked.
- **LiveBench** (White et al., ICLR 2025, arXiv:2406.19314): the SOTA
  protocol for contamination-resistant eval. Three strategies —
  frequently-updated questions from fresh sources (arxiv, competitions),
  automatic objective scoring, wide task variety.
- **Reusable Holdout** (Dwork et al., Science 2015, arXiv:1506.02629):
  differentially-private noise injection on reported scores lets us
  reuse a holdout across thousands of adaptive queries without leaking
  it; formal max-information bound on cumulative privacy.
- **Emperor's New Clothes** (arXiv:2503.16402, 2025): empirically, *no*
  rephrase/regeneration strategy for mitigating contamination works
  across benchmarks. Must be structural (rotation + private pool) not
  algorithmic.

**Implementation.**

1. Prompt bank P = 5,000 prompts from ClimbMix-style open data, pretest
   each with teacher-rollout quality (filter out prompts teacher fails
   on — otherwise noise dominates).
2. Per tempo, sample a private subset of size k=300 via
   H(block_hash ‖ validator_hotkey ‖ tempo_index). Miners see stats from
   the *aggregate* across all tempos, never one tempo's raw prompts.
3. Add Laplace noise to each reported score with scale matching the
   per-tempo privacy budget ε. `noise ~ Lap(Δ/ε)` where Δ is the
   sensitivity of the score to one prompt.
4. Track cumulative privacy budget per miner; reject further submissions
   once budget is exhausted until the pool is refreshed.

### What shipped (April 2026, v1 — `eval/private_pool.py`)

- **Private holdout** (`state/private_prompt_pool.json`): 198 hand-curated
  prompts (instruction-style, recipe, translation, eulogy, format-compliance
  — deliberately *off-distribution* from ClimbMix web text). Validator-only;
  not in the public repo. Extend by appending JSON entries; rotation handles
  the rest.
- **Per-round mix**: `service.py` allocates 10% (`DEFAULT_PRIVATE_FRACTION`)
  of `n_prompts` to a deterministic, block-seeded private subset; the
  remaining 90% are public ClimbMix shards as before.
- **Commit-reveal**:
  - **Commit** (`state/private_pool_commit.json`): sha256 root over the
    sorted per-prompt sha256 list, written *before* eval starts. Exposed via
    `GET /api/private-pool-commit` so anyone can record the commit at the
    block boundary.
  - **Reveal** (`state/private_pool_reveal.json`): per-prompt sha256 list,
    written *after* the round completes. Available via the same endpoint.
    Auditors verify `sha256(sorted(prompt_hashes)) == root`.
- **Public-API redaction**: `pod_session.py` redacts the private tail of
  `eval_data_latest.json` and the per-round files in `state/eval_data/`
  (which the public `/api/eval-data` serves) — the original raw copy is
  retained under `state/eval_data_private/` for offline validator audit.
- **DP-noise scaffolding** (`dp_noise_for`): per-prompt Laplace draw with
  scale ∝ uses, seeded by `sha256(prompt ‖ uses)` so the same draw is
  reproduced if asked again. Not yet wired into the score — composite
  `_axis_kl` still uses raw KL — because we want one round of telemetry to
  size the noise scale before perturbing rankings. Roll-out plan: shadow
  noise for 7 days, compare ranking stability with/without, then enable.

Still to do:
- Multi-validator independence: each validator should sample its private
  subset using `H(block_hash ‖ validator_hotkey)` so cross-validator
  triangulation can't reduce-by-intersection. Currently all validators
  share the same subset per block.
- Cross-validator union pool growth (~5,000 prompts per memo §A7) so the
  use rate doesn't outpace the holdout size; current 198 prompts is enough
  for ~6-8 rounds at 10% before each prompt has been used twice.
- Cumulative ε privacy budget tracking per *miner* (per memo §A7 step 4).

---

## Axis A8 — Dethronement ε(Δblocks) margin

Today: paired t-test at p<0.05 promotes a challenger over the king.
Problem: a copy-miner with near-zero perturbation passes this by chance
~50% of the time. The **SN37 pattern**
(docs.macrocosmos.ai/subnet-37-finetuning/subnet-37-incentive-mechanism):
require `KL_new < KL_king · (1 − ε(Δblocks))` where
`ε(Δblocks) = ε_0 · exp(−λ · Δblocks)`. A challenger must beat the king
by an *epsilon* in mean KL, not just in p-value; and the epsilon decays
with commit-time separation so that obviously-later copies need a much
bigger margin.

The theory is **Blum & Hardt "The Ladder"** (ICML 2015,
arXiv:1502.04585): provable bounds on adaptive leaderboard overfitting
under repeated submission. A score is revealed only if it strictly
beats the submitter's previous best by a threshold — otherwise the old
score is returned. Our paired t-test is a weak, buggy Ladder; replace
with a real one.

---

## Axis A9 — TOPLOC attestations (anti model-swap)

**TOPLOC** (Ong et al., ICML 2025, arXiv:2501.16007): hashes top-k of
the last hidden state per window into a 258-byte polynomial proof per
32 tokens. 1000× smaller than raw embeddings, 100× faster to verify
than re-running inference. 100% true positive / 0 false positive across
hardware, GPU type, tensor-parallel dim, kernel choice.

**Why we need it.** Today SN97 trusts that the HF repo at the committed
hash is what the miner actually served for the eval. A miner can serve
different weights through their inference endpoint and nothing checks.
TOPLOC makes the miner commit an attestation on their inference; the
validator verifies by re-running a 1-of-32-token spot-check. ~0.2%
compute overhead, kills model-swap cold.

---

## Axis A10 — Commit-reveal weights v3 + Yuma 3

Standard Bittensor hardening. *The Weight Copying paper* (BT whitepaper
v7.0.1, May 2024, docs.learnbittensor.org) documents the stake-weighted
averaging attack: copier validators compute median of revealed weights
and submit it, achieving maximal vtrust, earning higher dividends than
honest validators. Commit-reveal v3 uses Drand time-lock encryption so
no one (including submitter) can decrypt before the target round.

**Caveat from Inference Labs' analysis** (Medium, 2024): commit-reveal
only works if consensus weights change enough over the interval. If
SN97's king is stable for weeks (likely), stale weights ≈ fresh weights
and copying still wins. The defense is to **force churn** via A7
(rotating prompts) and A8 (tight dethronement margins).

Yuma Consensus 3 (YC3, docs.bittensor.com/learn/yc3-blog, 2025) adds
per-bond EMA scaling so validators who identify a strong miner *early*
keep dividend share even after consensus catches up — directly
incentivizes honest scoring over copying.

Settings: `commit_reveal_weights_enabled = true`,
`commit_reveal_period = 2 tempos`, tempo = 360 blocks (~72 min).
Immunity period > commit-reveal interval.

---

## Axis A11 — Behavioral dedup via logit fingerprints

Two models with different weights can have identical logit patterns on
our eval set (e.g., miner applies an orthogonal transform, or trains a
new student to imitate the old king exactly). AWM/REEF catch
weight-level similarity; this catches *output* similarity.

Implementation: for each evaluated model, hash the top-k logit indices
at every position for a fixed canary-prompt set. Two models whose
canary-logit-hash agree at > threshold are flagged as near-copies; the
later-committed one is zeroed. SN25's duplicate detection (Macrocosmos
protein-folding subnet) uses this pattern.

---

## Axis A12 — Adversarial prompt generation (Dynabench)

Optional, spec only for now. A validator-side LLM (~7B) generates
prompts that maximize KL-delta between the current king and the teacher
under on-policy rollout. Those prompts become a dedicated axis in the
next tempo's eval. Pattern from **Dynabench** (Kiela et al., NAACL
2021) and the 2025 red-team literature (arXiv:2510.08329 *AutoRed*,
arXiv:2407.17447 *FLRT*).

Low-cost alternative: a manually-curated battery of ~200 short prompts
known to crush "always-think" miners (greetings, one-word-answer,
format-constrained, refusal probes). Near zero maintenance.

---

## Implementation sequence (the concrete plan)

**Next tempo (P0, this commit cycle):**
- [x] Shipped: teacher-anchored degeneracy probe (A2 subset).
- [x] Shipped: teacher-probe reference collection while teacher is
      loaded (populates the dormant `_TEACHER_PROBE_SAMPLES` so the
      MAD-z branch actually activates; 15 prompts, ~10s extra per round).
- [x] Shipped: A3 length penalty, computed as `min(1, 2·teacher_mean /
      student_mean)` on the think-probe prompts. Stored as
      `length_axis.penalty` per student.
- [x] Shipped: A4-mini verifiable-rewards capability probe, 10
      short prompts (arithmetic, yes/no, one-word facts) scored by
      regex extraction, normalized against teacher pass rate. Runs
      per-student in `pod_eval_vllm.py::capability_probe`.
- [x] Shipped: `scripts/validator/composite.py` multi-axis score,
      with both worst-axis (min-form, Coste 2024 / Pan 2025 PURE) and
      weighted-mean aggregation. **Shadow mode** — logged and surfaced
      in H2H/API but KL still decides the king. 14-day grace before
      switchover.
- [ ] A1 on-policy reverse-KL as a proper axis (requires holding the
      teacher or reloading for a short forward pass over all-students'
      rollouts). Deferred to next tempo — composite already anti-games
      KL-farming via the other axes. Prototype exists at
      `scripts/on_policy_rkl_probe.py`.

**Next month (P1):**
- [ ] Grace-period expiry: composite becomes the canonical ranking
      key. Existing KL field preserved for transparency.
- [ ] A4-full verifiable rewards (GSM8K-mini, HumanEval-mini, IFEval-mini)
      run in a sandbox on the eval pod.
- [ ] A5 teacher ensemble (add DeepSeek-R1-Distill-Qwen-7B and
      Llama-3.1-8B-Instruct to eval pod; score as min across).
- [ ] A6 AWM + REEF weight fingerprint at registration.
- [ ] A8 ε(Δblocks) dethronement margin.
- [ ] A10 enable commit-reveal v3 + YC3.

**Next quarter (P2):**
- [ ] A7 private rotating prompt pool with DP noise.
- [ ] A9 TOPLOC attestation of miner inference.
- [ ] A11 behavioral dedup via canary-logit fingerprints.

**Spec-only (P3):**
- A12 adversarial prompt generation.
- zkPoT / Kaizen-style proof-of-training (arXiv 2024/162; today takes
  ~hours per step at 5B, reconsider in 12–24 months).
- SN56-style full script-execution tournaments (costs $$$; revisit
  once subnet alpha price supports it).

---

## What about transitioning the scorer from KL to RL?

Thinking Machines' post and the GKD paper both argue on-policy
distillation *is* a clean, low-compute RL. The Qwen3 report and the
Tülu 3 recipe both do essentially the same thing: SFT → DPO → RLVR. Our
current mechanism is SFT-only. Adding A1 (on-policy) moves us to
something DPO-adjacent; adding A4 (RLVR) moves us to full Tülu 3. This
is a natural 3-tempo migration and does not require blowing up the
subnet — at each step the old scorer coexists, weighted down over time.

---

## What about the teacher itself?

A periodic reminder: **Qwen3.5-4B is small enough to have limited
reasoning.** The *Looping in Reasoning Models* finding — 4B students
loop most — applies to the teacher too. At some point the subnet should
upgrade to a larger teacher (Qwen3.5-32B / DeepSeek-V3 / Llama-3.1-70B).
Doing that *before* A1 is dangerous because compounding errors on
off-policy distillation are worse with a larger teacher. Doing it
*after* A1 is fine and probably a quality win. Rough plan: A1 → A4 → A5
ensemble → upgrade primary teacher.

---

## References

All papers are cited inline above. Master list for convenience (all
2024–2026 unless noted):

**On-policy distillation.** GKD arXiv:2306.13649 (ICLR 2024) · MiniLLM
arXiv:2306.08543 (ICLR 2024) · Thinking Machines *On-Policy
Distillation* blog, Oct 2025, DOI 10.64434/tml.20251026 · DistiLLM
arXiv:2402.03898 (ICML 2024) · DistiLLM-2 arXiv:2503.07067 (ICML 2025
oral) · EOPD arXiv:2603.07079 (ICLR 2026 submission) · SKD
arXiv:2410.11325 (ICLR 2025) · Teacher Hacking arXiv:2502.02671 (ICML
2025) · Looping in Reasoning Models arXiv:2512.12895 · RL's Razor
arXiv:2509.04259.

**Reward hacking / Goodhart.** DPO Overoptimization arXiv:2406.02900 ·
RM Ensembles arXiv:2406.01013 / 2310.02743 · Iterated RLHF
arXiv:2505.18126 · CoT Monitoring / Obfuscation arXiv:2503.11926
(OpenAI, 2025) · PURE min-form arXiv:2504.15275 · Sycophancy → Subterfuge
arXiv:2406.10162 · School of Reward Hacks arXiv:2508.17511 · Reward
Hacking Survey OpenReview ENQrLsePEa (2025) · Reward Hacking as
Equilibrium arXiv:2603.28063 (2026).

**Text degeneracy.** Text Diversity Standardization arXiv:2403.00553
(IJCNLP 2025) · SpecRA OpenReview xVO4BqmzVD · Correlation Dimension
arXiv:2510.21258 · Prometheus-2 arXiv:2405.01535 (2024) · LLM-Judge
Bias arXiv:2410.02736 · SOS-BENCH OpenReview MzHNftnAM1 (ICLR 2025) ·
Perplexity Brittleness OpenReview bdef329fe64d…. Classical anchors:
Holtzman *Curious Case* arXiv:1904.09751 (2019) as baseline only.

**RLVR & verifiable rewards.** DeepSeek-R1 arXiv:2501.12948 · Kimi k1.5
arXiv:2501.12599 · Tülu 3 arXiv:2411.15124 · ProcessBench
arXiv:2412.06559 · PRMBench arXiv:2501.03124 · Rubric Rewards
arXiv:2510.07774 · Spurious Rewards arXiv:2506.10947 · Imperfect
Verifiers arXiv:2510.00915.

**Contamination & held-out.** LiveBench arXiv:2406.19314 (ICLR 2025) ·
Retro-Holdouts arXiv:2410.09247 · ConStat arXiv:2405.16281 · Emperor's
New Clothes arXiv:2503.16402 · CoDeC arXiv:2510.27055 · Perplexity
Signatures arXiv:2509.23488 · Leaderboard Illusion arXiv:2504.20879 ·
SWE-bench Verified (OpenAI blog 8/2024). Historical anchors: Dwork et
al. *Reusable Holdout* arXiv:1506.02629 · Blum & Hardt *Ladder*
arXiv:1502.04585.

**Bittensor & decentralized ML.** IOTA arXiv:2507.17766 · BT Weight
Copier whitepaper v7.0.1 (docs.learnbittensor.org, May 2024) · Commit
Reveal docs (learnbittensor, Feb 2026) · YC3 blog
(docs.bittensor.com/learn/yc3-blog) · SN37 docs
(docs.macrocosmos.ai/subnet-37-finetuning) · SN56 Gradients
(simplytao.ai, 2025) · SN25 Protein Folding
(docs.macrocosmos.ai/subnet-25-mainframe) · TOPLOC arXiv:2501.16007 ·
INTELLECT-2 arXiv:2505.07291 · AWM arXiv:2510.06738 · REEF ICLR 2025 ·
DuFFin ACL-EACL 2026 · Sentient OML
(github.com/sentient-agi/OML-1.0-Fingerprinting) · Proof-of-Learning
arXiv:2103.05633 + attacks arXiv:2108.09454, arXiv:2208.03567 ·
Probabilistic Audits + PoI (ICLR 2026 phXKOiXC4x) · Kaizen ZKP-of-Training
ePrint 2024/162.

**Mechanism design.** SPIN arXiv:2401.01335 · Best-of-N
arXiv:2503.21878 · ATLAS arXiv:2511.04689 · LLM Fine-tune Mechanism
Design OpenReview 3JUhkxVlyF (NeurIPS 2024) · DSIR arXiv:2302.03169 ·
Peer Prediction arXiv:2505.13636, arXiv:2506.02259 · Chatbot Arena
Vote Rigging arXiv:2501.17858, ICML 2025 proc.mlr.press/v267/huang25z ·
MLPerf inference policies (mlcommons, 2024–2026).

Total: ~55 distinct references, ~80% of which are 2024–2026.
