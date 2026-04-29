# 2026-04-29 v30 strategic rollup

Single-document rollup of everything shipped on 2026-04-29 plus the
forward path validated against April-2026 SOTA distillation research.

## What shipped today (chronological)

| Tag | Time UTC | What |
|---|---|---|
| v29.1 | ~21:30 | Per-axis baseline-relative penalty (regress vs Qwen-4B-base ⇒ axis docked α×gap) + KL probe skill diversification |
| v29.2 | ~22:00 | `long_context_bench` multi-needle (was 93 % saturated → now discriminating); new `debug_bench` axis; per-axis correlation telemetry |
| v29.3 | ~23:00 | `per_src` aggregation per bench probe + per-template saturation audit script |
| v29.4 | ~02:30 | 4 new SOTA-aligned bench axes (`correction_bench`, `multi_doc_synthesis_bench`, `calibration_bench`, `refactor_bench`) + capability tightening + teacher-swap-readiness (vocab + heuristic) |
| v29.5 | ~03:00 | Procedural lexicon — replace every static name/topic/distractor list with synthesisers (28M+ unique names, 446M+ unique distractor sentences) |
| v29.6 | ~03:30 | Procedural chat-prompt synthesizer (eliminates 178 static prompts in ON_POLICY_RKL / JUDGE / CHAT_TURNS pools) + entry-point name rotation in coding benches |
| v29.7 | ~03:30 | vLLM crash recovery (threading.Event-backed dead-signal so workers fail-fast) + `[teacher_path]` log tags + concurrency cap 48→24 + eval-weight rebalance + Qwen3.6-35B-A3B teacher swap + maxStudentParams 5.25B→7B |
| v29.7.1 | ~03:35 | vLLM teacher MM disable (`--limit-mm-per-prompt {"image": 0, "video": 0}`) for Qwen3.6 vision encoder |
| v29.7.2 | ~03:45 | `_teacher_cache_complete` actually verifies safetensors weights present (was returning True with only config + tokenizer cached) + `_stub_missing_preprocessor_config` stubs Qwen3.6's missing preprocessor |

## v30 strategic audit (delivered as canvas)

The canvas at `canvases/sn97-eval-and-teacher-audit.canvas.tsx`
identifies and quantifies:

* **Eval axis status**: 11 core, 4 auxiliary, 5 redundant / saturated /
  dead. Three high-weight axes (`capability` 0.25, `kl` 0.15,
  `length` 0.10) carry 0.50 of total weight but contribute minimal
  discrimination — kl is 91 % saturated, capability overlaps math /
  reasoning, length is 60 % saturated. Rebalance flows that 0.40 of
  weight onto the core SOTA-correlated axes (math / code / reasoning /
  ifeval / calibration). Shipped today via `distil.env` env-overrides
  and `composite.py` env-driven AXIS_WEIGHTS.

* **Teacher options April 2026**: Kimi K2.6 (1T MoE, 32B active, SOTA
  on agentic SWE-Bench Pro), DeepSeek V4 (~600B+ MoE, strong reasoning),
  GLM-5 (744B MoE, verbose CoT — confirmed worse for 4B distillation
  than Kimi), Qwen3.6-35B-A3B (drop-in compatible — same 248320 vocab
  as current). **Recommendation**: Stage 1 swap to Qwen3.6 (shipped
  today), Stage 2 evaluate Kimi K2.6 (planned).

* **Coverage gaps**: Knowledge / factual recall (knowledge_bench was
  muted in v28; needs open-ended QA replacement), theory of mind /
  pragmatics (no axis), long-form generation quality (judge_probe
  only short responses).

## SOTA distillation research findings

`reports/2026-04-29-distillation-sota-synthesis.md` is the full
synthesis. Highlights for SN97 alignment:

### Validates current approach
* RKL primary (35 %) + FKL secondary (15 %) is the canonical 2026
  recipe — Thinking Machines blog [3], Qwen3 tech report [5], TRL
  `DistillationTrainer` all use this exact split.
* Block-seeded fresh prompts + procedural everything = better than
  80 % of public distillation eval setups.
* Capability + degeneracy + reasoning_density catch real failure
  modes (the "decorative reasoning that diverges in token-space"
  the 2026-04-17 spiral king exhibited).

### Two clear-impact upgrades identified

1. **Top-K token overlap axis** ([20]: "Rethinking OPD" 2026 arXiv
   2604.13016) — *the* single most predictive signal of OPD success.
   Successful runs converge to 97-99 % shared mass on a small token
   set. Our top-128 cache makes this nearly free to compute. Adds a
   17th composite axis with ~5 % weight.

2. **Entropy-aware FKL/RKL switching** ([4]: Entropy-Aware OPD,
   OpenReview WSRQ37tzk1) — dynamically weight RKL when teacher is
   confident, FKL when teacher is uncertain. On Qwen3-{0.6B, 1.7B, 4B}
   shows +1.37 / +2.39 / +5.05 Pass@8 across 6 math benchmarks vs
   vanilla OPD. Direct upgrade to the current fixed-weight RKL/FKL
   composition.

### Bias fixes worth doing

* **Importance-sampled KL** instead of top-K renormalised KL ([13]:
  Sparse Logit Sampling, ACL 2025). Current path renormalises over
  teacher's top-128 support — biased. The IS fix has < 10 % overhead
  and gives unbiased KL nat values.

* **Tail-decoupled KL** ([22]: TAD, arXiv 2602.20816) — splits the KL
  into top-K and tail components so top-K gradient doesn't dominate.
  Useful as a *separate* "tail mass alignment" axis.

### Real-world data validating teacher choice

The brianmeyer/distillreasoning April-2026 experiment ([2]) is the
strongest evidence we have for our planned Stage 2:

* Same student (Qwen3.5-4B), same problem set (2,083 GSM8K + MATH +
  ARC + HumanEval), same SFT-then-GRPO pipeline.
* Kimi K2.5 traces (325-tok median, concise) → distilled 4B scored
  **72.6 % GSM8K** (zero-shot).
* GLM-5 traces (433-tok median, verbose) → distilled 4B scored
  **53 % GSM8K**.
* Combined traces (3,196) → 67 %, *worse* than Kimi-only (1,624).
* Reference: raw Qwen3-8B (2× larger, no distill) scored **63 %** —
  the distilled 4B beat it.

**Conclusion**: Concise teachers transfer better to small students.
For Stage 2, prefer Kimi K2.6 (concise) over GLM-5.1 (verbose). The
Qwen3.6 Stage 1 swap is in the same family / size as our previous
3.5 teacher so this concise / verbose distinction doesn't apply yet.

### 4B distillation success benchmarks (SOTA targets, April 2026)

For a 4B-class student distilled from a 35B-class teacher:

| Benchmark | Target | Reference |
|---|---|---|
| AIME 2024 / 2025 (Pass@1) | 50-75 % | Thinking Machines + Qwen3 + Phi-4-Mini-Reasoning |
| MATH-500 | 88-95 % | DeepSeek-R1-Distill-7B 92.8 |
| GPQA Diamond | 45-55 % | Phi-4-Mini-Reasoning 52.0 |
| GSM8K (5-shot CoT) | 88-94 % | DeepSeek-R1-Distill-7B 95.3 |
| MMLU-Pro | 50-60 % | Open LLM Leaderboard v2 |
| IFEval | 75-85 % (post-OPD) | Thinking Machines 83 % |
| **On-policy RKL vs teacher** | **< 0.10 nats** average | brianmeyer + TM reproductions |
| **Forward-KL on teacher continuations** | **< 0.15 nats** | Same |
| **Top-K overlap @ K=64** | **> 85 %** | [20] (97-99 % is achievable) |
| **Reasoning density** | **> 0.6** | Calibrated against Qwen3-4B-Thinking |

Our Qwen3.5-4B reference baseline is currently ~0.10-0.15 nats KL
**without any distillation training** (same family overlap). Post-
distillation miners should target **< 0.05 nats** to be competitive.

### Recommended miner pipeline (4-stage, $100-500 cloud cost)

```
Stage 1: Mid-train (large-scale off-policy SFT)
  - 50-200k teacher rollouts, rejection-sample on correctness
  - LoRA rank 128, lr 1e-5, batch ≤ 32, 1-3 epochs

Stage 2: SFT on high-quality compact set
  - 800-2000 hand-checked problems (LIMO/s1-style)
  - 1 epoch full-FT or LoRA-r-128

Stage 3: On-Policy Distillation (per Thinking Machines / Qwen3 / TRL)
  - TRL DistillationTrainer with:
      lmbda=1.0 (fully on-policy)
      beta=1.0 (reverse KL)  -- or 0.7 (mostly RKL with FKL safety)
      loss_top_k=1
  - 100-200 steps if Stage 1 was solid

Stage 4 (optional): RL with verifiable reward (Phi-4-Mini style)
  - GRPO on math/code with answer-correctness reward

Eval:
  - lm-evaluation-harness: gsm8k_cot 8-shot, minerva_math 4-shot,
    arc_challenge 25-shot, gpqa_diamond 0-shot, mmlu_pro 5-shot
  - Plus RKL/FKL against committed teacher (matches SN97)
```

**Critical warning** ([19], Phi-4-Mini-Reasoning paper): Naively
applying LIMO / s1 to a 4B base **regresses** performance:

| Model | AIME24 | MATH-500 | GPQA |
|---|---|---|---|
| Phi-4-Mini base | 10.0 | 71.8 | 36.9 |
| Phi-4-Mini + LIMO | 6.7 | 57.8 | 24.8 |
| Phi-4-Mini + S1K | 3.0 | 47.0 | 26.3 |
| Phi-4-Mini-Reasoning (full 4-stage) | **57.5** | **94.6** | **52.0** |

Small students lack the latent capacity that LIMO / s1 assume. **Need
the big mid-training phase first.** Inform miners of this trap.

## Forward path (next iterations)

In priority order:

1. **Top-K token overlap axis** (research-validated highest single
   impact). Add as a 17th composite axis, weight ~0.05. Free to
   compute from existing top-128 cache.

2. **Knowledge / factual recall axis** (replaces muted MC `knowledge_bench`).
   Open-ended factual QA with regex / exact-match grading; replaces
   the random-pick-floor failure mode of the 4-option MC version.

3. **Pragmatic / theory-of-mind axis** — false-belief, scalar
   implicature. Procedural multi-character scenario items.

4. **Long-form judge sub-axis** — teacher rubric on 300-500 word
   essay responses, distinct from the current short-response
   `judge_probe`.

5. **Entropy-aware RKL/FKL adaptive weighting** ([4]). Bigger code
   change but research-backed direct improvement.

6. **Mining guide v2** — codify the 4-stage pipeline + the
   LIMO/s1 warning into `MINER_FAQ.md` so miners don't waste
   compute on regressing-on-small-models recipes.

7. **Stage 2 teacher evaluation: Kimi K2.6** — pull the model on a
   separate pod, run a small distillation experiment (50 traces),
   measure 4B student delta vs Qwen3.6 baseline. GO/NO-GO based on
   actual data.

8. **Importance-sampled KL** ([13]) — drop-in unbiased estimator for
   the existing top-K cached KL; <10 % runtime overhead.

9. **Tail-decoupled KL** ([22]) — separate "tail mass alignment"
   surface as a new metric.

## Files touched today

```
M  scripts/pod_eval_vllm.py                         (~1500 lines net add across v29.1-v29.7.2)
M  scripts/validator/composite.py                   (axis weight env-overrides, baseline penalty, broken-axes set, per_src forwarding)
M  scripts/validator/results.py                     (annotate_h2h_with_composite plumbing)
M  scripts/validator/state_manager.py               (king_canary_streak)
M  scripts/audit/axis_correlation.py                (NEW v29.2)
M  scripts/audit/per_template_saturation.py         (NEW v29.3)
M  eval/state.py                                    (KING_CANARY_FILE)
M  eval/private_pool.py                             (env-driven fraction)
M  state/private_prompt_pool.json                   (198 → 278 prompts)
M  frontend/src/lib/subnet-config.json              (teacher Qwen3.5 → Qwen3.6, maxStudentParams 5.25B → 7B)
M  /home/distil/.secrets/distil.env                 (axis weight rebalance, vLLM concurrency cap, teacher-swap envs)

A  reports/2026-04-28-grief-tiebreaker-and-canary-streak.md
A  reports/2026-04-28-v29.1-baseline-relative-penalty.md
A  reports/2026-04-29-v29.2-debug-bench-and-lc-fix.md
A  reports/2026-04-29-eval-roadmap.md
A  reports/2026-04-29-v29.4-sota-capability-sweep.md
A  reports/2026-04-29-v29.5-procedural-everything.md
A  reports/2026-04-29-v29.6-procedural-chat-prompts.md
A  reports/2026-04-29-distillation-sota-synthesis.md  (this push)
A  reports/2026-04-29-v30-strategic-rollup.md         (this file)
A  canvases/sn97-eval-pipeline.canvas.tsx
A  canvases/sn97-eval-and-teacher-audit.canvas.tsx
```

## Open Discord report status

**coffiee 2026-04-29 01:07 UTC**: "fix eval it is using HF again."

* Root cause confirmed: vLLM teacher deadlocked at ~prompt 200/300
  (engine queue stuck with 41 pending requests, KV cache only 16.7%
  — NOT OOM). Retry loop kept hammering port 9100 for ~45 min before
  giving up and falling to HF generate, which is ~10× slower.

* Fix shipped in v29.7 (commit cc5e7a1):
  - `_vllm_dead_event()` threading.Event short-circuit
  - Explicit `[teacher_path]` log tags so the bot's snapshot can
    correctly identify HF vs vLLM without back-deriving from timing
  - vLLM concurrency capped 48 → 24

* Follow-up posted to thread; user-visible behaviour:
  - `[teacher_path] vllm` — happy path
  - `[teacher_path] hf-fallback-vllm-failed-to-start` — startup fail
  - `[teacher_path] hf-fallback-after-vllm-crash` — mid-eval crash
  - `[teacher_path] hf-generation-active` — phase-1 fallback

* Apology issued to coffiee — they were right from the start that
  timing is the only thing that matters here, and the bot should
  have led with timing math instead of pushing back from VRAM /
  proc-count which look identical on both paths.

## Bot lesson learned (for future Discord triage)

When asked "is this on HF?", **always lead with wall-clock timing**.
VRAM and proc-count alone don't disambiguate vLLM vs HF on a model
this size — both paths hold ~67 GB for the weights. The signal that
matters is generation speed (3-5 min for 300 prompts on vLLM vs
30+ min on HF). The new `[teacher_path]` log tags surface this in
the 40-line tail snapshot directly so the bot can answer correctly
the first time.
