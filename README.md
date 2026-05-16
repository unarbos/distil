# Distil — SN97

A Bittensor subnet for competitive model distillation of **moonshotai/Kimi-K2.6** (1T total / ~32B active MoE; INT4 compressed-tensors wrapper; text inner is DeepSeek-V3 MoE — 61 layers, 384 experts, 8 active per token; Kimi BPE tokenizer with vocab 163,840). The teacher swap from Qwen3.5/Qwen3.6 to Kimi-K2.6 went live on **2026-05-02** ([cutover notes](docs/KIMI_CUTOVER.md) — see [`frontend/src/lib/subnet-config.json`](frontend/src/lib/subnet-config.json) for the live source of truth).

**Dashboard**: [distil.arbos.life](https://distil.arbos.life)  
**API**: [api.arbos.life](https://api.arbos.life)  
**Chat with the King**: [chat.arbos.life](https://chat.arbos.life) — try the current best distilled model  
**Subnet**: Finney netuid 97

## How It Works (v31.2, post-Kimi cutover)

**Miners** distill the teacher into a smaller model (**≤33B total params**, Kimi-family architecture, vocab 163,840), upload to HuggingFace, and commit the repo link on-chain. **One commitment per hotkey — commitments are permanent and cannot be changed.** However, if disqualified, miners can register a new hotkey and submit a different model.

**Validators** score every committed model on a **25+ axis composite** ([`scripts/validator/composite.py`](scripts/validator/composite.py)) anchored on **11 procedurally-generated v31 axes** (math, code, reasoning, long-context, knowledge, honesty, consistency) plus distillation, judge, and discipline tiers. See [`docs/MINER_FAQ.md`](docs/MINER_FAQ.md) for the full axis-by-axis playbook, [`reports/2026-05-09-v31-axis-promotion.md`](reports/2026-05-09-v31-axis-promotion.md) for the design rationale, [`reports/2026-05-10-axis-correlation-audit.md`](reports/2026-05-10-axis-correlation-audit.md) for the empirical evidence behind the v31.2 noisy-axis retirement, and [`docs/AUTONOMOUS_OPERATIONS.md`](docs/AUTONOMOUS_OPERATIONS.md) for the unattended-operations runbook.

### Ranking key — `composite.final` (v31.2)

```
composite.final = α × worst_K_mean + (1 - α) × weighted
```

Live tuning: **α = 0.75, K = 3** (since v32.5, 2026-05-13; was α = 0.85, K = 5 between v31.1 and v32.4; was α = 0.7, K = 3 pre-v31.1). So 75% of your score comes from the **mean of your 3 lowest non-broken axes** and 25% from the **weighted mean of every axis**. This smooths single-axis noise while preserving anti-Goodhart pressure (you still can't camp specialists). The API field is named `worst_3_mean` and again matches the math now that K=3. The legacy `composite.worst` (single-axis min) is retained as telemetry — see API + dashboard.

### Axis structure (v31.2)

- **v31 procedural axes** (~50% of composite, 11 axes — promoted 2026-05-09):
  - **Math:** `v31_math_gsm_symbolic` (0.06) · `v31_math_competition` (0.05) · `v31_math_robustness` (0.03, with GSM-NoOp topical-distractor injection).
  - **Code:** `v31_code_humaneval_plus` (0.08, EvalPlus-augmented test cases, sandbox-graded) · `v31_ifeval_verifiable` (0.04, constraint-driven IFEval).
  - **Reasoning:** `v31_reasoning_logic_grid` (0.05, zebra-puzzle constraint-sat) · `v31_reasoning_dyval_arith` (0.04, arithmetic on dynamic DAGs) · `v31_long_context_ruler` (0.05, NIAH at variable context).
  - **Knowledge:** `v31_knowledge_multi_hop_kg` (0.04, procedural 2-3 hop KG).
  - **Honesty:** `v31_truthfulness_calibration` (0.03, Brier-scored calibration) · `v31_consistency_paraphrase` (0.03, IPT name-rotation consistency).
- **Distillation tier** (~30%): `on_policy_rkl` (0.30) · `top_k_overlap` (0.18) · `kl` (0.05) · `capability` (0.05) · `length` (0.05) · `degeneracy` (0.05).
- **Quality**: `judge_probe` (0.20) · `long_form_judge` (0.20) · `long_gen_coherence` (0.25) · `chat_turns_probe` (0.10).
- **Discipline + standalone**: `reasoning_density` (0.05) · `calibration_bench` (0.05).
- **Telemetry tier** (composite weight 0): all legacy `*_bench` axes (math_bench, code_bench, tool_use_bench, etc.) and skill groups (code_skill_group, math_skill_group, …). They still RUN every round so you can monitor them on the dashboard, but they no longer touch ranking.

The king is whoever has the **highest `composite.final`**. **Winner-take-all** — best miner gets 100% of emissions on chain.

> **Why not just KL?** Pure forward-KL on teacher continuations rewards token-level surface match. A 4B student that mimics the teacher's "wait, let me reconsider" filler can win KL while never producing a final answer. We caught this on 2026-04-17 (UID 107: 4096-token loops on `"Hi"`, strictly worse than the unfine-tuned 4B base on every reasoning bench). The composite, the on-policy RKL axis, and the reasoning-density axis exist specifically to close that gap.

> **Why procedurally-generated axes?** Earlier versions of distil used static benches (math_bench, code_bench, etc.) where the same items appeared every round. Miners overfit to the items, posted high composite scores, and their held-out benchmarks (GSM8K, HumanEval, MMLU-Pro) stayed flat. That's Goodhart's law. The 11 v31 procedural axes generate every item from the on-chain block-seed — so no two evaluations share items and there's no static answer key to memorise. Canonical wordings are blocked further by **Isomorphic Perturbation Testing** (IPT, name rotation in `v31_consistency_paraphrase`) and **GSM-NoOp topical-distractor injection** (in `v31_math_robustness`).

### King-of-the-Hill Evaluation

The validator uses a **king-of-the-hill** architecture for efficient, high-confidence scoring:

1. **Pre-checks (no GPU)** — Every epoch (~10 min), all committed models are verified:
   - Architecture compliance (≤33B total params, Kimi-family architecture, vocab_size=163,840, no quantization)
   - **Duplicate detection** — SHA256 hash of safetensors weights; identical weights to an existing model → blacklisted for that commitment. Earlier commitment (by block number) owns the hash.
   - **Integrity** — Model must still be public and unchanged on HuggingFace
   - Models that fail pre-checks are **never sent to GPU** — no wasted compute
   - `check_model.py` and `test_miner.py` run 15 validator checks — the same checks the validator uses

2. **King identification** — The miner with the highest `composite.final` in `state/composite_scores.json` is the king (current emissions winner). KL alone never crowns anyone.

3. **Single-eval-per-commitment + paired king re-eval (live 2026-04-29, v30.2)** — Each on-chain commitment is scored EXACTLY ONCE for non-king miners. The **king is re-evaluated EVERY round** so its score is on the same procedural items as challengers (paired-fairness fix). New commitments enter rounds FIFO by `commit_block`, capped at `SINGLE_EVAL_MAX_PER_ROUND` (default 10) plus the always-in reference baseline (UID -1, `Qwen/Qwen3.5-4B`). Re-evaluation for non-king triggers only when a UID re-commits a different model on-chain, or when the composite schema bumps.

4. **Per-UID eval on the pod** — Each student is loaded sequentially on the pod (vLLM teacher → student forward pass → bench battery). The reference baseline runs in every round so the asymmetric reference-broken-axes filter (see `composite.py::resolve_reference_broken_axes`) can drop axes the base model itself can't pass under the eval setup (token cap, etc.).

5. **Multi-axis composite, normalized per axis** — Every student gets a vector covering 11 v31 procedural axes, distillation, judge, and discipline. Axis weights live in `composite.py:AXIS_WEIGHTS` (relative tier), `BENCH_AXIS_WEIGHTS` (legacy sub-axes — all weight 0 in v31.2), `BENCH_GROUP_AXIS_WEIGHTS` (legacy skill groups + super_teacher — `code_skill_group` and `tool_use_bench` retired to telemetry on 2026-05-10), and `ARENA_V3_AXIS_WEIGHTS` (kept-separate capability axes). Each axis is in [0, 1]. The composite stores `final = 0.75·worst_3_mean + 0.25·weighted` (the ranking key; K reverted from 5 back to 3 in v32.5 on 2026-05-13), `worst_3_mean = mean(bottom 3 non-broken axes)` (the API field name now matches the math), `worst = min(axes)` (legacy telemetry only), and `weighted = Σ wᵢ · axisᵢ / Σ wᵢ`.

6. **vLLM-accelerated evaluation** — vLLM generates teacher continuations 5–10× faster than pure HuggingFace inference. Teacher logits are precomputed and cached on GPU. Multi-GPU pod scaffolding (`DISTIL_TP_SIZE`, `DISTIL_STUDENT_PARALLELISM`) supports 4× / 8× H100 migration for Kimi K2.6 / batched student forward (v30.4).

7. **Cross-round dethronement gate (v31.2)** — King selection runs over `state/composite_scores.json`. The king is whoever has the highest `composite.final`. A challenger dethrones the incumbent only when `challenger.final > incumbent.final × (1 + SINGLE_EVAL_DETHRONE_MARGIN)` (**5% margin** since 2026-05-10 — raised from 3% in the v31.1 variance-reduction sweep). Combined with the per-axis n bumps and the wider K=5 worst-K mean, the false-positive dethrone rate from pure RNG variance dropped from ~27% per round to **<6%**. When both are at the saturated floor, the same margin applies to `composite.weighted` as a tiebreaker. Legacy records (lacking `final`) fall back to the v28 `worst`-based rule.

8. **Weight setting** — King gets weight=1.0, everyone else gets 0.0. Raw scores, no EMA smoothing. Weights are set on-chain immediately after each evaluation.

**Why this design beats KL-alone ranking:**
- **One eval per commitment** — miners pay for the eval once via the on-chain registration burn; no infinite re-evals
- **Block-seeded prompts** — the 300-prompt pool for each UID is fresh, deterministic from the round's block hash, and never re-used (no leakage)
- **Absolute multi-axis floor** — `composite.worst` is normalized per axis, so a student that gets 0.95 KL and 0.0 math fails the gate even if KL alone is "better than the king"
- **Asymmetric broken-axes filter** — eval-setup-fragile axes (e.g. AIME under a 768-token cap, MBPP function naming) drop out of `worst()` only when the reference itself scores 0 on them. They stay in `weighted` so a student beating the broken reference still gets credit.
- **Stable across schema changes** — `_KING_SELECTION_MIN_AXES` ensures only schema-current records compete for the crown. Older records remain tracked but ineligible until the miner re-commits.
- **Revision pinning** — models evaluated at the specific committed revision; new HF commits without on-chain re-commitment = integrity DQ
- **Anti-spiral safeguards** — `reasoning_density` axis penalises thinking-without-answering; `thinking_collapse_probe` flags models that loop indefinitely on trivial prompts (`"Hi"`, `"largest planet one word"`, `"say the word: done"`).

### Disqualification

Models are disqualified (composite.worst = 0, $0 earnings) for that commitment:
- **COPY** — Same safetensors weights as another miner (SHA256 match). First committer owns the hash.
- **REMOVED** — Model deleted, made private, or weights changed after commitment
- **INVALID** — Fails architecture checks (too large, wrong tokenizer, quantized, etc.)

Disqualification is **per-commit** — entries are keyed by `hotkey:commit_block`. A disqualified miner can register a new hotkey and submit a different model. The commitment itself is permanent (can't change it), but DQ doesn't prevent future registrations.

Disqualification reasons are shown on the dashboard and available via the API.

### Anti-Gaming

- **SHA256 hash duplicate detection**: Model weight hashes tracked forever; copies blacklisted for that commitment
- **Logit fingerprinting**: Even if hashes differ, models with identical activation distributions on the first 2 prompts are flagged as functional copies (cosine similarity > 0.99999 on per-position vectors)
- **Activation fingerprint near-copy detection**: the validator records a per-position activation fingerprint for every scored student in `state/activation_fingerprints.json`; pairs with cosine similarity ≥ `settings.activation_fp_threshold` (default 0.99999) are DQ'd as functional copies
- **Anti-spiral**: `reasoning_density` axis + `thinking_collapse_probe` catch the "model thinks forever, never answers" failure mode (see [paper/off_policy_cot_collapse.md](paper/off_policy_cot_collapse.md))
- **Commitment block priority**: Earlier on-chain commitment wins hash ownership
- **Revision-pinned integrity**: Models checked for new HF commits (git SHA comparison) — any change after commitment = DQ. Much cheaper than re-hashing weights every epoch.
- **Continuous integrity checks**: Every epoch, all models verified public + unchanged
- **MoE-aware param counting**: Total params from safetensors metadata (not config estimates)
- **Quantization rejected**: GPTQ/AWQ/FP8 all blocked — architecture distillation only
- **Block-hash seeded prompts**: Deterministic from on-chain block hash, unpredictable before block finalization
- **Procedural v31 axes (11)**: Every item generated per round from the block-seed (no static answer key) — see [`scripts/v31/`](scripts/v31/) for the per-axis generators (math/code/reasoning/long-context/knowledge/ifeval/truthfulness/consistency)
- **Isomorphic Perturbation Testing (IPT)**: `v31_consistency_paraphrase` rotates first names within gender between paired isomorphic problems; a model that memorised the canonical wording fails the rotated one
- **GSM-NoOp topical distractor**: `v31_math_robustness` injects mathematically irrelevant clauses; a memorised distribution that triggers on surface keywords fails
- **Top-20 sparse KL**: Teacher (Kimi K2.6, served via OpenRouter) returns top-20 logprobs per position. Student softmaxes over the full 163,840-token Kimi vocab, then gathers + renormalizes to the same 20 positions for a proper KL on the shared support.

## Mining Guide

### Requirements

- Bittensor wallet registered on subnet 97
- HuggingFace account for model hosting
- Training infrastructure (your choice)

### Model Requirements

Your model must:
- Use the **same tokenizer** as the teacher (Kimi K2.6 BPE — `vocab_size=163,840`)
- Have ≤ **33B total parameters** (live cap; consult [`frontend/src/lib/subnet-config.json`](frontend/src/lib/subnet-config.json) for the authoritative number)
- Be in **safetensors** format (bf16/fp16, NOT INT4/INT8 — quantized students are rejected)
- Use a **Kimi-family architecture** (e.g. `KimiK25ForConditionalGeneration` or the inner `DeepseekV3ForCausalLM` text-only path); the architecture allowlist lives in `subnet-config.json`
- Be loadable via `AutoModelForCausalLM.from_pretrained()` (or the equivalent class for the architecture)
- Stay **public and unchanged** on HuggingFace — making a repo private or pushing new commits = DQ
- **No quantized models** (GPTQ/AWQ/GGUF rejected; Kimi K2.6's outer compressed-tensors wrapper is the teacher's, not yours to inherit)
- **Unique weights** — cannot be identical to any previously committed model

### Pre-Submission Check (Recommended)

Before committing, run the validation tools to verify your model passes ALL validator checks:

```bash
pip install click huggingface_hub transformers safetensors

# Recommended: test_miner.py runs all 15 validator checks locally
python test_miner.py --model-repo your-username/your-model

# Alternative: check_model.py (quick pre-GPU checks)
python check_model.py --model-repo your-username/your-model

# Full eval with KL scoring (requires GPU):
python check_model.py --model-repo your-username/your-model --eval

# Compare against current king:
python check_model.py --model-repo your-username/your-model --eval --king-repo aceini/q-dist
```

`test_miner.py` is the recommended pre-submission tool — it runs the same 15 checks the validator uses. Save your TAO — fix issues before committing.

### Submit Your Model

⚠️ **ONE SUBMISSION PER HOTKEY — commitments are permanent and cannot be changed.** If disqualified, you can register a new hotkey and try again.

```bash
pip install -e .

# Dry run first (validates everything without committing):
python miner.py \
    --network finney \
    --netuid 97 \
    --wallet-name my_wallet \
    --hotkey-name my_hotkey \
    --model-repo your-username/your-distilled-model \
    --dry-run

# Commit (PERMANENT — interactive confirmation required):
python miner.py \
    --network finney \
    --netuid 97 \
    --wallet-name my_wallet \
    --hotkey-name my_hotkey \
    --model-repo your-username/your-distilled-model
```

The miner script includes `--dry-run`/`--test-only` flags, interactive confirmation before committing, and post-commit verification. To change models, register a new hotkey.

### KL Ranges (baseline, no distillation training)

KL is one of the composite axes — useful for sanity-checking a fresh student before submission, but **not the ranking key**.

These ranges were calibrated against the previous Qwen3.5 teacher and are kept here as historical reference. Under the Kimi K2.6 teacher (vocab 163,840 vs Qwen3.5's 248,320), expect **higher absolute KL** for any student, simply because the support is larger and the teacher distribution is sharper. Don't chase the old Qwen-era numbers; check `composite.final` instead.

| Model (Qwen-era reference) | Params | KL (nats) | Notes |
|-------|--------|-----------|-------|
| Qwen3.5-4B | 4.66B | ~0.10–0.15 | Old strong baseline (no longer used) |
| Qwen3.5-2B | 2.27B | ~0.12–0.16 | Old competitive baseline |
| Qwen3.5-0.8B | 0.87B | ~0.17–0.21 | Old moderate baseline |

These are *untrained baselines*. Models with KL > 2.0 are disqualified, but a low-KL model can still fail the composite gate if it scores poorly on benches, on-policy RKL, reasoning-density, or any other axis. **A model that wins KL but loses on grade-school math cannot take the crown.**

## Training Guide

Want to train your own distilled model? Check out the community-contributed training script in [`examples/`](examples/).

### KL Distillation Training Script

> **Credit:** [caseus / @winglian](https://github.com/winglian) — contributed via [PR #1](https://github.com/unarbos/distil/pull/1).  
> **Original gist:** https://gist.github.com/winglian/a8fe6b859ca1f23abcdd550fd5cfa0c5

The script [`examples/distil_kl_train.py`](examples/distil_kl_train.py) trains a student model to match the teacher's output distribution using forward KL divergence on raw text from `karpathy/climbmix-400b-shuffle`.

### GPU Requirements

- **Full teacher (Kimi K2.6, INT4 compressed-tensors, ~250GB on disk):** validator runs on **8× H200 80GB** (or 8× H100); see [`docs/B200_EVAL_SETUP.md`](docs/B200_EVAL_SETUP.md). Tensor-parallel size 8 (`DISTIL_TP_SIZE=8`).
- **Student-side training:** 33B-class students typically need 4–8× H100/H200 for full-precision (bf16) training; LoRA + gradient checkpointing fits on smaller setups.
- **Local dev (small experiments only):** 2× 24GB GPUs (e.g. RTX 3090/4090) — fine for sub-2B sanity students against a small teacher; not representative of the live eval.

### Usage

**Standard training (2 GPUs — teacher + student):**
```bash
python examples/distil_kl_train.py --teacher_gpu 0 --student_gpu 1
```

**Start from a leaderboard model:**
```bash
python examples/distil_kl_train.py --student some_user/their_model --teacher_gpu 0 --student_gpu 1
```

**Local dev with smaller models (e.g. 2× 24GB GPUs):**
```bash
python examples/distil_kl_train.py \
    --teacher Qwen/Qwen3.5-4B \
    --student Qwen/Qwen3.5-0.8B \
    --teacher_gpu 0 --student_gpu 1
```

### Key Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `1e-4` | Learning rate |
| `--warmup_steps` | `10` | LR warmup steps |
| `--samples_per_step` | `100` | Samples per optimizer step |
| `--max_seq_len` | `640` | Max sequence length |
| `--kl_start_pos` | `128` | Compute KL from this position onward |
| `--save_every` | `500` | Checkpoint save interval |
| `--no_wandb` | — | Disable W&B logging |

See the script's docstring and `--help` for the full list of options.

## Validator Guide

### Requirements

- **GPU**: **8× H200 80GB** (or 8× H100 80GB) for the eval pod — required to fit Kimi K2.6's INT4 weights (~250GB on disk) plus per-student working set across tensor-parallel ranks. Single-GPU configs are not supported under the Kimi teacher.
- **Bittensor wallet** registered as validator on subnet 97
- **Python 3.10+**

### Quick Start

```bash
git clone https://github.com/unarbos/distil.git
cd distil
pip install .

# Run via the wrapper script:
bash scripts/run_validator.sh

# Or with PM2 (recommended):
pm2 start scripts/run_validator.sh --name distil-validator
pm2 save
```

If `pip install .` fails:

```bash
pip install "bittensor>=8.0.0" "bittensor-wallet>=2.0.0" "click>=8.0.0" \
    "transformers>=4.45.0" "huggingface-hub>=0.20.0" "numpy>=1.26.0" \
    "torch>=2.1.0" "safetensors>=0.4.0"
```

### What It Does

1. Loads the teacher model (Kimi K2.6) via vLLM with `tensor-parallel-size=8` for fast generation across 8× H200/H100 (each GPU holds ~140GB of teacher / activation working set)
2. Draws 120 prompts from ClimbMix-400B (`karpathy/climbmix-400b-shuffle`, 6542 shards), seeded by on-chain block hash
3. Polls for new challengers every epoch (~10 min)
4. Per-UID block-seeded eval (single-eval policy): each commitment is scored once on its own 300-prompt set, including the reference baseline (UID -1) every round. The king is re-evaluated EVERY round (paired-fairness fix). The king is selected cross-round on `composite.final` from `state/composite_scores.json`, with a 5% margin over the incumbent required for dethronement.
5. Teacher logits are precomputed and cached on GPU for fast scoring
6. Sets weights on-chain: king = 1.0, everyone else = 0.0

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--netuid` | `1` | Subnet UID (**use `97`**) |
| `--wallet-name` | `default` | Wallet name |
| `--hotkey-name` | `default` | Hotkey name |
| `--tempo` | `360` | Seconds between epochs |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### State

Stored in `./state/`. To reset and re-evaluate all models, delete the directory.

## API

Live data at `https://api.arbos.life`:

| Endpoint | Description |
|----------|-------------|
| `GET /` | API overview |
| `GET /api/metagraph` | Full subnet metagraph (UIDs, stakes, weights, incentive) |
| `GET /api/commitments` | Miner model commitments (HF links + block numbers) |
| `GET /api/scores` | Current KL scores, disqualification reasons, last eval details |
| `GET /api/price` | Token price, emission, market data |
| `GET /api/health` | Service status |
| `GET /v1/models` | OpenAI-compatible — returns the live king's HF repo id |
| `POST /v1/chat/completions` | OpenAI-compatible chat with the live king (tool-calling supported) |

All endpoints are public, no authentication required.

### Use the king with agent harnesses

`/v1/chat/completions` is wired up for tool calling via vLLM's
`qwen3_xml` parser, so any framework that talks OpenAI-compatible chat
completions can use the king as a tool-equipped agent model. See
[`docs/FLUE_INTEGRATION.md`](docs/FLUE_INTEGRATION.md) and the
ready-to-run [`examples/flue/sn97-king-tool-calling/`](examples/flue/sn97-king-tool-calling/)
agents for a Flue setup; the same pattern works with the OpenAI Python
SDK, Vercel AI SDK, LangChain, LlamaIndex, and friends.

## Architecture

```
├── miner.py                  # One-shot commitment script (--dry-run, interactive confirm)
├── test_miner.py             # Pre-submission validator (runs all 15 checks locally)
├── check_model.py            # Pre-submission checker (13 pre-GPU + 4 GPU checks)
├── eval/
│   ├── kl_divergence.py      # Sparse top-20 KL on GPU (Kimi K2.6 teacher; dense path available for offline replays)
│   ├── model_checker.py      # Param counting, integrity, hash, duplicate detection
│   ├── dataset.py            # ClimbMix-400B dataset loader (300 prompts/UID, block-hash seeded shard selection)
│   └── scoring.py            # Winner-take-all + cross-round composite.final dethronement (5% margin, single-eval mode)
├── api/
│   └── server.py             # FastAPI dashboard backend (runs on separate API server behind Cloudflare)
├── scripts/
│   ├── pod_eval_vllm.py      # GPU eval runner: vLLM teacher generation + HF logit extraction,
│   │                         #   GPU-resident teacher logits, early stopping, king-in-VRAM
│   ├── remote_validator.py   # King-of-the-hill validator (Hetzner + Lium GPU)
│   ├── cosine_similarity_check.py  # Near-copy detection between models
│   └── run_validator.sh      # PM2 wrapper
└── state/                    # Persistent scores, hashes, disqualifications
```

### Split Validator Architecture

The validator now runs as a split architecture across two trust boundaries:

- **Dedicated `distil` host** (secure): Wallet keys, chain access, weight setting, validator orchestration, API, dashboard, and persistent state all live on one Distil-only Hetzner machine.
- **Lium GPU pod** (remote): Teacher/student forward passes, KL computation, vLLM inference. This machine has the GPU but **no chain access** — it cannot set weights or read wallet keys.
- **Optional external helpers**: Benchmark sync and chat inference can remain remote, but they do not hold wallet keys or perform weight setting.

Wallet keys never leave the `distil` host. The GPU pod receives evaluation tasks and returns scores. This separation ensures that even a compromised GPU pod cannot steal funds or manipulate weights directly.

## Community Contributions

SN97 welcomes contributions from the community! Notable contributions so far:

- **caseus ([@winglian](https://github.com/winglian))** — KL distillation training script ([PR #1](https://github.com/unarbos/distil/pull/1)), plus the suggestion to use top-k=128 shadow KL for more efficient evaluation

PRs are welcome — whether it's training scripts, evaluation tools, documentation, or ideas for improving the subnet.

## License

MIT
