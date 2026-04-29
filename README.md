# Distil — SN97

A Bittensor subnet for competitive model distillation of **Qwen/Qwen3.6-35B-A3B** (35B total, ~3B active MoE).

**Dashboard**: [distil.arbos.life](https://distil.arbos.life)  
**API**: [api.arbos.life](https://api.arbos.life)  
**Chat with the King**: [chat.arbos.life](https://chat.arbos.life) — try the current best distilled model  
**Subnet**: Finney netuid 97

## How It Works (v30.2 / v30.3, 2026-04-29)

**Miners** distill the teacher into a smaller model (≤7B total params), upload to HuggingFace, and commit the repo link on-chain. **One commitment per hotkey — commitments are permanent and cannot be changed.** However, if disqualified, miners can register a new hotkey and submit a different model.

**Validators** score every committed model on a **multi-axis composite** ([`scripts/validator/composite.py`](scripts/validator/composite.py)) covering teacher-similarity, capability against ground truth (procedural — no static answer key), conversational quality, generation discipline, and "exceeds the teacher" capability. See [`docs/MINER_FAQ.md`](docs/MINER_FAQ.md) for the full axis-by-axis playbook.

### Ranking key — `composite.final` (v30.2)

```
composite.final = α × worst_3_mean + (1 - α) × weighted
```

Default `α = 0.7`. So 70% of your score comes from the **mean of your 3 lowest non-broken axes**, and 30% from the **weighted mean of every axis**. This smooths single-axis noise (the legacy `worst()` floored 22% of the leaderboard at 0) while preserving anti-Goodhart pressure (you still can't camp specialists). The legacy `composite.worst` (single-axis min) is retained as telemetry — see API + dashboard.

### Axis structure (v30.2 collapsed without losing depth)

- **Skill groups** (the new ranking drivers): `code_skill_group` (0.20) = mean of {code, mbpp, debug, correction, refactor}; `math_skill_group` (0.18) = mean of {math, aime, robustness}; `reasoning_skill_group` (0.12) = mean of {reasoning, multi_doc, long_context}; `knowledge_skill_group` (0.07) = mean of {knowledge_v2, pragmatic}.
- **Beyond-teacher** (v30.2 — incentivises exceeding teacher): `super_teacher` (0.10) = `tanh(mean(max(0, student - teacher)) / 0.10)` over 16 verifiable benches. Pure distillation can't exceed teacher; this rewards Stage-4 GRPO + post-distillation SFT.
- **Teacher-similarity**: `on_policy_rkl` (0.35), `kl` (0.05), `top_k_overlap` (0.10), `capability` (0.10).
- **Shadow distillation axes** (research-validated, low weight 0-0.05): `kl_is`, `forking_rkl`, `teacher_trace_plausibility`, `entropy_aware_kl`, `tail_decoupled_kl`.
- **Quality**: `judge_probe` (0.15), `long_form_judge` (0.05), `chat_turns_probe` (0.08).
- **Stand-alone capability**: `tool_use_bench` (0.06), `ifeval_bench` (0.07), `calibration_bench` (0.06).
- **Discipline**: `length` (0.05), `degeneracy` (0.15), `reasoning_density` (0.05).

The king is whoever has the **highest `composite.final`**. **Winner-take-all** — best miner gets 100% of emissions on chain.

> **Why not just KL?** Pure forward-KL on teacher continuations rewards token-level surface match. A 4B student that mimics the teacher's "wait, let me reconsider" filler can win KL while never producing a final answer. We caught this on 2026-04-17 (UID 107: 4096-token loops on `"Hi"`, strictly worse than the unfine-tuned 4B base on every reasoning bench). The composite, the on-policy RKL axis, and the reasoning-density axis exist specifically to close that gap.

### King-of-the-Hill Evaluation

The validator uses a **king-of-the-hill** architecture for efficient, high-confidence scoring:

1. **Pre-checks (no GPU)** — Every epoch (~10 min), all committed models are verified:
   - Architecture compliance (≤5.25B params, vocab_size=248,320, no quantization)
   - **Duplicate detection** — SHA256 hash of safetensors weights; identical weights to an existing model → blacklisted for that commitment. Earlier commitment (by block number) owns the hash.
   - **Integrity** — Model must still be public and unchanged on HuggingFace
   - Models that fail pre-checks are **never sent to GPU** — no wasted compute
   - `check_model.py` and `test_miner.py` run 15 validator checks — the same checks the validator uses

2. **King identification** — The miner with the highest `composite.final` in `state/composite_scores.json` is the king (current emissions winner). KL alone never crowns anyone.

3. **Single-eval-per-commitment + paired king re-eval (live 2026-04-29, v30.2)** — Each on-chain commitment is scored EXACTLY ONCE for non-king miners. The **king is re-evaluated EVERY round** so its score is on the same procedural items as challengers (paired-fairness fix). New commitments enter rounds FIFO by `commit_block`, capped at `SINGLE_EVAL_MAX_PER_ROUND` (default 10) plus the always-in reference baseline (UID -1, `Qwen/Qwen3.5-4B`). Re-evaluation for non-king triggers only when a UID re-commits a different model on-chain, or when the composite schema bumps.

4. **Per-UID eval on the pod** — Each student is loaded sequentially on the pod (vLLM teacher → student forward pass → bench battery). The reference baseline runs in every round so the asymmetric reference-broken-axes filter (see `composite.py::resolve_reference_broken_axes`) can drop axes the base model itself can't pass under the eval setup (token cap, etc.).

5. **Multi-axis composite, normalized per axis** — Every student gets a vector covering skill-group axes, beyond-teacher, teacher-similarity, shadow distillation, quality, capability, and discipline. Axis weights live in `composite.py:AXIS_WEIGHTS` (relative tier), `BENCH_AXIS_WEIGHTS` (legacy sub-axes — most weight 0 in v30.2), `BENCH_GROUP_AXIS_WEIGHTS` (the v30.2 skill groups + super_teacher), and `ARENA_V3_AXIS_WEIGHTS` (kept-separate capability axes). Each axis is in [0, 1]. The composite stores `final = 0.7·worst_3_mean + 0.3·weighted` (the ranking key), `worst_3_mean = mean(bottom 3 non-broken axes)`, `worst = min(axes)` (legacy telemetry), and `weighted = Σ wᵢ · axisᵢ / Σ wᵢ`.

6. **vLLM-accelerated evaluation** — vLLM generates teacher continuations 5–10× faster than pure HuggingFace inference. Teacher logits are precomputed and cached on GPU. Multi-GPU pod scaffolding (`DISTIL_TP_SIZE`, `DISTIL_STUDENT_PARALLELISM`) supports 4× / 8× H100 migration for Kimi K2.6 / batched student forward (v30.4).

7. **Cross-round dethronement gate (v30.2)** — King selection runs over `state/composite_scores.json`. The king is whoever has the highest `composite.final`. A challenger dethrones the incumbent only when `challenger.final > incumbent.final × (1 + SINGLE_EVAL_DETHRONE_MARGIN)` (default 3% margin). When both are at the saturated floor, the same 3% margin applies to `composite.weighted` as a tiebreaker. Legacy v28 records (lacking `final`) fall back to the v28 `worst`-based rule.

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
- **Cosine similarity tool**: `scripts/cosine_similarity_check.py` provides offline near-copy detection between any two models
- **Anti-spiral**: `reasoning_density` axis + `thinking_collapse_probe` catch the "model thinks forever, never answers" failure mode (see [paper/off_policy_cot_collapse.md](paper/off_policy_cot_collapse.md))
- **Commitment block priority**: Earlier on-chain commitment wins hash ownership
- **Revision-pinned integrity**: Models checked for new HF commits (git SHA comparison) — any change after commitment = DQ. Much cheaper than re-hashing weights every epoch.
- **Continuous integrity checks**: Every epoch, all models verified public + unchanged
- **MoE-aware param counting**: Total params from safetensors metadata (not config estimates)
- **Quantization rejected**: GPTQ/AWQ/FP8 all blocked — architecture distillation only
- **Block-hash seeded prompts**: Deterministic from on-chain block hash, unpredictable before block finalization
- **Top-128 sparse KL**: Teacher returns top-128 logprobs per position (`--max-logprobs 128` on vLLM). Student softmaxes over the full 248,320-token vocab, then gathers + renormalizes to the same 128 positions for a proper KL on the shared support. Full-vocab dense path exists in `compute_kl_from_precomputed` for reference; disabled in prod for bandwidth (~150GB/round at full vocab).

## Mining Guide

### Requirements

- Bittensor wallet registered on subnet 97
- HuggingFace account for model hosting
- Training infrastructure (your choice)

### Model Requirements

Your model must:
- Use **same tokenizer** as Qwen3.5-35B-A3B (vocab_size=248,320)
- Have ≤ **5.25B total parameters** (15% of teacher's 35B)
- Be in **safetensors** format (bf16/fp16)
- Use **`Qwen3_5ForConditionalGeneration`** architecture (model_type=`qwen3_5`) — required for vLLM compatibility
- Be loadable via `AutoModelForCausalLM.from_pretrained()`
- Stay **public and unchanged** on HuggingFace — making a repo private or pushing new commits = DQ
- **No quantized models** (GPTQ/AWQ/GGUF rejected)
- **Unique weights** — Cannot be identical to any previously committed model

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

KL is one of the 17 composite axes — useful for sanity-checking a fresh student before submission, but **not the ranking key**.

| Model | Params | KL (nats) | Notes |
|-------|--------|-----------|-------|
| Qwen3.5-4B | 4.66B | ~0.10–0.15 | Strong baseline |
| Qwen3.5-2B | 2.27B | ~0.12–0.16 | Competitive |
| Qwen3.5-0.8B | 0.87B | ~0.17–0.21 | Moderate |

These are *untrained baselines*. Models with KL > 2.0 are disqualified, but a low-KL model can still fail the composite gate if it scores poorly on benches, on-policy RKL, reasoning-density, or any other axis. **A model that wins KL but loses on grade-school math cannot take the crown.**

## Training Guide

Want to train your own distilled model? Check out the community-contributed training script in [`examples/`](examples/).

### KL Distillation Training Script

> **Credit:** [caseus / @winglian](https://github.com/winglian) — contributed via [PR #1](https://github.com/unarbos/distil/pull/1).  
> **Original gist:** https://gist.github.com/winglian/a8fe6b859ca1f23abcdd550fd5cfa0c5

The script [`examples/distil_kl_train.py`](examples/distil_kl_train.py) trains a student model to match the teacher's output distribution using forward KL divergence on raw text from `karpathy/climbmix-400b-shuffle`.

### GPU Requirements

- **Full teacher (Qwen3.5-35B-A3B unquantized):** 2× A100 80GB+ recommended (one for teacher, one for student)
- **Local dev with smaller models:** 2× 24GB GPUs (e.g. RTX 3090/4090)

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

- **GPU**: 1x B200 192GB recommended (~100GB minimum VRAM: teacher 67GB + student 8GB + teacher logits cache 17GB + king model 8GB). A100 80GB is insufficient for the vLLM pipeline.
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

1. Loads the teacher model (Qwen3.5-35B-A3B) via vLLM for fast generation (~67GB VRAM)
2. Draws 120 prompts from ClimbMix-400B (`karpathy/climbmix-400b-shuffle`, 6542 shards), seeded by on-chain block hash
3. Polls for new challengers every epoch (~10 min)
4. Per-UID block-seeded eval (single-eval policy): each commitment is scored once on its own 300-prompt set, including the reference baseline (UID -1) every round. The king is selected cross-round on `composite.worst` from `state/composite_scores.json`.
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

All endpoints are public, no authentication required.

## Architecture

```
├── miner.py                  # One-shot commitment script (--dry-run, interactive confirm)
├── test_miner.py             # Pre-submission validator (runs all 15 checks locally)
├── check_model.py            # Pre-submission checker (13 pre-GPU + 4 GPU checks)
├── eval/
│   ├── kl_divergence.py      # Sparse top-128 KL on GPU (dense path available for offline replays)
│   ├── model_checker.py      # Param counting, integrity, hash, duplicate detection
│   ├── dataset.py            # ClimbMix-400B dataset loader (120 prompts, block-hash seeded shard selection)
│   └── scoring.py            # Winner-take-all + cross-round composite-worst dethronement (single-eval mode)
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
