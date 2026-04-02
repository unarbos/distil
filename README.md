# Distil — SN97

A Bittensor subnet for competitive model distillation of **Qwen/Qwen3.5-35B-A3B** (35B total, 3B active MoE).

**Dashboard**: [distil.arbos.life](https://distil.arbos.life)  
**API**: [api.arbos.life](https://api.arbos.life)  
**Subnet**: Finney netuid 97

## How It Works

**Miners** distill the teacher into a smaller model (≤5.25B total params), upload to HuggingFace, and commit the repo link on-chain. **One commitment per hotkey — commitments are permanent and cannot be changed.** However, if disqualified, miners can register a new hotkey and submit a different model.

**Validators** evaluate by computing full-distribution KL-divergence on GPU using a vLLM-accelerated pipeline. Lower KL = better distillation = higher rewards. **Winner-take-all** — best miner gets 100% of emissions.

### King-of-the-Hill Evaluation

The validator uses a **king-of-the-hill** architecture for efficient, high-confidence scoring:

1. **Pre-checks (no GPU)** — Every epoch (~10 min), all committed models are verified:
   - Architecture compliance (≤5.25B params, vocab_size=248,320, no quantization)
   - **Duplicate detection** — SHA256 hash of safetensors weights; identical weights to an existing model → blacklisted for that commitment. Earlier commitment (by block number) owns the hash.
   - **Integrity** — Model must still be public and unchanged on HuggingFace
   - Models that fail pre-checks are **never sent to GPU** — no wasted compute
   - `check_model.py` runs 13 pre-GPU + 4 GPU checks — the same checks the validator uses

2. **King identification** — The miner with the lowest KL score from state is the "king" (current emissions winner)

3. **Challenger detection** — Only models that haven't been evaluated yet are challengers. Already-evaluated models that didn't beat the king are not re-evaluated (their scores are final).

4. **Head-to-head GPU eval** — The king and all new challengers are scored together on the **same 60 ClimbMix-400B prompts** (block-hash seeded). Both models see identical teacher continuations, making the comparison fair. The king is only put on GPU when there's a challenger — no wasted compute on idle re-evaluation.

5. **vLLM-accelerated evaluation** — vLLM generates teacher continuations 5–10× faster than pure HuggingFace inference. The validator uses a hybrid approach: vLLM for fast teacher text generation, then HF for full-vocab logit extraction. Teacher logits are precomputed as softmax and cached on GPU, staying resident in VRAM during scoring.

6. **Early stopping** — Models clearly worse than the king are stopped early (`MIN_PROMPTS_EARLY_STOP=7`) to save GPU time. The king model also stays loaded in VRAM to avoid repeated loading.

7. **Epsilon threshold (1%)** — A challenger must achieve KL divergence **more than 1% lower** than the king's to dethrone it. For example, if the king has KL=0.097, a challenger needs KL < 0.096 (= 0.097 × 0.99). This prevents noisy near-ties from flipping the winner every epoch and rewards meaningful improvements.

8. **Weight setting** — King gets weight=1.0, everyone else gets 0.0. Raw scores, no EMA smoothing. Weights are set on-chain immediately after each evaluation.

**Why this is better than evaluating all models every epoch:**
- **More prompts per model** (60 vs 20) → tighter confidence intervals (95% CI, z=1.96), lower variance
- **GPU only runs when needed** — no challengers means no GPU eval at all
- **Fair comparison** — king and challenger scored on identical prompts in the same run
- **Epsilon prevents flip-flopping** — the king holds unless clearly beaten
- **Scales to many miners** — 100 miners with 1 new challenger = 2 models evaluated, not 100
- **Early stopping** saves GPU time on clearly inferior models

### Disqualification

Models are disqualified (KL=∞, $0 earnings) for that commitment:
- **COPY** — Same safetensors weights as another miner (SHA256 match). First committer owns the hash.
- **REMOVED** — Model deleted, made private, or weights changed after commitment
- **INVALID** — Fails architecture checks (too large, wrong tokenizer, quantized, etc.)

Disqualification is **per-commit** — entries are keyed by `hotkey:commit_block`. A disqualified miner can register a new hotkey and submit a different model. The commitment itself is permanent (can't change it), but DQ doesn't prevent future registrations.

Disqualification reasons are shown on the dashboard and available via the API.

### Anti-Gaming

- **SHA256 hash duplicate detection**: Model weight hashes tracked forever; copies blacklisted for that commitment
- **Logit fingerprinting**: Even if hashes differ, models with identical KL distributions on the first 2 prompts are flagged as functional copies (cosine similarity > 0.9999 on per-position KL vectors)
- **Commitment block priority**: Earlier on-chain commitment wins hash ownership
- **Integrity verification**: Models verified public + unchanged before every weight-set
- **MoE-aware param counting**: Total params from safetensors metadata (not config estimates)
- **Quantization rejected**: GPTQ/AWQ/FP8 all blocked — architecture distillation only
- **Block-hash seeded prompts**: Deterministic from on-chain block hash, unpredictable before block finalization
- **Full-distribution KL**: Scored on all 248,320 tokens, not top-k

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
- Be loadable via `AutoModelForCausalLM.from_pretrained()`
- **No quantized models** (GPTQ/AWQ/GGUF rejected)
- **Unique weights** — Cannot be identical to any previously committed model

### Pre-Submission Check (Recommended)

Before committing, run the checker to verify your model passes ALL validator checks:

```bash
pip install click huggingface_hub transformers safetensors

# Quick check (no GPU needed) — runs 13 pre-GPU checks:
python check_model.py --model-repo your-username/your-model

# Full eval with KL scoring (requires GPU) — runs 13 pre-GPU + 4 GPU checks:
python check_model.py --model-repo your-username/your-model --eval

# Compare against current king:
python check_model.py --model-repo your-username/your-model --eval --king-repo aceini/q-dist
```

This runs the same checks the validator uses. Save your TAO — fix issues before committing.

### Submit Your Model

⚠️ **ONE SUBMISSION PER HOTKEY — commitments are permanent and cannot be changed.** If disqualified, you can register a new hotkey and try again.

```bash
pip install -e .

python miner.py \
    --network finney \
    --netuid 97 \
    --wallet-name my_wallet \
    --hotkey-name my_hotkey \
    --model-repo your-username/your-distilled-model
```

To change models, register a new hotkey.

### KL Ranges (baseline, no distillation training)

| Model | Params | KL (nats) | Notes |
|-------|--------|-----------|-------|
| Qwen3.5-4B | 4.66B | ~0.10–0.15 | Strong baseline |
| Qwen3.5-2B | 2.27B | ~0.12–0.16 | Competitive |
| Qwen3.5-0.8B | 0.87B | ~0.17–0.21 | Moderate |

These are *untrained baselines* — purpose-built distillations should do significantly better. Models with KL > 2.0 are disqualified.

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
2. Draws 60 prompts from ClimbMix-400B (`karpathy/climbmix-400b-shuffle`, 6542 shards), seeded by on-chain block hash
3. Polls for new challengers every epoch (~10 min)
4. Head-to-head KL evaluation: king vs challengers on identical prompts, with early stopping for clearly worse models
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
├── miner.py                  # One-shot commitment script
├── check_model.py            # Pre-submission checker (13 pre-GPU + 4 GPU checks)
├── eval/
│   ├── kl_divergence.py      # Full-distribution KL on GPU
│   ├── model_checker.py      # Param counting, integrity, hash, duplicate detection
│   ├── dataset.py            # ClimbMix-400B dataset loader (60 prompts, block-hash seeded shard selection)
│   └── scoring.py            # Winner-take-all + disqualification tracking
├── api/
│   └── server.py             # FastAPI dashboard backend
├── scripts/
│   ├── pod_eval_vllm.py      # GPU eval runner: vLLM teacher generation + HF logit extraction,
│   │                         #   GPU-resident teacher logits, early stopping, king-in-VRAM
│   ├── remote_validator.py   # King-of-the-hill validator (Hetzner + Lium GPU)
│   └── run_validator.sh      # PM2 wrapper
└── state/                    # Persistent scores, hashes, disqualifications
```

### Split Validator Architecture

The validator runs as a split architecture across two machines:

- **Hetzner server** (secure): Wallet keys, chain access, weight setting, commitment monitoring. This machine has no GPU but holds all sensitive credentials.
- **Lium GPU pod** (remote): Teacher/student forward passes, KL computation, vLLM inference. This machine has the GPU but **no chain access** — it cannot set weights or read wallet keys.

Wallet keys never leave the Hetzner server. The GPU pod receives evaluation tasks and returns scores. This separation ensures that even a compromised GPU pod cannot steal funds or manipulate weights directly.

## Community Contributions

SN97 welcomes contributions from the community! Notable contributions so far:

- **caseus ([@winglian](https://github.com/winglian))** — KL distillation training script ([PR #1](https://github.com/unarbos/distil/pull/1)), plus the suggestion to use top-k=128 shadow KL for more efficient evaluation

PRs are welcome — whether it's training scripts, evaluation tools, documentation, or ideas for improving the subnet.

## License

MIT
