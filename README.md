# Distil — SN97

A Bittensor subnet for competitive model distillation of **Qwen/Qwen3.5-35B-A3B** (35B total, 3B active MoE).

**Dashboard**: [distil.arbos.life](https://distil.arbos.life)  
**API**: [api.arbos.life](https://api.arbos.life)  
**Subnet**: Finney netuid 97

## How It Works

**Miners** distill the teacher into a smaller model (≤5.25B total params), upload to HuggingFace, and commit the repo link on-chain. **One commitment per hotkey, permanently.**

**Validators** evaluate by computing full-distribution KL-divergence on GPU. Lower KL = better distillation = higher rewards. **Winner-take-all** — best miner gets 100% of emissions.

### King-of-the-Hill Evaluation

The validator uses a **king-of-the-hill** architecture for efficient, high-confidence scoring:

1. **Pre-checks (no GPU)** — Every epoch (~10 min), all committed models are verified:
   - Architecture compliance (≤5.25B params, vocab_size=248,320, no quantization)
   - **Duplicate detection** — SHA256 hash of safetensors weights; identical weights to an existing model → permanently blacklisted. Earlier commitment (by block number) owns the hash.
   - **Integrity** — Model must still be public and unchanged on HuggingFace
   - Models that fail pre-checks are **never sent to GPU** — no wasted compute

2. **King identification** — The miner with the lowest KL score from state is the "king" (current emissions winner)

3. **Challenger detection** — Only models that haven't been evaluated yet are challengers. Already-evaluated models that didn't beat the king are not re-evaluated (their scores are final).

4. **Head-to-head GPU eval** — The king and all new challengers are scored together on the **same 40 FineWeb prompts** (block-seeded). Both models see identical teacher continuations, making the comparison fair. The king is only put on GPU when there's a challenger — no wasted compute on idle re-evaluation.

5. **Epsilon threshold (1%)** — A challenger must achieve KL divergence **more than 1% lower** than the king's to dethrone it. For example, if the king has KL=0.097, a challenger needs KL < 0.096 (= 0.097 × 0.99). This prevents noisy near-ties from flipping the winner every epoch and rewards meaningful improvements.

6. **Weight setting** — King gets weight=1.0, everyone else gets 0.0. Raw scores, no EMA smoothing. Weights are set on-chain immediately after each evaluation.

**Why this is better than evaluating all models every epoch:**
- **2x more prompts per model** (40 vs 20) → tighter confidence intervals, lower variance
- **GPU only runs when needed** — no challengers means no GPU eval at all
- **Fair comparison** — king and challenger scored on identical prompts in the same run
- **Epsilon prevents flip-flopping** — the king holds unless clearly beaten
- **Scales to many miners** — 100 miners with 1 new challenger = 2 models evaluated, not 100

### Disqualification

Models are permanently disqualified (KL=∞, $0 earnings) for:
- **COPY** — Same safetensors weights as another miner (SHA256 match). First committer owns the hash.
- **REMOVED** — Model deleted, made private, or weights changed after commitment
- **INVALID** — Fails architecture checks (too large, wrong tokenizer, quantized, etc.)

Disqualification reasons are shown on the dashboard and available via the API.

### Anti-Gaming

- **SHA256 hash duplicate detection**: Model weight hashes tracked forever; copies permanently blacklisted
- **Logit fingerprinting**: Even if hashes differ, models with identical KL distributions on the first 2 prompts are flagged as functional copies (cosine similarity > 0.9999 on per-position KL vectors)
- **Commitment block priority**: Earlier on-chain commitment wins hash ownership
- **Integrity verification**: Models verified public + unchanged before every weight-set
- **MoE-aware param counting**: Total params from safetensors metadata (not config estimates)
- **Quantization rejected**: GPTQ/AWQ/FP8 all blocked — architecture distillation only
- **Block-seeded prompts**: Deterministic from block number, unpredictable in advance
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

### Submit Your Model

⚠️ **ONE SUBMISSION PER HOTKEY, PERMANENTLY.** Cannot update or re-commit.

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

## Validator Guide

### Requirements

- **GPU**: ~80GB+ VRAM (teacher model is 35B params in bf16 ≈ 70GB, plus student + overhead)
  - Recommended: 1x A100 80GB, 1x H100, or 2x RTX PRO 6000
- **Bittensor wallet** registered as a validator on subnet 97
- **Python 3.10+**

### Setup

```bash
git clone https://github.com/unarbos/distil.git
cd distil
pip install .
```

If `pip install .` fails, install dependencies manually:

```bash
pip install "bittensor>=8.0.0" "bittensor-wallet>=2.0.0" "click>=8.0.0" \
    "transformers>=4.45.0" "huggingface-hub>=0.20.0" "numpy>=1.26.0" \
    "torch>=2.1.0" "safetensors>=0.4.0"
```

### Run

```bash
python validator.py \
    --network finney \
    --netuid 97 \
    --wallet-name my_wallet \
    --hotkey-name my_hotkey
```

The validator will:
1. Load the teacher model (Qwen3.5-35B-A3B) — this takes ~70GB VRAM
2. Load 500 prompts from FineWeb (cached after first run)
3. Poll for new challenger models every epoch (~10 min)
4. When a new challenger is found, run head-to-head KL evaluation (40 prompts)
5. Set weights on-chain: king = 1.0, everyone else = 0.0

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `finney` | Bittensor network |
| `--netuid` | `1` | Subnet UID (use `97`) |
| `--wallet-name` | `default` | Wallet name |
| `--hotkey-name` | `default` | Hotkey name |
| `--tempo` | `360` | Seconds between epochs |
| `--samples-per-epoch` | `12` | Prompts per eval (increased to 40 in king-of-the-hill mode) |
| `--log-level` | `INFO` | Logging verbosity |

### State

The validator stores state in `./state/`:
- `scores.json` — Current KL scores per UID
- `model_hashes.json` — SHA256 hashes for copy detection
- `disqualified.json` — DQ'd UIDs and reasons
- `evaluated_uids.json` — Already-evaluated UIDs (won't re-eval)

To reset and re-evaluate all models, delete the `state/` directory.

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
├── eval/
│   ├── kl_divergence.py      # Full-distribution KL on GPU
│   ├── model_checker.py      # Param counting, integrity, hash, duplicate detection
│   ├── dataset.py            # FineWeb prompt loader (500 cached prompts)
│   └── scoring.py            # Winner-take-all + disqualification tracking
├── api/
│   └── server.py             # FastAPI dashboard backend
├── scripts/
│   ├── pod_eval.py           # GPU eval runner (runs on remote pod)
│   ├── remote_validator.py   # King-of-the-hill validator (Hetzner + Lium GPU)
│   └── run_validator.sh      # PM2 wrapper
└── state/                    # Persistent scores, hashes, disqualifications
```

### Split Validator Architecture

- **Hetzner server** (secure): Wallet keys, chain access, weight setting
- **Lium GPU pod** (remote): Teacher/student forward passes, KL computation
- Wallet keys never leave the Hetzner server. The GPU pod has no chain access.

## License

MIT
