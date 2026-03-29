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

1. **Pre-checks (no GPU)** — Every epoch, all models are checked for:
   - Architecture compliance (param count, vocab size, no quantization)
   - **Duplicate detection** — SHA256 hash of safetensors weights via HF API; identical weights to an existing model → permanently blacklisted. Earlier commitment (by block number) wins.
   - **Integrity** — Model must still be public and unchanged on HuggingFace
2. **King identification** — The miner with the lowest KL score is the "king"
3. **Challenger evaluation** — Only **new/unevaluated** models are scored head-to-head against the king on GPU
4. **Higher confidence** — 40 prompts per evaluation (vs 20 in broad-sweep mode) for tighter confidence intervals
5. **King re-validation** — The king is re-evaluated with fresh prompts every 6 epochs
6. **Crown transfer** — If a challenger beats the king's KL, it becomes the new king
7. **Weight setting** — King gets weight=1.0, everyone else gets 0.0. Raw scores, no EMA smoothing.

This means the validator spends GPU time only on models that could actually win, and gets 2x more samples per model for better point estimates.

### Disqualification

Models are permanently disqualified (KL=∞, $0 earnings) for:
- **COPY** — Same safetensors weights as another miner (SHA256 match). First committer owns the hash.
- **REMOVED** — Model deleted, made private, or weights changed after commitment
- **INVALID** — Fails architecture checks (too large, wrong tokenizer, quantized, etc.)

Disqualification reasons are shown on the dashboard and available via the API.

### Anti-Gaming

- **SHA256 hash duplicate detection**: Model weight hashes tracked forever; copies permanently blacklisted
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
