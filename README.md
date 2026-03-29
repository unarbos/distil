# Constantinople — SN97

A Bittensor subnet for competitive model distillation of **Qwen/Qwen3.5-35B-A3B** (35B total, 3B active MoE).

**Dashboard**: [distil.arbos.life](https://distil.arbos.life)  
**Subnet**: Finney netuid 97

## How It Works

**Miners** distill the teacher into a smaller model (≤5.25B total params), upload to HuggingFace, and commit the repo link on-chain. **One commitment per hotkey, permanently.**

**Validators** evaluate by computing full-distribution KL-divergence on GPU. Lower KL = better distillation = higher rewards. **Winner-take-all** — best miner gets 100% of emissions.

### Evaluation Pipeline

1. **Prompt sampling** — 20 prompts sampled from FineWeb pretraining corpus, seeded by block number
2. **Teacher continuation** — Teacher generates 512-token continuations per prompt
3. **Full-distribution KL** — Both models forward-pass the sequence; KL computed on 248K vocab at each continuation position
4. **EMA smoothing** — Scores smoothed with α=0.3 across epochs
5. **Winner-take-all** — Lowest KL miner gets weight=1.0, everyone else gets 0.0

### Anti-Gaming

- **Copy detection**: SHA256 of model weights (via HF API metadata)
- **Integrity checks**: Models verified public + unchanged before every weight-set
- **MoE-aware param counting**: Total params capped (not just active)
- **Quantization rejected**: GPTQ/AWQ/FP8 all blocked — architecture distillation only
- **Block-seeded prompts**: Unpredictable, reproducible selection

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

### Expected KL Ranges

| Model | Params | KL (nats) | Quality |
|-------|--------|-----------|---------|
| Qwen3.5-4B | 4.66B | ~0.24 | Strong |
| Qwen3.5-2B | 2.27B | ~0.35 | Good |
| Qwen3.5-0.8B | 0.87B | ~0.58 | Moderate |
| Random / broken | — | >5.0 | Rejected |

Target: KL < 0.3 for competitive mining. Models with KL > 2.0 receive zero weight.

## API

Live data at `https://api.arbos.life`:

- `GET /api/metagraph` — Full subnet metagraph
- `GET /api/commitments` — Miner model commitments
- `GET /api/scores` — Current KL scores + EMA
- `GET /api/price` — Token price, emission, market data
- `GET /api/health` — Service status

## Architecture

```
├── validator.py              # Chi-pattern validator
├── miner.py                  # One-shot commitment script
├── eval/
│   ├── kl_divergence.py      # Full-distribution KL on GPU
│   ├── model_checker.py      # Param counting, integrity, hash
│   ├── dataset.py            # FineWeb prompt loader
│   └── scoring.py            # EMA + winner-take-all
├── api/
│   └── server.py             # FastAPI dashboard backend
├── scripts/
│   ├── pod_eval.py           # GPU eval runner
│   └── remote_validator.py   # Split validator (Hetzner + Lium)
└── state/                    # Persistent scores, caches
```

## License

MIT
