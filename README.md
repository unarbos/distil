# Distillation Subnet

A Bittensor subnet that incentivizes miners to produce the best knowledge-distilled version of **Qwen/Qwen3.5-35B-A3B** (35B total parameters, 3B active, MoE architecture, vocab_size=248,320).

## How It Works

**Miners** train a distilled/compressed model (≤3.5B total params) that preserves the teacher's output distribution as closely as possible. They upload the model to HuggingFace and commit the repo link on-chain. **One commitment per hotkey, permanently** — choose your model carefully.

**Validators** evaluate submitted models by computing full-distribution KL-divergence between the teacher and student on GPU. Lower KL = better distillation = higher rewards.

### Evaluation Pipeline

1. **Block-seeded prompt selection** — Prompts are sampled from the SweInfinite dataset using the current block number as seed (unpredictable, reproducible)
2. **Teacher continuation** — The teacher generates 512-token greedy continuations for each prompt
3. **Full-sequence KL** — Both models forward-pass the full sequence (prompt + continuation); KL is computed on continuation positions only, using the full 248K vocabulary
4. **EMA smoothing** — KL scores are smoothed with exponential moving average (α=0.3) across epochs
5. **Proportional weights** — Rewards are distributed proportionally via inverse-KL weighting (not winner-take-all)

### Anti-Gaming

- **Copy detection**: SHA256 hash of model weights prevents re-uploading existing models
- **MoE-aware param counting**: Limits on total params (not just active) prevent gaming via huge sparse models
- **Staleness timeout**: 3 consecutive failures → zero weight until new submission
- **Dynamic prompts**: Block-seeded selection prevents overfitting to specific prompts

## Requirements

### Validator
- **GPU**: 80GB+ VRAM (H100/A100) — teacher model runs in bfloat16
- **RAM**: 64GB+
- **Python**: 3.10+
- **Bittensor wallet** with validator registration on the subnet

### Miner
- **Training infrastructure**: Whatever you need to train your distilled model
- **HuggingFace account**: To host your model
- **Bittensor wallet** with miner registration on the subnet

## Quick Start

### Validator

```bash
# Install
pip install -e .

# Run
python validator.py \
    --network finney \
    --netuid <NETUID> \
    --wallet-name default \
    --hotkey-name validator \
    --teacher-model Qwen/Qwen3.5-35B-A3B \
    --dataset-path ./dataset \
    --samples-per-epoch 12 \
    --tempo 360
```

### Miner

⚠️ **ONE SUBMISSION PER HOTKEY, PERMANENTLY.** You cannot update, replace, or re-commit. Choose wisely.

```bash
# Install
pip install -e .

# Submit your model (THIS IS PERMANENT)
python miner.py \
    --network finney \
    --netuid <NETUID> \
    --wallet-name my_wallet \
    --wallet-path ~/.bittensor/wallets \
    --hotkey-name my_hotkey \
    --model-repo your-username/your-distilled-model

# To change models, you must register a new hotkey.
```

### Model Requirements

Your distilled model must:
- Use the **same tokenizer** as Qwen3.5-35B-A3B (vocab_size=248,320)
- Have ≤ **3.5B total parameters** (10% of teacher)
- Be hosted on **HuggingFace** in safetensors format
- Be loadable via `transformers.AutoModelForCausalLM`

## Architecture

```
distillation/
├── validator.py          # Main validator (Chi pattern, single long-running process)
├── miner.py              # Miner commitment script
├── eval/
│   ├── kl_divergence.py  # Full-distribution KL on GPU tensors + teacher continuation
│   ├── model_checker.py  # MoE-aware param counting, tokenizer verification, hash identity
│   ├── dataset.py        # SweInfinite loader with block-seeded selection
│   └── scoring.py        # EMA tracking, proportional weights, staleness management
├── state/                # Persistent validator state (scores, failures, caches)
├── scripts/
│   ├── pod_eval.py       # Standalone GPU eval with teacher continuation
│   └── pod_eval_fast.py  # Fast prompt-only KL eval (no generation)
├── sim/
│   └── test_full.py      # Full simulation (run without GPU)
└── dataset/              # SweInfinite JSON files
```

## Scoring Details

**KL Divergence**: `KL(teacher || student) = Σ P_teacher(x) · log(P_teacher(x) / P_student(x))`

Computed on full vocabulary (248K tokens) at each position of the teacher's generated continuation. This measures how well the student predicts what the teacher would generate — the gold standard for distillation quality.

**Weight Formula**: `weight_i = (1/KL_i) / Σ(1/KL_j)` for all miners below the quality threshold.

Lower KL → higher weight → more rewards. Continuous incentive to improve.

## License

MIT
