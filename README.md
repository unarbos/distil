# 🧬 Distillation

**Competitive model compression for GLM-5 on Bittensor.**

Miners distill the 744B-parameter GLM-5 (MoE, 40B active) down to ≤74.4B parameters. Validators compare output distributions via KL-divergence on real coding tasks. The best distillation wins all emission weight.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Bittensor Chain                          │
│  ┌──────────────┐                        ┌───────────────────┐  │
│  │ Commitments   │◄── miner commits ────│  Yuma Consensus   │  │
│  │ Pallet        │    HF model URL       │  (weight setting) │  │
│  └──────┬───────┘                        └────────┬──────────┘  │
│         │                                         ▲              │
│         │ validator reads                         │ weights      │
│         ▼                                         │              │
│  ┌──────────────────────────────────────────────────┐           │
│  │                   Validator                       │           │
│  │                                                   │           │
│  │  1. Load GLM-5 (teacher) — kept in GPU memory     │           │
│  │  2. Sample N coding prompts from SweInfinite      │           │
│  │  3. Generate teacher logprobs                     │           │
│  │  4. For each miner:                               │           │
│  │     a. Check param count (≤74.4B)                 │           │
│  │     b. Check tokenizer compatibility              │           │
│  │     c. Load student model → inference → unload    │           │
│  │     d. Compute KL(teacher || student)             │           │
│  │  5. Winner-take-all: lowest KL → weight 1.0       │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture: Chi Pattern

This subnet follows the **Chi pattern** — no synapse, axon, or dendrite. Instead:

- **Miners** commit their HuggingFace model URL on-chain via the Commitments pallet
- **Validators** read commitments, download models, and run local GPU inference via vLLM

This eliminates network latency from scoring and enables rigorous logprob comparison.

---

## Scoring: KL-Divergence

The validator measures **KL(P_teacher || P_student)** — the Kullback-Leibler divergence from the teacher's output distribution to the student's, averaged across token positions and prompts.

**Why KL-divergence?**
- Directly measures how well the student approximates the teacher's *full* output distribution
- Rewards models that preserve uncertainty and confidence calibration, not just top-1 accuracy
- The natural loss function for knowledge distillation (Hinton et al., 2015)

**Winner-take-all**: The miner with the **lowest** average KL-divergence receives `weight = 1.0`. All other miners receive `weight = 0.0`. This creates maximum competitive pressure.

---

## Model Constraints

| Constraint | Value | Rationale |
|---|---|---|
| Max parameters | **74.4B** (10% of 744B) | Enforces actual compression |
| Tokenizer | Must match GLM-5 | Required for valid KL comparison |
| Hosting | Public HuggingFace Hub | Validators must download the model |

---

## Dataset: SweInfinite

Evaluation uses the **SweInfinite** dataset — real GitHub issues from open-source repositories. Each issue contains a `problem_statement` that becomes a coding prompt.

- ~20 JSON files, each with a real bug report / feature request
- Prompts are randomly sampled each epoch (prevents overfitting)
- Tests code understanding, reasoning, and generation

---

## For Miners

### 1. Distill GLM-5

Compress GLM-5 using any technique:
- Knowledge distillation (logit matching, hidden-state matching)
- Quantization (GPTQ, AWQ, GGUF)
- Pruning (structured, unstructured)
- Architecture search / layer dropping
- Any combination

**Requirements:**
- Same tokenizer as GLM-5 (`zai-org/GLM-5`)
- ≤ 74.4B total parameters
- Publicly hosted on HuggingFace Hub

### 2. Upload to HuggingFace

```bash
huggingface-cli upload your-username/distilled-glm5 ./model-directory
```

### 3. Run the Miner

```bash
# Install
pip install -e .

# Commit your model
python miner.py \
    --network finney \
    --netuid <NETUID> \
    --coldkey <WALLET_NAME> \
    --hotkey <HOTKEY_NAME> \
    --model-repo your-username/distilled-glm5
```

The miner will:
1. Verify your model's param count and tokenizer compatibility
2. Commit the model URL on-chain
3. Periodically re-commit to stay active

### Commitment Format

```json
{
    "model_repo": "username/distilled-glm5",
    "tokenizer": "zai-org/GLM-5",
    "params_b": 70.0
}
```

---

## For Validators

### GPU Requirements

Validators must run both the teacher (GLM-5, 744B MoE / ~40B active) and student models (up to 74.4B). This requires **serious GPU hardware**.

**Recommended:**
- 8× H100 80GB (or equivalent)
- vLLM with tensor parallelism

**Budget option:**
- Rent GPUs from [Lium.io](https://lium.io) (Bittensor SN51)

### Setup

```bash
# Clone and install
git clone https://github.com/Junjie4/distillation.git
cd distillation
pip install -e .

# Download the evaluation dataset
./scripts/download_dataset.sh

# Run the validator
python validator.py \
    --network finney \
    --netuid <NETUID> \
    --coldkey <WALLET_NAME> \
    --hotkey <HOTKEY_NAME> \
    --teacher-model zai-org/GLM-5 \
    --dataset-path ./dataset \
    --tensor-parallel-size 8 \
    --samples-per-epoch 5
```

### Validator CLI Options

| Flag | Default | Description |
|---|---|---|
| `--network` | `finney` | Bittensor network |
| `--netuid` | `1` | Subnet UID |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Wallet hotkey name |
| `--teacher-model` | `zai-org/GLM-5` | Teacher model HF repo |
| `--max-param-ratio` | `0.1` | Max student/teacher ratio |
| `--dataset-path` | `./dataset` | Path to SweInfinite data |
| `--samples-per-epoch` | `5` | Prompts per evaluation |
| `--max-tokens` | `128` | Generation length |
| `--top-k-logprobs` | `50` | Logprobs per token |
| `--tensor-parallel-size` | `1` | GPUs per model (vLLM TP) |
| `--gpu-memory-utilization` | `0.90` | vLLM GPU memory fraction |
| `--log-level` | `INFO` | Logging verbosity |

---

## Lium.io Integration

For validators who don't have local GPU capacity, [Lium.io](https://lium.io) (Bittensor SN51) provides on-demand GPU rental. Configure your validator to use a remote GPU endpoint for vLLM inference.

---

## Project Structure

```
distillation/
├── validator.py          # Single-file validator (Chi pattern)
├── miner.py              # Miner commitment script
├── eval/
│   ├── kl_divergence.py  # KL(teacher || student) computation
│   ├── inference.py      # vLLM inference wrapper
│   ├── dataset.py        # SweInfinite dataset loader
│   └── tokenizer.py      # Tokenizer compatibility checking
├── knowledge/            # Chi-style knowledge YAMLs
├── scripts/              # Shell scripts for running
├── tests/                # Unit tests
├── pyproject.toml        # Python package config
└── dataset/              # SweInfinite data (not in git)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]" pytest

# Run tests
pytest tests/

# Run with debug logging
python validator.py --log-level DEBUG
```

---

## License

MIT — see [LICENSE](LICENSE).
