# Distillation Subnet

Bittensor subnet for competitive model distillation of [GLM-5](https://huggingface.co/zai-org/GLM-5) (744B params).

Miners submit distilled models (≤74.4B params, same tokenizer). Validators evaluate by KL-divergence of logprobs on coding prompts. **Winner-take-all**: lowest divergence gets all weight.

## Architecture

```
┌──────────┐     commit(model, revision)      ┌───────────┐
│  Miner   │ ─────────────────────────────────→│  Chain     │
│          │   set_reveal_commitment()         │  (Finney)  │
└──────────┘                                   └─────┬─────┘
                                                     │
                 get_all_revealed_commitments()       │
┌──────────┐ ←───────────────────────────────────────┘
│ Validator│
│          │──→ 1. Read commitments (model repo + git SHA)
│          │──→ 2. Check architecture (config.json, param count, vocab)
│          │──→ 3. Load teacher (GLM-5) + student via vLLM
│          │──→ 4. Generate logprobs on SweInfinite prompts
│          │──→ 5. Compute KL(teacher || student)
│          │──→ 6. Set weight 1.0 on lowest KL-div miner
└──────────┘
```

## Key Design Choices

- **Quantization-proof model checking** — validates architecture via HuggingFace `config.json` fields (hidden_size, num_layers, vocab_size), not file sizes. Adapted from [Affine Cortex](https://github.com/AffineLabs/affine-cortex).
- **Revision pinning** — commitments include HF git SHA, so validators evaluate the exact snapshot.
- **Commit/reveal on-chain** — uses `set_reveal_commitment()` / `get_all_revealed_commitments()` for tamper-proof model submissions.
- **Winner-take-all** — simplest incentive: best distillation gets all weight.
- **SweInfinite dataset** — real-world coding problems for evaluation.

## File Structure

```
validator.py          Single-file validator (Click CLI)
miner.py              Miner commit script (Click CLI)
eval/
  kl_divergence.py    KL(P||Q) from top-k logprobs
  inference.py        vLLM wrapper with revision support
  dataset.py          SweInfinite JSON loader
  model_checker.py    HF config.json architecture validation
sim/
  test_full.py        Full end-to-end simulation (no GPU needed)
dataset/              SweInfinite JSON files
```

## Quick Start

### Run the Simulation (no GPU, no chain)

```bash
pip install numpy
cd /home/openclaw/distillation
python -m sim.test_full
```

This runs a complete end-to-end test with synthetic logprobs:
- Two mock miners with different distillation quality
- Real KL-divergence computation
- Winner-take-all weight assignment
- All assertions verified

### Run Validator (requires GPU + chain)

```bash
pip install -e .
python validator.py \
  --network finney \
  --netuid 1 \
  --wallet-name default \
  --hotkey-name default \
  --tensor-parallel-size 4
```

### Run Miner (requires chain)

```bash
python miner.py \
  --network finney \
  --netuid 1 \
  --model-repo your-username/distilled-glm5-70b \
  --revision abc123def456
```

## Miner Requirements

| Requirement | Value |
|---|---|
| Max parameters | 74.4B (10% of GLM-5's 744B) |
| Tokenizer | Must match GLM-5 vocab_size (151,552) |
| Format | HuggingFace model repo |
| Commitment | On-chain via `set_reveal_commitment()` |

## Evaluation

Each epoch, the validator:
1. Samples 5 random SweInfinite coding prompts
2. Generates teacher (GLM-5) logprobs via vLLM
3. For each miner: checks architecture → loads model → generates student logprobs
4. Computes `KL(teacher || student)` averaged across prompts
5. Winner (lowest KL-div) gets weight 1.0, all others get 0.0

## License

MIT
