# `check_model_validator.py`

Validator-aligned pre-submission checker for **Distil SN97**.

It performs static repository checks and (optionally) GPU scoring using live validator APIs:

- Prompts/continuation budget: [`/api/eval-data`](https://distil.arbos.life/api/eval-data)
- Current king metadata/model id: [`/api/h2h-latest`](https://distil.arbos.life/api/h2h-latest)

This README is only for [`check_model_validator.py`](check_model_validator.py).  
For the simpler non-validator-aligned checker, see [`check_model.py`](check_model.py).

## Requirements

**Always (CPU checks):**

```bash
pip install click huggingface_hub transformers safetensors
```

**With `--eval` (GPU + optional datasets):**

```bash
pip install torch datasets
```

Validator-aligned eval needs **network access** to `distil.arbos.life` unless you use `--offline-eval`.

## What it does

### 1. Pre-GPU checks (no CUDA)

- Repository public / accessible on Hugging Face  
- No disallowed custom Python in the model repo  
- Safetensors present, size within subnet limits  
- Config: architecture, vocab (teacher-aligned), no banned patterns  
- Tokenizer encoding matches teacher  
- Optional duplicate / integrity helpers (when `eval/` helpers are available)

If any **hard failure** occurs, the script exits non-zero before any GPU work.

### 2) GPU evaluation (`--eval`)

Two modes:

| Mode | Prompts | Teacher continuation length | King model |
|------|---------|-------------------------------|------------|
| **Default (validator-aligned)** | From [eval-data](https://distil.arbos.life/api/eval-data) | `max_new_tokens` from API (often 8192) | `king_model` from [h2h-latest](https://distil.arbos.life/api/h2h-latest), unless you pass `--king-repo` |
| **`--offline-eval`** | Local climbmix sampling (`--dataset`) | 512 | `--king-repo` or h2h fetch / fallback |

**Scoring details (GPU path):**

- Continuation-only forward KL(teacher ‖ student)
- fp32 log-softmax
- `F.kl_div(..., log_target=True)`
- Same teacher-generated continuation is reused when scoring student and king
- Reports both:
  - mean KL
  - paired stats (`delta_i = king_kl_i - student_kl_i`, one-sided p-value)
- Prints live reference thresholds when available (`paired_test_alpha`, `epsilon_threshold`)

**Anti-cheat-style runtime checks** (warnings / failures): student VRAM cap, generation speed benchmark, KL “too low” fraud band.

## CLI reference

| Option | Description |
|--------|-------------|
| `--model-repo` | **Required.** Hugging Face model id (e.g. `user/model`). |
| `--revision` | Optional commit SHA; if omitted, latest is resolved and pinned. |
| `--eval` | Run GPU evaluation after static checks. |
| `--prompts` | Max number of prompts. **0** = use the **full** list from eval-data (validator mode) or **20** prompts (offline mode when 0). |
| `--offline-eval` | Do **not** call eval-data; use local climbmix prompts instead. |
| `--validator-eval-data-url` | Override eval-data URL (default: `https://distil.arbos.life/api/eval-data`). |
| `--validator-h2h-url` | Override h2h-latest URL (default: `https://distil.arbos.life/api/h2h-latest`). |
| `--king-repo` | Force king HF repo; default is `king_model` from h2h-latest. |
| `--king-revision` | Optional king revision. |
| `--dataset` | Only for `--offline-eval`: HF dataset id for prompt sampling. |
| `--teacher-cache` | Optional `.pt` cache to skip teacher forward/generate if compatible. |

## Example commands

Static checks only (no GPU):

```bash
python check_model_validator.py --model-repo your-org/your-model
```

Full validator-aligned eval (network + GPU; can be slow and memory-heavy):

```bash
python check_model_validator.py --model-repo your-org/your-model --eval
```

Smoke test on the first 50 live prompts only:

```bash
python check_model_validator.py --model-repo your-org/your-model --eval --prompts 50
```

Force explicit king (if you want to compare against one specific repo/revision):

```bash
python check_model_validator.py \
  --model-repo your-org/your-model \
  --eval \
  --king-repo abacada/e
```

Offline / legacy local prompts:

```bash
python check_model_validator.py --model-repo your-org/your-model --eval --offline-eval --prompts 20
```

Custom API endpoints (debug/testing):

```bash
python check_model_validator.py \
  --model-repo your-org/your-model \
  --eval \
  --validator-eval-data-url https://distil.arbos.life/api/eval-data \
  --validator-h2h-url https://distil.arbos.life/api/h2h-latest
```

## Recommended workflow

1. **Train checkpoints** with your training script.
2. **Evaluate with this validator-aligned checker**:
   - quick loop: `--prompts 50`
   - final check: full prompts (`--prompts 0`, default)
3. **Pick checkpoint by paired stats**, not train loss alone:
   - positive mean delta (`king - student`)
   - low paired p-value
4. **Run one final full eval** before submit.

## Optional: one-time eval cache workflow

If teacher-target generation for eval is expensive, prebuild the eval cache once from live API prompts using `distil_kl_train_prebuilt.py`, then reuse it across training runs.

Build once:

```bash
python /root/distil/examples/distil_kl_train_prebuilt.py build_eval_cache_api \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --teacher_gpu 0 \
  --teacher_gpu_count 1 \
  --eval_data_url https://distil.arbos.life/api/eval-data \
  --eval_prompts 0 \
  --eval_cache_path /root/distil-online-checkpoints/eval_cache_api.pt
```

Reuse during training:

```bash
python /root/distil/examples/distil_kl_train_prebuilt.py train \
  ... \
  --eval_cache_path /root/distil-online-checkpoints/eval_cache_api.pt
```

Notes:

- `--eval_prompts 0` means “use all prompts returned by `eval-data`”.
- Add `--rebuild_eval_cache` when you want to refresh after API changes.
- Reusing this cache avoids re-running teacher continuation generation each training restart/resume.

## Practical notes

- **Cost:** Validator-style eval may use **very long** generations (`max_new_tokens` from eval-data). Use `--prompts` for iteration; reserve full runs for final confidence.
- **Teacher cache:** `--teacher-cache` is only useful if it matches the **same** prompt list and generation settings; otherwise the script regenerates.
- **Not a guarantee:** This tool aligns to published APIs and scoring style, but on-chain outcomes can still differ due to timing, infra, and evolving validator internals.
- **King can change:** always trust live [`/api/h2h-latest`](https://distil.arbos.life/api/h2h-latest) at evaluation time.
- **Prompt bundle can change:** always trust live [`/api/eval-data`](https://distil.arbos.life/api/eval-data).

## Exit status

- **0:** All static checks passed; if `--eval` was used, GPU eval finished without unhandled errors (warnings may still be printed).
- **Non-zero:** At least one failing check or eval error.
