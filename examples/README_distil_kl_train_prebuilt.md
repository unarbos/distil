# Online KL Distillation Training

This document explains how to use `examples/distil_kl_train_prebuilt.py` in its current **online-only** form.

## Commands

`distil_kl_train_prebuilt.py` supports:

1. `build_eval_cache_api`  
   Build deterministic eval cache from validator `eval_data` API prompts.
2. `train`  
   Train with **online teacher inference** from streaming dataset data.

## Requirements

```bash
pip install torch transformers datasets wandb
```

## Build eval cache (API)

```bash
python3 examples/distil_kl_train_prebuilt.py build_eval_cache_api \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --eval_data_url https://distil.arbos.life/api/eval-data \
  --eval_cache_path ./eval_cache_api.pt
```

## Train (online mode only)

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --student_gpu 1 --student_gpu_count 1 \
  --dataset karpathy/climbmix-400b-shuffle \
  --dataset_split train \
  --dataset_skip_rows 0 \
  --min_chars 2560 \
  --max_seq_len 1024 \
  --samples_per_step 100 \
  --max_steps 2000 \
  --output_dir ./distil-checkpoints
```

## Build a Benchmark-Mix JSONL

Use `examples/build_benchmark_mix_jsonl.py` to create a local `text` JSONL that targets the validator's composite axes: math, code, reasoning, instruction following, calibration, long-context retrieval, multi-document synthesis, and refactoring.

```bash
python3 examples/build_benchmark_mix_jsonl.py \
  --output ./distil-checkpoints/arena_mix.jsonl \
  --max_examples 50000
```

The builder streams from public instruction/code/math datasets and adds synthetic validator-style tasks. The generated file can be passed directly to `distil_kl_train_prebuilt.py` because local `.jsonl` files are supported as training datasets.

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student <YOUR_CURRENT_DISTILLED_MODEL_OR_CHECKPOINT> \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --student_gpu 1 --student_gpu_count 1 \
  --dataset ./distil-checkpoints/arena_mix.jsonl \
  --dataset_split train \
  --min_chars 0 \
  --max_seq_len 1536 \
  --kl_start_pos 64 \
  --lr 2e-6 \
  --samples_per_step 24 \
  --max_steps 1000 \
  --output_dir ./distil-checkpoints-arena
```

This is intended as a continued-training pass after KL distillation. Keep the learning rate low and continue checking KL/king metrics, because pure benchmark SFT can improve absolute axes while damaging KL, on-policy behavior, length discipline, or generation quality.

## Train with king comparison

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --student_gpu 1 --student_gpu_count 1 \
  --king_gpu 2 --king_gpu_count 1 \
  --king_repo <KING_REPO> \
  --max_steps 2000 \
  --eval_every_steps 200 \
  --eval_prompts 500 \
  --eval_use_api_prompts \
  --eval_data_url https://distil.arbos.life/api/eval-data \
  --eval_cache_path ./eval_cache_api.pt \
  --beat_king_p_threshold 0.05 \
  --beat_king_min_delta 0.0 \
  --early_stop_on_beat_consecutive 3 \
  --output_dir ./distil-checkpoints
```

## Resume

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --student_gpu 1 \
  --dataset karpathy/climbmix-400b-shuffle \
  --max_steps 2000 \
  --resume_latest \
  --output_dir ./distil-checkpoints
```

## Validator alignment notes

- `--eval_use_api_prompts` is the default for validator parity.
- If you choose dataset eval prompts (`--eval_use_dataset_prompts`), pass `--eval_block_hash` for better parity.
- Loaded eval cache metadata is validated for source/config compatibility.

## Important behavior

- `train` requires `--max_steps > 0` in online mode.
- Teacher and student are both loaded during training.
- `--dataset_skip_rows` is ignored when resuming (`--resume_from` / `--resume_latest`) because stream state is restored from checkpoint `data_state`.

