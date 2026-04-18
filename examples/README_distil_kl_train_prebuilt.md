# Prebuilt KL Distillation Training

This document explains how to use `examples/distil_kl_train_prebuilt.py`.

The script keeps the original KL distillation logic, but splits work into offline stages so repeated training runs are faster and use less VRAM.

## What this script does

`distil_kl_train_prebuilt.py` supports three commands:

1. `build`  
   Stream raw text from dataset and save tokenized samples to disk.
2. `build_teacher_cache`  
   Run the teacher model once on tokenized samples and save teacher log-prob targets.
3. `train`  
   Train the student model using either:
   - online teacher inference (default), or
   - cached teacher targets (`--use_teacher_cache`).

The `train` command can also run periodic head-to-head eval against a king model and emit JSON metrics in this shape:

- `eval_stats`: `mean/std/p50/min/max/n`
- `king_stats`: `mean/std/p50/min/max/n`
- `ttest`: one-sided paired t-test with `p`, `t`, `delta` (`delta = mean(king_kl - eval_kl)`)

For stability, periodic eval uses a deterministic teacher-continuation cache generated once at run start (`--eval_seed`), then reuses those exact continuations for all later student-vs-king comparisons.

## Why use this

- Avoid repeated dataset streaming + tokenization.
- Avoid loading the teacher during training when using `--use_teacher_cache` (see [Memory and OOM](#memory-and-oom)).
- Keep strict compatibility checks between built data and teacher cache.

## Dataset and eval alignment

- Validator prompt sourcing is primarily `karpathy/climbmix-400b-shuffle`.
- If ClimbMix loading fails, validator code falls back to `HuggingFaceFW/fineweb`.
- For best training/eval alignment, build training data from ClimbMix.

Sequence-length note:

- `max_seq_len` in training (default `640`) is a training-time tradeoff, not the full validator eval cap.
- Validator eval pipeline can run on longer contexts than `640`.
- If VRAM allows, training with `max_seq_len=1024` usually improves alignment versus `640`.

## Requirements

Install dependencies similar to the base trainer:

```bash
pip install torch transformers datasets wandb
```

You also need enough GPU memory for:

- `build_teacher_cache`: teacher model (one forward at a time per sample)
- `train`:
  - student only (with `--use_teacher_cache`)
  - teacher on `--teacher_gpu` (+ `--teacher_gpu_count`) and student on `--student_gpu` (+ `--student_gpu_count`) (without `--use_teacher_cache`)

## Memory and OOM

The usual **Linux OOM kill** during distillation is from holding the **teacher and student in the same training loop** (two large models, or one model per GPU with insufficient headroom).

**Recommended mitigation:** run `build_teacher_cache` once, then `train ... --use_teacher_cache`. In that path the teacher weights are **not** loaded during training; only the student runs forward/backward, and targets are read from disk. That removes the dominant “two models in train” failure mode.

**What can still OOM**

- **`train` without `--use_teacher_cache`:** the teacher is loaded on `--teacher_gpu`/`--teacher_gpu_count` and the student on `--student_gpu`/`--student_gpu_count`. You need enough VRAM on the selected GPU sets.
- **Student and batch shape:** large `--samples_per_step`, long `--max_seq_len`, or a big student can exhaust the student GPU even with a cache. Reduce `samples_per_step` and/or `max_seq_len` first.
- **`build_teacher_cache`:** VRAM is bounded by one teacher forward at a time, but each chunk’s targets are collected in memory before `torch.save`. A very large `--chunk_size` with long sequences and full-vocab log-probs can stress **CPU RAM**; try a smaller `chunk_size` if the process dies during cache build.
- **Cache dtype:** `bf16` / `fp16` shrink cache files and the tensors moved to GPU during KL; `fp32` is heavier.

**Optional environment variable (CUDA fragmentation):** some setups see fewer allocator OOMs with:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This does not fix running out of memory for a model that is simply too large for the card.

## Teacher cache and disk space

Teacher cache stores teacher log-probabilities for continuation positions. For long contexts and large vocab, cache can be large.

Practical options:

1. Use `--teacher_cache_dtype bf16` (or `fp16`) to reduce disk footprint.
2. Keep `--chunk_size` moderate during `build_teacher_cache` to avoid memory pressure.
3. Use online teacher mode (omit `--use_teacher_cache`) when you prefer compute over disk.

## Quick start

### 1) Build tokenized data

```bash
python3 examples/distil_kl_train_prebuilt.py build \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --dataset karpathy/climbmix-400b-shuffle \
  --data_dir ./prebuilt-data \
  --num_samples 200000 \
  --chunk_size 2000 \
  --max_seq_len 1024 \
  --kl_start_pos 128 \
  --min_chars 2560
```

Output:

- `./prebuilt-data/manifest.json`
- `./prebuilt-data/chunks/chunk_*.pt`

### 2) Build teacher cache (one-time)

For large `num_samples`, prefer `bf16` (or `fp16`) unless you explicitly need `fp32`.

```bash
python3 examples/distil_kl_train_prebuilt.py build_teacher_cache \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --teacher_gpu 0 \
  --teacher_gpu_count 1 \
  --data_dir ./prebuilt-data \
  --teacher_cache_dtype bf16
```

Output:

- `./prebuilt-data/teacher_cache/manifest.json`
- `./prebuilt-data/teacher_cache/chunk_*/000000.pt` … (default **sharded** layout: one small file per sample)
- `./prebuilt-data/token_chunk_hashes.json`

Notes:

- `--teacher_cache_dtype fp32` is the heaviest; `bf16`/`fp16` reduce size and are often sufficient.

### 3) Train with cached teacher targets (recommended)

This path loads **only the student** during training; the teacher is not kept in VRAM for the optimization loop.

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --student_gpu 1 \
  --data_dir ./prebuilt-data \
  --use_teacher_cache \
  --max_seq_len 1024 \
  --samples_per_step 100 \
  --max_steps 1000 \
  --output_dir ./distil-checkpoints \
  --save_every 500
```

### 3b) Train with periodic king comparison (dethrone mode)

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --student_gpu 1 --student_gpu_count 1 \
  --king_gpu 0 --king_gpu_count 1 \
  --data_dir ./prebuilt-data \
  --use_teacher_cache \
  --output_dir ./distil-checkpoints \
  --eval_every_steps 200 \
  --eval_prompts 500 \
  --eval_seed 12345 \
  --king_repo <KING_REPO> \
  --beat_king_p_threshold 0.05 \
  --beat_king_min_delta 0.0 \
  --early_stop_on_beat_consecutive 3
```

This emits JSON eval lines, appends them to `king_eval_metrics.jsonl`, saves best king-beating checkpoint to `best_beat_king`, and can early-stop after consecutive king-beating evals.

### 3c) Resume training

Resume from latest checkpoint in output directory:

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --data_dir ./prebuilt-data \
  --output_dir ./distil-checkpoints \
  --use_teacher_cache \
  --resume_latest \
  --eval_every_steps 200 \
  --eval_prompts 500 \
  --eval_seed 12345 \
  --king_repo <KING_REPO>
```

Resume from a specific checkpoint:

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --data_dir ./prebuilt-data \
  --output_dir ./distil-checkpoints \
  --use_teacher_cache \
  --resume_from ./distil-checkpoints/step_1400 \
  --eval_every_steps 200 \
  --eval_prompts 500 \
  --eval_seed 12345 \
  --king_repo <KING_REPO>
```

Without cache (online teacher path): the teacher stays loaded on `--teacher_gpu` for the whole run. Use a **second GPU** for the student (`--student_gpu`) or expect high VRAM on a single GPU.

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher_gpu 0 --teacher_gpu_count 1 \
  --student_gpu 1 --student_gpu_count 1 \
  --data_dir ./prebuilt-data \
  --output_dir ./distil-checkpoints
```

### 3d) Train without prebuilt data (streaming fallback)

If `<data_dir>/manifest.json` is missing, `train` automatically falls back to streaming raw text from dataset and tokenizing on the fly.

Notes:

- In this mode, `--use_teacher_cache` is not allowed.
- You must set `--max_steps > 0` (there is no finite prebuilt sample count).
- You can control dataset filtering with `--dataset` and `--min_chars`.
- **`--dataset_split`**: Hugging Face `datasets` split name (default `train`). Use e.g. `validation` if your dataset exposes it.
- **`--dataset_skip_rows`**: skip the first *N* raw rows of that streaming split before training. Uses `IterableDataset.skip` when available (fast). **Ignored when you pass `--resume_from` or `--resume_latest`**; in that case the stream position comes from `train_state.json` (`data_state` / `consumed` for streaming).
- Row count matches the training loop: every `next()` on the stream increments position, including rows dropped by `--min_chars`.

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --teacher_gpu_count 2 \
  --student_gpu 2 --student_gpu_count 2 \
  --dataset karpathy/climbmix-400b-shuffle \
  --dataset_split train \
  --dataset_skip_rows 0 \
  --min_chars 2560 \
  --max_seq_len 1024 \
  --max_steps 2000 \
  --output_dir ./distil-checkpoints
```

Example: start deep in the split (e.g. skip 500k raw rows) without resuming. Use a `--data_dir` that has **no** `manifest.json` so streaming fallback activates:

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --teacher_gpu 0 --student_gpu 1 \
  --data_dir ./streaming-data-no-manifest \
  --dataset karpathy/climbmix-400b-shuffle \
  --dataset_split train \
  --dataset_skip_rows 500000 \
  --max_steps 2000 \
  --output_dir ./distil-checkpoints
```

### 3e) 8-GPU split example (teacher/student/king)

Allocate teacher=3 GPUs, student=3 GPUs, king=2 GPUs:

```bash
python3 examples/distil_kl_train_prebuilt.py train \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --student Qwen/Qwen3.5-4B \
  --king_repo <KING_REPO> \
  --teacher_gpu 0 --teacher_gpu_count 3 \
  --student_gpu 3 --student_gpu_count 3 \
  --king_gpu 6 --king_gpu_count 2 \
  --data_dir ./prebuilt-data \
  --eval_every_steps 200 \
  --eval_prompts 300 \
  --max_steps 2000
```

#### GPU planning cheatsheet (8× GPU, disjoint spans)

Each model uses a **contiguous** block: `--*_gpu` is the first index, `--*_gpu_count` is the length. The three blocks must not overlap (teacher + student + king = 8 in the examples below).

| Scenario | `--teacher_gpu` + count | `--student_gpu` + count | `--king_gpu` + count | When to use |
|----------|-------------------------|-------------------------|----------------------|---------------|
| Default (balanced) | `0` + `3` | `3` + `3` | `6` + `2` | Online teacher + student + periodic king eval; good default. |
| Heavy teacher | `0` + `4` | `4` + `3` | `7` + `1` | Teacher barely fits; king is smaller or eval is light (`--eval_prompts` low). |
| Heavy student (online KL) | `0` + `2` | `2` + `4` | `6` + `2` | Student backward needs more VRAM than teacher forward; long `max_seq_len` or high `samples_per_step`. |
| Smaller king / faster eval | `0` + `3` | `3` + `4` | `7` + `1` | Give student an extra GPU; king still fits on one card. |
| No king eval | `0` + `4` | `4` + `4` | (omit) | Set `--eval_every_steps 0`: king model is not loaded; split all 8 between teacher and student only. |

If eval OOMs but training is fine, reduce `--eval_prompts` / `--eval_max_new_tokens` first, or give king one more GPU and shrink student count by one.

### 4) Evaluate trained checkpoint with validator-like local logic

Use `examples/eval_like_validator.py` to score your trained model with local logic aligned to
`check_model.py --eval` (continuation-only KL with teacher-generated continuations).

```bash
python3 examples/eval_like_validator.py \
  --student ./distil-checkpoints/step_1000 \
  --teacher Qwen/Qwen3.5-35B-A3B \
  --prompts 20
```

Optional: reuse teacher eval cache for faster repeated comparisons.

```bash
python3 examples/eval_like_validator.py \
  --student ./distil-checkpoints/step_1000 \
  --teacher-cache ./teacher_eval_cache.pt \
  --prompts 20
```

## Important compatibility rules

The script validates:

- `max_seq_len` matches built data manifest.
- `kl_start_pos` matches built data manifest.
- teacher in cache manifest matches `--teacher`.
- chunk filenames match between tokens and teacher cache.
- token chunk fingerprint matches cache manifest fingerprint.

If any of these fail, training stops with an explicit error.

## Checkpoints and origin `config.json`

After each `save_pretrained`, the script copies cached **origin** metadata into `step_<N>/` (and `best_beat_king/` when applicable) so checkpoints stay aligned with the **published** model on Hugging Face, not only the slimmer config `transformers` may serialize from memory.

- **Local `--student` path:** matching `*.json` and listed tokenizer/config files are copied from that directory into `<output_dir>/_origin_model_configs/` at run start, then merged into each checkpoint.
- **Hub `--student` id (`org/model`):** the same filenames are **downloaded** from the Hub into `_origin_model_configs/` (requires network once per run). Use **`--student_revision`** (branch / tag / commit) to pin an exact revision.

If Hub download fails for `config.json`, training still runs but checkpoints may miss Hub-only fields until the download succeeds.

## LR schedule

The cosine LR schedule’s **total step count** matches this run (`--max_steps` or one full pass over prebuilt data), not a fixed large constant. Warmup uses `--warmup_steps` as in Hugging Face `get_cosine_schedule_with_warmup`.

## Main arguments

### `build`

- `--teacher` (default: `Qwen/Qwen3.5-35B-A3B`)
- `--dataset` (default: `karpathy/climbmix-400b-shuffle`)
- `--data_dir`
- `--num_samples`
- `--chunk_size`
- `--max_seq_len`
- `--kl_start_pos`
- `--min_chars`

### `build_teacher_cache`

- `--teacher`
- `--teacher_gpu`
- `--teacher_gpu_count`
- `--data_dir`
- `--cache_dir` (optional, default: `<data_dir>/teacher_cache`)
- `--kl_start_pos`
- `--teacher_cache_dtype` (`bf16`, `fp16`, `fp32`)

### `train`

- `--teacher`
- `--student`
- `--student_revision` (optional Hub revision for origin config download when `--student` is a repo id)
- `--teacher_gpu`
- `--teacher_gpu_count`
- `--student_gpu`
- `--student_gpu_count`
- `--king_gpu`, `--king_gpu_count`
- `--data_dir`
- `--dataset`, `--dataset_split`, `--dataset_skip_rows`, `--min_chars` (streaming fallback mode)
- `--cache_dir`
- `--use_teacher_cache`
- `--shuffle_chunks`
- `--data_seed`
- `--lr`, `--warmup_steps`, `--weight_decay`
- `--samples_per_step`, `--max_steps`
- `--save_every`, `--output_dir`
- `--resume_from`, `--resume_latest`
- `--eval_every_steps`, `--eval_prompts`, `--eval_dataset`, `--eval_block_number`, `--eval_max_new_tokens`, `--eval_seed`
- `--king_repo`, `--king_revision`
- `--beat_king_p_threshold`, `--beat_king_min_delta`, `--early_stop_on_beat_consecutive`
- `--metrics_jsonl`
- `--wandb_project`, `--wandb_run`, `--wandb_init_timeout` (seconds for `wandb.init`; default 300), `--no_wandb`

## Artifacts

Saved checkpoints (every `save_every` steps):

- `<output_dir>/step_<N>/` with:
  - model weights
  - tokenizer files
  - origin config / tokenizer metadata merged after save (local `--student` dir or Hub download into `_origin_model_configs/`), including full **`config.json`** when available
  - `optimizer.pt`
  - `scheduler.pt`
  - `train_state.json` (written atomically; if missing or corrupt, resume can recover `global_step` from the `step_<N>` directory name)

And one run config:

- `<output_dir>/train_config.json`
- `<output_dir>/_origin_model_configs/` (cached origin files for Hub or local `--student`; internal use, safe to delete between runs if you accept re-download)
- `<output_dir>/king_eval_metrics.jsonl` (unless overridden by `--metrics_jsonl`)
- `<output_dir>/best_beat_king/` (best checkpoint that satisfies beat-king thresholds)

## Troubleshooting

- **`Teacher mismatch`**: rebuild cache with the same `--teacher`.
- **`kl_start_pos mismatch`**: keep `build`, `build_teacher_cache`, and `train` on same value.
- **`Token chunk fingerprint mismatch`**: token chunks changed after cache build; rebuild teacher cache.
- **Out of memory / process killed**: see [Memory and OOM](#memory-and-oom). Quick checks: `train --use_teacher_cache`, lower `--samples_per_step` / `--max_seq_len`, smaller `--chunk_size` during cache build, lighter `--teacher_cache_dtype`.
- **`Missing manifest` / no prebuilt data**: this now triggers streaming fallback automatically. Set `--max_steps > 0`, and do not use `--use_teacher_cache` in that mode.
- **Resume gives different sample order**: ensure checkpoint contains `train_state.json` with `data_state` (new checkpoints include this). Without it, resume can replay samples from the beginning. For **streaming** fallback, `data_state` includes `consumed` and `dataset_split`; restore uses `IterableDataset.skip` when possible instead of replaying millions of `next()` calls.
- **`train_state.json` empty / corrupt**: resume can rebuild minimal state from the `step_<N>` folder name; **`--dataset_skip_rows` is ignored** whenever `--resume_from` / `--resume_latest` is set.
- **`wandb` init timeout**: increase `--wandb_init_timeout` or use `--no_wandb` if `api.wandb.ai` is slow from your network.
- **No king eval output**: set `--eval_every_steps > 0` and provide `--king_repo`.
- **King comparison looks noisy between runs**: fix `--eval_seed` and keep `--eval_prompts`/`--eval_dataset`/`--eval_block_number` unchanged to compare runs fairly.
- **No early stop**: set `--early_stop_on_beat_consecutive > 0` and make sure thresholds (`--beat_king_p_threshold`, `--beat_king_min_delta`) are achievable.

## Suggested workflow for repeated experiments

1. Run `build` once.
2. Run `build_teacher_cache` once per teacher or dataset rebuild.
3. Run many `train --use_teacher_cache` experiments with different hyperparameters.
4. Run `eval_like_validator.py` on candidate checkpoints and keep the best KL.

