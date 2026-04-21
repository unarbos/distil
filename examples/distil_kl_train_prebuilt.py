#!/usr/bin/env python3
"""
KL Distillation with Prebuilt Training Data for Bittensor Subnet 97.

This script keeps the same core training logic as examples/distil_kl_train.py,
but supports an offline pipeline:

1) build: stream dataset once and prebuild tokenized samples to disk
2) build_teacher_cache: run teacher once on prebuilt tokens and save targets
3) train: load prebuilt samples (and optional teacher cache) for KL distillation

Examples:
    # Build 200k tokenized samples once
    python examples/distil_kl_train_prebuilt.py build \
      --teacher Qwen/Qwen3.5-35B-A3B \
      --data_dir ./prebuilt-data \
      --num_samples 200000

    # Teacher cache (sharded); add --resume_teacher_cache to continue after interruption
    python distil_kl_train_prebuilt.py build_teacher_cache \
      --teacher Qwen/Qwen3.5-35B-A3B --teacher_gpu 0 --data_dir ./prebuilt-data \
      --resume_teacher_cache

    # Train repeatedly without streaming dataset every run
    python examples/distil_kl_train_prebuilt.py train \
      --teacher_gpu 0 --student_gpu 1 \
      --data_dir ./prebuilt-data \
      --output_dir ./distil-checkpoints
"""

import argparse
import gc
import hashlib
import shutil
import json
import logging
import os
import random
import time
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Keep defaults aligned with distil_kl_train.py
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
STUDENT_MODEL = "Qwen/Qwen3.5-4B"
DATASET = "karpathy/climbmix-400b-shuffle"
LR = 1e-4
WARMUP = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 640
KL_START_POS = 128
SAMPLES_PER_STEP = 100
SAVE_EVERY = 500
MIN_CHARS = 2560


def _require_positive(name: str, value: int):
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def _validate_common_seq_args(max_seq_len: int, kl_start_pos: int):
    _require_positive("max_seq_len", max_seq_len)
    if kl_start_pos < 0:
        raise ValueError(f"kl_start_pos must be >= 0 (got {kl_start_pos})")
    # Keep same training safety margin used later in filtering.
    if max_seq_len <= kl_start_pos + 10:
        raise ValueError(
            f"max_seq_len must be > kl_start_pos + 10 "
            f"(got max_seq_len={max_seq_len}, kl_start_pos={kl_start_pos})"
        )


def _gpu_span(start_gpu: int, gpu_count: int, label: str) -> list[int]:
    if gpu_count <= 0:
        raise ValueError(f"{label}_gpu_count must be > 0 (got {gpu_count})")
    n_cuda = torch.cuda.device_count()
    if n_cuda <= 0:
        raise RuntimeError("No CUDA devices available.")
    if start_gpu < 0 or start_gpu >= n_cuda:
        raise ValueError(
            f"{label}_gpu start index out of range: {start_gpu} (available: 0..{n_cuda-1})"
        )
    end_gpu = start_gpu + gpu_count - 1
    if end_gpu >= n_cuda:
        raise ValueError(
            f"{label}_gpu span exceeds available GPUs: start={start_gpu}, count={gpu_count}, "
            f"needs up to {end_gpu}, available max index={n_cuda-1}"
        )
    return list(range(start_gpu, start_gpu + gpu_count))


def _device_map_and_memory(gpus: list[int]):
    if len(gpus) == 1:
        return {"": gpus[0]}, None
    # Constrain auto-sharding to the selected GPU subset.
    max_memory = {}
    for idx in gpus:
        total_bytes = torch.cuda.get_device_properties(idx).total_memory
        gib = max(1, int((total_bytes * 0.90) / (1024**3)))
        max_memory[idx] = f"{gib}GiB"
    max_memory["cpu"] = "256GiB"
    return "auto", max_memory


def _first_param_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def kl_loss(student_logits, teacher_logits, start_pos=KL_START_POS):
    """Forward KL(teacher || student) from start_pos onward."""
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(-1).mean()


def kl_loss_masked(
    student_logits,
    teacher_logits,
    start_pos: int,
    tail_mask: torch.Tensor,
):
    """
    Forward KL(teacher || student) from start_pos onward.

    Averages the same way as the per-sample loop: mean over valid positions within each
    sequence, then mean across batch. tail_mask: [B, L - start_pos], 1 on real tokens.
    """
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    per_pos = (t_p * (t_log_p - s_log_p)).sum(-1)
    m = tail_mask.to(device=per_pos.device, dtype=per_pos.dtype)
    per_seq = (per_pos * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    return per_seq.mean()


def _pad_token_batch(
    tokens: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad 1D token tensors to [B, Lmax] on CPU; return (input_ids, position_mask)."""
    lengths = [int(x.shape[0]) for x in tokens]
    lmax = max(lengths)
    b = len(tokens)
    out = torch.full((b, lmax), pad_id, dtype=torch.long)
    mask = torch.zeros((b, lmax), dtype=torch.float32)
    for i, x in enumerate(tokens):
        li = lengths[i]
        out[i, :li] = x
        mask[i, :li] = 1.0
    return out, mask


def _chunk_list(items: list, chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize empty metric list")
    vals = sorted(float(v) for v in values)
    n = len(vals)
    p50 = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {
        "mean": float(sum(vals) / n),
        "std": float(statistics.stdev(vals) if n > 1 else 0.0),
        "p50": float(p50),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "n": int(n),
    }


def _paired_t_stats(deltas: list[float]) -> tuple[float, float]:
    n = len(deltas)
    if n < 2:
        return 0.0, 1.0
    mean_delta = sum(deltas) / n
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    if se <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0
        return 0.0, 1.0
    t_stat = mean_delta / se
    cdf = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    p_one_sided = max(0.0, min(1.0, 1.0 - cdf))
    return float(t_stat), float(p_one_sided)


def _update_ema(prev: float | None, value: float, beta: float) -> float:
    if prev is None:
        return float(value)
    return float(beta * prev + (1.0 - beta) * value)


def _gate_score(delta_ema: float, p_value: float) -> float:
    """
    Composite ranking score for stable-vs-king checkpoints.

    Higher is better: reward stronger positive margin, penalize weaker significance.
    """
    return float(delta_ema - 0.5 * p_value)


def _json_lists_to_tuples(value):
    if isinstance(value, list):
        return tuple(_json_lists_to_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _json_lists_to_tuples(v) for k, v in value.items()}
    return value


def _prepare_eval_cache(
    teacher_eval,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    teacher_eval.eval()
    teacher_device = next(teacher_eval.parameters()).device
    cache = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(teacher_device)
            prompt_len = prompt_ids.shape[1]
            prompt_seed = int(seed) + i
            try:
                gen = torch.Generator(device=teacher_device)
                gen.manual_seed(prompt_seed)
                full_seq = teacher_eval.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    generator=gen,
                )
            except ValueError as e:
                # Some model wrappers/custom generation stacks reject `generator`.
                # Fallback keeps deterministic sampling by seeding within a forked RNG scope.
                if "not used by the model: ['generator']" not in str(e):
                    raise
                device_idx = teacher_device.index if teacher_device.type == "cuda" else None
                devices = [device_idx] if device_idx is not None else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(prompt_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(prompt_seed)
                    full_seq = teacher_eval.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                    )
            t_logits = teacher_eval(full_seq).logits.float()
            t_cont = t_logits[:, prompt_len - 1 : -1, :]
            t_log_p = F.log_softmax(t_cont, dim=-1).cpu()
            cache.append(
                {
                    "full_seq": full_seq.cpu(),
                    "prompt_len": int(prompt_len),
                    "teacher_log_probs": t_log_p,
                }
            )
            del prompt_ids, full_seq, t_logits, t_cont, t_log_p
    return cache


def _mean_token_kl_forward_chunked(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    vocab_chunk: int = 4096,
) -> float:
    """
    Forward KL(teacher || student) averaged over token positions.

    Same math as ``kl_loss`` / ``(t_p * (t_log_p - s_log_p)).sum(-1).mean()`` but
    sums over vocabulary in chunks so we never materialize full ``[B, T, V]`` tensors
    (``F.kl_div`` on large V still peaks VRAM for huge vocabs).
    """
    b, t, v = student_log_probs.shape
    if t <= 0 or v <= 0:
        return 0.0
    device = student_log_probs.device
    # [B, T] running sum of per-token KL contributions
    acc = torch.zeros((b, t), device=device, dtype=torch.float32)
    for v0 in range(0, v, vocab_chunk):
        v1 = min(v0 + vocab_chunk, v)
        t_lp = teacher_log_probs[:, :, v0:v1].float()
        s_lp = student_log_probs[:, :, v0:v1].float()
        t_p = t_lp.exp()
        acc += (t_p * (t_lp - s_lp)).sum(dim=-1)
    return float(acc.mean().item())


def _evaluate_against_king(
    student,
    king_eval,
    eval_cache: list[dict],
) -> tuple[list[float], list[float], dict]:

    king_eval.eval()
    student.eval()
    king_device = next(king_eval.parameters()).device
    student_device = next(student.parameters()).device

    eval_scores = []
    king_scores = []
    with torch.no_grad():
        for item in eval_cache:
            full_seq = item["full_seq"]
            prompt_len = int(item["prompt_len"])
            t_log_p = item["teacher_log_probs"].to(student_device)

            s_logits = student(full_seq.to(student_device)).logits.float()
            s_cont = s_logits[:, prompt_len - 1 : -1, :]
            k_logits = king_eval(full_seq.to(king_device)).logits.float()
            k_cont = k_logits[:, prompt_len - 1 : -1, :]

            min_len_s = min(s_cont.shape[1], t_log_p.shape[1])
            min_len_k = min(k_cont.shape[1], t_log_p.shape[1])
            s_lp = F.log_softmax(s_cont[:, :min_len_s, :], dim=-1)
            k_lp = F.log_softmax(k_cont[:, :min_len_k, :], dim=-1)
            # Contiguous copies so we can drop the full teacher log-prob tensor and free VRAM.
            t_lp_s = t_log_p[:, :min_len_s, :].contiguous()
            t_lp_k = t_log_p[:, :min_len_k, :].to(device=king_device, dtype=torch.float32).contiguous()
            del t_log_p

            s_kl = _mean_token_kl_forward_chunked(s_lp, t_lp_s)
            del s_logits, s_cont, s_lp, t_lp_s
            if student_device.type == "cuda":
                torch.cuda.empty_cache()
            k_kl = _mean_token_kl_forward_chunked(k_lp, t_lp_k)
            eval_scores.append(float(s_kl))
            king_scores.append(float(k_kl))

            del k_logits, k_cont, k_lp, t_lp_k

    student.train()
    eval_stats = _summary_stats(eval_scores)
    king_stats = _summary_stats(king_scores)
    deltas = [k - e for k, e in zip(king_scores, eval_scores)]
    t_stat, p_val = _paired_t_stats(deltas)
    return eval_scores, king_scores, {
        "p": p_val,
        "t": t_stat,
        "delta": float(sum(deltas) / len(deltas)),
    }


def _looks_like_hf_repo_id(ref: str) -> bool:
    ref = ref.strip()
    if "/" not in ref or ref.startswith(("/", ".", "~")):
        return False
    left, _, right = ref.partition("/")
    return bool(left) and bool(right) and "\\" not in ref


# Files to preserve from the origin checkpoint (Hub or local). `save_pretrained` may rewrite a
# slimmer config.json; we copy these into each step_* after save so validators see repo-identical JSON.
_ORIGIN_ARTIFACT_NAMES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "chat_template.jinja",
    "vocab.json",
    "added_tokens.json",
    "merges.txt",
    "tokenizer.model",
)


def _ensure_origin_model_configs_cached(
    origin_model_ref: str,
    revision: str | None,
    cache_dir: Path,
) -> list[Path]:
    """
    Cache origin model config/tokenizer metadata for later copy into checkpoints.

    - Local directory: copy matching files from disk (byte-identical).
    - Hugging Face repo id (``org/name``): download the same filenames from the Hub so
      ``config.json`` matches the published model, not only the in-memory config from
      ``save_pretrained``.
    """
    src = Path(origin_model_ref)
    if src.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        names = set(_ORIGIN_ARTIFACT_NAMES)
        for p in src.iterdir():
            if not p.is_file():
                continue
            if p.name in names or p.name.endswith(".json"):
                dst = cache_dir / p.name
                shutil.copy2(p, dst)
                copied.append(dst)
        return copied

    if not _looks_like_hf_repo_id(origin_model_ref):
        return []

    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    repo_id = origin_model_ref.strip()
    rev_kw = {} if revision in (None, "") else {"revision": str(revision)}
    for filename in _ORIGIN_ARTIFACT_NAMES:
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
                **rev_kw,
            )
            copied.append(Path(out))
        except Exception as e:
            if filename == "config.json":
                log.warning(
                    "Could not download origin config.json from Hub repo %s%s: %s",
                    repo_id,
                    f"@{revision}" if rev_kw else "",
                    e,
                )
            continue
    if not (cache_dir / "config.json").is_file():
        log.warning(
            "Origin Hub cache for %s has no config.json; checkpoints will only contain "
            "save_pretrained config (may omit Hub-only fields).",
            repo_id,
        )
    else:
        log.info(
            "Cached %s origin Hub artifact(s) from %s%s into %s",
            len(copied),
            repo_id,
            f" (revision={revision})" if rev_kw else "",
            cache_dir,
        )
    return copied


def _copy_cached_origin_configs(cache_dir: Path, target_dir: Path):
    if not cache_dir.exists():
        return
    for p in cache_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, target_dir / p.name)


def kl_loss_from_teacher_log_probs(student_logits, teacher_log_probs, start_pos=KL_START_POS):
    """
    Forward KL(teacher || student) using precomputed teacher log-probabilities.

    teacher_log_probs is expected to cover positions [start_pos:] only.
    """
    s = student_logits[:, start_pos:, :].contiguous()
    t_log_p = teacher_log_probs.to(s.device).contiguous()
    if t_log_p.dim() == 2:
        t_log_p = t_log_p.unsqueeze(0)
    min_len = min(s.shape[1], t_log_p.shape[1])
    if min_len <= 0:
        raise ValueError("No overlapping sequence length for KL loss")
    s_log_p = F.log_softmax(s[:, :min_len, :].float(), dim=-1)
    t_log_p = t_log_p[:, :min_len, :].float()
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(-1).mean()


def _save_manifest(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _read_json_file(path: Path) -> dict:
    """Load a JSON object from disk; raise with a clear message if empty or corrupt."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Cannot read {path}: {e}") from e
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    raw = raw.strip()
    if not raw:
        raise ValueError(
            f"{path} is empty or whitespace-only. The checkpoint may be incomplete "
            "(e.g. process killed while saving). Use an older step_* directory, or omit "
            "--resume_from / --resume_latest."
        )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{path} is not valid JSON ({e}). The file may be corrupted or partially written."
        ) from e
    if not isinstance(out, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(out).__name__}")
    return out


def _global_step_from_checkpoint_dir(checkpoint_dir: Path) -> int | None:
    """Parse training step from a directory named ``step_<N>``."""
    name = checkpoint_dir.name
    if not name.startswith("step_"):
        return None
    tail = name.split("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _read_train_state_or_repair(checkpoint_dir: Path) -> tuple[dict, bool]:
    """
    Load ``train_state.json`` from a checkpoint directory.

    If the file is missing, empty, or invalid JSON, recover a minimal dict when the
    directory name is ``step_<N>`` (so resume still works after a crash mid-save).

    Returns ``(state, repaired)`` where ``repaired`` is True if recovery was used.
    """
    train_state_path = checkpoint_dir / "train_state.json"
    err: Exception | None = None
    if train_state_path.is_file():
        try:
            return _read_json_file(train_state_path), False
        except (ValueError, OSError) as e:
            err = e
    else:
        err = FileNotFoundError(str(train_state_path))

    repaired = _global_step_from_checkpoint_dir(checkpoint_dir)
    if repaired is None:
        raise ValueError(
            f"Cannot load train state from {checkpoint_dir}: {err}. "
            f"Expected a readable {train_state_path.name} or a directory named step_<N>."
        ) from err

    log.warning(
        "train_state.json in %s is missing or unreadable (%s). "
        "Rebuilding minimal resume state from directory name (global_step=%s). "
        "data_state was not restored; the data stream may replay samples. "
        "beat_streak / best_beating_delta reset.",
        checkpoint_dir,
        err,
        repaired,
    )
    minimal = {
        "global_step": repaired,
        "data_state": None,
        "best_beating_delta": float("-inf"),
        "beat_streak": 0,
    }
    try:
        _atomic_write_json(train_state_path, minimal)
    except OSError as werr:
        log.warning("Could not write repaired train_state.json (%s).", werr)
    return minimal, True


def _atomic_write_json(path: Path, data: dict):
    """Write JSON atomically (tmp + replace) so crashes never leave an empty train_state.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_chunk_files(paths: list[Path], cache_file: Path | None = None) -> str:
    """
    Build a deterministic fingerprint across chunk files.

    Includes filename + exact file hash for each chunk, then hashes the manifest
    string to produce a single fingerprint.
    """
    if not paths:
        raise ValueError("Cannot fingerprint empty chunk list")

    cached_meta = {}
    if cache_file and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            if isinstance(payload, dict):
                cached_meta = payload.get("files", {}) or {}
        except Exception:
            cached_meta = {}

    files_meta = {}
    entries = []
    for path in sorted(paths, key=lambda p: p.name):
        st = path.stat()
        size = st.st_size
        mtime_ns = st.st_mtime_ns
        c = cached_meta.get(path.name, {})
        if (
            isinstance(c, dict)
            and c.get("size") == size
            and c.get("mtime_ns") == mtime_ns
            and isinstance(c.get("sha256"), str)
        ):
            file_hash = c["sha256"]
        else:
            file_hash = _sha256_file(path)

        files_meta[path.name] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "sha256": file_hash,
        }
        entries.append(f"{path.name}:{file_hash}")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"files": files_meta}, indent=2))

    joined = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _teacher_cache_shard_dir(cache_dir: Path, token_chunk_path: Path) -> Path:
    """On-disk directory for per-sample teacher shards (stem matches token chunk)."""
    return cache_dir / token_chunk_path.stem


def _sorted_teacher_shard_paths(shard_dir: Path) -> list[Path]:
    """Return shard paths 000000.pt, 000001.pt, ... (lexicographic order)."""
    paths = [p for p in shard_dir.glob("*.pt") if p.stem.isdigit() and len(p.stem) == 6]
    return sorted(paths, key=lambda p: p.name)


def resolve_teacher_cache_chunk(cache_dir: Path, token_chunk_path: Path) -> tuple[str, Path]:
    """
    Locate teacher cache for one token chunk file.

    Returns ("sharded", dir) if ``cache_dir / chunk_stem /`` holds numeric shards, else
    ("legacy", path) for a single ``chunk_XXXXXX.pt`` file.
    """
    shard_dir = _teacher_cache_shard_dir(cache_dir, token_chunk_path)
    legacy_pt = cache_dir / token_chunk_path.name
    if shard_dir.is_dir():
        shards = _sorted_teacher_shard_paths(shard_dir)
        if shards:
            return "sharded", shard_dir
    if legacy_pt.is_file():
        return "legacy", legacy_pt
    raise FileNotFoundError(
        f"No teacher cache for {token_chunk_path.name}: "
        f"expected {shard_dir}/{{000000..}}.pt or {legacy_pt}"
    )


def _token_chunk_num_samples(token_chunk_path: Path) -> int:
    payload = torch.load(token_chunk_path, map_location="cpu", weights_only=False)
    ids = payload.get("input_ids", [])
    if not ids:
        raise RuntimeError(f"Empty token chunk: {token_chunk_path}")
    return len(ids)


def _sharded_first_missing_sample(shard_dir: Path, n: int) -> int | None:
    """
    For a sharded teacher cache directory, return the first sample index whose shard
    is missing. Return None if shards ``000000`` … ``{n-1:06d}`` all exist (chunk done).
    If ``shard_dir`` does not exist, return 0 (write from the start).
    """
    if not shard_dir.is_dir():
        return 0
    for j in range(n):
        if not (shard_dir / f"{j:06d}.pt").is_file():
            return j
    return None


def build_dataset(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    _require_positive("num_samples", args.num_samples)
    _require_positive("chunk_size", args.chunk_size)
    _validate_common_seq_args(args.max_seq_len, args.kl_start_pos)
    if args.min_chars < 0:
        raise ValueError(f"min_chars must be >= 0 (got {args.min_chars})")

    data_dir = Path(args.data_dir)
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading streaming dataset...")
    ds_iter = iter(load_dataset(args.dataset, split="train", streaming=True))

    total_saved = 0
    total_scanned = 0
    chunk_idx = 0
    chunk_samples = []
    start_time = time.time()

    while total_saved < args.num_samples:
        try:
            item = next(ds_iter)
        except StopIteration:
            log.warning("Dataset exhausted early.")
            break

        total_scanned += 1
        text = item.get("text", "")
        if not text or len(text) < args.min_chars:
            continue

        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).input_ids.squeeze(0).to(torch.int32)

        if ids.shape[0] <= args.kl_start_pos + 10:
            continue

        chunk_samples.append(ids)
        total_saved += 1

        if len(chunk_samples) >= args.chunk_size:
            chunk_path = chunks_dir / f"chunk_{chunk_idx:06d}.pt"
            torch.save({"input_ids": chunk_samples}, chunk_path)
            chunk_idx += 1
            chunk_samples = []
            elapsed = time.time() - start_time
            rate = total_saved / max(elapsed, 1e-6)
            log.info(
                f"Saved chunk {chunk_idx:06d} | samples={total_saved}/{args.num_samples} "
                f"| scanned={total_scanned} | {rate:.1f} samp/s"
            )

    if chunk_samples:
        chunk_path = chunks_dir / f"chunk_{chunk_idx:06d}.pt"
        torch.save({"input_ids": chunk_samples}, chunk_path)
        chunk_idx += 1

    elapsed = time.time() - start_time
    manifest = {
        "teacher": args.teacher,
        "dataset": args.dataset,
        "max_seq_len": args.max_seq_len,
        "kl_start_pos": args.kl_start_pos,
        "min_chars": args.min_chars,
        "num_samples": total_saved,
        "num_chunks": chunk_idx,
        "build_scanned": total_scanned,
        "build_elapsed_sec": round(elapsed, 2),
    }
    _save_manifest(manifest_path, manifest)
    log.info(f"Build complete: {total_saved} samples, {chunk_idx} chunks, {elapsed/60:.1f} min")
    log.info(f"Manifest: {manifest_path}")


class PrebuiltTokenStream:
    """Iterates prebuilt token samples without loading all chunks into memory."""

    def __init__(self, data_dir: str, seed: int = 42, shuffle_chunks: bool = False):
        self.data_dir = Path(data_dir)
        self.chunks_dir = self.data_dir / "chunks"
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Missing chunks dir: {self.chunks_dir}")

        self.manifest = json.loads(self.manifest_path.read_text())
        self.chunk_paths = sorted(self.chunks_dir.glob("chunk_*.pt"))
        if not self.chunk_paths:
            raise RuntimeError(f"No chunk_*.pt files in {self.chunks_dir}")

        self.rng = random.Random(seed)
        self.shuffle_chunks = shuffle_chunks
        self._chunk_order = list(range(len(self.chunk_paths)))
        if self.shuffle_chunks:
            self.rng.shuffle(self._chunk_order)

        self._chunk_ptr = 0
        self._sample_ptr = 0
        self._current_samples = []
        self._consumed = 0
        self._load_current_chunk()

    def _load_current_chunk(self):
        if not self._chunk_order:
            raise RuntimeError("No chunk order available")
        chunk_idx = self._chunk_order[self._chunk_ptr]
        chunk_path = self.chunk_paths[chunk_idx]
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        self._current_samples = payload.get("input_ids", [])
        if not self._current_samples:
            raise RuntimeError(f"Empty chunk: {chunk_path}")
        self._sample_ptr = 0

    def _advance_chunk(self):
        self._chunk_ptr += 1
        if self._chunk_ptr >= len(self._chunk_order):
            self._chunk_ptr = 0
            if self.shuffle_chunks:
                self.rng.shuffle(self._chunk_order)
        self._load_current_chunk()

    def get_batch(self, n: int):
        out = []
        while len(out) < n:
            if self._sample_ptr >= len(self._current_samples):
                self._advance_chunk()
            out.append(self._current_samples[self._sample_ptr])
            self._sample_ptr += 1
            self._consumed += 1
        return out

    @property
    def position(self):
        return self._consumed

    def state_dict(self) -> dict:
        return {
            "chunk_order": list(self._chunk_order),
            "chunk_ptr": int(self._chunk_ptr),
            "sample_ptr": int(self._sample_ptr),
            "consumed": int(self._consumed),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict):
        chunk_order = state.get("chunk_order")
        if not isinstance(chunk_order, list) or len(chunk_order) != len(self.chunk_paths):
            raise ValueError("Invalid token stream state: chunk_order mismatch")
        self._chunk_order = [int(x) for x in chunk_order]
        self._chunk_ptr = int(state.get("chunk_ptr", 0))
        self._consumed = int(state.get("consumed", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(_json_lists_to_tuples(rng_state))
        self._load_current_chunk()
        self._sample_ptr = int(state.get("sample_ptr", 0))
        if not (0 <= self._sample_ptr <= len(self._current_samples)):
            raise ValueError("Invalid token stream state: sample_ptr out of range")


class PrebuiltTeacherTargetStream:
    """Iterates token IDs + cached teacher targets in lockstep."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: str | None = None,
        seed: int = 42,
        shuffle_chunks: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.chunks_dir = self.data_dir / "chunks"
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "teacher_cache"
        self.manifest_path = self.data_dir / "manifest.json"
        self.cache_manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.cache_manifest_path.exists():
            raise FileNotFoundError(f"Missing teacher cache manifest: {self.cache_manifest_path}")

        self.manifest = json.loads(self.manifest_path.read_text())
        self.cache_manifest = json.loads(self.cache_manifest_path.read_text())
        self.token_chunk_paths = sorted(self.chunks_dir.glob("chunk_*.pt"))
        if not self.token_chunk_paths:
            raise RuntimeError(f"No token chunks in {self.chunks_dir}")

        self._cache_chunk_specs: list[tuple[str, Path]] = []
        for tp in self.token_chunk_paths:
            try:
                self._cache_chunk_specs.append(resolve_teacher_cache_chunk(self.cache_dir, tp))
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Teacher cache incomplete relative to token chunks: {e}"
                ) from e

        self.rng = random.Random(seed)
        self.shuffle_chunks = shuffle_chunks
        self._chunk_order = list(range(len(self.token_chunk_paths)))
        if self.shuffle_chunks:
            self.rng.shuffle(self._chunk_order)

        self._chunk_ptr = 0
        self._sample_ptr = 0
        self._current_ids = []
        self._current_teacher_log_probs = []
        self._cache_mode = "legacy"
        self._shard_paths: list[Path] | None = None
        self._consumed = 0
        self._load_current_chunk()

    def _load_current_chunk(self):
        if not self._chunk_order:
            raise RuntimeError("No chunk order available")
        chunk_idx = self._chunk_order[self._chunk_ptr]
        token_payload = torch.load(self.token_chunk_paths[chunk_idx], map_location="cpu", weights_only=False)
        self._current_ids = token_payload.get("input_ids", [])
        if not self._current_ids:
            raise RuntimeError(f"Empty token chunk index {chunk_idx}")

        mode, cache_loc = self._cache_chunk_specs[chunk_idx]
        self._cache_mode = mode
        if mode == "legacy":
            cache_payload = torch.load(cache_loc, map_location="cpu", weights_only=False)
            self._current_teacher_log_probs = cache_payload.get("teacher_log_probs", [])
            self._shard_paths = None
            if not self._current_teacher_log_probs:
                raise RuntimeError(f"Empty teacher cache chunk index {chunk_idx}")
            if len(self._current_ids) != len(self._current_teacher_log_probs):
                raise RuntimeError(
                    f"Sample count mismatch in chunk {chunk_idx}: ids={len(self._current_ids)} "
                    f"teacher={len(self._current_teacher_log_probs)}"
                )
        else:
            self._shard_paths = _sorted_teacher_shard_paths(cache_loc)
            self._current_teacher_log_probs = []
            if len(self._shard_paths) != len(self._current_ids):
                raise RuntimeError(
                    f"Shard/sample count mismatch in chunk {chunk_idx}: "
                    f"ids={len(self._current_ids)} shards={len(self._shard_paths)}"
                )
        self._sample_ptr = 0

    def _advance_chunk(self):
        self._chunk_ptr += 1
        if self._chunk_ptr >= len(self._chunk_order):
            self._chunk_ptr = 0
            if self.shuffle_chunks:
                self.rng.shuffle(self._chunk_order)
        self._load_current_chunk()

    def get_batch(self, n: int):
        out = []
        while len(out) < n:
            if self._sample_ptr >= len(self._current_ids):
                self._advance_chunk()
            ids = self._current_ids[self._sample_ptr]
            if self._cache_mode == "legacy":
                t_lp = self._current_teacher_log_probs[self._sample_ptr]
            else:
                assert self._shard_paths is not None
                shard = torch.load(
                    self._shard_paths[self._sample_ptr],
                    map_location="cpu",
                    weights_only=False,
                )
                t_lp = shard["teacher_log_probs"]
            out.append((ids, t_lp))
            self._sample_ptr += 1
            self._consumed += 1
        return out

    @property
    def position(self):
        return self._consumed

    def state_dict(self) -> dict:
        return {
            "chunk_order": list(self._chunk_order),
            "chunk_ptr": int(self._chunk_ptr),
            "sample_ptr": int(self._sample_ptr),
            "consumed": int(self._consumed),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict):
        chunk_order = state.get("chunk_order")
        if not isinstance(chunk_order, list) or len(chunk_order) != len(self.token_chunk_paths):
            raise ValueError("Invalid teacher-target stream state: chunk_order mismatch")
        self._chunk_order = [int(x) for x in chunk_order]
        self._chunk_ptr = int(state.get("chunk_ptr", 0))
        self._consumed = int(state.get("consumed", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(_json_lists_to_tuples(rng_state))
        self._load_current_chunk()
        self._sample_ptr = int(state.get("sample_ptr", 0))
        if not (0 <= self._sample_ptr <= len(self._current_ids)):
            raise ValueError("Invalid teacher-target stream state: sample_ptr out of range")


def _open_hf_streaming_iterator(dataset: str, split: str, skip_rows: int):
    """
    Open a streaming HF ``datasets`` iterator, optionally skipping the first ``skip_rows``
    raw examples (fast path uses ``IterableDataset.skip`` when available).
    """
    from datasets import load_dataset

    skip_rows = int(max(0, skip_rows))
    ds = load_dataset(dataset, split=split, streaming=True)
    if skip_rows > 0 and hasattr(ds, "skip"):
        ds = ds.skip(skip_rows)
        return iter(ds)
    if skip_rows > 0:
        log.warning(
            "Streaming dataset has no .skip(); advancing %s rows with next() (slow for large skips).",
            skip_rows,
        )
        it = iter(ds)
        for _ in range(skip_rows):
            next(it)
        return it
    return iter(ds)


class StreamingTokenStream:
    """Streams raw dataset text and tokenizes on the fly (no prebuilt manifest needed)."""

    def __init__(
        self,
        teacher_model: str,
        dataset: str = DATASET,
        max_seq_len: int = MAX_SEQ_LEN,
        min_chars: int = MIN_CHARS,
        dataset_split: str = "train",
        dataset_skip_rows: int = 0,
    ):
        from transformers import AutoTokenizer

        if min_chars < 0:
            raise ValueError(f"min_chars must be >= 0 (got {min_chars})")
        self._teacher_model = teacher_model
        self.dataset = dataset
        self.dataset_split = str(dataset_split)
        self.max_seq_len = int(max_seq_len)
        self.min_chars = int(min_chars)
        skip = int(max(0, dataset_skip_rows))
        self._ds = _open_hf_streaming_iterator(self.dataset, self.dataset_split, skip)
        self._consumed = skip
        self.manifest = {
            "dataset": self.dataset,
            "max_seq_len": self.max_seq_len,
            "kl_start_pos": None,
            "num_samples": 0,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_batch(self, n: int):
        out = []
        scanned = 0
        max_scan = max(n * 20, n + 1)
        while len(out) < n and scanned < max_scan:
            try:
                item = next(self._ds)
            except StopIteration:
                break
            scanned += 1
            self._consumed += 1
            text = item.get("text", "")
            if not text or len(text) < self.min_chars:
                continue
            ids = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).input_ids.squeeze(0).to(torch.int32)
            out.append(ids)
        return out

    @property
    def position(self):
        return self._consumed

    def state_dict(self) -> dict:
        return {
            "variant": "streaming_token_stream",
            "consumed": int(self._consumed),
            "dataset": self.dataset,
            "dataset_split": self.dataset_split,
            "max_seq_len": int(self.max_seq_len),
            "min_chars": int(self.min_chars),
            "teacher_model": self._teacher_model,
        }

    def load_state_dict(self, state: dict):
        if state.get("variant") != "streaming_token_stream":
            raise ValueError(
                "Invalid streaming data_state: expected variant 'streaming_token_stream'"
            )
        if state.get("dataset") != self.dataset:
            raise ValueError(
                f"Streaming resume dataset mismatch: checkpoint={state.get('dataset')} "
                f"current={self.dataset}"
            )
        ckpt_split = state.get("dataset_split")
        if ckpt_split is not None and ckpt_split != self.dataset_split:
            raise ValueError(
                f"Streaming resume dataset_split mismatch: checkpoint={ckpt_split!r} "
                f"current={self.dataset_split!r}"
            )
        if int(state.get("max_seq_len", self.max_seq_len)) != int(self.max_seq_len):
            raise ValueError(
                f"Streaming resume max_seq_len mismatch: checkpoint={state.get('max_seq_len')} "
                f"current={self.max_seq_len}"
            )
        if int(state.get("min_chars", self.min_chars)) != int(self.min_chars):
            raise ValueError(
                f"Streaming resume min_chars mismatch: checkpoint={state.get('min_chars')} "
                f"current={self.min_chars}"
            )
        if state.get("teacher_model") != self._teacher_model:
            raise ValueError(
                f"Streaming resume teacher_model mismatch: checkpoint={state.get('teacher_model')} "
                f"current={self._teacher_model}"
            )
        target = int(state.get("consumed", 0))
        if target < 0:
            raise ValueError(f"Invalid consumed in checkpoint: {target}")
        if target > 0:
            log.info(
                "Restoring streaming dataset position (%s raw rows in split %r; using skip when available)...",
                target,
                self.dataset_split,
            )
        self._ds = _open_hf_streaming_iterator(self.dataset, self.dataset_split, target)
        self._consumed = target


def build_teacher_cache(args):
    from transformers import AutoModelForCausalLM

    data_dir = Path(args.data_dir)
    data_manifest_path = data_dir / "manifest.json"
    if not data_manifest_path.exists():
        raise FileNotFoundError(f"Missing data manifest: {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text())

    _validate_common_seq_args(
        int(data_manifest.get("max_seq_len", 0)),
        args.kl_start_pos,
    )

    token_chunks = sorted((data_dir / "chunks").glob("chunk_*.pt"))
    if not token_chunks:
        raise RuntimeError(f"No token chunks found in {data_dir / 'chunks'}")
    token_hash_cache_file = data_dir / "token_chunk_hashes.json"
    token_fingerprint = _fingerprint_chunk_files(token_chunks, cache_file=token_hash_cache_file)
    log.info(f"Token chunk fingerprint: {token_fingerprint[:12]}...")

    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_dir / "manifest.json"

    if data_manifest.get("teacher") != args.teacher:
        raise ValueError(
            f"Teacher mismatch: build_manifest={data_manifest.get('teacher')} cache_build={args.teacher}"
        )
    if int(data_manifest.get("kl_start_pos", args.kl_start_pos)) != int(args.kl_start_pos):
        raise ValueError(
            f"kl_start_pos mismatch: build_manifest={data_manifest.get('kl_start_pos')} "
            f"cache_build={args.kl_start_pos}"
        )

    if args.teacher_cache_dtype == "bf16":
        target_dtype = torch.bfloat16
    elif args.teacher_cache_dtype == "fp16":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    teacher_gpus = _gpu_span(args.teacher_gpu, args.teacher_gpu_count, "teacher")
    teacher_device_map, teacher_max_memory = _device_map_and_memory(teacher_gpus)

    monolithic = bool(getattr(args, "teacher_cache_monolithic", False))
    resume = bool(getattr(args, "resume_teacher_cache", False))
    if resume:
        log.info(
            "Resume enabled: skipping finished chunks; sharded chunks resume from the first "
            "missing shard. Monolithic chunks skip if the output .pt already exists."
        )
    if monolithic:
        log.warning(
            "teacher_cache_monolithic: each output chunk is one large .pt file holding all "
            "full-vocab log-probs in RAM before save — can OOM. Prefer default sharded layout."
        )
    else:
        log.info(
            "Writing sharded teacher cache (one small .pt per sample) to bound host RAM. "
            "Use --teacher_cache_monolithic only if you need legacy single-file chunks."
        )

    chunk_plans: list[dict] = []
    for tp in token_chunks:
        n = _token_chunk_num_samples(tp)
        if monolithic:
            out_pt = cache_dir / tp.name
            if resume and out_pt.is_file():
                chunk_plans.append({"n": n, "kind": "skip"})
            else:
                chunk_plans.append({"n": n, "kind": "monolithic"})
        else:
            out_dir = _teacher_cache_shard_dir(cache_dir, tp)
            if resume:
                start_j = _sharded_first_missing_sample(out_dir, n)
                if start_j is None:
                    chunk_plans.append({"n": n, "kind": "skip"})
                else:
                    chunk_plans.append({"n": n, "kind": "sharded", "start_j": start_j})
            else:
                chunk_plans.append({"n": n, "kind": "sharded", "start_j": 0})

    if all(p["kind"] == "skip" for p in chunk_plans):
        log.info("Teacher cache already complete for all chunks; not loading the teacher model.")
        total_samples = sum(p["n"] for p in chunk_plans)
        layout = "monolithic" if monolithic else "sharded"
        _save_manifest(
            cache_manifest_path,
            {
                "teacher": args.teacher,
                "kl_start_pos": args.kl_start_pos,
                "dtype": args.teacher_cache_dtype,
                "teacher_cache_layout": layout,
                "num_chunks": len(token_chunks),
                "num_samples": total_samples,
                "token_chunk_fingerprint": token_fingerprint,
                "token_hash_cache_file": str(token_hash_cache_file),
                "data_manifest_teacher": data_manifest.get("teacher"),
                "data_manifest_max_seq_len": data_manifest.get("max_seq_len"),
                "elapsed_sec": 0.0,
                "resume_teacher_cache": resume,
            },
        )
        log.info(f"Teacher cache manifest refreshed: {total_samples} samples in {cache_dir}")
        return

    log.info("Loading teacher (%s) on GPUs %s...", args.teacher, teacher_gpus)
    teacher_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": teacher_device_map,
        "trust_remote_code": True,
    }
    if teacher_max_memory is not None:
        teacher_load_kwargs["max_memory"] = teacher_max_memory
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_load_kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_in_dev = _first_param_device(teacher)
    if teacher_in_dev.type == "cuda":
        log.info(
            "Teacher loaded. Primary device %s VRAM: %.1fGB",
            teacher_in_dev,
            torch.cuda.memory_allocated(teacher_in_dev.index) / 1e9,
        )

    start_time = time.time()
    total_samples = 0
    for i, (token_chunk_path, plan) in enumerate(zip(token_chunks, chunk_plans)):
        chunk_t0 = time.time()
        n = plan["n"]
        if plan["kind"] == "skip":
            total_samples += n
            log.info(
                f"Resume: skip chunk {i+1}/{len(token_chunks)} ({token_chunk_path.name}, "
                f"{n} samples) | total_samples={total_samples}"
            )
            continue

        payload = torch.load(token_chunk_path, map_location="cpu", weights_only=False)
        input_ids_list = payload.get("input_ids", [])
        if not input_ids_list:
            raise RuntimeError(f"Empty token chunk: {token_chunk_path}")

        if monolithic:
            shard_dir = _teacher_cache_shard_dir(cache_dir, token_chunk_path)
            if shard_dir.is_dir():
                shutil.rmtree(shard_dir)
            teacher_log_probs = []
            with torch.no_grad():
                for j, ids in enumerate(input_ids_list):
                    ids = ids.unsqueeze(0).to(device=teacher_in_dev, dtype=torch.long)
                    logits = teacher(ids).logits[:, args.kl_start_pos:, :].float()
                    t_log_p = F.log_softmax(logits, dim=-1).to(target_dtype).cpu()
                    teacher_log_probs.append(t_log_p.squeeze(0))
                    del ids, logits, t_log_p
                    if (j + 1) % 100 == 0:
                        dt = time.time() - chunk_t0
                        log.info(
                            f"Building monolithic teacher cache | "
                            f"chunk {i+1}/{len(token_chunks)} | "
                            f"within-chunk {j+1}/{len(input_ids_list)} | "
                            f"{(j + 1) / max(dt, 1e-6):.2f} samp/s"
                        )

            out_chunk_path = cache_dir / token_chunk_path.name
            torch.save(
                {
                    "teacher_log_probs": teacher_log_probs,
                    "kl_start_pos": args.kl_start_pos,
                    "teacher": args.teacher,
                    "dtype": args.teacher_cache_dtype,
                },
                out_chunk_path,
            )
        else:
            out_dir = _teacher_cache_shard_dir(cache_dir, token_chunk_path)
            legacy_pt = cache_dir / token_chunk_path.name
            if legacy_pt.is_file():
                legacy_pt.unlink()
            start_j = plan["start_j"]
            if not resume:
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                for k in range(start_j, n):
                    (out_dir / f"{k:06d}.pt").unlink(missing_ok=True)
                for p in out_dir.glob("*.pt"):
                    if p.stem.isdigit() and len(p.stem) == 6 and int(p.stem) >= n:
                        p.unlink()

            with torch.no_grad():
                for j in range(start_j, len(input_ids_list)):
                    ids = input_ids_list[j]
                    ids = ids.unsqueeze(0).to(device=teacher_in_dev, dtype=torch.long)
                    logits = teacher(ids).logits[:, args.kl_start_pos:, :].float()
                    t_log_p = F.log_softmax(logits, dim=-1).to(target_dtype).cpu()
                    torch.save(
                        {
                            "teacher_log_probs": t_log_p.squeeze(0),
                            "kl_start_pos": args.kl_start_pos,
                            "teacher": args.teacher,
                            "dtype": args.teacher_cache_dtype,
                        },
                        out_dir / f"{j:06d}.pt",
                    )
                    del ids, logits, t_log_p
                    if (j + 1) % 100 == 0:
                        dt = time.time() - chunk_t0
                        log.info(
                            f"Building sharded teacher cache | "
                            f"chunk {i+1}/{len(token_chunks)} | "
                            f"within-chunk {j+1}/{len(input_ids_list)} | "
                            f"{(j + 1) / max(dt, 1e-6):.2f} samp/s"
                        )
                    if (j + 1) % 200 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

        total_samples += len(input_ids_list)
        elapsed = time.time() - start_time
        log.info(
            f"Cached teacher chunk {i+1}/{len(token_chunks)} | "
            f"samples={total_samples} | {total_samples/max(elapsed, 1e-6):.2f} samp/s"
        )
        del input_ids_list, payload
        gc.collect()
        torch.cuda.empty_cache()

    layout = "monolithic" if monolithic else "sharded"
    _save_manifest(
        cache_manifest_path,
        {
            "teacher": args.teacher,
            "kl_start_pos": args.kl_start_pos,
            "dtype": args.teacher_cache_dtype,
            "teacher_cache_layout": layout,
            "num_chunks": len(token_chunks),
            "num_samples": total_samples,
            "token_chunk_fingerprint": token_fingerprint,
            "token_hash_cache_file": str(token_hash_cache_file),
            "data_manifest_teacher": data_manifest.get("teacher"),
            "data_manifest_max_seq_len": data_manifest.get("max_seq_len"),
            "elapsed_sec": round(time.time() - start_time, 2),
            "resume_teacher_cache": resume,
        },
    )
    log.info(f"Teacher cache complete: {total_samples} samples in {cache_dir}")


def train_from_prebuilt(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from eval.dataset import sample_prompts_from_dataset, format_prompt

    _require_positive("samples_per_step", args.samples_per_step)
    _require_positive("save_every", args.save_every)
    _validate_common_seq_args(args.max_seq_len, args.kl_start_pos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.resume_from and args.resume_latest:
        raise ValueError("Use only one of --resume_from or --resume_latest.")
    if args.early_stop_on_beat_consecutive > 0 and args.eval_every_steps <= 0:
        raise ValueError(
            "--early_stop_on_beat_consecutive requires --eval_every_steps > 0."
        )

    def _resolve_resume_dir() -> Path | None:
        if args.resume_from:
            p = Path(args.resume_from)
            if not p.exists():
                raise FileNotFoundError(f"--resume_from path does not exist: {p}")
            return p
        if not args.resume_latest:
            return None
        candidates = []
        for d in output_dir.glob("step_*"):
            if not d.is_dir():
                continue
            stem = d.name.split("_", 1)[-1]
            if stem.isdigit():
                candidates.append((int(stem), d))
        if not candidates:
            raise FileNotFoundError(
                f"--resume_latest requested but no step_* directories found in {output_dir}"
            )
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    resume_dir = _resolve_resume_dir()

    data_manifest_path = Path(args.data_dir) / "manifest.json"
    using_streaming_fallback = not data_manifest_path.exists()
    if using_streaming_fallback:
        if args.use_teacher_cache:
            raise ValueError(
                "--use_teacher_cache requires prebuilt data + teacher cache manifests. "
                "Disable --use_teacher_cache when training directly from streaming data."
            )
        log.warning(
            "Prebuilt manifest not found at %s. Falling back to streaming dataset mode.",
            data_manifest_path,
        )
        ds_skip = int(getattr(args, "dataset_skip_rows", 0) or 0)
        if ds_skip < 0:
            raise ValueError("--dataset_skip_rows must be >= 0")
        if resume_dir and ds_skip > 0:
            log.warning(
                "--dataset_skip_rows is ignored when resuming from a checkpoint; "
                "streaming position comes from train_state.json data_state (or consumed)."
            )
        stream_skip_rows = 0 if resume_dir else ds_skip
        if stream_skip_rows > 0:
            log.info(
                "Streaming: skipping first %s raw example(s) in split %r of %s.",
                stream_skip_rows,
                args.dataset_split,
                args.dataset,
            )
        data = StreamingTokenStream(
            teacher_model=args.teacher,
            dataset=args.dataset,
            max_seq_len=args.max_seq_len,
            min_chars=args.min_chars,
            dataset_split=args.dataset_split,
            dataset_skip_rows=stream_skip_rows,
        )
    elif args.use_teacher_cache:
        data = PrebuiltTeacherTargetStream(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            seed=args.data_seed,
            shuffle_chunks=args.shuffle_chunks,
        )
    else:
        data = PrebuiltTokenStream(
            data_dir=args.data_dir,
            seed=args.data_seed,
            shuffle_chunks=args.shuffle_chunks,
        )

    # Basic compatibility guardrails to keep logic consistent
    manifest = data.manifest
    if not using_streaming_fallback:
        if int(manifest.get("max_seq_len", args.max_seq_len)) != int(args.max_seq_len):
            raise ValueError(
                f"max_seq_len mismatch: manifest={manifest.get('max_seq_len')} train={args.max_seq_len}"
            )
        if int(manifest.get("kl_start_pos", args.kl_start_pos)) != int(args.kl_start_pos):
            raise ValueError(
                f"kl_start_pos mismatch: manifest={manifest.get('kl_start_pos')} train={args.kl_start_pos}"
            )

    if not args.no_wandb:
        import wandb

        _wb_timeout = max(30, int(args.wandb_init_timeout))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or "distil-kl-prebuilt",
            config=vars(args),
            settings=wandb.Settings(init_timeout=_wb_timeout),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_gpus = _gpu_span(args.teacher_gpu, args.teacher_gpu_count, "teacher")
    student_gpus = _gpu_span(args.student_gpu, args.student_gpu_count, "student")
    teacher_device_map, teacher_max_memory = _device_map_and_memory(teacher_gpus)
    student_device_map, student_max_memory = _device_map_and_memory(student_gpus)

    teacher = None
    if not args.use_teacher_cache:
        log.info("Loading teacher (%s) on GPUs %s...", args.teacher, teacher_gpus)
        teacher_load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": teacher_device_map,
            "trust_remote_code": True,
        }
        if teacher_max_memory is not None:
            teacher_load_kwargs["max_memory"] = teacher_max_memory
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_load_kwargs)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_in_dev = _first_param_device(teacher)
        if teacher_in_dev.type == "cuda":
            log.info(
                "  Teacher primary %s: %.1fGB",
                teacher_in_dev,
                torch.cuda.memory_allocated(teacher_in_dev.index) / 1e9,
            )
    else:
        cache_manifest = data.cache_manifest
        if cache_manifest.get("teacher") != args.teacher:
            raise ValueError(
                f"Teacher mismatch: cache={cache_manifest.get('teacher')} train={args.teacher}"
            )
        if int(cache_manifest.get("kl_start_pos", args.kl_start_pos)) != int(args.kl_start_pos):
            raise ValueError(
                f"kl_start_pos mismatch: cache={cache_manifest.get('kl_start_pos')} "
                f"train={args.kl_start_pos}"
            )
        token_hash_cache_file = Path(args.data_dir) / "token_chunk_hashes.json"
        token_fingerprint = _fingerprint_chunk_files(
            data.token_chunk_paths, cache_file=token_hash_cache_file
        )
        expected_fingerprint = cache_manifest.get("token_chunk_fingerprint")
        if not expected_fingerprint:
            raise ValueError(
                "Teacher cache manifest missing token_chunk_fingerprint. "
                "Rebuild cache with current script."
            )
        if token_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Token chunk fingerprint mismatch: cache={expected_fingerprint[:12]}... "
                f"current={token_fingerprint[:12]}..."
            )
        log.info("Using cached teacher targets; teacher model will NOT be loaded for training.")

    student_source = str(resume_dir) if resume_dir else args.student
    log.info("Loading student (%s) on GPUs %s...", student_source, student_gpus)
    student_load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": student_device_map,
        "trust_remote_code": True,
    }
    if student_max_memory is not None:
        student_load_kwargs["max_memory"] = student_max_memory
    student = AutoModelForCausalLM.from_pretrained(student_source, **student_load_kwargs)
    student.train()
    student_in_dev = _first_param_device(student)
    log.info(
        f"  Student: {sum(p.numel() for p in student.parameters()):,} params, "
        f"{(torch.cuda.memory_allocated(student_in_dev.index)/1e9) if student_in_dev.type == 'cuda' else 0.0:.1f}GB"
    )

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    resume_global_step = 0
    resume_state: dict | None = None
    scheduler_path: Path | None = None
    if resume_dir:
        optimizer_path = resume_dir / "optimizer.pt"
        scheduler_path = resume_dir / "scheduler.pt"
        if not optimizer_path.exists():
            raise FileNotFoundError(f"Missing optimizer.pt in resume checkpoint: {resume_dir}")
        if not scheduler_path.exists():
            log.warning(
                "scheduler.pt missing in %s. Scheduler will start fresh from step 0.",
                resume_dir,
            )
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        resume_state, _train_state_repaired = _read_train_state_or_repair(resume_dir)
        resume_global_step = int(resume_state.get("global_step", 0))

    dataset_samples = int(manifest.get("num_samples", 0) or 0)
    if using_streaming_fallback and args.max_steps <= 0:
        raise ValueError(
            "Streaming fallback mode has no finite manifest sample count. "
            "Set --max_steps to a positive value."
        )
    if args.max_steps > 0:
        target_steps = args.max_steps
        stop_reason = f"max_steps={args.max_steps}"
    else:
        if dataset_samples <= 0:
            raise ValueError(
                "Manifest is missing a valid num_samples. Set --max_steps explicitly to control run length."
            )
        target_steps = max(1, math.ceil(dataset_samples / max(args.samples_per_step, 1)))
        stop_reason = (
            f"one full pass over prebuilt data ({dataset_samples} samples, "
            f"~{target_steps} steps at {args.samples_per_step} samples/step)"
        )
        log.info(
            "--max_steps not set (or 0). Training will stop after %s.",
            stop_reason,
        )
        if resume_dir and resume_global_step >= target_steps:
            extra = max(1, math.ceil(dataset_samples / max(args.samples_per_step, 1)))
            target_steps = resume_global_step + extra
            stop_reason = (
                f"resume extension by one more pass ({extra} steps) "
                f"from resumed step {resume_global_step}"
            )
            log.info(
                "Resumed step (%s) already reached auto target. Extending run: %s.",
                resume_global_step,
                stop_reason,
            )

    # Cosine schedule total must match this run length (hardcoding 100k flattens LR for short runs).
    scheduler_total_steps = max(int(target_steps), int(args.warmup_steps) + 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_steps), scheduler_total_steps
    )
    log.info(
        "LR scheduler: cosine over %s steps (warmup=%s).",
        scheduler_total_steps,
        args.warmup_steps,
    )
    if resume_dir and scheduler_path is not None and scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    origin_cfg_cache_dir = output_dir / "_origin_model_configs"
    copied_origin_cfgs = _ensure_origin_model_configs_cached(
        args.student,
        args.student_revision,
        origin_cfg_cache_dir,
    )
    if copied_origin_cfgs:
        log.info(
            "Cached %s origin model config files from %s",
            len(copied_origin_cfgs),
            args.student,
        )

    eval_prompts = []
    teacher_eval = None
    king_eval = None
    eval_cache = []
    if args.eval_every_steps > 0:
        king_start_gpu = args.king_gpu if args.king_gpu is not None else args.teacher_gpu
        king_gpus = _gpu_span(king_start_gpu, args.king_gpu_count, "king")
        king_device_map, king_max_memory = _device_map_and_memory(king_gpus)
        if not args.king_repo:
            raise ValueError("--king_repo is required when --eval_every_steps > 0")
        raw_prompts = sample_prompts_from_dataset(
            n=args.eval_prompts,
            block_number=args.eval_block_number,
            block_hash=None,
            dataset_name=args.eval_dataset,
        )
        for text in raw_prompts:
            formatted = format_prompt(text)
            if formatted:
                eval_prompts.append(formatted)
            if len(eval_prompts) >= args.eval_prompts:
                break
        if len(eval_prompts) < 2:
            raise RuntimeError(
                f"Need at least 2 eval prompts after filtering; got {len(eval_prompts)}"
            )
        if teacher is None:
            log.info("Eval enabled: loading teacher model for periodic king-comparison metrics...")
            teacher_eval_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": teacher_device_map,
                "trust_remote_code": True,
            }
            if teacher_max_memory is not None:
                teacher_eval_kwargs["max_memory"] = teacher_max_memory
            teacher_eval = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_eval_kwargs)
        else:
            teacher_eval = teacher
        log.info("Eval enabled: loading king model (%s) on GPUs %s...", args.king_repo, king_gpus)
        king_load_kwargs = {
            "revision": args.king_revision,
            "dtype": torch.bfloat16,
            "device_map": king_device_map,
            "trust_remote_code": True,
        }
        if king_max_memory is not None:
            king_load_kwargs["max_memory"] = king_max_memory
        king_eval = AutoModelForCausalLM.from_pretrained(args.king_repo, **king_load_kwargs)
        king_eval.eval()
        if teacher_eval is not None:
            teacher_eval.eval()
        log.info("Preparing deterministic eval cache from teacher continuations...")
        eval_cache = _prepare_eval_cache(
            teacher_eval=teacher_eval,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.eval_seed,
        )
        log.info("Prepared eval cache for %s prompts", len(eval_cache))
        log.info(
            "Periodic king eval configured: %s prompts every %s steps",
            len(eval_prompts),
            args.eval_every_steps,
        )

    sdev = _first_param_device(student)
    tdev = _first_param_device(teacher) if teacher is not None else None

    log.info("=== Starting training (prebuilt data) ===")
    log.info(
        f"  LR: {args.lr}, Warmup: {args.warmup_steps}, Samples/step: {args.samples_per_step}"
    )
    log.info(f"  Seq len: {args.max_seq_len}, KL from pos {args.kl_start_pos}")
    log.info(f"  Prebuilt samples: {manifest.get('num_samples')} from {args.data_dir}")
    if args.use_teacher_cache:
        log.info(f"  Teacher cache dir: {args.cache_dir or (Path(args.data_dir) / 'teacher_cache')}")
    else:
        log.info(
            "  Online teacher: chunked forwards (online_chunk_size=%s). "
            "Larger chunks = faster but more student-GPU memory for logits/activations.",
            getattr(args, "online_chunk_size", 1),
        )

    best_beating_delta = float("-inf")
    beat_streak = 0
    metrics_jsonl = Path(args.metrics_jsonl) if args.metrics_jsonl else (output_dir / "king_eval_metrics.jsonl")
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gate_delta_ema: float | None = None
    gate_streak = 0
    best_gate_score = float("-inf")
    best_gate_step = 0

    global_step = resume_global_step
    if resume_dir:
        assert resume_state is not None
        data_state = resume_state.get("data_state")
        if data_state is not None:
            data.load_state_dict(data_state)
            log.info("Resumed data stream state from checkpoint.")
        else:
            log.warning(
                "No data_state in checkpoint. Data stream starts fresh; this may replay samples."
            )
        best_beating_delta = float(resume_state.get("best_beating_delta", float("-inf")))
        beat_streak = int(resume_state.get("beat_streak", 0))
        log.info(
            "Resumed from %s | step=%s | data_pos=%s | best_delta=%.6f | beat_streak=%s",
            resume_dir,
            global_step,
            data.position,
            best_beating_delta if math.isfinite(best_beating_delta) else float("-inf"),
            beat_streak,
        )

    while global_step < target_steps:

        t0 = time.time()
        batch = data.get_batch(args.samples_per_step)
        if args.use_teacher_cache:
            pairs = [(ids, tlogp) for ids, tlogp in batch if ids.shape[0] > args.kl_start_pos + 10]
            if not pairs:
                log.warning("No valid token/teacher pairs after filtering.")
                break
        else:
            tokens = [t for t in batch if t.shape[0] > args.kl_start_pos + 10]
            if not tokens:
                log.warning("No valid tokens in batch after filtering.")
                break

        optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        if args.use_teacher_cache:
            for ids, teacher_log_probs in pairs:
                ids = ids.unsqueeze(0)
                s_logits = student(ids.to(device=sdev, dtype=torch.long)).logits
                loss = kl_loss_from_teacher_log_probs(
                    s_logits, teacher_log_probs, start_pos=args.kl_start_pos
                )
                (loss / len(pairs)).backward()
                total_loss += loss.item()
                n += 1
                del s_logits, loss
        else:
            # Batched forwards reduce kernel overhead; chunking caps peak VRAM (full logits [B,L,V]).
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            chunk_sz = max(1, int(getattr(args, "online_chunk_size", 1)))
            n_tokens = len(tokens)
            total_loss = 0.0
            _dt = next(student.parameters()).dtype
            for sub in _chunk_list(tokens, chunk_sz):
                n_sub = len(sub)
                input_ids_cpu, pos_mask = _pad_token_batch(sub, pad_id)
                attn_cpu = (input_ids_cpu != pad_id).long()
                input_ids_t = input_ids_cpu.to(device=tdev, dtype=torch.long, non_blocking=True)
                attn_t = attn_cpu.to(device=tdev, dtype=torch.long, non_blocking=True)
                with torch.no_grad():
                    t_logits = teacher(input_ids_t, attention_mask=attn_t).logits.to(
                        device=sdev, dtype=_dt, non_blocking=True
                    )
                input_ids_s = input_ids_cpu.to(device=sdev, dtype=torch.long, non_blocking=True)
                attn_s = attn_cpu.to(device=sdev, dtype=torch.long, non_blocking=True)
                s_logits = student(input_ids_s, attention_mask=attn_s).logits
                tail_mask = pos_mask[:, args.kl_start_pos :].to(device=sdev, dtype=torch.float32)
                loss_c = kl_loss_masked(s_logits, t_logits, args.kl_start_pos, tail_mask)
                (loss_c * (n_sub / n_tokens)).backward()
                total_loss += loss_c.item() * (n_sub / n_tokens)
                del t_logits, s_logits, loss_c
            n = n_tokens

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        global_step += 1

        elapsed = time.time() - t0
        avg_kl = total_loss / max(n, 1)
        lr = scheduler.get_last_lr()[0]
        log.info(
            f"Step {global_step} | KL: {avg_kl:.4f} | LR: {lr:.2e} | "
            f"{elapsed:.1f}s ({n/max(elapsed, 1e-6):.1f} samp/s) | pos: {data.position:,}"
        )

        if not args.no_wandb:
            import wandb

            wandb.log(
                {"train/kl": avg_kl, "train/lr": lr, "perf/step_time": elapsed},
                step=global_step,
            )

        if args.eval_every_steps > 0 and (
            global_step == 1 or global_step % args.eval_every_steps == 0
        ):
            eval_t0 = time.time()
            eval_scores, king_scores, ttest = _evaluate_against_king(
                student=student,
                king_eval=king_eval,
                eval_cache=eval_cache,
            )
            eval_stats = _summary_stats(eval_scores)
            king_stats = _summary_stats(king_scores)
            lr_scale = lr / max(args.lr, 1e-12)
            gate_delta_ema = _update_ema(
                gate_delta_ema, float(ttest["delta"]), float(args.gate_ema_beta)
            )
            gate_promotable = (
                gate_delta_ema >= float(args.gate_min_delta)
                and float(ttest["p"]) <= float(args.gate_max_p)
            )
            gate_streak = gate_streak + 1 if gate_promotable else 0
            gate_is_stable = gate_streak >= int(args.gate_patience_evals)
            gate_score = _gate_score(gate_delta_ema, float(ttest["p"]))
            entry = {
                "step": int(global_step),
                "lr_scale": float(lr_scale),
                "train_loss": float(avg_kl),
                "eval_stats": eval_stats,
                "king_stats": king_stats,
                "ttest": ttest,
                "gate": {
                    "delta_ema": float(gate_delta_ema),
                    "promotable": bool(gate_promotable),
                    "streak": int(gate_streak),
                    "is_stable": bool(gate_is_stable),
                    "score": float(gate_score),
                },
                "time_s": float(time.time() - eval_t0),
                "ts": _utc_now_iso(),
            }
            print(json.dumps(entry, ensure_ascii=True))
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

            beats_king = (
                eval_stats["mean"] < king_stats["mean"]
                and ttest["delta"] >= args.beat_king_min_delta
                and ttest["p"] <= args.beat_king_p_threshold
            )
            if beats_king:
                beat_streak += 1
                if ttest["delta"] > best_beating_delta:
                    best_beating_delta = ttest["delta"]
                    best_dir = output_dir / "best_beat_king"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    # Restore full origin config/tokenizer files over save_pretrained output when cached.
                    _copy_cached_origin_configs(origin_cfg_cache_dir, best_dir)
                    _atomic_write_json(
                        best_dir / "train_state.json",
                        {
                            "global_step": global_step,
                            "data_position": data.position,
                            "best_beating_delta": best_beating_delta,
                            "ttest": ttest,
                            "eval_stats": eval_stats,
                            "king_stats": king_stats,
                            "saved_at": _utc_now_iso(),
                        },
                    )
                    log.info(
                        "New best king-beating checkpoint saved at %s (delta=%.6f, p=%.6g)",
                        best_dir,
                        ttest["delta"],
                        ttest["p"],
                    )
            else:
                beat_streak = 0

            if gate_is_stable and gate_score > best_gate_score:
                best_gate_score = gate_score
                best_gate_step = int(global_step)
                stable_dir = output_dir / "best_stable_vs_king"
                stable_dir.mkdir(parents=True, exist_ok=True)
                student.save_pretrained(stable_dir)
                tokenizer.save_pretrained(stable_dir)
                _copy_cached_origin_configs(origin_cfg_cache_dir, stable_dir)
                _atomic_write_json(
                    stable_dir / "train_state.json",
                    {
                        "global_step": global_step,
                        "data_position": data.position,
                        "gate": {
                            "delta": float(ttest["delta"]),
                            "p": float(ttest["p"]),
                            "delta_ema": float(gate_delta_ema),
                            "streak": int(gate_streak),
                            "min_delta": float(args.gate_min_delta),
                            "max_p": float(args.gate_max_p),
                            "patience_evals": int(args.gate_patience_evals),
                            "ema_beta": float(args.gate_ema_beta),
                            "score": float(gate_score),
                        },
                        "eval_stats": eval_stats,
                        "king_stats": king_stats,
                        "saved_at": _utc_now_iso(),
                    },
                )
                log.info(
                    "New best stable-vs-king checkpoint saved at %s (score=%.6f, delta_ema=%.6f, p=%.6g)",
                    stable_dir,
                    gate_score,
                    gate_delta_ema,
                    ttest["p"],
                )

            if not args.no_wandb:
                import wandb

                wandb.log(
                    {
                        "eval/mean": eval_stats["mean"],
                        "eval/std": eval_stats["std"],
                        "eval/p50": eval_stats["p50"],
                        "king/mean": king_stats["mean"],
                        "king/std": king_stats["std"],
                        "ttest/p": ttest["p"],
                        "ttest/t": ttest["t"],
                        "ttest/delta": ttest["delta"],
                        "ttest/beats_king": int(beats_king),
                        "train/beat_streak": beat_streak,
                        "gate/delta_ema": gate_delta_ema,
                        "gate/promotable": int(gate_promotable),
                        "gate/streak": gate_streak,
                        "gate/is_stable": int(gate_is_stable),
                        "gate/score": gate_score,
                    },
                    step=global_step,
                )

            if args.early_stop_on_beat_consecutive > 0 and beat_streak >= args.early_stop_on_beat_consecutive:
                log.info(
                    "Early stop: beat-king criteria met for %s consecutive evals.",
                    beat_streak,
                )
                stop_reason = (
                    f"beat-king early stop (streak={beat_streak}, "
                    f"p<={args.beat_king_p_threshold}, delta>={args.beat_king_min_delta})"
                )
                break

        if global_step % args.save_every == 0:
            d = output_dir / f"step_{global_step}"
            d.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(d)
            tokenizer.save_pretrained(d)
            # Overwrite save_pretrained config.json with full origin Hub/local snapshot when cached.
            _copy_cached_origin_configs(origin_cfg_cache_dir, d)
            torch.save(optimizer.state_dict(), d / "optimizer.pt")
            torch.save(scheduler.state_dict(), d / "scheduler.pt")
            _atomic_write_json(
                d / "train_state.json",
                {
                    "global_step": global_step,
                    "data_position": data.position,
                    "data_state": data.state_dict(),
                    "best_beating_delta": best_beating_delta,
                    "beat_streak": beat_streak,
                },
            )
            log.info(f"  Saved: {d}")

        if global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    log.info(f"Done. Step {global_step} (stopped by {stop_reason})")
    if best_gate_step > 0:
        log.info(
            "Best stable-vs-king checkpoint selected at step %s (score=%.6f).",
            best_gate_step,
            best_gate_score,
        )
    if not args.no_wandb:
        import wandb

        wandb.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="KL distillation with prebuilt token dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser(
        "build", help="Prebuild tokenized samples from streaming dataset."
    )
    p_build.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    p_build.add_argument("--dataset", type=str, default=DATASET)
    p_build.add_argument("--data_dir", type=str, default="./prebuilt-data")
    p_build.add_argument("--num_samples", type=int, default=100000)
    p_build.add_argument("--chunk_size", type=int, default=2000)
    p_build.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p_build.add_argument("--kl_start_pos", type=int, default=KL_START_POS)
    p_build.add_argument("--min_chars", type=int, default=MIN_CHARS)

    p_cache = subparsers.add_parser(
        "build_teacher_cache",
        help="Precompute teacher log-prob targets from prebuilt token chunks.",
    )
    p_cache.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    p_cache.add_argument("--teacher_gpu", type=int, default=0)
    p_cache.add_argument(
        "--teacher_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to teacher model (starting at --teacher_gpu).",
    )
    p_cache.add_argument("--data_dir", type=str, default="./prebuilt-data")
    p_cache.add_argument("--cache_dir", type=str, default=None)
    p_cache.add_argument("--kl_start_pos", type=int, default=KL_START_POS)
    p_cache.add_argument(
        "--teacher_cache_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Stored dtype for teacher log-prob targets.",
    )
    p_cache.add_argument(
        "--teacher_cache_monolithic",
        action="store_true",
        help=(
            "Write one chunk_*.pt per token chunk (legacy). Holds every sample's full-vocab "
            "log-probs in RAM before torch.save — often OOM-killed for large models/chunks."
        ),
    )
    p_cache.add_argument(
        "--resume_teacher_cache",
        action="store_true",
        help=(
            "Skip token chunks that are already fully cached. For sharded layout, continue each "
            "chunk from the first missing ######.pt (tail shards are rewritten). Monolithic: skip "
            "if chunk_*.pt exists (assumed complete; delete the file to rebuild that chunk)."
        ),
    )

    p_train = subparsers.add_parser(
        "train", help="Train from prebuilt tokenized samples."
    )
    p_train.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    p_train.add_argument("--student", type=str, default=STUDENT_MODEL)
    p_train.add_argument(
        "--student_revision",
        type=str,
        default=None,
        help="Optional Hub revision/branch for --student when downloading origin config/tokenizer files.",
    )
    p_train.add_argument("--teacher_gpu", type=int, default=0)
    p_train.add_argument("--student_gpu", type=int, default=1)
    p_train.add_argument(
        "--teacher_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to teacher model (starting at --teacher_gpu).",
    )
    p_train.add_argument(
        "--student_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to student model (starting at --student_gpu).",
    )
    p_train.add_argument(
        "--king_gpu",
        type=int,
        default=None,
        help="Start GPU index for king model (defaults to --teacher_gpu).",
    )
    p_train.add_argument(
        "--king_gpu_count",
        type=int,
        default=1,
        help="Number of GPUs allocated to king model (starting at --king_gpu).",
    )
    p_train.add_argument("--data_dir", type=str, default="./prebuilt-data")
    p_train.add_argument("--cache_dir", type=str, default=None)
    p_train.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Dataset used when prebuilt manifest is missing (streaming fallback mode).",
    )
    p_train.add_argument(
        "--min_chars",
        type=int,
        default=MIN_CHARS,
        help="Minimum text length filter for streaming fallback mode.",
    )
    p_train.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Streaming fallback only: Hugging Face ``datasets`` split name (e.g. train, validation).",
    )
    p_train.add_argument(
        "--dataset_skip_rows",
        type=int,
        default=0,
        help=(
            "Streaming fallback only: skip this many raw examples from the start of the split "
            "before training (uses IterableDataset.skip when available). Ignored when resuming from a checkpoint."
        ),
    )
    p_train.add_argument("--use_teacher_cache", action="store_true")
    p_train.add_argument("--data_seed", type=int, default=42)
    p_train.add_argument("--shuffle_chunks", action="store_true")
    p_train.add_argument("--lr", type=float, default=LR)
    p_train.add_argument("--warmup_steps", type=int, default=WARMUP)
    p_train.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p_train.add_argument("--samples_per_step", type=int, default=SAMPLES_PER_STEP)
    p_train.add_argument(
        "--online_chunk_size",
        type=int,
        default=1,
        help=(
            "Online teacher only: max sequences per teacher+student forward (memory knob). "
            "1 matches the old per-sample memory; increase (e.g. 2–4) if GPU 1 has headroom for "
            "full logits [chunk,L,V] during backward."
        ),
    )
    p_train.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p_train.add_argument("--kl_start_pos", type=int, default=KL_START_POS)
    p_train.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Training steps. If 0, run one full pass over prebuilt samples and exit.",
    )
    p_train.add_argument("--output_dir", type=str, default="./distil-checkpoints")
    p_train.add_argument("--save_every", type=int, default=SAVE_EVERY)
    p_train.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint dir to resume from (e.g. ./distil-checkpoints/step_1400).",
    )
    p_train.add_argument(
        "--resume_latest",
        action="store_true",
        help="Resume from latest step_* checkpoint in --output_dir.",
    )
    p_train.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="If > 0, run periodic eval-vs-king and emit JSON metric lines.",
    )
    p_train.add_argument(
        "--eval_prompts",
        type=int,
        default=500,
        help="Number of prompts used for periodic king comparison.",
    )
    p_train.add_argument(
        "--eval_dataset",
        type=str,
        default=DATASET,
        help="Dataset used to sample eval prompts for king comparison.",
    )
    p_train.add_argument(
        "--eval_block_number",
        type=int,
        default=12345,
        help="Deterministic dataset block for prompt sampling.",
    )
    p_train.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=512,
        help="Teacher continuation length for periodic evaluation.",
    )
    p_train.add_argument(
        "--eval_seed",
        type=int,
        default=12345,
        help="Seed for deterministic teacher continuations used in periodic eval.",
    )
    p_train.add_argument(
        "--king_repo",
        type=str,
        default=None,
        help="King model repo for periodic eval-vs-king metrics.",
    )
    p_train.add_argument(
        "--king_revision",
        type=str,
        default=None,
        help="Optional king model revision.",
    )
    p_train.add_argument(
        "--beat_king_p_threshold",
        type=float,
        default=0.05,
        help="Beat-king criterion: require ttest.p <= this value.",
    )
    p_train.add_argument(
        "--beat_king_min_delta",
        type=float,
        default=0.0,
        help="Beat-king criterion: require ttest.delta >= this value.",
    )
    p_train.add_argument(
        "--early_stop_on_beat_consecutive",
        type=int,
        default=0,
        help="If > 0, stop when beat-king criteria hold this many evals in a row.",
    )
    p_train.add_argument(
        "--metrics_jsonl",
        type=str,
        default=None,
        help="Path to append eval-vs-king JSON lines (default: output_dir/king_eval_metrics.jsonl).",
    )
    p_train.add_argument(
        "--gate_min_delta",
        type=float,
        default=0.0025,
        help="Stable-gate minimum EMA delta (king - student) required for promotion.",
    )
    p_train.add_argument(
        "--gate_max_p",
        type=float,
        default=0.03,
        help="Stable-gate maximum one-sided p-value required for promotion.",
    )
    p_train.add_argument(
        "--gate_patience_evals",
        type=int,
        default=3,
        help="Stable-gate required consecutive promotable evals before checkpoint promotion.",
    )
    p_train.add_argument(
        "--gate_ema_beta",
        type=float,
        default=0.6,
        help="EMA smoothing factor for stable-gate delta.",
    )
    p_train.add_argument("--wandb_project", type=str, default="distil-subnet97")
    p_train.add_argument("--wandb_run", type=str, default=None)
    p_train.add_argument(
        "--wandb_init_timeout",
        type=int,
        default=300,
        help="Seconds to wait for wandb.run init (slow networks / api.wandb.ai).",
    )
    p_train.add_argument("--no_wandb", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "build":
        build_dataset(args)
    elif args.command == "build_teacher_cache":
        build_teacher_cache(args)
    elif args.command == "train":
        train_from_prebuilt(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
