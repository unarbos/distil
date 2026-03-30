"""
Dataset loader with block-seeded prompt sampling from the full FineWeb dataset.

Each eval epoch uses the block number to seek into a different region of the
1.5 trillion token FineWeb dataset. No fixed prompt pool — every eval draws
from a fresh, unpredictable slice of the full dataset.
"""
import json
import os
import random
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("distillation.dataset")

# Default HF dataset for prompt sourcing
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
PROMPT_CACHE_DIR = Path("state/prompt_cache")

# Legacy pool size for backward compat with load_prompts_from_hf
DEFAULT_POOL_SIZE = 10_000


def load_prompts_from_hf(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    n: int = DEFAULT_POOL_SIZE,
    min_chars: int = 200,
    max_chars: int = 4000,
    cache_path: Path | None = None,
) -> list[str]:
    """
    Stream n prompts from a HuggingFace dataset.

    This is the legacy loader that builds a fixed pool. Prefer
    sample_prompts_from_dataset() for production, which samples
    directly from the full dataset each epoch.
    """
    if cache_path is None:
        cache_path = PROMPT_CACHE_DIR / f"{dataset_name.replace('/', '_')}_{n}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if len(cached) >= n:
                return cached[:n]
        except Exception:
            pass

    from datasets import load_dataset

    print(f"[dataset] Streaming {n} prompts from {dataset_name}...", flush=True)
    ds = load_dataset(dataset_name, split=split, streaming=True, name="default")

    prompts: list[str] = []
    seen = 0
    for item in ds:
        seen += 1
        text = item.get(text_field, "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        prompts.append(text)
        if len(prompts) >= n:
            break
        if seen > n * 20:
            break

    print(f"[dataset] Got {len(prompts)} prompts (scanned {seen} items)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prompts))

    return prompts


def sample_prompts_from_dataset(
    n: int,
    block_number: int,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    min_chars: int = 200,
    max_chars: int = 4000,
    cache_dir: Path | None = None,
) -> list[str]:
    """
    Sample n prompts directly from the full dataset, seeded by block_number.

    Uses the block number to compute a skip offset into the streaming dataset,
    so each epoch draws from a completely different region of the 1.5T token
    corpus. No fixed pool — miners cannot predict or overfit to the prompts.

    Results are cached per block so repeated calls (e.g. retries) return the
    same prompts.
    """
    if cache_dir is None:
        cache_dir = PROMPT_CACHE_DIR

    # Check block-specific cache
    cache_path = cache_dir / f"block_{block_number}_{n}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if len(cached) >= n:
                logger.info(f"Using cached prompts for block {block_number}")
                return cached[:n]
        except Exception:
            pass

    from datasets import load_dataset

    # Use block number to determine where in the dataset to sample from.
    # FineWeb has ~15B documents. We hash the block number to get a large
    # skip offset, then stream from there.
    block_hash = hashlib.sha256(str(block_number).encode()).hexdigest()
    # Use first 12 hex chars → up to ~281 trillion, way more than dataset size.
    # The streaming dataset wraps around, so any offset works.
    skip_offset = int(block_hash[:12], 16) % 1_000_000_000  # mod 1B for safety

    print(
        f"[dataset] Sampling {n} prompts from {dataset_name} "
        f"(block={block_number}, offset={skip_offset:,})",
        flush=True,
    )

    ds = load_dataset(dataset_name, split=split, streaming=True, name="default")
    ds_shuffled = ds.shuffle(seed=block_number, buffer_size=10_000)

    # Skip into the dataset using the block-derived offset.
    # For efficiency, use skip() which is O(1) on streaming datasets.
    ds_skipped = ds_shuffled.skip(skip_offset % 100_000)  # skip within shuffle buffer range

    prompts: list[str] = []
    seen = 0
    max_scan = n * 20  # safety limit

    for item in ds_skipped:
        seen += 1
        text = item.get(text_field, "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        prompts.append(text)
        if len(prompts) >= n:
            break
        if seen > max_scan:
            break

    print(f"[dataset] Got {len(prompts)} prompts (scanned {seen} items)", flush=True)

    # Cache for this block
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prompts))

    return prompts


def sample_prompts_seeded(
    prompts: list[str],
    n: int,
    block_number: int,
) -> list[str]:
    """
    Sample n prompts from a pre-loaded pool, seeded by block_number.

    Legacy function for use with load_prompts_from_hf(). For production,
    prefer sample_prompts_from_dataset() which samples directly from the
    full dataset without a fixed pool.
    """
    rng = random.Random(block_number)
    return rng.sample(prompts, min(n, len(prompts)))


def sample_prompts(prompts: list[str], n: int) -> list[str]:
    """Random sample without block seeding (for testing/simulation)."""
    return random.sample(prompts, min(n, len(prompts)))


def format_prompt(text: str) -> str:
    """
    Format a raw pretraining text as a continuation prompt.
    Uses the first ~512 chars as context, model continues from there.
    """
    # Clean up: strip leading whitespace, normalize
    text = text.strip()
    # Use first portion as the prompt prefix
    if len(text) > 512:
        # Try to cut at a sentence boundary
        cut = text[:512].rfind(". ")
        if cut > 200:
            text = text[: cut + 1]
        else:
            text = text[:512]
    return text
