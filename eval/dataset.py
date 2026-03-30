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

    # Use block number to seed a deterministic but unpredictable prompt selection.
    # We use the block as both the shuffle seed and a skip offset so each epoch
    # draws from a different region of the dataset.
    block_hash = hashlib.sha256(str(block_number).encode()).hexdigest()
    skip_offset = int(block_hash[:8], 16) % 50_000  # moderate skip to avoid slow seeks

    print(
        f"[dataset] Sampling {n} prompts from {dataset_name} "
        f"(block={block_number}, skip={skip_offset:,})",
        flush=True,
    )

    # Stream with a small buffer to keep RAM usage low.
    # The shuffle buffer_size controls how many items are held in memory for
    # pseudo-random sampling. 5000 is enough for diversity without OOM.
    ds = load_dataset(dataset_name, split=split, streaming=True, name="default")
    ds_shuffled = ds.shuffle(seed=block_number, buffer_size=5_000)
    ds_skipped = ds_shuffled.skip(skip_offset)

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


def format_prompt(text: str, max_chars: int = 512) -> str:
    """
    Format a raw pretraining text as a continuation prompt.
    Uses the first ~max_chars as context, model continues from there.

    Includes sanitization to prevent malformed inputs from crashing
    the tokenizer or model:
    - Strips control characters (except newlines/tabs)
    - Removes null bytes
    - Limits total length
    - Rejects prompts that are mostly non-text (binary garbage)
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove null bytes and control chars (keep \n, \t, \r)
    text = text.replace("\x00", "")
    text = "".join(
        c for c in text
        if c in ("\n", "\t", "\r") or (ord(c) >= 32) or (ord(c) >= 128)
    )

    text = text.strip()
    if not text:
        return ""

    # Reject if >50% non-printable/non-ASCII after cleanup (likely binary)
    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
    if printable_count < len(text) * 0.5:
        return ""

    # Truncate to max_chars at a sentence boundary
    if len(text) > max_chars:
        cut = text[:max_chars].rfind(". ")
        if cut > max_chars // 3:
            text = text[: cut + 1]
        else:
            text = text[:max_chars]

    return text
