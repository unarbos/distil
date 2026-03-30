"""
Dataset loader with block-seeded prompt selection.

Uses HuggingFace pretraining datasets (streamed) for diverse, unpredictable prompts.
Prompts are selected deterministically based on the current block number.

The prompt pool is large (10,000+) to prevent miners from overfitting to a small set.
Block-seeded sampling selects a small eval subset each epoch.
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

# Large pool — miners can't overfit to 40 prompts if the pool is 10k+
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
    Stream n prompts from a HuggingFace dataset to build a large pool.

    The pool is cached locally. Block-seeded sampling (sample_prompts_seeded)
    draws a small eval subset each epoch, so the full pool is never exposed
    to miners at once.

    Args:
        n: Pool size. Default 10,000. Larger = more diversity per eval.
    """
    if cache_path is None:
        cache_path = PROMPT_CACHE_DIR / f"{dataset_name.replace('/', '_')}_{n}.json"

    # Return cached if we have enough
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if len(cached) >= n:
                return cached[:n]
            # Cache exists but is too small (e.g. old 500-entry cache) — rebuild
            logger.info(f"Cache has {len(cached)} prompts but need {n}, rebuilding...")
        except Exception:
            pass

    # Stream from HF
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
        # Truncate long texts to max_chars
        if len(text) > max_chars:
            text = text[:max_chars]
        prompts.append(text)
        if len(prompts) >= n:
            break
        if seen > n * 20:  # Safety: don't scan forever
            break

    print(f"[dataset] Got {len(prompts)} prompts (scanned {seen} items)", flush=True)

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prompts))

    return prompts


def sample_prompts_seeded(
    prompts: list[str],
    n: int,
    block_number: int,
) -> list[str]:
    """
    Sample n prompts deterministically seeded by block_number.

    Using the block number as seed makes prompt selection unpredictable
    (miners can't know which prompts will be used until the block is produced)
    and reproducible (any validator can verify the same prompts were used).
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
