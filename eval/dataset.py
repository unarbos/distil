"""
SweInfinite dataset loader with block-seeded prompt selection.

Prompts are selected deterministically based on the current block number,
making it impossible for miners to predict/overfit to specific prompts.
"""
import glob
import json
import os
import random
import logging

logger = logging.getLogger("distillation.dataset")


def load_swe_infinite_prompts(dataset_path: str) -> list[dict]:
    """
    Load all problem statements from SweInfinite JSON files.

    Returns list of dicts with keys: instance_id, repo, problem_statement.
    """
    prompts: list[dict] = []
    for filepath in sorted(glob.glob(os.path.join(dataset_path, "*.json"))):
        with open(filepath) as f:
            data = json.load(f)
        prompts.append({
            "instance_id": data["instance_id"],
            "repo": data["repo"],
            "problem_statement": data["problem_statement"],
        })
    return prompts


def sample_prompts_seeded(
    prompts: list[dict],
    n: int,
    block_number: int,
) -> list[dict]:
    """
    Sample n prompts deterministically seeded by block_number.

    Using the block number as seed makes prompt selection unpredictable
    (miners can't know which prompts will be used until the block is produced)
    and reproducible (any validator can verify the same prompts were used).
    """
    rng = random.Random(block_number)
    return rng.sample(prompts, min(n, len(prompts)))


def sample_prompts(prompts: list[dict], n: int) -> list[dict]:
    """Random sample without block seeding (for testing/simulation)."""
    return random.sample(prompts, min(n, len(prompts)))


def format_coding_prompt(problem: dict) -> str:
    """Format a SweInfinite problem as a coding prompt for model inference."""
    return (
        "You are an expert software engineer. "
        "Analyze the following GitHub issue and provide a solution.\n\n"
        f"Repository: {problem['repo']}\n\n"
        f"Issue:\n{problem['problem_statement']}\n\n"
        "Provide your analysis and proposed code changes:"
    )
