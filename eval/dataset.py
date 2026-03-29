"""Load coding prompts from the SweInfinite dataset."""

import glob
import json
import os
import random


def load_swe_infinite_prompts(dataset_path: str) -> list[dict]:
    """
    Load all problem statements from SweInfinite JSON files.

    Each JSON file contains at minimum:
        - instance_id
        - repo
        - problem_statement

    Returns:
        List of dicts with keys: instance_id, repo, problem_statement.
    """
    prompts: list[dict] = []
    for filepath in sorted(glob.glob(os.path.join(dataset_path, "*.json"))):
        with open(filepath) as f:
            data = json.load(f)
        prompts.append(
            {
                "instance_id": data["instance_id"],
                "repo": data["repo"],
                "problem_statement": data["problem_statement"],
            }
        )
    return prompts


def sample_prompts(prompts: list[dict], n: int) -> list[dict]:
    """Randomly sample *n* prompts (or all if fewer than *n* available)."""
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
