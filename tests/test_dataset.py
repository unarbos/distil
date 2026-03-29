"""Tests for SweInfinite dataset loading."""

import json
import os
import tempfile

import pytest

from eval.dataset import format_coding_prompt, load_swe_infinite_prompts, sample_prompts


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a temporary dataset directory with sample JSON files."""
    for i in range(3):
        data = {
            "instance_id": f"test__repo-{i}",
            "repo": f"org/repo-{i}",
            "problem_statement": f"Fix bug number {i} in the parser module.",
            "extra_field": "ignored",
        }
        filepath = tmp_path / f"test_repo-{i}.json"
        filepath.write_text(json.dumps(data))
    return str(tmp_path)


class TestLoadSweInfinitePrompts:
    def test_loads_all_files(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        assert len(prompts) == 3

    def test_required_keys(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        for p in prompts:
            assert "instance_id" in p
            assert "repo" in p
            assert "problem_statement" in p

    def test_empty_directory(self, tmp_path):
        prompts = load_swe_infinite_prompts(str(tmp_path))
        assert prompts == []

    def test_extra_fields_excluded(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        for p in prompts:
            assert "extra_field" not in p


class TestSamplePrompts:
    def test_sample_less_than_available(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        sampled = sample_prompts(prompts, 2)
        assert len(sampled) == 2

    def test_sample_more_than_available(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        sampled = sample_prompts(prompts, 100)
        assert len(sampled) == len(prompts)

    def test_sample_zero(self, tmp_dataset):
        prompts = load_swe_infinite_prompts(tmp_dataset)
        sampled = sample_prompts(prompts, 0)
        assert len(sampled) == 0


class TestFormatCodingPrompt:
    def test_contains_repo(self):
        problem = {
            "instance_id": "x",
            "repo": "org/repo",
            "problem_statement": "Fix the bug.",
        }
        result = format_coding_prompt(problem)
        assert "org/repo" in result

    def test_contains_problem_statement(self):
        problem = {
            "instance_id": "x",
            "repo": "org/repo",
            "problem_statement": "Unique problem description XYZ123.",
        }
        result = format_coding_prompt(problem)
        assert "Unique problem description XYZ123." in result

    def test_is_nonempty_string(self):
        problem = {
            "instance_id": "x",
            "repo": "a/b",
            "problem_statement": "desc",
        }
        result = format_coding_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 50
