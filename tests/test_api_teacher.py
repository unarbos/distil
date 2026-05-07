from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import api_teacher


class TinyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=False):
        n = max(1, len(text.split()))
        return SimpleNamespace(input_ids=torch.arange(n, dtype=torch.long).unsqueeze(0))


def _cfg():
    return api_teacher.APIConfig(
        base_url="https://example.test/api",
        api_key="test-key",
        model="teacher/test",
        endpoint="chat",
        top_logprobs=2,
        concurrency=1,
        timeout_s=15,
    )


def test_generate_single_prompt_retries_empty_logprobs(monkeypatch):
    calls = []

    def fake_call(prompt_text, cfg, max_new_tokens, block_seed, idx, request):
        calls.append(idx)
        if len(calls) < 3:
            return {"text": " hello", "top_logprobs_list": [], "raw": {}}
        return {"text": " hello", "top_logprobs_list": [{" hello": -0.1}], "raw": {}}

    monkeypatch.setattr(api_teacher, "_call_api_chat", fake_call)
    monkeypatch.setattr(api_teacher.time, "sleep", lambda _seconds: None)

    idx, result = api_teacher._generate_single_prompt_api(
        7,
        "prompt",
        8,
        None,
        _cfg(),
        TinyTokenizer(),
        {" hello": 1},
        lambda top_lp, token_to_id, tokenizer, k: {"indices": [1], "values": [-0.1]},
        lambda full_text, prompt_text, full_ids, tokenizer: 1,
    )

    assert idx == 7
    assert len(calls) == 3
    assert result["sparse_logprobs"] == {"indices": [1], "values": [-0.1]}
    assert "api_missing_logprobs" not in result


def test_generate_single_prompt_marks_missing_logprobs_after_retries(monkeypatch):
    calls = []

    def fake_call(prompt_text, cfg, max_new_tokens, block_seed, idx, request):
        calls.append(idx)
        return {"text": " hello", "top_logprobs_list": [], "raw": {}}

    monkeypatch.setattr(api_teacher, "_call_api_chat", fake_call)
    monkeypatch.setattr(api_teacher.time, "sleep", lambda _seconds: None)

    _idx, result = api_teacher._generate_single_prompt_api(
        3,
        "prompt",
        8,
        None,
        _cfg(),
        TinyTokenizer(),
        {" hello": 1},
        lambda top_lp, token_to_id, tokenizer, k: {"indices": [1], "values": [-0.1]},
        lambda full_text, prompt_text, full_ids, tokenizer: 1,
    )

    assert len(calls) == 6
    assert result["api_missing_logprobs"] is True
    assert "sparse_logprobs" not in result


def test_api_health_check_rejects_zero_logprobs(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "{}"
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "OK"}, "logprobs": {"content": []}},
                ],
            }

    import requests

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="no logprobs"):
        api_teacher.api_health_check(_cfg(), max_attempts=1)
