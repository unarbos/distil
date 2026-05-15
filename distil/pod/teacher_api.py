"""Cloud-API teacher path (OpenRouter / OpenAI-compatible).

The 1T-parameter Kimi-K2.6 teacher does NOT fit on a single B200 (or
even a single 8xB200 pod in bf16 — the model is ~2 TB while the pod
has ~500 GB usable VRAM). Production has been running the teacher
through OpenRouter for months, with student scoring still local on
the pod.

This module is the distil-stack equivalent of the legacy
``scripts/api_teacher.py`` (669 LoC, tightly coupled to pod_eval_vllm's
sparse-logprobs encoding). Here we return :class:`TeacherOutput`
directly — the same shape ``distil/pod/teacher.py:generate_continuations``
produces — so the rest of the pipeline is mode-agnostic.

Activated by ``DISTIL_TEACHER_MODE=api``. Required env:

* ``DISTIL_TEACHER_API_KEY`` (or ``OPENROUTER_API_KEY``)
* ``DISTIL_TEACHER_API_MODEL``  (default: moonshotai/kimi-k2.6)
* ``DISTIL_TEACHER_API_BASE``   (default: https://openrouter.ai/api)
* ``DISTIL_TEACHER_API_PROVIDERS``  comma list; default ``Inceptron``
  (the only K2.6 endpoint on OpenRouter that exposes logprobs)
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from distil.pod.teacher import TeacherOutput

logger = logging.getLogger("distil.pod.teacher_api")

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_CONCURRENCY = 12
DEFAULT_TIMEOUT_S = 120
# Inceptron is the only OpenRouter provider that returns per-token
# logprobs for K2.6 — other providers respond with ``logprobs: null``
# which would zero our KL signal.
DEFAULT_PROVIDERS = ("Inceptron",)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _config():
    api_key = _env("DISTIL_TEACHER_API_KEY") or _env("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DISTIL_TEACHER_MODE=api requires DISTIL_TEACHER_API_KEY "
            "(or OPENROUTER_API_KEY) in the environment."
        )
    providers_env = _env("DISTIL_TEACHER_API_PROVIDERS")
    providers = (
        tuple(p.strip() for p in providers_env.split(",") if p.strip())
        if providers_env
        else DEFAULT_PROVIDERS
    )
    return {
        "base_url": _env("DISTIL_TEACHER_API_BASE", DEFAULT_BASE_URL),
        "api_key": api_key,
        "model": _env("DISTIL_TEACHER_API_MODEL", DEFAULT_MODEL),
        "concurrency": int(_env("DISTIL_TEACHER_API_CONCURRENCY", str(DEFAULT_CONCURRENCY))),
        "timeout_s": int(_env("DISTIL_TEACHER_API_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))),
        "providers": providers,
    }


def _call_once(prompt: str, max_new_tokens: int, top_k: int, cfg: dict) -> TeacherOutput:
    """One chat-completions call returning a TeacherOutput."""
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": min(top_k, 20),
        "reasoning": {"enabled": False},
    }
    if cfg["providers"]:
        body["provider"] = {"order": list(cfg["providers"]), "allow_fallbacks": False}

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://distil.sn97.io",
        "X-Title": "distil-sn97-validator",
    }
    r = requests.post(
        f"{cfg['base_url']}/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=cfg["timeout_s"],
    )
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    text = choice["message"]["content"] or ""
    lp_payload = choice.get("logprobs") or {}
    content_tokens = lp_payload.get("content") or []

    # Tokenization detail: OpenRouter / OpenAI logprobs report tokens
    # as decoded strings (e.g. " the", "ization"), not int IDs. The
    # downstream KL/RKL pipeline keys on integer token IDs, so we hash
    # token strings into a deterministic dense ID space [0, 2**31).
    # This loses identity with the local tokenizer, BUT the comparison
    # is teacher-vs-student where BOTH sides use the same hashed IDs
    # (we hash the student's predicted-string tokens through the same
    # function before comparing). For top-K overlap & KL the only
    # property we need is consistent identity within a single round.
    token_ids: list[int] = []
    completion_logprobs: list[dict[int, float]] = []
    for entry in content_tokens:
        tok_str = entry.get("token", "")
        tok_lp = float(entry.get("logprob", 0.0) or 0.0)
        token_ids.append(_str_id(tok_str))
        per_pos: dict[int, float] = {_str_id(tok_str): tok_lp}
        for alt in entry.get("top_logprobs") or []:
            alt_str = alt.get("token", "")
            alt_lp = float(alt.get("logprob", 0.0) or 0.0)
            per_pos.setdefault(_str_id(alt_str), alt_lp)
        completion_logprobs.append(per_pos)

    return TeacherOutput(
        prompt=prompt,
        continuation=text,
        completion_token_ids=token_ids,
        completion_logprobs=completion_logprobs,
    )


def _str_id(s: str) -> int:
    """Deterministic non-negative 31-bit hash of a token string.

    Same hash function on both the teacher (here) and the student
    (distil.pod.student._encode_str_token) so KL / top-K overlap can
    key on a shared integer ID space when the underlying tokenizers
    differ between sides.
    """
    import hashlib

    h = hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def _call_with_retry(
    prompt: str, max_new_tokens: int, top_k: int, cfg: dict, max_attempts: int = 5
) -> TeacherOutput:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _call_once(prompt, max_new_tokens, top_k, cfg)
        except Exception as exc:
            last_exc = exc
            wait = min(2**attempt, 30)
            logger.warning(
                f"api teacher attempt {attempt + 1}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc} — retrying in {wait}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"api teacher exhausted retries: {last_exc}") from last_exc


def generate_continuations_api(
    prompts: list[str],
    *,
    max_new_tokens: int,
    top_k: int,
) -> list[TeacherOutput]:
    """Drop-in for ``teacher.generate_continuations`` running via API.

    Concurrent execution preserving prompt order in the output list.
    """
    cfg = _config()
    logger.info(
        f"teacher mode=api model={cfg['model']!r} concurrency={cfg['concurrency']} "
        f"providers={cfg['providers']!r}"
    )

    n = len(prompts)
    results: list[TeacherOutput | None] = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, cfg["concurrency"])) as ex:
        futures = {
            ex.submit(_call_with_retry, p, max_new_tokens, top_k, cfg): i
            for i, p in enumerate(prompts)
        }
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            completed += 1
            if completed % 10 == 0 or completed == n:
                logger.info(f"api teacher progress {completed}/{n}")

    out: list[TeacherOutput] = []
    for i, r in enumerate(results):
        if r is None:
            raise RuntimeError(f"api teacher missing result for prompt {i}")
        out.append(r)
    return out


__all__ = ["generate_continuations_api"]
