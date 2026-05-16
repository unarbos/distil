"""Cloud-API teacher path (OpenRouter / OpenAI-compatible).

The 1T-parameter Kimi-K2.6 teacher does NOT fit in vLLM on the
8xB200 pod (~500 GB usable VRAM vs ~2 TB needed in bf16), so the
production validator has run the teacher through OpenRouter for
months while keeping student scoring local. This module is the
distil-stack equivalent of ``scripts/api_teacher.py`` (669 LoC,
coupled to the legacy pod_eval_vllm sparse-tensor format).

Distil's KL/RKL/top-K scoring keys on ``dict[int, float]`` per
position with real Kimi vocab IDs. The OpenRouter API returns
tokens as decoded strings, so this module:

1. loads the local Kimi-K2.6 tokenizer once;
2. builds a ``token_to_id`` map (vocab entries + decoded fallback);
3. for each prompt: chat-completions call, then maps API token
   strings → vocab IDs using the same fallback chain the legacy
   ``vllm_logprobs_to_sparse`` used;
4. tokenizes ``prompt + continuation`` locally and slices off the
   prompt via ``_align_prompt_boundary`` so ``completion_token_ids``
   are EXACTLY the IDs the student's vLLM will see when it
   re-tokenizes the same full string — KL stays meaningful even
   when BPE merges across the prompt/continuation join.

Activated by ``DISTIL_TEACHER_MODE=api``. Required env:

* ``DISTIL_TEACHER_API_KEY`` (or ``OPENROUTER_API_KEY``)
* ``DISTIL_TEACHER_API_MODEL``  (default: ``moonshotai/kimi-k2.6``)
* ``DISTIL_TEACHER_API_BASE``   (default: ``https://openrouter.ai/api``)
* ``DISTIL_TEACHER_API_PROVIDERS``  comma list; default ``Inceptron``
  (the only OpenRouter provider that returns logprobs for K2.6)
* ``DISTIL_TEACHER_REPO``       (default: ``moonshotai/Kimi-K2.6``)
  — used to load the local tokenizer for vocab-ID alignment.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from distil.pod.teacher import TeacherOutput

logger = logging.getLogger("distil.pod.teacher_api")

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_TOKENIZER_REPO = "moonshotai/Kimi-K2.6"
DEFAULT_CONCURRENCY = 12
DEFAULT_TIMEOUT_S = 120
DEFAULT_PROVIDERS = ("Inceptron",)

_TOKENIZER = None
_TOKEN_TO_ID: dict[str, int] | None = None
_LOCK = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _config() -> dict:
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
        "tokenizer_repo": _env("DISTIL_TEACHER_REPO", DEFAULT_TOKENIZER_REPO),
        "concurrency": int(_env("DISTIL_TEACHER_API_CONCURRENCY", str(DEFAULT_CONCURRENCY))),
        "timeout_s": int(_env("DISTIL_TEACHER_API_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))),
        "providers": providers,
    }


def _load_tokenizer(repo: str):
    """Lazy-load + cache the local Kimi tokenizer (thread-safe).

    The tokenizer is the same one the student vLLM uses to tokenize
    ``prompt + continuation``. Sharing it here is what keeps teacher
    vocab IDs aligned with the IDs the student sees, so KL is meaningful.
    """
    global _TOKENIZER, _TOKEN_TO_ID
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _LOCK:
        if _TOKENIZER is not None:
            return _TOKENIZER
        from transformers import AutoTokenizer

        logger.info(f"loading tokenizer for vocab-ID alignment: {repo}")
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        vocab = tok.get_vocab()
        mapping: dict[str, int] = {}
        for tok_str, tok_id in vocab.items():
            mapping[tok_str] = tok_id
            decoded = tok.decode([tok_id])
            mapping.setdefault(decoded, tok_id)
        _TOKENIZER = tok
        _TOKEN_TO_ID = mapping
        logger.info(f"tokenizer loaded: {len(vocab)} vocab entries, {len(mapping)} keys w/ decoded fallback")
        return _TOKENIZER


def _str_to_vocab_id(token_str: str) -> int | None:
    """API token string → Kimi vocab ID, with ``encode()`` fallback.

    Returns ``None`` when neither the direct vocab lookup nor the
    encode-fallback maps the API-returned string to a valid token id.
    Pre-fix, this returned ``0`` (the BOS / pad id), which silently
    piled logprob mass at id 0 in the teacher distribution. That
    mass intersects with student id 0 too often to be noise — it
    inflated KL on some prompts and broke top-K overlap on others.
    Callers must now skip ``None`` entries rather than persist them.
    """
    assert _TOKEN_TO_ID is not None
    tid = _TOKEN_TO_ID.get(token_str)
    if tid is not None:
        return int(tid)
    assert _TOKENIZER is not None
    try:
        encoded = _TOKENIZER.encode(token_str, add_special_tokens=False)
        return int(encoded[0]) if encoded else None
    except Exception:
        return None


def _align_prompt_boundary(full_text: str, prompt_text: str, full_ids) -> int:
    """Return the index in ``full_ids`` that separates prompt from continuation.

    Ported from ``scripts/pod_eval_vllm._align_prompt_boundary``. BPE
    tokenizers can merge characters across the prompt/continuation
    join (e.g. "Hello " + "world" → ["Hello", " world"]), so a naive
    ``len(tokenizer(prompt))`` drifts from the true boundary inside
    ``tokenizer(prompt + cont)``. Wrong boundary leaks prompt tokens
    into the KL slice (or vice versa) and inflates KL on the
    mismatched side.
    """
    prompt_char_len = len(prompt_text)
    if prompt_char_len == 0:
        return 0
    n_full = full_ids.shape[1]
    if prompt_char_len >= len(full_text):
        return n_full
    try:
        assert _TOKENIZER is not None
        enc = _TOKENIZER(
            full_text,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"][0].tolist()
        last_prompt_tok = None
        for k, (start, end) in enumerate(offsets):
            if end <= prompt_char_len:
                last_prompt_tok = k
            elif start >= prompt_char_len:
                break
            else:
                break
        return 0 if last_prompt_tok is None else last_prompt_tok + 1
    except Exception:
        ids_list = full_ids[0].tolist()
        assert _TOKENIZER is not None
        for k in range(1, len(ids_list) + 1):
            try:
                decoded = _TOKENIZER.decode(ids_list[:k], skip_special_tokens=False)
            except Exception:
                continue
            if decoded == prompt_text:
                return k
            if len(decoded) > prompt_char_len:
                return max(k - 1, 0)
        return len(ids_list)


def _call_once(prompt: str, max_new_tokens: int, top_k: int, cfg: dict) -> TeacherOutput:
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "logprobs": True,
        # OpenRouter caps top_logprobs at 20.
        "top_logprobs": min(max(top_k, 1), 20),
    }
    if cfg["providers"]:
        body["provider"] = {"only": list(cfg["providers"])}
    if "openrouter" in cfg["base_url"]:
        # Raw next-token logprobs (skip K2.6's reasoning loop).
        body["reasoning"] = {"enabled": False}

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    if "openrouter" in cfg["base_url"]:
        headers["HTTP-Referer"] = _env(
            "DISTIL_OPENROUTER_REFERER", "https://chat.arbos.life"
        )
        headers["X-Title"] = _env(
            "DISTIL_OPENROUTER_TITLE", "Bittensor SN97 distil validator"
        )

    r = requests.post(
        f"{cfg['base_url']}/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=cfg["timeout_s"],
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"API error: {data['error']}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    # K2.6 sometimes returns ``content: null`` if reasoning ate the
    # token budget — treat as empty continuation rather than crash.
    cont_text = msg.get("content") or ""
    lp_payload = choice.get("logprobs") or {}
    content_tokens = lp_payload.get("content") or []

    full_text = prompt + cont_text
    enc = _TOKENIZER(full_text, return_tensors="pt", truncation=False)  # type: ignore[misc]
    full_ids = enc.input_ids
    prompt_len = _align_prompt_boundary(full_text, prompt, full_ids)
    completion_token_ids = full_ids[0][prompt_len:].tolist()

    per_pos_logprobs: list[dict[int, float]] = []
    for entry in content_tokens:
        chosen = entry.get("token", "")
        chosen_lp = entry.get("logprob")
        d: dict[int, float] = {}
        if chosen_lp is not None and chosen != "":
            cid = _str_to_vocab_id(chosen)
            if cid is not None:
                d[cid] = float(chosen_lp)
        for alt in entry.get("top_logprobs") or []:
            t = alt.get("token", "")
            lp = alt.get("logprob")
            if lp is not None and t != "":
                aid = _str_to_vocab_id(t)
                if aid is not None:
                    d.setdefault(aid, float(lp))
        per_pos_logprobs.append(d)

    # Length harmonization: KL keys on min(len) positions, but mid-stream
    # nan or empty positions are fine — distil.eval.scoring tolerates
    # missing keys per-position.
    return TeacherOutput(
        prompt=prompt,
        continuation=cont_text,
        completion_token_ids=[int(x) for x in completion_token_ids],
        completion_logprobs=per_pos_logprobs,
    )


def _call_with_retry(
    prompt: str, max_new_tokens: int, top_k: int, cfg: dict, max_attempts: int = 6
) -> TeacherOutput:
    """Retry transient 429/5xx with exp backoff, honoring Retry-After.

    Non-transient HTTP errors (e.g. ``401`` unauthorized, ``403``
    forbidden, ``404`` not-found, ``422`` unprocessable) raise
    *immediately* — exp-backoffing those just wastes wall time and
    drowns the logs in spurious retry warnings. The model name in the
    body is the only knob that changes between rounds; if the
    OpenRouter / Inceptron auth is wrong or the model id was retired,
    the next 5 retries will fail too.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _call_once(prompt, max_new_tokens, top_k, cfg)
        except requests.HTTPError as exc:
            last_exc = exc
            sc = getattr(exc.response, "status_code", None)
            transient = sc == 429 or (sc is not None and 500 <= sc < 600)
            if not transient:
                raise
            if attempt >= max_attempts - 1:
                raise
            retry_after = None
            try:
                ra_hdr = exc.response.headers.get("Retry-After")
                if ra_hdr:
                    retry_after = float(ra_hdr)
            except Exception:
                retry_after = None
            wait = max(min(retry_after or (2**attempt), 30), 1)
            logger.warning(
                f"api teacher HTTP {sc} (attempt {attempt + 1}/{max_attempts}); "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = min(2**attempt, 30)
                logger.warning(
                    f"api teacher attempt {attempt + 1}/{max_attempts} failed: "
                    f"{type(exc).__name__}: {exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"api teacher exhausted retries: {last_exc}") from last_exc


# Outer retry loop: when OpenRouter's Inceptron route hits a sustained
# 429 storm (which happens during peak hours on the kimi-k2.6 endpoint),
# the per-prompt 6-attempt × ~30s backoff isn't enough — every prompt in
# the batch can fail simultaneously after exhausting its inner budget.
# Legacy ``scripts/api_teacher.generate_via_api`` solved this with a
# second-tier outer retry pass: collect every failure from the main
# fan-out, then sequentially re-issue them with multi-minute cool-downs
# between passes. This buys ~12 min of additional stall budget per
# prompt without thrashing the API. We mirror that behaviour here.
_OUTER_RETRY_PASSES = 5
_OUTER_RETRY_COOLDOWNS_S = (60, 90, 120, 180, 300)


def generate_continuations_api(
    prompts: list[str],
    *,
    max_new_tokens: int,
    top_k: int,
) -> list[TeacherOutput]:
    """Drop-in for ``teacher.generate_continuations`` running via API.

    Loads the local tokenizer once (vocab-ID alignment), then fans out
    one chat-completions call per prompt across a ThreadPoolExecutor.
    Output order matches input order; raises on any unrecoverable prompt
    (no silent holes — the round is poisoned if even one prompt is
    missing, which is detectable at the validator and triggers a
    deterministic retry rather than a corrupted leaderboard write).

    Two-tier retry strategy:
    * **Inner** (``_call_with_retry``) — per-prompt 6 attempts with
      exp-backoff (1s → 30s), honoring ``Retry-After`` on transient
      429/5xx. Non-transient HTTP (401/404/422) raises immediately.
    * **Outer** (this function) — after the main fan-out, collect every
      prompt that failed and re-issue it across up to 5 sequential
      passes with 60s → 300s cool-downs between passes. Buys ~12 min of
      additional stall budget per prompt vs. a single-tier retry, which
      is what we needed to survive the 2026-05-16 OpenRouter outage
      that killed round 1778905724 Phase 1.
    """
    cfg = _config()
    _load_tokenizer(cfg["tokenizer_repo"])
    logger.info(
        f"teacher mode=api model={cfg['model']!r} concurrency={cfg['concurrency']} "
        f"providers={cfg['providers']!r}"
    )

    n = len(prompts)
    results: list[TeacherOutput | None] = [None] * n
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, cfg["concurrency"])) as ex:
        futures = {
            ex.submit(_call_with_retry, p, max_new_tokens, top_k, cfg): i
            for i, p in enumerate(prompts)
        }
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
                completed += 1
                if completed % 10 == 0 or completed == n:
                    logger.info(f"api teacher progress {completed}/{n}")
            except Exception as exc:
                failed.append((idx, f"{type(exc).__name__}: {exc}"))
                logger.warning(
                    f"api teacher prompt {idx} failed after inner retries: "
                    f"{type(exc).__name__}: {exc} (deferred to outer retry)"
                )

    # Outer retry pass for any prompts that failed during the fan-out.
    if failed:
        logger.warning(
            f"api teacher main fan-out completed with {len(failed)} failed prompt(s); "
            f"entering outer retry (up to {_OUTER_RETRY_PASSES} passes)"
        )
        for pass_idx in range(_OUTER_RETRY_PASSES):
            if not failed:
                break
            logger.info(
                f"api teacher outer retry pass {pass_idx + 1}/{_OUTER_RETRY_PASSES} "
                f"on {len(failed)} failed prompt(s)"
            )
            still_failed: list[tuple[int, str]] = []
            for idx, _err in failed:
                try:
                    results[idx] = _call_with_retry(
                        prompts[idx], max_new_tokens, top_k, cfg
                    )
                    completed += 1
                    logger.info(
                        f"api teacher outer retry prompt {idx}: OK "
                        f"({completed}/{n} total)"
                    )
                except Exception as exc:
                    still_failed.append(
                        (idx, f"{type(exc).__name__}: {exc}")
                    )
                    logger.warning(
                        f"api teacher outer retry pass {pass_idx + 1} "
                        f"prompt {idx} failed: {type(exc).__name__}: {exc}"
                    )
            failed = still_failed
            if failed and pass_idx < _OUTER_RETRY_PASSES - 1:
                cool_s = _OUTER_RETRY_COOLDOWNS_S[pass_idx]
                logger.warning(
                    f"api teacher {len(failed)} prompt(s) still failing — "
                    f"cooling {cool_s}s before next pass (likely sustained 429 storm)"
                )
                time.sleep(cool_s)
        if failed:
            first_idx, first_err = failed[0]
            raise RuntimeError(
                f"api teacher: {len(failed)} prompt(s) failed after "
                f"{_OUTER_RETRY_PASSES} outer retry passes "
                f"(first failure: prompt {first_idx}: {first_err})"
            )

    out: list[TeacherOutput] = []
    for i, r in enumerate(results):
        if r is None:
            raise RuntimeError(f"api teacher missing result for prompt {i}")
        out.append(r)
    return out


# ── greedy text-only API grading (Phase 3 judge path) ─────────────────


def _greedy_text_once(prompt: str, max_new_tokens: int, cfg: dict) -> str:
    """One chat-completions call returning text only.

    Used by Phase 3 (judge) rubric grading where we only need a single
    digit / short response from the teacher — no logprobs, no
    tokenization alignment. This mirrors
    ``scripts.api_teacher._greedy_text_one`` from the legacy stack.
    """
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    if cfg["providers"]:
        body["provider"] = {"only": list(cfg["providers"])}
    if "openrouter" in cfg["base_url"]:
        body["reasoning"] = {"enabled": False}

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    if "openrouter" in cfg["base_url"]:
        headers["HTTP-Referer"] = _env(
            "DISTIL_OPENROUTER_REFERER", "https://chat.arbos.life"
        )
        headers["X-Title"] = _env(
            "DISTIL_OPENROUTER_TITLE", "Bittensor SN97 distil validator"
        )

    r = requests.post(
        f"{cfg['base_url']}/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=cfg["timeout_s"],
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"API error: {data['error']}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return msg.get("content") or ""


def _greedy_text_with_retry(
    prompt: str,
    max_new_tokens: int,
    cfg: dict,
    max_attempts: int = 6,
) -> str:
    """Same exp-backoff retry envelope as :func:`_call_with_retry`.

    Non-transient HTTP errors raise immediately (see that function's
    docstring for rationale).
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _greedy_text_once(prompt, max_new_tokens, cfg)
        except requests.HTTPError as exc:
            last_exc = exc
            sc = getattr(exc.response, "status_code", None)
            transient = sc == 429 or (sc is not None and 500 <= sc < 600)
            if not transient:
                raise
            if attempt >= max_attempts - 1:
                raise
            retry_after = None
            try:
                ra_hdr = exc.response.headers.get("Retry-After")
                if ra_hdr:
                    retry_after = float(ra_hdr)
            except Exception:
                retry_after = None
            wait = max(min(retry_after or (2**attempt), 30), 1)
            logger.warning(
                f"api judge HTTP {sc} (attempt {attempt + 1}/{max_attempts}); "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = min(2**attempt, 30)
                logger.warning(
                    f"api judge attempt {attempt + 1}/{max_attempts} failed: "
                    f"{type(exc).__name__}: {exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"api judge exhausted retries: {last_exc}") from last_exc


def greedy_batch_api(
    prompts: list[str],
    *,
    max_new_tokens: int = 8,
    concurrency: int | None = None,
) -> list[str]:
    """Greedy text-only batch generation via the OpenAI-compatible API.

    Phase 3 judge / long-form / chat-turns rubric grading routes through
    this when ``DISTIL_TEACHER_MODE=api`` because Kimi-K2.6 cannot fit
    on the 8xB200 pod. The rubrics return a single digit (1-5) or a
    short verdict, so we cap ``max_new_tokens`` at 8 by default and fan
    out with ``min(concurrency, len(prompts))`` workers.

    Concurrency defaults to ``DISTIL_TEACHER_API_JUDGE_CONCURRENCY``
    (env), falling back to ``max(1, teacher_concurrency // 2)`` so that
    the burst stays under the Inceptron rate-limit bucket — matching
    the legacy 2026-05-04 fix that solved the same 429-storm issue
    when probe batches followed Phase 1 sustained traffic. Failed
    prompts return ``""`` (the probe's ``_extract_score`` then returns
    ``None`` and the prompt drops out of the mean) rather than poisoning
    the whole round — judge axes are advisory, not gating.
    """
    cfg = _config()
    n = len(prompts)
    if n == 0:
        return []
    if concurrency is None:
        env_c = _env("DISTIL_TEACHER_API_JUDGE_CONCURRENCY")
        if env_c:
            try:
                concurrency = int(env_c)
            except ValueError:
                concurrency = None
        if concurrency is None:
            concurrency = max(1, cfg["concurrency"] // 2)
    concurrency = max(1, min(int(concurrency), n))

    logger.info(
        f"api judge model={cfg['model']!r} prompts={n} concurrency={concurrency}"
    )
    out: list[str] = [""] * n
    if concurrency == 1:
        for i, p in enumerate(prompts):
            try:
                out[i] = _greedy_text_with_retry(p, max_new_tokens, cfg)
            except Exception as exc:
                logger.warning(f"api judge prompt {i} failed: {exc}")
                out[i] = ""
        return out

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_greedy_text_with_retry, p, max_new_tokens, cfg): i
            for i, p in enumerate(prompts)
        }
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:
                logger.warning(f"api judge prompt {i} failed: {exc}")
                out[i] = ""
            completed += 1
            if completed % 8 == 0 or completed == n:
                logger.info(f"api judge progress {completed}/{n}")
    return out


__all__ = ["generate_continuations_api", "greedy_batch_api"]
