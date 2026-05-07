"""Cloud-API teacher generation path for SN97 evaluation.

Why this exists
===============
Loading the 1T-param Kimi K2.6 teacher onto a Lium pod has two real costs:

* **VRAM** — even FP8/INT8, the model needs ≥600 GB across 8×H200, which means
  the eval pod has to be a "tensor-parallel monster" before it can even start.
* **Cold-start latency** — vLLM startup on Kimi K2.6 takes ~6 minutes per
  attempt, plus Lium provisioning if we don't already have a hot pod. Every
  vLLM crash burns more wall clock.

For our subnet's purposes we don't need to *run* the teacher; we only need
its logprob distribution at every generated token so we can compute KL
divergence against student outputs. Cloud inference providers (OpenRouter,
Moonshot direct, Cloudflare Workers AI, Together AI) expose that exact
information via OpenAI-compatible ``logprobs`` + ``top_logprobs`` (≤20).
So we can replace the local teacher with an API call and free the pod's GPU
budget for student scoring (which we *cannot* outsource — students are
unique per-miner and do need local FP forward passes).

Key trade-off: cloud APIs cap ``top_logprobs`` at 20 vs vLLM's 128. For our
KL signal that's fine — top-20 captures >99% of the teacher's distribution
mass on confident tokens, and ranking signal between miners is preserved
because the same approximation applies to every student.

Public API
==========
* :func:`generate_via_api` — drop-in replacement for ``generate_via_vllm``
  in :mod:`scripts.pod_eval_vllm`. Returns the same
  ``[{"full_ids", "prompt_len", "gen_len", "sparse_logprobs"}, ...]``
  shape, so downstream Phase-2 student scoring is unchanged.
* :func:`prepare_teacher_probe_refs_api` — drop-in replacement for
  ``prepare_teacher_probe_refs_vllm``. Greedy text-only (no logprobs),
  used for calibration probes (think samples / capability answers /
  chat-probe gen lens).
* :func:`api_health_check` — quick sanity test before the eval round
  commits to the API path.

Provider notes
==============
The default provider is OpenRouter (it confirmed K2.6 availability and
``top_logprobs`` support) but any OpenAI-compatible endpoint works:

  ============= ============================== ===================== =====
  Provider       Base URL                       Model ID              Logp
  ============= ============================== ===================== =====
  OpenRouter     https://openrouter.ai/api      moonshotai/kimi-k2.6  ≤20
  Moonshot       https://api.moonshot.ai        kimi-k2.6             ≤20
  Cloudflare     https://api.cloudflare.com/... @cf/moonshotai/...    ≤20
  ============= ============================== ===================== =====

Configure via environment variables:

* ``DISTIL_TEACHER_API_BASE`` — defaults to ``https://openrouter.ai/api``
* ``DISTIL_TEACHER_API_KEY``  — required
* ``DISTIL_TEACHER_API_MODEL`` — defaults to ``moonshotai/kimi-k2.6``
* ``DISTIL_TEACHER_API_ENDPOINT`` — ``chat`` (default) or ``completions``
* ``DISTIL_TEACHER_API_CONCURRENCY`` — parallel requests (default 8)
* ``DISTIL_TEACHER_API_TOP_LOGPROBS`` — top-K per position (default 20)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_ENDPOINT = "chat"
DEFAULT_CONCURRENCY = 12
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_TIMEOUT_S = 120
# OpenRouter routes through 14 providers for kimi-k2.6 — but only ONE
# (Inceptron, int4) actually exposes ``logprobs``. The rest silently
# return ``logprobs: null``, which would zero our KL signal. Pin the
# provider on OpenRouter unless DISTIL_TEACHER_API_PROVIDERS overrides.
# See ``GET /api/v1/models/{id}/endpoints`` for the live list.
DEFAULT_OPENROUTER_PROVIDERS = ("Inceptron",)


class EmptyLogprobsError(RuntimeError):
    """Provider returned HTTP 200 but no per-token logprobs."""


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model: str
    endpoint: str  # "chat" | "completions"
    top_logprobs: int
    concurrency: int
    timeout_s: int
    # Provider-routing hint — only meaningful for OpenRouter. Empty list
    # = let OpenRouter pick freely. Names must match
    # ``provider_name`` in the OpenRouter endpoints API.
    providers: Tuple[str, ...] = ()
    # Disable Kimi's "thinking" reasoning loop. K2.6 defaults to
    # thinking-on, which (a) burns ``reasoning_tokens`` against
    # ``max_tokens`` before any actual content is emitted, and (b)
    # produces ``content: null`` if the budget runs out mid-thought.
    # For SN97's KL-based eval we want pure instant-mode raw next-token
    # distributions, not chain-of-thought.
    disable_reasoning: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "APIConfig":
        env = os.environ.get
        base_url = overrides.get("base_url") or env("DISTIL_TEACHER_API_BASE", DEFAULT_BASE_URL)
        # ``DISTIL_TEACHER_API_KEY`` is the canonical name. We also accept
        # ``OPENROUTER_API_KEY`` for OpenRouter-default deployments since
        # that is the env var name OpenRouter's own examples use.
        api_key = (
            overrides.get("api_key")
            or env("DISTIL_TEACHER_API_KEY")
            or env("OPENROUTER_API_KEY", "")
        )
        model = overrides.get("model") or env("DISTIL_TEACHER_API_MODEL", DEFAULT_MODEL)
        endpoint = overrides.get("endpoint") or env("DISTIL_TEACHER_API_ENDPOINT", DEFAULT_ENDPOINT)
        top_lp = overrides.get("top_logprobs") or int(env("DISTIL_TEACHER_API_TOP_LOGPROBS", DEFAULT_TOP_LOGPROBS) or DEFAULT_TOP_LOGPROBS)
        conc = overrides.get("concurrency") or int(env("DISTIL_TEACHER_API_CONCURRENCY", DEFAULT_CONCURRENCY) or DEFAULT_CONCURRENCY)
        timeout = overrides.get("timeout_s") or int(env("DISTIL_TEACHER_API_TIMEOUT_S", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)

        # Provider list: comma-separated env var, e.g. "Inceptron" or
        # "Inceptron,DeepInfra". Empty string disables pinning.
        providers_env = env("DISTIL_TEACHER_API_PROVIDERS")
        if "providers" in overrides:
            providers = tuple(overrides["providers"]) if overrides["providers"] else ()
        elif providers_env is not None:
            providers = tuple(p.strip() for p in providers_env.split(",") if p.strip())
        elif "openrouter" in base_url:
            providers = DEFAULT_OPENROUTER_PROVIDERS
        else:
            providers = ()

        # Reasoning toggle. Default OFF for the API-teacher KL path. The
        # provider often omits visible content/logprobs when it spends the
        # budget in hidden reasoning, which wastes calls and shrinks the KL
        # prompt set. Chat/probe quality is tested separately in pod_eval.py.
        dr_env = env("DISTIL_TEACHER_API_DISABLE_REASONING", "1")
        disable_reasoning = overrides.get("disable_reasoning",
                                          dr_env not in ("0", "false", "False", "no", ""))

        if endpoint not in ("chat", "completions"):
            raise ValueError(f"DISTIL_TEACHER_API_ENDPOINT must be 'chat' or 'completions' (got {endpoint!r})")
        if not api_key:
            raise ValueError(
                "DISTIL_TEACHER_API_KEY (or OPENROUTER_API_KEY) is not set. "
                "Configure your inference provider's API key (OpenRouter / "
                "Moonshot / etc.) in the validator's environment file."
            )
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            top_logprobs=max(1, min(top_lp, 20)),  # API cap
            concurrency=max(1, conc),
            timeout_s=max(15, timeout),
            providers=providers,
            disable_reasoning=disable_reasoning,
        )

    def headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter recommends adding referer/title headers for analytics &
        # rate-limit tier; harmless to other providers (they ignore unknowns).
        if "openrouter" in self.base_url:
            h["HTTP-Referer"] = os.environ.get("DISTIL_OPENROUTER_REFERER", "https://chat.arbos.life")
            h["X-Title"] = os.environ.get("DISTIL_OPENROUTER_TITLE", "Bittensor SN97 distil validator")
        return h

    def extra_body(self) -> Dict[str, Any]:
        """Provider-routing + reasoning hints to merge into chat/completions payloads."""
        body: Dict[str, Any] = {}
        if self.providers and "openrouter" in self.base_url:
            body["provider"] = {"only": list(self.providers)}
        if self.disable_reasoning and "openrouter" in self.base_url:
            # OpenRouter's normalized field; underlying providers ignore
            # if they don't support reasoning.
            body["reasoning"] = {"enabled": False}
        return body


# ---------------------------------------------------------------------------
# Response → top_logprobs list-of-dicts conversion
# ---------------------------------------------------------------------------

def _chat_logprobs_to_dict_list(content_lp: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Convert chat-completions logprobs format to vLLM-compatible list-of-dicts.

    Chat completions API returns:
        logprobs.content = [
            {"token": "Hello", "logprob": -0.123,
             "top_logprobs": [{"token": "Hello", "logprob": -0.12}, ...]},
            ...
        ]

    We want the same shape vLLM/text-completions returns:
        [{"Hello": -0.12, "Hi": -2.5, ...}, ...]
    """
    out = []
    for tok_entry in content_lp or []:
        d: Dict[str, float] = {}
        # The chosen token's own logprob
        chosen = tok_entry.get("token", "")
        chosen_lp = tok_entry.get("logprob")
        if chosen_lp is not None and chosen != "":
            d[chosen] = float(chosen_lp)
        # All siblings in top_logprobs
        for sib in tok_entry.get("top_logprobs", []) or []:
            t = sib.get("token", "")
            lp = sib.get("logprob")
            if lp is not None and t != "":
                d[t] = float(lp)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Single-prompt API call
# ---------------------------------------------------------------------------

def _call_api_chat(
    prompt_text: str,
    cfg: APIConfig,
    max_new_tokens: int,
    block_seed: Optional[int],
    idx: int,
    request: "Any",  # requests module
) -> Dict[str, Any]:
    """Call /v1/chat/completions. Returns dict with text, top_logprobs_list."""
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if block_seed is not None else 0.0,
        "top_p": 0.9 if block_seed is not None else 1.0,
        "logprobs": True,
        "top_logprobs": cfg.top_logprobs,
    }
    if block_seed is not None:
        payload["seed"] = block_seed + idx
    payload.update(cfg.extra_body())
    resp = request.post(
        f"{cfg.base_url}/v1/chat/completions",
        headers=cfg.headers(),
        data=json.dumps(payload),
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"API error: {data['error']}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    # K2.6 emits ``content: null`` if the request landed in
    # reasoning-mode and ``max_tokens`` was eaten by the chain-of-
    # thought before any visible content. With ``disable_reasoning``
    # this should be rare, but guard anyway and surface as empty.
    text = msg.get("content") or ""
    lp_field = choice.get("logprobs") or {}
    content_lp = lp_field.get("content") or []
    top_lp_list = _chat_logprobs_to_dict_list(content_lp)
    return {"text": text, "top_logprobs_list": top_lp_list, "raw": choice}


def _call_api_completions(
    prompt_text: str,
    cfg: APIConfig,
    max_new_tokens: int,
    block_seed: Optional[int],
    idx: int,
    request: "Any",
) -> Dict[str, Any]:
    """Call /v1/completions (legacy text completion). Returns dict with text, top_logprobs_list."""
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if block_seed is not None else 0.0,
        "top_p": 0.9 if block_seed is not None else 1.0,
        "logprobs": cfg.top_logprobs,
    }
    if block_seed is not None:
        payload["seed"] = block_seed + idx
    payload.update(cfg.extra_body())
    resp = request.post(
        f"{cfg.base_url}/v1/completions",
        headers=cfg.headers(),
        data=json.dumps(payload),
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"API error: {data['error']}")
    choice = data["choices"][0]
    text = choice.get("text") or ""
    lp_field = choice.get("logprobs") or {}
    top_lp_list = lp_field.get("top_logprobs") or []
    return {"text": text, "top_logprobs_list": top_lp_list, "raw": choice}


# ---------------------------------------------------------------------------
# Per-prompt worker
# ---------------------------------------------------------------------------

def _generate_single_prompt_api(
    idx: int,
    prompt_text: str,
    max_new_tokens: int,
    block_seed: Optional[int],
    cfg: APIConfig,
    tokenizer,
    token_to_id: Optional[Dict[str, int]],
    sparse_converter: Callable,
    align_prompt_boundary: Callable,
) -> Tuple[int, Dict[str, Any]]:
    """Worker: one API call → one sequence_data dict.

    Args:
        sparse_converter: function (top_lp_list, token_to_id, tokenizer, k) ->
            sparse-tensor dict {indices, values}. We pass it in to avoid a
            circular import with pod_eval_vllm.
        align_prompt_boundary: function (full_text, prompt_text, full_ids,
            tokenizer) -> int. Same reason.
    """
    import requests

    last_err: Optional[Exception] = None
    # 6 attempts with exp backoff covers the worst Inceptron-side rate-limit
    # bursts we've observed (60 prompts × concurrency=4 occasionally hits a
    # cluster of 4-8 consecutive 429s when a coincident user spikes the
    # provider's queue). 1+2+4+8+16+30 = ~61 s of budget per prompt before
    # we give up; far cheaper than a 5 min vLLM cold-start.
    for attempt in range(6):
        try:
            if cfg.endpoint == "chat":
                api_out = _call_api_chat(prompt_text, cfg, max_new_tokens, block_seed, idx, requests)
            else:
                api_out = _call_api_completions(prompt_text, cfg, max_new_tokens, block_seed, idx, requests)

            cont_text = api_out["text"]
            full_text = prompt_text + cont_text

            full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
            prompt_len = align_prompt_boundary(full_text, prompt_text, full_ids, tokenizer)

            result: Dict[str, Any] = {
                "full_ids": full_ids,
                "prompt_len": prompt_len,
                "gen_len": full_ids.shape[1] - prompt_len,
            }

            top_lp_list = api_out["top_logprobs_list"]
            if token_to_id is not None and not top_lp_list:
                last_err = EmptyLogprobsError(
                    "API returned no top_logprobs; retrying as transient provider miss"
                )
                if attempt < 5:
                    time.sleep(min(2.0 ** attempt, 30.0))
                    continue
                result["api_missing_logprobs"] = True
                result["api_missing_logprobs_reason"] = "empty_top_logprobs_after_retries"
                return idx, result
            if top_lp_list and token_to_id is not None:
                result["sparse_logprobs"] = sparse_converter(
                    top_lp_list, token_to_id, tokenizer, k=cfg.top_logprobs,
                )
            return idx, result
        except requests.HTTPError as e:
            last_err = e
            sc = getattr(e.response, "status_code", None)
            is_rate_limited = sc == 429 or (sc is not None and 500 <= sc < 600)
            if is_rate_limited and attempt < 5:
                retry_after_hdr = e.response.headers.get("Retry-After") if e.response is not None else None
                try:
                    backoff_s = float(retry_after_hdr) if retry_after_hdr else (2.0 ** attempt)
                except Exception:
                    backoff_s = 2.0 ** attempt
                backoff_s = min(backoff_s, 30.0)
                # Provider-assigned Retry-After can be 0; force at least 1 s
                # so we don't immediately retry into the same closed window.
                backoff_s = max(backoff_s, 1.0)
                time.sleep(backoff_s)
                continue
            if attempt < 5:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise RuntimeError(f"API generation failed for prompt {idx}: HTTP {sc}") from e
        except Exception as e:
            last_err = e
            if attempt < 5:
                time.sleep(min(2.0 ** attempt, 30.0))
            else:
                raise RuntimeError(f"API generation failed for prompt {idx}: {e}") from e
    # unreachable
    raise RuntimeError(f"API generation failed for prompt {idx}: {last_err}")


# ---------------------------------------------------------------------------
# Public entry: generate_via_api
# ---------------------------------------------------------------------------

def generate_via_api(
    prompts: List[str],
    tokenizer,
    max_new_tokens: int,
    block_seed: Optional[int] = None,
    logprobs_k: int = 20,
    token_to_id: Optional[Dict[str, int]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    concurrency: Optional[int] = None,
    sparse_converter: Optional[Callable] = None,
    align_prompt_boundary: Optional[Callable] = None,
    config: Optional[APIConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate teacher continuations + top-K logprobs via cloud inference API.

    Drop-in replacement for ``generate_via_vllm``. Same return shape:
    list of dicts with keys ``full_ids``, ``prompt_len``, ``gen_len``, and
    optionally ``sparse_logprobs`` ({"indices": tensor, "values": tensor}).

    Args:
        prompts: pre-rendered prompt strings (already chat-templated).
        tokenizer: HF tokenizer matching the teacher (Kimi BPE).
        max_new_tokens: max output tokens.
        block_seed: deterministic-block seed; None = greedy.
        logprobs_k: requested top-K. Will be capped at the API max (20).
            If 0, no logprobs requested.
        token_to_id: pre-built {token_str: token_id} (from
            ``_build_token_to_id_map`` in pod_eval_vllm).
        progress_cb: callback(done, total) for live progress reporting.
        concurrency: parallel API requests; defaults to env / 8.
        sparse_converter: pod_eval_vllm.vllm_logprobs_to_sparse, passed in
            to avoid a circular import.
        align_prompt_boundary: pod_eval_vllm._align_prompt_boundary.
        config: optional pre-built APIConfig.
    """
    if sparse_converter is None or align_prompt_boundary is None:
        raise ValueError(
            "generate_via_api requires sparse_converter and "
            "align_prompt_boundary callables (passed in by pod_eval_vllm)"
        )

    overrides: Dict[str, Any] = {}
    if logprobs_k:
        overrides["top_logprobs"] = min(logprobs_k, 20)
    if concurrency:
        overrides["concurrency"] = concurrency
    cfg = config or APIConfig.from_env(**overrides)

    print(
        f"  [api] {cfg.model} via {cfg.base_url} "
        f"(endpoint={cfg.endpoint} concurrency={cfg.concurrency} "
        f"top_logprobs={cfg.top_logprobs})",
        flush=True,
    )

    n = len(prompts)
    result_slots: List[Optional[Dict[str, Any]]] = [None] * n
    completed = 0
    last_log = 0

    if cfg.concurrency <= 1:
        for idx, prompt_text in enumerate(prompts):
            _, result = _generate_single_prompt_api(
                idx, prompt_text, max_new_tokens, block_seed, cfg,
                tokenizer, token_to_id, sparse_converter, align_prompt_boundary,
            )
            result_slots[idx] = result
            completed += 1
            if completed - last_log >= 10 or completed == n:
                has_lp = "sparse_logprobs" in result
                print(
                    f"  [{completed}/{n}] latest: {result['prompt_len']}+{result['gen_len']} tokens"
                    f"{' (logprobs✓)' if has_lp else ''}",
                    flush=True,
                )
                last_log = completed
            if progress_cb:
                progress_cb(completed, n)
        return [r for r in result_slots if r is not None]  # type: ignore[return-value]

    # Concurrent path
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
        futures = {}
        for idx, prompt_text in enumerate(prompts):
            fut = executor.submit(
                _generate_single_prompt_api, idx, prompt_text, max_new_tokens,
                block_seed, cfg, tokenizer, token_to_id,
                sparse_converter, align_prompt_boundary,
            )
            futures[fut] = idx

        failed = []
        for fut in as_completed(futures):
            orig_idx = futures[fut]
            try:
                _, result = fut.result()
                result_slots[orig_idx] = result
                completed += 1
                if completed - last_log >= 10 or completed == n:
                    has_lp = "sparse_logprobs" in result
                    print(
                        f"  [{completed}/{n}] latest: {result['prompt_len']}+{result['gen_len']} tokens"
                        f"{' (logprobs✓)' if has_lp else ''}",
                        flush=True,
                    )
                    last_log = completed
                if progress_cb:
                    progress_cb(completed, n)
            except Exception as e:
                failed.append((orig_idx, str(e)))
                print(f"  [api] Prompt {orig_idx} failed: {e}", flush=True)

    # Retry failures sequentially
    if failed:
        print(f"  [api] Retrying {len(failed)} failed prompts sequentially...", flush=True)
        for idx, _err in failed:
            try:
                _, result = _generate_single_prompt_api(
                    idx, prompts[idx], max_new_tokens, block_seed, cfg,
                    tokenizer, token_to_id, sparse_converter, align_prompt_boundary,
                )
                result_slots[idx] = result
                completed += 1
                print(f"  [api] Retry prompt {idx}: OK", flush=True)
                if progress_cb:
                    progress_cb(completed, n)
            except Exception as e2:
                raise RuntimeError(f"API generation failed for prompt {idx} after retry: {e2}") from e2

    if any(r is None for r in result_slots):
        # Should not happen — every prompt either succeeded or raised above.
        raise RuntimeError("generate_via_api: some prompts have no result (internal bug)")
    return result_slots  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry: prepare_teacher_probe_refs_api (greedy text only, no logprobs)
# ---------------------------------------------------------------------------

def _greedy_text_one(prompt_text: str, cfg: APIConfig, max_new_tokens: int, idx: int) -> str:
    """Single greedy generation, text only (no logprobs). Used for probe refs.

    Uses the same exp-backoff retry strategy as the logprob-fetching worker
    (:func:`_generate_single_prompt_api`). Probe refs are fired in waves
    (think-probe ≈ 30, capability ≈ 36, chat-probe ≈ 4) right after the
    main 60-prompt logprob batch finishes, so the Inceptron rate-limit
    bucket is still cold and the first dozen+ requests routinely 429.
    Without retries that would cascade into 'API teacher capability
    failed' for every probe and the round would lose all benchmark axes.
    """
    import requests

    if cfg.endpoint == "chat":
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        url = f"{cfg.base_url}/v1/chat/completions"
    else:
        payload = {
            "model": cfg.model,
            "prompt": prompt_text,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        url = f"{cfg.base_url}/v1/completions"
    payload.update(cfg.extra_body())

    last_err: Optional[Exception] = None
    for attempt in range(6):
        try:
            resp = requests.post(url, headers=cfg.headers(), data=json.dumps(payload),
                                 timeout=cfg.timeout_s)
            sc = resp.status_code
            if sc == 429 or 500 <= sc < 600:
                retry_after_hdr = resp.headers.get("Retry-After")
                try:
                    backoff_s = float(retry_after_hdr) if retry_after_hdr else (2.0 ** attempt)
                except Exception:
                    backoff_s = 2.0 ** attempt
                backoff_s = max(min(backoff_s, 30.0), 1.0)
                last_err = RuntimeError(f"HTTP {sc}: {resp.text[:160]}")
                if attempt < 5:
                    time.sleep(backoff_s)
                    continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and not data.get("choices"):
                raise RuntimeError(f"API error: {data['error']}")
            choice = data["choices"][0]
            if cfg.endpoint == "chat":
                return (choice.get("message") or {}).get("content") or ""
            return choice.get("text") or ""
        except requests.RequestException as e:
            last_err = e
            if attempt < 5:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise
    raise RuntimeError(f"_greedy_text_one[{idx}] exhausted retries: {last_err}")


def greedy_batch_api(
    prompts: List[str],
    max_new_tokens: int = 512,
    config: Optional[APIConfig] = None,
    concurrency: Optional[int] = None,
) -> List[str]:
    """Greedy text-only generation for a batch of prompts via API.

    Used by :func:`prepare_teacher_probe_refs_api` (and any caller that
    wants probe-style refs without logprob overhead).
    """
    cfg = config or APIConfig.from_env(
        **({"concurrency": concurrency} if concurrency else {})
    )
    n = len(prompts)
    out: List[Optional[str]] = [None] * n
    if cfg.concurrency <= 1:
        for i, p in enumerate(prompts):
            out[i] = _greedy_text_one(p, cfg, max_new_tokens, i)
        return [s or "" for s in out]
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futs = {ex.submit(_greedy_text_one, p, cfg, max_new_tokens, i): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                print(f"  [api] probe prompt {i} failed: {e}", flush=True)
                out[i] = ""
    return [s or "" for s in out]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def api_health_check(config: Optional[APIConfig] = None, prompt: str = "Say OK and nothing else.",
                     max_attempts: int = 6) -> Dict[str, Any]:
    """Quick connectivity + logprobs sanity test. Raises on failure.

    Returns ``{"ok": True, "text": "...", "n_logprobs": N, "model": "...", "elapsed_s": ...}``.
    Run this once before committing the eval round to the API path.

    Retries on 429 (rate limit) and 5xx with exponential backoff. The OpenRouter
    Inceptron route has been observed to bounce 429 in clusters when several
    workers compete for the same backend; the per-prompt worker uses the same
    retry logic, so the health check needs to be at least as patient — failing
    fast here means the round aborts immediately on a transient burst.
    """
    import requests

    cfg = config or APIConfig.from_env()
    t0 = time.time()
    if cfg.endpoint == "chat":
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": cfg.top_logprobs,
        }
        url = f"{cfg.base_url}/v1/chat/completions"
    else:
        payload = {
            "model": cfg.model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0.0,
            "logprobs": cfg.top_logprobs,
        }
        url = f"{cfg.base_url}/v1/completions"
    payload.update(cfg.extra_body())

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=cfg.headers(), data=json.dumps(payload), timeout=cfg.timeout_s)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Honour Retry-After if the provider sets it; otherwise
                # exponential backoff with full jitter, capped at 30s.
                retry_after_hdr = resp.headers.get("Retry-After")
                try:
                    backoff_s = float(retry_after_hdr) if retry_after_hdr else (2.0 ** attempt)
                except Exception:
                    backoff_s = 2.0 ** attempt
                backoff_s = min(backoff_s, 30.0)
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff_s)
                    continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise
    else:
        raise RuntimeError(f"api_health_check exhausted {max_attempts} attempts: {last_err}")

    data = resp.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"API error: {data['error']}")
    choice = data["choices"][0]
    if cfg.endpoint == "chat":
        text = (choice.get("message") or {}).get("content") or ""
        content_lp = ((choice.get("logprobs") or {}).get("content") or [])
        n_lp = len(content_lp)
        per_pos_topk = len((content_lp[0] or {}).get("top_logprobs", [])) if n_lp else 0
    else:
        text = choice.get("text") or ""
        top_lp_list = (choice.get("logprobs") or {}).get("top_logprobs") or []
        n_lp = len(top_lp_list)
        per_pos_topk = len(top_lp_list[0]) if n_lp else 0
    if n_lp <= 0 or per_pos_topk <= 0:
        raise RuntimeError(
            "API health check returned no logprobs; KL teacher path is not usable "
            f"(endpoint={cfg.endpoint}, model={cfg.model})"
        )
    return {
        "ok": True,
        "text": text,
        "n_logprob_positions": n_lp,
        "per_position_topk": per_pos_topk,
        "model": cfg.model,
        "endpoint": cfg.endpoint,
        "base_url": cfg.base_url,
        "elapsed_s": round(time.time() - t0, 3),
    }


__all__ = [
    "APIConfig",
    "generate_via_api",
    "greedy_batch_api",
    "api_health_check",
]
