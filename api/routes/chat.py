"""Chat endpoints: proxy to king model on GPU pod, OpenAI-compatible endpoints.

2026-05-04 (Sebastian's "chat doesn't work / Service Unavailable" report):
The chat router used to spawn an ``ssh root@chat-pod -- curl ...``
subprocess for every request. Each call carried ~150-300 ms of SSH
handshake overhead and held a uvicorn worker thread for the duration
of the model's response (5-60 s on long generations). Combined with
~1 GET/s polling from /api/chat/status across many dashboard tabs,
this was saturating the API's ``--limit-concurrency 2000`` budget
and surfacing as ``503 Service Unavailable`` for chat *and* every
other dashboard endpoint sharing the same uvicorn worker.

Fix: route everything through the existing ``chat-tunnel.service``
SSH forward (``localhost:8100 → chat-pod:8100``) using a single
async httpx client. This:

* Eliminates per-request SSH process spawn (~10 ms instead of
  ~250 ms steady-state).
* Frees the uvicorn worker during long generations (httpx async
  yields control instead of blocking on subprocess.wait).
* Uses connection pooling — a single TCP keep-alive to localhost
  serves thousands of requests instead of one ssh socket per call.

The legacy SSH helpers in ``api/helpers/ssh.py`` are retained for
``chat_pod_admin`` and the chat-keeper script, but the chat router
no longer touches them on the hot path. If the local tunnel is
down, ``chat-keeper.timer`` (every 3 min) re-establishes it and
heals vLLM via ``scripts.validator.chat_pod_admin``.
"""

import json
import os
import threading
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    STATE_DIR,
)
from helpers.rate_limit import _chat_rate_limiter, _openai_api_rate_limiter
from helpers.sanitize import _safe_json_load
from state_store import h2h_latest, read_cache, read_state, uid_hotkey_map

router = APIRouter()


# ── King info helper ──────────────────────────────────────────────────────────

def _get_king_info():
    h2h = h2h_latest()
    king_uid = h2h.get("king_uid")
    if king_uid is not None:
        for r in h2h.get("results", []):
            if r.get("is_king") or r.get("uid") == king_uid:
                return king_uid, r.get("model")
        commitments_data = read_cache("commitments", {})
        commitments = (
            commitments_data.get("commitments", commitments_data)
            if isinstance(commitments_data, dict)
            else {}
        )
        king_hotkey = uid_hotkey_map().get(str(king_uid))
        if king_hotkey and king_hotkey in commitments:
            info = commitments[king_hotkey]
            return king_uid, info.get("model") if isinstance(info, dict) else info
        return king_uid, None

    # 2026-05-04 (Kimi cutover follow-up): h2h_latest can sit at king_uid=None
    # for an entire eval generation when the prior king is DQ'd by a hard
    # arch-cutover (e.g. Qwen→Kimi) and no Kimi-arch student has been
    # crowned yet. The chat-keeper's vLLM is still serving the *previous*
    # king (state/chat_pod.json keeps a reference for exactly this
    # scenario), so we surface that to the chat router instead of going
    # 503 — chat staying live during the gap is more important than
    # leaderboard purity in the API surface (the dashboard still reads
    # h2h_latest directly so the leaderboard correctly shows "no king").
    chat_pod_state = read_state("chat_pod.json", {}) or {}
    fallback_model = chat_pod_state.get("model")
    if fallback_model:
        return -1, fallback_model
    return None, None


# ── Local httpx client ───────────────────────────────────────────────────────
# chat_server.py always serves the king under the stable name "sn97-king".
# The HF repo id changes every time a new king is crowned, but vLLM only
# registers what we boot it with, so any client-sent model name has to be
# rewritten before forwarding or vLLM 404s with `does not exist`.
CHAT_POD_SERVED_MODEL = "sn97-king"

# The chat-tunnel.service systemd unit forwards 127.0.0.1:8100 →
# chat-pod:8100 over autossh. Going through localhost lets us:
#   1. Reuse a TCP keep-alive instead of opening a fresh ssh socket
#      per request.
#   2. Detect tunnel-down conditions in <2 s (connection refused)
#      instead of waiting for a 10 s ssh ConnectTimeout.
_LOCAL_CHAT_BASE = f"http://127.0.0.1:{CHAT_POD_PORT}"

# Single shared async client — pooled connections, sane timeouts.
# We deliberately keep ``connect`` short (3 s) so a dead tunnel
# fails fast and we can return 503 to the client; vLLM generations
# can take a while so ``read`` is generous (90 s for sync, the
# stream paths use their own client without a read cap).
_chat_http_client: httpx.AsyncClient | None = None
_chat_http_lock = threading.Lock()


def _get_http_client() -> httpx.AsyncClient:
    """Return the module-level pooled httpx client, creating it on first use."""
    global _chat_http_client
    if _chat_http_client is None:
        with _chat_http_lock:
            if _chat_http_client is None:
                _chat_http_client = httpx.AsyncClient(
                    base_url=_LOCAL_CHAT_BASE,
                    timeout=httpx.Timeout(connect=3.0, read=90.0, write=10.0, pool=5.0),
                    limits=httpx.Limits(
                        max_connections=64,
                        max_keepalive_connections=32,
                        keepalive_expiry=30.0,
                    ),
                )
    return _chat_http_client


def _normalize_chat_payload(payload: dict) -> dict:
    """Rewrite the OpenAI-shaped payload for the chat pod's vLLM.

    1. Force ``model`` to the stable served name. We expose the live king's
       repo id at ``/v1/models`` and at the response level (so clients can
       attribute completions correctly), but vLLM is booted with a fixed
       served name so it can't honor anything else.
    2. Default ``enable_thinking`` off. Distil's king models are reasoners;
       leaving thinking on means small ``max_tokens`` budgets get eaten by
       the reasoning trace and ``content`` comes back null. Clients that
       want thinking can opt in via ``chat_template_kwargs``.
    3. Sane anti-derail sampling defaults (2026-04-30): see
       ``_normalize_chat_payload`` history; rationale unchanged.
    """
    payload = dict(payload)
    payload["model"] = CHAT_POD_SERVED_MODEL
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", False)
    payload["chat_template_kwargs"] = kwargs
    # 2026-05-01 (v30.4 patch v3): chat.arbos.life is a transparent
    # window into the king's behaviour. We do NOT mask poor model
    # quality. No sampling caps, no derail truncation — clients see
    # exactly what the model produces.
    #
    # 2026-05-02 (v30.5 patch): floor max_tokens to keep Open-WebUI's
    # restrictive default from cutting Fermi-style answers mid-paragraph.
    #
    # 2026-05-04 (chat-recovery patch): the previous floor (24576) was
    # interacting badly with degraded post-Kimi-cutover kings whose
    # output never terminates — every Open-WebUI session would hold
    # the vLLM slot for the full max-model-len and our timeout fired
    # before the user saw a single token. We now use a tiered approach:
    #   * if the client explicitly passed any value, respect it (test
    #     harnesses, agent loops, even Open-WebUI's 1200 — clients
    #     opting into a small budget want to bail out fast on a
    #     looping king).
    #   * if no value was supplied, default to 1024 — enough for a
    #     concise reply plus a paragraph of explanation, bounded so
    #     even a stuck king finishes within ~10-15 s on a 1xH200.
    #     Long-form answers still need an explicit ``max_tokens``;
    #     that's the OpenAI default contract anyway.
    if payload.get("max_tokens") is None:
        payload["max_tokens"] = 1024
    # 2026-05-02 (v30.5 patch): math-formatting system prompt — only
    # injected when the client did not provide its own system prompt.
    msgs = list(payload.get("messages") or [])
    has_system = any(
        (isinstance(m, dict) and m.get("role") == "system") for m in msgs
    )
    if not has_system:
        formatting_guide = (
            "You are a helpful, concise assistant. When you write math, "
            "ALWAYS use LaTeX with consistent delimiters: ``$...$`` for "
            "inline math (e.g., the speed $v=d/t$) and ``$$...$$`` on "
            "their OWN lines for block math. Never emit bare LaTeX "
            "commands (``\\text``, ``\\frac``, ``\\times``, "
            "``\\approx``) outside of these delimiters. For simple "
            "arithmetic prefer plain text (``2.36 × 10^22`` is fine) — "
            "reserve LaTeX for multi-line derivations and equations "
            "with fractions, integrals, or sums. Use Markdown for "
            "headers, lists, and code blocks."
        )
        payload["messages"] = [
            {"role": "system", "content": formatting_guide}
        ] + msgs
    return payload


# ── Local chat helpers ───────────────────────────────────────────────────────

class _ChatPodUnavailable(RuntimeError):
    """Raised when the local tunnel to the chat pod is not reachable."""


async def _local_chat_post(payload: dict, *, timeout: float = 90.0) -> dict:
    """Async POST to the local tunnel; returns parsed JSON.

    Raises :class:`_ChatPodUnavailable` for connection / DNS / timeout
    failures so the caller can map to a clean 503. Other exceptions
    propagate.
    """
    client = _get_http_client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json=_normalize_chat_payload(payload),
            timeout=httpx.Timeout(connect=3.0, read=timeout, write=10.0, pool=5.0),
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
        raise _ChatPodUnavailable(str(e)) from e
    if resp.status_code >= 500:
        # vLLM crashed or the tunnel is half-open; surface as unavailable
        # so chat-keeper picks it up on the next tick.
        raise _ChatPodUnavailable(f"vLLM returned {resp.status_code}")
    return resp.json()


async def _local_models_probe(timeout: float = 2.5) -> str | None:
    """Cheap async probe: returns the served model id, or None on failure."""
    client = _get_http_client()
    try:
        resp = await client.get(
            "/v1/models",
            timeout=httpx.Timeout(connect=1.5, read=timeout, write=2.0, pool=2.0),
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for m in (data.get("data") or []):
            mid = m.get("id")
            if mid:
                return mid
    except Exception:
        return None
    return None


# ── Streaming response helpers ───────────────────────────────────────────────

_SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable Cloudflare/Caddy buffering so SSE deltas reach the
    # client without a multi-second batched flush at the proxy.
    "X-Accel-Buffering": "no",
}


def _sse_response(generator) -> StreamingResponse:
    """Wrap an SSE async generator with the standard headers used by both
    the dashboard chat proxy and the OpenAI-compatible passthrough."""
    return StreamingResponse(
        generator, media_type="text/event-stream",
        headers=_SSE_RESPONSE_HEADERS,
    )


# ── Chat-side coherence helper (eval-side parity, kept for reference) ────────
# These helpers used to feed an in-proxy truncator that we removed on
# 2026-05-01 (chat.arbos.life is a transparent window — derail belongs
# in the eval). Retained verbatim so any future re-enable can flip the
# call site, and so the eval-side detector in pod_eval_vllm.py has a
# textually identical sibling for cross-reference when tuning signals.

def _coherence_factor_chat(text: str) -> float:
    if not text:
        return 1.0
    text_len = len(text)
    if text_len < 50:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_frac = non_ascii / text_len
    non_ascii_factor = max(0.0, 1.0 - min(1.0, non_ascii_frac * 4.0))
    seen = set()
    repeats = 0
    for i in range(0, text_len - 50, 25):
        s = text[i:i + 50]
        if s in seen:
            repeats += 1
        seen.add(s)
    repeats_factor = max(0.0, 1.0 - min(1.0, repeats * 0.05))
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return 0.0
    long_words = sum(1 for w in words if len(w) > 50)
    word_list_factor = max(
        0.0, 1.0 - min(1.0, (long_words / n_words) * 1.5),
    )
    word_lens = [len(w) for w in words[:1000]]
    mean_word_len = sum(word_lens) / max(1, len(word_lens))
    meaningful_factor = max(
        0.0, 1.0 - max(0.0, (mean_word_len - 20.0) * 0.1),
    )
    punct_chars = sum(1 for c in text if c in ".,;:?!\"'()[]{}—–-")
    punct_frac = punct_chars / max(1, text_len)
    if text_len < 600:
        punctuation_factor = 1.0
    elif punct_frac >= 0.015:
        punctuation_factor = 1.0
    else:
        punctuation_factor = max(0.0, min(1.0, punct_frac / 0.015))
    norm_words = [w.strip(".,;:?!\"'()[]{}").lower() for w in words]
    norm_words = [
        w for w in norm_words
        if w and w.replace("-", "").isalpha()
    ]
    if len(norm_words) >= 150:
        unique_frac = len(set(norm_words)) / len(norm_words)
        if unique_frac < 0.85:
            unique_word_factor = 1.0
        else:
            unique_word_factor = max(
                0.0, 1.0 - (unique_frac - 0.85) / 0.10,
            )
    else:
        unique_word_factor = 1.0
    coh = (
        non_ascii_factor * repeats_factor * word_list_factor
        * meaningful_factor * punctuation_factor * unique_word_factor
    )
    return max(0.05, min(1.0, coh))


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _extract_message_content(message: dict) -> tuple[str, str | None]:
    """Pull (content, thinking) from a vLLM choices[0].message."""
    content = message.get("content") or ""
    thinking = message.get("reasoning") or message.get("thinking")
    if not content and thinking:
        content = thinking
        thinking = None
    return content, thinking


async def _sync_chat(payload, king_uid, king_model):
    payload["stream"] = False
    try:
        data = await _local_chat_post(payload, timeout=90.0)
    except _ChatPodUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": "chat server unavailable",
                "detail": str(e)[:200],
                "king_uid": king_uid,
                "king_model": king_model,
            },
        )
    if "choices" in data:
        message = data["choices"][0].get("message") or {}
        content, thinking = _extract_message_content(message)
        resp = {
            "response": content,
            "model": king_model,
            "king_uid": king_uid,
        }
        if thinking:
            resp["thinking"] = thinking
        if "usage" in data:
            resp["usage"] = data["usage"]
        _log_chat_turn(
            _normalize_chat_payload(payload),
            content, king_uid, king_model, data,
        )
        return resp
    return {"error": "unexpected response from chat server"}


def _stream_chat(payload, king_uid, king_model):
    payload["stream"] = True
    norm = _normalize_chat_payload(payload)

    async def generate():
        # 2026-05-04: streaming via httpx async — no proxy-side
        # truncation. Forward every SSE delta as-is, accumulate
        # ``acc`` only for the chat_turns.jsonl audit log at the end.
        acc = ""
        client = _get_http_client()
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=norm,
                # vLLM streams forever until the model stops; we cap
                # read at 5 min as a safety belt against runaway
                # generations from a degraded king.
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 500:
                    yield (
                        f"data: {json.dumps({'error': f'chat server returned {resp.status_code}'})}\n\n"
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        yield f"data: {raw}\n\n"
                        continue
                    choices = parsed.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        msg = choices[0].get("message") or {}
                        delta_content = (
                            delta.get("content")
                            or msg.get("content")
                            or ""
                        )
                        if delta_content:
                            acc += delta_content
                    parsed["king_uid"] = king_uid
                    parsed["king_model"] = king_model
                    yield f"data: {json.dumps(parsed)}\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            yield (
                f"data: {json.dumps({'error': 'chat server unavailable', 'detail': str(e)[:200]})}\n\n"
            )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"
        finally:
            try:
                _log_chat_turn(norm, acc, king_uid, king_model, None)
            except Exception:
                pass

    return _sse_response(generate())


# ── Chat turn logging ─────────────────────────────────────────────────────────
# 2026-04-30: minimal request/response audit log so derail complaints can be
# diagnosed after the fact. We log to a JSONL file under STATE_DIR with one
# line per completed turn:
#   { ts, king_uid, king_model, prompt_chars, response_chars,
#     non_ascii_frac, top_repeated_50char_count, completion_tokens,
#     temperature, top_p, repetition_penalty, frequency_penalty,
#     prompt_preview (first 200 chars), response_preview (first 200 + last 200 chars) }
# We deliberately do NOT log full conversations — privacy + disk space.
# When a miner reports "the king derailed", grep this log for high
# non_ascii_frac or non-zero repeated-substring counts.
_CHAT_LOG_PATH = os.path.join(STATE_DIR, "chat_turns.jsonl")
_chat_log_lock = threading.Lock()
_CHAT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB rotation


def _detect_repeated_substring(text: str, win: int = 50, step: int = 25) -> int:
    """Cheap repetition heuristic: count how many ``win``-char windows
    starting at multiples of ``step`` repeat in ``text``.
    """
    seen = set()
    repeats = 0
    if not text or len(text) < win * 2:
        return 0
    for i in range(0, len(text) - win, step):
        s = text[i:i + win]
        if s in seen:
            repeats += 1
        seen.add(s)
    return repeats


def _log_chat_turn(payload, response_text, king_uid, king_model, raw_data=None):
    """Append a one-line JSON record of a completed chat turn.

    Best-effort and non-blocking on errors — never let logging take down a
    user-facing request.
    """
    try:
        prompt_text = ""
        for msg in (payload.get("messages") or []):
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    prompt_text += c + "\n"
        response_text = response_text or ""
        non_ascii = sum(1 for c in response_text if ord(c) > 127)
        non_ascii_frac = non_ascii / len(response_text) if response_text else 0.0
        repeats = _detect_repeated_substring(response_text)
        usage = (raw_data or {}).get("usage") or {}
        rec = {
            "ts": time.time(),
            "king_uid": king_uid,
            "king_model": king_model,
            "prompt_chars": len(prompt_text),
            "response_chars": len(response_text),
            "non_ascii_frac": round(non_ascii_frac, 4),
            "repeats_50char": repeats,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "repetition_penalty": payload.get("repetition_penalty"),
            "frequency_penalty": payload.get("frequency_penalty"),
            "max_tokens": payload.get("max_tokens"),
            "prompt_preview": prompt_text[-400:],
            "response_head": response_text[:200],
            "response_tail": response_text[-200:],
        }
        with _chat_log_lock:
            try:
                if (
                    os.path.exists(_CHAT_LOG_PATH)
                    and os.path.getsize(_CHAT_LOG_PATH) > _CHAT_LOG_MAX_BYTES
                ):
                    bak = _CHAT_LOG_PATH + ".1"
                    if os.path.exists(bak):
                        os.remove(bak)
                    os.rename(_CHAT_LOG_PATH, bak)
            except OSError:
                pass
            try:
                with open(_CHAT_LOG_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
    except Exception:
        # Logging is strictly best-effort.
        pass


# ── Status caching ───────────────────────────────────────────────────────────
# /api/chat/status is hit by every dashboard tab on a 30 s polling
# interval. With ~50 simultaneous viewers and the previous SSH probe
# (~250 ms each), the endpoint alone consumed ~12 worker-seconds per
# minute. We now cache the local probe result for 10 s; the king's
# quality scores already come from h2h_latest (cheap file read), so
# this is mostly about the live vLLM probe.
_status_cache: dict | None = None
_status_cache_ts: float = 0.0
_STATUS_CACHE_TTL = 10.0
_status_cache_lock = threading.Lock()


def _cached_status_lookup() -> dict | None:
    now = time.time()
    with _status_cache_lock:
        if _status_cache is not None and now - _status_cache_ts < _STATUS_CACHE_TTL:
            return _status_cache
    return None


def _store_status_cache(snapshot: dict) -> None:
    global _status_cache, _status_cache_ts
    with _status_cache_lock:
        _status_cache = snapshot
        _status_cache_ts = time.time()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod."""
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)
    try:
        max_tokens = min(int(max_tokens), 6144)
    except (TypeError, ValueError):
        max_tokens = 4096
    stream = body.get("stream", False)

    if not messages:
        return {"error": "messages required"}
    if not isinstance(messages, list) or len(messages) > 50:
        return JSONResponse(
            status_code=400,
            content={"error": "messages must be an array with at most 50 entries"},
        )
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and len(content) > 10000:
            return JSONResponse(
                status_code=400,
                content={"error": "message content too long (max 10000 chars)"},
            )
    if not isinstance(max_tokens, (int, float)) or max_tokens < 1:
        max_tokens = 4096
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    body_rep = body.get("repetition_penalty")
    body_freq = body.get("frequency_penalty")
    body_pres = body.get("presence_penalty")
    pod_payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if isinstance(body_rep, (int, float)) and 1.0 <= body_rep <= 2.0:
        pod_payload["repetition_penalty"] = body_rep
    if isinstance(body_freq, (int, float)) and -2.0 <= body_freq <= 2.0:
        pod_payload["frequency_penalty"] = body_freq
    if isinstance(body_pres, (int, float)) and -2.0 <= body_pres <= 2.0:
        pod_payload["presence_penalty"] = body_pres

    if stream:
        return _stream_chat(pod_payload, king_uid, king_model)
    return await _sync_chat(pod_payload, king_uid, king_model)


@router.get("/api/chat/status")
async def chat_status():
    """Check if the king chat server is available.

    2026-05-04: cached for 10 s to keep a 50-tab dashboard from
    pinging the chat pod 50 times per second. Quality scores are
    pulled from h2h_latest (cheap file read), the only network cost
    is a single 2 s GET to the local tunnel per refresh window.
    """
    cached = _cached_status_lookup()
    if cached is not None:
        return cached

    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    server_ok = False
    served_model: str | None = None
    if CHAT_POD_HOST:
        served_model = await _local_models_probe(timeout=2.5)
        if served_model:
            # vLLM serves the king under the stable "sn97-king" name regardless
            # of which HF repo is loaded; treat any successful probe as healthy.
            server_ok = True

    quality = {
        "long_form_judge": None,
        "long_gen_coherence": None,
        "judge_probe": None,
        "composite_final": None,
    }
    if king_uid is not None and king_uid >= 0:
        try:
            h2h = h2h_latest()
            for r in (h2h.get("results") or []):
                if r.get("uid") == king_uid:
                    comp = r.get("composite") or {}
                    axes = comp.get("axes") or {}
                    quality["long_form_judge"] = axes.get("long_form_judge")
                    quality["long_gen_coherence"] = axes.get("long_gen_coherence")
                    quality["judge_probe"] = axes.get("judge_probe")
                    quality["composite_final"] = comp.get("final")
                    break
        except Exception:
            pass

    snapshot = {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "served_model": served_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "quality": quality,
        "note": (
            "King model is loaded on GPU and ready for chat."
            if server_ok
            else (
                "Chat pod is not configured."
                if not CHAT_POD_HOST
                else (
                    "Chat paused while the eval pipeline holds the GPU."
                    if eval_active
                    else "Chat server is starting or unavailable."
                )
            )
        ),
    }
    _store_status_cache(snapshot)
    return snapshot


# ── OpenAI-compatible endpoints (for Open WebUI etc.) ─────────────────────────

@router.get("/v1/models")
def openai_models():
    """OpenAI-compatible models list. Returns the current king model."""
    king_uid, king_model = _get_king_info()
    model_id = king_model or "distil-king"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
        }],
    }


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint. Proxies to the king model.

    2026-05-02 (v30.5): this endpoint is the entry point for agent
    harnesses (Flue, OpenAI Agents SDK, Vercel AI SDK, LangChain, …)
    that loop the king for many tool-calling rounds. We use a
    dedicated, more generous rate limiter (``_openai_api_rate_limiter``,
    240/min) instead of the strict ``_chat_rate_limiter`` (10/min)
    that throttles direct browser-driven chat.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _openai_api_rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
        )

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return JSONResponse(status_code=503, content={"error": {"message": "no king model available"}})

    stream = body.get("stream", False)
    if stream:
        norm = _normalize_chat_payload(body)

        async def generate():
            client = _get_http_client()
            try:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=norm,
                    timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
                ) as resp:
                    if resp.status_code >= 500:
                        yield (
                            f"data: {json.dumps({'error': {'message': f'chat server returned {resp.status_code}'}})}\n\n"
                        )
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                yield 'data: {"error": {"message": "chat server unavailable"}}\n\n'
            except Exception:
                yield 'data: {"error": {"message": "stream interrupted"}}\n\n'

        return _sse_response(generate())

    try:
        data = await _local_chat_post(body, timeout=120.0)
    except _ChatPodUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "chat server unavailable", "detail": str(e)[:200]}},
        )
    if isinstance(data, dict) and king_model:
        # Stamp the response with the live king's HF repo id so OpenAI
        # clients (Open WebUI etc.) display the correct lineage even
        # though vLLM serves under the stable "sn97-king" name.
        data["model"] = king_model
        data["king_uid"] = king_uid
    return JSONResponse(content=data)
