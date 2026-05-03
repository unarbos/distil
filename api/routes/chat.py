"""Chat endpoints: proxy to king model on GPU pod, OpenAI-compatible endpoints."""

import base64
import json
import os
import subprocess
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    CHAT_RESTART_COOLDOWN,
    CHAT_SERVER_SCRIPT,
    STATE_DIR,
)
from helpers.rate_limit import _chat_rate_limiter, _openai_api_rate_limiter
from helpers.sanitize import _safe_json_load
from helpers.ssh import _ssh_exec, SshExecError
from state_store import h2h_latest, read_cache, uid_hotkey_map

router = APIRouter()


# ── King info helper ──────────────────────────────────────────────────────────

def _get_king_info():
    h2h = h2h_latest()
    king_uid = h2h.get("king_uid")
    if king_uid is None:
        return None, None
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


# ── Shared SSH-curl helpers ──────────────────────────────────────────────────

def _ssh_args():
    if not CHAT_POD_HOST:
        raise SshExecError(1, "chat pod is not configured")
    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-i", CHAT_POD_SSH_KEY,
        "-p", str(CHAT_POD_SSH_PORT),
        f"root@{CHAT_POD_HOST}",
    ]


# chat_server.py always serves the king under the stable name "sn97-king".
# The HF repo id changes every time a new king is crowned, but vLLM only
# registers what we boot it with, so any client-sent model name has to be
# rewritten before forwarding or vLLM 404s with `does not exist`.
CHAT_POD_SERVED_MODEL = "sn97-king"


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
    3. Sane anti-derail sampling defaults (2026-04-30): king models are
       distilled 4B students, and post-distillation they often have
       narrow-attractor failure modes — once a phrase template starts
       repeating, the model will loop on it for the rest of the budget.
       Reproduced on UID 85 (levikross127/131004_v1) with "list 50 cat
       facts" at temp=1.0: facts 1–43 were coherent, then 44–50 looped
       on "Cats Have a 20-Foot X Range / 300-Mile Y Range" with X/Y
       randomly drawn from {Wind, Rain, Snow, Earth, Stone, Metal}.
       At higher temperature the loop wanders into CJK/non-Latin
       vocabulary, which is the multi-language/character "derailing"
       miners are reporting on Discord.
       Default ``repetition_penalty=1.05`` (mild — anything > 1.1 makes
       the king sound robotic and hurts essay quality) and
       ``frequency_penalty=0.3`` (suppresses re-use of recent tokens
       without penalising domain vocabulary). Clients that need raw
       probs can pass an explicit value.
    """
    payload = dict(payload)
    payload["model"] = CHAT_POD_SERVED_MODEL
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", False)
    payload["chat_template_kwargs"] = kwargs
    # 2026-05-01 (v30.4 patch v3): chat.arbos.life is a transparent
    # window into the king's behaviour. We do NOT mask poor model
    # quality. No sampling caps, no derail truncation — clients see
    # exactly what the model produces. If the king derails, the
    # chat exposes it. The eval-side ``long_gen_coherence`` axis
    # will dethrone broken kings.
    #
    # 2026-05-02 (v30.5 patch): the ONE exception is a
    # ``max_tokens`` floor. Open-WebUI's default for the
    # sn97-king model card is 1200, so user questions that
    # need a long answer (Fermi math, multi-step proofs,
    # essays) hit ``finish_reason=length`` mid-paragraph
    # before the model has reached its natural stop token.
    # We raise the floor to 24576 (3/4 of the chat pod's
    # 32768 max-model-len) so a 4K-token prompt + reasonable
    # answer always has room to terminate cleanly. Clients
    # that want a tight cap can still pass an explicit value
    # ≥ 24576; we never lower a client-supplied cap.
    client_max_tokens = payload.get("max_tokens")
    floor = 24576
    if client_max_tokens is None or (
        isinstance(client_max_tokens, (int, float))
        and client_max_tokens < floor
    ):
        payload["max_tokens"] = floor
    # 2026-05-02 (v30.5 patch): math-formatting system prompt.
    # User report: Fermi-style math answers ("how many jelly beans
    # fill the ocean") were rendering with raw red LaTeX in
    # chat.arbos.life because the model emits a mix of (a) bare
    # ``\text{Volume}=3.55\times10^{23}`` (no delimiter), (b)
    # ``$$ ... $$`` block math, and (c) ``$...$`` inline math.
    # Open-WebUI 0.8.12 KaTeX renders (c) reliably but stumbles on
    # (b) when the closing ``$$`` is on the same line as text, and
    # never renders (a) at all. The model is doing this because no
    # system prompt told it to stay consistent.
    #
    # We inject a tiny formatting guide IFF the client did not
    # provide its own system prompt. This is the smallest delta
    # that fixes the user-visible "red LaTeX" without overriding
    # custom system prompts (Open-WebUI lets users set per-chat
    # system prompts; we never clobber those).
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


def _curl_cmd(payload: dict, stream: bool) -> str:
    payload = _normalize_chat_payload(payload)
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    flag = "-sN" if stream else "-s"
    return (
        f"echo '{payload_b64}' | base64 -d | curl {flag} "
        f"-X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d @-"
    )


def run_remote_chat(payload: dict, stream: bool = False, timeout: int = 60) -> str:
    """Non-streaming path: execute the payload via ssh+curl and return raw stdout."""
    return _ssh_exec(_curl_cmd(payload, stream=stream), timeout=timeout, check=False)


def stream_remote_chat(payload: dict):
    """Yield ssh+curl stdout lines for a streaming chat request."""
    cmd = _curl_cmd(payload, stream=True)
    proc = subprocess.Popen(
        _ssh_args() + [cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        for line in proc.stdout:
            yield line
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


# ── Chat-side derail detection (REMOVED 2026-05-01) ───────────────────────────
# Earlier in this session we added an aggressive proxy-side
# truncator that hid the king's long-form derail from users.
# That was the wrong call — chat.arbos.life is the operator's
# transparent window into model quality and should NEVER mask
# poor performance. The derail belongs in the eval, where it
# will dethrone the broken king. Helpers below are kept (unused
# by chat) as reference implementation for the eval-side
# detector in scripts/pod_eval_vllm.py — both share the same
# six-signal heuristic so signal tuning stays in sync.


def _coherence_factor_chat(text: str) -> float:
    """Six-signal statistical coherence detector — copy of the one in
    pod_eval_vllm.py, kept here so the chat proxy can run it without
    importing the eval module. See the original for full docstring.

    Returns coherence in [0.05, 1.0]. 1.0 = clean prose. <0.3 = derail.
    """
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
    # 2026-05-01 (v30.4 patch v2): raised threshold 10 → 20. Academic
    # prose ("Philosophical Inquiry into Artificial Intelligence") has
    # 10-15 char words frequently and was scoring meaningful_factor
    # ~0.6, false-positiving the truncator on legitimate long
    # responses. The signal is meant to catch nonsense compound
    # coinage ("jovialincarnacioappreciable", "boblynberry-vogesters")
    # which has mean word length 50+ in single-word strings.
    meaningful_factor = max(
        0.0, 1.0 - max(0.0, (mean_word_len - 20.0) * 0.1),
    )
    punct_chars = sum(1 for c in text if c in ".,;:?!\"'()[]{}—–-")
    punct_frac = punct_chars / max(1, text_len)
    # 2026-05-01 (v30.4 patch v2): lowered floor 0.03 → 0.015 and
    # raised the ≥400 chars gate to ≥600. Academic / multi-paragraph
    # prose with markdown headers (asterisks, no terminal
    # punctuation) was hitting a 1.5-2.5 percent punct rate and
    # getting falsely penalized. Real word-list derail mode runs
    # 0-0.5 percent punctuation across long stretches.
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


def _truncate_at_derail(
    text: str,
    window: int = 800,
    threshold: float = 0.5,
) -> tuple[str, bool]:
    """Find the last coherent prefix of ``text`` and truncate there.

    Returns ``(truncated_text, was_truncated)``. The algorithm slides
    a ``window``-char detector across the text in ``window // 2`` steps
    and finds the FIRST window whose coherence drops below
    ``threshold``. The cut is placed at the last sentence boundary
    before the derail starts (period, question mark, exclamation
    point, or newline), with a graceful "[truncated]" tail so the
    user knows the rest was discarded.

    The window is 800 chars so that BOTH the punctuation_factor (≥600
    chars threshold) AND the unique_word_factor (≥150 words threshold)
    are active in each detector pass. The previous 400-char window
    let derail chunks slip past those signals.

    Cheap O(N): each window's coherence is computed on an 800-char
    slice and the loop stops at the first bad window. For a clean
    coherent response the loop runs through every window once
    (typical 5000 chars → ~12 windows × ~1ms each = ~12ms).
    """
    if not text or len(text) < window:
        return text, False
    text_len = len(text)
    step = max(1, window // 2)
    derail_start = None
    for end in range(window, text_len + step, step):
        end = min(end, text_len)
        chunk = text[max(0, end - window):end]
        if _coherence_factor_chat(chunk) < threshold:
            derail_start = end - window
            break
    if derail_start is None:
        return text, False
    cutoff = max(0, derail_start)
    sentence_breaks = ".?!"
    paragraph_breaks = "\n"
    for i in range(cutoff, max(0, cutoff - 600), -1):
        if i < text_len and (
            text[i] in paragraph_breaks
            or (
                text[i] in sentence_breaks
                and (i + 1 >= text_len or text[i + 1] in " \n\t")
            )
        ):
            return (
                text[:i + 1]
                + "\n\n"
                + "_[Response truncated — the model began producing "
                + "incoherent text past this point. This is a known "
                + "failure mode of the current king on long generations; "
                + "the next eval round should dethrone this model.]_"
            ), True
    return (
        text[:cutoff]
        + "\n\n"
        + "_[Response truncated — incoherent text past this point.]_"
    ), True


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _extract_message_content(message: dict) -> tuple[str, str | None]:
    """Pull (content, thinking) from a vLLM choices[0].message.

    vLLM in reasoner mode puts the assistant text in ``content`` when
    thinking is disabled. When thinking is enabled, it splits the model
    output: ``reasoning`` holds the chain-of-thought, ``content`` may end
    up null if max_tokens cuts the reply mid-thought. We always fall back
    to ``reasoning`` so chat.arbos.life never shows a blank bubble even
    when a client opts back into thinking.
    """
    content = message.get("content") or ""
    thinking = message.get("reasoning") or message.get("thinking")
    if not content and thinking:
        content = thinking
        thinking = None
    return content, thinking


def _sync_chat(payload, king_uid, king_model):
    payload["stream"] = False
    stdout = run_remote_chat(payload, stream=False, timeout=60)
    try:
        data = json.loads(stdout)
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
            # Log the chat turn for derail audits in chat_turns.jsonl
            # (raw content, no truncation).
            _log_chat_turn(
                _normalize_chat_payload(payload),
                content, king_uid, king_model, data,
            )
            return resp
        return {"error": "unexpected response from chat server"}
    except json.JSONDecodeError:
        return {"error": "chat server not responding - may be starting up"}


def _stream_chat(payload, king_uid, king_model):
    payload["stream"] = True

    def generate():
        # 2026-05-01 (v30.4 patch v3): no proxy-side truncation. We
        # forward every delta as-is and accumulate in ``acc`` only
        # for the chat_turns.jsonl audit log at the end. Derail is
        # caught by the eval, not hidden by the chat.
        acc = ""
        try:
            for line in stream_remote_chat(payload):
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
        except Exception as e:
            err = str(e)
            if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
                err = "chat server connection failed"
            yield f"data: {json.dumps({'error': err[:200]})}\n\n"
        try:
            _log_chat_turn(
                _normalize_chat_payload(payload),
                acc, king_uid, king_model, None,
            )
        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_chat_restart_lock = threading.Lock()
_last_chat_restart = 0.0


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
    starting at multiples of ``step`` repeat in ``text``. Used as a
    derail signal (>0 means at least one verbatim ~50-char repeat).
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


def _ensure_chat_server(king_model=None):
    """Auto-start chat server if not running or wrong model.

    Rate-limited to once per :data:`CHAT_RESTART_COOLDOWN` seconds.
    """
    global _last_chat_restart
    with _chat_restart_lock:
        if time.time() - _last_chat_restart < CHAT_RESTART_COOLDOWN:
            return
        _last_chat_restart = time.time()

    model_name = king_model or "unknown"
    try:
        stdout = _ssh_exec(
            f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models || echo not_running",
            check=False,
        )
        if "not_running" in stdout:
            print(f"[chat] Auto-starting chat server for {model_name}", flush=True)
            _ssh_exec(
                f"nohup python3 {CHAT_SERVER_SCRIPT} '{model_name}' {CHAT_POD_PORT} "
                f"> /root/chat.log 2>&1 &",
                timeout=10, check=False,
            )
        elif model_name != "unknown" and model_name not in stdout:
            print(
                f"[chat] Chat server running wrong model, restarting for {model_name}",
                flush=True,
            )
            _ssh_exec(
                "pkill -f 'vllm.entrypoints.openai.api_server|chat_server.py' || true",
                timeout=10, check=False,
            )
            time.sleep(2)
            _ssh_exec(
                f"nohup python3 {CHAT_SERVER_SCRIPT} '{model_name}' {CHAT_POD_PORT} "
                f"> /root/chat.log 2>&1 &",
                timeout=10, check=False,
            )
    except Exception as e:
        print(f"[chat] Auto-restart failed: {e}", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod.

    Supports streaming via ``stream=true``.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    body = await request.json()
    messages = body.get("messages", [])
    # 2026-05-01 (v30.4 patch v3): no masking. Default max_tokens is
    # 4096 (a typical assistant default), bounded only by the model's
    # context window (6144 hard cap = 8192 max_model_len minus prompt
    # headroom). If the king derails, the chat exposes it.
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
    # 2026-05-01 (v30.4 patch v3): standard assistant defaults — no
    # anti-derail bias. Chat exposes model behaviour as-is.
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    # Anti-derail defaults — see _normalize_chat_payload docstring for rationale.
    # 2026-04-30: client can override but we set a non-zero floor so the chat
    # path never falls into the all-defaults vLLM behaviour where every
    # repetition / frequency / presence penalty is 0.
    body_rep = body.get("repetition_penalty")
    body_freq = body.get("frequency_penalty")
    body_pres = body.get("presence_penalty")
    try:
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
        return _sync_chat(pod_payload, king_uid, king_model)

    except Exception as e:
        err = str(e)
        if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
            return {"error": "chat server connection failed - try again in a moment"}
        return {"error": f"chat error: {err[:200]}"}


@router.get("/api/chat/status")
def chat_status():
    """Check if the king chat server is available. Auto-starts if down."""
    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    server_ok = False
    if CHAT_POD_HOST:
        try:
            stdout = _ssh_exec(
                f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models >/dev/null && cat /root/model_name.txt 2>/dev/null",
                check=False,
            )
            served = (stdout or "").strip()
            if served and (king_model is None or served == king_model):
                server_ok = True
            elif not eval_active:
                _ensure_chat_server(king_model)
        except SshExecError:
            pass

    return {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "note": (
            "King model is loaded on GPU and ready for chat."
            if server_ok
            else (
                "Chat pod is not configured."
                if not CHAT_POD_HOST
                else "Chat server is starting or unavailable."
            )
        ),
    }


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
    that throttles direct browser-driven chat. See
    ``examples/flue/sn97-king-tool-calling/`` for a working integration.
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
    # 2026-05-01 (v30.4 patch v3): no truncation. Pass through the
    # model's response as-is. Chat is a transparent surface; if the
    # king derails, we expose it.
    try:
        if stream:
            def generate():
                try:
                    for line in stream_remote_chat(body):
                        line = line.strip()
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                break
                except Exception:
                    yield 'data: {"error": "stream interrupted"}\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        stdout = run_remote_chat(body, stream=False, timeout=120)
        try:
            data = json.loads(stdout)
            # Stamp the response with the live king's HF repo id so OpenAI
            # clients (Open WebUI etc.) display the correct lineage even
            # though vLLM serves under the stable "sn97-king" name.
            if isinstance(data, dict) and king_model:
                data["model"] = king_model
                data["king_uid"] = king_uid
            return JSONResponse(content=data)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "chat server not responding"}},
            )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "chat server connection failed"}},
        )
