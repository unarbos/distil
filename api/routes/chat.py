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

import ast
import html
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    STATE_DIR,
)
from external import get_model_info as fetch_model_info_data
from helpers.rate_limit import _chat_rate_limiter, _openai_api_rate_limiter
from helpers.sanitize import _safe_json_load
from state_store import (
    eval_progress,
    h2h_latest,
    read_cache,
    read_state,
    top4_leaderboard,
    uid_hotkey_map,
)

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


# ── Product-grade chat orchestration ─────────────────────────────────────────

_CHAT_ORCHESTRATION_ENABLED = (
    os.environ.get("DISTIL_CHAT_ORCHESTRATION", "1") != "0"
)
_CHAT_QUALITY_FALLBACK_ENABLED = (
    os.environ.get("DISTIL_CHAT_QUALITY_FALLBACK", "1") != "0"
)

_SN97_RE = re.compile(
    r"\b(sn\s?97|sn-97|subnet|king|leaderboard|miner|uid|eval|round|"
    r"validator|arbos|distil)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"\b(run|execute|python|script|code|calculate|compute|evaluate|math|"
    r"fibonacci|fib(?:onacci)?_?sequence)\b",
    re.IGNORECASE,
)
_UID_RE = re.compile(r"\buid\s*[:=#]?\s*(\d{1,3})\b", re.IGNORECASE)
_MODEL_PATH_RE = re.compile(
    r'"model_path"\s*:\s*"([^"]+)"|'
    r"\b([A-Za-z0-9._-]{2,40}/[A-Za-z0-9._-]{2,100})\b"
)
_WEB_SEARCH_RE = re.compile(
    r"\b(search (?:the )?web|web search|look up|google|duckduckgo|latest news|"
    r"current (?:news|price|weather|release|version))\b",
    re.IGNORECASE,
)
_FIB_INDEX_RE = re.compile(
    r"(?:(\d+)\s*\^\s*(\d+)(?:st|nd|rd|th)?|\b(\d{1,8})(?:st|nd|rd|th)?)"
    r".{0,40}\b(?:fib(?:onacci)?|fibonacci)\b",
    re.IGNORECASE,
)
_FACTORIAL_RE = re.compile(
    r"(?:factorial\s*(?:of)?|(\d{1,6})\s*!)\s*(\d{1,6})?",
    re.IGNORECASE,
)
_DERAIL_MARKERS = (
    "Use the tool ",
    "get_model_info",
    "get_subnet_overview",
    "get_leaderboard",
    "knowledge base",
    "Update knowledge base",
    "outlook steady",
    "Outlook steady",
    "steady demand for clear",
    "Persistent File System",
    "Pyodide",
)


def _latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                return "\n".join(parts)
    return ""


def _clean_client_messages(messages: list[dict], *, system: str, max_history: int = 8) -> list[dict]:
    """Keep recent user/assistant text only; never forward tool specs/messages.

    Open-WebUI native function calling sends ``tools`` outside the messages,
    but failed turns can leave tool-like prose in assistant history. The weak
    king tends to imitate that, so the proxy feeds vLLM a clean chat transcript
    and handles tools itself.
    """
    cleaned = [{"role": "system", "content": system}]
    for msg in (messages or [])[-max_history:]:
        if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if any(marker in content for marker in _DERAIL_MARKERS):
            continue
        cleaned.append({"role": msg["role"], "content": content[:4000]})
    return cleaned


def _detect_repeated_line_run(text: str) -> bool:
    lines = [ln.strip().lower() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    seen = {}
    for line in lines:
        if len(line) < 18:
            continue
        seen[line] = seen.get(line, 0) + 1
        if seen[line] >= 3:
            return True
    return False


def _detect_repeated_short_tokens(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]{3,24}", text.lower())
    if len(tokens) < 24:
        return False
    for n in (1, 2, 3):
        grams = {}
        for i in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            grams[gram] = grams.get(gram, 0) + 1
            if grams[gram] >= 8:
                return True
    return False


def _answer_looks_bad(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return True
    if sum(1 for c in text if ord(c) > 127) / max(1, len(text)) > 0.03:
        return True
    if "</return>" in text or "```text\nsteady" in text.lower():
        return True
    if any(marker.lower() in text.lower() for marker in _DERAIL_MARKERS):
        return True
    if _detect_repeated_substring(text, win=60, step=20) >= 2:
        return True
    if _detect_repeated_line_run(text):
        return True
    if _detect_repeated_short_tokens(text):
        return True
    # The current weak kings often drift into all-caps first-person snippets.
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) >= 20:
        upper = sum(1 for w in words if w.isupper())
        if upper / max(1, len(words)) > 0.35:
            return True
    return False


def _trim_derail(text: str, *, max_chars: int = 2400) -> str:
    text = (text or "").strip()
    for marker in _DERAIL_MARKERS:
        idx = text.lower().find(marker.lower())
        if idx > 80:
            text = text[:idx].rstrip()
    # Drop repeated lines while preserving the first occurrence.
    out = []
    seen = set()
    for line in text.splitlines():
        key = line.strip().lower()
        if len(key) > 20 and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line.rstrip())
    text = "\n".join(out).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        stop = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = (cut[: stop + 1] if stop > 400 else cut).rstrip()
    return text


def _live_sn97_context(user_text: str, king_uid: int | None, king_model: str | None) -> tuple[str, str | None]:
    """Return (thinking, deterministic answer if obvious)."""
    top4 = top4_leaderboard() or {}
    progress = eval_progress() or {}
    h2h = h2h_latest() or {}
    king = dict(top4.get("king") or {})
    if king_uid is not None:
        king.setdefault("uid", king_uid)
    if king_model:
        king.setdefault("model", king_model)
    contenders = [dict(c) for c in (top4.get("contenders") or [])[:4]]
    eval_active = bool(progress.get("active") or progress.get("phase") not in (None, "", "idle"))

    thought_lines = [
        "I checked the live SN97 state before answering.",
        f"Current king: UID {king.get('uid', king_uid)}, model {king.get('model') or king_model or 'unknown'}.",
    ]
    if progress:
        stage = progress.get("phase") or progress.get("stage") or progress.get("status")
        thought_lines.append(f"Eval status: active={eval_active}, stage={stage or 'unknown'}.")

    lower = user_text.lower()
    answer = None
    if "leaderboard" in lower or "top" in lower:
        rows = [king] + contenders
        lines = ["Current SN97 leaderboard:"]
        for i, row in enumerate(rows[:5], start=1):
            comp = row.get("composite") or {}
            score = comp.get("final") or row.get("score") or row.get("h2h_kl")
            score_txt = f", score {score:.4f}" if isinstance(score, (int, float)) else ""
            label = "king" if i == 1 else "contender"
            lines.append(
                f"{i}. UID {row.get('uid')} ({label}): {row.get('model') or 'unknown'}{score_txt}"
            )
        answer = "\n".join(lines)
    elif "king" in lower or "who" in lower:
        answer = (
            f"The current SN97 king is UID {king.get('uid', king_uid)}"
            f" running `{king.get('model') or king_model or 'unknown'}`."
        )
    elif "eval" in lower or "round" in lower or "status" in lower:
        stage = progress.get("phase") or progress.get("stage") or progress.get("status") or "unknown"
        done = progress.get("students_done")
        total = progress.get("students_total")
        block = progress.get("completed_block") or h2h.get("block")
        bits = [f"Eval active: {eval_active}", f"stage: {stage}"]
        if done is not None and total is not None:
            bits.append(f"students: {done}/{total}")
        if block is not None:
            bits.append(f"block: {block}")
        answer = "Current SN97 eval status: " + ", ".join(bits) + "."
    else:
        m = _UID_RE.search(user_text)
        if m:
            uid = int(m.group(1))
            commitments_data = read_cache("commitments", {}) or {}
            commitments = commitments_data.get("commitments", commitments_data)
            hotkey = uid_hotkey_map().get(str(uid))
            model = None
            if hotkey and isinstance(commitments, dict):
                c = commitments.get(hotkey) or {}
                if isinstance(c, dict):
                    model = c.get("model") or c.get("repo")
            answer = (
                f"UID {uid} is registered as `{model or 'unknown model'}`. "
                "For the full score/history view, use the dashboard miner page."
            )

    return "\n".join(thought_lines), answer


def _model_info_answer(user_text: str) -> tuple[str, str] | None:
    if "get_model_info" not in user_text and "model_path" not in user_text:
        return None
    m = _MODEL_PATH_RE.search(user_text)
    if not m:
        return (
            "I recognized a model-info request but did not find a valid HuggingFace repo path.",
            "I need a HuggingFace repo path like `owner/repo` to look up model info.",
        )
    model_path = (m.group(1) or m.group(2) or "").strip()
    info = fetch_model_info_data(model_path)
    thought = f"I executed the model-info lookup for `{model_path}` via the Distil API helper."
    if not isinstance(info, dict) or info.get("error"):
        return thought, f"I could not fetch reliable model info for `{model_path}`: {info.get('error') if isinstance(info, dict) else 'unknown error'}"
    fields = []
    for key in ("params_b", "is_moe", "num_experts", "num_active_experts", "pipeline_tag", "license", "downloads", "likes"):
        if info.get(key) is not None:
            fields.append(f"- `{key}`: {info.get(key)}")
    if not fields:
        fields.append("- No detailed metadata was available in the cache/provider response.")
    return thought, f"Model info for `{model_path}`:\n" + "\n".join(fields)


def _format_tool_context_for_display(tool_context: list[str], *, max_chars: int = 5000) -> str:
    if not tool_context:
        return ""
    text = "\n\n".join(tool_context).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n... [tool output truncated]"
    return (
        "Runtime tool output (not model text):\n\n"
        "```text\n"
        f"{text}\n"
        "```\n\n"
    )


def _strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


async def _web_search_tool(query: str, *, limit: int = 5) -> str:
    q = query.strip()
    q = re.sub(r"^\s*(search (?:the )?web for|web search for|look up|google)\s+", "", q, flags=re.I)
    if not q:
        return "No search query detected."
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DistilSN97Chat/1.0)"}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(12.0, connect=4.0),
            headers=headers,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.text
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {str(exc)[:200]}"

    results = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        flags=re.I | re.S,
    ):
        href = html.unescape(m.group(1))
        title = _strip_html_tags(m.group(2))
        if href.startswith("//duckduckgo.com/l/?uddg="):
            parsed = urllib.parse.urlparse("https:" + href)
            href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
        if title:
            results.append((title, href))
        if len(results) >= limit:
            break
    if not results:
        return f"No web results parsed for query: {q}"
    lines = [f"query = {q}"]
    for i, (title, href) in enumerate(results, 1):
        lines.append(f"{i}. {title}\n   {href}")
    return "\n".join(lines)


def _extract_python_code(user_text: str) -> tuple[str | None, str]:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", user_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip(), "python code block"

    inline = re.search(r"`([^`]{3,400})`", user_text)
    if inline and any(ch in inline.group(1) for ch in "+-*/()%"):
        expr = inline.group(1).strip()
        return f"print({expr})", "inline arithmetic expression"

    factorial = _FACTORIAL_RE.search(user_text)
    if factorial:
        n_txt = factorial.group(1) or factorial.group(2)
        try:
            n = int(n_txt)
        except (TypeError, ValueError):
            n = -1
        if 0 <= n <= 100_000:
            return (
                "import math, sys\n"
                "if hasattr(sys, 'set_int_max_str_digits'):\n"
                "    sys.set_int_max_str_digits(0)\n"
                f"n = {n}\n"
                "value = math.factorial(n)\n"
                "s = str(value)\n"
                "print('n =', n)\n"
                "print('digits =', len(s))\n"
                "if len(s) <= 1000:\n"
                "    print('value =', s)\n"
                "else:\n"
                "    print('value_head_80 =', s[:80])\n"
                "    print('value_tail_80 =', s[-80:])",
                "factorial runtime",
            )

    # Conservative expression extraction for requests like
    # "calculate ((88+43)**(2-2))//9".
    expr_match = re.search(
        r"(?:calculate|compute|evaluate|what is|what's|solve|result of|math)\s+([0-9\s+\-*/%().,_<>=!&|^~]+)",
        user_text,
        re.IGNORECASE,
    )
    if expr_match:
        expr = expr_match.group(1).strip(" .?\n\t")
        if expr:
            return f"print({expr})", "arithmetic expression"
    return None, ""


def _python_code_is_reasonable(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    banned = {"socket", "subprocess", "shutil", "pathlib", "requests", "httpx", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in banned:
                    return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"open", "exec", "eval", "compile", "__import__"}:
                return False
    return True


def _run_python_code(code: str) -> tuple[str, str | None]:
    if not _python_code_is_reasonable(code):
        return "", "I refused to run that code because it uses unsafe imports or dynamic execution."
    wrapper = (
        "import math, statistics, fractions, decimal, itertools, functools, collections\n"
        + code.strip()
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="distil_chat_py_") as td:
        path = os.path.join(td, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapper)
        try:
            proc = subprocess.run(
                ["python3", "-I", path],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=4,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", "Python execution timed out after 4 seconds."
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return out, f"Python exited with code {proc.returncode}: {err[-800:]}"
    return out[:4000], None


def _extract_fibonacci_tool(user_text: str) -> tuple[str | None, str]:
    if "fib" not in user_text.lower():
        return None, ""
    m = _FIB_INDEX_RE.search(user_text)
    if not m:
        return None, ""
    if m.group(1) and m.group(2):
        n = int(m.group(1)) ** int(m.group(2))
        desc = f"Fibonacci index {m.group(1)}^{m.group(2)} = {n}"
    else:
        n = int(m.group(3))
        desc = f"Fibonacci index {n}"
    if n < 0 or n > 1_000_000:
        return None, f"Requested Fibonacci index {n} is outside the runtime limit (0..1,000,000)."
    code = f"""
import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

def fib_pair(n):
    if n == 0:
        return (0, 1)
    a, b = fib_pair(n // 2)
    c = a * (2 * b - a)
    d = a * a + b * b
    if n % 2:
        return (d, c + d)
    return (c, d)

n = {n}
value = fib_pair(n)[0]
s = str(value)
print("index =", n)
print("digits =", len(s))
if len(s) <= 1000:
    print("value =", s)
else:
    print("value_head_80 =", s[:80])
    print("value_tail_80 =", s[-80:])
"""
    return code.strip(), desc


async def _call_quality_fallback(messages: list[dict], *, max_tokens: int = 700) -> str | None:
    if not _CHAT_QUALITY_FALLBACK_ENABLED:
        return None
    key = (
        os.environ.get("CHAT_FALLBACK_API_KEY")
        or os.environ.get("DISTIL_TEACHER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        return None
    base = (
        os.environ.get("CHAT_FALLBACK_API_BASE")
        or os.environ.get("DISTIL_TEACHER_API_BASE")
        or "https://openrouter.ai/api"
    ).rstrip("/")
    model = (
        os.environ.get("CHAT_FALLBACK_API_MODEL")
        or os.environ.get("DISTIL_TEACHER_API_MODEL")
        or "moonshotai/kimi-k2.6"
    )
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.25,
        "top_p": 0.9,
    }
    if "openrouter" in base:
        providers = tuple(
            p.strip()
            for p in os.environ.get("DISTIL_TEACHER_API_PROVIDERS", "").split(",")
            if p.strip()
        )
        if providers:
            payload["provider"] = {"only": list(providers)}
        payload["reasoning"] = {"enabled": False}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(f"{base}/v1/chat/completions", json=payload, headers=headers)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            return msg.get("content") or None
    except Exception:
        return None


async def _prepare_orchestrated_chat(
    body: dict,
    king_uid: int | None,
    king_model: str | None,
    *,
    stream: bool = False,
) -> tuple[dict, list[str], list[str]]:
    """Build the vLLM request plus deterministic runtime tool context.

    Tool execution happens before the model call. The streaming path uses this
    to emit tool output immediately, then streams the raw vLLM answer.
    """
    messages = list(body.get("messages") or [])
    user_text = _latest_user_text(messages)
    runtime_trace = ["Runtime trace, not hidden model reasoning: inspected the latest user request."]
    tool_context: list[str] = []

    model_info = _model_info_answer(user_text or "")
    if model_info:
        t, answer = model_info
        runtime_trace.append(t)
        tool_context.append(f"MODEL_INFO_RESULT:\n{answer}")

    if _WEB_SEARCH_RE.search(user_text or ""):
        result = await _web_search_tool(user_text or "")
        runtime_trace.append("Executed a DuckDuckGo web search from the chat runtime.")
        tool_context.append(f"WEB_SEARCH_RESULT:\n{result}")

    # Tool execution: SN97 live state.
    if _SN97_RE.search(user_text or ""):
        t, answer = _live_sn97_context(user_text, king_uid, king_model)
        runtime_trace.append(t)
        if answer:
            tool_context.append(f"SN97_LIVE_RESULT:\n{answer}")

    fib_code, fib_desc = _extract_fibonacci_tool(user_text or "")
    if fib_desc and not fib_code:
        runtime_trace.append(f"Fibonacci runtime rejected the request: {fib_desc}")
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{fib_desc}")
    elif fib_code:
        out, err = _run_python_code(fib_code)
        runtime_trace.append(f"Executed Python for {fib_desc}.")
        if err:
            result = f"Python failed:\n{err}"
            if out:
                result += f"\nPartial stdout:\n{out}"
        else:
            result = f"Python stdout:\n{out or '(no stdout)'}"
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{result}")

    # Tool execution: Python code / arithmetic.
    code, code_kind = _extract_python_code(user_text or "")
    if code and _CODE_RE.search(user_text or "") and not fib_code:
        out, err = _run_python_code(code)
        runtime_trace.append(f"Executed a {code_kind} in a short-lived Python sandbox.")
        if err:
            result = f"Python failed:\n{err}"
            if out:
                result += f"\nPartial stdout:\n{out}"
        else:
            result = f"Python stdout:\n{out or '(no stdout)'}"
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{result}")
    elif _CODE_RE.search(user_text or "") and "python" in (user_text or "").lower():
        runtime_trace.append("The user asked for Python execution but no runnable code/expression was detected.")
        tool_context.append(
            "PYTHON_EXECUTION_RESULT:\nNo runnable Python code or arithmetic expression was detected."
        )

    system = (
        "You are SN97 chat. Answer the user directly. If tool results are "
        "provided, use them as authoritative context and do not invent another "
        "tool name. Do not claim you used a tool unless a tool result is "
        "present in the context."
    )
    clean_messages = _clean_client_messages(messages, system=system)
    if tool_context:
        clean_messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Tool results from the chat runtime:\n\n"
                    + "\n\n".join(tool_context)
                    + "\n\nUse these results when relevant. The final answer should still be your own model output."
                ),
            },
        )
    vllm_body = {
        "model": CHAT_POD_SERVED_MODEL,
        "messages": clean_messages,
        "stream": stream,
        "max_tokens": body.get("max_tokens") or 700,
        "temperature": body.get("temperature", 0.6),
        "top_p": body.get("top_p", 0.9),
        "frequency_penalty": body.get("frequency_penalty", 0.0),
        "presence_penalty": body.get("presence_penalty", 0.0),
        "repetition_penalty": body.get("repetition_penalty", 1.0),
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    }
    return vllm_body, tool_context, runtime_trace


async def _orchestrated_chat_completion(body: dict, king_uid: int | None, king_model: str | None) -> dict:
    """Transparent tool runtime around fast vLLM.

    This function may execute tools and add their outputs to the model context,
    but it must not quality-filter, rewrite, or replace the model's final text.
    chat.arbos.life is intentionally a window into the current king's behavior;
    if the king derails after seeing useful tool results, that failure should
    remain visible.
    """
    vllm_body, tool_context, runtime_trace = await _prepare_orchestrated_chat(
        body, king_uid, king_model, stream=False,
    )
    try:
        raw = await _local_chat_post(vllm_body, timeout=120.0)
        msg = (raw.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        model_reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        usage = raw.get("usage")
    except _ChatPodUnavailable as exc:
        content = f"chat server unavailable: {str(exc)[:200]}"
        model_reasoning = ""
        usage = None
    runtime_trace.append("Forwarded the final prompt to the fast local vLLM king and returned its raw answer.")
    if tool_context:
        content = (
            _format_tool_context_for_display(tool_context)
            + "Raw model answer:\n\n"
            + (content or "(empty)")
        )
    reasoning_parts = []
    if model_reasoning:
        reasoning_parts.append(model_reasoning)
    if tool_context:
        reasoning_parts.append("Tool outputs supplied to the model:\n" + _format_tool_context_for_display(tool_context))
    reasoning_parts.append("\n".join(runtime_trace))
    reasoning = "\n\n".join(reasoning_parts)
    now = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    usage = usage or {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": now,
        "model": king_model or CHAT_POD_SERVED_MODEL,
        "king_uid": king_uid,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    "reasoning_content": reasoning,
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _stream_openai_response(data: dict):
    async def generate():
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        base = {
            "id": data.get("id"),
            "object": "chat.completion.chunk",
            "created": data.get("created"),
            "model": data.get("model"),
        }
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if reasoning:
            yield "data: " + json.dumps({
                **base,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning": reasoning,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": None,
                }],
            }) + "\n\n"
        content = msg.get("content") or ""
        # Coarse chunks keep the implementation simple while preserving
        # streaming semantics for Open-WebUI.
        for i in range(0, len(content), 240):
            yield "data: " + json.dumps({
                **base,
                "choices": [{"index": 0, "delta": {"content": content[i:i + 240]}, "finish_reason": None}],
            }) + "\n\n"
        yield "data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"
    return _sse_response(generate())


def _openai_sse_chunk(base: dict, delta: dict, *, finish_reason=None) -> str:
    return "data: " + json.dumps({
        **base,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }) + "\n\n"


def _delta_reasoning(delta: dict, msg: dict | None = None) -> str:
    msg = msg or {}
    return (
        delta.get("reasoning")
        or delta.get("reasoning_content")
        or delta.get("thinking")
        or msg.get("reasoning")
        or msg.get("reasoning_content")
        or msg.get("thinking")
        or ""
    )


def _stream_orchestrated_openai_response(body: dict, king_uid: int | None, king_model: str | None):
    """True SSE streaming for the OpenAI-compatible endpoint.

    Runtime tools execute first and are emitted immediately as visible content;
    then the raw king answer is streamed from vLLM token-by-token. This keeps
    chat transparent while avoiding the previous "wait for full completion,
    then chunk a finished string" behavior.
    """
    async def generate():
        response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        base = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": king_model or CHAT_POD_SERVED_MODEL,
            "king_uid": king_uid,
        }
        acc = ""
        reasoning_acc: list[str] = []
        finish_sent = False
        vllm_body: dict | None = None

        try:
            vllm_body, tool_context, runtime_trace = await _prepare_orchestrated_chat(
                body, king_uid, king_model, stream=True,
            )
        except Exception as exc:
            yield "data: " + json.dumps({
                "error": {"message": f"chat orchestration failed: {str(exc)[:200]}"},
            }) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        trace = "\n".join(runtime_trace)
        if trace:
            reasoning_acc.append(trace)
            yield _openai_sse_chunk(
                base,
                {
                    "role": "assistant",
                    "reasoning": trace,
                    "reasoning_content": trace,
                },
            )
        else:
            yield _openai_sse_chunk(base, {"role": "assistant"})

        if tool_context:
            prefix = _format_tool_context_for_display(tool_context) + "Raw model answer:\n\n"
            acc += prefix
            yield _openai_sse_chunk(base, {"content": prefix})

        client = _get_http_client()
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_normalize_chat_payload(vllm_body),
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                    yield "data: " + json.dumps({
                        "error": {
                            "message": f"chat server returned {resp.status_code}",
                            "detail": detail,
                        },
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choice = (parsed.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    msg = choice.get("message") or {}
                    out_delta = {}
                    content = delta.get("content") or msg.get("content") or ""
                    if content:
                        acc += content
                        out_delta["content"] = content
                    reasoning = _delta_reasoning(delta, msg)
                    if reasoning:
                        reasoning_acc.append(reasoning)
                        out_delta["reasoning"] = reasoning
                        out_delta["reasoning_content"] = reasoning
                    if delta.get("tool_calls"):
                        out_delta["tool_calls"] = delta.get("tool_calls")
                    finish_reason = choice.get("finish_reason")
                    if out_delta or finish_reason:
                        yield _openai_sse_chunk(base, out_delta, finish_reason=finish_reason)
                    if finish_reason:
                        finish_sent = True
            if not finish_sent:
                yield _openai_sse_chunk(base, {}, finish_reason="stop")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            yield 'data: {"error": {"message": "chat server unavailable"}}\n\n'
        except Exception as exc:
            yield "data: " + json.dumps({
                "error": {"message": f"stream interrupted: {str(exc)[:200]}"},
            }) + "\n\n"
        finally:
            yield "data: [DONE]\n\n"
            try:
                _log_chat_turn(vllm_body or body, acc, king_uid, king_model, {
                    "choices": [{
                        "message": {
                            "content": acc,
                            "reasoning": "\n".join(reasoning_acc),
                        }
                    }]
                })
            except Exception:
                pass

    return _sse_response(generate())


def _stream_orchestrated_chat_response(payload: dict, king_uid: int | None, king_model: str | None):
    """Streaming variant for the legacy `/api/chat` endpoint."""
    async def generate():
        acc = ""
        vllm_body: dict | None = None
        try:
            vllm_body, tool_context, runtime_trace = await _prepare_orchestrated_chat(
                payload, king_uid, king_model, stream=True,
            )
            trace = "\n".join(runtime_trace)
            if trace:
                yield f"data: {json.dumps({'thinking': trace})}\n\n"
            if tool_context:
                prefix = _format_tool_context_for_display(tool_context) + "Raw model answer:\n\n"
                acc += prefix
                yield f"data: {json.dumps({'response': prefix, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"

            client = _get_http_client()
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_normalize_chat_payload(vllm_body),
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 400:
                    yield f"data: {json.dumps({'error': f'chat server returned {resp.status_code}'})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choice = (parsed.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    msg = choice.get("message") or {}
                    reasoning = _delta_reasoning(delta, msg)
                    if reasoning:
                        yield f"data: {json.dumps({'thinking': reasoning, 'delta': True})}\n\n"
                    content = delta.get("content") or msg.get("content") or ""
                    if content:
                        acc += content
                        yield f"data: {json.dumps({'response': content, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            yield f"data: {json.dumps({'error': 'chat server unavailable', 'detail': str(exc)[:200]})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)[:200]})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            try:
                _log_chat_turn(vllm_body or payload, acc, king_uid, king_model, None)
            except Exception:
                pass

    return _sse_response(generate())


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
    if _CHAT_ORCHESTRATION_ENABLED:
        data = await _orchestrated_chat_completion(payload, king_uid, king_model)
        message = data["choices"][0]["message"]
        resp = {
            "response": message.get("content") or "",
            "model": king_model,
            "king_uid": king_uid,
            "thinking": message.get("reasoning") or "",
            "usage": data.get("usage"),
        }
        _log_chat_turn(payload, resp["response"], king_uid, king_model, data)
        return resp
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
    if _CHAT_ORCHESTRATION_ENABLED:
        return _stream_orchestrated_chat_response(payload, king_uid, king_model)
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
    models = [{
        "id": CHAT_POD_SERVED_MODEL,
        "object": "model",
        "created": int(time.time()),
        "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
    }]
    if model_id != CHAT_POD_SERVED_MODEL:
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
        })
    return {
        "object": "list",
        "data": models,
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
    if _CHAT_ORCHESTRATION_ENABLED:
        if stream:
            return _stream_orchestrated_openai_response(body, king_uid, king_model)
        data = await _orchestrated_chat_completion(body, king_uid, king_model)
        return JSONResponse(content=data)

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
