"""Pre-flight non-python tools the current king can't invoke itself.

The current king is NOT fine-tuned to emit native ``tool_calls``, so it
can't invoke ``web_search`` / ``sn97_state`` / ``model_info`` on its
own. We detect user intent up-front via the intent regexes below and
call the underlying helpers, then inject the JSON results as an
authoritative user message so the agent loop sees the data as ground
truth. After the king is upgraded to emit native tool_calls these
pre-flights become redundant — the SDK loop will trigger the same
helpers via ``tool_choice="auto"``.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any

_SN97_RE = re.compile(
    r"\b(sn\s?97|sn-97|subnet|king|leaderboard|miner|uid|eval|round|"
    r"validator|arbos|distil)\b",
    re.IGNORECASE,
)
_WEB_SEARCH_RE = re.compile(
    r"(?:"
    r"\b(?:search\s+(?:the\s+)?web|web\s+search|look\s+up|google\s+for|google\s+the|"
    r"duckduckgo|find\s+me|fetch\s+(?:the\s+)?(?:latest|news|results?))\b"
    r"|\b(?:right\s+now|today|tonight|this\s+(?:morning|afternoon|evening|week|month|year))\b"
    r"|\b(?:latest|breaking|current|today'?s|recent|live|real[-\s]?time)\b"
    r"|\b(?:weather|forecast|temperature|stock\s+price|share\s+price|exchange\s+rate|"
    r"headline[s]?|news|score)\b"
    r"|\b(?:bitcoin|btc|ethereum|eth|tesla|gold|silver|oil|sp500|s&p\s*500)\b"
    r"|\bprice\s+of\s+\w+\b|\b\w+\s+price\b"
    r")",
    re.IGNORECASE,
)
_MODEL_PATH_RE = re.compile(
    r'"model_path"\s*:\s*"([^"]+)"|'
    r"\b([A-Za-z0-9._-]{2,40}/[A-Za-z0-9._-]{2,100})\b"
)
_SEARCH_PREFIX_RE = re.compile(
    r"^\s*("
    r"search\s+(?:the\s+)?web\s+for|web\s+search\s+for|"
    r"look\s+up|google\s+for|google\s+the|google|"
    r"find\s+me|fetch\s+(?:the\s+)?(?:latest|news|results?\s+(?:about|on|for))|"
    r"(?:can\s+you|could\s+you|please|tell\s+me)\s+"
    r"(?:search|find|look\s+up|google|fetch|tell\s+me)|"
    r"what(?:'s|\s+is)|what\s+are|what\s+were|"
    r"how\s+much\s+(?:is|does|costs?)|how\s+do|how\s+many|"
    r"who\s+(?:is|are|was|were)|where\s+(?:is|are|was|were)|"
    r"when\s+(?:is|are|was|were|did|does|do)|"
    r"explain|tell\s+me\s+about"
    r")\s+",
    re.IGNORECASE,
)


def _normalize_search_query(user_text: str) -> str:
    """Compact a chatty user prompt into a search-friendly query.

    DuckDuckGo's HTML endpoint returns zero hits when the query is too
    long, has trailing punctuation, or contains compound clauses like
    ``"X, and how much is Y worth?"``. We strip leading "what is",
    trailing punctuation, and chop everything after the first comma so
    the search engine sees the user's primary intent.
    """
    q = (user_text or "").strip()
    # Take only the first clause (before a comma or semicolon) -- the
    # rest is usually a follow-up like "and convert to USD" that the
    # search engine doesn't need.
    for sep in (",", ";", " and ", "?"):
        idx = q.find(sep) if sep != "?" else q.rfind(sep)
        if idx > 0 and (sep != "?" or idx == len(q) - 1):
            q = q[:idx]
            break
    q = q.strip()
    # Strip leading interrogative / search-request prefix.
    q = _SEARCH_PREFIX_RE.sub("", q).strip()
    # Strip trailing punctuation.
    q = q.rstrip("?!.,;:").strip()
    # Cap to ~12 words; longer queries usually have noise.
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    return q or (user_text or "").strip()


async def _preflight_tools(
    user_text: str, king_uid: int | None, king_model: str | None,
) -> tuple[list[str], list[str]]:
    """Run the non-python tools that the model can't invoke itself.

    Returns ``(context_blocks, runtime_trace)`` where ``context_blocks``
    is a list of pretty-printed result strings to inject as a system
    message and ``runtime_trace`` is human-readable lines for the
    reasoning panel.
    """
    blocks: list[str] = []
    trace: list[str] = []
    # Captured by the sn97_state branch below; used as a fallback
    # candidate for the implicit ``model_info`` pre-flight when the
    # user asks about size/params of "the king" without naming a path.
    king_model_path: str = ""
    if not user_text:
        return blocks, trace

    # web_search
    if _WEB_SEARCH_RE.search(user_text):
        try:
            from agent_tools import _parse_duckduckgo_html, _WEB_HEADERS, _WEB_TIMEOUT
            import httpx as _httpx
            import urllib.parse as _up
            q = _normalize_search_query(user_text)
            url = "https://duckduckgo.com/html/?" + _up.urlencode({"q": q})
            async with _httpx.AsyncClient(
                timeout=_WEB_TIMEOUT, headers=_WEB_HEADERS, follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                results = _parse_duckduckgo_html(resp.text, query=q, limit=5)
            blocks.append(
                "WEB_SEARCH_RESULT:\n" + json.dumps(
                    {"query": q, "results": results}, indent=2,
                )[:4000]
            )
            trace.append(f"web_search ({len(results)} hits)")
        except Exception as exc:
            blocks.append(f"WEB_SEARCH_ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            trace.append(f"web_search FAILED: {type(exc).__name__}")

    # sn97_state
    if _SN97_RE.search(user_text):
        try:
            from state_store import (
                eval_progress, h2h_latest, read_cache, top4_leaderboard,
                uid_hotkey_map,
            )
            top4 = top4_leaderboard() or {}
            king = dict(top4.get("king") or {})
            if king_uid is not None:
                king.setdefault("uid", king_uid)
            if king_model:
                king.setdefault("model", king_model)
            contenders = [dict(c) for c in (top4.get("contenders") or [])[:4]]
            progress = eval_progress() or {}
            h2h = h2h_latest() or {}
            payload: dict[str, Any] = {
                "king": king,
                "contenders": contenders,
                "eval_progress": {
                    "active": bool(progress.get("active") or progress.get("phase") not in (None, "", "idle")),
                    "phase": progress.get("phase") or progress.get("stage") or progress.get("status"),
                    "students_done": progress.get("students_done"),
                    "students_total": progress.get("students_total"),
                    "block": progress.get("completed_block") or h2h.get("block"),
                },
            }
            uid_match = re.search(r"\buid\s*[:=#]?\s*(\d{1,3})\b", user_text, re.IGNORECASE)
            if uid_match:
                with suppress(Exception):
                    uid = int(uid_match.group(1))
                    hotkey = uid_hotkey_map().get(str(uid))
                    commitments_data = read_cache("commitments", {}) or {}
                    commitments = commitments_data.get("commitments", commitments_data)
                    model = None
                    if hotkey and isinstance(commitments, dict):
                        c = commitments.get(hotkey) or {}
                        if isinstance(c, dict):
                            model = c.get("model") or c.get("repo")
                    payload["uid_lookup"] = {"uid": uid, "hotkey": hotkey, "model": model}
            blocks.append("SN97_LIVE_STATE:\n" + json.dumps(payload, indent=2)[:4000])
            trace.append(f"sn97_state (king UID={king.get('uid')})")
            # Capture king model path for the implicit ``model_info``
            # pre-flight below ("what's the king's param count?" etc.)
            with suppress(Exception):
                kmp = king.get("model")
                if isinstance(kmp, str) and "/" in kmp:
                    king_model_path = kmp
        except Exception as exc:
            trace.append(f"sn97_state FAILED: {type(exc).__name__}")

    # model_info -- explicit request OR implicit "size/params/bytes of
    # current king" follow-up that piggybacks on a prior sn97_state hit.
    explicit_model_info = (
        "get_model_info" in user_text
        or "model_path" in user_text
        or "tell me about" in user_text.lower()
    )
    implicit_king_size = bool(king_model_path) and bool(
        re.search(
            r"\b(size|params?|parameters?|bytes?|gb|mb|memory|"
            r"quantiz(?:e|ed|ation)|footprint|vram|disk)\b",
            user_text, re.IGNORECASE,
        )
    )
    if explicit_model_info or implicit_king_size:
        candidate_path = ""
        m = _MODEL_PATH_RE.search(user_text)
        if m:
            candidate_path = (m.group(1) or m.group(2) or "").strip()
        if not candidate_path and implicit_king_size:
            candidate_path = king_model_path or ""
        if "/" in candidate_path and len(candidate_path) >= 3:
            with suppress(Exception):
                from external import get_model_info as fetch_model_info_data
                info = fetch_model_info_data(candidate_path)
                if isinstance(info, dict):
                    blocks.append(
                        "MODEL_INFO_RESULT:\n" + json.dumps(
                            {**info, "model_path": candidate_path}, indent=2,
                        )[:2000]
                    )
                    trace.append(f"model_info ({candidate_path})")
    return blocks, trace


def _inject_preflight_context(
    inputs: list[dict], blocks: list[str],
) -> list[dict]:
    """Prepend a synthetic user turn carrying the pre-flight tool results.

    We use ``role=user`` (not ``system``) because the SDK passes the
    Agent's ``instructions`` as the only true system prompt; a second
    system message would either be merged or lost on some providers.
    Wrapping the data in a clearly delineated user turn keeps it visible
    to the model without poisoning the conversation history if the agent
    is long-running.
    """
    if not blocks:
        return inputs
    header = (
        "[runtime tool results — authoritative ground truth, pre-fetched "
        "by the chat orchestrator before this turn. Treat these values as "
        "true and reference them in your answer; do not contradict them "
        "or claim you cannot access them.]\n\n"
    )
    payload = header + "\n\n".join(blocks)
    augmented = list(inputs)
    # Splice before the LAST user turn so the data lands right next to
    # the question. Falls back to prepend if no user turn is present.
    insert_at = len(augmented)
    for i in range(len(augmented) - 1, -1, -1):
        if augmented[i].get("role") == "user":
            insert_at = i
            break
    augmented.insert(insert_at, {"role": "user", "content": payload})
    return augmented


__all__ = [
    "_MODEL_PATH_RE",
    "_SEARCH_PREFIX_RE",
    "_SN97_RE",
    "_WEB_SEARCH_RE",
    "_inject_preflight_context",
    "_normalize_search_query",
    "_preflight_tools",
]
