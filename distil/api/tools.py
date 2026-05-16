"""Chat-agent tools (preflight intent dispatch + agent loop tool calls)."""

from __future__ import annotations

import io
import re
import textwrap
import traceback
from contextlib import redirect_stdout
from typing import Any

import httpx

from distil.api.external import hf_model_info
from distil.settings import settings
from distil.state.store import store

# ── python_exec ─────────────────────────────────────────────────────────

_DENYLIST = (
    "open(",
    "subprocess",
    "socket",
    "import os",
    "import sys",
    "import shutil",
    "input(",
    "exec(",
    "eval(",
    "__import__",
)


def python_exec(code: str, *, timeout_s: int = 6) -> dict[str, Any]:
    """Execute a small Python snippet; return ``{ok, stdout, error}``."""
    if any(b in code for b in _DENYLIST):
        return {"ok": False, "stdout": "", "error": "blocked: dangerous module/builtin"}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(textwrap.dedent(code), {"__builtins__": __builtins__}, {})
        return {"ok": True, "stdout": buf.getvalue()[-8000:], "error": ""}
    except Exception:
        return {
            "ok": False,
            "stdout": buf.getvalue()[-2000:],
            "error": traceback.format_exc()[-2000:],
        }


# ── web_search ──────────────────────────────────────────────────────────


def web_search(query: str, *, n: int = 5) -> dict[str, Any]:
    if not settings.web_search_api_key:
        return {"ok": False, "error": "web_search disabled (no TAVILY_API_KEY)", "results": []}
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.web_search_api_key,
                "query": query,
                "max_results": int(n),
                "search_depth": "basic",
            },
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        return {
            "ok": True,
            "results": [
                {"title": x.get("title"), "url": x.get("url"), "content": x.get("content")}
                for x in results[:n]
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}


# ── sn97_state ──────────────────────────────────────────────────────────


def sn97_state(query: str = "") -> dict[str, Any]:
    """Single source of truth for live SN97 numbers (king, leaderboard, eval)."""
    return {
        "leaderboard": store.top4_leaderboard(),
        "h2h_latest": store.h2h_latest(),
        "eval_progress": store.eval_progress(),
        "king_history_tail": store.h2h_history()[-1:],
    }


# ── model_info ──────────────────────────────────────────────────────────


def model_info(model_repo: str) -> dict[str, Any]:
    return hf_model_info(model_repo)


# ── summarise_history ──────────────────────────────────────────────────


def summarise_history(n: int = 5) -> dict[str, Any]:
    rows = store.h2h_history()[-int(n) :]
    summary = []
    for r in rows:
        winner = r.get("king_after") or r.get("king_name")
        summary.append(
            {
                "block": r.get("block"),
                "winner": winner,
                "students": [
                    s.get("name") or s.get("model")
                    for s in (r.get("results") or r.get("students") or [])
                ],
                "broken_axes": r.get("broken_axes"),
            }
        )
    return {"recent_rounds": summary}


# ── Preflight intent dispatch ──────────────────────────────────────────


_PREFLIGHT = (
    (re.compile(r"\b(king|leader|leaderboard|score)\b", re.I), "sn97_state"),
    (re.compile(r"\b(model[- ]info|huggingface|hf model)\b", re.I), "model_info"),
    (re.compile(r"\b(search|google|web)\b", re.I), "web_search"),
    (re.compile(r"\b(history|recent rounds)\b", re.I), "summarise_history"),
)


def preflight_intent(text: str) -> str | None:
    for pattern, tool in _PREFLIGHT:
        if pattern.search(text or ""):
            return tool
    return None


TOOL_REGISTRY = {
    "python_exec": python_exec,
    "web_search": web_search,
    "sn97_state": sn97_state,
    "model_info": model_info,
    "summarise_history": summarise_history,
}
