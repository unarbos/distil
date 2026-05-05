"""
title: SN97 Subnet Status
author: distil
author_url: https://arbos.life
funding_url: https://github.com/arbos-ai
description: Live SN97 subnet data — current king, leaderboard, miner info, eval round status.
required_open_webui_version: 0.4.0
version: 1.0.0
license: MIT
"""

# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes,too-few-public-methods

# ---------------------------------------------------------------------------
# Open-WebUI Tools class.
#
# Each public method on ``Tools`` becomes a callable function the king model
# can invoke through native OpenAI tool calling. Open-WebUI auto-generates the
# OpenAI ``function`` spec from the type hints + docstring, so we keep types
# explicit and docstrings short and prescriptive.
#
# All methods are async and return JSON strings (which is what Open-WebUI
# forwards back to the model as the tool message body). Networking is done
# with ``aiohttp`` (already a dep of Open-WebUI). We point at the public API
# at api.arbos.life so this works the same in dev and prod, no internal
# routing required.
#
# 2026-05-02: introduced for chat.arbos.life native tool calling. See
# scripts/chat_pod/configure_webui_tools.py for installation logic.
# ---------------------------------------------------------------------------

import json
from typing import Optional

import aiohttp


_API_BASE = "https://api.arbos.life"
_REQ_TIMEOUT = aiohttp.ClientTimeout(total=12)


def _trim(obj, max_chars: int = 6000):
    """Best-effort truncate of a JSON-serialisable payload to keep tool
    messages under the king's effective context budget. Returns the original
    object when it serialises under ``max_chars``."""
    try:
        s = json.dumps(obj)
    except (TypeError, ValueError):
        return obj
    if len(s) <= max_chars:
        return obj
    return {
        "_truncated": True,
        "_original_chars": len(s),
        "preview": s[:max_chars],
    }


async def _get_json(path: str, params: Optional[dict] = None):
    url = f"{_API_BASE}{path}"
    async with aiohttp.ClientSession(timeout=_REQ_TIMEOUT) as session:
        async with session.get(url, params=params) as resp:
            text = await resp.text()
            if resp.status >= 400:
                return {
                    "error": f"HTTP {resp.status}",
                    "url": str(resp.url),
                    "body": text[:1000],
                }
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text[:2000]}


class Tools:
    def __init__(self):
        self.citation = True

    async def get_subnet_overview(self) -> str:
        """Get a one-shot overview of the SN97 subnet: current king model,
        top of leaderboard, and current eval round status. Use this whenever
        a user asks generic questions about "what's happening on SN97" or
        "who is the current king"."""
        async with aiohttp.ClientSession(timeout=_REQ_TIMEOUT) as session:
            async def fetch(path):
                async with session.get(f"{_API_BASE}{path}") as r:
                    text = await r.text()
                    if r.status >= 400:
                        return {"error": f"HTTP {r.status}", "path": path}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw": text[:500]}

            leaderboard, king_history, eval_status, eval_stats = (
                await fetch("/api/leaderboard"),
                await fetch("/api/king-history"),
                await fetch("/api/eval-status"),
                await fetch("/api/eval-stats"),
            )

        # /api/leaderboard returns ``{"leaderboard": {"king": {...}, "miners": [...]}}``
        lb_data = leaderboard.get("leaderboard") if isinstance(leaderboard, dict) else None
        if not isinstance(lb_data, dict):
            lb_data = {}

        current_king = lb_data.get("king") or lb_data.get("current_king")

        # Trim the king's verbose composite/axes blob — keep only the
        # high-signal fields. The model can call get_miner_info if it wants
        # the full picture.
        if isinstance(current_king, dict):
            comp = (current_king.get("composite") or {})
            current_king = {
                "uid": current_king.get("uid"),
                "model": current_king.get("model"),
                "h2h_kl": current_king.get("h2h_kl"),
                "block": current_king.get("block"),
                "composite_final": comp.get("final"),
                "composite_weighted": comp.get("weighted"),
            }

        # /api/leaderboard contender rows have ``uid``, ``model``, ``h2h_kl``,
        # ``composite``. Trim to a comparable shape across endpoints.
        contenders = lb_data.get("contenders") or []
        if not isinstance(contenders, list):
            contenders = []
        top_5 = []
        for m in contenders[:5]:
            if not isinstance(m, dict):
                continue
            comp = m.get("composite") or {}
            top_5.append(
                {
                    "uid": m.get("uid"),
                    "model": m.get("model"),
                    "h2h_kl": m.get("h2h_kl"),
                    "composite_final": comp.get("final"),
                }
            )

        # /api/eval-status has a long ``statuses`` map keyed by uid; surface
        # just the king + a small count of pending tests.
        eval_summary = None
        if isinstance(eval_status, dict):
            statuses = eval_status.get("statuses") or {}
            counts: dict[str, int] = {}
            for entry in statuses.values():
                s = (entry or {}).get("status", "unknown")
                counts[s] = counts.get(s, 0) + 1
            eval_summary = {
                "king_uid": eval_status.get("king_uid"),
                "block": eval_status.get("block"),
                "status_counts": counts,
                "miners_total": len(statuses),
            }

        # /api/king-history is most-recent-first. Surface the last 3
        # dethronements as a "what changed recently" hint.
        recent_changes = []
        if isinstance(king_history, list):
            for ev in king_history[:3]:
                if not isinstance(ev, dict):
                    continue
                recent_changes.append(
                    {
                        "block": ev.get("block"),
                        "old_king_uid": ev.get("old_king_uid"),
                        "new_king_uid": ev.get("new_king_uid"),
                        "new_king_model": ev.get("new_king_model"),
                        "p_value": ev.get("p_value"),
                    }
                )

        # /api/eval-stats has aggregate timing / counts.
        last_round = (eval_stats or {}).get("last_round") if isinstance(eval_stats, dict) else None

        out = {
            "current_king": current_king,
            "top_5": top_5,
            "eval_status": eval_summary,
            "last_round": last_round,
            "recent_dethronements": recent_changes,
            "links": {
                "dashboard": "https://arbos.life",
                "leaderboard_api": f"{_API_BASE}/api/leaderboard",
            },
        }
        return json.dumps(_trim(out))

    async def get_leaderboard(self, limit: int = 10) -> str:
        """Get the SN97 miner leaderboard sorted by composite score.

        :param limit: How many rows to return (1..50). Defaults to 10.
        """
        n = max(1, min(int(limit or 10), 50))
        data = await _get_json("/api/leaderboard")
        # /api/leaderboard returns
        #   {"leaderboard": {"king": {...}, "contenders": [...], ...}}
        lb = data.get("leaderboard") if isinstance(data, dict) else None
        if not isinstance(lb, dict):
            return json.dumps({"error": "unexpected leaderboard shape", "raw": str(data)[:500]})
        contenders = lb.get("contenders") or []
        trimmed = []
        for m in contenders[:n]:
            if not isinstance(m, dict):
                continue
            comp = m.get("composite") or {}
            trimmed.append(
                {
                    "uid": m.get("uid"),
                    "model": m.get("model"),
                    "h2h_kl": m.get("h2h_kl"),
                    "composite_final": comp.get("final"),
                    "composite_weighted": comp.get("weighted"),
                }
            )
        king = lb.get("king") or {}
        out = {
            "king_uid": king.get("uid") if isinstance(king, dict) else None,
            "king_model": king.get("model") if isinstance(king, dict) else None,
            "contenders": trimmed,
            "count": len(trimmed),
            "phase": lb.get("phase"),
        }
        return json.dumps(_trim(out))

    async def get_miner_info(self, uid: int) -> str:
        """Get full info about a specific SN97 miner (their model, scores,
        recent rounds).

        :param uid: Miner UID (integer 0..255).
        """
        try:
            uid_int = int(uid)
        except (TypeError, ValueError):
            return json.dumps({"error": "uid must be an integer"})
        if not 0 <= uid_int <= 255:
            return json.dumps({"error": "uid out of range (0..255)"})

        async with aiohttp.ClientSession(timeout=_REQ_TIMEOUT) as session:
            async def fetch(path):
                async with session.get(f"{_API_BASE}{path}") as r:
                    text = await r.text()
                    if r.status >= 400:
                        return {"error": f"HTTP {r.status}", "path": path}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw": text[:500]}

            miner, rounds = (
                await fetch(f"/api/miner/{uid_int}"),
                await fetch(f"/api/miner/{uid_int}/rounds"),
            )

        if isinstance(rounds, list):
            rounds = rounds[:5]
        elif isinstance(rounds, dict):
            rounds = rounds.get("rounds", [])[:5]

        return json.dumps(_trim({"miner": miner, "recent_rounds": rounds}))

    async def get_eval_status(self) -> str:
        """Get the current SN97 evaluation round status: round number,
        progress, ETA, and which model is being evaluated."""
        data = await _get_json("/api/eval-status")
        return json.dumps(_trim(data))

    async def get_model_info(self, model_path: str) -> str:
        """Look up info about a HuggingFace model that has been evaluated on
        SN97 (parameter count, architecture, scores).

        :param model_path: HuggingFace ``user/repo`` style path.
        """
        model_path = (model_path or "").strip().strip("/")
        if not model_path or "/" not in model_path:
            return json.dumps({"error": "model_path must look like 'user/repo'"})
        data = await _get_json(f"/api/model-info/{model_path}")
        return json.dumps(_trim(data))

    async def get_eval_stats(self) -> str:
        """Get aggregate SN97 evaluation statistics: distribution of scores,
        recent round summaries, miner counts."""
        data = await _get_json("/api/eval-stats")
        return json.dumps(_trim(data))

    async def get_announcement(self) -> str:
        """Get the latest pinned subnet announcement (if any)."""
        data = await _get_json("/api/announcement")
        return json.dumps(_trim(data, max_chars=2000))
