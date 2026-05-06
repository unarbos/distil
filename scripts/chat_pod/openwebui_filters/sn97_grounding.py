"""
title: SN97 Subnet Grounding
author: distil
author_url: https://arbos.life
description: Pre-fetches live SN97 subnet data (king, leaderboard, eval status,
  miner details) from the local distil-api when the user asks an SN97-related
  question, and injects the data as a system message so the answering king has
  grounded facts to summarise. Also writes a synthetic ``reasoning`` blob on
  the assistant message in ``outlet`` so the Thinking pane shows what data was
  looked up — important because the current weak king never emits ``</think>``
  natively, leaving the Thinking pane empty.
required_open_webui_version: 0.4.0
version: 1.1.0
license: MIT
"""

# pylint: disable=too-few-public-methods,broad-except,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field


_LOG = logging.getLogger("sn97_grounding_filter")


_DEFAULT_API_BASES = (
    "http://127.0.0.1:3710",
    "http://172.17.0.1:3710",
    "http://host.docker.internal:3710",
)

# Keyword groups that trigger specific API lookups. Order matters only for
# readability — multiple groups can fire on the same prompt.
_RX_OVERVIEW = re.compile(
    r"\b("
    r"king|champion|winner|leader|leaderboard|"
    r"top|rank|ranking|"
    r"subnet|sn\s?97|sn-97|"
    r"validator|miner|miners|"
    r"score|scores|composite|"
    r"distil|arbos"
    r")\b",
    re.IGNORECASE,
)
_RX_EVAL = re.compile(
    r"\b("
    r"eval|evals|evaluation|evaluating|"
    r"round|rounds|"
    r"queue|progress|benchmark|benchmarks|"
    r"running|active|status|"
    r"h2h|head.to.head|dethron|dethrone"
    r")\b",
    re.IGNORECASE,
)
_RX_UID = re.compile(r"\buid\s*[:=#]?\s*(\d{1,3})\b", re.IGNORECASE)
_RX_HF_REPO = re.compile(r"\b([A-Za-z0-9._-]{2,40})/([A-Za-z0-9._-]{2,80})\b")


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=10, description="Lower runs earlier.")
        api_base: str = Field(
            default="http://127.0.0.1:3710",
            description=(
                "Base URL of distil-api. The filter falls back through a few "
                "common Docker-host candidates if this one fails."
            ),
        )
        timeout_s: float = Field(default=2.5)
        cache_ttl_s: float = Field(default=5.0)
        inject_thinking: bool = Field(
            default=True,
            description=(
                "Set ``message.reasoning`` on the assistant outlet so the "
                "Thinking pane shows the live SN97 data we looked up."
            ),
        )
        debug: bool = Field(default=False)

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._cache: dict[str, tuple[float, Any]] = {}
        # chat_id -> reasoning blob captured during ``inlet``, picked up in ``outlet``.
        self._reasoning_state: dict[str, str] = {}

    # ------------------------------------------------------------------
    # API fetch helpers
    # ------------------------------------------------------------------
    async def _fetch(self, path: str, params: Optional[dict] = None) -> Any:
        params = params or {}
        cache_key = f"{path}|" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self.valves.cache_ttl_s:
            return cached[1]

        bases = []
        primary = (self.valves.api_base or "").strip().rstrip("/")
        if primary:
            bases.append(primary)
        for fallback in _DEFAULT_API_BASES:
            if fallback not in bases:
                bases.append(fallback)

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for base in bases:
                try:
                    async with session.get(f"{base}{path}", params=params) as resp:
                        if resp.status >= 400:
                            continue
                        data = await resp.json(content_type=None)
                        self._cache[cache_key] = (now, data)
                        return data
                except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
                    if self.valves.debug:
                        _LOG.debug("fetch %s via %s failed: %s", path, base, exc)
                    continue
        return None

    # ------------------------------------------------------------------
    # Grounding builders
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_score(v) -> str:
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v) if v is not None else "?"

    async def _gather_grounding(self, user_text: str) -> Optional[str]:
        snippets: list[str] = []

        wants_overview = bool(_RX_OVERVIEW.search(user_text))
        wants_eval = bool(_RX_EVAL.search(user_text))
        uid_match = _RX_UID.search(user_text)
        # Match HF repo only if the user_text contains "/", to avoid accidental
        # matches on words like "and/or".
        hf_match = _RX_HF_REPO.search(user_text) if "/" in user_text else None

        # Always include a tiny health snippet — it tells the model what
        # subnet, what king, and how fresh the data is. Cheap.
        health = await self._fetch("/api/health")
        if health:
            snippets.append(
                "SUBNET_HEALTH: "
                f"netuid={health.get('netuid')} "
                f"king_uid={health.get('king_uid')} "
                f"king_h2h_kl={self._fmt_score(health.get('king_kl'))} "
                f"n_scored={health.get('n_scored')} "
                f"n_disqualified={health.get('n_disqualified')} "
                f"last_eval_age_min={health.get('last_eval_age_min')} "
                f"eval_active={health.get('eval_active')} "
                f"code_revision={health.get('code_revision')}"
            )

        if wants_overview:
            lb = await self._fetch("/api/leaderboard")
            if lb:
                board = lb.get("leaderboard", lb) or {}
                king = board.get("king") or {}
                if king:
                    comp = king.get("composite") or {}
                    snippets.append(
                        "CURRENT_KING: "
                        f"uid={king.get('uid')} "
                        f"model={king.get('model')} "
                        f"composite_final={self._fmt_score(comp.get('final'))} "
                        f"worst_axis={self._fmt_score(comp.get('worst'))} "
                        f"h2h_kl={self._fmt_score(king.get('h2h_kl'))}"
                    )
                top = (
                    board.get("top5")
                    or board.get("top4")
                    or board.get("rows")
                    or board.get("miners")
                    or []
                )
                if top:
                    rows = []
                    for r in top[:5]:
                        comp = (r.get("composite") or {}) if isinstance(r, dict) else {}
                        rows.append(
                            f"  uid={r.get('uid')} score={self._fmt_score(comp.get('final'))} {r.get('model','?')}"
                        )
                    snippets.append("TOP_MINERS:\n" + "\n".join(rows))

        if uid_match:
            uid = int(uid_match.group(1))
            mi = await self._fetch(f"/api/miner/{uid}")
            if mi:
                # Trim aggressively — miner blobs can be huge.
                snippets.append(
                    f"MINER_{uid}_DETAIL: " + json.dumps(mi)[:800]
                )

        if hf_match:
            repo = f"{hf_match.group(1)}/{hf_match.group(2)}"
            # Avoid matching obvious non-repos like "n/a".
            if len(hf_match.group(1)) >= 2 and len(hf_match.group(2)) >= 2:
                info = await self._fetch(f"/api/model-info/{repo}")
                if info:
                    snippets.append(f"MODEL_INFO[{repo}]: " + json.dumps(info)[:600])

        if wants_eval:
            ep = await self._fetch("/api/eval-progress")
            if ep:
                snippets.append("EVAL_PROGRESS: " + json.dumps(ep)[:400])
            qq = await self._fetch("/api/queue")
            if qq:
                snippets.append("EVAL_QUEUE: " + json.dumps(qq)[:400])
            kh = await self._fetch("/api/king-history", {"limit": 3})
            if kh:
                # Only the most recent dethronement is usually interesting.
                first = kh[0] if isinstance(kh, list) and kh else kh
                snippets.append(
                    "LATEST_DETHRONEMENT: " + json.dumps(first)[:500]
                )

        # Return None if nothing useful was found (only the always-on health
        # snippet) AND the user clearly wasn't asking about SN97. In that
        # case we shouldn't pollute the context.
        if (
            not wants_overview
            and not wants_eval
            and not uid_match
            and not hf_match
        ):
            return None

        if not snippets:
            return None

        return (
            "## Live SN97 data (use these facts, do not invent UIDs/scores):\n\n"
            + "\n\n".join(snippets)
            + "\n\n"
            "## Instructions\n"
            "Answer the user's question USING ONLY the facts above. If the "
            "data above doesn't cover what they asked, say so plainly — do "
            "NOT make up UIDs, model names, scores, or block numbers."
        )

    # ------------------------------------------------------------------
    # OWUI filter hooks
    # ------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        try:
            msgs = body.get("messages") or []
            if not msgs:
                return body

            last_user = None
            last_user_idx = -1
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if m.get("role") == "user":
                    last_user = m
                    last_user_idx = i
                    break
            if not last_user:
                return body

            content = last_user.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            if not isinstance(content, str) or not content.strip():
                return body

            grounding = await self._gather_grounding(content)
            if not grounding:
                return body

            sys_msg = {"role": "system", "content": grounding}
            msgs.insert(last_user_idx, sys_msg)
            body["messages"] = msgs

            if self.valves.inject_thinking:
                chat_id = (
                    body.get("chat_id")
                    or (body.get("metadata") or {}).get("chat_id")
                    or body.get("id")
                    or "noid"
                )
                first_q = content.strip().split("\n")[0][:160]
                self._reasoning_state[chat_id] = (
                    f'User asked: "{first_q}"\n\n'
                    f"Looking up live SN97 data via distil-api:\n\n"
                    f"{grounding}\n\n"
                    "Now drafting a concise, factual answer based on the data above."
                )
        except Exception as exc:
            _LOG.warning("sn97_grounding inlet failed: %s", exc, exc_info=self.valves.debug)
        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.inject_thinking:
            return body
        try:
            chat_id = (
                body.get("chat_id")
                or (body.get("metadata") or {}).get("chat_id")
                or body.get("id")
                or "noid"
            )
            reasoning = self._reasoning_state.pop(chat_id, None)
            if not reasoning:
                return body

            msgs = body.get("messages") or []
            last_assistant = None
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    last_assistant = m
                    break
            if last_assistant is None:
                return body

            existing = (
                last_assistant.get("reasoning")
                or last_assistant.get("reasoning_content")
                or ""
            )
            # Only inject when the model itself emitted nothing — never overwrite
            # genuine reasoning if a future stronger king starts producing it.
            if existing.strip():
                return body
            last_assistant["reasoning"] = reasoning
            last_assistant["reasoning_content"] = reasoning
        except Exception as exc:
            _LOG.warning("sn97_grounding outlet failed: %s", exc, exc_info=self.valves.debug)
        return body
