"""Chat surface — ``/api/chat`` + OpenAI-compatible ``/v1/*`` endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from distil.api.agent import run_agent_chat, stream_agent_chat
from distil.api.helpers import _Bucket, client_real_ip, sanitize_text
from distil.settings import settings
from distil.state.files import append_jsonl
from distil.state.store import store

logger = logging.getLogger("distil.api.chat")

router = APIRouter()
_chat_bucket = _Bucket(capacity=settings.api_chat_rate_limit_per_minute, window_s=60)


def _ratelimit(request: Request) -> None:
    if not _chat_bucket.hit(client_real_ip(request)):
        raise HTTPException(status_code=429, detail="chat rate limit exceeded")


def _audit(req_payload: dict, resp_text: str, ip: str) -> None:
    try:
        append_jsonl(
            settings.chat_audit_log,
            {
                "ts": time.time(),
                "ip": ip,
                "model": req_payload.get("model"),
                "messages": [
                    {"role": m.get("role"), "len": len(m.get("content") or "")}
                    for m in req_payload.get("messages") or []
                ],
                "response_chars": len(resp_text or ""),
            },
            max_rows=20000,
        )
    except Exception as exc:
        logger.debug(f"chat audit failed: {exc}")


def _normalise_messages(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    total_chars = 0
    for m in raw or []:
        role = m.get("role") or "user"
        content = sanitize_text(m.get("content") or "")
        if not content:
            continue
        if total_chars + len(content) > 24000:
            content = content[: max(0, 24000 - total_chars)]
        total_chars += len(content)
        out.append({"role": role, "content": content})
    return out


@router.get("/api/chat/status")
def chat_status() -> dict[str, Any]:
    h2h = store.h2h_latest() or {}
    return {
        "king_model": h2h.get("king_after") or h2h.get("king_name"),
        "block": h2h.get("block"),
        "model_name": settings.chat_model_name,
        "pod_url": settings.chat_pod_url.rsplit(":", 1)[0],
    }


@router.post("/api/chat")
async def chat(request: Request) -> JSONResponse:
    _ratelimit(request)
    body = await request.json()
    messages = _normalise_messages(body.get("messages") or [])
    if not messages:
        raise HTTPException(status_code=400, detail="no messages")
    try:
        msg = run_agent_chat(messages)
    except Exception as exc:
        logger.warning(f"chat upstream failed: {exc}")
        raise HTTPException(status_code=503, detail=f"chat pod unavailable: {exc}") from exc
    _audit(body, msg.get("content") or "", client_real_ip(request))
    return JSONResponse({"message": msg})


# ── OpenAI-compatible surface ───────────────────────────────────────────


@router.get("/v1/models")
def v1_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": settings.chat_model_name, "object": "model", "owned_by": "sn97"}],
    }


@router.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    _ratelimit(request)
    body = await request.json()
    messages = _normalise_messages(body.get("messages") or [])
    if not messages:
        raise HTTPException(status_code=400, detail="no messages")
    if body.get("stream"):
        try:
            it = stream_agent_chat(messages, model=body.get("model"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return StreamingResponse(it, media_type="text/event-stream")
    try:
        msg = run_agent_chat(messages, model=body.get("model"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _audit(body, msg.get("content") or "", client_real_ip(request))
    return JSONResponse(
        {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or settings.chat_model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        }
    )
