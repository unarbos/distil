"""FastAPI app factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from distil.api.chat import router as chat_router
from distil.api.compat import load_prod_routers
from distil.api.helpers import _Bucket, client_real_ip
from distil.api.routes import router as api_router
from distil.settings import settings

logger = logging.getLogger("distil.api.server")

_global_bucket = _Bucket(capacity=settings.api_rate_limit_per_minute, window_s=60)


def prime_caches() -> None:
    """Best-effort warm-up of the StateStore + external caches at boot."""
    from distil.api.external import tao_price
    from distil.state.store import store

    try:
        store.scores()
        store.h2h_latest()
        store.top4_leaderboard()
        tao_price()
    except Exception as exc:
        logger.warning(f"cache prime failed: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prime_caches()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Distil SN97 API (v2)",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def rate_limit_mw(request: Request, call_next):
        if request.url.path.startswith("/api/") or request.url.path.startswith("/v1/"):
            ip = client_real_ip(request)
            if not _global_bucket.hit(ip):
                return JSONResponse(
                    {"error": "rate limit exceeded"},
                    status_code=429,
                )
        start = time.time()
        try:
            response = await call_next(request)
        except HTTPException:
            raise
        except Exception:
            logger.exception(f"unhandled error on {request.url.path}")
            return JSONResponse({"error": "internal_server_error"}, status_code=500)
        response.headers["x-response-ms"] = f"{int((time.time() - start) * 1000)}"
        return response

    # Mount prod routers FIRST so they win first-match on any overlapping
    # path. This guarantees zero response-shape regression for the live
    # frontend during the rewrite cutover — the prod implementations remain
    # the source of truth for every endpoint the dashboard already calls.
    # Skipped silently if the prod ``api/`` package isn't on the filesystem.
    for r in load_prod_routers():
        app.include_router(r)
    # Native distil routers are mounted last and only "win" for new paths
    # that prod doesn't define (e.g. /api/miner/{uid}, /api/telemetry/*,
    # /api/incidents, /api/model-info/{owner}/{name}). Existing prod paths
    # are unaffected.
    app.include_router(api_router)
    app.include_router(chat_router)
    return app


app = create_app()
