"""External integrations — HuggingFace model info + TAO price."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from distil.settings import settings
from distil.state.store import store

logger = logging.getLogger("distil.api.external")


def hf_model_info(repo: str) -> dict[str, Any]:
    cached = store.read_cache(f"hf_{repo.replace('/', '__')}", ttl=900)
    if cached is not None:
        return cached
    headers = {}
    if settings.hf_dl_token:
        headers["Authorization"] = f"Bearer {settings.hf_dl_token}"
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{repo}", headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning(f"hf_model_info({repo}) failed: {exc}")
        data = {"error": str(exc)}
    store.write_cache(f"hf_{repo.replace('/', '__')}", data)
    return data


def tao_price() -> dict[str, Any]:
    cached = store.read_cache("tao_price", ttl=120)
    if cached is not None:
        return cached
    try:
        r = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd",
            timeout=8,
        )
        r.raise_for_status()
        data = {"usd": float(r.json()["bittensor"]["usd"]), "ts": time.time()}
    except Exception as exc:
        logger.warning(f"tao_price failed: {exc}")
        data = {"usd": None, "ts": time.time(), "error": str(exc)}
    store.write_cache("tao_price", data)
    return data
