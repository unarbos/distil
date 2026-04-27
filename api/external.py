import json
import os
import sys
import threading

from config import CACHE_TTL
from helpers.cache import _bg_refresh, _get_cached, _get_stale, _set_cached
from helpers.fetch import _fetch_commitments, _fetch_metagraph, _fetch_price


# Cap concurrent HF model_info subprocesses. The dashboard fans out
# /api/model-info/{repo} for every UID on every page load; before this cap a
# single page render would spawn 25+ subprocess calls and saturate uvicorn's
# thread pool, causing /api/leaderboard to 503 with "Exceeded concurrency limit".
_MODEL_INFO_SEM = threading.Semaphore(int(os.environ.get("DISTIL_API_HF_PARALLEL", "4")))
_MODEL_INFO_INFLIGHT_LOCK = threading.Lock()
_MODEL_INFO_INFLIGHT: set[str] = set()
_MODEL_INFO_TTL = 3600
_MODEL_INFO_STALE_TTL = 7 * 24 * 3600


def _cached(name, ttl, fetcher, fallback):
    cached = _get_cached(name, ttl)
    if cached:
        return cached
    stale = _get_stale(name)
    if stale:
        _bg_refresh(name, fetcher)
        return stale
    try:
        result = fetcher()
        _set_cached(name, result)
        return result
    except Exception as exc:
        return fallback(exc)


def get_commitments():
    return _cached("commitments", CACHE_TTL, _fetch_commitments, lambda exc: {"commitments": {}, "count": 0, "error": str(exc)})


def get_metagraph():
    return _cached("metagraph", CACHE_TTL, _fetch_metagraph, lambda exc: {"error": str(exc)})


def get_price():
    return _cached("price", 30, _fetch_price, lambda exc: {"error": str(exc)})


_MODEL_INFO_SCRIPT = """
import json, os
from huggingface_hub import model_info as hf_model_info, hf_hub_download

model_path = os.environ["MODEL_PATH"]
info = hf_model_info(model_path, files_metadata=True)

params_b = None
if info.safetensors and hasattr(info.safetensors, "total"):
    params_b = round(info.safetensors.total / 1e9, 2)

active_params_b = None
is_moe = False
num_experts = None
num_active_experts = None
try:
    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    ne = config.get("num_local_experts", config.get("num_experts", 1))
    is_moe = ne > 1
    if is_moe:
        num_experts = ne
        num_active_experts = config.get("num_experts_per_tok", config.get("num_active_experts", ne))
except Exception:
    pass

card = info.card_data
print(json.dumps({
    "model": model_path,
    "author": info.author or model_path.split("/")[0],
    "tags": list(info.tags) if info.tags else [],
    "downloads": info.downloads,
    "likes": info.likes,
    "created_at": info.created_at.isoformat() if info.created_at else None,
    "last_modified": info.last_modified.isoformat() if info.last_modified else None,
    "params_b": params_b,
    "active_params_b": active_params_b,
    "is_moe": is_moe,
    "num_experts": num_experts,
    "num_active_experts": num_active_experts,
    "license": getattr(card, "license", None) if card else None,
    "pipeline_tag": info.pipeline_tag,
    "base_model": getattr(card, "base_model", None) if card else None,
}))
"""


def _fetch_model_info(model_path):
    """Spawn the HF subprocess. Bounded by ``_MODEL_INFO_SEM``."""
    import subprocess

    env = os.environ.copy()
    env["MODEL_PATH"] = model_path
    with _MODEL_INFO_SEM:
        result = subprocess.run(
            [sys.executable, "-c", _MODEL_INFO_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:] or "hf model_info failed")
    return json.loads(result.stdout)


def get_model_info(model_path):
    """Stale-while-refresh wrapper around the HF subprocess.

    The dashboard fans out /api/model-info/{repo} for every miner on every page
    render. Synchronously waiting for HF on each one used to saturate uvicorn's
    request thread pool and stall /api/leaderboard with HTTP 503 ("Exceeded
    concurrency limit"). Now we serve any cached/stale entry immediately and
    schedule a single background refresh per repo so the request thread frees
    up in microseconds.
    """
    cache_key = f"model_info:{model_path}"
    fresh = _get_cached(cache_key, _MODEL_INFO_TTL)
    if fresh:
        return fresh

    def _refresh():
        try:
            return _fetch_model_info(model_path)
        except Exception as exc:
            return {"error": str(exc), "model": model_path}

    stale = _get_stale(cache_key)
    if stale:
        with _MODEL_INFO_INFLIGHT_LOCK:
            already = model_path in _MODEL_INFO_INFLIGHT
            if not already:
                _MODEL_INFO_INFLIGHT.add(model_path)
        if not already:
            def _bg():
                try:
                    data = _refresh()
                    if data and not data.get("error"):
                        _set_cached(cache_key, data)
                finally:
                    with _MODEL_INFO_INFLIGHT_LOCK:
                        _MODEL_INFO_INFLIGHT.discard(model_path)
            threading.Thread(target=_bg, daemon=True).start()
        return stale

    try:
        data = _fetch_model_info(model_path)
        _set_cached(cache_key, data)
        return data
    except Exception as exc:
        return {"error": str(exc), "model": model_path}
