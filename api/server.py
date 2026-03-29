from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import json
import traceback
import os
import requests as req

app = FastAPI(title="Distillation Subnet API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NETUID = 97
CACHE_TTL = 60
TMC_KEY = "***REMOVED***"
TMC_BASE = "https://api.taomarketcap.com"
TMC_HEADERS = {"Authorization": TMC_KEY}

# ── Caches ────────────────────────────────────────────────────────────────────
_meta_cache = {"data": None, "ts": 0}
_commit_cache = {"data": None, "ts": 0}
_price_cache = {"data": None, "ts": 0}
_table_cache = {"data": None, "ts": 0}


def _tmc_get(path: str, timeout: int = 10):
    """GET from TaoMarketCap API."""
    r = req.get(f"{TMC_BASE}{path}", headers=TMC_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── Metagraph via TMC neurons endpoint ────────────────────────────────────────

@app.get("/api/metagraph")
def get_metagraph():
    """Get metagraph from TMC (no bittensor SDK needed)."""
    now = time.time()
    if _meta_cache["data"] and now - _meta_cache["ts"] < CACHE_TTL:
        return _meta_cache["data"]

    try:
        import bittensor as bt
        sub = bt.Subtensor(network="finney")
        meta = sub.metagraph(NETUID)
        block = sub.block

        neurons = []
        for uid in range(meta.n):
            neurons.append({
                "uid": uid,
                "hotkey": str(meta.hotkeys[uid]),
                "coldkey": str(meta.coldkeys[uid]),
                "stake": float(meta.S[uid]),
                "trust": float(meta.T[uid]),
                "consensus": float(meta.C[uid]),
                "incentive": float(meta.I[uid]),
                "emission": float(meta.E[uid]),
                "dividends": float(meta.D[uid]),
                "is_validator": float(meta.S[uid]) > 1000,
            })

        result = {
            "netuid": NETUID,
            "block": int(block),
            "n": int(meta.n),
            "neurons": neurons,
            "timestamp": now,
        }
        _meta_cache["data"] = result
        _meta_cache["ts"] = now
        return result
    except Exception as e:
        if _meta_cache["data"]:
            return _meta_cache["data"]
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Commitments (still needs bt SDK for now) ──────────────────────────────────

@app.get("/api/commitments")
def get_commitments():
    """Get all revealed commitments (miner model submissions)."""
    now = time.time()
    if _commit_cache["data"] and now - _commit_cache["ts"] < CACHE_TTL:
        return _commit_cache["data"]

    try:
        import bittensor as bt
        sub = bt.Subtensor(network="finney")
        revealed = sub.get_all_revealed_commitments(NETUID)
        commits = {}
        for hotkey, entries in revealed.items():
            if entries:
                block, data = entries[0]
                try:
                    parsed = json.loads(data)
                    commits[str(hotkey)] = {"block": block, **parsed}
                except Exception:
                    commits[str(hotkey)] = {"block": block, "raw": str(data)}

        result = {"commitments": commits, "count": len(commits)}
        _commit_cache["data"] = result
        _commit_cache["ts"] = now
        return result
    except Exception as e:
        if _commit_cache["data"]:
            return _commit_cache["data"]
        return {"commitments": {}, "count": 0, "error": str(e)}


# ── Scores (local file) ──────────────────────────────────────────────────────

@app.get("/api/scores")
def get_scores():
    """Get latest eval scores from state files."""
    state_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
    result = {"ema_scores": {}, "last_eval": None}

    scores_path = os.path.join(state_dir, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            result["ema_scores"] = json.load(f)

    eval_path = os.path.join(state_dir, "last_eval.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            result["last_eval"] = json.load(f)

    return result


# ── Price via TaoMarketCap (no Subtensor!) ────────────────────────────────────

@app.get("/api/price")
def get_price():
    """Get SN97 alpha price, TAO/USD, market data from TaoMarketCap."""
    now = time.time()
    if _price_cache["data"] and now - _price_cache["ts"] < 30:
        return _price_cache["data"]

    try:
        data = _tmc_get(f"/public/v1/subnets/table/")
        sn97 = None
        for item in data:
            if item.get("subnet") == NETUID:
                sn97 = item
                break

        if not sn97:
            raise ValueError("Subnet 97 not found in TMC table")

        alpha_price_tao = sn97.get("price", 0)

        # Get TAO/USD from coingecko (lightweight)
        try:
            r = req.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd",
                timeout=5,
            )
            tao_usd = r.json().get("bittensor", {}).get("usd", 0)
        except Exception:
            tao_usd = _price_cache.get("data", {}).get("tao_usd", 0) if _price_cache["data"] else 0

        alpha_price_usd = alpha_price_tao * tao_usd
        alpha_in_pool = sn97.get("alpha_liquidity", 0) / 1e9
        tao_in_pool = sn97.get("tao_liquidity", 0) / 1e9

        result = {
            "alpha_price_tao": round(alpha_price_tao, 6),
            "alpha_price_usd": round(alpha_price_usd, 4),
            "tao_usd": round(tao_usd, 2),
            "alpha_in_pool": round(alpha_in_pool, 2),
            "tao_in_pool": round(tao_in_pool, 2),
            "marketcap_tao": round(sn97.get("marketcap", 0), 2),
            "emission_pct": round(sn97.get("emission", 0), 4),
            "volume_tao": round(sn97.get("volume", 0), 2),
            "price_change_1h": round(sn97.get("price_difference_hour", 0), 2),
            "price_change_24h": round(sn97.get("price_difference_day", 0), 2),
            "price_change_7d": round(sn97.get("price_difference_week", 0), 2),
            "miners_tao_per_day": round(sn97.get("miners_tao_per_day", 0), 2),
            "block_number": sn97.get("block_number", 0),
            "name": sn97.get("name", ""),
            "symbol": sn97.get("symbol", ""),
            "timestamp": now,
        }
        _price_cache["data"] = result
        _price_cache["ts"] = now
        return result
    except Exception as e:
        if _price_cache["data"]:
            return _price_cache["data"]
        return {"error": str(e)}


# ── TMC SSE config (for frontend to connect directly) ────────────────────────

@app.get("/api/tmc-config")
def get_tmc_config():
    """Return TMC SSE endpoint config for frontend to connect directly."""
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "api_key": TMC_KEY,
        "netuid": NETUID,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "netuid": NETUID,
        "cache_age_meta": time.time() - _meta_cache["ts"] if _meta_cache["ts"] else None,
        "cache_age_price": time.time() - _price_cache["ts"] if _price_cache["ts"] else None,
        "price_source": "taomarketcap",
    }
