from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import bittensor as bt
import time
import json
import traceback

app = FastAPI(title="Distillation Subnet API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NETUID = 97
CACHE_TTL = 60

_meta_cache = {"data": None, "ts": 0}
_commit_cache = {"data": None, "ts": 0}


@app.get("/api/metagraph")
def get_metagraph():
    now = time.time()
    if _meta_cache["data"] and now - _meta_cache["ts"] < CACHE_TTL:
        return _meta_cache["data"]

    try:
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
        # Return cached data if available, even if stale
        if _meta_cache["data"]:
            return _meta_cache["data"]
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/commitments")
def get_commitments():
    """Get all revealed commitments (miner model submissions)."""
    now = time.time()
    if _commit_cache["data"] and now - _commit_cache["ts"] < CACHE_TTL:
        return _commit_cache["data"]

    try:
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


@app.get("/api/health")
def health():
    return {"status": "ok", "netuid": NETUID, "cache_age_meta": time.time() - _meta_cache["ts"] if _meta_cache["ts"] else None}
