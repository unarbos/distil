from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import json
import traceback
import os
import threading
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
TMC_KEY = os.environ.get("TMC_API_KEY", "")
TMC_BASE = "https://api.taomarketcap.com"
TMC_HEADERS = {"Authorization": TMC_KEY}

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
DISK_CACHE_DIR = os.path.join(STATE_DIR, "api_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

# ── Disk-backed cache ────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")

def _disk_read(name: str):
    path = os.path.join(DISK_CACHE_DIR, f"{_safe_filename(name)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _disk_write(name: str, data):
    path = os.path.join(DISK_CACHE_DIR, f"{_safe_filename(name)}.json")
    with open(path, "w") as f:
        json.dump(data, f)

# In-memory caches (fast path)
_mem = {
    "metagraph": {"data": None, "ts": 0},
    "commitments": {"data": None, "ts": 0},
    "price": {"data": None, "ts": 0},
}

def _get_cached(name: str, ttl: int):
    """Return cached data if fresh enough, from memory or disk."""
    now = time.time()
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    mc = _mem[name]
    if mc["data"] and now - mc["ts"] < ttl:
        return mc["data"]
    # Try disk
    disk = _disk_read(name)
    if disk and now - disk.get("_ts", 0) < ttl:
        mc["data"] = disk
        mc["ts"] = disk.get("_ts", 0)
        return disk
    return None

def _set_cached(name: str, data: dict):
    now = time.time()
    data["_ts"] = now
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    _mem[name]["data"] = data
    _mem[name]["ts"] = now
    _disk_write(name, data)

def _get_stale(name: str):
    """Return ANY cached data, even stale — for fallback."""
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    mc = _mem[name]
    if mc["data"]:
        return mc["data"]
    return _disk_read(name)


# ── Background refresh (non-blocking) ────────────────────────────────────────

_refresh_lock = threading.Lock()
_refreshing = set()

def _bg_refresh(name: str, fn):
    """Refresh cache in background thread. Non-blocking."""
    if name in _refreshing:
        return
    def _do():
        try:
            _refreshing.add(name)
            result = fn()
            if result:
                _set_cached(name, result)
        except Exception as e:
            print(f"[bg_refresh] {name} failed: {e}")
        finally:
            _refreshing.discard(name)
    t = threading.Thread(target=_do, daemon=True)
    t.start()


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_metagraph():
    """Fetch metagraph via subprocess to avoid loading bittensor/torch in the API process."""
    import subprocess
    script = """
import bittensor as bt, json
sub = bt.Subtensor(network="finney")
meta = sub.metagraph(97)
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
print(json.dumps({"netuid": 97, "block": int(block), "n": int(meta.n), "neurons": neurons}))
"""
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"metagraph fetch failed: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    data["timestamp"] = time.time()
    return data

def _fetch_commitments():
    """Fetch commitments via subprocess to avoid loading bittensor/torch in the API process."""
    import subprocess
    script = """
import bittensor as bt, json
sub = bt.Subtensor(network="finney")
revealed = sub.get_all_revealed_commitments(97)
commits = {}
for hotkey, entries in revealed.items():
    if entries:
        block, data_str = entries[0]
        try:
            parsed = json.loads(data_str)
            commits[str(hotkey)] = {"block": block, **parsed}
        except Exception:
            commits[str(hotkey)] = {"block": block, "raw": str(data_str)}
print(json.dumps({"commitments": commits, "count": len(commits)}))
"""
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"commitments fetch failed: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    return data

def _fetch_price():
    data = req.get(f"{TMC_BASE}/public/v1/subnets/table/", headers=TMC_HEADERS, timeout=10).json()
    sn97 = next((item for item in data if item.get("subnet") == NETUID), None)
    if not sn97:
        raise ValueError("Subnet 97 not found")

    alpha_price_tao = sn97.get("price", 0)
    try:
        r = req.get("https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd", timeout=5)
        tao_usd = r.json().get("bittensor", {}).get("usd", 0)
    except Exception:
        tao_usd = (_get_stale("price") or {}).get("tao_usd", 0)

    miners_tao_per_day = sn97.get("miners_tao_per_day", 0) or 0

    return {
        "alpha_price_tao": round(alpha_price_tao, 6),
        "alpha_price_usd": round(alpha_price_tao * tao_usd, 4),
        "tao_usd": round(tao_usd, 2),
        "alpha_in_pool": round(sn97.get("alpha_liquidity", 0) / 1e9, 2),
        "tao_in_pool": round(sn97.get("tao_liquidity", 0) / 1e9, 2),
        "marketcap_tao": round(sn97.get("marketcap", 0), 2),
        "emission_pct": round(sn97.get("emission", 0), 4),
        "volume_tao": round(sn97.get("volume", 0), 2),
        "price_change_1h": round(sn97.get("price_difference_hour", 0), 2),
        "price_change_24h": round(sn97.get("price_difference_day", 0), 2),
        "price_change_7d": round(sn97.get("price_difference_week", 0), 2),
        "miners_tao_per_day": round(miners_tao_per_day, 4),
        "block_number": sn97.get("block_number", 0),
        "name": sn97.get("name", ""),
        "symbol": sn97.get("symbol", ""),
        "timestamp": time.time(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """API overview — shown when visiting api.arbos.life directly."""
    return {
        "name": "Distil — Subnet 97 API",
        "dashboard": "https://distil.arbos.life",
        "github": "https://github.com/unarbos/distil",
        "endpoints": {
            "/api/metagraph": "Full subnet metagraph (UIDs, stakes, weights, incentive)",
            "/api/commitments": "Miner model commitments (HuggingFace links)",
            "/api/scores": "Current KL scores, disqualifications, last eval details",
            "/api/price": "Token price, emission, market data",
            "/api/model-info/{repo}": "HuggingFace model card info (params, MoE, tags, etc.)",
            "/api/history": "Score history over time (for trendline chart)",
            "/api/eval-progress": "Live eval progress (phase, models, prompts done)",
            "/api/announcement": "Pending announcements (new king, etc.)",
            "/api/health": "Service health check",
        },
    }


@app.get("/api/metagraph")
def get_metagraph():
    # Fast: return cache immediately, refresh in background if stale
    cached = _get_cached("metagraph", CACHE_TTL)
    if cached:
        return cached
    # No fresh cache — return stale if available, and refresh in background
    stale = _get_stale("metagraph")
    if stale:
        _bg_refresh("metagraph", _fetch_metagraph)
        return stale
    # No cache at all — must block (first ever request)
    try:
        result = _fetch_metagraph()
        _set_cached("metagraph", result)
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/commitments")
def get_commitments():
    cached = _get_cached("commitments", CACHE_TTL)
    if cached:
        return cached
    stale = _get_stale("commitments")
    if stale:
        _bg_refresh("commitments", _fetch_commitments)
        return stale
    try:
        result = _fetch_commitments()
        _set_cached("commitments", result)
        return result
    except Exception as e:
        return {"commitments": {}, "count": 0, "error": str(e)}


@app.get("/api/scores")
def get_scores():
    result = {"scores": {}, "ema_scores": {}, "disqualified": {}, "last_eval": None, "last_eval_time": None, "tempo_seconds": 600}
    scores_path = os.path.join(STATE_DIR, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            s = json.load(f)
            result["scores"] = s
            result["ema_scores"] = s  # backward compat
    dq_path = os.path.join(STATE_DIR, "disqualified.json")
    if os.path.exists(dq_path):
        with open(dq_path) as f:
            result["disqualified"] = json.load(f)
    eval_path = os.path.join(STATE_DIR, "last_eval.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            result["last_eval"] = json.load(f)
        result["last_eval_time"] = os.path.getmtime(eval_path)
    return result


@app.get("/api/price")
def get_price():
    cached = _get_cached("price", 30)
    if cached:
        return cached
    stale = _get_stale("price")
    if stale:
        _bg_refresh("price", _fetch_price)
        return stale
    try:
        result = _fetch_price()
        _set_cached("price", result)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model-info/{model_path:path}")
def get_model_info(model_path: str):
    """Fetch HuggingFace model card info (cached 1h)."""
    cache_key = f"model_info:{model_path}"
    cached = _get_cached(cache_key, 3600)
    if cached:
        return cached
    try:
        import subprocess
        script = f"""
import json
from huggingface_hub import model_info as hf_model_info, hf_hub_download

info = hf_model_info("{model_path}", files_metadata=True)

params_b = None
if info.safetensors and hasattr(info.safetensors, "total"):
    params_b = round(info.safetensors.total / 1e9, 2)

active_params_b = None
is_moe = False
num_experts = None
num_active_experts = None
try:
    config_path = hf_hub_download(repo_id="{model_path}", filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    ne = config.get("num_local_experts", config.get("num_experts", 1))
    is_moe = ne > 1
    if is_moe:
        hidden = config.get("hidden_size", 0)
        num_experts = ne
        num_active_experts = config.get("num_experts_per_tok", config.get("num_active_experts", ne))
except Exception:
    pass

card = info.card_data
result = {{
    "model": "{model_path}",
    "author": info.author or "{model_path}".split("/")[0],
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
}}
print(json.dumps(result))
"""
        result_proc = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        if result_proc.returncode != 0:
            raise RuntimeError(result_proc.stderr[-300:])
        result = json.loads(result_proc.stdout)
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        return {"error": str(e), "model": model_path}


@app.get("/api/announcement")
def get_announcement():
    """Pending announcements (e.g., new king). Mark as posted via POST."""
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True):
                return ann
        except Exception:
            pass
    return {"type": None}


@app.post("/api/announcement/claim")
def claim_announcement():
    """Atomically claim a pending announcement — returns it and marks posted in one call."""
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True):
                ann["posted"] = True
                with open(ann_path, "w") as f:
                    json.dump(ann, f, indent=2)
                return ann
        except Exception:
            pass
    return {"type": None}


@app.post("/api/announcement/posted")
def mark_announcement_posted():
    """Mark the current announcement as posted (legacy compat)."""
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            ann["posted"] = True
            with open(ann_path, "w") as f:
                json.dump(ann, f, indent=2)
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}
    return {"ok": True, "note": "no announcement"}


@app.get("/api/eval-progress")
def get_eval_progress():
    """Live eval progress — what the validator is currently doing."""
    progress_path = os.path.join(STATE_DIR, "eval_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": False}


@app.get("/api/h2h-latest")
def get_h2h_latest():
    """Latest H2H round results — king vs challengers on same prompts."""
    path = os.path.join(STATE_DIR, "h2h_latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"error": "No H2H data yet"}


@app.get("/api/h2h-history")
def get_h2h_history():
    """H2H round history (last 50 rounds)."""
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


@app.get("/api/tmc-config")
def get_tmc_config():
    """SSE config — key is proxied server-side, not exposed to frontend."""
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "netuid": NETUID,
    }


@app.get("/api/history")
def get_history():
    """Return score history for trendline chart."""
    history_path = os.path.join(STATE_DIR, "score_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                return json.load(f)
        except Exception:
            return []
    return []


@app.get("/api/health")
def health():
    import time as _time
    # Check last eval time from h2h-latest
    last_eval_block = None
    last_eval_age_min = None
    eval_active = False
    try:
        h2h_path = STATE_DIR / "h2h_latest.json"
        if h2h_path.exists():
            h2h = json.loads(h2h_path.read_text())
            last_eval_block = h2h.get("block")
            ts = h2h.get("timestamp")
            if ts:
                last_eval_age_min = round((_time.time() - ts) / 60, 1)
        progress_path = STATE_DIR / "eval_progress.json"
        if progress_path.exists():
            prog = json.loads(progress_path.read_text())
            eval_active = prog.get("active", False)
    except Exception:
        pass
    return {
        "status": "ok",
        "netuid": NETUID,
        "has_metagraph_cache": _get_stale("metagraph") is not None,
        "has_commit_cache": _get_stale("commitments") is not None,
        "has_price_cache": _get_stale("price") is not None,
        "last_eval_block": last_eval_block,
        "last_eval_age_min": last_eval_age_min,
        "eval_active": eval_active,
    }


# ── Startup: prime caches ────────────────────────────────────────────────────

@app.on_event("startup")
def prime_caches():
    """On startup, kick off background refreshes so first request is fast."""
    _bg_refresh("metagraph", _fetch_metagraph)
    _bg_refresh("commitments", _fetch_commitments)
    _bg_refresh("price", _fetch_price)
