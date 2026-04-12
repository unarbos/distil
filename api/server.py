from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json
import math
import traceback
import os
import threading
import requests as req
from collections import defaultdict
import time as _rate_time


def _sanitize_floats(obj):
    """Replace inf/nan floats with None so JSON serialization doesn't break."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


# ── Rate Limiting ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = _rate_time.time()
        window_start = now - self.window_sec
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

_rate_limiter = RateLimiter(max_requests=60, window_sec=60)
_chat_rate_limiter = RateLimiter(max_requests=10, window_sec=60)  # Stricter for chat

# Load .env from repo root
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

API_DESCRIPTION = """
# Distil — Subnet 97 API

Public API for [Distil](https://distil.arbos.life), a Bittensor subnet where miners compete to produce the best knowledge-distilled small language models.

## How It Works

Miners submit distilled models (currently ≤4B params, based on Qwen 3.5). A validator evaluates them head-to-head against the reigning **king** model using KL-divergence on shared prompts. Lower KL = better distillation = higher rewards.

## Quick Start

```bash
# Who's the current king?
curl https://api.arbos.life/api/health

# Get all miner scores
curl https://api.arbos.life/api/scores

# Get token price
curl https://api.arbos.life/api/price
```

## Links

- **Dashboard**: [distil.arbos.life](https://distil.arbos.life)
- **GitHub**: [github.com/unarbos/distil](https://github.com/unarbos/distil)
- **TaoMarketCap**: [taomarketcap.com/subnets/97](https://taomarketcap.com/subnets/97)
- **Twitter**: [@arbos_born](https://x.com/arbos_born)
"""

app = FastAPI(
    title="Distil — Subnet 97 API",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Overview", "description": "API info and health checks"},
        {"name": "Metagraph", "description": "On-chain subnet data — UIDs, stakes, weights, incentive"},
        {"name": "Miners", "description": "Miner model commitments and scores"},
        {"name": "Evaluation", "description": "Live eval progress, head-to-head rounds, and score history"},
        {"name": "Market", "description": "Token pricing, emission, and market data"},
        {"name": "Chat", "description": "Chat with the current king model (when GPU is available)"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://distil.arbos.life", "http://localhost:3000", "http://localhost:5173"],
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

def _safe_json_load(path: str, default=None):
    """Load JSON file, returning default on any error (missing, empty, corrupt)."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 2:
            return default
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def _safe_filename(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")

def _disk_read(name: str):
    path = os.path.join(DISK_CACHE_DIR, f"{_safe_filename(name)}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Corrupt cache file — delete it silently
            try:
                os.remove(path)
            except OSError:
                pass
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
import bittensor as bt, json, sys
sub = bt.Subtensor(network="finney")
revealed = sub.get_all_revealed_commitments(97)
commits = {}
for hotkey, entries in revealed.items():
    if not entries:
        continue
    try:
        # Take the LATEST revealed commitment for this hotkey.
        # Using the first/original entry leaves the dashboard permanently stale
        # when a miner updates their model.
        block, data_str = max(entries, key=lambda x: x[0])
    except (ValueError, TypeError) as e:
        print(f"[commitments] bad entry for {hotkey}: {e}", file=sys.stderr)
        continue
    # Robust parsing: try JSON first, then hex decode, then raw string
    parsed = None
    if isinstance(data_str, str):
        # Try JSON directly
        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try hex-encoded JSON
        if parsed is None and data_str.startswith("0x"):
            try:
                decoded = bytes.fromhex(data_str[2:]).decode("utf-8")
                parsed = json.loads(decoded)
            except Exception:
                pass
        # Try hex without prefix
        if parsed is None:
            try:
                decoded = bytes.fromhex(data_str).decode("utf-8")
                parsed = json.loads(decoded)
            except Exception:
                pass
    elif isinstance(data_str, dict):
        parsed = data_str
    if parsed and isinstance(parsed, dict):
        commits[str(hotkey)] = {"block": block, **parsed}
    else:
        print(f"[commitments] unparseable for {hotkey}: {str(data_str)[:100]}", file=sys.stderr)
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


@app.get("/", include_in_schema=False)
def root():
    """Redirect to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/api/metagraph", tags=["Metagraph"], summary="Full subnet metagraph",
         description="""Returns all 256 UIDs with on-chain data: hotkey, coldkey, stake, trust, consensus, incentive, emission, and dividends.

**Cached for 60s** — background refreshes keep data fresh without blocking requests.

Response includes:
- `block`: Current Bittensor block number
- `n`: Number of UIDs in the subnet (256)
- `neurons[]`: Array of all UIDs with their on-chain metrics
""",
         response_description="Metagraph with all 256 UIDs and their on-chain metrics")
def get_metagraph():
    # Fast: return cache immediately, refresh in background if stale
    cached = _get_cached("metagraph", CACHE_TTL)
    if cached:
        return JSONResponse(content=cached, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    # No fresh cache — return stale if available, and refresh in background
    stale = _get_stale("metagraph")
    if stale:
        _bg_refresh("metagraph", _fetch_metagraph)
        return JSONResponse(content=stale, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    # No cache at all — must block (first ever request)
    try:
        result = _fetch_metagraph()
        _set_cached("metagraph", result)
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/commitments", tags=["Miners"], summary="Miner model commitments",
         description="""Returns all miner HuggingFace model commitments (on-chain).

Each commitment contains:
- `model`: HuggingFace repo (e.g. `aceini/q-dist`)
- `revision`: Git commit SHA of the submitted model
- `block`: Block number when the commitment was made

**Cached for 60s.**
""")
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


@app.get("/api/scores", tags=["Miners"], summary="Current KL scores and disqualifications",
         description="""Returns the latest KL-divergence scores for all evaluated miners, plus disqualification status.

Response includes:
- `scores`: Map of UID → KL score (lower is better)
- `ema_scores`: Same as scores (backward compat)
- `disqualified`: Map of UID → disqualification reason
- `last_eval`: Details of the most recent evaluation round
- `last_eval_time`: Unix timestamp of last eval
- `tempo_seconds`: Seconds between evaluation rounds (currently 600)
""")
def get_scores(fields: str = ""):
    result = {"scores": {}, "ema_scores": {}, "disqualified": {}, "last_eval": None, "last_eval_time": None, "tempo_seconds": 600}
    scores_path = os.path.join(STATE_DIR, "scores.json")
    s = _safe_json_load(scores_path, {})
    result["scores"] = s
    result["ema_scores"] = s  # backward compat
    dq_path = os.path.join(STATE_DIR, "disqualified.json")
    result["disqualified"] = _safe_json_load(dq_path, {})
    eval_path = os.path.join(STATE_DIR, "last_eval.json")
    last_eval = _safe_json_load(eval_path)
    if last_eval is not None:
        result["last_eval"] = last_eval
        try:
            result["last_eval_time"] = os.path.getmtime(eval_path)
        except OSError:
            result["last_eval_time"] = last_eval.get("timestamp")
        result["last_eval_block"] = last_eval.get("block")
        result["last_eval_type"] = last_eval.get("type")
    # Filter fields if requested
    if fields:
        requested = set(f.strip() for f in fields.split(","))
        result = {k: v for k, v in result.items() if k in requested}
    return JSONResponse(
        content=_sanitize_floats(result),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@app.get("/api/price", tags=["Market"], summary="Token price and market data",
         description="""Returns SN97 alpha token pricing, TAO/USD rate, pool liquidity, emission, and volume.

Response includes:
- `alpha_price_tao` / `alpha_price_usd`: Current alpha token price
- `tao_usd`: TAO/USD exchange rate (via CoinGecko)
- `alpha_in_pool` / `tao_in_pool`: DEX pool liquidity
- `marketcap_tao`: Total market cap in TAO
- `emission_pct`: Current emission allocation percentage
- `price_change_1h`, `_24h`, `_7d`: Price change percentages
- `miners_tao_per_day`: Total TAO earned by miners per day

**Cached for 30s.**
""")
def get_price():
    cached = _get_cached("price", 30)
    if cached:
        return JSONResponse(content=cached, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    stale = _get_stale("price")
    if stale:
        _bg_refresh("price", _fetch_price)
        return JSONResponse(content=stale, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    try:
        result = _fetch_price()
        _set_cached("price", result)
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model-info/{model_path:path}", tags=["Miners"], summary="HuggingFace model info",
         description="""Fetches model card metadata from HuggingFace for a given repo.

**Example**: `/api/model-info/aceini/q-dist`

Response includes:
- `params_b`: Total parameters in billions
- `is_moe`: Whether the model uses Mixture of Experts
- `num_experts` / `num_active_experts`: MoE configuration
- `tags`, `license`, `pipeline_tag`: HuggingFace metadata
- `downloads`, `likes`: Popularity metrics
- `base_model`: Parent model (if distilled/fine-tuned)

**Cached for 1 hour.**
""")
def get_model_info(model_path: str):
    cache_key = f"model_info:{model_path}"
    cached = _get_cached(cache_key, 3600)
    if cached:
        return cached
    try:
        import subprocess
        script = """
import json, os, sys
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
        hidden = config.get("hidden_size", 0)
        num_experts = ne
        num_active_experts = config.get("num_experts_per_tok", config.get("num_active_experts", ne))
except Exception:
    pass

card = info.card_data
result = {
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
}
print(json.dumps(result))
"""
        env = os.environ.copy()
        env["MODEL_PATH"] = model_path
        result_proc = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        if result_proc.returncode != 0:
            raise RuntimeError(result_proc.stderr[-300:])
        result = json.loads(result_proc.stdout)
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        return {"error": str(e), "model": model_path}


@app.get("/api/leaderboard", tags=["Evaluation"], summary="Top-4 leaderboard",
         description="Returns the top-4 leaderboard — current king and contenders. Dethronement uses paired t-test (p < 0.05).")
def get_leaderboard():
    top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {}) or {}

    leaderboard = {
        "king": dict(top4.get("king") or {}) if top4.get("king") else None,
        "contenders": [dict(c) for c in (top4.get("contenders") or [])],
        "phase": top4.get("phase", "unknown"),
        "initial_eval_complete": top4.get("initial_eval_complete", False),
        "completed_at": top4.get("completed_at"),
    }

    return JSONResponse(
        content={
            "leaderboard": leaderboard,
            "phase": leaderboard["phase"],
        },
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


def _is_announcement_claimed(ann: dict) -> bool:
    """Check if an announcement has already been claimed, using the claims log.

    The claims log prevents re-posting when rsync overwrites announcement.json
    with an older copy that still has posted=False.
    """
    claims_path = os.path.join(STATE_DIR, "announcement_claims.json")
    claims = _safe_json_load(claims_path, [])
    ann_ts = ann.get("timestamp", 0)
    ann_type = ann.get("type", "")
    # Match by timestamp + type (unique per announcement)
    for claim in claims:
        if claim.get("timestamp") == ann_ts and claim.get("type") == ann_type:
            return True
    return False


def _record_announcement_claim(ann: dict):
    """Record that an announcement was claimed, in a separate file rsync won't overwrite."""
    claims_path = os.path.join(STATE_DIR, "announcement_claims.json")
    claims = _safe_json_load(claims_path, [])
    claims.append({
        "timestamp": ann.get("timestamp", 0),
        "type": ann.get("type", ""),
        "claimed_at": time.time(),
    })
    # Keep only last 50 claims
    claims = claims[-50:]
    with open(claims_path, "w") as f:
        json.dump(claims, f, indent=2)


@app.get("/api/announcement", tags=["Evaluation"], summary="Pending announcements",
         description="Returns pending announcements (e.g., new king crowned). Returns `{type: null}` if none pending.")
def get_announcement():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True) and not _is_announcement_claimed(ann):
                return ann
        except Exception:
            pass
    return {"type": None}


@app.post("/api/announcement/claim", tags=["Evaluation"], summary="Claim pending announcement",
          description="Atomically reads and marks an announcement as posted. Returns the announcement content, or `{type: null}` if none pending. "
                      "Uses a claims log to prevent re-posting after rsync overwrites.")
def claim_announcement():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True) and not _is_announcement_claimed(ann):
                # Record the claim FIRST (idempotent protection)
                _record_announcement_claim(ann)
                # Also mark posted in the file (best effort — rsync may overwrite)
                ann["posted"] = True
                with open(ann_path, "w") as f:
                    json.dump(ann, f, indent=2)
                return ann
        except Exception:
            pass
    return {"type": None}


@app.post("/api/announcement/posted", tags=["Evaluation"], summary="Mark announcement as posted",
          description="Marks the current announcement as posted. Legacy endpoint — prefer `/api/announcement/claim`.")
def mark_announcement_posted():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            _record_announcement_claim(ann)
            ann["posted"] = True
            with open(ann_path, "w") as f:
                json.dump(ann, f, indent=2)
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}
    return {"ok": True, "note": "no announcement"}


@app.get("/api/eval-progress", tags=["Evaluation"], summary="Live evaluation progress",
         description="""Shows what the validator is currently doing in real-time.

When `active: true`, the response includes:
- `phase`: Current eval phase (e.g. `teacher_generation`, `student_eval`)
- `students_total`: How many miners are being evaluated
- `completed[]`: UIDs that have finished this round
- `current`: Details on the student being evaluated right now (name, prompts done, running KL mean)
- `prompts_total`: Total prompts in this round

When `active: false`, the validator is idle between rounds.
""")
def get_eval_progress():
    progress_path = os.path.join(STATE_DIR, "eval_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": False}


@app.get("/api/h2h-latest", tags=["Evaluation"], summary="Latest head-to-head round",
         description="""Returns results from the most recent evaluation round where miners compete against the king.

Response includes:
- `block`: Block when this round was scored
- `king_uid`: Current king's UID
- `king_h2h_kl`: King's KL score in this round
- `king_global_kl`: King's smoothed global KL
- `p_value`: Paired t-test p-value for the challenger vs king comparison
- `n_prompts`: Number of prompts used
- `results[]`: Array of `{uid, model, kl, is_king, vs_king}` for each evaluated miner
- `king_changed`: Whether the king was dethroned this round (requires p < 0.05)
""")
def get_h2h_latest():
    path = os.path.join(STATE_DIR, "h2h_latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return _sanitize_floats(data)
        except Exception:
            pass
    return {"error": "No H2H data yet"}


@app.get("/api/h2h-history", tags=["Evaluation"], summary="Head-to-head round history",
         description="Returns evaluation rounds with pagination. Supports `?limit=N` (default 50, max 200) and `?page=N` (1-indexed, default 1). Returns newest rounds first when paginated.")
def get_h2h_history(limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            # Annotate exploit rounds: king_changed but new_king not in results
            for entry in data:
                if entry.get("king_changed") and entry.get("new_king_uid"):
                    result_uids = [r.get("uid") for r in entry.get("results", [])]
                    if entry["new_king_uid"] not in result_uids:
                        entry["_exploit"] = True
                        entry["_exploit_note"] = "King promoted from cached scores without evaluation this round (fixed in 579b17b)"
            total = len(data)
            # Reverse so newest first, then paginate
            data_rev = list(reversed(data))
            start = (page - 1) * limit
            end = start + limit
            page_data = data_rev[start:end]
            return JSONResponse(
                content=_sanitize_floats({"rounds": page_data, "total": total, "page": page, "limit": limit, "has_more": end < total}),
                headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
            )
        except Exception:
            pass
    return JSONResponse(
        content={"rounds": [], "total": 0, "page": 1, "limit": limit, "has_more": False},
        headers={"Cache-Control": "public, max-age=10"},
    )


@app.get("/api/king-history", tags=["Evaluation"], summary="King dethronement history",
         description="Returns the chain of king changes (dethronements). Each entry shows the block, new king, and the dethroned UID with margin of victory.")
def get_king_history():
    """Extract all king changes from h2h_history.json."""
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            history = json.load(f)
    except Exception:
        return []

    changes = []
    for entry in history:
        if not entry.get("king_changed"):
            continue
        new_king_uid = entry.get("new_king_uid") or entry.get("king_uid")
        prev_king_uid = entry.get("prev_king_uid")
        # Find king model name from results
        king_model = None
        king_kl = None
        prev_kl = None
        for r in entry.get("results", []):
            if r.get("uid") == new_king_uid:
                king_model = r.get("model")
                king_kl = r.get("kl")
            if r.get("uid") == prev_king_uid:
                prev_kl = r.get("kl")
        margin = None
        if king_kl is not None and prev_kl is not None and prev_kl > 0:
            margin = round((prev_kl - king_kl) / prev_kl, 6)
        # Detect exploit rounds: new king not in results
        result_uids = [r.get("uid") for r in entry.get("results", [])]
        is_exploit = new_king_uid not in result_uids and entry.get("king_changed")
        change = {
            "block": entry.get("block"),
            "timestamp": entry.get("timestamp"),
            "king_uid": new_king_uid,
            "king_model": king_model,
            "dethroned_uid": prev_king_uid,
            "margin": margin,
        }
        if is_exploit:
            change["_exploit"] = True
        changes.append(change)
    return changes


@app.get("/api/tmc-config", tags=["Market"], summary="TaoMarketCap SSE config",
         description="Returns SSE (Server-Sent Events) URLs for real-time price and subnet data from TaoMarketCap. Used by the dashboard for live price updates.")
def get_tmc_config():
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "netuid": NETUID,
    }




@app.get("/api/history", tags=["Evaluation"], summary="Score history over time",
         description="Returns historical KL scores for all miners over time. Supports `?limit=N` (default 50) to return only the latest N entries. Response includes `full_eval_block` if a full eval round exists.")
def get_history(limit: int = 50):
    limit = max(1, min(limit, 500))
    history_path = os.path.join(STATE_DIR, "score_history.json")
    entries = []
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                data = json.load(f)
            entries = data[-limit:] if len(data) > limit else data
        except Exception:
            pass

    # Find full_eval block from h2h_history.
    # Older state may have accidentally stored a timestamp in `block`, so when
    # that happens we normalize to the nearest score_history entry by timestamp.
    full_eval_block = None
    h2h_path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(h2h_path):
        try:
            with open(h2h_path) as f:
                h2h_data = json.load(f)
            full_eval_round = next((r for r in reversed(h2h_data) if r.get("type") == "full_eval"), None)
            if full_eval_round:
                raw_block = full_eval_round.get("block")
                full_eval_ts = full_eval_round.get("timestamp")
                if isinstance(raw_block, int) and raw_block < 100_000_000:
                    full_eval_block = raw_block
                elif full_eval_ts and entries:
                    nearest = min(entries, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
                    full_eval_block = nearest.get("block")
                elif full_eval_ts and os.path.exists(history_path):
                    with open(history_path) as f:
                        all_history = json.load(f)
                    if all_history:
                        nearest = min(all_history, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
                        full_eval_block = nearest.get("block")
        except Exception:
            pass

    return JSONResponse(
        content={"entries": entries, "full_eval_block": full_eval_block},
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )


@app.get("/api/pod-logs", tags=["Debugging"], summary="Pod eval logs",
         description="Access pod eval logs. Use `?list_files=true` to list available logs, or `?file=<name>&lines=N&offset=N` to read.")
def get_pod_logs(list_files: bool = False, file: str = None, lines: int = 200, offset: int = 0):
    logs_dir = os.path.join(STATE_DIR, "pod_logs")
    if not os.path.exists(logs_dir):
        return {"files": [], "error": "No logs directory"}
    if list_files:
        files = sorted([f for f in os.listdir(logs_dir) if f.endswith(".log")], reverse=True)
        return {"files": files, "count": len(files)}
    if not file:
        return {"error": "Specify ?file=<name> or ?list_files=true"}
    # Sanitize filename
    safe_name = os.path.basename(file)
    path = os.path.join(logs_dir, safe_name)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    try:
        with open(path) as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if offset > 0:
            selected = all_lines[offset:offset + lines]
        else:
            selected = all_lines[-lines:] if lines < total else all_lines
        return {"file": safe_name, "lines": [l.rstrip() for l in selected], "count": len(selected), "total": total, "offset": offset}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/eval-data", tags=["Evaluation"], summary="Eval data (prompts + completions)",
         description="Returns eval round data. Use `?list=true` for available files, or `?file=<name>` for a specific round.")
def get_eval_data(list: bool = False, file: str = None):
    data_dir = os.path.join(STATE_DIR, "eval_data")
    latest = os.path.join(STATE_DIR, "eval_data_latest.json")
    if list:
        if not os.path.exists(data_dir):
            return {"files": []}
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")], reverse=True)
        return {"files": files, "count": len(files)}
    if file:
        safe_name = os.path.basename(file)
        path = os.path.join(data_dir, safe_name)
        if not os.path.exists(path):
            return JSONResponse(content={"error": "File not found"}, status_code=404)
    else:
        path = latest
    if not os.path.exists(path):
        return JSONResponse(content={"error": "No eval data available"}, status_code=404)
    try:
        with open(path) as f:
            return JSONResponse(content=json.load(f), headers={"Cache-Control": "public, max-age=60"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/benchmarks", tags=["Evaluation"], summary="Benchmark results for king models",
         description="Returns benchmark scores for evaluated king models. Scores are from lm-eval-harness full eval sets.")
def get_benchmarks():
    benchmarks_dir = os.path.join(STATE_DIR, "benchmarks")
    if not os.path.exists(benchmarks_dir):
        return JSONResponse(content={"models": [], "baseline": None}, headers={"Cache-Control": "public, max-age=60"})
    models = []
    baseline = None
    for fname in sorted(os.listdir(benchmarks_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(benchmarks_dir, fname)) as f:
                data = json.load(f)
            if data.get("is_baseline"):
                baseline = data
            else:
                models.append(data)
        except Exception:
            pass
    return JSONResponse(
        content=_sanitize_floats({"models": models, "baseline": baseline}),
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )


@app.get("/api/health", tags=["Overview"], summary="Service health and quick status",
         description="""One-stop health check that returns the current state of the validator and subnet.

Response includes:
- `status`: `ok` if the API is running
- `king_uid` / `king_kl`: Current king and their KL score (lower = better)
- `n_scored` / `n_disqualified`: Number of active vs disqualified miners
- `last_eval_block` / `last_eval_age_min`: When the last eval happened
- `eval_active`: Whether an evaluation round is in progress right now
- `eval_progress`: Detailed progress if eval is active (phase, students done, current KL, etc.)

This is the best endpoint to start with — gives you a quick overview of the entire subnet state.
""")
def health():
    import time as _time
    last_eval_block = None
    last_eval_age_min = None
    eval_active = False
    king_uid = None
    king_kl = None
    n_scored = 0
    n_dq = 0
    eval_students_done = 0
    eval_students_total = 0
    try:
        h2h = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
        last_eval_block = h2h.get("block")
        ts = h2h.get("timestamp")
        if ts:
            last_eval_age_min = round((_time.time() - ts) / 60, 1)
        king_uid = h2h.get("king_uid")
        # Get king KL from scores
        scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
        n_scored = len(scores)
        if king_uid and str(king_uid) in scores:
            king_kl = scores[str(king_uid)]
        dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
        n_dq = len(dq)
        prog = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
        eval_active = prog.get("active", False)
        if eval_active:
            eval_students_done = len(prog.get("completed", []))
            eval_students_total = prog.get("students_total", 0)
    except Exception:
        pass
    return {
        "status": "ok",
        "netuid": NETUID,
        "dethrone_method": "paired_t_test",
        "king_uid": king_uid,
        "king_kl": round(king_kl, 6) if king_kl else None,
        "n_scored": n_scored,
        "n_disqualified": n_dq,
        "last_eval_block": last_eval_block,
        "last_eval_age_min": last_eval_age_min,
        "eval_active": eval_active,
        "eval_progress": {
            "phase": prog.get("phase"),
            "students_total": prog.get("students_total"),
            "students_done": len(prog.get("completed", [])),
            "prompts_total": prog.get("prompts_total"),
            "current_student": prog.get("current", {}).get("student_name") if isinstance(prog.get("current"), dict) else None,
            "current_prompt": prog.get("current", {}).get("prompts_done") if isinstance(prog.get("current"), dict) else None,
            "current_kl": prog.get("current", {}).get("kl_running_mean") if isinstance(prog.get("current"), dict) else None,
            "current_best": prog.get("current", {}).get("best_kl_so_far") if isinstance(prog.get("current"), dict) else None,
            "teacher_prompts_done": prog.get("teacher_prompts_done"),
        } if eval_active else None,
    }


import re as _re
_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*m')
_SECRET_RE = _re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}|ssh-(?:rsa|ed25519|dss|ecdsa)\s+[A-Za-z0-9+/=]{20,}|AAAA[A-Za-z0-9+/=]{50,}')
_SENSITIVE_KW = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret", "PRIVATE KEY", "ssh-rsa", "ssh-ed25519", "credentials")
_INTERNAL_PATHS = ("/root/", "/home/pod/", "/home/openclaw/")
_ALLOWED_PREFIXES = ("[GPU]", "[eval]", "[VALIDATOR]", "[pod_eval]", "[vLLM]", "[PHASE]", "[Cache]", "#")


def _sanitize_log_line(line: str) -> str | None:
    """Sanitize a single log line. Returns None if the line should be dropped."""
    cleaned = _ANSI_RE.sub('', line).strip()
    if not cleaned:
        return None
    if any(kw in cleaned for kw in _SENSITIVE_KW):
        return None
    if any(p in cleaned for p in _INTERNAL_PATHS):
        return None
    cleaned = _SECRET_RE.sub('[REDACTED]', cleaned)
    return cleaned


@app.get("/api/validator-logs", tags=["Evaluation"], summary="Real-time validator activity logs",
         description="""Returns structured validator log entries showing round progress, precheck results, eval outcomes, and errors.

Query parameters:
- `limit`: Number of entries to return (default 50, max 200)

Each entry contains:
- `ts`: Unix timestamp
- `level`: Log level (info, warn, error)
- `msg`: Human-readable log message

Entries are returned chronologically (oldest first).
""")
def get_validator_logs(limit: int = 50):
    limit = max(1, min(limit, 200))
    log_path = os.path.join(STATE_DIR, "validator_log.json")
    entries = _safe_json_load(log_path, [])
    if not isinstance(entries, list):
        entries = []
    # Most recent first, limited
    entries = entries[-limit:]
    return JSONResponse(
        content={"entries": entries, "count": len(entries)},
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=10"},
    )


@app.get("/api/gpu-logs", tags=["Evaluation"], summary="Recent GPU evaluation logs",
         description="""Returns sanitized recent logs from the GPU evaluation pod and validator process.

Query parameters:
- `lines`: Number of log lines to return (default 50, max 200)

Logs are sanitized — API keys, internal paths, and sensitive data are stripped. Lines are prefixed with source tags like `[GPU]`, `[eval]`, `[VALIDATOR]`.
""")
def gpu_logs(lines: int = 50):
    import subprocess
    max_lines = min(lines, 200)
    log_lines = []

    # Source 1: Live GPU eval output from pod (streamed by poll thread, pre-sanitized)
    gpu_log_path = os.path.join(STATE_DIR, "gpu_eval.log")
    if os.path.exists(gpu_log_path):
        try:
            with open(gpu_log_path) as f:
                pod_lines = f.read().strip().split('\n')
            for line in pod_lines:
                cleaned = _sanitize_log_line(line)
                if cleaned:
                    prefixed = cleaned if any(cleaned.startswith(p) for p in _ALLOWED_PREFIXES) else f"[GPU] {cleaned}"
                    log_lines.append(prefixed)
        except Exception:
            pass

    # Source 2: Validator PM2 logs (orchestrator events)
    try:
        result = subprocess.run(
            ["pm2", "logs", "distill-validator", "--lines", str(max_lines), "--nostream"],
            capture_output=True, text=True, timeout=5
        )
        raw = result.stdout + result.stderr
        for line in raw.split('\n'):
            cleaned = _ANSI_RE.sub('', line)
            if '|' in cleaned:
                cleaned = cleaned.split('|', 1)[-1].strip()
            if not cleaned:
                continue
            # Only allow lines with known prefixes
            if not any(cleaned.startswith(p) for p in _ALLOWED_PREFIXES):
                continue
            sanitized = _sanitize_log_line(cleaned)
            if sanitized:
                log_lines.append(sanitized)
    except Exception:
        pass

    return {
        "lines": log_lines[-max_lines:],
        "count": len(log_lines),
    }


# ── Miner lookup endpoints ────────────────────────────────────────────────────


@app.get("/api/eval-status", tags=["Evaluation"], summary="Eval status for all miners",
         description="""Returns why each miner is or isn't being evaluated.
Statuses: king, queued, tested, stale, untested, disqualified.""")
def get_eval_status():
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
    h2h_tracker = _safe_json_load(os.path.join(STATE_DIR, "h2h_tested_against_king.json"), {})
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    current_king_uid = h2h_latest.get("king_uid")
    current_block = h2h_latest.get("block", 0)
    stale_threshold = 50

    result = {}
    for uid_str in scores:
        if uid_str in dq:
            result[uid_str] = {"status": "disqualified"}
            continue
        if current_king_uid is not None and int(uid_str) == current_king_uid:
            result[uid_str] = {"status": "king"}
            continue
        tracker_entry = h2h_tracker.get(uid_str, {})
        if tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
            last_block = tracker_entry["block"]
            epochs_since = (current_block - last_block) // 360 if current_block > last_block else 0
            if epochs_since < stale_threshold:
                result[uid_str] = {"status": "tested", "epochs_ago": epochs_since}
            else:
                result[uid_str] = {"status": "stale", "epochs_ago": epochs_since}
        else:
            result[uid_str] = {"status": "untested"}
    return JSONResponse(
        content={"king_uid": current_king_uid, "block": current_block, "statuses": result},
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@app.get("/api/miner/{uid}", tags=["Miners"], summary="Full miner details by UID",
         description="""Returns everything known about a specific miner UID.

Response includes:
- `hotkey` / `coldkey`: On-chain keys
- `commitment`: Model repo, revision, and commitment block
- `kl_score`: Current KL-divergence score (lower = better)
- `disqualified`: Disqualification status and reason (if any)
- `h2h_history`: Last 10 head-to-head rounds involving this UID
- `in_top5`: Whether this UID is in the top 5 (king or contender)
- `is_king`: Whether this UID is the current king
- `registered`: Whether this UID is registered in the metagraph
""")
def get_miner(uid: int):
    result = {"uid": uid, "registered": False}

    # Metagraph data
    metagraph = _get_stale("metagraph") or {}
    neurons = metagraph.get("neurons", [])
    neuron = None
    for n in neurons:
        if n.get("uid") == uid:
            neuron = n
            break
    if neuron:
        result["registered"] = True
        result["hotkey"] = neuron.get("hotkey")
        result["coldkey"] = neuron.get("coldkey")
        result["stake"] = neuron.get("stake")
        result["incentive"] = neuron.get("incentive")
        result["emission"] = neuron.get("emission")
        result["is_validator"] = neuron.get("is_validator", False)
    else:
        result["hotkey"] = None
        result["coldkey"] = None

    # Commitment
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    hotkey = result.get("hotkey")
    if hotkey and hotkey in commitments:
        result["commitment"] = commitments[hotkey]
    else:
        result["commitment"] = None

    # KL score
    scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {})
    uid_str = str(uid)
    result["kl_score"] = scores.get(uid_str)

    # Disqualification — check per-commit key first, fall back to legacy keys
    # only if no commit_block is known (same logic as eval/scoring.py is_disqualified)
    dq = _safe_json_load(os.path.join(STATE_DIR, "disqualified.json"), {})
    commit_block = result.get("commitment", {}).get("block") if result.get("commitment") else None
    dq_reason = None
    if commit_block is not None and hotkey:
        dq_reason = dq.get(f"{hotkey}:{commit_block}")
    if dq_reason is None and commit_block is None:
        # Only use legacy bare keys when we don't know the commit block
        dq_reason = dq.get(uid_str) or dq.get(hotkey) if hotkey else dq.get(uid_str)
    result["disqualified"] = dq_reason

    # Top 5 / king status
    top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {})
    king = top4.get("king") or {}
    contenders = top4.get("contenders") or []
    result["is_king"] = king.get("uid") == uid
    top5_uids = set()
    if king.get("uid") is not None:
        top5_uids.add(king["uid"])
    for c in contenders:
        if c.get("uid") is not None:
            top5_uids.add(c["uid"])
    result["in_top5"] = uid in top5_uids

    # Eval status: why (not) evaluated
    h2h_tracker = _safe_json_load(os.path.join(STATE_DIR, "h2h_tested_against_king.json"), {})
    h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    current_king_uid = h2h_latest.get("king_uid")
    current_block = h2h_latest.get("block", 0)
    tracker_entry = h2h_tracker.get(uid_str, {})
    eval_status = {}
    if result.get("disqualified"):
        eval_status["status"] = "disqualified"
        eval_status["reason"] = "Model is disqualified and won't be evaluated"
    elif result.get("is_king"):
        eval_status["status"] = "king"
        eval_status["reason"] = "Evaluated every round as the defending king"
    elif not result.get("kl_score"):
        eval_status["status"] = "queued"
        eval_status["reason"] = "Waiting for first evaluation — new submissions get priority"
    elif tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
        last_block = tracker_entry["block"]
        epochs_since = (current_block - last_block) // 360 if current_block > last_block else 0
        stale_threshold = 50
        if epochs_since < stale_threshold:
            eval_status["status"] = "tested"
            eval_status["reason"] = f"Already tested against current king ({epochs_since} epochs ago, re-test after {stale_threshold})"
            eval_status["last_test_block"] = last_block
            eval_status["epochs_since"] = epochs_since
            eval_status["stale_after"] = stale_threshold
        else:
            eval_status["status"] = "stale"
            eval_status["reason"] = f"Due for re-test ({epochs_since} epochs since last H2H, threshold is {stale_threshold})"
            eval_status["last_test_block"] = last_block
            eval_status["epochs_since"] = epochs_since
    else:
        eval_status["status"] = "untested"
        eval_status["reason"] = "Not yet tested against the current king — will be scheduled"
    result["eval_status"] = eval_status

    # H2H history (last 10 rounds involving this UID)
    h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    relevant = []
    for rnd in reversed(h2h_history):
        for r in rnd.get("results", []):
            if r.get("uid") == uid:
                relevant.append({
                    "block": rnd.get("block"),
                    "timestamp": rnd.get("timestamp"),
                    "kl": r.get("kl"),
                    "is_king": r.get("is_king", False),
                    "king_changed": rnd.get("king_changed", False),
                    "type": rnd.get("type"),
                })
                break
        if len(relevant) >= 10:
            break
    result["h2h_history"] = relevant

    return JSONResponse(
        content=result,
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@app.get("/api/miner/{uid}/rounds", tags=["Miners"], summary="H2H rounds for a specific miner",
         description="""Returns all head-to-head rounds where a specific UID participated.

Supports `?limit=N` (default 50, max 200) and `?page=N` (1-indexed). Newest rounds first.

Each round entry includes:
- `block`: Block number of the round
- `timestamp`: Unix timestamp
- `kl`: This miner's KL score in that round
- `is_king`: Whether the miner was king during this round
- `king_changed`: Whether the king was dethroned
- `type`: Round type (h2h or full_eval)
- `king_uid`: Who was king that round
- `n_prompts`: Number of prompts used
""")
def get_miner_rounds(uid: int, limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    try:
        h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
        if not isinstance(h2h_history, list):
            h2h_history = []

        # Filter rounds where this UID participated
        relevant = []
        for rnd in reversed(h2h_history):
            for r in rnd.get("results", []):
                if r.get("uid") == uid:
                    relevant.append({
                        "block": rnd.get("block"),
                        "timestamp": rnd.get("timestamp"),
                        "kl": r.get("kl"),
                        "model": r.get("model"),
                        "is_king": r.get("is_king", False),
                        "vs_king": r.get("vs_king"),
                        "king_changed": rnd.get("king_changed", False),
                        "king_uid": rnd.get("king_uid"),
                        "new_king_uid": rnd.get("new_king_uid"),
                        "type": rnd.get("type"),
                        "n_prompts": rnd.get("n_prompts"),
                        "p_value": rnd.get("p_value"),
                    })
                    break

        total = len(relevant)
        start = (page - 1) * limit
        end = start + limit
        page_data = relevant[start:end]

        return JSONResponse(
            content=_sanitize_floats({
                "uid": uid,
                "rounds": page_data,
                "total": total,
                "page": page,
                "limit": limit,
                "has_more": end < total,
            }),
            headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch miner rounds: {str(e)}"},
        )


@app.get("/api/commitment/{hotkey}", tags=["Miners"], summary="Lookup commitment by hotkey",
         description="""Lookup a miner's on-chain model commitment by their hotkey (ss58 address).

Useful for miners to verify the validator sees their commitment after submitting.

Response includes:
- `commitment`: Model repo, revision, and commitment block (if found)
- `uid`: Registered UID (if registered in metagraph)
- `registered`: Whether this hotkey is registered
""")
def get_commitment_by_hotkey(hotkey: str):
    result = {"hotkey": hotkey, "registered": False, "uid": None, "commitment": None}

    # Find UID from metagraph
    metagraph = _get_stale("metagraph") or {}
    for n in metagraph.get("neurons", []):
        if n.get("hotkey") == hotkey:
            result["registered"] = True
            result["uid"] = n.get("uid")
            result["coldkey"] = n.get("coldkey")
            result["stake"] = n.get("stake")
            result["incentive"] = n.get("incentive")
            break

    # Commitment data
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    if hotkey in commitments:
        result["commitment"] = commitments[hotkey]

    return JSONResponse(
        content=result,
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


# ── Chat with king model ──────────────────────────────────────────────────────


# Chat server runs on the GPU pod at port 8100 (scripts/chat_server.py).
# We proxy requests via direct SSH + curl (no lium dependency).
CHAT_POD_PORT = 8100
CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST", "91.224.44.81")
CHAT_POD_SSH_PORT = int(os.environ.get("CHAT_POD_SSH_PORT", "20300"))
CHAT_POD_SSH_KEY = os.environ.get("CHAT_POD_SSH_KEY", "/home/distil/.ssh/id_ed25519")


def _get_king_info():
    """Get king UID and model name."""
    h2h = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    king_uid = h2h.get("king_uid")
    if king_uid is None:
        return None, None

    # Try h2h results first (has model directly)
    for r in h2h.get("results", []):
        if r.get("is_king") or r.get("uid") == king_uid:
            return king_uid, r.get("model")

    # Fallback: commitments cache (hotkey-keyed, need metagraph for UID→hotkey)
    metagraph = _safe_json_load(os.path.join(DISK_CACHE_DIR, "metagraph.json"), {})
    commitments_data = _safe_json_load(os.path.join(DISK_CACHE_DIR, "commitments.json"), {})
    commitments = commitments_data.get("commitments", commitments_data) if isinstance(commitments_data, dict) else {}

    # Build UID→hotkey from metagraph
    uids = metagraph.get("uids", [])
    hotkeys = metagraph.get("hotkeys", [])
    king_hotkey = None
    for i, uid in enumerate(uids):
        if uid == king_uid and i < len(hotkeys):
            king_hotkey = hotkeys[i]
            break

    if king_hotkey and king_hotkey in commitments:
        info = commitments[king_hotkey]
        return king_uid, info.get("model") if isinstance(info, dict) else info

    return king_uid, None


def _ssh_exec(cmd: str, timeout: int = 30) -> str:
    """Execute command on chat pod via SSH. Returns stdout."""
    import subprocess
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
        f"root@{CHAT_POD_HOST}", cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout



@app.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod. Supports streaming via stream=true."""
    # Rate limit: 10 req/min per IP for chat
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 8192)
    stream = body.get("stream", False)

    if not messages:
        return {"error": "messages required"}

    # Input validation
    if not isinstance(messages, list) or len(messages) > 50:
        return JSONResponse(status_code=400, content={"error": "messages must be an array with at most 50 entries"})
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and len(content) > 10000:
            return JSONResponse(status_code=400, content={"error": "message content too long (max 10000 chars)"})
    if not isinstance(max_tokens, (int, float)) or max_tokens < 1:
        max_tokens = 8192
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    try:
        pod_payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            return _stream_chat(pod_payload, king_uid, king_model)
        else:
            return _sync_chat(pod_payload, king_uid, king_model)

    except Exception as e:
        # Sanitize error — don't leak SSH commands/paths to frontend
        err = str(e)
        if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
            return {"error": "chat server connection failed — try again in a moment"}
        return {"error": f"chat error: {err[:200]}"}


def _sync_chat(payload, king_uid, king_model):
    """Non-streaming chat proxy via SSH + curl."""
    import base64
    payload["stream"] = False
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    cmd = f"echo '{payload_b64}' | base64 -d | curl -s -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
    stdout = _ssh_exec(cmd, timeout=60)

    try:
        data = json.loads(stdout)
        if "choices" in data:
            resp = {
                "response": data["choices"][0]["message"]["content"],
                "model": king_model,
                "king_uid": king_uid,
            }
            if "thinking" in data:
                resp["thinking"] = data["thinking"]
            if "usage" in data:
                resp["usage"] = data["usage"]
            return resp
        return {"error": "unexpected response from chat server"}
    except json.JSONDecodeError:
        return {"error": "chat server not responding — may be starting up"}


def _stream_chat(payload, king_uid, king_model):
    """Streaming chat proxy via SSE. SSH + curl -N pipes pod SSE → client."""
    import base64, subprocess
    payload["stream"] = True
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    cmd = f"echo '{payload_b64}' | base64 -d | curl -sN -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"

    def generate():
        try:
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
                f"root@{CHAT_POD_HOST}", cmd,
            ]
            proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(raw)
                        parsed["king_uid"] = king_uid
                        parsed["king_model"] = king_model
                        yield f"data: {json.dumps(parsed)}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {raw}\n\n"
            proc.wait(timeout=5)
        except Exception as e:
            err = str(e)
            if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
                err = "chat server connection failed"
            yield f"data: {json.dumps({'error': err[:200]})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_chat_restart_lock = threading.Lock()
_last_chat_restart = 0.0


def _ensure_chat_server(king_model=None):
    """Auto-start chat server if not running or running wrong model. Rate-limited to once per 2 min."""
    global _last_chat_restart
    with _chat_restart_lock:
        if time.time() - _last_chat_restart < 120:
            return  # Already tried recently
        _last_chat_restart = time.time()

    model_name = king_model or "unknown"
    try:
        stdout = _ssh_exec(f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models || echo not_running")
        if "not_running" in stdout:
            print(f"[chat] Auto-starting chat server for {model_name}", flush=True)
            _ssh_exec(f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_PORT} > /root/chat.log 2>&1 &", timeout=10)
        elif model_name != "unknown" and model_name not in stdout:
            # Server running but wrong model — restart
            print(f"[chat] Chat server running wrong model, restarting for {model_name}", flush=True)
            _ssh_exec("pkill -f 'vllm.entrypoints.openai.api_server|chat_server.py' || true", timeout=10)
            import time as _t; _t.sleep(2)
            _ssh_exec(f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_PORT} > /root/chat.log 2>&1 &", timeout=10)
    except Exception as e:
        print(f"[chat] Auto-restart failed: {e}", flush=True)


@app.get("/api/chat/status")
def chat_status():
    """Check if the king chat server is available. Auto-starts if down."""
    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    # Try health check on pod
    server_ok = False
    try:
        stdout = _ssh_exec(f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models")
        if stdout and (king_model is None or king_model in stdout):
            server_ok = True
        elif not eval_active:
            # Server not responding and no eval in progress — auto-restart
            _ensure_chat_server(king_model)
    except Exception:
        pass

    return {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "note": "King model is loaded on GPU and ready for chat." if server_ok else "Chat server is starting or unavailable.",
    }


# ── OpenAI-compatible endpoints (for Open WebUI etc.) ─────────────────────────


@app.get("/v1/models")
def openai_models():
    """OpenAI-compatible models list. Returns the current king model."""
    king_uid, king_model = _get_king_info()
    model_id = king_model or "distil-king"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
        }],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint. Proxies to the king model.
    Used by Open WebUI and other OpenAI-compatible clients."""
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}})

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    # Default to non-thinking chat output for Open WebUI / normal chat UX.
    # Callers can still override this explicitly.
    body.setdefault("chat_template_kwargs", {})
    body["chat_template_kwargs"].setdefault("enable_thinking", False)

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return JSONResponse(status_code=503, content={"error": {"message": "no king model available"}})

    # Forward the request as-is to the chat server (it already speaks OpenAI format)
    import base64
    stream = body.get("stream", False)
    payload_b64 = base64.b64encode(json.dumps(body).encode()).decode()

    try:
        if stream:
            # Stream SSE directly from chat server
            import subprocess
            cmd = f"echo '{payload_b64}' | base64 -d | curl -sN -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
                f"root@{CHAT_POD_HOST}", cmd,
            ]

            def generate():
                try:
                    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for line in proc.stdout:
                        line = line.strip()
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                break
                    proc.wait(timeout=5)
                except Exception:
                    yield 'data: {"error": "stream interrupted"}\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        else:
            # Non-streaming: return raw OpenAI response
            cmd = f"echo '{payload_b64}' | base64 -d | curl -s -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
            stdout = _ssh_exec(cmd, timeout=120)
            try:
                data = json.loads(stdout)
                return JSONResponse(content=data)
            except json.JSONDecodeError:
                return JSONResponse(status_code=502, content={"error": {"message": "chat server not responding"}})
    except Exception:
        return JSONResponse(status_code=502, content={"error": {"message": "chat server connection failed"}})


# ── Startup: prime caches ────────────────────────────────────────────────────

# ── Rate limiting middleware for all endpoints ────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip rate limiting for docs
        if request.url.path in ("/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        # Chat/OpenAI endpoints have their own stricter limiter applied in the handler
        if request.url.path in ("/api/chat", "/v1/chat/completions", "/v1/models"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        # Exempt localhost — dashboard SSR makes many internal requests
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)
        if not _rate_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)


@app.on_event("startup")
def prime_caches():
    """On startup, kick off background refreshes so first request is fast."""
    _bg_refresh("metagraph", _fetch_metagraph)
    _bg_refresh("commitments", _fetch_commitments)
    _bg_refresh("price", _fetch_price)
