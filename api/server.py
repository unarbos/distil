from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import json
import traceback
import os
import threading
import requests as req

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

from fastapi.responses import RedirectResponse

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
def get_scores():
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
        result["last_eval_time"] = os.path.getmtime(eval_path)
    return result


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


@app.get("/api/announcement", tags=["Evaluation"], summary="Pending announcements",
         description="Returns pending announcements (e.g., new king crowned). Returns `{type: null}` if none pending.")
def get_announcement():
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


@app.post("/api/announcement/claim", tags=["Evaluation"], summary="Claim pending announcement",
          description="Atomically reads and marks an announcement as posted. Returns the announcement content, or `{type: null}` if none pending.")
def claim_announcement():
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


@app.post("/api/announcement/posted", tags=["Evaluation"], summary="Mark announcement as posted",
          description="Marks the current announcement as posted. Legacy endpoint — prefer `/api/announcement/claim`.")
def mark_announcement_posted():
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
- `epsilon` / `epsilon_threshold`: How close a challenger must get to dethrone the king
- `n_prompts`: Number of prompts used
- `results[]`: Array of `{uid, model, kl, is_king, vs_king}` for each evaluated miner
- `king_changed`: Whether the king was dethroned this round
""")
def get_h2h_latest():
    path = os.path.join(STATE_DIR, "h2h_latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"error": "No H2H data yet"}


@app.get("/api/h2h-history", tags=["Evaluation"], summary="Head-to-head round history",
         description="Returns the last 50 evaluation rounds. Each entry has the same structure as `/api/h2h-latest`.")
def get_h2h_history():
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


@app.get("/api/tmc-config", tags=["Market"], summary="TaoMarketCap SSE config",
         description="Returns SSE (Server-Sent Events) URLs for real-time price and subnet data from TaoMarketCap. Used by the dashboard for live price updates.")
def get_tmc_config():
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "netuid": NETUID,
    }


@app.get("/api/history", tags=["Evaluation"], summary="Score history over time",
         description="Returns historical KL scores for all miners over time. Used by the dashboard trendline chart. Each entry contains a timestamp and UID → score mapping.")
def get_history():
    history_path = os.path.join(STATE_DIR, "score_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                return json.load(f)
        except Exception:
            return []
    return []


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
_SECRET_RE = _re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}')
_SENSITIVE_KW = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret")
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


# ── Chat with king model ──────────────────────────────────────────────────────

from fastapi import Request

# Chat server runs on the GPU pod at port 8100 (scripts/chat_server.py).
# We proxy requests via Lium SSH exec + curl.
CHAT_POD_PORT = 8100


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


def _lium_pod():
    """Get Lium client and pod."""
    from lium import Lium, Config
    from pathlib import Path
    lium_key = os.environ.get("LIUM_API_KEY")
    if not lium_key:
        return None, None
    lium = Lium(config=Config(api_key=lium_key, ssh_key_path=str(Path.home() / ".ssh" / "id_ed25519")))
    for p in lium.ps():
        if "distil" in str(getattr(p, "name", "")).lower():
            return lium, p
    return None, None


@app.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod."""
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = min(body.get("max_tokens", 512), 1024)

    if not messages:
        return {"error": "messages required"}

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    try:
        lium, pod = _lium_pod()
        if not pod:
            return {"error": "GPU pod not found"}

        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
        }).replace("'", "'\\''")

        cmd = f"curl -s -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{payload}'"
        result = lium.exec(pod, command=cmd)
        stdout = result.get("stdout", "") if isinstance(result, dict) else str(result)

        try:
            data = json.loads(stdout)
            if "choices" in data:
                return {
                    "response": data["choices"][0]["message"]["content"],
                    "model": king_model,
                    "king_uid": king_uid,
                }
            return {"error": "unexpected response", "details": stdout[:300]}
        except json.JSONDecodeError:
            return {"error": "chat server not responding — may be starting up", "details": stdout[:300]}

    except Exception as e:
        return {"error": f"chat error: {str(e)[:200]}"}


@app.get("/api/chat/status")
def chat_status():
    """Check if the king chat server is available."""
    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    # Try health check on pod
    server_ok = False
    try:
        lium, pod = _lium_pod()
        if pod:
            result = lium.exec(pod, command=f"curl -s http://localhost:{CHAT_POD_PORT}/health")
            stdout = result.get("stdout", "") if isinstance(result, dict) else ""
            if '"status": "ok"' in stdout or '"status":"ok"' in stdout:
                server_ok = True
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


# ── Startup: prime caches ────────────────────────────────────────────────────

@app.on_event("startup")
def prime_caches():
    """On startup, kick off background refreshes so first request is fast."""
    _bg_refresh("metagraph", _fetch_metagraph)
    _bg_refresh("commitments", _fetch_commitments)
    _bg_refresh("price", _fetch_price)
