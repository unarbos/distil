import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.runtime import (
    ALLOWED_ORIGINS,
    CACHE_TTL,
    CHAT_POD_APP_PORT,
    CHAT_POD_HOST,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    DASHBOARD_URL,
    DISK_CACHE_DIR,
    NETUID,
    STATE_DIR,
    TMC_BASE,
    TMC_HEADERS,
)

CHAT_POD_PORT = CHAT_POD_APP_PORT

# ── Tunables (single source of truth, duplicated in several places before) ──
# 2026-04-28: STALE_EVAL_BLOCKS retired (was a misleading time-based cooldown
# in the dashboard ``eval_status`` text). Single-eval mode re-evaluates a UID
# only when its on-chain commitment changes; there is no time-based re-test.
# Kept here only so any external consumer that imports the constant still
# resolves (returns the legacy default), but the API routes no longer use it.
STALE_EVAL_BLOCKS = 50  # deprecated, retained for backward-compat import only
EPOCH_BLOCKS = 360
CHAT_RESTART_COOLDOWN = 120
CHAT_SERVER_SCRIPT = "/root/chat_server.py"
MAX_COMPARE_UIDS = 10
MAX_BATCH_UIDS = 64
ANNOUNCEMENT_CLAIMS_KEEP = 50

API_DESCRIPTION = f"""
# Distil - Subnet {NETUID} API

Public API for [Distil]({DASHBOARD_URL}), a Bittensor subnet where miners compete to produce the best knowledge-distilled small language models.

## How It Works

Miners submit distilled models and a validator scores each commitment on a **17-axis composite** (`scripts/validator/composite.py`) covering distribution match (KL, on-policy RKL, capability, length, degeneracy), absolute capability vs ground truth (math, code, reasoning, IFEval, AIME, MBPP, tool-use, long-context, robustness), conversational quality (judge-probe, chat-turns), and generation discipline (reasoning-density). The king is whoever has the highest `composite.worst` — the lowest score across the 17 axes. KL is one of the 17 axes, not the gate.

## Quick Start

```bash
curl https://api.arbos.life/api/health
curl https://api.arbos.life/api/scores
curl https://api.arbos.life/api/price
```
"""
