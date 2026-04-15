"""Shared configuration constants for the Distil API."""

import os

# Load .env from repo root
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

NETUID = 97
CACHE_TTL = 60
TMC_KEY = os.environ.get("TMC_API_KEY", "")
TMC_BASE = "https://api.taomarketcap.com"
TMC_HEADERS = {"Authorization": TMC_KEY}

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
DISK_CACHE_DIR = os.path.join(STATE_DIR, "api_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

# Chat pod SSH connection
CHAT_POD_PORT = 8100
CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST", "91.224.44.207")
CHAT_POD_SSH_PORT = int(os.environ.get("CHAT_POD_SSH_PORT", "40070"))
CHAT_POD_SSH_KEY = os.environ.get("CHAT_POD_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

API_DESCRIPTION = """
# Distil - Subnet 97 API

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
