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

Miners submit distilled models and a validator scores each commitment on a **multi-axis composite** (`scripts/validator/composite.py`) covering:

- **Teacher-similarity axes**: KL, on-policy RKL, top-K overlap, IS-KL (unbiased), forking-RKL, teacher-trace plausibility, entropy-aware KL, tail-decoupled KL
- **Skill-group axes** (v30.2 — collapse without losing depth):
  - `code_skill_group` = mean of {{code_bench, mbpp_bench, debug_bench, correction_bench, refactor_bench}}
  - `math_skill_group` = mean of {{math_bench, aime_bench, robustness_bench}}
  - `reasoning_skill_group` = mean of {{reasoning_bench, multi_doc_synthesis_bench, long_context_bench}}
  - `knowledge_skill_group` = mean of {{knowledge_bench, pragmatic_bench}}
- **Stand-alone capability axes**: tool_use_bench, ifeval_bench, calibration_bench
- **Quality axes**: judge_probe (short answers), long_form_judge (300-500 word essays), chat_turns_probe (3-turn coherence)
- **Discipline axes**: length, degeneracy, capability, reasoning_density
- **Super-teacher axis** (v30.2): rewards exceeding the teacher on verifiable benches (incentivises GRPO + post-distillation SFT)

**Ranking key** (v30.2+): `composite.final = 0.7 × worst_3_mean + 0.3 × weighted` where:
- `worst_3_mean` = equal-weighted mean of the 3 lowest non-broken axes (smooths single-axis noise while preserving anti-Goodhart pressure)
- `weighted` = standard weighted convex combination of all axes

The king is whoever has the highest `composite.final`. A challenger dethrones only when its final beats the incumbent's by `SINGLE_EVAL_DETHRONE_MARGIN` (default 3%). The legacy `composite.worst` (single-axis min) is retained as telemetry but is no longer the dethrone gate.

## Quick Start

```bash
curl https://api.arbos.life/api/health
curl https://api.arbos.life/api/scores
curl https://api.arbos.life/api/price
```
"""
