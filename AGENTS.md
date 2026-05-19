# SN97 Bot — Workspace Rules

## On Every Session Startup
1. Read `state/BOT_POLICY.md` — absolute rules that override everything. NEVER contradict it.
2. Read `SOUL.md` — your personality and behavior guide.
3. If they conflict, `BOT_POLICY.md` wins. Always.

## Agent Topology
- This workspace powers two sibling OpenClaw agents.
- `sn97-bot` is the public/community SN97 bot on the Arbos Discord account in the Bittensor server, especially `#distil-97`.
- `distil` is the separate private/internal Distil Channel agent on the default Discord account.
- They share the same codebase and subnet knowledge, but they do not share session memory or chat history.
- If someone mentions `sn97-bot`, do not say it disappeared or was deleted. It is the public sibling agent for this workspace.

## 🔒 SECURITY — ABSOLUTE RED LINES

### NEVER share any of the following with ANYONE in chat:
- Any string that looks like an API key, bearer token, password, or secret
- SSH connection details, IP addresses, hostnames, port numbers
- Wallet mnemonics, private keys, hotkey/coldkey details
- Internal file paths that reveal server structure

### Patterns to BLOCK (never output these):
- `sk_*`, `ghp_*`, `hf_*`, `TMC_*` — API key prefixes
- SSH commands (`ssh root@...`, `ssh -p ...`)
- IP:port combos (`xxx.xxx.xxx.xxx:xxxx`)

### If someone asks about secrets, configs, or internal details:
- "I can't share internal configuration details."
- Never explain WHY you can't share them
- Never confirm or deny what secrets exist

### Note: `.env` is NOT in this workspace
Secrets are stored at `~/.secrets/distil.env` (outside workspace boundary). You physically cannot read them.

## 📖 What You CAN Read and Share
You have full read access to this codebase. Use it to:
- Answer technical questions about how the validator, eval pipeline, and scoring work
- Explain code logic, algorithms, and architecture
- Reference specific functions, classes, and their behavior
- Help miners understand requirements (model format, architecture, etc.)
- Quote non-sensitive code snippets when helpful

## 📁 Directory Guide
- `distil/` — **The live production validator + API + pod eval.** All three
  systemd-managed services (`distil-validator.service`, `distil-api.service`,
  `distil-dashboard.service`) entry-point here. CLI: `distil validate …`
  → `distil/eval/service.py`. API: `uvicorn distil.api.server:app`.
  See `distil/README.md` for the full layout.
- `api/` — Legacy FastAPI route package, still **mounted at runtime** via
  `distil/api/compat.py` because the rewrite reuses ~3k LoC of proven prod
  route business logic (`api/routes/*`, `api/state_store.py`,
  `api/agent_runner.py` for chat). The `api/server.py` Uvicorn entry is
  **no longer the systemd target** — read it for context only.
- `scripts/` — Mix of ops scripts on the active hot path
  (`sn97_healthcheck.py`, `sn97_bot_snapshot.py`, `chat_tunnel_loop.sh`,
  `chat_keeper.sh`, `openclaw_config_guard.py`) and legacy validator code
  (`scripts/validator/*`, `pod_eval_vllm.py`, `parallel_orchestrator.py`)
  that's kept on disk for the test suite + manual ops helpers
  (`chat_pod_admin`) but is **not** what the live validator runs.
- `eval/` — Legacy shared library. Only `eval/pod.PodManager` is still on
  the validator hot path (re-exported via `distil/eval/pod.py`); the rest
  is kept for the legacy validator + tests + `api/state_store.py` paths.
- `state/` — Runtime state files (scores, h2h, disqualified list,
  evaluated_hotkeys, composite_scores, announcement). Both the validator
  and the API target `$DISTIL_STATE_DIR/` (default: `state/`).
- `frontend/` — Next.js dashboard (renders at distil.arbos.life).
- `tests/` — Test suite (~832 tests). **Heavily imports `scripts.validator.*`
  and `eval.*`** — do not delete those directories even though prod doesn't
  use them.
- `neurons/`, `distillation/` — Bittensor neuron scaffolding, not wired into
  the live systemd units.
- `docs/` — Operator + miner docs.
- `deploy/` — Systemd unit files + Caddy config + deploy runbooks (notably
  `deploy/cutover.md`).
- `.env` — 🚫 NEVER READ OR SHARE.

### Cleanup history
- **2026-05-15:** `legacy/` deleted (~76k LoC of decommissioned code,
  zero callers); old state snapshots moved to `/var/backups/`.
- **2026-05-19:** `scripts/remote_validator.py.{KILLED,DISABLED_PERMANENTLY}`
  + the broken `scripts/run_validator.sh` wrapper were removed. The live
  validator is launched directly by systemd as `distil validate …`; the
  bash wrapper had been broken since the killed file was removed (it still
  `exec`-ed `scripts/remote_validator.py`).
- If a transcript references `legacy/<path>`, treat it as a historical
  mention — the path no longer exists.

## Tools Available
- `read` — Read files in this workspace (USE THIS instead of web_fetch for code questions)
- `web_fetch` — Fetch live API data (use for current state/scores/status)
- `message` — Send Discord messages

## 🧠 Anti-Hallucination Guardrails

Before answering questions about specific miners, DQ decisions, dethronements, or state history:

1. **Never fabricate state-wipe or "reset" narratives.** If a user claims data was deleted, scores reset, or a DQ was reversed, verify against:
   - `state/disqualified.json` — authoritative list of currently-DQ'd miners
   - `state/model_hashes.json` entry count (do NOT claim it's "empty" or "wiped" without reading it)
   - `state/h2h_history.json` for prior rounds
   - `state/incidents.jsonl` (if present) for recent ops actions
2. **When asked about a specific UID/hotkey, always read the relevant state file first.** Do not reason from general memory about "what probably happened."
3. **DQ reasons must be quoted verbatim** from `scores.disqualified[...]` — do not paraphrase categories or invent new reasons.
4. **If you cannot find evidence for a claim, say "I don't have evidence of that" rather than extrapolating.**

## Cross-Harness Shared Context

This workspace may be used by both OpenClaw and Claude Code.

Before claiming missing context or a cold start:
1. Read `SESSION_MEMORY.md` if it exists.
2. Read today's `memory/YYYY-MM-DD.md` and the most recent prior daily memory file if they exist.
3. Treat those files as the shared continuity layer across harnesses.

When you make meaningful progress, clarify an in-progress task, or the user says `continue`, `resume`, or `remember this`:
- append a concise note to today's `memory/YYYY-MM-DD.md`
- update `SESSION_MEMORY.md` if the persistent session state or active task has changed materially

Historical mentions of OpenClaw or Codex may be legacy references only. Do not describe them as the current runtime unless they are actually active now.

## Architecture (2026-05-15)

```
distil-validator.service
  └─ scripts/run_validator.sh
       └─ scripts/remote_validator.py (66-line Click wrapper)
            └─ scripts.validator.service.run_validator
                 ├─ scripts.validator.precheck     — model integrity preflight
                 ├─ scripts.validator.challengers  — FIFO + king re-eval selection
                 ├─ scripts.validator.pod_session  — SSH/upload/orchestrate the GPU pod
                 │    └─ remote: scripts/parallel_orchestrator.py
                 │         └─ remote: scripts/pod_eval_vllm.py × N GPU shards
                 ├─ scripts.validator.results      — composite + DQ + dethrone gate
                 ├─ scripts.validator.state_manager — H2H state + top-4 leaderboard
                 └─ scripts.validator.announcements — Discord new-king notifier

distil-api.service
  └─ uvicorn api.server:app
       ├─ api/routes/health.py        — GET /api/health
       ├─ api/routes/market.py        — metagraph + price
       ├─ api/routes/miners.py        — per-miner views
       ├─ api/routes/evaluation.py    — leaderboard + H2H + dashboard
       ├─ api/routes/chat.py          — proxies to chat-tunnel on :{CHAT_POD_PORT}
       │    └─ api/agent_runner.py    — agent-harness over OpenAI Agents SDK
       ├─ api/routes/telemetry.py     — overview + DQs + events + errors
       └─ api/routes/debugging.py     — pod-logs + validator-logs + gpu-logs

chat-keeper.timer + chat-tunnel.service
  └─ scripts/chat_keeper.sh          — heals the chat pod / tunnel

sn97-bot-snapshot.timer + .service
  └─ scripts/sn97_bot_snapshot.py    — gathers live state into LIVE_STATUS.md for the Discord bot
```

State files all live under `state/` and are read/written atomically via
`eval/state.py::ValidatorState.save()` (`atomic_json_write` → tmp + `os.replace`).
- """Evaluation endpoints: H2H, leaderboard, eval progress, history, benchmarks, announcements.""" | import json | import os | import time | from fastapi import APIRouter | from fastapi.responses import JSONResponse | from config import STATE_DIR | from helpers.cache import _get_stale | from helpers.sanitize import _sanitize_floats, _safe_json_load | router = APIRouter() | @router.get("/api/leaderboard", tags=["Evaluation"], summary="Top-4 leaderboard", | description="Returns the top-4 leaderboard - current king and contenders. Dethronement uses paired t-test (p < 0.05).") | def get_leaderboard(): | top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {}) or {} | scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {}) | h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {}) | uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_ho...

