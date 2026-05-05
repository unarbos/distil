#!/bin/bash
# chat-king watchdog (Directive 5: keep chat.arbos.life alive).
#
# Runs every few minutes via chat-keeper.timer. Sequence:
#   1. Curl the local SSH tunnel (127.0.0.1:8100/v1/models). If it answers,
#      bail out — both tunnel + vLLM are healthy.
#   2. If the tunnel is up but vLLM is silent, call chat_pod_admin probe to
#      distinguish a vLLM crash from a tunnel hiccup. Probe SSHes through
#      the chat pod's own SSH port, so it bypasses our local forward.
#   3. On vLLM-down, re-launch via chat_pod_admin heal --model <king>.
#      The king model is read from state/king.json (same source the API uses)
#      with state/chat_pod.json as a fallback.
#
# Why a watchdog instead of relying on validator side_effects:
#   sync_king_runtime fires once per eval round (~60-90 min). That's far
#   too long an outage window for a public chat surface. The watchdog
#   closes the gap and is idempotent against double-launch since
#   chat_server.py kills stale workers before bind.
#
# Why not embed in distil-api: keeping the keeper in its own systemd unit
# means a stuck API process can't take chat down with it, and ops can
# disable just the keeper for emergency manual control via `systemctl`.
set -uo pipefail

REPO_ROOT="${DISTIL_REPO_ROOT:-/opt/distil/repo}"
STATE_DIR="${DISTIL_STATE_DIR:-${REPO_ROOT}/state}"
APP_PORT="${CHAT_KEEPER_PORT:-8100}"
TIMEOUT="${CHAT_KEEPER_TIMEOUT:-8}"

log() {
  echo "[chat-keeper] $*"
}

read_field() {
  local file="$1" key="$2"
  python3 - "$file" "$key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        data = json.load(f) or {}
except Exception:
    data = {}
val = data.get(key)
print("" if val is None else val)
PY
}

resolve_king_model() {
  # Authority order: h2h_latest.json (validator's live king), then
  # chat_pod.json (last successful heal). Both are JSON; h2h_latest stamps
  # ``king_model`` while chat_pod.json keys on ``model``.
  local h2h_file="${STATE_DIR}/h2h_latest.json"
  local chat_file="${STATE_DIR}/chat_pod.json"
  local model=""
  [ -f "$h2h_file" ] && model="$(read_field "$h2h_file" king_model)"
  if [ -z "$model" ] && [ -f "$chat_file" ]; then
    model="$(read_field "$chat_file" model)"
  fi
  echo "$model"
}

probe_local() {
  curl -fsS --max-time "$TIMEOUT" "http://127.0.0.1:${APP_PORT}/v1/models" >/dev/null 2>&1
}

# probe_local_tool_calls — verify vLLM accepts ``tool_choice: "auto"`` requests.
#
# 2026-05-05: the chat-bench pod can be "alive" (``/v1/models`` returns 200)
# but still serve every chat.arbos.life turn as HTTP 400 if it was launched
# without ``--tool-call-parser``. Open-WebUI sends every conversation with
# the SN97 status toolkit attached and ``function_calling=native``, so a
# missing parser breaks chat for every user while ``probe_local`` says
# everything is fine. We flush this whole class of regressions out by
# sending a *minimal* tool-enabled completion (``max_tokens=1``, empty
# tool-args schema) and treating an HTTP 400 response as broken even if
# ``/v1/models`` is happy. Anything else (200 / 500 / network error) is
# treated as "tool-call wiring is at least configured" — we don't want to
# trigger a heal on transient model-side issues like OOM or context
# overflow.
probe_local_tool_calls() {
  local body
  body='{"model":"sn97-king","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"tools":[{"type":"function","function":{"name":"_keeper_probe","description":"noop","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto"}'
  local code
  code=$(curl -s -o /dev/null --max-time "$TIMEOUT" \
    -H 'Content-Type: application/json' \
    -X POST -d "$body" \
    -w '%{http_code}' \
    "http://127.0.0.1:${APP_PORT}/v1/chat/completions" 2>/dev/null)
  [ "$code" = "400" ] && return 1
  return 0
}

probe_remote() {
  cd "$REPO_ROOT" || return 1
  python3 -m scripts.validator.chat_pod_admin probe >/dev/null 2>&1
}

heal_remote() {
  local model="$1"
  cd "$REPO_ROOT" || return 1
  if [ -n "$model" ]; then
    python3 -m scripts.validator.chat_pod_admin heal --model "$model"
  else
    python3 -m scripts.validator.chat_pod_admin heal
  fi
}

if probe_local; then
  if probe_local_tool_calls; then
    log "ok (local tunnel + vLLM + tool-call wiring)"
    exit 0
  fi
  # vLLM is alive but rejecting ``tool_choice: "auto"`` with HTTP 400.
  # That means it was launched without ``--tool-call-parser`` (or with a
  # name vLLM doesn't recognise) — every chat.arbos.life turn is broken
  # for end users even though models endpoint is happy. Force a full
  # heal so the bootstrapper re-runs and picks up the right
  # family-specific parser flags.
  log "vLLM up but rejects tool_choice=auto (HTTP 400) — forcing heal"
  KING_MODEL="$(resolve_king_model)"
  if [ -z "$KING_MODEL" ]; then
    log "no king model recorded yet; cannot heal tool-call wiring"
    exit 0
  fi
  if heal_remote "$KING_MODEL"; then
    log "heal command issued for tool-call wiring — vLLM warm-up takes ~60s"
    exit 0
  fi
  log "heal failed"
  exit 1
fi

log "local probe failed — investigating"

# If the remote vLLM responds via the dedicated SSH probe but the local
# tunnel doesn't, the issue is the tunnel, not the chat server. Bouncing
# chat-tunnel.service is cheap and safe (it auto-reads state file on
# start). Skip the heavy heal in that case.
if probe_remote; then
  log "remote vLLM healthy but local tunnel stale; restarting chat-tunnel"
  systemctl restart chat-tunnel.service 2>&1 | sed 's/^/[chat-keeper] /'
  exit 0
fi

KING_MODEL="$(resolve_king_model)"
if [ -z "$KING_MODEL" ]; then
  log "no king model recorded yet (state/king.json + chat_pod.json both empty); cannot heal"
  exit 0
fi

log "remote vLLM down; healing with model=${KING_MODEL}"
if heal_remote "$KING_MODEL"; then
  log "heal command issued — vLLM warm-up takes ~60s"
else
  log "heal failed"
  exit 1
fi
