#!/bin/bash
# SSH tunnel for chat.arbos.life — reads coordinates from state/chat_pod.json
#
# Pre-2026-04-26 the tunnel ExecStart hardcoded the chat pod IP/port, so any
# Lium reprovisioning meant manually editing the systemd unit. The state-file
# layout combined with chat-tunnel.path lets ops rotate the chat pod with one
# CLI command (``python -m scripts.validator.chat_pod_admin set --host ...``)
# and have the tunnel pick it up without daemon-reload.
#
# This script intentionally exits (instead of looping) on missing/invalid
# state so systemd's Restart=always + chat-tunnel.path watcher together
# drive convergence: state changes trigger a service restart, and a missing
# pod just sleeps + retries until ops re-registers a host.
set -euo pipefail

STATE_DIR="${DISTIL_STATE_DIR:-/opt/distil/repo/state}"
STATE_FILE="$STATE_DIR/chat_pod.json"
DEFAULT_KEY="${HOME:-/root}/.ssh/id_ed25519"

if [ ! -f "$STATE_FILE" ]; then
  echo "[chat-tunnel] state file missing: $STATE_FILE" >&2
  # Sleep so systemd Restart=always doesn't spin at 100% CPU when chat is
  # intentionally undefined. Restart picks up the file on next iteration.
  sleep 30
  exit 1
fi

read_field() {
  python3 - "$STATE_FILE" "$1" <<'PY'
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

HOST="$(read_field host)"
SSH_PORT="$(read_field ssh_port)"
APP_PORT="$(read_field app_port)"
SSH_KEY="$(read_field ssh_key)"

[ -z "$APP_PORT" ] && APP_PORT=8100
[ -z "$SSH_KEY" ] && SSH_KEY="$DEFAULT_KEY"

if [ -z "$HOST" ] || [ -z "$SSH_PORT" ] || [ "$SSH_PORT" = "0" ]; then
  echo "[chat-tunnel] chat pod not configured (host=$HOST port=$SSH_PORT)" >&2
  sleep 30
  exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
  echo "[chat-tunnel] SSH key missing: $SSH_KEY" >&2
  sleep 30
  exit 1
fi

echo "[chat-tunnel] forwarding localhost:${APP_PORT} -> ${HOST}:${SSH_PORT}/${APP_PORT} (key=${SSH_KEY})"

# autossh -M 0 disables the monitoring port; ServerAliveInterval/CountMax
# do the keepalive instead. ExitOnForwardFailure makes us bounce on
# host-key churn, which the systemd Restart=always covers.
exec /usr/bin/autossh -M 0 \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "StrictHostKeyChecking=no" \
  -o "UserKnownHostsFile=/dev/null" \
  -o "ExitOnForwardFailure=yes" \
  -o "BatchMode=yes" \
  -i "$SSH_KEY" \
  -N -L "${APP_PORT}:localhost:${APP_PORT}" \
  -p "$SSH_PORT" "root@$HOST"
