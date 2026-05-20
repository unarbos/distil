#!/bin/bash
# distil-wipe-watch — real-time wipe forensics
#
# Watches /opt/distil/repo for mass deletions and, the moment one is
# detected, snapshots every piece of context that could fingerprint
# the perpetrator. The first three wipes (May 18, May 19 22:51, May 20
# 12:20) all happened within ~30-50 seconds of a fresh Cursor remote
# dev session opening from 81.17.123.x — but the actual `rm`/`git
# clean`/`git checkout` command never showed up in journald. Cursor
# tool calls bypass bash's history file, and the bash that hosts the
# agent's Shell tool is short-lived, so its process listing vanishes
# immediately after the call returns.
#
# Strategy: inotifywait on the tracked subdirs (scripts/, distil/, api/,
# tests/, eval/, etc.) in `delete,move_self,delete_self` mode. As soon
# as ≥5 deletions land within a 2s window we snapshot:
#
#   * lsof -nP on /opt/distil/repo (which processes hold file
#     descriptors to anything inside)
#   * Full process tree for every process whose cwd or any open fd
#     touches /opt/distil/repo (the "smoking gun")
#   * `ss -tnp | grep <cursor-server-pid>` — to confirm whether an
#     active Cursor remote session was holding fds
#   * Recent journalctl excerpt for sshd + sudo + python
#   * The bash-history files for every shell that exists right now
#
# Output: /opt/distil/repo/state/wipe_forensics/<timestamp>.json plus a
# Discord post (suppressed in QUIET mode).
#
# Runs as a long-lived systemd service. Lightweight — inotify is kernel
# event-driven, idle CPU near zero.

set -uo pipefail

REPO_ROOT="${DISTIL_REPO_ROOT:-/opt/distil/repo}"
OUT_DIR="$REPO_ROOT/state/wipe_forensics"
mkdir -p "$OUT_DIR"

# Threshold: ≥N deletions within WINDOW_S seconds → snapshot.
THRESHOLD="${DISTIL_WIPE_THRESHOLD:-5}"
WINDOW_S="${DISTIL_WIPE_WINDOW_S:-2}"
# Cooldown so a single wipe doesn't fire 100 snapshots.
COOLDOWN_S="${DISTIL_WIPE_COOLDOWN_S:-300}"

# Subdirs to watch. .git is intentionally excluded — git's own
# operations create constant churn there and would mask the signal.
WATCH_PATHS=(
  "$REPO_ROOT/scripts"
  "$REPO_ROOT/distil"
  "$REPO_ROOT/api"
  "$REPO_ROOT/tests"
  "$REPO_ROOT/eval"
  "$REPO_ROOT/deploy"
)
WATCH_EXISTING=()
for p in "${WATCH_PATHS[@]}"; do
  [[ -d "$p" ]] && WATCH_EXISTING+=("$p")
done

if [[ ${#WATCH_EXISTING[@]} -eq 0 ]]; then
  echo "no watch paths exist — sleeping" >&2
  sleep 60
  exit 0
fi

LAST_FIRE=0
DELETE_COUNT=0
WINDOW_START=0

snapshot() {
  local ts now
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  now="$(date +%s)"
  local out="$OUT_DIR/$ts.json"
  {
    echo "{"
    echo "  \"ts\": $now,"
    echo "  \"ts_iso\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"trigger\": \"inotify delete burst (≥$THRESHOLD in ${WINDOW_S}s)\","
    echo "  \"deletes_in_window\": $DELETE_COUNT,"
    echo "  \"lsof\": $(lsof -nP +D "$REPO_ROOT" 2>/dev/null | head -200 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"ps_full\": $(ps -ef --forest 2>/dev/null | head -120 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"cursor_sessions\": $(ps -ef 2>/dev/null | grep -E 'cursor-server|cursor-agent' | grep -v grep | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"ss_to_cursor\": $(ss -tnp 2>/dev/null | grep -E 'cursor|node' | head -20 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"recent_sshd\": $(journalctl --since '3 minutes ago' --no-pager 2>/dev/null | grep -iE 'sshd|sudo|session opened' | tail -40 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"recent_python\": $(journalctl _COMM=python --since '3 minutes ago' --no-pager 2>/dev/null | tail -40 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"git_status_tail\": $(cd "$REPO_ROOT" && git status --short 2>/dev/null | head -50 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"open_terminals\": $(ls /root/.cursor/projects/opt-distil/terminals/ 2>/dev/null | head -20 | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')"
    echo "}"
  } > "$out"
  echo "[wipe-watch] $(date -u +%H:%M:%S) snapshot $out (deletes=$DELETE_COUNT)" >&2
}

# Stream every delete event from inotifywait.
# --format outputs WATCHED_DIR FILENAME ACTION — we just count.
inotifywait -m -r -q -e delete -e delete_self -e move_self \
  --format '%T %w %f' --timefmt '%s' \
  "${WATCH_EXISTING[@]}" 2>/dev/null | while read -r event_ts watched fname; do
  now="$(date +%s)"
  # roll window
  if (( now - WINDOW_START > WINDOW_S )); then
    WINDOW_START=$now
    DELETE_COUNT=1
  else
    DELETE_COUNT=$((DELETE_COUNT + 1))
  fi
  # fire?
  if (( DELETE_COUNT >= THRESHOLD )) && (( now - LAST_FIRE > COOLDOWN_S )); then
    LAST_FIRE=$now
    snapshot
    DELETE_COUNT=0
  fi
done
