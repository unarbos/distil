#!/bin/bash
# Sync state files to the separate API server every 15s
# This runs on the main server (openclaw-shabor) and pushes to distil-api
#
# Uses --temp-dir and --delay-updates to prevent partial reads on remote:
#   --temp-dir: writes to a temp dir first, not directly to target
#   --delay-updates: moves all files into place at the end atomically

REMOTE="distil-api"
REMOTE_STATE="/opt/distil/state/"
LOCAL_STATE="/home/openclaw/distillation/state/"

rsync -az --timeout=10 \
  --temp-dir=/tmp \
  --delay-updates \
  --include='*.json' \
  --exclude='*.tmp' \
  --exclude='*' \
  "$LOCAL_STATE" "$REMOTE:$REMOTE_STATE" 2>/dev/null
