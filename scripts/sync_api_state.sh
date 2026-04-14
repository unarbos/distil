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
  --exclude='announcement.json' \
  --include='*.json' \
  --include='api_cache/' \
  --include='api_cache/*.json' \
  --include='pod_logs/' \
  --include='pod_logs/*.log' \
  --include='eval_data/' \
  --include='eval_data/*.json' \
  --include='benchmarks/' \
  --include='benchmarks/*.json' \
  --exclude='*.tmp' \
  --exclude='*' \
  "$LOCAL_STATE" "$REMOTE:$REMOTE_STATE" 2>/dev/null

# Sync code revision for /api/health display
git -C /home/openclaw/distillation rev-parse --short HEAD 2>/dev/null > /tmp/distil-revision.txt \
  && rsync -az /tmp/distil-revision.txt "$REMOTE:/opt/distil/REVISION" 2>/dev/null
