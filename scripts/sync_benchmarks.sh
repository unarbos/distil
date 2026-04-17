#!/bin/bash
# Sync benchmark results from the eval host into local state/benchmarks/.
# Exits 0 even when the remote is unreachable so systemd does not log-spam.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EVAL_POD="${DISTIL_BENCHMARK_HOST:-root@213.13.7.110}"
EVAL_PORT="${DISTIL_BENCHMARK_PORT:-6039}"
REMOTE_DIR="${DISTIL_BENCHMARK_REMOTE_DIR:-/root/benchmark_results/}"
LOCAL_DIR="${DISTIL_BENCHMARK_LOCAL_DIR:-$REPO_ROOT/state/benchmarks/}"

mkdir -p "$LOCAL_DIR"

RSYNC_ERR="$(mktemp)"
trap 'rm -f "$RSYNC_ERR"' EXIT

if ! rsync -az --timeout=10 \
      -e "ssh -p $EVAL_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no" \
      --include='uid_*_summary.json' \
      --exclude='*' \
      "$EVAL_POD:$REMOTE_DIR" "$LOCAL_DIR" 2>"$RSYNC_ERR"; then
    REASON="$(tr '\n' ' ' < "$RSYNC_ERR" | sed 's/  */ /g' | head -c 200)"
    echo "benchmark-sync: remote unreachable (${REASON:-unknown}); skipping this cycle" >&2
    exit 0
fi

NEW=$(find "$LOCAL_DIR" -name 'uid_*_summary.json' -newer "$LOCAL_DIR/.last_sync" 2>/dev/null | wc -l)
touch "$LOCAL_DIR/.last_sync"
if [ "$NEW" -gt 0 ]; then
    echo "Synced $NEW new benchmark summaries"
fi
