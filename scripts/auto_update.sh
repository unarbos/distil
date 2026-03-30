#!/bin/bash
# Auto-update loop for distil validator
# Checks for new commits every 5 minutes, pulls and restarts if updated
#
# Usage: bash scripts/auto_update.sh [--interval 300]

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL=${1:-300}  # seconds between checks, default 5 min

cd "$REPO_DIR" || exit 1

echo "[auto-update] Watching $REPO_DIR for updates (every ${INTERVAL}s)"

while true; do
    sleep "$INTERVAL"

    # Fetch latest
    git fetch origin main --quiet 2>/dev/null || continue

    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/main 2>/dev/null)

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "[auto-update] New commits detected, pulling..."
        git pull origin main --quiet 2>&1

        # Reinstall deps if pyproject.toml changed
        if git diff "$LOCAL" "$REMOTE" --name-only | grep -q "pyproject.toml"; then
            echo "[auto-update] pyproject.toml changed, reinstalling..."
            pip install . --quiet 2>&1 || pip install "bittensor>=8.0.0" "bittensor-wallet>=2.0.0" "click>=8.0.0" "transformers>=4.45.0" "huggingface-hub>=0.20.0" "numpy>=1.26.0" "torch>=2.1.0" "safetensors>=0.4.0" --quiet 2>&1
        fi

        echo "[auto-update] Updated to $(git rev-parse --short HEAD). Restarting validator..."

        # Kill the validator process (it will be restarted by the run script)
        pkill -f "python3.*validator.py" 2>/dev/null || true
        pkill -f "python3.*remote_validator.py" 2>/dev/null || true

        echo "[auto-update] Validator stopped. It should auto-restart via PM2/systemd/supervisor."
    fi
done
