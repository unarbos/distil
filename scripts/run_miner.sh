#!/usr/bin/env bash
# Run the Distillation subnet miner.
# Usage: ./scripts/run_miner.sh --model-repo username/distilled-glm5 [extra options]
#
# Environment variables (all optional, CLI flags override):
#   NETWORK     - finney | test | local  (default: finney)
#   NETUID      - subnet UID             (default: 1)
#   WALLET_NAME - coldkey name           (default: default)
#   HOTKEY_NAME - hotkey name            (default: default)

set -euo pipefail
cd "$(dirname "$0")/.."

exec python miner.py "$@"
