#!/usr/bin/env bash
# Run the Distillation subnet validator.
# Usage: ./scripts/run_validator.sh [extra click options]
#
# Environment variables (all optional, CLI flags override):
#   NETWORK             - finney | test | local  (default: finney)
#   NETUID              - subnet UID             (default: 1)
#   WALLET_NAME         - coldkey name           (default: default)
#   HOTKEY_NAME         - hotkey name            (default: default)
#   TEACHER_MODEL       - HF repo for teacher    (default: zai-org/GLM-5)
#   DATASET_PATH        - path to SweInfinite    (default: ./dataset)
#   TENSOR_PARALLEL_SIZE - GPUs per model         (default: 1)
#   LOG_LEVEL           - DEBUG|INFO|WARNING|ERROR

set -euo pipefail
cd "$(dirname "$0")/.."

exec python validator.py "$@"
