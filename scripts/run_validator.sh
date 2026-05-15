#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${DISTIL_ENV_FILE:-$HOME/.secrets/distil.env}"
PYTHON_BIN="${DISTIL_PYTHON:-}"

if [ -z "$PYTHON_BIN" ]; then
    for candidate in "$REPO_ROOT/.venv/bin/python" "/opt/distil/venv/bin/python" "$(command -v python3)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

cd "$REPO_ROOT"

if [ -f "$ENV_FILE" ]; then
    # Auto-export all sourced vars so python subprocess inherits them.
    # Matters for BENCH_BATTERY_SHADOW_AXES (2026-04-24) and any future
    # tunables we want to flip without editing this script.
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo '')}"

# 2026-05-15: do NOT pass --lium-api-key on argv — it would be visible
# via ``ps auxf`` to anyone with shell access. ``remote_validator.py``'s
# click option declares ``envvar="LIUM_API_KEY"`` and is no longer
# ``required=True`` on the CLI, so passing it via the inherited env is
# sufficient (and what every other secret in distil.env already does).
if [ -z "${LIUM_API_KEY:-}" ]; then
    echo "[run_validator] FATAL: LIUM_API_KEY not in env; check $ENV_FILE" >&2
    exit 2
fi
export LIUM_API_KEY

exec "$PYTHON_BIN" scripts/remote_validator.py \
  --wallet-name "${DISTIL_WALLET_NAME:-affine}" \
  --hotkey-name "${DISTIL_HOTKEY_NAME:-validator}" \
  --wallet-path "${DISTIL_WALLET_PATH:-$HOME/.bittensor/wallets}" \
  --state-dir "${DISTIL_STATE_DIR:-$REPO_ROOT/state}" \
  --lium-pod-name "${DISTIL_LIUM_POD_NAME:-distil-eval}" \
  --tempo "${DISTIL_VALIDATOR_TEMPO:-600}" \
  --use-vllm
