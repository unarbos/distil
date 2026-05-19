#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC_DIR="$SCRIPT_DIR/systemd"
UNIT_DST_DIR="/etc/systemd/system"

install -m 0644 "$UNIT_SRC_DIR/distil-api.service" "$UNIT_DST_DIR/distil-api.service"
install -m 0644 "$UNIT_SRC_DIR/distil-dashboard.service" "$UNIT_DST_DIR/distil-dashboard.service"
install -m 0644 "$UNIT_SRC_DIR/distil-validator.service" "$UNIT_DST_DIR/distil-validator.service"
install -m 0644 "$UNIT_SRC_DIR/distil-benchmark-sync.service" "$UNIT_DST_DIR/distil-benchmark-sync.service"
install -m 0644 "$UNIT_SRC_DIR/distil-benchmark-sync.timer" "$UNIT_DST_DIR/distil-benchmark-sync.timer"
install -m 0644 "$UNIT_SRC_DIR/distil-tree-guard.service" "$UNIT_DST_DIR/distil-tree-guard.service"
install -m 0644 "$UNIT_SRC_DIR/distil-tree-guard.timer" "$UNIT_DST_DIR/distil-tree-guard.timer"
if [[ -f "$UNIT_SRC_DIR/openclaw.service" ]]; then
  install -m 0644 "$UNIT_SRC_DIR/openclaw.service" "$UNIT_DST_DIR/openclaw.service"
fi

systemctl daemon-reload
systemctl enable distil-api distil-dashboard distil-validator
systemctl enable --now distil-benchmark-sync.timer
# Tree-guard catches working-tree mass deletion (observed twice in May
# 2026) and auto-checkpoints stable WIP so edits never live only on
# disk. See scripts/tree_guard.py for the rate-limiting + restore
# semantics.
systemctl enable --now distil-tree-guard.timer
if [[ -f /root/.openclaw/openclaw.json ]] && command -v openclaw >/dev/null 2>&1; then
  systemctl enable --now openclaw
fi
