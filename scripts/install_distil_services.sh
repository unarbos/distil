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

systemctl daemon-reload
systemctl enable distil-api distil-dashboard distil-validator
systemctl enable --now distil-benchmark-sync.timer
