#!/usr/bin/env bash
# Cutover from scripts/ → distil/ for one service at a time.
#
# Usage:
#   sudo bash deploy/cutover.sh api        # flip distil-api only
#   sudo bash deploy/cutover.sh validator  # flip distil-validator only (DANGEROUS — see parity gap)
#   sudo bash deploy/cutover.sh status     # show current ExecStart of both
#   sudo bash deploy/cutover.sh rollback api
#   sudo bash deploy/cutover.sh rollback validator
#
# Pre-cutover gates:
#   * api:        distil.api.server must expose all routes the frontend uses.
#                 See REWRITE_PLAN.md "Remaining gaps before cutover" item 6.
#   * validator:  distil.eval.composite must score the same axes as
#                 scripts.validator.composite for a fixed pod_results.json.
#                 See docs/CUTOVER_PARITY_2026-05-15.md.
#
# This script ONLY touches /etc/systemd/system/<unit>. It tarballs the prior
# unit file under /var/backups/distil-cutover-<date>/ before overwriting so
# rollback is `cp` + daemon-reload + restart.

set -euo pipefail

REPO="/opt/distil/repo"
LIVE="/etc/systemd/system"
DRAFT="${REPO}/deploy/systemd"
BACKUP="/var/backups/distil-cutover-$(date +%Y%m%d-%H%M%S)"

cmd="${1:-status}"
target="${2:-}"

status() {
  for u in distil-api distil-validator; do
    echo "=== ${u}.service ==="
    if [ -f "${LIVE}/${u}.service" ]; then
      grep -E "^ExecStart" "${LIVE}/${u}.service" || true
    else
      echo "  (not installed)"
    fi
  done
}

cutover_one() {
  local unit="$1"
  local src="${DRAFT}/${unit}.service"
  local dst="${LIVE}/${unit}.service"
  if [ ! -f "$src" ]; then
    echo "FATAL: $src missing"; exit 2
  fi
  if [ -f "$dst" ]; then
    mkdir -p "$BACKUP"
    cp "$dst" "${BACKUP}/${unit}.service.bak"
    echo "Backed up live unit → ${BACKUP}/${unit}.service.bak"
  fi
  cp "$src" "$dst"
  systemctl daemon-reload
  systemctl restart "$unit"
  systemctl status "$unit" --no-pager | head -15
}

rollback_one() {
  local target="$1"
  # Accept either "api" / "validator" (the short form passed by callers
  # of cutover.sh ... rollback api) or the full unit name. Either way
  # we restore the most-recent ``distil-${short}.service.bak`` from
  # /var/backups/.
  local unit
  case "$target" in
    api|validator) unit="distil-${target}" ;;
    distil-api|distil-validator) unit="$target" ;;
    *) echo "FATAL: rollback: unknown target '$target' (expected api|validator)"; exit 3 ;;
  esac
  local bak
  bak=$(ls -1t /var/backups/distil-cutover-*/"${unit}.service.bak" 2>/dev/null | head -1 || true)
  if [ -z "$bak" ]; then
    echo "FATAL: no backup of ${unit}.service found under /var/backups/"; exit 3
  fi
  cp "$bak" "${LIVE}/${unit}.service"
  systemctl daemon-reload
  systemctl restart "$unit"
  echo "Rolled back ${unit}.service from $bak"
}

case "$cmd" in
  status)   status ;;
  api)      cutover_one distil-api ;;
  validator)
    echo "WARNING: validator cutover changes how every UID is scored."
    echo "Confirm you've read docs/CUTOVER_PARITY_2026-05-15.md and"
    echo "accept the leaderboard shift, then re-run with --confirm-shift:"
    if [ "${target:-}" != "--confirm-shift" ]; then
      exit 4
    fi
    cutover_one distil-validator
    ;;
  rollback)
    if [ -z "$target" ]; then echo "Usage: $0 rollback (api|validator)"; exit 5; fi
    rollback_one "$target"
    ;;
  *)
    echo "Usage: $0 (status | api | validator --confirm-shift | rollback (api|validator))"
    exit 1
    ;;
esac
