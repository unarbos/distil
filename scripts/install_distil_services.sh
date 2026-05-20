#!/bin/bash
set -euo pipefail

# Unit-file source layout (post 2026-05 rewrite-v2 cutover):
#
#   deploy/systemd/distil-{api,validator}.service
#       — the LIVE rewrite-v2 units (``distil`` CLI entrypoint,
#         vLLM-aware ExecStart). These are what production runs.
#   scripts/systemd/distil-{api,validator}.service
#       — the LEGACY pre-cutover units. Kept for reference but NOT
#         installed; they reference ``scripts/run_validator.sh`` which
#         was retired in the rewrite-v2 cutover.
#   scripts/systemd/distil-{dashboard,benchmark-sync,tree-guard,
#                            openclaw-config-guard,owui-patches}.*
#       — the smaller cron-style units; same in legacy + rewrite-v2.
#
# This footgun bit production on 2026-05-20 when ``install_distil_services.sh``
# was re-run after a working-tree wipe: the script was sourcing
# ``scripts/systemd/`` for ALL units, including the validator, which
# crashed on startup with ``run_validator.sh: No such file or directory``.
# Fix: pull the two cutover units from ``deploy/systemd/`` and everything
# else from ``scripts/systemd/``.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LEGACY_DIR="$SCRIPT_DIR/systemd"            # pre-cutover units (dashboard, sync, etc.)
CUTOVER_DIR="$REPO_ROOT/deploy/systemd"     # rewrite-v2 (api, validator)
UNIT_DST_DIR="/etc/systemd/system"

# Rewrite-v2 services — MUST come from deploy/systemd/.
install -m 0644 "$CUTOVER_DIR/distil-api.service" "$UNIT_DST_DIR/distil-api.service"
install -m 0644 "$CUTOVER_DIR/distil-validator.service" "$UNIT_DST_DIR/distil-validator.service"

# Cron-style + dashboard units — same across legacy + cutover, live in scripts/systemd/.
install -m 0644 "$LEGACY_DIR/distil-dashboard.service" "$UNIT_DST_DIR/distil-dashboard.service"
install -m 0644 "$LEGACY_DIR/distil-benchmark-sync.service" "$UNIT_DST_DIR/distil-benchmark-sync.service"
install -m 0644 "$LEGACY_DIR/distil-benchmark-sync.timer" "$UNIT_DST_DIR/distil-benchmark-sync.timer"
install -m 0644 "$LEGACY_DIR/distil-tree-guard.service" "$UNIT_DST_DIR/distil-tree-guard.service"
install -m 0644 "$LEGACY_DIR/distil-tree-guard.timer" "$UNIT_DST_DIR/distil-tree-guard.timer"
install -m 0644 "$LEGACY_DIR/distil-wipe-watch.service" "$UNIT_DST_DIR/distil-wipe-watch.service"
# CRITICAL: both the tripwire and the wipe watcher must survive a
# wipe of /opt/distil/repo (otherwise the watchdog dies exactly when
# it's needed). Copy the in-repo sources to /usr/local/sbin/ so the
# systemd ExecStarts can point at stable out-of-repo paths. install
# -C only writes if the bytes differ, so re-running this script
# after a code update is a no-op when there's nothing new to deploy.
install -C -m 0755 "$REPO_ROOT/scripts/tree_guard.py" /usr/local/sbin/distil-tree-guard
install -C -m 0755 "$REPO_ROOT/scripts/distil-wipe-watch.sh" /usr/local/sbin/distil-wipe-watch
if [[ -f "$LEGACY_DIR/openclaw.service" ]]; then
  install -m 0644 "$LEGACY_DIR/openclaw.service" "$UNIT_DST_DIR/openclaw.service"
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
