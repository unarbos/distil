#!/usr/bin/env bash
# Watches state/h2h_latest.json for the next round-publish event, then
# flips distil-validator from the legacy `scripts/` stack to the rewrite
# `distil/` stack via deploy/cutover.sh.
#
# Why this script: flipping the validator in the middle of a round
# destroys the in-flight teacher-API output (kept in process memory +
# pod-side scratch, not picked up by the resume-on-attach hook in
# distil/eval/service.py). The safe window is the inter-round tempo gap
# AFTER h2h_latest.json + scores.json + activation_fingerprints.json are
# all on disk and BEFORE the next round starts.
#
# Lifecycle, by design:
#   1. Stamp the initial mtime of state/h2h_latest.json.
#   2. Poll every 30s. When the mtime advances → the just-completed
#      round has committed its dashboard payload.
#   3. Wait 45s for trailing writes (composite_scores.json, scores.json,
#      activation_fingerprints.json, dq_history.json) and for the
#      validator's set_weights() extrinsic to confirm.
#   4. Run `cutover.sh validator --confirm-shift` (one systemctl restart;
#      legacy unit is backed up under /var/backups/distil-cutover-<ts>/).
#   5. Sleep 30s, then verify distil-validator is `active (running)` and
#      not crash-looping. If it is, log and bail (the operator can roll
#      back with `cutover.sh rollback validator`).
#
# Hard timeout: 120 min. The current round should finish in ≤45 min;
# 120 min covers a 2x worst case before declaring something went wrong.
#
# Log: journalctl -u distil-validator-cutover.service -f

set -euo pipefail

LOG_TAG="cutover-watcher"
H2H=/opt/distil/repo/state/h2h_latest.json
DEADLINE_MIN=120
POLL_S=30
TRAILING_GRACE_S=45
POST_FLIP_GRACE_S=30

log() { echo "$(date -u +%H:%M:%S) [$LOG_TAG] $*"; }

if [ ! -f "$H2H" ]; then
    log "FATAL: $H2H missing"; exit 2
fi
START_MTIME=$(stat -c %Y "$H2H")
log "starting; initial h2h_latest mtime = $START_MTIME ($(date -u -d "@$START_MTIME" +%H:%M:%S))"
log "polling every ${POLL_S}s; deadline = ${DEADLINE_MIN} min"

t0=$(date +%s)
while true; do
    now=$(date +%s)
    elapsed_min=$(( (now - t0) / 60 ))
    if [ "$elapsed_min" -ge "$DEADLINE_MIN" ]; then
        log "FATAL: deadline reached without round publish; aborting"
        exit 3
    fi
    cur=$(stat -c %Y "$H2H" 2>/dev/null || echo 0)
    if [ "$cur" != "$START_MTIME" ]; then
        log "h2h_latest.json updated: $START_MTIME -> $cur"
        log "round published; sleeping ${TRAILING_GRACE_S}s for trailing writes + set_weights to confirm"
        sleep "$TRAILING_GRACE_S"
        break
    fi
    sleep "$POLL_S"
done

log "running cutover.sh validator --confirm-shift"
if ! bash /opt/distil/repo/deploy/cutover.sh validator --confirm-shift; then
    log "FATAL: cutover.sh exited non-zero; legacy unit still backed up under /var/backups/"
    exit 4
fi

log "sleeping ${POST_FLIP_GRACE_S}s before health check"
sleep "$POST_FLIP_GRACE_S"

state=$(systemctl is-active distil-validator.service || true)
log "distil-validator.service is-active = $state"
log "--- last 20 journal lines ---"
journalctl -u distil-validator.service --since "1 minute ago" --no-pager | tail -20

if [ "$state" != "active" ]; then
    log "FATAL: distil-validator failed to start. Rollback: sudo bash /opt/distil/repo/deploy/cutover.sh rollback validator"
    exit 5
fi

log "cutover complete; distil-validator is active. exiting cleanly."
