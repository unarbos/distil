#!/usr/bin/env bash
# Download/copy the SweInfinite dataset into the repo.
# If /tmp/SweInfinite/dataset exists (local), copy from there.
# Otherwise, clone the repo (placeholder URL — replace with actual).
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="./dataset"

if [ -d /tmp/SweInfinite/dataset ]; then
    echo "Copying SweInfinite dataset from local cache …"
    cp -r /tmp/SweInfinite/dataset "$DEST"
    echo "Done — $(ls "$DEST"/*.json 2>/dev/null | wc -l) JSON files copied."
else
    echo "Local dataset not found at /tmp/SweInfinite/dataset."
    echo "Please clone or download the SweInfinite dataset and place it at $DEST"
    exit 1
fi
