#!/bin/bash
cd /home/openclaw/distillation
source .env
exec python3 scripts/remote_validator.py \
  --lium-api-key "$LIUM_API_KEY" \
  --lium-pod-name "overnight-train" \
  --tempo 600
