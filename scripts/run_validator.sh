#!/bin/bash
cd /home/openclaw/distillation
source .env
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
exec python3 scripts/remote_validator.py \
  --lium-api-key "$LIUM_API_KEY" \
  --lium-pod-name "distil-eval" \
  --tempo 600 \
  --use-vllm
