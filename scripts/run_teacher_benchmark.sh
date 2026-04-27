#!/bin/bash
# Run the auto-bench evalscope battery against the teacher pod.
#
# Usage:
#   ./scripts/run_teacher_benchmark.sh [LIMIT]
#
# LIMIT defaults to 100 — same cut as run_king_benchmark.py uses for
# the king. This is the methodologically consistent comparison the
# dashboard's Bench tab needs (vs the model-card numbers the teacher
# row currently displays as a placeholder).
#
# Prereq: a teacher vLLM serving Qwen3.5-35B-A3B reachable over SSH
# tunnel on localhost:8101 (one port up from the chat-king tunnel at
# 8100, so they don't collide). The validator's eval pod hosts the
# teacher during eval rounds; spin a separate Lium pod when eval is
# in flight.
#
# After this completes, state/benchmarks/teacher_qwen35_35b.json is
# regenerated with first-party numbers (real eval_seconds, real
# counts) and the dashboard auto-picks it up.
set -euo pipefail

LIMIT="${1:-100}"
TEACHER_VLLM_URL="${TEACHER_VLLM_URL:-http://127.0.0.1:8101/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-35B-A3B}"

# Probe the teacher vLLM is up
if ! curl -fsS "${TEACHER_VLLM_URL}/models" >/dev/null 2>&1; then
    echo "Teacher vLLM not reachable at ${TEACHER_VLLM_URL}" >&2
    echo "Start a teacher pod (35B vLLM) and SSH-tunnel it to 8101 first." >&2
    exit 1
fi

cd "$(dirname "$0")/.."

env LIMIT="${LIMIT}" \
    KING_UID=-1 \
    KING_MODEL="${TEACHER_MODEL}" \
    BENCHMARKS=gsm8k,humaneval,ifeval,bbh,arc,mmlu_pro \
    VLLM_URL="${TEACHER_VLLM_URL}" \
    VLLM_MODEL_ID="$(curl -sS "${TEACHER_VLLM_URL}/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" \
    OUT_NAME=teacher_qwen35_35b \
    EVAL_WORK_DIR=/tmp/teacher_benchmark \
    /opt/distil/venv/bin/python scripts/run_king_benchmark.py

# Patch the file with the teacher flag so BenchPanel finds it as the
# teacher row (run_king_benchmark sets is_king=True by default).
python3 << 'PY'
import json, time
p = '/opt/distil/repo/state/benchmarks/teacher_qwen35_35b.json'
d = json.load(open(p))
d.pop('is_king', None)
d['is_teacher'] = True
d['label'] = 'Teacher · Qwen3.5-35B-A3B (first-party evalscope, 100-item cut)'
d.pop('uid', None)
d.pop('source', None)
d['note'] = 'First-party evalscope run on the teacher pod for methodological consistency with the king bench.'
open(p, 'w').write(json.dumps(d, indent=2))
print('patched', p)
PY

echo "Done. The dashboard's Bench tab will pick this up on next refresh."
