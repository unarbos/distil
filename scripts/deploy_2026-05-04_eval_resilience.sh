#!/usr/bin/env bash
# Deploy script for the 2026-05-04 eval-resilience fixes.
# Run as the `distil` user (or root) on the validator host.
#
# Context: previous round (epoch 00:49 UTC) crashed when student
# bodenmaurice/distil-new-v2 tripped a CUDA index-out-of-bounds
# device-side assertion in F.embedding (token id >= embed_size).
# The assertion poisoned the CUDA context, free_gpu()'s
# torch.cuda.empty_cache() then re-raised, killing the entire
# pod_eval before it wrote results.json. The validator's
# pod.download() got FileNotFoundError and cleared the round —
# all 9 other students lost.
#
# Fixes (already on disk in /opt/distil/repo, committed below):
#   1. free_gpu() now wraps every CUDA call in try/except so a
#      poisoned context can't propagate out of cleanup.
#   2. _refill_cache() validates input_ids vs the student's actual
#      embed_tokens.weight.shape[0] BEFORE the forward pass and
#      raises a normal Python ValueError("vocab_oob: ...") that the
#      per-prompt try/except DQ's cleanly. No CUDA crash, eval
#      continues with the next student.
#   3. After each student, _cuda_alive() probes the context with a
#      1-element add. If poisoned, _cuda_poisoned=True and every
#      remaining student is recorded as deferred_cuda_poisoned and
#      skipped (no GPU work).
#   4. results.py treats deferred_cuda_poisoned as "skip, do NOT
#      record_failure" so deferred students aren't pushed toward
#      the 3-strikes stale DQ for a problem that wasn't theirs.
#   5. Top-level entrypoint always writes a (partial) results.json
#      via _ensure_results_file before re-raising any exception,
#      so the validator's pod.download() never gets FileNotFoundError.
#
# Plus uncommitted Kimi cutover changes from the in-progress branch:
#   - eval/pod.py: idempotent vLLM kimi_k25 patches (multimodal
#     fallback, dummy_text bypass, etc.) applied via PodManager.
#     ensure_dependencies for every fresh pod.
#   - check_model.py: read max-student-params/vocab/arch-allowlist
#     from frontend/src/lib/subnet-config.json instead of hardcoded
#     7B/Qwen3.5 constants.
#   - README.md, docs/MINER_FAQ.md, docs/MINING_GUIDE_V2.md,
#     SOUL.md, pyproject.toml: post-Kimi-cutover documentation
#     (33B cap, vocab 163,840, Kimi-family arch allowlist).

set -euo pipefail

cd /opt/distil/repo

echo "=== Status ==="
git status -s

echo
echo "=== Commit Kimi cutover docs + check_model.py + eval/pod.py ==="
git add eval/pod.py check_model.py README.md SOUL.md docs/MINER_FAQ.md docs/MINING_GUIDE_V2.md pyproject.toml
git commit -m "$(cat <<'EOF'
chore(kimi-cutover): land docs + check_model + pod-side patches

- check_model.py: read max-student-params (33B), vocab (163,840) and
  arch allowlist from frontend/src/lib/subnet-config.json so the
  pre-submission script tracks live config drift instead of
  hardcoded Qwen3.5 numbers (Discord complaint: "the bot keeps
  saying 7B but the cap is 33B now").
- README.md, docs/MINER_FAQ.md, docs/MINING_GUIDE_V2.md, SOUL.md,
  pyproject.toml: align with the live Kimi K2.6 config (33B cap,
  163,840 vocab, Kimi-family architecture allowlist).
- eval/pod.py: idempotent vLLM kimi_k25.py patches applied per pod
  spin-up (media_tokens_calculator fallback, get_dummy_mm_items
  short-circuit, get_dummy_text bypass) so vLLM serves K2.6
  multimodal stub even without Kimi-specific image processor attrs.
EOF
)"

echo
echo "=== Commit eval-resilience fixes ==="
git add scripts/pod_eval_vllm.py scripts/validator/results.py
git commit -m "$(cat <<'EOF'
fix(pod_eval): never crash the round on a single broken student

Root cause (distil-97 incident 2026-05-04 epoch 00:49 UTC):
student bodenmaurice/distil-new-v2 tripped CUDA's vectorized_gather
"index out of bounds" device-side assertion in F.embedding when a
prompt token's id >= the student's actual embed_tokens.weight.shape[0]
(student claims vocab=163,840 but embedding table is smaller — config
edited to match teacher without retraining embeddings). The assertion
poisoned the CUDA context; the per-prompt RuntimeError handler then
called free_gpu(), whose torch.cuda.empty_cache() re-raised the same
device-side assertion — killing the whole pod_eval process before any
results.json write. validator's pod.download() got FileNotFoundError
and the entire 10-student round was discarded.

Fixes:
1. free_gpu() wraps every CUDA call (empty_cache, ipc_collect,
   synchronize) in try/except so a poisoned context can't propagate
   out of cleanup and kill main().
2. _refill_cache() reads student.get_input_embeddings().weight.shape[0]
   once and validates input_ids before each forward pass. Out-of-range
   ids raise a normal Python ValueError("vocab_oob: ...") caught by
   the per-prompt handler — clean DQ, no CUDA crash, next student
   proceeds.
3. After each student, _cuda_alive() probes the context (zeros+add+
   sync). If poisoned, _cuda_poisoned=True and every remaining
   student is recorded as status=deferred_cuda_poisoned with no GPU
   work, so the final results.json still downloads.
4. validator/results.py treats deferred_cuda_poisoned as a no-op
   (no record_failure) so deferred students aren't pushed toward
   the 3-strikes stale DQ for a problem that wasn't theirs.
5. Top-level entrypoint always writes a (partial or stub)
   results.json via _ensure_results_file before re-raising any
   exception, so pod.download() never gets FileNotFoundError again.

Net effect: the broken student now DQ's at "scoring_error: vocab_oob"
on first prompt (no GPU crash), the other 9 students score normally,
and even if some other student manages a non-CUDA crash the partial
results land in state/last_eval.json.
EOF
)"

echo
echo "=== Push ==="
git push origin HEAD

echo
echo "=== Restart validator (pod_eval upload happens at next round) ==="
sudo systemctl restart distil-validator.service
sleep 3
systemctl status distil-validator.service --no-pager | head -8

echo
echo "=== Tail validator log to watch first epoch ==="
echo "Run this manually if you want to follow:"
echo "  journalctl -u distil-validator.service -f --since now"
