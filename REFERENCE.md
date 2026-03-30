# Distil SN97 — Technical Reference

Auto-generated reference for common questions. Source of truth: the code in this repo.

## Commitment System

- **Commitments are PERMANENT.** One model per hotkey, forever.
- You CANNOT resubmit, change, or update your committed model.
- To try a different model, you need a new hotkey + new UID.
- This prevents gaming: submit → see score → swap if bad.
- Commitment is done on-chain via `btcli commit` with model repo + revision hash.
- The validator reads commitments from the Bittensor metagraph.

## King/Challenger (H2H Tournament)

- The **king** is the miner with the lowest KL divergence score.
- Each eval epoch, the king defends against new **challengers**.
- All models (king + challengers) are evaluated on the **same prompts** in the same epoch.
- The king is re-evaluated every epoch (not just once).
- Once a challenger **loses** (scored higher KL than king), they are added to `evaluated_uids` and **never re-evaluated** — the model is permanent, so the result won't change.
- New miners who haven't been evaluated yet are picked as challengers.

## Epsilon Threshold

- A challenger must beat the king's KL by more than **1% (epsilon)** to dethrone them.
- If a challenger beats the king but not by >1%, their score gets **pinned** to prevent false leaderboard flips.
- This prevents noise from causing unnecessary king changes.

## Scoring

- Metric: **KL divergence** — KL(teacher || student) on teacher-generated continuations.
- Lower KL = better (student distribution is closer to teacher).
- Teacher model: `Qwen/Qwen3.5-35B-A3B` (35B parameter MoE).
- Student models: up to ~5B parameters (configurable via `--max-params-b`).
- Prompts are sampled from the full **FineWeb** dataset (1.5T tokens) using block-seeded shuffle.
- Each epoch samples fresh prompts — no fixed prompt pool.
- The validator generates teacher continuations (512 tokens per prompt), then scores each student on those same continuations.

## Disqualification

- **Architecture check**: Model must be ≤ max params, correct vocab size, loadable.
- **Integrity check**: Model hash must match the committed revision.
- **Copy detection**: Logit fingerprinting detects functional copies (cosine similarity > 0.9999).
- **Transient errors** (429 rate limits, timeouts, connection errors): Model is **skipped and retried next epoch**, NOT permanently DQ'd.
- Permanent DQ reasons: wrong architecture, hash mismatch, functional copy.

## Evaluation Pipeline

1. Validator syncs metagraph to get all committed models.
2. Pre-checks: architecture, integrity, tokenizer compatibility (no GPU needed).
3. Prompts sampled from FineWeb using current block as seed.
4. Teacher generates continuations + logits on GPU.
5. Each student model scored against teacher logits (KL divergence per token, averaged).
6. Early stopping: if a student's running KL mean + confidence interval exceeds the best KL so far, scoring stops early.
7. Results written to state files, weights set on-chain.

## Earnings

- Total subnet emissions: check TaoMarketCap API.
- King gets weight 1.0 (all emissions).
- Challengers get weight based on their score relative to king.
- API endpoint: `https://api.arbos.life/api/scores`

## Validator Setup

- Hardware: 1 GPU with ≥48GB VRAM recommended for eval.
- The validator runs locally and dispatches GPU evaluation to a remote pod.
- Auto-update: `pm2 start scripts/auto_update.sh --name distil-autoupdate`
- Quick start in README.

## API Endpoints

- `GET /api/scores` — current scores, king, disqualified UIDs
- `GET /api/miners` — all committed miners with model info
- `GET /api/progress` — current eval progress (live during scoring)
- Dashboard: `https://distil.arbos.life`
