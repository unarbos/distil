# UID 18 Infinite Re-Eval Loop Fix

**Date:** 2026-04-02
**Branch:** `improvements/validator-fixes-v2`
**File:** `scripts/remote_validator.py`

## Problem

`loveisgone/toanzenai_v10` (UID 18) scored KL=11.36, far above `MAX_KL_THRESHOLD` (2.0). Two bugs conspired to create an infinite re-evaluation loop:

1. **`validate_state_consistency` (Check 6, line ~292):** Removed both the score AND the UID from `evaluated_uids` for any model scoring >= MAX_KL_THRESHOLD.
2. **`model_score_history` recording (line ~1379):** Only recorded scores when `0 < score <= MAX_KL_THRESHOLD`.

Result: The model's score was wiped, its evaluated status was cleared, AND it was never recorded in history. Every epoch it appeared as a "new" challenger and got re-evaluated, wasting GPU time.

## Fixes Applied

### Fix 1: State Validation — Keep High-KL UIDs Evaluated
- High-KL scores are now **capped** at `MAX_KL_THRESHOLD + 1` (sentinel value) instead of being removed
- The UID **stays** in `evaluated_uids` — it's marked as evaluated
- Only truly garbage scores (NaN, Inf, negative) still remove the UID

### Fix 2: Record ALL Scores in Model History
- Models scoring above threshold now get recorded with a `worst_kl` key
- If no `best_kl` exists, the bad score is used as a floor for the skip logic
- This ensures the challenger selection's "historically bad" skip works for terrible models

### Fix 3: Permanently Bad Models List
- New file: `state/permanently_bad_models.json`
- Models scoring > 10x king's KL are added to this list
- Challenger selection checks this list first (fastest skip path)
- Belt-and-suspenders: even if history gets corrupted, these models won't waste GPU

### Fix 4: SFTP Poll Interval
- Changed `poll_stop.wait(5)` → `poll_stop.wait(15)` in `_poll_pod_progress`
- Reduces SFTP connection noise (progress data is informational only)

## Verification
- Python syntax check: ✅ passes `ast.parse()`
- All changes in single file: `scripts/remote_validator.py`
- Net: +78 lines, -16 lines
