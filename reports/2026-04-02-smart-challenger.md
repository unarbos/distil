# Smart Challenger Selection

**Date:** 2026-04-02  
**Branch:** `improvements/validator-fixes-v2`  
**File:** `scripts/remote_validator.py`

## Problem

The periodic re-challenge system (`RE_CHALLENGE_INTERVAL=30`, `RE_CHALLENGE_TOP_N=3`) was wasteful:
1. Re-tested the same top 3 models every 30 epochs regardless of whether anything changed
2. 87 out of 91 scored models had NEVER been H2H tested due to the UID 18 bug monopolizing eval slots
3. The best models by global score never got a chance to face the king

## Solution

Replaced the periodic re-challenge with priority-based smart challenger selection:

### Priority System

1. **Priority 1: Best untested models** — Models with global scores that have NEVER been H2H tested against the CURRENT king. Sorted by global KL ascending (best first). Picks top 1 per epoch. This ensures every competitive model eventually faces the king.

2. **Priority 2: New submissions** — Models not yet evaluated at all (already handled by the existing challenger loop). Now logged explicitly.

3. **Priority 3: Stale re-tests** — Models whose last H2H was >50 epochs ago AND whose global score is within 2x of the king's KL. These might have been unlucky on a particular prompt set.

### State Tracking

- New file: `state/h2h_tested_against_king.json` — maps `{uid_str: {"king_uid", "epoch", "block", "kl", "model", "timestamp"}}`
- When the king changes, old records become irrelevant (they reference a different king_uid), so untested models naturally surface as Priority 1 candidates
- Updated after every H2H round with all challenger results

### Removed

- `RE_CHALLENGE_INTERVAL` constant (was 30)
- `RE_CHALLENGE_TOP_N` constant (was 3)
- All `rechallenge_history.json` references and tracking
- The entire periodic re-challenge block (~55 lines)

### Added

- `STALE_H2H_EPOCHS = 50` constant
- Smart challenger selection block with P1/P2/P3 logic (~85 lines)
- Post-round h2h_tested_against_king persistence (~25 lines)
- Logging: `[VALIDATOR] 🎯 SMART CHALLENGER: UID {uid} ({model}) selected — Priority {N}: {reason}`

## Key Behavior Change

- **Before:** Same 3 models re-tested every 30 epochs; 87/91 models never got H2H
- **After:** Every model with a valid global score will eventually get an H2H test against the current king, prioritized by how good their global score is. When a new king is crowned, the cycle resets — all models become "untested vs this king."

## Files Changed

- `scripts/remote_validator.py`: +108/-53 lines
