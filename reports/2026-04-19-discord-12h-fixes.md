# Discord 12h triage and fixes (2026-04-19)

## Summary

Reviewed 12 hours of SN97 public-channel discussion (174 messages,
14 distinct miners) and shipped fixes for the issues miners correctly
flagged. Two of the items were already fixed in earlier rounds today
(`ca5d4ed` think-probe disable, `2d14f6f` contender sort); the rest are
addressed by this commit.

## What miners flagged and what we did

### apple_2357 — activation copy threshold too strict (FIXED)

> "I finetuned top model for 24+ hours but still can't pass activation
> fingerprint check."

Bumped `activationCopyThreshold` from 0.9999 → 0.99999. Calibrated against
the 27 stored fingerprints in `state/activation_fingerprints.json`:

- 13 pairs sit between 0.9999 and 0.99999 — these are legitimate fine-tunes
  (e.g. `best26/sn97-best6` vs `sn97-best7` at 0.9999875, `allforone111`
  V1/V2 sibling models at 0.9999870). They were getting false-positive
  copy-DQ'd at the old threshold.
- Only 2 pairs sit at >= 0.99999, both strongly suspected copies:
  - `best26/sn97-best6` vs `gtensorapp/slim-ivory-2577` at 1.0000000
    (different owners, identical activations).
  - `best26/sn97-best4` vs `sniper999/nova-distilled-V9` at 0.9999998.

Apple's empirical claim ("copies can't pass 0.99999, real fine-tunes can")
holds against the data. Existing DQ entries don't auto-clear (they're
keyed to `hotkey:commit_block`); only future evaluations use the new
threshold.

### apple_2357 — anti-spam attack vector (FIXED)

> "Let's image that prev king KL is 0.1 and dethroned models KL are 0.9,
> 0.901, 0.902. But 0.9, 0.901 is just copy of 0.902 model. Current
> mechanism selects 0.9 as king."

Implemented a pairwise-paired-t-test tiebreaker in
`scripts/validator/results.py`:

1. Collect EVERY challenger that passes the king t-test (instead of the
   old "lowest-KL passing" pick).
2. Pairwise paired-t-test all dethroners against each other (two-sided,
   p > `PAIRED_TEST_ALPHA` ⇒ statistically tied).
3. Build equivalence classes from "tied" edges. Find the cluster
   containing the global lowest-KL dethroner.
4. Within that cluster: earliest `commit_block` wins; ties broken by KL.
   A genuinely-better outlier sits in its own cluster and wins on merit.

leeroyjkin's pushback was incorporated: the rule does NOT pick an earlier
model with a worse score. It only invokes commit_block when the
challengers can't be statistically distinguished.

Validated end-to-end with 5 synthetic scenarios in `/tmp/test_tiebreak.py`
including Apple's exact attack (3 noise-copies of an early-committer where
the late copies have lowest mean KL by noise). Old behaviour → late copy
wins. New behaviour → original wins, late copies deferred to next round.

Comprehensive `[tiebreak]` log lines on every multi-dethrone round so any
challenged decision can be reconstructed from the journal.

### itorgov — stale FAQ "greedy/temp=0" claim (FIXED)

> "The FAQ says eval is greedy/temp=0 at docs/MINER_FAQ.md (line 86).
> Production code currently does not match that when --block-seed is
> passed: vLLM uses temperature=0.7, top_p=0.9, and seed=block_seed + idx
> at scripts/pod_eval_vllm.py (line 2100)."

Updated `docs/MINER_FAQ.md` to match production:

```
Eval runs vLLM with temperature=0.7, top_p=0.9 and a per-prompt seed
derived from the round's block hash (seed = block_seed + prompt_idx).
Generation is deterministic per round but varies between rounds. Greedy
(temp=0) only applies to local dev runs that don't pass --block-seed.
```

### itorgov — misleading "Integrity check failed" advice (FIXED)

> Quoting itorgov: "do I always need to register a new hotkey?"

The previous FAQ entry conflated permanent DQs (copy / anti-finetune /
arch — those need a new hotkey) with transient integrity DQs (deleted /
private repo — those clear themselves once the repo is reachable again).
Rewrote the entry to distinguish them.

### lucker03877 — README "pushing new commits = DQ" confusion (NO CHANGE)

The README says pushing new revisions to an already-evaluated repo
triggers an integrity DQ. This is correct: the validator pins the git SHA
at evaluation time and refuses to use a different revision later.

The Arbos bot's earlier reply to apple_2357 that said "you can update
weights in the same HF repo" was wrong; the bot publicly self-corrected
in the channel. The README itself doesn't need a change.

## Items intentionally deferred

- **Per-prompt KL correlation copy detection** (leeroyjkin's idea, bot
  acknowledged). Better long-term defense against noise-injected copies
  than the activation fingerprint, but needs calibration against known
  merges (e.g. SLERP/TIES) before it can ship without false-positiving
  legitimate model merges. Adding to backlog.
- **Activation fingerprint at precheck** (apple_2357's queue-flooding
  concern). Currently runs on the eval pod, so copies still consume an
  eval slot before being caught. Moving the fingerprint to precheck
  requires either CPU-side forward passes (teacher is 70 GB) or a
  dedicated lightweight precheck pod. Out of scope for this round.
- **Revision-aware DQ clearing** (itorgov + lucker03877). Pushing new
  weights to a copy-DQ'd repo currently can't clear the DQ. The right
  fix is to invalidate `hotkey:commit_block` DQs when the HF revision
  changes AND the new weights actually differ from the DQ'd ones.
  Larger refactor of the DQ key scheme; deferred.
- **Anti-finetune regression** (Discord bot claim). Verified false in
  the previous round — bot was hallucinating against stale data; live
  state shows the probe is working as designed.

## Files changed

- `frontend/src/lib/subnet-config.json` — threshold 0.9999 → 0.99999
- `eval/runtime.py` — derives from config (no code change needed)
- `check_model.py` — local test threshold mirrored
- `README.md` — copy-detection section threshold updated
- `docs/MINER_FAQ.md` — temperature claim + integrity-DQ entry
- `scripts/validator/results.py` — pairwise tiebreaker + helper

## Deployment

Ship with the next safe restart window (between rounds). The threshold
bump and FAQ updates are inert if not deployed. The tiebreaker activates
only when 2+ challengers pass the king t-test in the same round, which
happens roughly every 2-3 rounds at current activity levels.

## Follow-ups after the initial deploy (22:28-22:45 UTC)

Discovered three gaps during the deploy rollout — all fixed in `e2c534a`:

1. **Stale activation-copy DQs don't auto-clear with the threshold bump**
   (sebastian_020521's complaint about UID 191 and UID 193). Bumping
   `activationCopyThreshold` in config alone does nothing for already-DQ'd
   miners — precheck short-circuits on the cached `hotkey:commit_block`
   entry in `disqualified.json` before any fresh comparison runs. Added
   `scripts/maintenance/clear_activation_dqs_below_new_threshold.py`
   which removes DQ entries whose recorded cosine sim falls in
   `[0.9999, 0.99999)`. Ran against prod state: cleared 21 DQs, unbanned
   5 models from `permanently_bad_models.json`.

2. **Penalty history in `model_score_history.json` also needed reset.**
   Even after un-DQing, `select_challengers` pruned these UIDs via the
   `best_ever > king_kl*2` rule because their history was stamped with
   `best_kl=3.0` (the sentinel assigned when the DQ was applied). Added
   `scripts/maintenance/reset_penalty_history_for_cleared_uids.py`; ran
   against prod and reset 16 penalty entries.

3. **`cap_challengers` silently dropped top-4 contenders.** The cap
   sorts by `state.scores.get(uid, 999)`; unscored challengers all tie
   on the 999 default and top-4 contenders (added last in
   `add_top5_contenders`) got truncated first by dict-insertion order.
   UID 193 (`best26/sn97-best50622-2550`) hit this on the first round
   after un-DQing — the existing `assert_top_contenders_present`
   regression check caught it loudly but didn't fix it. Modified
   `cap_challengers` to pin top-4 contenders before the sort; smoke-
   tested with 3 scenarios (cap exceeded with top-4, cap exceeded
   without top-4, under cap).

Validator was restarted four times during rollout: (a) pick up the
1c61941 commit, (b) reload in-memory `state.dq_reasons` after clearing
21 entries, (c) reload `model_score_history` after resetting 16 penalty
entries, (d) pick up the cap fix. After the last restart the H2H round
correctly includes UID 193 via the newly-protected top-4 path, and the
15 other cleared UIDs are visible in the precheck-valid pool for
subsequent rounds.

Files added:

- `scripts/maintenance/clear_activation_dqs_below_new_threshold.py`
- `scripts/maintenance/reset_penalty_history_for_cleared_uids.py`

Files changed:

- `scripts/validator/challengers.py` — `cap_challengers` protects top-4
