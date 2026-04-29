# Anti-griefing tiebreaker + king canary streak — 2026-04-28 (late)

A short follow-up to `2026-04-28-goodhart-deep-pass.md` and
`2026-04-28-discord-12h-fixes.md`, addressing two issues surfaced by
crypsick on the Discord channel.

## What happened

### crypsick: "truke02/golden_v1 was uploaded 34 mins ago this seems like a bug"

The bot had reported earlier in the session that
`UID 228 (allforone111/nimbus-V48)` was DQ'd as a copy of
`UID 241 (truke02/golden_v1)` because UID 241 "committed first" on
chain.

Investigation:

| field | UID 228 (allforone111) | UID 241 (truke02) |
|---|---|---|
| HF created_at | 2026-04-28 11:49:31 UTC | **2026-04-28 19:07:14 UTC** |
| HF last_modified | 2026-04-28 11:50:17 UTC | 2026-04-28 19:17:49 UTC |
| chain commit_block | 8067271 (~13:46 UTC today) | 8034134 (~5 days ago) |

The miner reserved a UID slot 5 days ago, committed the bare model
name `truke02/golden_v1` on chain pointing to a non-existent HF
repo, watched the chain for the next king, and today (after
allforone111 was crowned) copied the king's weights into their
pre-reserved repo. Their on-chain block then "predates" the
legitimate commit, fooling the SHA256/content/activation copy
detectors into DQ-ing the victim.

The previous anti-griefing guard (commits `beafc1f`, `24d94ba`)
only kicked in when the copy was evaluated and the original was
not. Once both miners get evaluated, the guard goes silent and the
chain order wins again — so a determined attacker who waits long
enough will always win the order.

### The three-of-seven canary regression

Audit run on `state/benchmarks/uid_*.json`: 6/7 of the recent kings
are >5pp below Qwen3.5-4B baseline on the held-out canary axes
(gsm8k+humaneval+bbh+ifeval), with -8pp to -12pp gaps. The dethrone
gates can't see this — they protect the king's
``composite.worst``. So the king persists indefinitely once
crowned.

## What we shipped

### 1. HF upload time as authoritative tiebreaker — `5b61851`

* `eval/hf_upload_meta.py` (new, 181 lines)
  - `get_first_upload_epoch(repo, revision)` hits HfApi
    `list_repo_commits` and falls back to `created_at` /
    `last_modified`
  - cached on disk in `state/hf_upload_meta.json` with a 24h TTL
    so precheck does not hammer HF every round

* `scripts/validator/precheck.py`
  - `_hf_upload_griefing_swap(...)` resolves the (copy, original)
    pair using HF time as authoritative tiebreaker
  - SHA256-hash and content-hash branches now consult the swap
    *before* any chain-order branch fires; if HF shows the
    would-be original was uploaded *after* the would-be copy, the
    DQ direction is reversed before the legacy logic runs
  - `check_activation_fingerprint` accepts `revision`,
    `uid_to_model`, `uid_to_revision` and applies the same
    reversal after the evaluated-state guard

* `scripts/validator/results.py`
  - passes the round's per-UID `model` + `revision` maps so the
    activation-fingerprint guard can resolve the right HF repos

End-to-end smoke test on the real truke02 vs allforone111 case:

```
scenario 1 (chain says truke02 earlier, HF says truke02 later)
  -> swap triggered, UID 241 flagged as the copy, UID 228 protected
scenario 2 (chain says allforone111 earlier, HF says allforone111 earlier)
  -> no swap (chain order matches HF order, legitimate)
```

Also forward-ports the existing UID 241 griefing DQ from block
8034134 to its current commit_block 8069266 so the attacker cannot
escape the existing DQ by re-committing before the next precheck
pass, and bumps `state.model_hashes['241_block']` accordingly.

### 2. King canary regression streak + waive — `b74c3c2`

* `eval/state.py`
  - new `KING_CANARY_FILE = "king_canary_streak.json"`
  - `state.king_canary_streak` (dict keyed by stringified king UID)

* `scripts/validator/composite.py`
  - new tunables `KING_CANARY_GATE`, `KING_CANARY_MARGIN=0.05`,
    `KING_CANARY_MIN_STREAK=2`,
    `KING_CANARY_AXES=gsm8k,humaneval,bbh,ifeval`
  - `_compute_king_canary_regression(uid, state_dir)` loads the
    king's `state/benchmarks/uid_<UID>.json` + the Qwen 4B
    baseline, computes mean-across-axes for each, returns
    `at_risk` if king mean < base mean - margin

* `scripts/validator/state_manager.py`
  - each canonical round, runs the regression check; when at-risk,
    increments `state.king_canary_streak[king_uid]`; resets when
    the king changes or is no longer at-risk; persists to
    `king_canary_streak.json`

* `scripts/validator/results.py`
  - `_king_regression_floor_waived` now returns True on **either**
    a `king_regression_streak >= KING_REGRESSION_MIN_STREAK` (3
    canonical rounds, internal composite at-risk) or a
    `king_canary_streak >= KING_CANARY_MIN_STREAK` (2 canonical
    rounds, held-out canary at-risk). Once either fires, the
    composite-floor veto is waived and a strong challenger can
    dethrone even if it slips on a saturated relative axis.

Verified on current state: gate fires correctly for UIDs 18, 107,
118, 123, 156, 217 (gap >= 5pp) and abstains for UID 149 (3.7pp).

## Why this is not double-counting

The composite-floor veto is **challenger-side**: it stops a narrow
KL specialist from taking the crown. The two streaks are
**king-side** at-risk signals that say "the king itself has been
weak across canonical rounds; the floor is now asymmetric
protection." Waiving the floor when both signals fire still
requires the challenger to pass KL significance + Pareto + per-axis
rules; it just removes the no-regression-on-composite check that
was protecting an under-performing king.

## Why HF time + chain block both, not just one

* HF time alone misses the case where a miner uploads a fresh
  legitimate model and the chain commit lags. Chain block is the
  miner's commitment to a specific weight set.
* Chain block alone misses the griefing pattern (this incident).

Using **HF time as the tiebreaker only when both lookups succeed
and disagree** preserves the chain-block primary while patching the
griefing hole. If HF is unreachable, we fall back to chain order
(no behaviour change).

## Open follow-up

* The 24h cache TTL on `hf_upload_meta.json` means a slow grief
  attempt (attacker uploads days after victim) is caught instantly
  but a same-day grief race within the cache window may still
  succeed if the cache was warmed against an empty repo. Mitigation
  is the existing per-block precheck pass + 24h TTL together: the
  attacker would have to win a same-block race AND beat the cache
  warm.
* When the king canary streak triggers and a dethrone happens, we
  should re-eval the **previous** king on the new canonical round
  to confirm the regression wasn't a benchmark-runner outage. The
  baseline-floor veto already does this for composite; canary will
  inherit it on the next pass.

## Validator state at end of session

```
[validator] Git: 5b61851 fix: HF upload time as authoritative tiebreaker for copy detection
```

Active PRs / pending tasks:

* perf-improve (in progress) — continue improving model performance
  across evalscope benchmarks without using evalscope as the eval.
  v29 procedural rebalance (math/code/reasoning/ifeval) is in
  place; next king benchmark will tell us whether transfer
  improved.
