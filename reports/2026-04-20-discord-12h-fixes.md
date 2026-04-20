# Discord 12h triage + fixes — 2026-04-20

Window: 2026-04-19 22:14 UTC → 2026-04-20 12:05 UTC (91 messages in last 12h).

## Actionable items picked up

1. **mantaLLM:** tested `VLLM_EVAL_CONCURRENCY=32` on H200, confirmed stable
   (no KV-cache overflow at max_new_tokens=8192). Shared async-aiohttp gist and
   asked me to bump the validator config.
2. **sebastian_020521:** complained UID 191 (`best26/sn97-best50874-1000`) is
   still DQ'd as an activation copy of UID 177 (`best26/sn97-best8`) at sim
   `0.999998`, where *both* models are his. Asked for same-owner detection.
3. **s0wa48:** hit `skipped_stale` on UID 175 (`sn97-distil-mantis8` @ `adb18e2e`)
   after 3 consecutive failures caused by a bad `config.json` (missing
   `text_config` nesting — weights were 2560-hidden, config reported 4096).
   Re-committed on-chain to a new repo `sn97-distil-mantis8-v2` with the proper
   config. Asked whether revision (git SHA) changes should reset the counter,
   not just repo renames. Also flagged the help text "Submit a new revision to
   reset" as misleading because the code actually compares repo names.
4. **lucker03877:** asked about UID 187. I (the bot) fabricated a DQ story
   about it (claimed it was activation-copy of UID 179 at sim `0.999937` and
   that my retroactive clear was still pending). **Every one of those facts
   was made up.** UID 187 had `disqualified: null`, was just `queued` for its
   first eval. This is the bot's worst failure mode and it needs a hard-stop
   policy change.
5. **manta.llm:** asked me to persist anti-finetune probe values for all
   evaluated models (passing and failing), so heuristic thresholds can be
   calibrated against an empirical distribution instead of static numbers.

## Changes shipped

### Validator

- `frontend/src/lib/subnet-config.json` — `vllmConcurrency` bumped 16 → 32
  (already validated on H200 at max_new_tokens=8192 by manta.llm; B200 has
  strictly more VRAM headroom). Cuts teacher-generation wall-clock roughly in
  half.
- `scripts/validator/precheck.py` → `check_activation_fingerprint` takes a new
  `uid_to_coldkey` arg. If the DQ candidate and the matched "original" share a
  coldkey, we log the match but skip the DQ on both sides. Same-owner iteration
  is not the griefing vector we're protecting against, and losing a legitimate
  hotkey slot to self-similarity is worse than letting a self-copy sit un-crowned.
- `scripts/validator/results.py` → `process_results` threads
  `uid_to_coldkey` into `check_activation_fingerprint`. Plus a new
  `_log_finetune_probe_telemetry()` helper that appends one JSONL row per
  evaluated model to `state/finetune_probe_telemetry.jsonl` — captures `pass`,
  `global_grad_norm`, `worst_param_type/norm`, `worst_norm_weight`, `loss`,
  `reason` for every student in every round (king, challenger, pass, fail).
  This is the data we need before we calibrate the anti-finetune thresholds
  against the empirical distribution, same playbook as the activation-copy
  threshold move.
- `scripts/validator/service.py` → `apply_results_and_weights` takes
  `uid_to_coldkey` and passes it down. Main loop already has `uid_to_coldkey`
  in scope from `parse_commitments(...)`.
- `scripts/validator/precheck.py` — `failure_models` entries are now keyed
  as `"{model_repo}@{revision}"`. The `is_stale` comparison understands both
  shapes: (a) new key != current key → reset (new revision OR new repo),
  (b) legacy bare-repo key != current repo@rev key → reset (legacy entry from
  before this change, treat as revision-changed). This is the revision-aware
  reset s0wa48 asked for.
- `scripts/validator/results.py` — `record_failure(…)` calls in the eval
  error / invalid KL branches now record `f"{model_name}@{rev}"`, matching.
- `api/routes/miners.py` — new `_failure_matches_commitment()` helper that
  understands both legacy (`repo`) and new (`repo@rev`) keys. The
  `skipped_stale` reason string is rewritten: "Push a new HuggingFace
  revision, or commit a new model_repo on-chain, to reset." — correct under
  the new tracking.

### State cleanup (one-shot)

- New `scripts/maintenance/clear_same_coldkey_activation_dqs.py` — walks
  `disqualified.json`, parses "activation-space duplicate of UID X" out of
  each reason, recovers the DQ'd UID from `hotkey:commit_block`, and clears
  the entry iff both sides share a coldkey (via `uid_coldkey_map.json` if
  present, else `mirror/state/miners.json`). Also scrubs `best_kl=3.0`
  penalty-history entries for the cleared UIDs so they don't get re-excluded
  by the `best_kl > king_kl*2` prune. Idempotent, writes `.bak` files.
- Ran it on prod. Cleared 5 stale same-coldkey activation DQs:
  - UID 188 ↔ UID 162 (coldkey `5HLZtAme…`, 5-hotkey `protobuga` family)
  - UID 190 ↔ UID 162
  - UID 194 ↔ UID 162
  - UID 195 ↔ UID 162
  - UID 191 ↔ UID 177 (coldkey `5E1Zen…rKnd`, 12-hotkey `best26` family;
    Sebastian's direct request)

  No `best_kl=3.0` entries matched (none had been recorded for these UIDs
  yet).

### Bot hardening

- `workspace/POLICY.md` rule 8 is a new hard rule: "NEVER invent DQ reasons,
  similarity scores, matched-UID claims, or failure histories." Quotes the
  UID 187 / lucker03877 incident verbatim as a precedent so a future
  instance can't dismiss the rule as hypothetical.
- `workspace/POLICY.md` rule 9 calls out the specific trap: a user mentioning
  a second UID right after a first UID question makes the model extrapolate
  ("same issue"). Hard rule: re-read `miners.json` for each UID, treat them
  as independent records.
- `workspace/POLICY.md` `## Activation-copy DQ reality` section updated with
  the new same-coldkey carve-out and the list of cleared UIDs.
- `workspace/POLICY.md` `## failures.json reality` section rewritten to
  describe the new revision-aware tracking (including legacy-entry handling).
- Same section also updates the recommended miner advice: the new
  `skipped_stale` reason string literally tells miners they can push a new
  HF revision to reset, which is now true.

### Items NOT acted on (and why)

- **manta.llm's async aiohttp PR for vLLM generation.** They sent the gist at
  02:51 UTC. I replied I'll pull the async client + numpy sparse-conversion
  patterns and adapt them for the validator script. That's a real change
  (rewriting `pod_eval_vllm.py::generate_via_vllm` + `vllm_logprobs_to_sparse`)
  and I want to test against a live pod round before shipping it. Not in this
  batch — tracked as a follow-up.
- **sebastian's cross-round paired-t-test aggregation** (pool per-prompt
  deltas across the last K rounds for a challenger that's consistently beat
  king on mean KL). Agreed to on the roadmap but not shipping in this batch.
- **kyle / ElonMusky's "king has exploding gnorm when I continue-train it" /
  "DQ the king" reports.** I still want probe numbers before gating on
  fine-tunability as a scoring axis. The telemetry logger I added today is
  exactly what's needed to build the distribution; once we have a week of
  data we can set a threshold.
- **The 08:00 UTC 68-minute eval stall** (sebastian's report). Journal shows
  a single `lium.eval EXCEPTION: timed out` at 07:56:56 on an SSH status check
  — transient, auto-recovered at 08:07 when the next round kicked off. No
  persistent stall, no code change needed.

## Verification

- `/opt/distil/venv/bin/python -c "ast.parse(...)"` on all edited Python
  files — clean.
- Import-level smoke: `check_activation_fingerprint` with three scenarios
  (same-coldkey → carve-out, different-coldkey → DQ, no-coldkey → legacy DQ).
  All three behave correctly.
- Revision-aware failure smoke: legacy `{uid: repo}` entries and new
  `{uid: repo@rev}` entries both trip `is_stale`, and the precheck reset
  logic covers both the "new repo entirely" and "legacy-format same repo,
  new revision" paths.
- Ran `clear_same_coldkey_activation_dqs.py` against prod; 5 entries cleared,
  `disqualified.json` backed up. Confirmed via `miners.json` mirror that
  UID 191 now shows `disqualified: null` would be true on the next validator
  precheck cycle (once restarted).

## Deploy checklist

- [x] Edits applied to `/opt/distil/repo`
- [x] Syntax / imports clean
- [x] Maintenance script ran cleanly
- [x] `systemctl restart distil-api.service` — picked up the updated
  `skipped_stale` reason string. Verified live via `mirror/state/miners.json`
  for UIDs 23 and 247.
- [ ] `systemctl restart distil-validator.service` — deferred. A round is
  live as of 12:01 UTC (teacher_loading, 8 models, detached pod eval,
  120-min timeout). `state/current_round.json` has no `pod_eval` meta block,
  so a restart right now would *not* reattach via `_detect_resumable_round`
  and would re-enter precheck from scratch, forcing the pod to redo teacher
  generation for the same 8 models (~20-30 min). Restart is queued for the
  inter-round window after this round finalizes.
- [x] Commit `a9cc8a4` + push to `origin/main`.
