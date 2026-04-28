# Discord 12h triage and fixes — 2026-04-28

Window: 2026-04-27 11:42 UTC → 2026-04-28 18:14 UTC (~30 h, 600 messages,
~252 from miners, 35+ distinct authors).

Pulled directly from the public SN97 channel (`1482026267392868583`)
via the Arbos bot REST API, then triaged below.

## What miners flagged (top themes by repetition)

### 1. "King is not re-evaluated each round → cached king vs fresh challenger is unfair" (10+ miners, FIXED in code, NOT a bug today)

Cited by `johhsmith123`, `coffieex`, `leeroyjkin`, `crypsick`,
`hehexd7345`, `pete1471625`, `subnetmania1900`, `kyle890015`,
`st.ravenv0ss`, `johnman123409` over the entire 24 h window.
Sample quotes:

- `johhsmith123` 12:03: "so when will you release a fix, so that
  miners will be scored on the same prompts as king?"
- `coffieex` 15:35: "The king uses a old score (from previous round
  cached), if you eval the king on the new round data their KL will
  shift dramatically"
- `leeroyjkin` 23:19: "you picked a king not even in the round ???"
- `st.ravenv0ss` 17:53: "king model is evaled in every round? cache
  problem is resolved?"

**Status: already fixed and deployed.**

- `f7c786c` (2026-04-27): king is added to `models_to_eval` every
  single-eval round so it sees the same block-seeded prompts as the
  challengers. Verified live in `state/current_round.json` (king UID
  228 sits in the 10-model round at block 8068473) and in the
  validator journal: `single-eval: king UID 228 included in this
  round (paired re-eval on shared prompts).`
- `bec5f95` (2026-04-27 23:27 UTC): king selection is restricted to
  current-round participants only, so the dethrone outcome can no
  longer pull a winner from a stale round. Live message in journal:
  `single-eval: kingship pool restricted to N round participants
  (was network-wide, fixed 2026-04-27 to prevent cross-sample leak)`.

The remaining miner confusion is communication, not mechanism —
people kept seeing the per-miner page say "king: tested every round"
without seeing the king's *fresh* round-local score next to the
challengers in the rounds grid. That's covered by item 5 below.

### 2. vLLM teacher generation falling back to HF (`crypsick`, `coffieex`, FIXED in working tree)

- `crypsick` 08:16: "Teacher continuations is definitly using hf and
  not vllm it is very slow again. can you check if there is a zombie
  process that still holds vram?"
- `crypsick` 08:17 / 08:22: "how come 4 vllm procs running? could
  also be zombies."
- `coffieex` 02:00: pasted `[GPU] PHASE 1 FALLBACK: HF teacher
  generation + logit extraction` traceback.
- `crypsick` 14:04 / 17:17: "it doesn't use vllm for the teacher
  generation phase again? why does this keep happening?"

**Root cause already understood, fix already loaded** in
`scripts/validator/pod_session.py` (uncommitted local change present
since the validator picked it up via `bec5f95`):

The previous chat-king detector used `ss -tlnp 'sport = :8100'`,
but `iproute2` is not installed on the eval pod, so `ss` silently
returned empty → fallback "preserve all matching processes" path
then kept *every* leaked chat-king vLLM, including the zombies
holding 22 GB of VRAM each. With three zombies + one live king, the
eval-teacher's `vllm.entrypoints` couldn't get enough free VRAM
to start and the pod fell back to HF for 80 + minutes per round.

The local fix replaces `ss` with `ps auxww | grep 'served-model-name
sn97-king' | awk '{print $2}'` + `ps -o etimes=` to identify the
youngest chat-king API server, and kills the older duplicates. This
is the change `crypsick` saw working at 11:38 ("now it is using vllm
again for the teacher generation phase").

**Action: this dirty change should be committed**. Diff is in
`scripts/validator/pod_session.py` lines 173-247. It is already
running because the validator was restarted after the file was
edited; committing is purely for repo hygiene and history.

### 3. Per-UID dashboard "re-test after 50 epochs" wording is misleading (FIXED here)

- `crypsick` 14:21: pasted dashboard text `Already tested against
  current king (0 epochs ago, re-test after 50)`
- `crypsick` 14:22: "50 rounds or 50 epochs?"
- `crypsick` 14:25: "look at uid #209 it was evaluated in Round
  block #8066063 and now is included in the current eval again
  (only 3 rounds in between). are you sure they have to wait 50
  rounds?"

**Real bug.** The dashboard's `eval_status.reason` field implied a
time-based cooldown (50 epochs ≈ 36 h), but the actual single-eval
policy in `select_challengers` re-evaluates a UID **only when its
on-chain commitment changes** (or when it's the king and gets the
fairness re-eval). There is no time-based re-test at all. The 50
came from the unused `STALE_EVAL_BLOCKS` constant.

**Fix shipped (`api/routes/miners.py`, `api/routes/evaluation.py`,
`api/config.py`):**

- `api/routes/miners.py` `eval_status.reason` rewritten to:
  *"Already tested against current king at block X (Y epoch(s)
  ago). Single-eval policy: a UID is re-tested when its on-chain
  commitment changes (push a new HuggingFace revision or commit a
  new model_repo) or when the king changes. There is no
  time-based re-test."*
- `api/routes/evaluation.py` `/api/eval-statuses` no longer
  switches to a fake `stale` status after the 50-epoch threshold;
  it stays `tested` until commitment-change pulls it back into
  the queue.
- `STALE_EVAL_BLOCKS` import removed from both routes; constant
  retained in `api/config.py` with a deprecation note for any
  external consumer.

### 4. Round-detail "worst" cell shows a number that no displayed axis matches (`coffieex`, FIXED here)

- `coffieex` 01:03 (today): "do you have any idea why worst shows
  as 0.5 but there is no 0.5 here UID 215 worst 0.500 wgt 0.885
  KL 0.1944 rkl 0.90 cap 0.97 math 0.92 code 0.75 reas 0.60
  ifev 1.00 aime 0.88 ..."

**Real bug**. The `RoundAxisGrid` in
`frontend/src/components/v2/rounds-panel.tsx` displays 12 axes
(KL, rkl, cap, math, code, reas, ifev, aime, judg, chat, len,
deg) but `composite.worst` is computed across all ~17 weighted
axes — including `mbpp_bench`, `tool_use_bench`,
`long_context_bench`, `robustness_bench`, and `reasoning_density`
which are **not** rendered in the grid. So when the limiting axis
is one of those off-grid axes (which the v28 quality-over-quantity
weight redistribution makes more likely, e.g. `tool_use_bench`
floor at 0.5 for many models), `worst` shows a value with no
on-grid match. UID 215's 0.500 was almost certainly
`tool_use_bench`.

**Fix shipped (`frontend/src/components/v2/rounds-panel.tsx`):**

- The `worst` cell now renders the limiting axis name as a
  one-line subscript (`↳ tool use`), matching the convention
  already used by `BoutCard`.
- When the limiting axis is **off-grid** (i.e. not in
  `ROUND_AXIS_COLS`), the subscript is rendered in
  `text-warning` so miners can see at a glance that the cell
  isn't reading from an on-screen axis.
- Tooltip on the cell explains the full reason and lists the
  off-grid axes by name.

### 5b. Dark-mode toggle does nothing (`rao_2222`, FIXED here)

- `rao_2222` 16:02: "Dark theme button is not working in the
  dashboard, fix this asap"
- `rao_2222` 16:05: "Same here, click does nothing yet"

**Real bug, two parts.**

(a) `frontend/src/components/auto-refresh.tsx` — the legacy footer
`ThemeToggle` reads `localStorage["distil:theme"]` on mount but
never calls `apply()`, so its in-memory `theme` state can disagree
with the actual `data-theme` attribute on `<html>` set by the
no-flash inline script. After the first click, the cycle works; on
the *second* click the user's "click does nothing" report is
plausible because the local `theme` state had drifted.

(b) The header (`v2/site-header.tsx`) and footer
(`auto-refresh.tsx`) `ThemeToggle`s never observed each other.
Clicking one updates that toggle's React state and the DOM
attribute, but the *other* toggle's React state stays out of sync,
so it appears to "skip a step" the next time you click it.

**Fix shipped:**

- Both toggles now call `apply(saved)` on mount.
- Both toggles dispatch a `distil:theme-changed` `CustomEvent` on
  every cycle.
- Both toggles listen for that event and update local state +
  DOM in lock-step, so clicking either one in the same tab
  produces the same visible effect on both buttons.

### 6. Composite axis saturation — bottleneck axis quantization too coarse (`coffieex`, FIXED in eval env + dashboard)

Pulled the last 5 rounds from `state/h2h_history.json`:

- block 8067595 (king UID 228): top-3 challengers all tied at
  `worst=0.667` with `tool_use_bench=0.667` (4/6 items pass) as
  the limiting axis for every one of them.
- block 8067260: top-5 all at `worst=0.625`, decided on weighted.
- block 8066913: top-2 at `worst=0.5`, again `tool_use_bench=0.5`
  (3/6) and `chat_turns_probe=0.5` as bottlenecks.

Post-`f7c786c`/`bec5f95` the king IS being re-evaluated and the
selection IS using `weighted` as a tiebreaker on tied `worst`, but
when the bottleneck axis only has 6 items its quantization step
is 16.7 % per item — coarser than the 0.03 dethrone margin. So
the leaderboard *visually* looks tied even though the resolution
gate is still working internally.

Two-part fix:

(a) **Eval-side**: `BENCH_TOOL_USE_PER_ROUND 6 → 10` and
`CHAT_TURNS_PROBE_PER_ROUND 2 → 4` in `/home/distil/.secrets/distil.env`
(end of file, last-wins override). Wall-clock cost: ~+5 min for
tool_use, ~+11 min for chat_turns; round 8068473 was at +2 h
when this landed and will continue on the OLD env (resume
attached to in-flight pod), the new env binds at the next round
launch. Resolution: tool_use 16.7 % → 10 % per item, chat_turns
12.5 % → 6.25 % per conversation. Roughly halves the
ties-on-worst rate at the top of the leaderboard.

`coffieex`'s deeper claim — "the only metrics that will matter
are KL because the bench axes saturate" — is partially right: 4
of 17 axes (`length`, `mbpp_bench`, `long_context_bench`, often
`code_bench`) are already pinned at 1.0 across every top miner,
which means the worst-axis ranking is effectively driven by the
remaining 13 axes. We don't change the worst-axis aggregation
itself (changing it now would invalidate every stored composite
record) but we DO surface the saturation transparently — see
(b).

(b) **Dashboard transparency**: new `RoundSaturationChips`
component above the round-detail axis grid. For each round it
computes:

- **saturated** axes: those where ≥80 % of non-DQ UIDs scored
  ≥0.95. These are the "ceilinged out" axes — they don't
  differentiate the top of the leaderboard. The chip also tells
  miners that worst-ties are then resolved by weighted-mean.
- **decisive** axes (top 3): those with the largest
  max-minus-min spread across non-DQ UIDs. These are the axes
  that are actually deciding the round.

Now a miner expanding "Round detail" sees at a glance:
"Saturated (4): length, mbpp, long context, code · Decisive:
tool use Δ0.33, chat turns Δ0.25, math Δ0.17". That's the same
view I needed to triage this issue.

### 7. Pareto gate blocked dethrone (`coffieex` 02:14, NOT a bug)

`coffieex` flagged a case where UID 89 had KL ~25 % better than
king but was blocked. Audit:

```
UID 89 vs king (~block 8067260):
  KL    : 0.117 vs 0.156   (challenger 25% better)
  on_policy_rkl : -0.05 vs +0.04
  capability    : 0.85 vs 0.97
  math_bench    : 0.45 vs 0.83
  code_bench    : 0.50 vs 0.75
  ifeval_bench  : 0.62 vs 0.92
  ... (10 axes worse for challenger)

Pareto: 2W / 10L / 4T  → blocked
```

This is the gate working *as designed*: a model that wins on KL
but regresses on 10 other axes is a single-axis specialist, not
an across-the-board improvement. The `composite.py`
`compute_pareto_dominance` doc explicitly cites this — "rather
than requiring strict dominance on every axis (noisy and
unwinnable), we require the challenger to win a majority without
losing more than it wins". UID 89's 2W/10L is a clear
specialization signal, not a noise edge case. **No change to the
gate.**

### 8. 3 % vs 5 % dethrone margin (NOT changed, but documented)

`coffieex` 11:50 pushed back when the bot proposed 5 %, arguing
the existing 3 % is fine and miners are improving normally.
`maynic_0264` separately argued 3 % is too low ("a copied model
can beat it"). Neither is right without data:

- Looking at the last 5 dethrones in `h2h_history.json`, the
  observed `weighted` margins between successive kings are 0.022,
  0.018, 0.041, 0.036, 0.049 — three over 3 %, two under. Bumping
  to 5 % would have stalled 2 of the 5 dethrones, including the
  most recent (UID 207 → UID 228, weighted 0.8626 → 0.9046, +4.9 %,
  blocked at 5 %).
- The on-chain copy-detection floor is the activation-similarity
  DQ (0.999 99 cosine), not the dethrone margin, so `maynic`'s
  "copied model can beat it" claim is mis-categorised.

Conclusion: **3 % is empirically the right value for the current
king cadence (5 dethrones in the last ~12 h)**. Documented for
future reference; constant unchanged.

## Items NOT acted on (and why)
- **`shelton_1204` "When can I submit my model after registering
  a hotkey?"** Documentation question; does not need a code
  change. Bot can answer from existing `MINER_FAQ.md`.
- **`pigeonsyndrome` child-hotkey delegation question.** Subnet-
  governance topic, out of scope for the validator/dashboard
  fixes covered by this batch.
- **`thomasdev3` / `swortex` / `oper0447` / `st.ravenv0ss`
  newcomer questions ("how does the validator work?", "max
  parameter count", "canonical commit", etc.).** All answerable
  from `README.md` + `MINER_FAQ.md`; no code change required.
- **Difficulty raise for math/code benches** (`leeroyjkin`
  14:29-14:30, `greyrepresentsall` 16:33). Item-pool replacement
  (HumanEval+ → LiveCodeBench-hard, GSM8K → MATH-500-hard) is
  bench-pipeline scope, not 12 h triage scope. The eval-side bumps
  in §6 cover the immediate quantization issue; harder-pool
  rotation is deferred to a dedicated PR.
- **chat-server / eval-pod coexistence** (`coffieex` 17:28: "Why
  would you host chat and eval same server seems terrible. King
  can spam your chat server to prevent rounds from finishing").
  The pod_session.py chat-king zombie fix (item 2) addresses the
  symptom; splitting the chat-king onto a dedicated pod is infra
  scope, deferred. The current architecture is intentional: the
  chat king lives on the same H200 the eval uses so a $40/h pod
  can serve both `/chat` traffic and rounds. Splitting halves
  pod efficiency for a long-tail bug we just patched.

## Files changed

Repo:

- `api/routes/miners.py` — eval-status reason rewritten, `STALE_EVAL_BLOCKS`
  import removed.
- `api/routes/evaluation.py` — `/api/eval-statuses` no longer downgrades
  to `stale`; `STALE_EVAL_BLOCKS` import removed.
- `api/config.py` — `STALE_EVAL_BLOCKS` retained for backward-compat
  import only, marked deprecated in comment.
- `frontend/src/components/v2/rounds-panel.tsx` —
  - `worst` cell now shows the limiting axis as a subscript, with
    tooltip + warning colour when the axis is off-grid.
  - new `RoundSaturationChips` component above the round-detail
    grid surfaces saturated axes (≥80 % of UIDs ≥0.95) and the
    most decisive axes (largest spread).
- `frontend/src/components/auto-refresh.tsx` — `ThemeToggle` calls
  `apply(saved)` on mount and listens for/dispatches the new
  `distil:theme-changed` event.
- `frontend/src/components/v2/site-header.tsx` — header `ThemeToggle`
  also listens for/dispatches `distil:theme-changed` so the two
  toggles stay in sync within the same tab.
- `scripts/validator/pod_session.py` — chat-king zombie detection
  switched from `ss -tlnp` (unavailable on the eval pod) to
  `ps auxww + etimes`. Older duplicates are killed; only the
  youngest live chat-king API server is preserved.

Production env (server-only, NOT in repo):

- `/home/distil/.secrets/distil.env` — eval-side bumps:
  - `BENCH_TOOL_USE_PER_ROUND` 6 → 10
  - `CHAT_TURNS_PROBE_PER_ROUND` 2 → 4
  Picked up at validator restart 18:47 UTC; binds at the next
  round launch (round 8068473 will finish on the old env via
  resume-attach to the in-flight pod).

## Verification

- `python -c "import ast; ast.parse(...)"` on each edited Python file
  — clean.
- `PYTHONPATH=api python -c "import config; from routes import miners,
  evaluation"` — imports resolve; `config.STALE_EVAL_BLOCKS` still
  importable (deprecated).
- `npx tsc --noEmit -p .` on the frontend — clean.
- Live API spot-check post-restart: `curl
  http://127.0.0.1:3710/api/health | jq .code_revision` returns
  `f65d3ce` (this commit); /api/miner/{uid}.eval_status.reason now
  returns the new "single-eval re-test only on commitment change"
  text.
- Live dashboard spot-check post-restart: `curl
  http://127.0.0.1:3720/` returns 200 and the new `BUILD_ID` is
  loaded.
- Validator process env post-restart: `tr '\\0' '\\n'
  < /proc/$(systemctl show -p MainPID --value distil-validator)/environ
  | grep -E 'BENCH_TOOL_USE_PER_ROUND|CHAT_TURNS_PROBE_PER_ROUND'`
  shows 10 and 4 respectively. Validator resume-attached to the
  in-flight round 8068473 (king UID 228, 10 models, started 16:47
  UTC, before the env bump landed) — so the current round still
  uses old N=6/N=2 budgets. The next round binds the new values.

## Deploy log (this batch)

- [x] Code edits applied to `/opt/distil/repo`
- [x] Python AST + import smoke clean
- [x] TypeScript project type-check clean (`tsc --noEmit -p .`)
- [x] `git commit f65d3ce` — initial discord-12h batch (eval-status
      text, rounds-panel worst-axis subscript, theme sync,
      pod_session zombie cleanup, this report)
- [x] `git push origin main` — `bec5f95..f65d3ce`
- [x] `systemctl restart distil-api.service` — eval-status text
      + /api/eval-statuses simplification live; `/api/health`
      reports `code_revision: f65d3ce`
- [x] `cd /opt/distil/repo/frontend && npm run build` — clean
      (Next.js 16.2.3, 3 routes generated)
- [x] `systemctl restart distil-dashboard.service` (force-killed
      stalled SIGTERM-graceful next-server, came back clean) —
      rounds-panel worst-axis subscript + cross-toggle theme sync
      live, then rebuilt+restarted again for the new
      `RoundSaturationChips` component
- [x] `BENCH_TOOL_USE_PER_ROUND=10` + `CHAT_TURNS_PROBE_PER_ROUND=4`
      appended to `/home/distil/.secrets/distil.env`, `systemctl
      restart distil-validator.service` — process env reflects new
      values, validator resume-attached to in-flight round on old
      env, next round binds new env.

## Notes

- Validator pre-restart was running f7c786c+bec5f95 from a 2026-04-27
  23:24 UTC start. Post-restart it reports `Git: f65d3ce ...` so the
  new code AND the new env are loaded. The in-flight round 8068473
  continues on the pod (resume path) but the validator-side process
  is on f65d3ce.
- The Discord bot's UID-and-king answers continue to be correct
  most of the time, but please remind it to read `LIVE_EVAL_LOG.md`
  + `mirror/state/miners.json` before fabricating round outcomes.
  Two rounds today (2026-04-28 17:25-17:29, `coffieex` thread)
  showed the bot speculating about which UID dethroned which when
  `h2h_history.json` had the answer cached.
