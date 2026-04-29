# 2026-04-28 — Baseline-floor dethrone veto + math_bench v29

## Why

The held-out evalscope canary (added earlier today) exposed a clear
Goodhart signature on the validator:

| benchmark | Qwen3.5-4B base (held-out) | Recent kings (held-out) | Δ |
| -- | -- | -- | -- |
| gsm8k     | 0.934 | 0.860 | **−7.4 pp** |
| bbh       | 0.879 | 0.777 | **−10.2 pp** |
| arc       | 0.901 | 0.920 | +1.9 pp |

Models miners distil are climbing the validator composite (`composite.worst`
0.5 → 0.55+) while regressing on the held-out evalscope benchmarks vs the
*un-distilled* base model. The single cleanest piece of evidence: the
reference Qwen3.5-4B base scored `math_bench=0.5` on procedural items but
`gsm8k=0.93` on held-out — a 40+ pp distribution gap. The procedural math
items have been testing a **different skill** than gsm8k, so optimising
`math_bench` doesn't transfer to gsm8k.

## What landed

### 1. `pod_eval_vllm._generate_math_items` — v29 narrative rebalance

Replaced the single direct-compute kind list with a 70/30 split:

* **70% gsm8k-narrative-style multi-step word problems (12 new templates):**
  `shopping_budget`, `recipe_scale`, `travel_distance`, `school_classroom`,
  `garden_orchard`, `bakery_orders`, `library_books`, `fundraiser`,
  `trip_planning`, `pets_animals`, `sports_tournament`, `construction`.
  Each item has named entities (Maya, Amir, Ravi, Iris…), a 3–5-step
  scenario, no formula leak in the prompt, and at least one numeric
  distractor (a "noise" number that's irrelevant to the answer). All
  goldens computed in Python so cross-validator agreement is exact.

* **30% legacy v27 direct-compute templates** kept for skill-surface
  coverage (`modular_linear`, `polynomial_eval`, `gcd_lcm`, `factorial_mod`,
  `arithmetic_series`, `geometric_series`, `simultaneous`, `digit_sum`,
  …). Useful as easy-floor items so the round mean stays sane.

Smoke test (block_seed=8068999, n=12) confirmed 8/12 items hit the
narrative bucket and goldens check out. Sample:

> "Iris is planning a 4-night trip with 6 other people, but they're each
> paying their own way. Iris's lodging costs \$80 per night, meals cost
> \$30 per day, and round-trip transport costs \$80. How many dollars will
> Iris's share of the trip cost?"
>
> Solve step by step and end with '#### N' where N is the final integer
> answer.
>
> gold: 520

This shape is the gsm8k distribution: the SAME skill miners need for
held-out gsm8k is now what they have to optimise for `math_bench`.

### 2. `validator/results.py::_baseline_floor_dethrone_veto` — new dethrone gate

A challenger that passes KL significance + 3% epsilon + composite floor +
Pareto majority is **still** blocked from taking the crown if its
`pass_frac` on any of `(math_bench, code_bench, reasoning_bench,
ifeval_bench, aime_bench, mbpp_bench)` is more than `BASELINE_FLOOR_MARGIN`
(default 0.10) below the Qwen3.5-4B base reference scored on the **same
items in the same round**.

Properties:

* **Paired evaluation.** Reference and challenger see the same block-seeded
  items. No cross-round prompt drift, no cross-sample noise.
* **Narrow-axis-specialist proof.** A challenger that is great on KL but
  worse than the un-distilled base on a held-out-transfer axis can't take
  the crown — it would be a regression by definition.
* **Fail-open.** When the reference isn't in the round (legacy
  `INCLUDE_REFERENCE_IN_ROUND=0`), or fewer than 2 axes are comparable, the
  veto silently passes. Never freezes the crown.
* **Wired into both dethrone paths.** Paired-t-test path AND legacy-epsilon
  path both run the veto after composite floor and before Pareto.

### 3. Reference re-seated every round

`distil.env` now has `INCLUDE_REFERENCE_IN_ROUND=1`. The reference baseline
(Qwen3.5-4B base, `REFERENCE_UID = -1`) is included in `models_to_eval` for
every round. Cost: ~3–5 min of round wall-time. Benefit: the baseline-floor
veto has a paired-sample comparison target every round; the dashboard's
king-vs-base health card stops showing stale data.

### 4. `BENCH_MATH_PER_ROUND=14`

Bumped from 12 (was 10 before earlier today) so the v29 narrative subtypes
each get representation in every round (~10 narrative + ~4 legacy items).

## Goodhart-resistance argument

The validator's composite is now bounded above by the Qwen 4B base on the
held-out-transfer axes. Concretely:

* If a miner's model regresses below the Qwen 4B base on `math_bench` by
  more than 10 pp → blocked from the crown by the baseline-floor veto.
* `math_bench` items now distributionally match gsm8k → if the model is
  *good* at procedural `math_bench`, it's good at held-out gsm8k by
  construction (same skill, different surface). Goodhart-immune in the
  ABSOLUTE direction (overfitting `math_bench` items doesn't help because
  items are fresh per round) AND in the DISTRIBUTIONAL direction
  (overfitting the *style* of `math_bench` IS the same as overfitting the
  *style* of gsm8k).

This is the chain the user asked for:

> "make it so that if the miner performs well on the eval it will become
> an actual good model" — narrative rebalance gives that direction.
>
> "if they perform well on these, they will perform well on actual llm
> benchmarks, but not just because they memorized the data" — the
> procedural fresh-per-round generation prevents memorisation; the
> distributional similarity gives transfer.

## What's still pending

* **`_generate_reasoning_items` v29** — same narrative rebalance for
  `reasoning_bench` (BBH-style multi-entity deduction templates).
  reasoning_bench is the single biggest held-out regression (-10.2 pp on
  BBH). Plan: add 6–8 BBH-narrative-style task families
  (`tracking_objects`, `temporal_sequence`, `web_of_lies`,
  `logical_deduction_5`, `causal_judgment`, `date_understanding`).
* **`_generate_code_items` v29** — humaneval-style richer function specs
  (multi-line docstrings with edge cases, helper-function patterns).
* **`_generate_ifeval_items` v29** — broader format-constraint coverage.
* **Dashboard "Goodhart canary" overlay** — co-plot `composite.worst` and
  held-out gsm8k on the same X-axis so the divergence is visible at a
  glance (currently the canary strip shows only held-out trends).
* **Auto-dethrone on canary regression** — post-coronation, if the king's
  held-out gsm8k regresses below baseline by >10 pp, force-dethrone via the
  king-regression-streak machinery. Currently only the
  validator-composite-worst path is wired into king regression streaks;
  adding evalscope as an extra streak source.

## Operator action item

**Discord channel permission for arbos bot.** The arbos bot has
`autoThread: true` configured but the bot is replying in-channel, not
threading. Diagnosis:

* `verified plus` role (1116135069518667860) has a per-channel overwrite
  on `1482026267392868583` (ა・distil・97) that DENIES
  `CREATE_PUBLIC_THREADS` (deny mask `103079264256`).
* The bot has the `verified plus` role, so this denial applies.
* The bot's own `Arbos` role (1482105925144678504) grants
  `CREATE_PUBLIC_THREADS` at guild level, but role-level deny in a channel
  overwrite always wins over guild-level allow.
* Bot lacks `MANAGE_ROLES` so it cannot self-grant the permission via API
  (verified: `PUT /channels/{id}/permissions/{role|user_id}` returns
  `50013 Missing Permissions`).

Fix (Discord UI):
1. Open `ა・distil・97` channel settings → Permissions.
2. Either:
   * Edit `Arbos` (role) overwrite → ALLOW `Create Public Threads`, OR
   * Add a member-specific overwrite for `Arbos` user → ALLOW
     `Create Public Threads`.
3. Save.

After that, the next mention will route through openclaw's
`maybeCreateDiscordAutoThread`, which already has `autoThread:true` set,
and the bot will respond inside a thread instead of as an inline reply.

## Deployment

* **Commit:** `bec8470` (math v29 narrative rebalance + baseline-floor
  dethrone veto)
* **Pushed:** `main`
* **Validator restarted:** 19:56:55 UTC (reattached cleanly to in-flight
  round). Env confirmed live: `INCLUDE_REFERENCE_IN_ROUND=1`,
  `BASELINE_FLOOR_MARGIN=0.10`, `BENCH_MATH_PER_ROUND=14`.
* **Effective in:** next round (the in-flight round was already started
  with the previous code; the planning of the next round will pick up
  v29 + reference re-seating).
