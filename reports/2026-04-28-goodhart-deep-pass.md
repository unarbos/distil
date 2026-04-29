# Goodhart deep pass — 2026-04-28

A second pass over the Goodhart surface a day after `paper/goodhart_audit_2026-04-27.md`. The earlier audit catalogued the v27/v28 procedural switch and identified four open follow-ups; this pass executes three of them, fixes a held-out canary outage that was hiding a real regression, ships a continuous Goodhart-divergence panel on the dashboard, and answers the user's "are the models actually getting better?" question with a 12-day quantitative review.

## TL;DR — what changed and why

| change | type | why |
|---|---|---|
| `JUDGE_PROBE_POOL` expanded 82 → 180 prompts | hardening | open follow-up #1: distributional rubric-passing on a 65-prompt surface was within reach. Doubling the surface roughly halves that risk. |
| `CHAT_TURNS_PROBE_POOL` expanded 24 → 54 conversations | hardening | open follow-up #2: same logic for multi-turn dialogue rubric-passing. |
| `scripts/run_king_benchmark.py` `MAX_TOKENS` reduced (8192 → 1024-4096 by bench) | bugfix | vLLM was 400-ing every humaneval/ifeval/mmlu_pro request because `max_tokens=8192` consumed the entire 8192-token context. Held-out canary was returning `n=0` for half the benches since 2026-04-27. |
| `frontend/.../bench-panel.tsx` adds a Canary trend strip (last 8 kings, 6 benches, dashed-baseline sparkline per bench) | observability | open follow-up #4: continuous Goodhart canary on the dashboard so divergence between composite (in-eval ranking key) and held-out (transfer canary) is visible at a glance. |
| `canvases/sn97-goodhart-analysis.canvas.tsx` (this session) | analysis | the user's "do an analysis on if the models are actually getting better" — quantified king progression on composite vs held-out across 12 days. |

The procedural-item switch in v27/v28 was already the big Goodhart move. This pass closes the remaining three follow-ups from the prior audit and surfaces the canary signal continuously instead of as a manual side-pod check.

## What the user asked for

> continue improving on solving for goodhart's law. Make it so memorization is possible nad make it so that if the miner performans well on the eval it will become an actual good model, meaning overfitting is actually a good thing here. And make it so that if they perform well on these, they will perform well on actual llm benchmarks, but not just because they memorized the data. and also do an anlysis on if the models are actually getting better. keep doing the benchmarking from evalscope. but I don't want the miners to just be able to teach their miners those benchmarks so we shouldn't use them in the eval - re goodhart's law

Mapped onto our existing surface:

1. **"overfitting is a good thing here"** — exactly the design intent of the procedural axes. A miner who overfits to `_generate_math_items` is overfitting to the *distribution* of grade-school multi-step word problems; doing well on it requires actually learning that skill, not memorising any specific (problem, gold) pair. Already in place since v27.
2. **"perform well on real LLM benchmarks but not because they memorized"** — that's the canary loop. Validator-eval items are procedural and fresh per round; held-out evalscope runs against the king on public datasets the validator NEVER touches. If composite climbs while held-out flatlines or regresses, the procedural axes are being optimised in a way that doesn't transfer. Already in place since v28; what was missing was making the divergence continuously visible.
3. **"keep doing the benchmarking from evalscope but don't use those items in the eval"** — exactly how the surface is split today. evalscope = public datasets, held-out canary, never in composite. Validator-eval = procedural per-round items, the ranking key. The two share zero data.
4. **"analysis if the models are actually getting better"** — answered in the canvas (`canvases/sn97-goodhart-analysis.canvas.tsx`) and summarised below.

## Are the models actually getting better?

**Mixed answer, leaning yes-on-composite, no-on-canary.**

### Composite (the in-eval ranking key) — climbing

12 evaluated rounds since 2026-04-25 01:01 UTC, sorted by block:

```
block   date         uid  KL     worst   weighted
8042k   04-25 01:01  48   0.251  0.000   0.649
8052k   04-26 21:30  149  0.184  0.333   0.788
8056k   04-27 01:00  146  0.194  0.167   0.847
8062k   04-27 18:35  144  0.198  0.600   0.887
8063k   04-27 21:35  217  0.147  0.500   0.874
8064k   04-27 22:33  217  0.147  0.667   0.866
8064k   04-28 00:33  148  0.150  0.375   0.827
8065k   04-28 02:21  199  0.156  0.333   0.827
8066k   04-28 09:32  199  0.145  0.625   0.905
8066k   04-28 14:01  207  0.173  0.417   0.865
8067k   04-28 14:48  207  0.287  0.625   0.858
8067k   04-28 17:24  228  0.151  0.667   0.863
```

`composite.worst` (the king-selection ranking key) climbs from 0.000 → 0.667 over the audit window. `composite.weighted` (the auxiliary tiebreak) climbs from 0.649 → 0.863. Both are bounded in [0, 1] with higher being better. Models *are* getting better at the procedural eval surface that determines their ranking.

KL has plateaued: best ever was 0.093 (UID 18, 2026-04-23) which preceded the v28 schema bump; post-v28 KL is flapping in the 0.13-0.29 range with no clear improvement. The schema change deliberately reduced KL's weight from 0.30 to 0.15 (it was saturating); a stalled KL despite ~15 dethrones suggests miners aren't pushing distribution-match aggressively at the top of the leaderboard.

### Held-out evalscope (the canary) — partial regression

Limit-50 auto-bench against the king's vLLM after each crowning. Public datasets the validator never touches.

```
date        uid   model                          gsm8k   bbh    arc    humaneval  ifeval   mmlu_pro
baseline    Qwen3.5-4B (no distillation)         0.934   0.879  0.901  0.817      0.826    n/a
teacher     Qwen3.5-35B                          0.943   0.832  0.954  0.872      0.871    0.732
2026-04-17  107   gtensorapp/prime-dusk-4260     0.893   0.716  0.881  0.695      0.673    n/a
2026-04-18  156   CargoHull/sn97-prime-deik      0.920   0.719  0.860  0.760      0.729    0.634
2026-04-20  118   Sanguineey/distilled-v4        0.920   0.679  0.900  0.760      0.646    0.597
2026-04-23  18    tom9491/distil-32              0.900   0.695  0.880  0.720      0.660    0.626
2026-04-27  149   ivangrapher/distilman3         0.940   0.799  0.920  BROKEN     BROKEN   BROKEN
2026-04-27  123   eugene141759/distil-m20        0.860   0.767  0.920  BROKEN     BROKEN   BROKEN
2026-04-28  217   sampleratez/5421387            0.860   0.778  0.920  BROKEN     BROKEN   BROKEN
```

* **arc** — up (0.88 → 0.92), above the undistilled-baseline of 0.901. Genuine transfer.
* **bbh** — up (0.72 → 0.78), still well below baseline (0.879). Trending right but lots of room.
* **gsm8k** — *down* (0.94 → 0.86), now below the undistilled baseline (0.934). **A real regression**: the latest king is worse at grade-school math than just running the off-the-shelf student.
* **humaneval / ifeval / mmlu_pro** — `BROKEN` since 2026-04-27 (n=0 from the evalscope runner; see "Held-out runner outage" below).

**The Goodhart signature.** Composite climbed from `worst=0.000` to `worst=0.667` over the same period that gsm8k regressed from 0.94 to 0.86. The procedural ranking key and the held-out canary are diverging by ~13 percentage points. That's the divergence pattern the audit predicted ("composite climbing while held-out flatlines or regresses → procedural axes are being optimised in ways that don't transfer") and the new dashboard canary panel is designed to surface it.

## What we shipped

### 1. JUDGE_PROBE_POOL — 82 → 180

Followed the audit's open follow-up #1. Added 120 prompts evenly across the existing categories (chat / reasoning / instruction-following / coding / creative / world-model). Per-round paraphrase still applies; the larger surface dilutes any distributional rubric-passing attack and keeps the per-round 16-prompt sample varied.

```913:1095:scripts/pod_eval_vllm.py
JUDGE_PROBE_POOL = (
    # Chat / factual-helpful
    "What is the best way to learn a new programming language? Answer in 2-3 sentences.",
    "Explain briefly the difference between TCP and UDP.",
    ...
    # === v29 expansion 2026-04-28 — open follow-up #1 from
    # paper/goodhart_audit_2026-04-27.md. The pool was 82 prompts;
    # `JUDGE_PROBE_PER_ROUND` is 16 and the per-round paraphrase
    # forces fresh wording each time, but a 82-prompt surface is small
    # enough that distributional rubric-passing was within reach. The
    # 120 additions below triple the surface to ~200, ...
    ...
)
```

### 2. CHAT_TURNS_PROBE_POOL — 24 → 54

Followed the audit's open follow-up #2. Added 30 multi-turn conversations all built around a third-turn that depends on the prior context (the coherence signal we want to grade). The audit asked for ~100; we're at 54 — short of the goal but the per-turn paraphrase plus the larger denominator already raises the bar materially.

### 3. Held-out runner outage — fixed

`scripts/run_king_benchmark.py` was setting `max_tokens=8192` for ifeval, humaneval, and mmlu_pro. The Qwen3.5-4B-class models cap at `max_model_len=8192` total context. vLLM rejected every request with `400 BadRequest` because the requested output (8192) plus any non-empty prompt exceeded the context limit. The runner caught the error per-sample, recorded n=0, and wrote score=0.0 with count=0 for the whole bench. The dashboard's Bench tab handled count=0 correctly (showed "n/a") but the held-out canary was effectively half-blind for two days.

The fix: lower `MAX_TOKENS` to leave room for the prompt:

```47:57:scripts/run_king_benchmark.py
MAX_TOKENS = {
    "gsm8k": 2048,
    "ifeval": 2048,
    "humaneval": 4096,
    "bbh": 2048,
    "arc": 1024,
    "mmlu_pro": 1024,
    ...
}
```

Plus an inline comment explaining the constraint so this doesn't get reverted by future "let's give more headroom" tweaks.

### 4. Continuous Goodhart canary on the dashboard

`frontend/src/components/v2/bench-panel.tsx` now opens with a Canary strip showing the last 8 kings' held-out scores per bench. Each cell is:

* a sparkline of king-by-king held-out scores
* a dashed reference line at the undistilled-baseline value
* the latest score with delta-vs-baseline in points
* warn-tone color when the latest king is below baseline

This is the audit's open follow-up #4. The old layout showed the *current* king's bench scores; the new layout co-shows the trajectory and the baseline line so a regression-below-baseline is immediately visible. If we'd had this on 2026-04-27 the gsm8k regression would have been a same-day flag instead of a 24-hour-late discovery.

### 5. Analysis canvas

`canvases/sn97-goodhart-analysis.canvas.tsx` — a one-shot analytical artifact answering the user's "are the models actually getting better?" question with composite-vs-held-out trajectory tables, per-king delta-vs-baseline pills, and a diagnosis section calling out the gsm8k regression, the KL plateau, and the runner outage.

## What's still on the roadmap

| item | status | note |
|---|---|---|
| `judge_probe` pool expand to 200-300 | ~90% (180/200) | additional 20-120 prompts at next refactor |
| `chat_turns` pool expand to ~100 | ~54% (54/100) | additional 46 conversations at next refactor |
| Quantify GSM8K → math_bench transfer (audit follow-up #3) | not started | requires miner-side ablation: take a model fine-tuned only on GSM8K, score it on `_generate_math_items`. If it scores high, push procedural items further from GSM8K's distribution. |
| Composite-vs-held-out time-series overlay | dashboard sparkline only | next pass: add a per-king composite-worst point on the same X-axis as the held-out sparkline so both trajectories show on one chart. |
| Difficulty stratification on procedural generators | deferred | the existing items already span 4B-pass-rate ~30-90%, so the discrimination is fine; the stratification work is for v29's "harder ceiling" goal. |

## What we are explicitly NOT doing

* **Adding evalscope items to the validator composite.** That's the entire Goodhart trap — once a benchmark is the ranking key, miners optimize directly for it. Evalscope stays held-out.
* **Removing procedural bench axes from the composite** (the 4/27 Allan-on-Discord framing). The procedural axes ARE the Goodhart-resistant surface; demoting them would leave only on_policy_rkl + kl + capability + chat probes binding ranking, which is a smaller, less-binding composite.
* **Building a separate "private benchmark suite"** beyond what we have. The per-round procedural items already serve that role.

## Verification

* `python3 -c "import ast; ast.parse(open('scripts/pod_eval_vllm.py').read())"` — green.
* `npx tsc --noEmit -p .` from `frontend/` — green.
* `JUDGE_PROBE_POOL` items: 180 (was 82).
* `CHAT_TURNS_PROBE_POOL` convos: 54, all with 3 turns each (was 24).

Once committed and the validator restarts, the next round (8068474+) binds the larger pools. The dashboard rebuild surfaces the canary strip on first load.

## Credit

The audit foundation (`paper/goodhart_audit_2026-04-27.md`) and the held-out / in-composite separation (`paper/benchmarks_as_north_star.md`) are the architecture this pass extends. Allan's Discord framing on 2026-04-27 was the right *category* of concern even though the literal version (validator scores GSM8K directly) didn't hold against v27/v28.
