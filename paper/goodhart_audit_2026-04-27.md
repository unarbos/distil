# Goodhart audit — composite v28 — 2026-04-27

Per-axis review of the v28 composite's memorisation surface. For each
axis we record the item source, what a memorisation attack would
need to do, and whether such an attack is practically feasible
today.

The bar for "Goodhart-resistant" is: a miner who has the entire
public training corpus (GSM8K, HumanEval, MBPP, BBH, MMLU-Pro,
IFEval, AIME, ARC) downloaded and built a `{prompt → answer}` lookup
should NOT be able to saturate the axis. They should still need to
have a model that genuinely solves the underlying skill.

## Tier 1 — Distribution-match axes (relative to teacher)

| axis | weight | source | rotation | residual surface |
|---|---|---|---|---|
| `kl` | 0.15 | climbmix-400b prompts → teacher continuation, KL on student | block-seed sampler over 400B-token corpus, no on-disk pool | none — corpus is too big to memorise; teacher continuation is non-deterministic at temperature |
| `on_policy_rkl` | 0.35 | 80-prompt static pool, **paraphrased per round** via `_paraphrase_chat_prompt` (Session 3.19) | per-(block_seed, prompt) chat-domain synonym swap; sampling seed XOR block_seed (Session 3.10) | a miner could pre-train on the 80 prompts' canonical wordings — but the per-round paraphrase forces them to generalise across phrasings, which is the point of distillation |
| `capability` | 0.25 | 24/36 procedural (number theory, string ops, arithmetic) + 12/36 static-pool trivia | static portion rotates 24/200 entries per block_seed; procedural portion is fresh (Session 3.12) | the 12 trivia items per round draw from a 200-entry pool; a miner who memorises all 200 saturates that 12-item slice. Mitigated by the procedural 24/36 majority |

**Verdict:** kl is uncheatable (massive corpus). on_policy_rkl and
capability are paraphrased + mostly-procedural; saturation requires
genuine skill.

## Tier 2 — Discipline axes (validator-internal)

| axis | weight | source | rotation | residual surface |
|---|---|---|---|---|
| `length` | 0.10 | derived from chat-probe response lengths | n/a — measurement on student's own output | not gameable; you optimise by being concise |
| `degeneracy` | 0.15 | n-gram repeat / self-BLEU / gzip-ratio on student rollouts | distribution-test against teacher's degeneracy on the same prompts (Session 3.7) | not gameable; you optimise by writing diverse text |
| `reasoning_density` | 0.05 | pass_frac × length_bonus per bench, target tokens per task | derived from student's behaviour on rotating bench items (above) | not gameable independently |

**Verdict:** all three are derived measurements on student behaviour
on otherwise-rotated content. No memorisation surface.

## Tier 3 — Dialogue probes (rubric-graded)

| axis | weight | source | rotation | residual surface |
|---|---|---|---|---|
| `judge_probe` | 0.15 | 65-prompt static pool, **paraphrased per round** (Session 3.18) | per-(block_seed, prompt) chat-domain synonym swap; teacher 1-5 rubric grading | a miner who pre-distils canonical 5/5 responses for all 65 prompts faces rotated phrasings; same defence as on_policy_rkl |
| `chat_turns_probe` | 0.08 | ~25-conversation static pool of 3-turn dialogues, paraphrased per round (Session 3.18) | per-(block_seed, prompt) per-turn paraphrase | same; multi-turn coherence is harder to memorise blindly |

**Verdict:** static prompt pools, but each round's specific phrasing is fresh.
Direct lookup attacks fail. The 65 + 25 prompt surface is small enough
that a miner could in principle learn rubric-passing behaviour
distributionally on it; that would manifest as judge_probe → 1.0 with
poor performance on other axes (a Goodhart signature we'd catch on
held-out display).

## Tier 4 — Procedural bench axes (Session 3.20)

All seven of these are **procedurally generated** per round from
`block_seed`. The (problem, gold) tuples exist nowhere on disk.

| axis | weight | generator | covers |
|---|---|---|---|
| `math_bench` | 0.14 | `_generate_math_items` — 18 procedural subtypes (modular_linear, rate_distance, mixture, percentage, gcd_lcm, polynomial_eval, arithmetic_series, geometric_series, digit_sum, unit_conversion, simultaneous, factorial_mod, set_intersect, probability_count, triangle_area, coin_change, time_arithmetic, proportion) | grade-school math, similar in distribution to GSM8K |
| `code_bench` | 0.14 | `_generate_code_items` — function-completion problems with executable test harness | HumanEval-style |
| `reasoning_bench` | 0.10 | `_generate_reasoning_items` — BBH-style logic with inline-options | BBH 27-task suite |
| `ifeval_bench` | 0.07 | `_generate_ifeval_items` — instruction-following with strict format checks | IFEval |
| `aime_bench` | 0.10 | `_generate_aime_items` — olympiad-style integer-answer math | AIME |
| `mbpp_bench` | 0.08 | `_generate_code_items` second config | MBPP+ |
| `tool_use_bench` | 0.06 | `_generate_math_items` (different stream) — agentic Python | tool-use |
| `long_context_bench` | 0.04 | `_generate_long_context_items` — needle-in-haystack over generated documents | long-context |
| `robustness_bench` | 0.07 | `_generate_math_items` (different stream) under k paraphrase + noise wrappers | distribution-shift robustness |

### Residual concern — distribution-similarity attack

A miner who fine-tunes on GSM8K's training set will generalise well
to `_generate_math_items` because the procedural generator covers
similar word-problem patterns. This is NOT memorisation in the
canonical sense — the (problem, answer) pair from GSM8K won't match
any procedural item. But it IS a form of overfitting to the
*distribution* the procedural generator samples from.

**Why this is acceptable:** a model that fine-tunes on GSM8K and
genuinely improves at multi-step word problems IS a more capable
model. That's a feature, not a bug. The composite is doing exactly
what we want.

**The Goodhart signal we still want to catch:** a model that
saturates `math_bench` while regressing on the held-out auto-bench
(GSM8K, MATH-500). That's overfitting-to-the-procedural-distribution
without underlying generalisation. We watch for this in the
dashboard's Bench tab.

The 2026-04-27 UID 160 → UID 123 transition showed an early
indicator: gsm8k regressed -5pp, bbh -9pp on held-out while
composite.worst held flat at 0.667. That's the signal to track.

## Held-out auto-bench (display only, NOT in composite)

These ARE the public datasets. They're served via
`scripts/run_king_benchmark.py` against the live king's vLLM, AFTER
the round, and emitted to `state/benchmarks/uid_<N>.json` for
dashboard display. They have **zero effect** on composite, ranking,
or king selection.

Their job is precisely to detect the Goodhart signature: composite
climbs while held-out flatlines or regresses → procedural axes are
being optimised in ways that don't transfer.

## Open follow-ups

1. **Larger judge_probe pool.** 65 entries is small; expanding to
   200-300 makes distributional rubric-passing harder. Roadmap for v29.
2. **Larger chat_turns pool.** Same.
3. **Quantify the GSM8K → math_bench transfer.** A miner-side ablation
   would help: take a model fine-tuned only on GSM8K, score it on
   `_generate_math_items`, see how high it goes. If high, we may want
   to push the procedural generator further from GSM8K's distribution
   (e.g. less word-problem-shaped, more proof-shaped).
4. **Continuous Goodhart canary.** Currently the held-out canary is
   a manual bench run. Wire it into the dashboard so the divergence
   between composite.worst and held-out is plotted as a time series
   per king.

## Bottom line

The v28 composite is in solid shape against the canonical
"miner has the public dataset and builds a lookup" attack. The
remaining attack surface is distribution-similarity overfitting,
which is partially what the eval is supposed to reward, and we have
a held-out canary for the rest.

The bot's earlier agreement that "we encourage model overfit on
these benchmark" was wrong against the v27/v28 procedural switch.
The category of concern is real and we already mitigate it.
