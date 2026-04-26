# SN97 Pareto Holistic Eval v2 — "overfit-to-eval ⇒ SOTA model"

**Author:** distil-validator maintenance rotation
**Status:** Session 2 design. Land this doc + probes shadow-mode on 2026-04-24. Promote to dethrone gate on 2026-04-26 (48h notice + baseline pass).
**Supersedes:** `reports/2026-04-23-goodhart-immune-eval.md` (Session 1 is still in force; this extends it).
**Inspiration:** [AffineFoundation/affine-cortex](https://github.com/AffineFoundation/affine-cortex) — Bittensor Subnet 120's Pareto-frontier winner-take-all contest across 14+ independent real-world environments (SWE-bench on real repos, ARC, browser tasks, MATH/GSM8K, MemoryGym, MCP tool-use, OpenSpiel games, etc.). Affine distills only as *one* environment among many; we adopt the "absolute correctness across many held-out capabilities" pattern but keep our teacher-KD backbone.

## 1. The problem Session 1 did not solve

Session 1 (2026-04-23) closed the obvious KL-Goodhart holes: it rotated
prompt pools, added an absolute-correctness floor to the capability axis,
introduced a teacher sanity gate, and shadowed a `judge_probe` axis. That
stopped the "ramble-for-10x and still win on KL" failure mode visible on
the 2026-04-22 king, and the Pareto-worst dethrone gate now blocks any
challenger that collapses a single axis.

**What it did not fix.** Every axis in Session 1 is still a *relative*
metric — scored against the teacher, or normalized against the king.
Five relative axes, all pointing at the same signal ("look like the
teacher"), composed with a worst-case rule, is stronger than one relative
axis, but it is still not the signal the user actually wants. Concretely:

- A student that matches the teacher closely on 80-token prompts but
  cannot correctly solve a held-out GSM8K problem still passes every
  Session 1 axis.
- A student whose KL and RKL to the teacher are within 3% but that
  cannot write a working Python function still passes every Session 1
  axis.
- A student that is extremely well-aligned on 52 hand-curated capability
  prompts but collapses on MMLU-Pro questions it has never seen still
  passes every Session 1 axis.

The goal is **"if a miner overfits our eval they produce a SOTA small
model."** Session 1 does not deliver that because our evals don't
require being good at anything in particular, only being close to the
teacher. The teacher is not SOTA; pushing toward "be the teacher, but
cheaper" is not the same as pushing toward "be the best small model
possible."

## 2. What Affine gets right

Affine-cortex has 14+ environments:

| Env                     | What it tests                                   | Scoring                         |
|-------------------------|-------------------------------------------------|---------------------------------|
| `affine:ded-v2`         | Program abduction (derive program from I/O)     | Exact-match                     |
| `affine:abd-v2`         | Program deduction (execute program in head)     | Exact-match                     |
| `cde`                   | Code editing / generation                       | Pass/fail on tests              |
| `lgc`, `lgc-v2`         | Logic / long-chain reasoning                    | Exact-match                     |
| `swe-pro/synth/inf`     | **SWE-bench on real repos** (docker-in-docker)  | Tests pass                      |
| `arc-gen`               | ARC abstract reasoning                          | Exact-match on held-out         |
| `liveweb`               | Browser-based web interaction                   | Task success                    |
| `memory`                | MemoryGym episodic memory over 30+ min          | Episode score                   |
| `navworld`              | MCP tool-use travel planning (AMap + transport) | Anti-hack hardened score        |
| `game`                  | OpenSpiel                                       | Win rate                        |
| `knowledge-eval`        | GPQA + MMLU-Pro + HLE + IFEval                  | Exact-match / instruction-follow|
| `corpus-eval` + `distill`| Teacher distillation (one axis among many)      | KL                              |

Three design choices are load-bearing:

1. **Pareto dominance.** A model wins iff it exceeds every other model
   on every environment. You cannot overfit one environment because you
   will lose on another.
2. **Absolute correctness.** Most axes score against ground truth, not
   against a reference model. "SWE-bench tests pass" is an unforgeable
   claim.
3. **Distillation is one axis, not the axis.** `distill` / `knowledge-eval`
   / `corpus-eval` are three of fourteen+ environments. Losing the
   distillation axis does not lose the contest.

We cannot adopt Affine's infra wholesale (Chutes inference, docker-in-
docker, MCP servers, OpenSpiel — SN97 runs on one eval pod with ~192GB
VRAM). But we can adopt the **pattern**.

## 3. The SN97 adaptation: 5 held-out absolute-correctness axes

New axes added on top of Session 1's five (rkl / kl / capability /
length / degeneracy / judge). All five are drawn from public benchmarks
already cached on the eval pod (`~/.cache/huggingface/datasets`, 187MB
pre-staged by prior evalscope work), so no new network dependencies in
the hot path.

| Axis               | Source                            | Pool  | Per-round  | Scoring                                      |
|--------------------|-----------------------------------|-------|------------|----------------------------------------------|
| `math_bench`       | GSM8K test + MATH-500 test        | 1819  | 8          | Final-answer regex match, `#### N` / `\boxed`|
| `code_bench`       | HumanEval test                    | 164   | 4          | Subprocess sandbox runs unit tests           |
| `reasoning_bench`  | BBH (21 objective subtasks)       | ~5250 | 8          | Exact-match on `\boxed` or final line        |
| `knowledge_bench`  | MMLU-Pro test                     | 12032 | 8          | MC letter (A-J) extraction                   |
| `ifeval_bench`     | google/IFEval train               | 541   | 8          | Instruction-following verifiers (11 kinds)   |

**Rotation.** Each axis samples its per-round items via
`_pick_bench_items(name, block_seed)` using a benchmark-specific stream
key (`seed ^ 0xA13A_MATH` / `…_CODE` / etc.) derived from `random.Random`.
Sampling is without replacement within a round and with replacement
across rounds, as with the hardened `on_policy_rkl` / `capability` /
`judge` pools. A single-round contest therefore sees a tiny, random-
looking subset that is nevertheless reproducible across validators from
the same on-chain seed.

**Combinatorics.** With pool sizes P and sample k:

```
math:      C(1819, 8)  ≈ 2.6e20
code:      C(164, 4)   ≈ 2.8e7
reasoning: C(5250, 8)  ≈ 1.9e24
knowledge: C(12032, 8) ≈ 5.2e27
ifeval:    C(541, 8)   ≈ 5.4e16
```

Memorizing any one of these is effectively impossible under the round
cadence. Memorizing *all five in parallel* requires storing the
benchmarks, which defeats the purpose (you now have the SOTA model).

**Anti-memorization: private holdouts.** Session 3 (next week) will
bump the private-pool fraction from 10% → 30% per axis. The private
pool is generated by the validator and released to miners only on a
commit-reveal cadence. Until then, Session 2 relies on rotation alone
which, at these pool sizes, is already stronger than any single current
axis.

## 4. Axis details

### 4.1 `math_bench` (GSM8K + MATH-500)

**Prompt.** Exact GSM8K text ("Janet's ducks lay 16 eggs…"), no
CoT-forcing. Student generates freely up to 384 tokens.

**Extraction.** For GSM8K-sourced items: grep final integer after
`####` if present in student output; otherwise grep last integer. For
MATH-500: grep `\boxed{...}` first; fall back to last number. Both use
the ground truth's post-`####` answer.

**Score.** Per-item binary exact-match on the normalized numeric
(strip commas, strip leading zeros, compare as int/float with 1e-6
tolerance). Axis = mean pass fraction in [0, 1].

**Why this is Goodhart-immune.** You cannot "almost solve" GSM8K — the
answer is a single integer. A model that beats us on rotated GSM8K/MATH
is a model that can do grade-school-through-competition math on held-
out problems. That is a capability. Overfitting here literally means
getting better at math.

### 4.2 `code_bench` (HumanEval)

**Prompt.** The HumanEval `prompt` field (function signature +
docstring with examples). Student generates freely up to 512 tokens.

**Execution.** Minimal subprocess sandbox: write
`{prompt}{generation}{test}{check(candidate)}` to a tempfile, run with
`python3 -I -s` under `subprocess.run(..., timeout=10, cwd=/tmp)`, kill
tree on timeout. Exit code 0 → pass.

**Safety.** Sandbox runs in the eval pod's container, not the miner's
process. Per-problem timeout (10s), memory limit (256MB via `prlimit`),
`-I` isolates from user site-packages, and we never pass untrusted
stdin. This is tighter than HumanEval's reference implementation which
uses raw `exec()` — we spawn a subprocess so a student's arbitrary code
cannot crash the eval loop.

**Score.** Axis = pass fraction over 4 randomly sampled HumanEval
problems.

**Why Goodhart-immune.** The unit tests are the ground truth. A model
that "overfits" is a model whose greedy generations pass the tests,
which is the definition of a working function.

### 4.3 `reasoning_bench` (BBH, 21 objective subtasks)

**Subtasks used.** `boolean_expressions`, `causal_judgement`,
`date_understanding`, `disambiguation_qa`, `formal_fallacies`,
`geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`,
`logical_deduction_seven_objects`, `logical_deduction_three_objects`,
`movie_recommendation`, `navigate`, `object_counting`, `penguins_in_a_table`,
`reasoning_about_colored_objects`, `ruin_names`, `snarks`,
`sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`,
`web_of_lies`. Excluded: `word_sorting` (string-prefix matching noise)
and multi-lingual ones.

**Prompt.** Exact BBH input text. Student generates up to 128 tokens.

**Extraction.** Each BBH target is a short string (e.g. `(A)`, `True`,
`sortedlist`). Exact-match after strip + case-insensitive + optional
parenthesis strip.

**Score.** Mean pass fraction across 8 sampled items (stratified by
subtask: at most one item per subtask per round).

### 4.4 `knowledge_bench` (MMLU-Pro)

**Prompt.** `{question}\nOptions:\n(A) ...\n(B) ...\n...\n\nRespond
with only the letter.`

**Extraction.** Regex `\b([A-J])\b` on stripped student output. Compare
to `answer` field (letter).

**Score.** Pass fraction over 8 items sampled uniformly at random
from the 12k pool.

### 4.5 `ifeval_bench` (IFEval)

**Prompt.** The IFEval `prompt` field, verbatim.

**Verifiers.** We vendor a minimal subset of Google's official
`instruction_following_eval` verifiers (11 of 25 instruction types,
chosen to cover >70% of IFEval prompts by coverage). See
`scripts/ifeval_vendor.py`. Types covered:

```
length_constraints:number_words
length_constraints:number_sentences
length_constraints:number_paragraphs
keywords:existence
keywords:forbidden_words
keywords:frequency
detectable_format:number_bullet_lists
detectable_format:constrained_response
detectable_format:json_format
detectable_format:title
detectable_format:multiple_sections
punctuation:no_comma
startend:end_checker
startend:quotation
change_case:english_capital
change_case:english_lowercase
change_case:capital_word_frequency
```

Items that require other instruction types are skipped; we pre-filter
the 541-item pool to the subset where *every* `instruction_id_list`
entry is supported (yields ~200 eligible prompts — still plenty for 8
per round).

**Score.** For each prompt, all verifiers for its `instruction_id_list`
must pass. Axis = fraction of items where all verifiers passed.

## 5. Composite integration

All five axes land in shadow mode (2026-04-24). Composite weight layout
once promoted (2026-04-26):

```python
AXIS_WEIGHTS_V2 = {
    # Tier 1 — distributional / teacher-relative (Session 1)
    "on_policy_rkl": 0.15,   # was 0.35
    "kl":            0.05,   # was 0.15
    "capability":    0.05,   # was 0.25 — subsumed by the bench axes
    "length":        0.05,   # was 0.10
    "degeneracy":    0.05,   # was 0.15
    "judge_probe":   0.10,   # was 0.20 in Session 2 plan
    # Tier 2 — absolute-correctness, held-out benchmarks (Session 2)
    "math_bench":       0.15,
    "code_bench":       0.15,
    "reasoning_bench":  0.10,
    "knowledge_bench":  0.10,
    "ifeval_bench":     0.05,
}
```

The `worst` rule keeps its dominance: a challenger wins only if its
min-axis exceeds the king's min-axis by `COMPOSITE_DETHRONE_MIN_MARGIN`.
Tier 2 axes dominate the bound because they are harder to saturate —
which is exactly the point. If you cannot dethrone without first
matching the king on GSM8K and HumanEval, then the thing miners
optimize for is GSM8K and HumanEval.

**Teacher sanity.** The teacher-as-student pass from Session 1
automatically extends to the new axes. If Qwen3.5-35B scores below
`TEACHER_SANITY_FLOOR` (0.70) on any bench axis in a round, that axis
drops out for the round. In practice the teacher scores ~0.92 on
HumanEval, ~0.82 on MATH-500, ~0.71 on MMLU-Pro; the floor is
approximately calibrated for that.

## 6. Cost

Per student, per round, worst-case forward-pass budget:

| Axis              | Prompts | Max tokens | Approx wall (B200)   |
|-------------------|---------|------------|----------------------|
| math_bench        | 8       | 384        | ~15 s                |
| code_bench        | 4       | 512        | ~10 s + ~4 s sandbox |
| reasoning_bench   | 8       | 128        | ~6 s                 |
| knowledge_bench   | 8       | 64         | ~3 s                 |
| ifeval_bench      | 8       | 512        | ~20 s                |
| **Total added**   | **36**  | –          | **~60 s / student**  |

Our 6-student round takes ~45 min end-to-end today, 10 min of which is
probe / scoring per student. Adding 60s per student ⇒ +6 min per round,
~10% eval cost. Acceptable.

## 7. Staged rollout

- **2026-04-24** (this PR): Ship the probes + composite axes in
  shadow mode (`BENCH_AXES_IN_COMPOSITE=0`). Dashboard shows them;
  dethrone gate ignores them.
- **2026-04-24** (same PR): Discord announcement with 48-hour notice.
- **2026-04-26**: Flip `BENCH_AXES_IN_COMPOSITE=1`. Judge promoted to
  production in the same flip (this folds Session 1's Session-2 plan
  into Session 2's Session-3 plan, cleanly).
- **2026-05-02** (Session 3): Private-pool fraction 10%→30%.
  Commit-reveal plumbing. Optional: add a sandbox tool-use axis if
  MCP infra is wired up by then.

## 8. Invariants this design upholds

1. **Absolute > relative.** ≥5 of the 11 axes score against ground truth,
   not a reference model. The composite worst-case cannot silently
   collapse to "match the teacher" because at least half of the axes
   do not measure that.
2. **Pool ≫ sample.** Every rotated pool is >20× the per-round sample.
   Memorization attacks require storing the underlying benchmark, at
   which point the miner has a SOTA model by construction.
3. **Teacher sanity defended.** Every axis is screened against the
   teacher itself; axes the teacher fails are dropped for the round.
4. **Worst-case > mean.** The dethrone gate is on `composite.worst`,
   so gaming any one axis costs you the crown by hurting the min.
5. **Scoring is reproducible.** All probes are seeded by the round's
   on-chain `block_seed`. Any validator can rerun and get the same
   score given the same snapshot + seed.
6. **Overfit ⇒ SOTA.** If a miner cracks this composite — greedy
   generations achieve >0.9 on math / code / reasoning / knowledge /
   ifeval plus >0.9 on rkl / kl / capability / length / degeneracy /
   judge — they have trained a 4B model that genuinely does grade-school
   math, writes working code, reasons, knows facts, follows instructions,
   *and* aligns with a 35B teacher. That model is SOTA at its scale.

## 9. What this does not give us (honest delta vs Affine)

- **No SWE-bench on real repos.** Docker-in-docker on the eval pod is
  out of scope; Chutes-style isolated inference is a different
  architecture.
- **No agentic / tool-use axis.** Would require MCP servers; candidate
  for Session 3 if we stand up the infra.
- **No multi-turn axis.** All probes are single-turn. We can add
  multi-turn debate / persuasion in a future round; for now single-turn
  is enough to catch the "thinks too long about simple questions"
  pathology.
- **Private pools are small.** 30% private is meaningful but less than
  Affine's commit-reveal-backed opaque environments. We chase this in
  Session 3.

What we *do* give miners is the user's ask: if you overfit this, you
ship a model that actually does the work. That is the contract.
