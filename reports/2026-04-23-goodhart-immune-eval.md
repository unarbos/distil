# Goodhart-immune eval design (2026-04-23)

## Invariant we're targeting

> *A miner who completely overfits to the eval should produce the best possible model.*

If overfitting to our eval produces a bad model, our eval is the problem. The
2026-04-22 king (`tom9491/distil-32`) is a live counterexample: it wins the
current eval handsomely and is nearly unusable as a chat assistant because it
rambles into 8192-token timeouts on trivial prompts. That pathology is
invisible to our primary ranking signal (forward-KL on pretraining-text
continuations) because per-token distribution-matching on web text is
orthogonal to "answers questions well." The fix is not more gates on top of
KL — the fix is for the eval to *be* the quality signal, so there's no
Goodhart gap between "scores well" and "is actually good."

This doc specifies the invariants, audits every signal against them, and
stages the migration in two sessions so miner incentives don't flip
overnight.

## Design invariants

An eval that satisfies the core invariant must satisfy all of:

1. **Cover every quality dimension users care about.** If users care
   about correctness, instruction-following, length discipline,
   coherence, and reasoning, then each must be an independent axis.
   Missing axes == free Goodhart dimensions.
2. **Measure absolute quality, not just similarity-to-teacher, wherever
   possible.** A student that matches the teacher's wrong answers
   should not score 1.0. A student that produces more correct, more
   helpful text than the teacher should be able to exceed 1.0 on the
   axis (or at least tie cleanly at 1.0).
3. **Anti-memorization by construction.** Every per-round prompt set
   must be drawn from a pool much larger than a miner could memorize,
   with rotation driven by unpredictable entropy (on-chain block hash).
   Fixed prompt lists are memorizable; they belong to the previous
   design.
4. **Teacher-sanity check on every axis every round.** The reference
   teacher (Qwen3.5-35B) should, by construction, be a strong model on
   every axis we rank on. If the teacher itself scores < 0.70 on an
   axis in a given round, the axis is miscalibrated and must be
   dropped from composite ranking automatically — not wait for a human
   incident response like the 2026-04-19 Wilson-anchor outage.
5. **Worst-axis dominance.** The axis that would be easiest to game in
   isolation should be the axis that decides the ranking. This is the
   existing ``composite.worst`` rule, promoted to production on
   2026-04-19 and given a dethrone veto floor on 2026-04-22. It stays.
6. **Gates fail open for pod-side outages, fail closed for
   incentive-corrupting bugs.** A missing axis drops out (renormalize
   over the surviving axes). A populated axis where the teacher scores
   < 0.70 is explicitly dropped with a warning. A populated axis where
   the teacher scores >= 0.70 is trusted.

## Audit of the pre-2026-04-23 eval against these invariants

### 1. KL global avg — the primary ranking signal

| Invariant | Pass/fail |
| --- | --- |
| Covers a user-visible quality dimension | FAIL. Measures per-token output distribution on pretraining-text continuations, not answer quality. |
| Absolute quality | FAIL. Zero KL ≠ good model — it means logit-identical to teacher on noise. |
| Anti-memorization | PASS. ~300 prompts sampled per-round from climbmix shard selected by on-chain block hash. Private-pool (10%) with commit-reveal. |
| Teacher sanity | N/A. Teacher is the anchor by definition. |
| Worst-axis dominance | WAS the dominant axis until 2026-04-19; still the only axis that actually gates dethrone via the paired t-test + 3% epsilon. The composite is a veto floor on top, not a replacement. |

**Verdict.** This is the textbook teacher-hacking attractor. The mitigation
path is to demote KL from the primary dethronement gate to one axis among
many, which Session 2 does.

### 2. on_policy_rkl

| Invariant | Pass/fail |
| --- | --- |
| Covers a user-visible quality dimension | PARTIAL. On-policy mode-seeking is closer to what users experience than off-policy KL, but the signal is "do your own rollouts resemble teacher's preferences" — still a proxy. |
| Absolute quality | FAIL. Normalized by king RKL, so a student tied with a bad king scores 1.0. |
| Anti-memorization | FAIL. 16 completely static prompts. Memorizable. |
| Teacher sanity | The teacher trivially scores perfectly here by construction (RKL vs itself is 0). |
| Worst-axis dominance | Enforced via composite. |

**Session 1 fix.** Expand to an 80-prompt pool, sample 16/round via
``block_seed`` (mirrors the 2026-04-19 think-probe hardening). Pool covers
chat-style queries, instruction-following, arithmetic, reasoning,
creative writing, and translation, so overfitting to the set means having
real coverage across all of them.

### 3. capability

| Invariant | Pass/fail |
| --- | --- |
| Covers a user-visible quality dimension | PASS. Verifiable correctness is the single most Goodhart-resistant axis we have — the answer is the answer. Tülu 3 RLVR (Lambert et al. arXiv:2411.15124) established this line. |
| Absolute quality | FAIL. Normalized ``pass_frac / teacher_pass_frac``. A student that matches the teacher's wrong answers scores 1.0. |
| Anti-memorization | PARTIAL. 52-item static pool + 12 procedural math/round, rotated by block_seed. 52 is memorizable. |
| Teacher sanity | Teacher scores well but its errors should not become student's ceiling. |

**Session 1 fix.** (a) Expand static pool to 160+ items with coverage of
trivia, arithmetic, IFEval-style format compliance, simple reading
comprehension, word problems, small code outputs. (b) Add an absolute-
correctness floor: ``score = (pass_frac + min(pass_frac / max(teacher, 0.5), 1.0)) / 2``.
A student and teacher both scoring 30% now maps to 0.55, not 1.0. A student
scoring 100% with teacher at 80% still maxes at 1.0. The shape preserves
"tied with teacher == ~teacher's quality" without rewarding matched
failure. (c) Per-round rotation already works — no change to that.

### 4. length

| Invariant | Pass/fail |
| --- | --- |
| Covers a user-visible quality dimension | PASS. "Don't ramble on trivial prompts" is genuinely what users complain about. |
| Absolute quality | N/A. The axis is a ratio against teacher, which is the right anchor for length. |
| Anti-memorization | FAIL. chat_probe path uses 4 fixed trivial prompts; think_probe path is 32 rotated but disabled. Memorizable. |
| Teacher sanity | Teacher trivially satisfies ratio = 1.0 vs itself. |

**Session 1 fix.** Same treatment as on_policy_rkl — broaden the chat-probe
prompt set to a 40-prompt pool, sample 4-8/round via block_seed. Keeps the
"trivial prompts, short answers" character but stops memorization.
Deferred to Session 2 because chat_probe length also feeds the production
length axis and I do not want to change what the existing length axis
measures in a single commit; Session 1 only expands the *ranking* surface.

### 5. degeneracy

Currently off (THINK_COLLAPSE_PROBE=0) after the 2026-04-19 outage, see
``reports/2026-04-19-think-probe-disabled.md``. When on, prompt pool is
rotated and the axis is well-designed. It comes back in Session 2 after
the teacher-sanity gate is in place to prevent the original miscalibration
failure mode.

### 6. finetune probe / activation fingerprint

Both are DQ signals, not ranking axes. Both are anti-adversarial by
design (anti-finetune probes train-time resistance; activation fingerprint
detects functional copies). Keep as-is.

## Session 1 — shipped 2026-04-23

All changes in commit ``<TBD>`` on ``/opt/distil/repo``, deployed to the
eval pod same day.

### 1. on_policy_rkl pool expansion (anti-memorization)

- Pool expanded from 16 static → 80 candidate prompts covering chat,
  instruction-following, reasoning, creative writing, translation,
  arithmetic.
- Per-round sampling of 16 via ``_pick_on_policy_rkl_prompts(block_seed)``
  using the same ``random.Random(int(seed))`` pattern as
  ``_pick_think_probe_prompts``.
- Combinatorial coverage: C(80, 16) ≈ 4.88·10^14 distinct sets per round.
  Memorizing every possible set is out of reach.
- Unchanged: temperature, top_p, max_new_tokens, top_k_logits, seed.

### 2. capability pool expansion (anti-memorization)

- Static pool expanded from 52 → ~160 items, same categorical mix:
  word, word_alt, int, yesno, phrase/regex, format_re, word_count,
  rhyme, mc.
- Per-round rotation preserved (block_seed shuffles the pool, then
  takes the first ``CAPABILITY_PROBE_N``). Existing
  ``set_capability_block_seed`` path unchanged.

### 3. capability absolute-correctness floor

- Current formula: ``score = clip(pass_frac / teacher_pass_frac, 0, 1)``.
- New formula: ``score = (pass_frac + min(pass_frac / max(teacher, 0.5), 1.0)) / 2``.
- Rationale: the relative term credits "you matched teacher on a set
  where teacher got 30%"; the absolute term penalizes 30% accuracy
  regardless. Averaging both gives a signal that is monotonic in both
  absolute and relative correctness. A miner's straightest path to 1.0
  is to *be 100% correct*.

### 4. Teacher sanity gate

- New per-round pass: score the teacher as a pseudo-student on each axis
  where it isn't trivially 1.0 by construction.
- Axes where the teacher scores < ``TEACHER_SANITY_FLOOR`` (0.70) are
  flagged ``teacher_broken: true`` in the results and dropped from
  composite normalization for that round with a warning log.
- Covers the 2026-04-19 failure mode (probe miscalibration DQs teacher
  too) automatically; no more stale-failure-counter recovery needed.

### 5. judge_probe — shadow only

- 64-prompt pool of realistic user queries (chat, coding, reasoning,
  instruction-following, creative). Rotated via block_seed to 16/round.
- Student generates greedy, 256 new tokens, ``enable_thinking=False``.
- Teacher (already loaded in vLLM for Phase 1) is prompted:
    "Rate this response on a 1-5 scale. 5 = excellent, 1 = poor.
     Output only the digit." — parsed as int, clipped to [1, 5],
     mapped to [0, 1] by ``(score - 1) / 4``.
- Per-prompt scores averaged to the axis value.
- **Shadow only in this session.** Computed, logged, shown on the
  dashboard under ``shadow_judge``, but NOT included in
  ``AXIS_WEIGHTS`` for ranking. Promotion to production is gated on
  48h of clean shadow data showing teacher-judged-on-teacher scores
  > 0.85 and challenger distribution not collapsing at the extremes.

## Session 2 — promised for 2026-04-25 (or +48h from this post)

Held for a separate commit + 48h miner notice:

1. Flip ``judge_probe`` from shadow to production. Proposed weights:
   ``on_policy_rkl=0.25, kl=0.10, capability=0.20, length=0.10, degeneracy=0.10, judge=0.25``.
   Net effect: KL stays the same (0.10, already down from 0.35 pre-2026-04-19);
   judge becomes the single largest weight, capability stays high, RKL slightly
   reduced because the judge axis overlaps with it conceptually.
2. External benchmark axis. Pull `evalscope` subsets (gsm8k, humaneval, ifeval, bbh)
   into a rolling per-UID-revision cache, refreshed daily. Emit as one composite
   axis ``benchmark`` weighted 0.15, with the other weights renormalized.
3. Replace the KL paired t-test + 3% epsilon dethrone gate with a
   ``composite.worst > king.worst + epsilon`` gate. KL-specific epsilon is
   retained only as a sanity tripwire (challenger must still not regress
   KL by more than 10%).
4. Bump private-pool fraction from 10% → 30% of per-round KL prompts.
5. Chat-probe length pool expanded from 4 fixed → 40-prompt pool with
   rotation. (Held to Session 2 because chat_probe length is what the
   current composite veto fires on, and I do not want to change the
   thing we just shipped a veto against in the same week.)

## Migration ethics

This sequence gives every miner at least 48 hours' notice before their
incentive surface changes in a material way. Session 1 ships today
because it is either (a) pure anti-memorization hardening that only
degrades miners who were gaming the specific fixed prompts — that
degradation is the point of the fix — or (b) a new shadow signal with
zero ranking impact. Session 2 is a genuine weight change and gets a
full 48h notice via the companion Discord post.

## References

- 2026-04-19 outage report: ``reports/2026-04-19-think-probe-disabled.md``
- Composite docstring: ``scripts/validator/composite.py``
- Composite dethrone veto: ``scripts/validator/results.py::_composite_dethrone_veto``
- Empirical counterexample: UID 18 full ``evalscope`` run, collapsing
  under realistic 8192-token budgets while still passing our KL gate.
