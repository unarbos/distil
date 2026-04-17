# Off-Policy CoT Collapse — Diagnosis & Plan

Raised by `@allan_ww` in #distil-97 on 2026-04-17. Corroborated by bench data
collected the same day.

## Diagnosis

On the same single-turn prompt `"Hi"`:

| model | finish | completion tokens | thinking chars | final answer |
|---|---|---|---|---|
| baseline `Qwen/Qwen3.5-4B` | `stop` | 104 | 407 | `"Hello! How can I assist you today?"` |
| king `gtensorapp/prime-dusk-4260` (UID 107) | `length` (4096 cap) | 4096 | 14 396 | *none* (never emits content) |

The king's thinking block contains the 6-word phrase `I'll write:* "Hello! How
are you` repeated 102 times and `Is there anything I can help` 91 times before
the generator is cut off. A fresh probe across benchmark-style prompts shows
the king is **10–80×** slower than baseline on every task (`Hi`: 266s vs 11s;
`arc`: 206s vs 3s).

The full published benchmark numbers (run with `enable_thinking=False`, so
the loop is suppressed) confirm the underlying weights are damaged:

| benchmark | king | baseline | delta |
|---|---|---|---|
| gsm8k | 0.893 | 0.934 | −4.1pp |
| ifeval | 0.673 | 0.826 | −15.3pp |
| humaneval | 0.695 | 0.817 | −12.2pp |
| bbh | 0.716 | 0.879 | −16.3pp |
| arc | 0.881 | 0.901 | −2.0pp |

The king — picked by the scoreboard as the top model in the subnet — is
**strictly worse than the untrained base model on 5 / 5 reasoning benchmarks**.

## Root cause

Two independently-published failure modes stack:

1. **Off-policy teacher forcing**
   ([thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/))
   Miners train students on teacher rollouts with token-level CE / KL loss.
   The student is only ever exposed to the teacher's on-distribution context,
   so the moment it samples anything the teacher wouldn't have said, compounding
   error takes it off the manifold and it has no recovery mechanism.

2. **CoT-complexity mismatch between teacher and 4B student**
   ([arXiv:2502.07266](https://arxiv.org/abs/2502.07266))
   A 4B model lacks the capacity to execute the 35B teacher's multi-step
   "Wait, let me reconsider" reasoning pattern. It learns the cosmetic surface
   form (think longer → lower KL loss) without the underlying reasoning,
   so the thinking block becomes filler.

Our KL scoring rewards surface-form match on prefix-forced continuations; it
provides no gradient toward "can the student stop". The anti-finetune probe
catches norm inflation but not generation pathology. The benchmark runner
disables thinking (`enable_thinking=false`) so the collapse is invisible to
the public leaderboard.

## Plan

### Short term — ship this week

- [x] **`thinking_collapse_probe`** in `scripts/pod_eval_vllm.py`: runs three
  trivial prompts (`"Hi"`, `"largest planet one word"`, `"say the word: done"`)
  with `enable_thinking=True`, greedy, 1024-token budget. Flags a model as
  `thinking_collapse` if any 6-word phrase repeats ≥ 15× on any prompt, or
  if fewer than 2/3 of the prompts reach EOS without looping. Set
  `kl_global_avg = inf` so the model never wins H2H.
- [x] **`retroactive_probe.py`** extended to run `thinking_collapse_probe`
  alongside `finetunability_probe` so we can sweep the last 3 rounds of
  kings and challengers and retroactively disqualify the ones that collapse.
- [x] **Publish the pathology** on the docs tab (`#chat-collapse`) with links
  to the two papers, the exact probe thresholds, and a recommended self-test
  miners can run before submitting.
- [ ] **Retroactive DQ sweep** — run `UIDS=107,125,157,101 retroactive_probe.py`
  on the eval pod and feed the failures into `state/disqualified.json`.
- [ ] **Benchmarks with thinking on** — add a second column to the benchmarks
  tab showing the same tasks with `enable_thinking=True`, temperature 0.7. This
  is the "real chat" column and will make collapse obvious from the dashboard.

### Medium term — next 2 weeks

- [ ] **Add a generation-quality signal to H2H scoring**. Right now H2H is pure
  token-level KL under teacher-forcing. Add a second term: for 16 prompts per
  round, let the student free-generate 256 tokens with thinking on; compute
  teacher-vs-student KL *over the student's own continuation*. This is the
  canonical on-policy distillation loss from the Thinking Machines post and
  gives a gradient against compounding error.
- [ ] **Per-task thinking budgets**. 2502.07266 shows the optimal CoT length
  varies by task and scales with model size. Rather than letting the student
  decide, pin the thinking budget per benchmark prompt class (e.g. 128 tokens
  for arc, 512 for gsm8k) and compare against the teacher's output truncated
  to the same budget.
- [ ] **Publish a baseline SFT/distillation training recipe** so miners aren't
  reinventing it. arxiv.org/html/2604.13016 (cold-start with offline distill
  then on-policy RL) is a reasonable template; a minimal reference trainer
  would keep the population from re-deriving the same CoT-collapse bug.

### Long term — architectural

- [ ] **On-policy distillation loop**. Validator does `N_policy_samples` of
  student free-generation per prompt, runs them through the teacher for
  teacher-reference logprobs, and scores student on the KL of its own samples.
  Non-deterministic, more expensive, but directly optimizes what we care about
  (the student generating coherent answers at chat.arbos.life). We can keep
  off-policy KL as a cheap first-pass filter and only run on-policy for the
  top-N contenders.

- [ ] **Mixed-objective reward**: on-policy KL (70%) + off-policy teacher-forced
  KL (20%) + answer-quality reward-model score on free generations (10%). The
  last term is what prevents models from just learning to emit EOS instantly.

## Credits

- `@allan_ww` — original diagnosis and the two paper references (#distil-97, 2026-04-17)
- `@mianzaz` — independent confirmation that "almost all models" exhibit the
  same failure on #distil-97 earlier the same day
- `@fishandconst` — chat.arbos.life reproduction link
