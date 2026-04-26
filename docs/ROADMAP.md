# Distil SN97 — Roadmap

Last updated: 2026-04-26

## ✅ Recently Shipped

- **Single-eval policy (Session 3.7, live 2026-04-25)** — One commitment, one eval. King not re-evaluated. Cross-round dethronement on absolute composite-worst (margin 3%, saturated-floor tiebreaker on weighted)
- **Arena v3.7 composite (20+ axes)** — KL + on-policy RKL + capability + length + judge probe + 15 absolute benches (math, code, reasoning, knowledge, ifeval, aime, mbpp, tool_use, self_consistency, arc, truthful, long_context, procedural, robustness, noise_resistance) + chat-turns probe + reasoning-density. `_KING_SELECTION_MIN_AXES = 17` ensures schema-fair king comparison.
- **Asymmetric reference-broken-axes filter** — `worst()` drops axes the base reference itself fails (e.g. AIME under 768-token cap), `weighted` keeps them so beating a broken-on-the-reference axis still pays
- **Long-context confuser needles** — `long_context_bench` was 100% saturated at 1.0 ("dead axis"); confuser-needle templates now force true question-needle matching
- **Eval speed** — Sub-1hr rounds achieved (Round 7: 18.2 min, Round 8: ~14 min on H200 NVL)
- **Anti-exploit** — Pre-dethronement integrity checks, shard-invariant hashing, copy detection (cosine threshold 0.99999, same-coldkey carve-out)
- **Transparency** — `/api/evaluated_uids`, `/api/dq_reasons`, `/api/model_hashes`, `/api/eval-stats`, `/api/compare`, `/api/miner/{uid}` with `composite.broken_axes`
- **Dashboard** — Live eval progress, per-miner detail page (`/miner/{uid}`), reference baseline (Qwen3.5-4B), broken-axes strikethrough rendering
- **Performance** — Concurrent teacher generation, vLLM student scoring, lium-based pod resume on validator restart
- **Documentation** — Miner FAQ, MINER_FAQ.md, CHANGELOG.md, API transparency endpoints

## 🔄 In Progress

- **Sample-count tuning per axis** — Round-3 speed cuts left some bench axes (self_consistency, procedural, robustness, noise) under-sampled (n≤4), causing severe quantization. Increasing per-round sample counts now that the round is well under the 60-min target.
- **Discord bot accuracy** — Mirror-source docs lagging behind single-eval policy was misinforming the bot; in-progress sync.
- **Reasoning density / length penalty** — Composite axis live; tuning vs base-model rambling baseline

## 📋 Planned

- **Local eval guide** — Step-by-step for miners to reproduce scores locally
- **Training starter code** — Example distillation script with recommended hyperparameters
- **Eval data explorer** — Browse prompts, completions, and per-prompt KL on the dashboard
- **Historical leaderboard** — Track miner ranking over time, not just current snapshot
- **Multi-teacher** — Evaluate against multiple teachers for robustness (research phase)

## 💡 Under Consideration

- **Partial credit** — Proportional emissions instead of winner-takes-all (needs community input)
- **Dynamic prompt selection** — Weight harder prompts more heavily
- **Model quality metrics** — Benchmark scores alongside KL for a holistic view

---

Have ideas? Drop them in the `ა・distil・97` channel.
