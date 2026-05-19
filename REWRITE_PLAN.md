# Distil — Rewrite Plan (v2) — **CUTOVER COMPLETE**

Last updated: **2026-05-19** · current state: **`distil/` IS production**.

> This document captures the historical rewrite plan. The cutover landed
> 2026-05-15 and `distil/` has been serving live traffic since. Keeping
> this file around for the rollback runbook in `deploy/cutover.md` and so
> we can refer back to the parity-snapshot evidence. New work should
> happen directly in `distil/`.

## Current state (truth)

The repo has two implementations still on disk; only one is live:

| Track | Lives in | Status | Used by prod? |
|---|---|---|---|
| **Live** | `distil/` | **Production** — all three systemd units entry-point here | **YES** — `distil-validator.service`, `distil-api.service`, `distil-dashboard.service` |
| **Legacy** | `scripts/`, `api/`, `eval/` | **Kept on disk** for tests (~832 tests still import `scripts.validator.*`) and for two narrowly-scoped shims (`api/routes/*` mounted by `distil/api/compat.py`, `eval/pod.PodManager` used by `distil/eval/pod.py`) | **NO** as a standalone validator path |

### Sizes (as of 2026-05-15)

```
scripts/                  ~44k LoC   ← prod, dominated by scripts/pod_eval_vllm.py (20,240 LoC)
api/                      ~6.9k LoC  ← prod FastAPI app (108 LoC server + 26 route modules)
eval/                     ~4.1k LoC  ← prod KL / state / model_checker utilities
frontend/src/             ~7.5k LoC  ← prod Next.js dashboard
distil/                   ~6.0k LoC  ← rewrite-v2 (WIP, NOT in pyproject)
tests/                    ~16k LoC   ← 832 tests, all targeting prod (scripts.validator.* + eval.*)
```

The `legacy/` tree (~76k LoC of decommissioned code) was deleted in **2026-05-15**. A
tarball lives at `/var/backups/distil-cleanup-2026-05-15/legacy-snapshot.tar.zst`.

## Goal

Single canonical implementation under `distil/`, packaged via `pyproject.toml`,
covered by tests, served by systemd, with `scripts/` retired.

End state target: **\~12k LoC of organized prod code** (down from ~50k+).

## Remaining gaps before cutover

> **2026-05-15 final parity snapshot:** both composite engines agree to
> within **0.13% max** on `final` when fed the same input row, on real
> prod data. The 23-axis weighted set is **byte-identical**. The 26
> "extra" axes prod records every round are all `weight=0.0` — telemetry
> only, never affecting the composite or leaderboard.
>
> **Cutting over distil end-to-end will produce composite finals that
> differ from prod end-to-end by < 1% per UID on the day of cutover**,
> well within normal between-round noise. No leaderboard-shift risk from
> the composite formula itself.

All four pre-cutover gates have landed (see commits since 2026-05-15):

1. **DQ thresholds** — DONE. `distil/eval/results.py:process_round` now
   applies the long-form derail DQ (per-hotkey permanent ban when
   coherence falls below the floor), the activation-fingerprint
   cross-round dedup with same-coldkey carve-out and commit-block
   ordering, and the composite dethrone floor via `can_dethrone()`.

2. **Resume-on-attach** — DONE. `state.current_round` records the
   in-progress round across validator restarts. `run_eval_on_pod`
   launches the orchestrator under `setsid -f` so it survives SSH
   disconnects, and `_pod_run_state` lets a restarted validator detect
   "completed" / "in-progress" runs and tail them rather than starting
   fresh.

3. **Activation-fingerprint dedup across rounds** — DONE. The new
   `_check_activation_copy()` reads `state.activation_fingerprints`
   (cross-round, with the `{model, layer_fingerprints, commit_block,
   coldkey, updated}` prod-compatible JSON shape) and applies the same
   per-layer cosine averaging + commit-block ordering prod uses.

4. **API route compat layer** — DONE. `distil/api/compat.py` loads the
   production `api/routes/*` routers in-process (with `sys.path`
   amended) and mounts them after the native distil routers. Result:
   `distil.api.server:app` now exposes **48 routes**, a superset of
   what the frontend consumes. Native distil routes win where both
   define the same path. Future per-route migrations to the cleaner
   distil pattern can happen one at a time without breaking the
   dashboard.

5. **Validator-side stage-stall watchdog** — still pending. The
   pod-side `LineStallDetector` kills hung student workers; the
   validator-side equivalent (kill the orchestrator subprocess if it
   wedges in `loading_weights` for > 25 min) is not yet ported. Low
   impact in persistent-pod mode (loading happens once, not per round)
   but still a fair completeness gap.

6. **Snapshot regression as a pytest gate** — still pending.
   `scripts/parity_check.py` runs interactively and surfaces diffs
   (current: max 0.0013 across 15 non-erroring students), but it isn't
   yet a `pytest` assertion. Useful for CI, not blocking.

## Promotion checklist (path to retire `scripts/`)

Do these in order; each is independently shippable:

### Phase A — make `distil/` real (no production impact) — **DONE 2026-05-15**

- [x] Port the **procedural item generators** from `scripts/pod_eval_vllm.py:_generate_*`
      into `distil/pod/axes/v31/`. All 11 v31 axes (math_gsm_symbolic,
      math_competition, math_robustness, code_humaneval_plus, ifeval,
      reasoning_logic_grid, reasoning_dyval, long_context_ruler, knowledge_kg,
      truthfulness, consistency) + `calibration_bench` are wired with the
      same generators + graders prod uses.
- [x] Port the **judge / long-form judge / chat-turns probes** so they run for
      real. Split-pass design: collect on student in phase 2, grade on teacher
      in phase 3 (avoids holding both engines in GPU memory).
- [ ] Port the **vLLM teacher/student lifecycle + sparse top-K KL** (mostly
      done; the `_coherence_factor` heuristic and activation fingerprint logic
      are landed but **fingerprint-history dedup** still missing).
- [ ] Port **prompt sampling + block-seed + IPT perturbations** (block-seed is
      done; IPT perturbations not yet).
- [ ] Port the **anti-finetune / fraud / DQ logic** from `scripts/validator/results.py`
      (happy path landed; per-axis DQ thresholds + `dq_history.json` migration
      not yet).
- [ ] Add `distil/` to `pyproject.toml` `packages.find.include`.
- [ ] Add tests: bring every existing `tests/test_*.py` that imports
      `scripts.validator.*` over to import from `distil.*` and make them pass.
- [ ] Add **a snapshot regression test**: feed a fixed `pod_results.json` into
      both `scripts.validator.process_results` and `distil.eval.results.process_round`
      and assert the per-student composite vectors match within a tight tolerance.

### Phase B — pod-side parity (no production impact) — **DONE 2026-05-15**

- [x] Add **parallel multi-GPU orchestration** to `distil/pod/orchestrator.py`
      (~280 LoC, mirrors `scripts/parallel_orchestrator.py` but cleaner).
- [x] Add the **stage-stall watchdog** with repeated-line detection
      (`distil/pod/watchdog.py:LineStallDetector`, replicates the diffuznik fix).
- [x] Add the **HF cache cleanup** honoring `HF_HOME` / `HF_HUB_CACHE`
      (`distil/pod/cache.py:sweep` + `clean_model`).
- [x] Add the **Phase 1 unified progress writer** so the dashboard shows multi-shard
      activity during teacher generation (orchestrator writes
      `eval_progress.json` with `shards[]` every poll).
- [x] Add **persistent-pod mode** (`distil/eval/pod.py:attach_pod` reuses an
      existing Lium pod by `DISTIL_LIUM_POD_NAME`).

### Phase C — shadow validation (no production impact)

- [ ] Stand up a **second wallet** with a non-trusted netuid stake, run
      `python -m distil.cli validate --wallet shadow` against the live chain.
- [ ] For **10 consecutive rounds**, diff the shadow validator's
      `state/composite_scores.json` vs the prod validator's. Require **byte-identical
      composite vectors** (modulo timestamps) for every shared UID.
- [ ] Fix every diff. Don't proceed to Phase D until 10 rounds in a row are clean.

### Phase D — cutover (production impact)

- [ ] Promote `distil/cli/entry.py` as the CLI; update `scripts/run_validator.sh`
      to call `python -m distil.cli validate ...` instead of
      `python scripts/remote_validator.py`.
- [ ] Switch `distil-api.service` `ExecStart` to `uvicorn distil.api.server:app`
      (the `deploy/systemd/distil-api.service` already points there).
- [ ] **Hold for one full epoch** with both validators running (shadow + prod)
      on the same pod, observing the dashboard.
- [ ] If clean: `rm -rf scripts/ api/ eval/` (after tarballing to
      `/var/backups/distil-cutover-<date>/`).

## Cleanup landed 2026-05-15 (already done)

- [x] `legacy/` deleted (~76k LoC, no callers anywhere in the repo).
- [x] `state.bak-1778765711/` (671 MB) and `state.bak-pre-publish-1778737047/`
      (420 KB) moved to `/var/backups/`.
- [x] Frontend Docs SSR `fetchKingKl` URL fixed (`/api/h2h/latest` → `/api/h2h-latest`).
- [x] Frontend `EvalProgress` / `ShardProgress` / `CompletedStudent` types
      deduplicated — single source in `frontend/src/lib/api.ts`, imported by
      `live-panel.tsx`.
- [x] `.gitignore` updated to keep `legacy/` and `state.bak-*/` from sneaking back.
- [x] `.pre-commit-config.yaml` and `deploy/` tracked in git.
- [x] `distil/` tracked in git as WIP (this file is its roadmap).
- [x] systemd unit files committed under `scripts/systemd/`.
- [x] **Security fix:** `LIUM_API_KEY` no longer visible via `ps auxf`
      (was passed as `--lium-api-key <key>` in `scripts/run_validator.sh`;
      now passed via env only — see commit `4a19eb4`).
- [x] **6 dead python files** removed from `scripts/` (~1,500 LoC, no
      callers, tarballed to `/var/backups/distil-cleanup-2026-05-15/dead-scripts/`):
      `bootstrap_weight_hashes`, `on_policy_rkl_probe`, `retroactive_probe`,
      `run_king_benchmark`, `verify_round`, `reproduce_prompts`.
- [x] **`distil/` is a real package** (`pyproject.toml` v0.9.0,
      `pip install -e .` installs it, `distil` CLI registered).
- [x] **Parity snapshot** captured (`docs/CUTOVER_PARITY_2026-05-15.md`).

## In-place cleanup of `scripts/` (parallel track to the rewrite)

Even if `distil/` never lands, the existing prod code has hot spots worth
fixing in place. Highest leverage (see `docs/CODE_REVIEW_2026-05-15.md` for the
full list):

- [ ] Decompose `scripts/validator/pod_session.py:run_eval_on_pod` (815 LoC).
      Move the 90-line bash cleanup into a versioned
      `scripts/pod/cleanup_eval.sh` file.
- [ ] Decompose `scripts/validator/results.py:process_results` (350 LoC) into
      a per-student pipeline of small functions.
- [ ] Replace `scripts/parallel_orchestrator.py:_CHALLENGERS_SPAWNED` module
      global with an `OrchestrationContext` dataclass.
- [ ] Move the watchdog from log-tail substring parsing to a structured
      per-shard heartbeat in `eval_progress.json`.
- [ ] Audit and fix the 101 `except Exception: pass` blocks in `scripts/`.
- [ ] Introduce Pydantic response models for `/api/health`, `/api/eval-progress`,
      `/api/h2h-latest`, `/api/leaderboard`.
- [ ] Split `scripts/pod_eval_vllm.py` (20,240 LoC) into a package:
      `scripts/pod_eval/{cli.py, runtime.py, benchmarks/, probes/, kl/, generators/}`.
      Keep `scripts/pod_eval_vllm.py` as a thin shim (`from pod_eval.cli import main`).
