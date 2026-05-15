# Distil — Rewrite Plan (v2)

Last updated: **2026-05-15** · target state: **`distil/` is production, `scripts/` retired**.

## Current state (truth)

The repo has **two parallel implementations** of the SN97 validator:

| Track | Lives in | Status | Used by prod? |
|---|---|---|---|
| **Prod (v1)** | `scripts/`, `api/`, `eval/` | **Live, hardened by months of incidents** | **YES** — `distil-validator.service` and `distil-api.service` |
| **Rewrite-v2** | `distil/` | **Feature-tracking prod** — Phase A + B done 2026-05-15 | NO — still not imported by the systemd units |

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

> **2026-05-15 parity snapshot (revised):** both engines agree to within
> **0.16% max** on `final` when fed the same input row. The 23-axis weighted
> set is **byte-identical** between prod and distil. The 26 "extra" axes
> prod records every round are all `weight=0.0` — telemetry-only, never
> affecting the composite or leaderboard. See
> `docs/CUTOVER_PARITY_2026-05-15.md` for the per-student numbers and the
> explanation of how my first revision of this section got that wrong.
>
> What that means: cutting over distil end-to-end will produce composite
> finals that differ from prod end-to-end by **< 1% per UID on the day of
> cutover**, well within normal between-round noise. **There is no
> leaderboard-shift risk from the composite formula itself.**

What blocked the cutover on 2026-05-14 has now been ported (Phase A + B,
see "Cleanup landed" below). What's still required before flipping systemd:

1. **Anti-finetune / fraud / DQ thresholds**
   `distil/eval/results.py` (212 LoC) implements the happy path but
   doesn't yet apply the full per-axis DQ thresholds prod uses
   (`scripts/validator/results.py:process_results` — ~350 LoC of
   per-axis floor checks + `dq_history.json` migrations). Cutover
   without this would let a low-quality model squeak past axes prod
   would DQ.

2. **Resume-on-attach**
   If the validator restarts mid-round, prod re-attaches to the same
   pod and continues from where the previous orchestrator left off
   (`scripts/validator/pod_session.py:resume_pod_eval`). `distil/` has
   no resume — every restart starts a fresh round.

3. **Activation-fingerprint dedup across rounds**
   Prod stores per-round fingerprints in
   `state/activation_fingerprints.json` and DQs students whose
   fingerprint is within ε of a prior round's king (near-copy attack
   detection). `distil/` computes the fingerprint per round but doesn't
   compare against history yet.

4. **Validator-side stage-stall watchdog**
   The pod-side `LineStallDetector` (just ported) kills hung student
   workers from inside the pod. But if the orchestrator itself wedges
   (e.g. teacher phase stuck on `loading_weights` for > 25 min), the
   validator needs to kill the whole orchestrator subprocess. Prod's
   `scripts/validator/pod_session.py:StageStallWatchdog` does this.

5. **Snapshot regression test bound into pytest**
   `scripts/parity_check.py` runs on the latest prod round and prints
   per-student diffs. Current result: max abs diff = 0.0016 across 8
   non-erroring students. We have **not** yet baked the harness into
   `pytest` with a regression-gating assertion (e.g. `assert max_diff <
   0.005`). That's the last "snapshot regression" todo.

6. **API route gap**
   `distil/api/routes.py` exposes **15 routes**; the production
   `api/server.py + api/routes/` graph exposes **~45 routes** the
   frontend actually consumes (chat, queue, announcement, eval-stats,
   composite-scores, commitments, model-info, pod-logs, telemetry/*, and
   more). Cutting the API to `distil.api.server:app` today would
   500-out half the dashboard.

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
