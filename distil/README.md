# `distil/` — Rewrite-v2 (parity-tracking)

This package is the **clean architectural target** of the Distil SN97 validator.
The live `distil-validator.service` still calls into `scripts/validator/` and
`scripts/pod_eval_vllm.py` — but the gap to `distil/` is now feature-level,
not architecture-level. Cutover plan in `REWRITE_PLAN.md`.

## Layout

```
distil/
├─ settings.py             Typed env-backed config (Pydantic)
├─ state/                  Atomic JSON state + migration helpers
├─ chain/                  Bittensor: metagraph, commitments, set_weights
├─ eval/                   Validator-side: round spec, pod orchestration, composite, king
├─ pod/                    GPU-pod runner
│  ├─ __main__.py          3-phase entrypoint: --phase teacher|students|judge|all
│  ├─ orchestrator.py      Multi-GPU fan-out across N shards
│  ├─ cache.py             HF_HUB_CACHE-aware sweep + per-student broom
│  ├─ watchdog.py          WallClock + LineStallDetector + cuda_alive
│  ├─ axes/                12 procedural axes (11 v31 + calibration_bench)
│  │  ├─ v31/              Verbatim copy of scripts/v31/*.py (generator + grader per axis)
│  │  ├─ _math.py          Shared boxed{} / #### N answer extractor
│  │  ├─ _runner.py        Generic generate-and-score loop
│  │  └─ <axis>.py         ~30-LoC thin wrapper per axis
│  └─ probes/              judge / long_form_judge / chat_turns (split: collect on
│                          student in phase 2, grade on teacher in phase 3)
├─ api/                    FastAPI v2 dashboard
├─ chat_pod/               Kimi/vLLM bootstrap for the chat endpoint
└─ cli/                    Unified Click CLI
```

## Status (2026-05-15)

| Module | Status | Notes |
|---|---|---|
| `distil/eval/service.py` (190 LoC) | **Functional** | Persistent-pod mode via `attach_pod`. No resume-on-attach yet (skip on round restart). |
| `distil/eval/pod.py` (~180 LoC) | **Functional** | Both `acquire_pod` (ephemeral) and `attach_pod` (persistent, prod default). |
| `distil/eval/composite.py` (452 LoC) | **Needs parity test** | Composite math looks equivalent; needs snapshot regression vs `scripts.validator.composite`. |
| `distil/eval/results.py` (212 LoC) | **Needs parity test** | Same shape as prod; needs side-by-side scoring diff. |
| `distil/pod/__main__.py` | **Functional** | 3-phase (teacher → students → judge) with --shard support. |
| `distil/pod/orchestrator.py` | **Functional** | Spawns one shard per GPU, line-stall watchdog, unified progress. |
| `distil/pod/cache.py` | **Functional** | Same HF_HUB_CACHE resolution as prod's hardened sweeper. |
| `distil/pod/watchdog.py` | **Functional** | LineStallDetector replicates the diffuznik fix. |
| `distil/pod/axes/*.py` (12 axes) | **Functional** | All 11 v31 procedural axes + calibration_bench wired with prod's generators + graders. |
| `distil/pod/probes/*.py` | **Functional** | judge / long_form_judge / chat_turns now run for real (was hardcoded to `n=0`). |
| `distil/api/server.py` (82 LoC) | **Skeleton** | A subset of `api/routes/*.py` — needs full port for cutover. |
| `distil/cli/entry.py` (105 LoC) | **Working** | `distil validate --once`, `distil api`, etc. |

## Not yet ported

- Full anti-finetune / fraud / DQ logic from `scripts/validator/results.py`
  (`distil/eval/results.py` has the happy path; DQ flags wired but the
  per-axis DQ thresholds need a sweep against `state/dq_history.json`).
- Resume-on-attach (if a previous round's `eval_progress.json` survives
  on the pod and shows incomplete shards, reuse them).
- Activation fingerprint dedup across rounds (prod stores per-round
  fingerprints and refuses near-duplicates).
- Stage-stall watchdog at the validator side
  (`scripts/validator/pod_session.py:StageStallWatchdog`). The pod-side
  `LineStallDetector` covers most of this, but the validator should
  still time out a pod that's stuck in `loading_weights` for > 25 min.

## Used by

- `deploy/systemd/distil-api.service` targets `uvicorn distil.api.server:app`,
  but that unit isn't the one systemd loads today.
- All 848 tests under `tests/` still pass with this package present —
  no shared-import collisions.

## Not used by

- `distil-validator.service` (prod) → `scripts/run_validator.sh` →
  `scripts/remote_validator.py` → `scripts.validator.service.run_validator`.
- `distil-api.service` (prod, mounted from `scripts/systemd/`) →
  `uvicorn server:app` in `api/`.

## Running locally

```bash
# Validator dry-run (no chain side effects):
python -m distil.cli validate --once --dry-run

# Pod-side eval against a local round_spec.json (single GPU):
python -m distil.pod ./round_spec.json --out /tmp/results.json

# Multi-GPU orchestrator (assumes you're on the pod):
python -m distil.pod.orchestrator ./round_spec.json \
    --workdir /tmp/round \
    --out /tmp/round/results.json \
    --progress /tmp/round/eval_progress.json \
    --n-gpus 8

# FastAPI app:
uvicorn distil.api.server:app --port 3711
```
