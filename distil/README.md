# `distil/` — Rewrite-v2 (WIP — not in production)

This package is the **clean architectural sketch** of the Distil SN97 validator.
It is **not** what runs in production today — see `scripts/validator/`,
`scripts/pod_eval_vllm.py`, and `api/server.py` for the live implementation.

## Layout

```
distil/
├─ settings.py             Typed env-backed config
├─ state/                  Atomic JSON state + migration helpers
├─ chain/                  Bittensor: metagraph, commitments, set_weights
├─ eval/                   Validator-side: round spec, pod orchestration, composite
├─ pod/                    GPU-pod runner: axes, probes, KL, vLLM lifecycle
├─ api/                    FastAPI v2 dashboard (`app = create_app()`)
├─ chat_pod/               Kimi/vLLM bootstrap for the chat endpoint
└─ cli/                    Unified Click CLI (`distil validate`, `distil api`, …)
```

## Status (2026-05-15)

| Module | Status | Notes |
|---|---|---|
| `distil/eval/service.py` (177 LoC) | **Skeleton** | Single-loop validator; no resume-on-attach, no DQ migrations, no announcements wiring |
| `distil/eval/pod.py` (131 LoC) | **Skeleton** | `acquire_pod` creates+terminates per round; prod uses a persistent 8×B200 pod |
| `distil/eval/composite.py` (452 LoC) | **Likely-equivalent** | Clean refactor of the prod composite math; needs a snapshot regression test |
| `distil/eval/results.py` (212 LoC) | **Likely-equivalent** | Same shape as prod; needs side-by-side scoring diff |
| `distil/pod/__main__.py` (285 LoC) | **PARTIALLY STUBBED** | `judge_probe`, `long_form_judge_probe`, `chat_turns_probe` are hardcoded to `{"n": 0, "n_valid": 0}` — those composite axes drop out |
| `distil/pod/axes/*.py` | **PARTIALLY STUBBED** | e.g. `math_gsm.py` has 1 template vs prod's 832-LoC procedural generator |
| `distil/api/server.py` (82 LoC) | **Skeleton** | Routes only stub a subset of `api/routes/*.py` |
| `distil/cli/entry.py` (105 LoC) | **Working** | `distil validate --once`, `distil api`, etc. |

## Why this is checked in (rather than deleted)

It is the **target** for a future clean prod stack. The roadmap to make it
production-ready lives in `/REWRITE_PLAN.md`. Don't import from this package
in production code paths until that roadmap is complete.

## Running it locally (no chain side effects)

```bash
# Validator dry-run (does not call set_weights):
python -m distil.cli validate --once --dry-run

# Pod-side eval against a local round_spec.json:
python -m distil.pod ./round_spec.json --out /tmp/results.json

# FastAPI app:
uvicorn distil.api.server:app --port 3711
```

## Not used by

- `distil-validator.service` (prod) → `scripts/run_validator.sh` → `scripts/remote_validator.py` → `scripts.validator.service.run_validator`
- `distil-api.service` (prod, mounted from `scripts/systemd/`) → `uvicorn server:app` in `api/`
- Any of the 832 tests under `tests/`

`deploy/systemd/distil-api.service` does target `uvicorn distil.api.server:app` —
but that unit file is NOT the one systemd is loading on this host.
