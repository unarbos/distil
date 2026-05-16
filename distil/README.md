# `distil/` — production validator + API + pod-side eval

This package is the **live** Distil SN97 stack. As of 2026-05-15, all three
systemd-managed services on the validator host are entry-pointed here:

| systemd unit | entry point |
|---|---|
| `distil-validator.service` | `distil validate --wallet-name … --hotkey-name …` (CLI → `distil/eval/service.py`) |
| `distil-api.service`       | `uvicorn distil.api.server:app` |
| `distil-dashboard.service` | `next start` on `frontend/` (talks to `distil-api`) |

The legacy `scripts/` + `api/` + `eval/` packages are kept on disk for two
narrowly-scoped reasons (both documented in code where they cross the
boundary):

1.  `distil/api/compat.py` re-mounts the ~3000 LoC of proven prod route
    business logic from `api/routes/*.py` underneath the distil FastAPI
    app so the dashboard frontend keeps working byte-identically.
2.  `distil/eval/pod.py` re-uses `eval/pod.PodManager` (364 LoC of
    battle-tested paramiko + SFTP + lium SDK plumbing, including the
    B200 grouped_mm patches and the Kimi-K2.6 transformers-5.x compat
    patches) instead of re-implementing SSH transport.

Both shims are clearly marked and can be inlined once we want to drop the
legacy directories. They were intentional zero-regression carve-outs for
the cutover; the rest of the validator and API logic is native to
`distil/`.

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
│  ├─ teacher_api.py       Cloud-teacher (OpenRouter / Kimi-K2.6) logprob harvester
│  ├─ axes/                12 procedural axes (11 v31 + calibration_bench)
│  │  ├─ v31/              Procedural axis generators (verbatim port of legacy scripts/v31/*.py)
│  │  ├─ _math.py          Shared boxed{} / #### N answer extractor
│  │  ├─ _runner.py        Generic generate-and-score loop
│  │  └─ <axis>.py         ~30-LoC thin wrapper per axis
│  └─ probes/              judge / long_form_judge / chat_turns (split: collect on
│                          student in phase 2, grade on teacher in phase 3)
├─ api/                    FastAPI app (native distil routes + compat-mounted prod routers)
├─ chat_pod/               Kimi/vLLM bootstrap for the chat endpoint
└─ cli/                    Unified Click CLI (`distil validate`, `distil api`, `distil pod`)
```

## Running

```bash
# Validator (production unit: `systemctl start distil-validator.service`).
distil validate --wallet-name affine --hotkey-name validator

# Validator dry-run (no chain side effects):
distil validate --once --dry-run

# Pod-side eval against a local round_spec.json (single GPU):
python -m distil.pod ./round_spec.json --out /tmp/results.json

# Multi-GPU orchestrator (on the pod):
python -m distil.pod.orchestrator ./round_spec.json \
    --workdir /tmp/round \
    --out /tmp/round/results.json \
    --progress /tmp/round/eval_progress.json \
    --n-gpus 8

# FastAPI app:
uvicorn distil.api.server:app --port 3710
```

## Testing

The full test suite (`pytest tests/`) covers the cutover plus the legacy
parity harnesses (see `tests/test_parity_*.py`).

```bash
pytest -q tests/
```
