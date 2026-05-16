"""Regression: pod orchestrator must signal completion via a SENTINEL
(``results.done``) — not via the existence of ``results.json``.

The bug we are guarding against: ``distil.pod.orchestrator`` writes
``results.json`` TWICE per round — once after Phase 2 (parallel student
shards merged) and a second time after Phase 3 (judge / long_form_judge
/ chat_turns_probe grading via the cloud teacher API). Between those
two writes there is a ~30-60 second window where ``results.json``
exists but is **missing every Phase 3 axis**.

The validator host polls the pod every 20 s. If ``_pod_run_state``
detects completion by ``results.json`` existence alone, it will race
into that window, download the partial file, and silently lose every
judge axis for the round — exactly what happened on round
1778903632 (king + every challenger had judge_probe/long_form_judge/
chat_turns_probe = None even though the pod-side results.json had
them populated).

These tests pin three invariants:

1. The orchestrator writes ``results.done`` ONLY after Phase 3 finishes.
2. ``_pod_run_state`` gates ``complete`` on ``results.done``, not on
   ``results.json``.
3. The orchestrator's last action of the success path is the sentinel
   write (so a crash between Phase 3 and the sentinel is observable as
   "stale", not as "complete with truncated results.json").
"""

from __future__ import annotations

from pathlib import Path

import distil.eval.pod as eval_pod
import distil.pod.orchestrator as orch


def test_orchestrator_writes_results_done_sentinel_after_phase3():
    """The ``run`` function must touch ``results.done`` only after
    Phase 3 (judge) succeeds. Reading the source is enough — exercising
    the full flow requires a real pod + spec, which is out of scope
    for unit tests.
    """
    src = Path(orch.__file__).read_text()

    # Sentinel exists and is named ``results.done``.
    assert 'results.done' in src, (
        "orchestrator must write a ``results.done`` sentinel file to "
        "signal Phase 3 completion to the host"
    )
    # Sentinel is written AFTER the phase-3 spawn (judge grading).
    phase3_idx = src.index('phase="judge"')
    sentinel_idx = src.index('results.done')
    assert sentinel_idx > phase3_idx, (
        "sentinel must be written AFTER Phase 3 spawn — otherwise the "
        "host can observe completion before judge scores are merged"
    )
    # Sentinel write is guarded by phase-3 non-zero return-code early
    # exit (we should NOT write it on failure).
    pre_sentinel = src[phase3_idx:sentinel_idx]
    assert 'return rc' in pre_sentinel, (
        "phase-3 failure must short-circuit BEFORE the sentinel write "
        "so a failed Phase 3 is observable as 'stale', not 'complete'"
    )


def test_pod_run_state_gates_on_results_done_not_results_json():
    """``_pod_run_state`` must check ``results.done`` for completion."""
    src = Path(eval_pod.__file__).read_text()
    func = src[src.index('def _pod_run_state'):src.index('def _pod_run_state') + 2000]

    assert 'results.done' in func, (
        "_pod_run_state must check for results.done to detect completion"
    )
    # Critically: results.json existence should NOT be the completion
    # trigger. We assert this by checking that the shell condition
    # tests results.done, not results.json.
    assert '[ -f {remote_run}/results.done ]' in func, (
        "completion check must be ``[ -f $remote_run/results.done ]`` "
        "— do NOT regress to checking results.json (race vs Phase 3)"
    )
    assert '[ -f {remote_run}/results.json ]' not in func, (
        "results.json existence is NOT a completion signal (Phase 2 "
        "writes it before Phase 3 merges judge axes) — use results.done"
    )
