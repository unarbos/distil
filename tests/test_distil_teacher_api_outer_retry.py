"""Regression: ``generate_continuations_api`` must survive sustained 429
storms via a second-tier outer retry loop.

The bug we are guarding against: round 1778905724 (2026-05-16) crashed
in Phase 1 (teacher) because OpenRouter's Inceptron route was
serving Kimi-K2.6 with a sustained 429 throttling pattern. The
per-prompt 6-attempt × ~30 s exp-backoff was exhausted within ~60 s
per prompt, the future raised ``HTTPError: 429``, the
``ThreadPoolExecutor.as_completed`` propagated it to the caller,
``_phase_teacher`` died with rc=1, and the whole round was lost.

The legacy stack solved this in ``scripts/api_teacher.generate_via_api``
with a two-tier retry: inner per-prompt exp-backoff, plus outer
sequential retry passes (5 passes × 60s-300s cool-downs) on any
prompts that survived the main fan-out without a result. This pins
that behaviour in the ``distil/pod/teacher_api`` port.
"""

from __future__ import annotations

import re
from pathlib import Path

import distil.pod.teacher_api as teacher_api


def test_outer_retry_constants_are_defined():
    """Outer retry must be configured with non-trivial cooldowns."""
    assert hasattr(teacher_api, "_OUTER_RETRY_PASSES"), (
        "teacher_api must define _OUTER_RETRY_PASSES (legacy used 5)"
    )
    assert teacher_api._OUTER_RETRY_PASSES >= 3, (
        f"outer retry passes ({teacher_api._OUTER_RETRY_PASSES}) too few; "
        f"legacy uses 5 to survive sustained 429 storms"
    )

    assert hasattr(teacher_api, "_OUTER_RETRY_COOLDOWNS_S"), (
        "teacher_api must define _OUTER_RETRY_COOLDOWNS_S (legacy used 60,90,120,180,300)"
    )
    cooldowns = teacher_api._OUTER_RETRY_COOLDOWNS_S
    assert len(cooldowns) >= teacher_api._OUTER_RETRY_PASSES - 1, (
        "cooldown list must cover at least N-1 inter-pass gaps"
    )
    assert all(c >= 30 for c in cooldowns), (
        "every cooldown must be ≥30s to let the 429 storm subside"
    )
    assert max(cooldowns) >= 180, (
        "max cooldown must be ≥180s — peak Inceptron throttle windows "
        "have been observed to last >2 minutes"
    )


def test_generate_continuations_api_implements_outer_retry():
    """Source inspection: the outer retry loop must be in place AND
    catch per-prompt exceptions rather than letting fut.result() raise."""
    src = Path(teacher_api.__file__).read_text()

    # The fan-out must NOT let a single future failure tear down the
    # whole batch — otherwise the outer retry can never collect.
    fanout = src[src.index("with ThreadPoolExecutor"):src.index("# Outer retry pass")]
    assert "fut.result()" in fanout
    assert "except Exception" in fanout, (
        "main fan-out must catch per-prompt exceptions (deferring to "
        "the outer retry loop) — otherwise ThreadPoolExecutor will "
        "propagate the first 429-storm failure and kill the round"
    )
    assert "failed.append" in fanout, (
        "main fan-out must collect failed prompt indices for the "
        "outer retry pass"
    )

    # The outer retry loop must exist and use the cooldowns.
    outer = src[src.index("# Outer retry pass"):]
    assert "_OUTER_RETRY_PASSES" in outer
    assert "_OUTER_RETRY_COOLDOWNS_S" in outer
    assert "_call_with_retry" in outer, (
        "outer retry must re-invoke the inner retry function so each "
        "outer attempt still benefits from per-prompt exp-backoff"
    )


def test_pod_polling_loop_fails_fast_on_dead_orchestrator():
    """The host-side polling loop must NOT keep polling for the full
    deadline_s when the pod-side orchestrator has crashed. Otherwise
    a Phase-1 failure stalls the validator for ~60 minutes before
    the deadline raises."""
    import distil.eval.pod as eval_pod

    src = Path(eval_pod.__file__).read_text()
    # Locate the in-flight polling loop (NOT the resume one).
    fresh_loop = src[src.index("orchestrator launch failed"):]
    # Must check for absent/stale state, and must have a startup-grace
    # window so brief pgrep races don't false-positive.
    assert "startup_grace" in fresh_loop, (
        "polling loop must include a startup grace window so initial "
        "pgrep races during setsid+fork don't trigger a false abort"
    )
    assert 'state == "absent"' in fresh_loop, (
        "polling loop must detect a dead orchestrator (state=='absent') "
        "and raise instead of waiting the full deadline"
    )
    assert re.search(r"orchestrator died before producing", fresh_loop), (
        "the failure message must clearly indicate a crashed "
        "orchestrator (helps operators triage from logs)"
    )
