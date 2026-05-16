"""Regression tests for the 2026-05-16 pre-merge audit batch.

Pins seven fixes (in priority order):

* **F1.1** (HIGH) — Activation fingerprint MUST return a dict that the
  host's ``_check_activation_copy`` recognizes; previously a flat
  ``list[float]`` silently disabled copy-detection DQ.
* **F1.2** (HIGH) — ``is_dethrone`` MUST veto when the challenger's
  worst axis is below ``composite_dethrone_floor``; previously a
  ``can_dethrone`` helper held this guard but was dead code.
* **F3.1** (HIGH) — ``/api/miner/{uid}`` MUST look up composite by
  UID-key, not by a non-existent ``uid`` field on the value.
* **F2.1** (MEDIUM) — Per-shard ``raw.json`` write MUST be atomic to
  avoid dropping judge / long-form / chat-turns axes on crash.
* **F1.3** (MEDIUM) — ``state.save()`` MUST persist ``h2h_history``
  so post-``resolve_king`` dethrone metadata is durable.
* **F2.2** (MEDIUM) — Host's env whitelist and pod's reader MUST use
  the same name for the watchdog repeat-window variable.
* **F1.4** (LOW) — ``service._round`` MUST pass ``reference_repo`` as
  ``reference_name=`` and ``teacher_repo`` as ``teacher_name=``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import distil.api.routes as api_routes
import distil.eval.composite as composite
import distil.eval.pod as eval_pod
import distil.eval.service as service
import distil.pod.__main__ as pod_main
import distil.pod.probes.activation_fingerprint as actfp
import distil.state.files as state_files
from distil.settings import settings


# ── F1.1: activation fingerprint schema ────────────────────────────


def test_activation_fingerprint_run_returns_dict_with_layer_fingerprints():
    """The host's ``_check_activation_copy`` gates DQ behind
    ``isinstance(fingerprint, dict) and fingerprint.get('layer_fingerprints')``.
    The pod's ``run()`` must satisfy both guards."""
    src = Path(actfp.__file__).read_text()
    assert "-> dict" in src, "run() return type must be dict, not list[float]"
    assert '"layer_fingerprints"' in src, (
        "run() output must include a 'layer_fingerprints' key — without "
        "it the host's DQ guard returns False unconditionally and the "
        "copy-detection feature is silently disabled"
    )
    # Sanity: the projection helper still produces a unit vector.
    vec = actfp._project([1, 2, 3, 4, 5], 8)
    assert len(vec) == 8
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9, "fingerprint vector must be unit-norm"


# ── F1.2: dethrone floor guard ─────────────────────────────────────


def test_is_dethrone_vetoes_when_challenger_worst_below_floor():
    """A challenger with great ``final`` but a single terrible axis
    must NOT dethrone — even when the margin gate would otherwise pass."""
    margin = settings.composite_dethrone_margin
    floor = settings.composite_dethrone_floor

    challenger = {
        "final": 0.99,
        "worst": floor - 0.01,  # below floor
        "present_count": 25,
    }
    king = {"final": 0.5, "worst": 0.4, "present_count": 25}
    do, why = composite.is_dethrone(challenger, king, margin=margin)
    assert do is False, (
        f"challenger with worst={floor - 0.01:.3f} (below floor "
        f"{floor:.2f}) must NOT dethrone — got {why}"
    )
    assert "worst_below_floor" in why


def test_is_dethrone_allows_when_challenger_worst_at_floor():
    """Worst at exactly ``floor`` should NOT be vetoed (strict less-than)."""
    margin = settings.composite_dethrone_margin
    floor = settings.composite_dethrone_floor
    challenger = {
        "final": 0.99,
        "worst": floor,  # exactly at floor
        "present_count": 25,
    }
    king = {"final": 0.5, "worst": 0.4, "present_count": 25}
    do, _why = composite.is_dethrone(challenger, king, margin=margin)
    assert do is True, "challenger at floor (not below) should be allowed to dethrone"


def test_is_dethrone_skips_floor_check_when_worst_is_none():
    """``worst is None`` means we couldn't compute the worst axis (e.g.
    Phase 2 produced no KL signal). Don't apply the floor in that case
    — the rest of the gate (sparse, margin) handles it."""
    challenger = {"final": 0.99, "worst": None, "present_count": 25}
    king = {"final": 0.5, "worst": 0.4, "present_count": 25}
    do, _why = composite.is_dethrone(challenger, king)
    assert do is True, "missing worst should not trigger the floor veto"


# ── F3.1: /api/miner/{uid} composite lookup ────────────────────────


def test_miner_route_composite_lookup_uses_uid_key():
    """The lookup MUST be by UID-string key, not by a non-existent
    ``uid`` field on the value."""
    src = Path(api_routes.__file__).read_text()
    # Locate the miner-detail route.
    func = src[src.index("def miner_detail"):src.index("def telemetry_timings")]
    assert "composites.get(str(uid))" in func, (
        "miner route must do a UID-keyed lookup; pre-fix code used "
        "(c for k, c in composites.items() if c.get('uid') == int(uid)) "
        "which always returned None"
    )


# ── F2.1: atomic raw.json write ────────────────────────────────────


def test_phase_students_raw_json_write_is_atomic():
    """The shard raw-json side-car must use the same tmp+rename pattern
    as ``_flush_partial`` — a crash mid-write otherwise destroys 75%
    of the composite weight (judge + long_form + long_gen + chat_turns)
    for every student in the shard."""
    src = Path(pod_main.__file__).read_text()
    func = src[src.index("args.phase == \"students\""):src.index("# ── Phase 3:")]
    assert ".tmp" in func, (
        "raw.json write must go through a .tmp file before rename"
    )
    assert "os.replace" in func, (
        "raw.json write must use os.replace for atomic rename"
    )


# ── F1.3: h2h_history persisted on save ────────────────────────────


def test_state_save_persists_h2h_history():
    """``state.save()`` must write ``h2h_history.json`` so post-
    ``resolve_king`` mutations to the last row (king_after, king_changed,
    prev_king_uid, new_king_uid, dethrone_method) are durable."""
    src = Path(state_files.__file__).read_text()
    save_fn = src[src.index("def save"):src.index("def append_round")]
    assert "H2H_HISTORY_FILE" in save_fn, (
        "save() must write H2H_HISTORY_FILE"
    )
    assert "self.h2h_history" in save_fn, (
        "save() must reference self.h2h_history"
    )


def test_state_save_round_trip_preserves_h2h_history_tail_mutations():
    """End-to-end: simulating service._round's pattern of mutating
    h2h_history[-1] after append_round must result in h2h_history.json
    on disk reflecting the mutation after save()."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        s = state_files.ValidatorState(state_dir=d)
        s.append_round({"round_id": 1, "king_uid": 47})
        s.h2h_history[-1]["king_after"] = "47"
        s.h2h_history[-1]["king_changed"] = False
        s.h2h_history[-1]["dethrone_method"] = "king_keeps_crown"
        s.save()
        on_disk = json.loads((d / state_files.H2H_HISTORY_FILE).read_text())
        assert on_disk[-1]["king_after"] == "47"
        assert on_disk[-1]["king_changed"] is False
        assert on_disk[-1]["dethrone_method"] == "king_keeps_crown"


# ── F2.2: watchdog env var name alignment ──────────────────────────


def test_watchdog_env_var_names_match_between_host_and_pod():
    """The host's env-forwarding whitelist and the pod's reader MUST
    agree on the variable name. A trailing ``_N`` mismatch makes
    operator tuning silently ineffective."""
    host_src = Path(eval_pod.__file__).read_text()
    orch_src = Path(
        Path(eval_pod.__file__).parent.parent / "pod" / "orchestrator.py"
    ).read_text()

    # The orchestrator reads ``DISTIL_ORCH_WATCHDOG_REPEAT`` (no _N).
    assert "os.environ.get(\"DISTIL_ORCH_WATCHDOG_REPEAT\")" in orch_src, (
        "orchestrator must read DISTIL_ORCH_WATCHDOG_REPEAT (no _N)"
    )
    # The host whitelist must use the same name.
    assert '"DISTIL_ORCH_WATCHDOG_REPEAT"' in host_src, (
        "host whitelist must include DISTIL_ORCH_WATCHDOG_REPEAT — "
        "trailing _N would silently break operator tuning"
    )
    assert '"DISTIL_ORCH_WATCHDOG_REPEAT_N"' not in host_src, (
        "host whitelist must NOT include the dead _N variant"
    )


# ── F1.4: teacher / reference args correctly bound ─────────────────


def test_process_round_call_passes_repo_args_correctly():
    """``service._round`` must pass ``spec['reference_repo']`` as the
    ``reference_name=`` kwarg and ``spec['teacher_repo']`` as the
    ``teacher_name=`` kwarg."""
    src = Path(service.__file__).read_text()
    start = src.index("record = process_round")
    # ``state.current_round`` appears twice (set at line ~237, mutated
    # again at ~306). We want the SECOND occurrence — the one AFTER
    # ``record = process_round`` — as the end-bound for our slice.
    end = src.index("state.current_round = {", start)
    call = src[start:end]
    assert "reference_name=spec[\"reference_repo\"]" in call, (
        "reference_name must bind to reference_repo (the Qwen3.5 baseline)"
    )
    assert "teacher_name=spec[\"teacher_repo\"]" in call, (
        "teacher_name must bind to teacher_repo (Kimi-K2.6 cloud teacher)"
    )
