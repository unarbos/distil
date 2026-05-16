"""Validator state layer — atomic R/W with drop-in compat for state/*.json.

The legacy validator wrote ~25 distinct JSON shards under ``state/``. Only
~15 are read by anything (the other ~10 were sidecars for endpoints that
no longer have frontend consumers). This package keeps writing the live
set verbatim so the new validator is drop-in replaceable.

Public surface:
- :class:`ValidatorState` — the in-memory aggregate, persisted as one bag
- :func:`atomic_json_write` / :func:`safe_json_load` — primitives
- file-name constants — single source of truth for paths
"""

from distil.state.files import (
    ANNOUNCEMENT_FILE,
    COMPOSITE_SCORES_FILE,
    CURRENT_ROUND_FILE,
    DISQUALIFIED_FILE,
    EVAL_PROGRESS_FILE,
    EVALUATED_HOTKEYS_FILE,
    EVALUATED_UIDS_FILE,
    FAILURE_MODELS_FILE,
    FAILURES_FILE,
    H2H_HISTORY_FILE,
    H2H_LATEST_FILE,
    H2H_TESTED_KING_FILE,
    INCIDENTS_FILE,
    MODEL_HASHES_FILE,
    RECENT_KINGS_FILE,
    SCORES_FILE,
    TOP4_LEADERBOARD_FILE,
    UID_HOTKEY_MAP_FILE,
    VALIDATOR_LOG_FILE,
    ValidatorState,
    atomic_json_write,
    safe_json_load,
    sanitize_for_json,
)
from distil.state.store import StateStore, store

__all__ = [
    "ANNOUNCEMENT_FILE",
    "COMPOSITE_SCORES_FILE",
    "CURRENT_ROUND_FILE",
    "DISQUALIFIED_FILE",
    "EVAL_PROGRESS_FILE",
    "EVALUATED_HOTKEYS_FILE",
    "EVALUATED_UIDS_FILE",
    "FAILURE_MODELS_FILE",
    "FAILURES_FILE",
    "H2H_HISTORY_FILE",
    "H2H_LATEST_FILE",
    "H2H_TESTED_KING_FILE",
    "INCIDENTS_FILE",
    "MODEL_HASHES_FILE",
    "RECENT_KINGS_FILE",
    "SCORES_FILE",
    "TOP4_LEADERBOARD_FILE",
    "UID_HOTKEY_MAP_FILE",
    "VALIDATOR_LOG_FILE",
    "StateStore",
    "ValidatorState",
    "atomic_json_write",
    "safe_json_load",
    "sanitize_for_json",
    "store",
]
