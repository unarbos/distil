"""Validator state — schemas, atomic writes, in-memory aggregate.

Drop-in compatible with the legacy ``state/*.json`` shapes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.state")


# ── File-name constants (single source of truth) ─────────────────────────

SCORES_FILE = "scores.json"
FAILURES_FILE = "failures.json"
FAILURE_MODELS_FILE = "failure_models.json"
DISQUALIFIED_FILE = "disqualified.json"
EVALUATED_UIDS_FILE = "evaluated_uids.json"
EVALUATED_HOTKEYS_FILE = "evaluated_hotkeys.json"
COMPOSITE_SCORES_FILE = "composite_scores.json"
H2H_LATEST_FILE = "h2h_latest.json"
H2H_HISTORY_FILE = "h2h_history.json"
H2H_TESTED_KING_FILE = "h2h_tested_against_king.json"
UID_HOTKEY_MAP_FILE = "uid_hotkey_map.json"
EVAL_PROGRESS_FILE = "eval_progress.json"
CURRENT_ROUND_FILE = "current_round.json"
TOP4_LEADERBOARD_FILE = "top4_leaderboard.json"
RECENT_KINGS_FILE = "recent_kings.json"
MODEL_HASHES_FILE = "model_hashes.json"
ACTIVATION_FP_FILE = "activation_fingerprints.json"
INCIDENTS_FILE = "incidents.jsonl"
VALIDATOR_LOG_FILE = "validator_log.json"
ANNOUNCEMENT_FILE = "announcement.json"
ANNOUNCEMENT_CLAIMS_FILE = "announcement_claims.json"
EVAL_BACKLOG_FILE = "eval_backlog.json"
LAST_EVAL_FILE = "last_eval.json"
CHAT_POD_FILE = "chat_pod.json"


# ── JSON primitives ──────────────────────────────────────────────────────

def sanitize_for_json(obj: Any) -> Any:
    """Replace inf/nan floats with None so JSON serialization never fails."""
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [sanitize_for_json(v) for v in sorted(obj)]
    return obj


def atomic_json_write(path: str | os.PathLike, data: Any, indent: int | None = 2) -> None:
    """Atomic JSON write (tmp -> os.replace). Sanitises inf/nan."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sanitize_for_json(data), f, indent=indent)
    os.replace(tmp, p)


def safe_json_load(path: str | os.PathLike, default: Any = None) -> Any:
    """Load a JSON file, returning ``default`` on missing/corrupt."""
    if default is None:
        default = {}
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning(f"safe_json_load({p}) failed: {exc}; returning default")
        return default


def append_jsonl(path: str | os.PathLike, row: dict, max_rows: int = 5000) -> None:
    """Append a row to a JSONL file, capped at max_rows (rewrites tail on overflow)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(sanitize_for_json(row)) + "\n")
    # Cheap line-count check; only rewrite if grossly over budget.
    try:
        if p.stat().st_size > max_rows * 4096:
            lines = p.read_text().splitlines()[-max_rows:]
            p.write_text("\n".join(lines) + "\n")
    except Exception:
        pass


# ── ValidatorState — the in-memory aggregate ─────────────────────────────

@dataclass
class ValidatorState:
    """One bag for every persistent JSON shard the validator owns.

    Construct with :meth:`load`; persist with :meth:`save`. Each field maps
    1:1 to a file under ``state_dir`` so the legacy reader sees the same
    shapes.
    """

    state_dir: Path

    scores: dict[str, float] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    failure_models: dict[str, str] = field(default_factory=dict)
    disqualified: dict[str, str] = field(default_factory=dict)
    evaluated_uids: list[str] = field(default_factory=list)
    evaluated_hotkeys: dict[str, dict] = field(default_factory=dict)
    composite_scores: dict[str, dict] = field(default_factory=dict)
    h2h_latest: dict = field(default_factory=dict)
    h2h_history: list[dict] = field(default_factory=list)
    h2h_tested_against_king: dict[str, dict] = field(default_factory=dict)
    uid_hotkey_map: dict[str, str] = field(default_factory=dict)
    top4_leaderboard: dict = field(default_factory=dict)
    recent_kings: list[int] = field(default_factory=list)
    model_hashes: dict[str, str] = field(default_factory=dict)
    activation_fingerprints: dict[str, dict] = field(default_factory=dict)
    eval_backlog: dict = field(default_factory=dict)
    current_round: dict = field(default_factory=dict)  # in-progress round (for resume-on-attach)

    @classmethod
    def load(cls, state_dir: Path | None = None) -> "ValidatorState":
        d = Path(state_dir or settings.state_dir)
        d.mkdir(parents=True, exist_ok=True)
        return cls(
            state_dir=d,
            scores=safe_json_load(d / SCORES_FILE, {}),
            failures=safe_json_load(d / FAILURES_FILE, {}),
            failure_models=safe_json_load(d / FAILURE_MODELS_FILE, {}),
            disqualified=safe_json_load(d / DISQUALIFIED_FILE, {}),
            evaluated_uids=list(safe_json_load(d / EVALUATED_UIDS_FILE, [])),
            evaluated_hotkeys=safe_json_load(d / EVALUATED_HOTKEYS_FILE, {}),
            composite_scores=safe_json_load(d / COMPOSITE_SCORES_FILE, {}),
            h2h_latest=safe_json_load(d / H2H_LATEST_FILE, {}),
            h2h_history=list(safe_json_load(d / H2H_HISTORY_FILE, [])),
            h2h_tested_against_king=safe_json_load(d / H2H_TESTED_KING_FILE, {}),
            uid_hotkey_map=safe_json_load(d / UID_HOTKEY_MAP_FILE, {}),
            top4_leaderboard=safe_json_load(d / TOP4_LEADERBOARD_FILE, {}),
            recent_kings=list(safe_json_load(d / RECENT_KINGS_FILE, [])),
            model_hashes=safe_json_load(d / MODEL_HASHES_FILE, {}),
            activation_fingerprints=safe_json_load(d / ACTIVATION_FP_FILE, {}),
            eval_backlog=safe_json_load(d / EVAL_BACKLOG_FILE, {}),
            current_round=safe_json_load(d / CURRENT_ROUND_FILE, {}),
        )

    def save(self) -> None:
        d = self.state_dir
        atomic_json_write(d / SCORES_FILE, self.scores)
        atomic_json_write(d / FAILURES_FILE, self.failures)
        atomic_json_write(d / FAILURE_MODELS_FILE, self.failure_models)
        atomic_json_write(d / DISQUALIFIED_FILE, self.disqualified)
        atomic_json_write(d / EVALUATED_UIDS_FILE, self.evaluated_uids)
        atomic_json_write(d / EVALUATED_HOTKEYS_FILE, self.evaluated_hotkeys)
        atomic_json_write(d / COMPOSITE_SCORES_FILE, self.composite_scores)
        atomic_json_write(d / UID_HOTKEY_MAP_FILE, self.uid_hotkey_map)
        atomic_json_write(d / TOP4_LEADERBOARD_FILE, self.top4_leaderboard)
        atomic_json_write(d / RECENT_KINGS_FILE, self.recent_kings)
        atomic_json_write(d / MODEL_HASHES_FILE, self.model_hashes)
        atomic_json_write(d / ACTIVATION_FP_FILE, self.activation_fingerprints)
        atomic_json_write(d / EVAL_BACKLOG_FILE, self.eval_backlog)
        atomic_json_write(d / CURRENT_ROUND_FILE, self.current_round)
        # h2h_latest is the live snapshot; h2h_history is append-only and
        # already persisted by :meth:`append_round`.
        if self.h2h_latest:
            atomic_json_write(d / H2H_LATEST_FILE, self.h2h_latest)
        if self.h2h_tested_against_king:
            atomic_json_write(d / H2H_TESTED_KING_FILE, self.h2h_tested_against_king)

    def append_round(self, round_record: dict, max_history: int = 1000) -> None:
        """Append one round to h2h_history.json (capped) and update h2h_latest."""
        self.h2h_history.append(round_record)
        if len(self.h2h_history) > max_history:
            self.h2h_history = self.h2h_history[-max_history:]
        atomic_json_write(self.state_dir / H2H_HISTORY_FILE, self.h2h_history)
        self.h2h_latest = round_record
        atomic_json_write(self.state_dir / H2H_LATEST_FILE, round_record)

    # ── Convenience helpers ────────────────────────────────────────────

    def disqualify(self, hotkey: str, reason: str) -> None:
        if not hotkey:
            return
        self.disqualified[hotkey] = reason

    def is_disqualified(self, hotkey: str | None, uid: int | None = None) -> bool:
        if hotkey and hotkey in self.disqualified:
            return True
        if uid is not None and str(uid) in self.disqualified:
            return True
        return False

    def dq_reason(self, hotkey: str | None, uid: int | None = None) -> str:
        if hotkey and hotkey in self.disqualified:
            return self.disqualified[hotkey]
        if uid is not None:
            return self.disqualified.get(str(uid), "")
        return ""

    def record_failure(self, uid: int, model_name: str | None = None) -> int:
        k = str(uid)
        self.failures[k] = self.failures.get(k, 0) + 1
        if model_name:
            self.failure_models[k] = model_name
        return self.failures[k]

    def reset_failures(self, uid: int) -> None:
        self.failures.pop(str(uid), None)

    def push_king(self, uid: int) -> None:
        """Push a new king to the front of recent_kings, dedupe, cap."""
        history = [u for u in self.recent_kings if int(u) != int(uid)]
        history.insert(0, int(uid))
        self.recent_kings = history[: settings.recent_kings_max]


# ── Validator log + incidents (append-only) ──────────────────────────────

def log_event(message: str, level: str = "info", state_dir: Path | None = None) -> None:
    """Append a structured log row (capped) and an incidents.jsonl row."""
    d = Path(state_dir or settings.state_dir)
    rows = safe_json_load(d / VALIDATOR_LOG_FILE, [])
    rows.append({"ts": time.time(), "level": level, "msg": str(message)[:500]})
    if len(rows) > 2000:
        rows = rows[-2000:]
    atomic_json_write(d / VALIDATOR_LOG_FILE, rows)
    if level in ("error", "warning"):
        append_jsonl(d / INCIDENTS_FILE, {"ts": time.time(), "level": level, "msg": message})
