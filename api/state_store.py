import json
import os
import sys

from config import DISK_CACHE_DIR, STATE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.state import (
    ANNOUNCEMENT_FILE,
    CURRENT_ROUND_FILE,
    DISQUALIFIED_FILE,
    EVAL_PROGRESS_FILE,
    H2H_HISTORY_FILE,
    H2H_LATEST_FILE,
    H2H_TESTED_KING_FILE,
    MODEL_HASHES_FILE,
    MODEL_SCORE_HISTORY_FILE,
    SCORE_HISTORY_FILE,
    SCORES_FILE,
    TOP4_LEADERBOARD_FILE,
    UID_HOTKEY_MAP_FILE,
)
from helpers.sanitize import _safe_json_load
from progress import normalize_eval_progress, progress_value


def _read(path, default=None):
    if default is None:
        default = {}
    return _safe_json_load(path, default)


def state_path(filename):
    return os.path.join(STATE_DIR, filename)


def cache_path(name):
    return os.path.join(DISK_CACHE_DIR, f"{name}.json")


def read_state(filename, default=None):
    return _read(state_path(filename), default)


def read_cache(name, default=None):
    return _read(cache_path(name), default)


def scores():
    return read_state(SCORES_FILE, {})


def disqualified():
    return read_state(DISQUALIFIED_FILE, {})


def last_eval():
    return read_state("last_eval.json", None)


def eval_progress():
    return read_state(EVAL_PROGRESS_FILE, {})


def current_round():
    return read_state(CURRENT_ROUND_FILE, {})


def h2h_latest():
    return read_state(H2H_LATEST_FILE, {})


def h2h_history():
    data = read_state(H2H_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def score_history():
    data = read_state(SCORE_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def top4_leaderboard():
    return read_state(TOP4_LEADERBOARD_FILE, {})


def uid_hotkey_map():
    return read_state(UID_HOTKEY_MAP_FILE, {})


def h2h_tested_against_king():
    return read_state(H2H_TESTED_KING_FILE, {})


def announcement():
    return read_state(ANNOUNCEMENT_FILE, {})


def model_score_history():
    return read_state(MODEL_SCORE_HISTORY_FILE, {})


def model_hashes():
    return read_state(MODEL_HASHES_FILE, {})


def benchmarks():
    path = state_path("benchmarks")
    if not os.path.exists(path):
        return [], None
    models = []
    baseline = None
    entries = []
    for name in os.listdir(path):
        if not name.endswith(".json"):
            continue
        full = os.path.join(path, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0
        entries.append((mtime, name, full))
    entries.sort(key=lambda e: e[0], reverse=True)
    # 2026-04-28: enrich each benchmark payload with the validator's
    # composite.worst for that UID so the dashboard Goodhart canary can
    # co-plot the validator score line against the held-out trend on
    # the same X-axis. Loaded once outside the per-file loop because
    # composite_scores.json is shared state.
    composite_scores_data = read_state("composite_scores.json", {})
    if not isinstance(composite_scores_data, dict):
        composite_scores_data = {}
    for mtime, _, full in entries:
        data = _read(full, None)
        if not isinstance(data, dict):
            continue
        data.setdefault("fetched_at", mtime)
        uid = data.get("uid")
        if isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit()):
            comp = composite_scores_data.get(str(uid))
            if isinstance(comp, dict):
                # The composite record can be either a flat dict (post-
                # 2026-04-26 schema) or nested under "composite" (legacy).
                inner = comp.get("composite") if isinstance(comp.get("composite"), dict) else comp
                worst = inner.get("worst")
                weighted = inner.get("weighted")
                # v30.2 — surface the new ranking key + worst_3_mean.
                final_score = inner.get("final")
                worst_3_mean = inner.get("worst_3_mean")
                axes = inner.get("axes") or {}
                if isinstance(worst, (int, float)):
                    data["composite_worst"] = worst
                if isinstance(weighted, (int, float)):
                    data["composite_weighted"] = weighted
                if isinstance(final_score, (int, float)):
                    data["composite_final"] = final_score
                if isinstance(worst_3_mean, (int, float)):
                    data["composite_worst_3_mean"] = worst_3_mean
                # v30.2 — surface group + shadow axes.
                # 2026-05-02 (v30.5): super_teacher kept here for
                # backwards-compat (older clients may still expect the
                # field) but the value is always 0.0 since the axis
                # was retired. New clients should ignore it.
                for axis_name, payload_key in (
                    ("code_skill_group", "axis_code_skill_group"),
                    ("math_skill_group", "axis_math_skill_group"),
                    ("reasoning_skill_group", "axis_reasoning_skill_group"),
                    ("knowledge_skill_group", "axis_knowledge_skill_group"),
                    ("super_teacher", "axis_super_teacher"),
                    ("top_k_overlap", "axis_top_k_overlap"),
                    ("kl_is", "axis_kl_is"),
                    ("forking_rkl", "axis_forking_rkl"),
                    ("teacher_trace_plausibility", "axis_teacher_trace_plausibility"),
                    ("entropy_aware_kl", "axis_entropy_aware_kl"),
                    ("tail_decoupled_kl", "axis_tail_decoupled_kl"),
                ):
                    v = axes.get(axis_name)
                    if isinstance(v, (int, float)):
                        data[payload_key] = v
        if data.get("is_baseline"):
            if baseline is None:
                baseline = data
        else:
            models.append(data)
    return models, baseline


def eval_data_file(name=None):
    if name:
        return os.path.join(state_path("eval_data"), os.path.basename(name))
    return state_path("eval_data_latest.json")


def read_json_file(path, default=None):
    return _read(path, default)


def write_json_file(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
