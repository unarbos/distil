"""
Scoring logic: EMA tracking, proportional weights, staleness management.

Key design decisions:
- Inverse-KL weighting (lower KL = higher weight) instead of winner-take-all
- EMA smoothing (alpha=0.3) prevents single-epoch flukes from dominating
- Quality floor: models with KL > threshold get zero weight
- Staleness: 3 consecutive failures → weight 0 until new commitment
- All state persisted to disk for restart survival
"""
import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger("distillation.scoring")

STATE_DIR = Path("state")
DEFAULT_EMA_ALPHA = 0.3
DEFAULT_MAX_KL = 10.0  # Quality floor — reject if KL above this
MIN_KL_FLOOR = 1e-6  # Prevents div-by-zero for near-perfect models


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ── EMA Scores ────────────────────────────────────────────────────────────


def load_ema_scores(state_dir: Path = STATE_DIR) -> dict[str, float]:
    """Load EMA KL scores. Keys are string UIDs."""
    return _load_json(state_dir / "scores.json")


def save_ema_scores(scores: dict[str, float], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "scores.json", scores)


def update_ema(
    uid: int,
    new_kl: float,
    ema_scores: dict[str, float],
    alpha: float = DEFAULT_EMA_ALPHA,
) -> float:
    """
    Update EMA for a miner. Returns new EMA value.

    ema_kl = alpha * new_kl + (1 - alpha) * old_ema_kl
    First observation: EMA = new_kl (no history)
    """
    uid_str = str(uid)
    if uid_str in ema_scores:
        old_ema = ema_scores[uid_str]
        new_ema = alpha * new_kl + (1 - alpha) * old_ema
    else:
        new_ema = new_kl
    ema_scores[uid_str] = new_ema
    return new_ema


# ── Failure Tracking ──────────────────────────────────────────────────────


def load_failures(state_dir: Path = STATE_DIR) -> dict[str, int]:
    return _load_json(state_dir / "failures.json")


def save_failures(failures: dict[str, int], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "failures.json", failures)


def record_failure(uid: int, failures: dict[str, int]) -> int:
    """Record a failure for a miner. Returns new failure count."""
    uid_str = str(uid)
    failures[uid_str] = failures.get(uid_str, 0) + 1
    return failures[uid_str]


def reset_failures(uid: int, failures: dict[str, int]):
    """Reset failure count (e.g., after successful eval)."""
    failures.pop(str(uid), None)


def is_stale(uid: int, failures: dict[str, int], max_failures: int = 3) -> bool:
    """Check if a miner is stale (too many consecutive failures)."""
    return failures.get(str(uid), 0) >= max_failures


# ── Commitment Cache ──────────────────────────────────────────────────────


def load_commitment_cache(state_dir: Path = STATE_DIR) -> dict[str, dict]:
    """Load cached commitments. Keys are string UIDs, values have 'model', 'revision', 'kl'."""
    return _load_json(state_dir / "commitment_cache.json")


def save_commitment_cache(cache: dict[str, dict], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "commitment_cache.json", cache)


def commitment_changed(
    uid: int, model: str, revision: str, cache: dict[str, dict],
) -> bool:
    """Check if a miner's commitment has changed since last eval."""
    uid_str = str(uid)
    if uid_str not in cache:
        return True
    cached = cache[uid_str]
    return cached.get("model") != model or cached.get("revision") != revision


# ── Weight Computation ────────────────────────────────────────────────────


def compute_proportional_weights(
    ema_scores: dict[str, float],
    failures: dict[str, int],
    n_uids: int,
    max_kl: float = DEFAULT_MAX_KL,
    max_failures: int = 3,
) -> list[float]:
    """
    Compute proportional weights using inverse-KL weighting.

    - Filters out miners above max_kl threshold (quality floor)
    - Filters out stale miners (too many failures)
    - Weight_i = (1/KL_i) / sum(1/KL_j) for all valid miners
    - Lower KL → higher weight (continuous incentive to improve)

    Returns list of weights indexed by UID (length n_uids).
    """
    weights = [0.0] * n_uids

    # Collect valid miners
    valid = {}
    for uid_str, kl in ema_scores.items():
        uid = int(uid_str)
        if uid >= n_uids:
            continue
        if kl <= 0 or kl > max_kl:
            continue
        if is_stale(uid, failures, max_failures):
            continue
        valid[uid] = kl

    if not valid:
        return weights

    # Inverse-KL weighting with floor to prevent div-by-zero on perfect copies
    MIN_KL_FLOOR = 1e-6
    inv_kls = {uid: 1.0 / max(kl, MIN_KL_FLOOR) for uid, kl in valid.items()}
    total = sum(inv_kls.values())

    for uid, inv_kl in inv_kls.items():
        weights[uid] = inv_kl / total

    return weights
