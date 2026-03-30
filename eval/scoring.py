"""
Scoring logic: winner-take-all weights.

- Winner-take-all: best KL miner gets ALL the weight (1.0), everyone else gets 0.0
- No EMA — models are permanently committed, scores converge naturally
- Quality floor: KL > threshold gets zero weight
- All state persisted to disk for restart survival
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("distillation.scoring")

STATE_DIR = Path("state")
DEFAULT_MAX_KL = 2.0


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


# ── Scores ────────────────────────────────────────────────────────────────


def load_scores(state_dir: Path = STATE_DIR) -> dict[str, float]:
    """Load KL scores. Keys are string UIDs."""
    return _load_json(state_dir / "scores.json")


def save_scores(scores: dict[str, float], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "scores.json", scores)


# Aliases for EMA-style access (validator.py uses these names)
load_ema_scores = load_scores
save_ema_scores = save_scores


def update_ema(
    uid: int,
    new_kl: float,
    scores: dict[str, float],
    alpha: float = 0.3,
) -> float:
    """
    Update EMA score for a UID.

    EMA = alpha * new_kl + (1 - alpha) * old_kl
    If no prior score exists, uses the new value directly.
    """
    uid_str = str(uid)
    old_kl = scores.get(uid_str)
    if old_kl is None:
        ema = new_kl
    else:
        ema = alpha * new_kl + (1 - alpha) * old_kl
    scores[uid_str] = ema
    return ema


# ── Disqualification Tracking ─────────────────────────────────────────────


def load_disqualified(state_dir: Path = STATE_DIR) -> dict[str, str]:
    """Load disqualification reasons. Keys are string UIDs, values are reason strings."""
    return _load_json(state_dir / "disqualified.json")


def save_disqualified(dq: dict[str, str], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "disqualified.json", dq)


def disqualify(uid: int, reason: str, dq: dict[str, str]):
    """Record a disqualification with reason."""
    dq[str(uid)] = reason


# ── Failure Tracking ──────────────────────────────────────────────────────


def load_failures(state_dir: Path = STATE_DIR) -> dict[str, int]:
    return _load_json(state_dir / "failures.json")


def save_failures(failures: dict[str, int], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "failures.json", failures)


def record_failure(uid: int, failures: dict[str, int]) -> int:
    uid_str = str(uid)
    failures[uid_str] = failures.get(uid_str, 0) + 1
    return failures[uid_str]


def reset_failures(uid: int, failures: dict[str, int]):
    failures.pop(str(uid), None)


def is_stale(uid: int, failures: dict[str, int], max_failures: int = 3) -> bool:
    return failures.get(str(uid), 0) >= max_failures


# ── Commitment Cache ──────────────────────────────────────────────────────


def load_commitment_cache(state_dir: Path = STATE_DIR) -> dict[str, dict]:
    return _load_json(state_dir / "commitment_cache.json")


def save_commitment_cache(cache: dict[str, dict], state_dir: Path = STATE_DIR):
    _save_json(state_dir / "commitment_cache.json", cache)


def commitment_changed(
    uid: int, model: str, revision: str, cache: dict[str, dict],
) -> bool:
    uid_str = str(uid)
    if uid_str not in cache:
        return True
    cached = cache[uid_str]
    return cached.get("model") != model or cached.get("revision") != revision


# ── Weight Computation ────────────────────────────────────────────────────


# ── Score History ──────────────────────────────────────────────────────────


def load_score_history(state_dir: Path = STATE_DIR) -> list[dict]:
    """Load score history array from disk."""
    path = state_dir / "score_history.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def append_score_history(
    block: int,
    timestamp: float,
    scores: dict[str, float],
    king_uid: int | None,
    state_dir: Path = STATE_DIR,
    max_entries: int = 500,
):
    """Append a score snapshot to history, capping at max_entries."""
    history = load_score_history(state_dir)
    entry = {
        "block": block,
        "timestamp": timestamp,
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "king_uid": king_uid,
    }
    history.append(entry)
    if len(history) > max_entries:
        history = history[-max_entries:]
    path = state_dir / "score_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    logger.info(f"Score history: {len(history)} entries (block {block})")


# ── Weight Computation ────────────────────────────────────────────────────


def compute_winner_weights(
    scores: dict[str, float],
    failures: dict[str, int],
    n_uids: int,
    max_kl: float = DEFAULT_MAX_KL,
    max_failures: int = 3,
    epsilon: float = 0.0,
) -> tuple[list[float], int | None, float]:
    """
    Winner-take-all with epsilon threshold.

    The current king (lowest KL) holds unless a challenger beats it by
    more than epsilon (relative). With epsilon=0.01, a challenger must
    have KL < king_kl * 0.99 to dethrone.

    This prevents noisy near-ties from flipping the winner every epoch.
    """
    # Ensure n_uids covers all scored UIDs (metagraph.n can be stale after sync failures)
    max_scored_uid = max((int(k) for k in scores), default=-1)
    if max_scored_uid >= n_uids:
        logger.warning(
            f"n_uids={n_uids} but scores contain UID {max_scored_uid} — "
            f"expanding weights array (metagraph.n likely stale)"
        )
        n_uids = max_scored_uid + 1

    weights = [0.0] * n_uids

    # Find all eligible candidates
    candidates = []
    for uid_str, kl in scores.items():
        uid = int(uid_str)
        if uid >= n_uids:
            continue
        if kl <= 0 or kl > max_kl:
            continue
        if is_stale(uid, failures, max_failures):
            continue
        candidates.append((uid, kl))

    if not candidates:
        return weights, None, float("inf")

    # Sort by KL ascending
    candidates.sort(key=lambda x: x[1])
    best_uid, best_kl = candidates[0]

    # With epsilon, the king (best) only changes if someone is strictly
    # better by epsilon margin. Since we pick the absolute best here,
    # the epsilon is enforced by the caller (remote_validator) which
    # decides whether to update scores for near-ties.
    weights[best_uid] = 1.0
    return weights, best_uid, best_kl
