"""On-chain weight builders + ``set_weights`` with retry."""

from __future__ import annotations

import logging
import time

from distil.settings import settings

logger = logging.getLogger("distil.chain.weights")


class SetWeightsError(RuntimeError):
    """Raised when ``set_weights`` exhausts all retries."""


def build_winner_take_all_weights(n_uids: int, winner_uid: int) -> list[float]:
    """Return a length-N float vector with 1.0 at ``winner_uid`` and 0 elsewhere."""
    n = max(int(n_uids), int(winner_uid) + 1)
    weights = [0.0] * n
    if 0 <= winner_uid < n:
        weights[winner_uid] = 1.0
    return weights


def build_recent_kings_weights(
    n_uids: int,
    recent_kings: list[int],
    *,
    max_kings: int | None = None,
) -> list[float]:
    """Equal-split emission across the last ``max_kings`` distinct UIDs.

    Order of ``recent_kings`` is most-recent-first (the live king at index 0).
    Re-crowned UIDs are deduped to their most recent occurrence.
    """
    cap = int(max_kings if max_kings is not None else settings.recent_kings_max)
    seen: list[int] = []
    for uid in recent_kings:
        try:
            uid_i = int(uid)
        except (TypeError, ValueError):
            continue
        if uid_i < 0 or uid_i in seen:
            continue
        seen.append(uid_i)
        if len(seen) >= cap:
            break
    n = max(int(n_uids), (max(seen) + 1) if seen else 0)
    weights = [0.0] * n
    if not seen:
        return weights
    share = 1.0 / len(seen)
    for uid in seen:
        weights[uid] = share
    return weights


def get_validator_weight_pairs(
    subtensor, netuid: int, validator_uid: int
) -> list[tuple[int, int]] | None:
    """Return ``[(uid, raw_weight), ...]`` for the validator, or None if unset."""
    try:
        rows = subtensor.weights(netuid)
    except Exception as exc:
        logger.warning(f"subtensor.weights failed: {exc}")
        return None
    for row_uid, pairs in rows or []:
        if int(row_uid) != int(validator_uid):
            continue
        if not pairs:
            return None
        return [(int(uid), int(w)) for uid, w in pairs]
    return None


def get_validator_weight_targets(subtensor, netuid: int, validator_uid: int) -> set[int] | None:
    """Return the set of UIDs with non-zero weight for this validator."""
    pairs = get_validator_weight_pairs(subtensor, netuid, validator_uid)
    if pairs is None:
        return None
    return {uid for uid, w in pairs if w > 0}


def set_weights(
    subtensor,
    wallet,
    *,
    netuid: int,
    n_uids: int,
    weights: list[float],
    label: str = "weights",
    attempts: int = 3,
) -> None:
    """Submit ``weights`` on-chain with retries; raise :class:`SetWeightsError` on final failure."""
    if len(weights) > n_uids:
        n_uids = len(weights)
    uids = list(range(n_uids))
    last_err: str | None = None
    for i in range(attempts):
        try:
            result = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            ok = result[0] if isinstance(result, (tuple, list)) else bool(result)
            if ok:
                logger.info(f"set_weights ok ({label}); winner-share max={max(weights):.3f}")
                return
            last_err = (
                result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else str(result)
            )
            logger.warning(f"set_weights attempt {i + 1} rejected: {last_err}")
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            logger.error(f"set_weights attempt {i + 1} crashed: {exc}")
        if i + 1 < attempts:
            time.sleep(30)
    raise SetWeightsError(
        f"set_weights failed after {attempts} attempts ({label}): {last_err or 'unknown'}"
    )
