"""King selection + emission-weight builder.

Decides who is king from the current ``composite_scores`` shard, applies
the dethrone margin, and builds the on-chain emission vector (winner-take-all
or recent-kings split per ``settings.multi_king_payout_enabled``).
"""

from __future__ import annotations

import logging
from typing import Any

from distil.chain.weights import (
    build_recent_kings_weights,
    build_winner_take_all_weights,
)
from distil.eval.composite import is_dethrone, select_king
from distil.settings import settings

logger = logging.getLogger("distil.eval.king")


def resolve_king(
    composite_scores: dict[str, dict[str, Any]],
    current_king_model: str | None,
) -> tuple[str | None, str]:
    """Return ``(king_model, reason)`` after applying the dethrone gate."""
    contender = select_king(composite_scores)
    if contender is None:
        return current_king_model, "no_contender"
    if not current_king_model or current_king_model == contender:
        return contender, "no_king" if not current_king_model else "king_keeps_crown"
    challenger_comp = composite_scores.get(contender) or {}
    king_comp = composite_scores.get(current_king_model) or {}
    do_dethrone, reason = is_dethrone(challenger_comp, king_comp)
    if do_dethrone:
        return contender, f"dethrone:{reason}"
    return current_king_model, f"keep_king:{reason}"


def build_emission_weights(
    *,
    n_uids: int,
    king_uid: int | None,
    recent_kings: list[int],
) -> list[float]:
    """Build the on-chain weight vector for this round."""
    if king_uid is None:
        return [0.0] * n_uids
    if not settings.multi_king_payout_enabled:
        return build_winner_take_all_weights(n_uids, king_uid)
    history = [king_uid, *(u for u in recent_kings if int(u) != int(king_uid))]
    return build_recent_kings_weights(n_uids, history, max_kings=settings.recent_kings_max)
