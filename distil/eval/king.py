"""King selection + emission-weight builder.

Decides who is king from the current ``composite_scores`` shard, applies
the dethrone margin, and builds the on-chain emission vector (winner-take-all
or recent-kings split per ``settings.multi_king_payout_enabled``).
"""

from __future__ import annotations

import logging
from typing import Any  # noqa: F401  # used by build_emission_weights type hint

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
    state: Any = None,
    uid_hotkey_map: dict[str, str] | None = None,
) -> list[float]:
    """Build the on-chain weight vector for this round.

    When ``state`` is provided, any ``recent_kings`` entry whose hotkey
    is currently disqualified is dropped from the emission split. This
    is the 2026-05-18 fix for the ``dethrone:no_king`` cascade: a king
    whose model has been DQ'd via load-failure strikes (e.g. UID 234
    with a broken tokenizer) should not continue receiving its
    ``recent_kings`` share until rotated out 4 crownings later. The
    seated ``king_uid`` is exempt from this filter — if the live king
    is somehow DQ'd we still emit their weight; that's a separate
    invariant violation that should be surfaced by alerting, not
    silently zeroed.
    """
    if king_uid is None:
        return [0.0] * n_uids
    if not settings.multi_king_payout_enabled:
        return build_winner_take_all_weights(n_uids, king_uid)
    history = [king_uid, *(u for u in recent_kings if int(u) != int(king_uid))]
    if state is not None and uid_hotkey_map is not None:
        filtered: list[int] = []
        for u in history:
            if int(u) == int(king_uid):
                filtered.append(int(u))
                continue
            hk = uid_hotkey_map.get(str(u))
            if hk and state.is_disqualified(hk, uid=int(u)):
                logger.info(
                    f"build_emission_weights: dropping DQ'd recent-king "
                    f"UID {u} (hk {hk[:14]}..) from emission split"
                )
                continue
            filtered.append(int(u))
        history = filtered
    return build_recent_kings_weights(n_uids, history, max_kings=settings.recent_kings_max)
