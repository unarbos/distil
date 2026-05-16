"""Auto-clear ``integrity:HuggingFace 404`` DQs when the repo returns.

The legacy pipeline disqualified miners with a stable
``integrity: Model <repo> no longer exists on HuggingFace (404)``
reason any time a precheck HEAD on the repo's ``config.json`` failed.
``disqualified.json`` is hotkey-keyed and otherwise permanent — the
new ``distil/`` package no longer writes ``integrity:`` DQs at all
(it relies on the 3-strikes Phase-2 load-failure counter from
2026-05-16 instead), but the 36 legacy ``integrity:HF 404`` rows
remain in ``state.disqualified`` and never clear on their own.

aizaysi's case (UID 233 / ``RLStepone/distil-success-h19``,
flagged in #distil 2026-05-16): repo was deleted → legacy precheck
DQ'd him → repo was made public again → ``api.arbos.life`` still
reports him DQ'd → he loses king-5 emission share until a human
clears it.

This sweeper runs at the top of every round and HEAD-checks each
``integrity:.*404`` DQ. If the repo is now reachable (HTTP 200 on
``config.json``) we drop the DQ row. Network/timeout/transient
errors fail open — we never auto-clear on a non-200 unless we have
a concrete 200 in hand.

Restoring the same repo is a strict superset of "honest re-commit"
because the original failure was about that specific repo URL; if
the miner pushed a different repo on the same hotkey the new repo
will show up via the standard ``select_challengers`` path.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from distil.settings import settings
from distil.state.files import ValidatorState

logger = logging.getLogger("distil.eval.dq_recovery")

# Repo extractor: matches the two legacy DQ message formats:
#  - ``integrity: Model <repo> no longer exists on HuggingFace (404)``
#  - ``integrity: HuggingFace 404 for <repo>``
# A ``404`` / ``no longer exists`` substring anywhere on the line is
# the gate before we try to extract; the repo itself is whichever of
# ``Model X`` or ``for X`` appears.
_REPO_FROM_MODEL = re.compile(r"Model\s+(\S+)")
_REPO_FROM_FOR = re.compile(r"\bfor\s+([A-Za-z0-9_./\-]+)")


def _extract_repo(reason: str) -> str | None:
    if not reason or not reason.lower().startswith("integrity:"):
        return None
    if "404" not in reason and "no longer exists" not in reason.lower():
        return None
    m = _REPO_FROM_MODEL.search(reason)
    if m:
        return m.group(1).rstrip(".,;:")
    m = _REPO_FROM_FOR.search(reason)
    if m:
        return m.group(1).rstrip(".,;:")
    return None


def _hf_head_ok(repo: str) -> tuple[bool, int]:
    """Return ``(ok, status_code)`` — ok=True iff HF returns 200 for
    ``{repo}/resolve/main/config.json``. Network errors map to
    ``(False, 0)`` so we DON'T clear a DQ on a transient HF blip.
    """
    import requests

    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    token = settings.hf_dl_token or os.environ.get("HF_TOKEN") or ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.head(url, allow_redirects=True, timeout=8, headers=headers)
    except Exception as exc:
        logger.debug(f"dq_recovery HEAD failed for {repo}: {type(exc).__name__}: {exc}")
        return False, 0
    return r.status_code == 200, r.status_code


def sweep_integrity_dq_recoveries(state: ValidatorState) -> list[dict[str, Any]]:
    """Walk ``state.disqualified`` and clear any ``integrity:.*404`` DQ
    whose repo is now reachable on HF. Returns a list of cleared
    entries (``[{"hotkey": ..., "model": ..., "prior_reason": ...}]``)
    suitable for logging.

    Mutates ``state`` in place. The caller is responsible for
    persisting via ``state.save()``.
    """
    if not state.disqualified:
        return []
    cleared: list[dict[str, Any]] = []
    for hotkey, reason in list(state.disqualified.items()):
        if not isinstance(reason, str):
            continue
        if not reason.startswith("integrity:"):
            continue
        repo = _extract_repo(reason)
        if not repo:
            continue
        ok, code = _hf_head_ok(repo)
        if not ok:
            continue
        prior = state.disqualified.pop(hotkey, None)
        cleared.append({"hotkey": hotkey, "model": repo, "prior_reason": prior})
        logger.info(
            f"dq_recovery: cleared integrity DQ for hotkey={hotkey[:20]}... "
            f"({repo}) — HF now returns {code}"
        )
        # Also reset the failure counter (if any) for the UID linked
        # to this hotkey — the prior strike sequence was on the
        # broken repo; the restored repo gets a clean budget.
        # ``failures`` is uid-keyed not hotkey-keyed so we walk
        # evaluated_hotkeys for the mapping.
        record = (state.evaluated_hotkeys or {}).get(hotkey) if isinstance(
            state.evaluated_hotkeys, dict
        ) else None
        if isinstance(record, dict) and "uid" in record:
            try:
                state.reset_failures(int(record["uid"]))
            except Exception:
                pass
    return cleared
