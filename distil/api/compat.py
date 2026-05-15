"""Compat shim: mount the production ``api/`` routers into the distil app.

Why this exists. The production ``api/`` package is ~3000 LoC of route
business logic the dashboard frontend already depends on
(``/api/queue``, ``/api/announcement``, ``/api/composite-scores``,
``/api/miner/{uid}/rounds``, ``/api/telemetry/*``, ``/api/chat/*``, etc.).
Re-translating it into ``distil/api/`` would risk subtle behavioral drift
and double the API LoC without functional gain.

This module loads the prod routers in-process (after putting ``api/`` on
``sys.path`` so its relative imports resolve) and re-exports them. The
distil FastAPI app then mounts them alongside the native distil routers.

After cutover, individual prod routes can be migrated to distil-native
implementations one at a time without breaking the frontend. The native
distil routers (in ``distil/api/routes.py``) take precedence: where a
route is defined in both, the distil version wins (FastAPI uses first
match).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("distil.api.compat")

_PROD_API = Path(__file__).resolve().parents[2] / "api"


def _ensure_prod_path() -> None:
    """Add the production ``api/`` dir to sys.path (idempotent)."""
    p = str(_PROD_API)
    if _PROD_API.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def load_prod_routers() -> list:
    """Return a list of FastAPI routers from the production ``api/`` package.

    Missing modules are logged and skipped (so partial migrations work).
    """
    _ensure_prod_path()
    out = []
    # Names: (module, attribute) — kept in include-order matching prod's
    # api/server.py so route precedence is identical.
    for mod_name, attr in [
        ("routes.health", "router"),
        ("routes.miners", "router"),
        ("routes.evaluation", "router"),
        ("routes.market", "router"),
        ("routes.chat", "router"),
        ("routes.debugging", "router"),
        ("routes.telemetry", "router"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            out.append(getattr(mod, attr))
        except Exception as exc:
            logger.warning(f"compat: failed to load {mod_name}: {exc}")
    return out


__all__ = ["load_prod_routers"]
