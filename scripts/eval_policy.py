"""Shared eval-policy loader.

The validator has historically treated environment variables as the policy
surface for eval sample counts, axis weights, and gates. This module keeps that
emergency-override behavior while moving defaults into a diffable JSON file.
It is intentionally dependency-free because it is uploaded beside
``pod_eval.py`` and imported inside remote GPU pods.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_POLICY_CACHE: dict[str, Any] | None = None
_MISSING = object()


def _candidate_paths() -> list[Path]:
    explicit = os.environ.get("DISTIL_EVAL_POLICY")
    here = Path(__file__).resolve()
    cwd = Path.cwd()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            here.with_name("eval_policy.json"),
            cwd / "eval_policy.json",
            here.parents[1] / "configs" / "eval_policy.json",
            cwd / "configs" / "eval_policy.json",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def policy_path() -> str | None:
    """Return the first existing policy path, if any."""
    for path in _candidate_paths():
        if path.is_file():
            return str(path)
    return None


def load_policy(*, force: bool = False) -> dict[str, Any]:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None and not force:
        return _POLICY_CACHE
    path = policy_path()
    if not path:
        _POLICY_CACHE = {}
        return _POLICY_CACHE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    _POLICY_CACHE = data if isinstance(data, dict) else {}
    return _POLICY_CACHE


def _walk_scalars(value: Any) -> dict[str, Any]:
    """Flatten sectioned policy maps into ENV_NAME -> scalar defaults."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out.update(_walk_scalars(item))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
    return out


def policy_defaults() -> dict[str, Any]:
    policy = load_policy()
    defaults: dict[str, Any] = {}
    for section in (
        "round",
        "samples",
        "samples_max_tokens",
        "weights",
        "gates",
        "runtime",
        "env",
    ):
        defaults.update(_walk_scalars(policy.get(section)))
    return defaults


def policy_env(name: str, default: Any = _MISSING) -> str | None:
    """Return an env-style string with process env taking precedence."""
    value = os.environ.get(name)
    if value is None:
        value = policy_defaults().get(name, default)
    if value is _MISSING:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return None
    return str(value)


def policy_bool(name: str, default: bool = False) -> bool:
    value = policy_env(name, "1" if default else "0")
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def policy_int(name: str, default: int) -> int:
    value = policy_env(name, str(default))
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return int(default)


def policy_float(name: str, default: float) -> float:
    value = policy_env(name, str(default))
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float(default)


def as_env() -> dict[str, str]:
    """Return policy defaults as env vars, excluding existing process env."""
    env: dict[str, str] = {}
    for key, value in policy_defaults().items():
        if key in os.environ or value is None:
            continue
        if isinstance(value, bool):
            env[key] = "1" if value else "0"
        else:
            env[key] = str(value)
    return env
