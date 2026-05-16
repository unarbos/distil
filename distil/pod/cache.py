"""Pod-side Hugging Face cache management.

Two operations:

* :func:`sweep` — Pre-round broom: delete any ``models--*`` snapshot
  that isn't in the keep-set (this round's students + teacher +
  ``DISTIL_CACHE_KEEP_MODELS``). Typically reclaims 200-500 GB on a
  pod that has been running for hours. Runs ONCE per round before any
  student is loaded.
* :func:`clean_model` — Post-student broom: drop one specific model's
  cache as soon as we're done with it, preserving the teacher cache
  unconditionally. Keeps disk steady during the round.

Both honour ``HF_HUB_CACHE`` / ``HF_HOME`` / ``XDG_CACHE_HOME`` via the
hub library's own resolver, AND sweep the legacy ``~/.cache/huggingface/hub``
path if it differs — that fallback caught a real ops bug on 2026-05-15
where ``HOME=/root`` but the actual cache lived under
``HF_HOME=/home/.cache/huggingface``, so the previous hard-coded
``Path.home()/.cache`` sweeper silently freed 0 GB while 300+ GB of
stale snapshots accumulated.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("distil.pod.cache")


def _cache_roots() -> list[Path]:
    """Return the HF cache roots to inspect, deduped, existing only.

    Order: ``HF_HUB_CACHE``-resolved root first (matches what
    ``snapshot_download`` would pick), then the legacy
    ``~/.cache/huggingface/hub`` if it's a different path that exists.
    """
    roots: list[Path] = []
    try:
        from huggingface_hub import constants as _hf

        roots.append(Path(_hf.HF_HUB_CACHE))
    except Exception:
        pass
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        key = str(rp)
        if key in seen or not r.exists():
            continue
        seen.add(key)
        out.append(r)
    return out


def _cache_name(repo: str) -> str:
    return f"models--{repo.replace('/', '--')}"


def _entry_size_gb(entry: Path) -> float:
    try:
        size = sum(
            p.stat().st_size for p in entry.rglob("*") if p.is_file() and not p.is_symlink()
        )
        return size / 1024**3
    except Exception:
        return 0.0


def sweep(keep_repos: list[str]) -> dict:
    """Delete ``models--*`` snapshots that aren't in ``keep_repos``.

    ``DISTIL_CACHE_KEEP_MODELS`` adds further keepers (used by the
    parallel orchestrator to preserve sibling-shard models loaded by
    other GPUs on the same pod).

    Returns ``{kept, deleted, freed_gb, roots}``.
    """
    extra = os.environ.get("DISTIL_CACHE_KEEP_MODELS", "").strip()
    keep_names = {_cache_name(r) for r in keep_repos}
    for repo in (s.strip() for s in extra.split(",")):
        if repo:
            keep_names.add(_cache_name(repo))

    roots = _cache_roots()
    if not roots:
        return {"kept": 0, "deleted": 0, "freed_gb": 0.0, "roots": []}

    kept = deleted = 0
    freed_gb = 0.0
    for root in roots:
        for entry in sorted(root.iterdir()):
            if not entry.name.startswith("models--") or not entry.is_dir():
                continue
            if entry.name in keep_names:
                kept += 1
                continue
            sz = _entry_size_gb(entry)
            try:
                shutil.rmtree(entry, ignore_errors=True)
                deleted += 1
                freed_gb += sz
            except Exception as exc:
                logger.warning(f"sweep: cannot remove {entry.name}: {exc}")
    logger.info(
        f"sweep kept {kept} (students+teacher), deleted {deleted} stale "
        f"snapshots, ~{freed_gb:.0f} GB freed",
    )
    return {
        "kept": kept,
        "deleted": deleted,
        "freed_gb": round(freed_gb, 2),
        "roots": [str(r) for r in roots],
    }


def clean_model(repo: str, *, keep_repos: tuple[str, ...] = ()) -> bool:
    """Remove one specific model's cache from every root, unless it's
    in ``keep_repos`` (typically the teacher repo).
    """
    target = _cache_name(repo)
    keep_names = {_cache_name(r) for r in keep_repos}
    if target in keep_names:
        return False
    removed = False
    for root in _cache_roots():
        path = root / target
        if path.exists():
            try:
                shutil.rmtree(path, ignore_errors=True)
                removed = True
                logger.info(f"clean_model removed {target} from {root}")
            except Exception as exc:
                logger.warning(f"clean_model {target} in {root}: {exc}")
    return removed


__all__ = ["sweep", "clean_model"]
