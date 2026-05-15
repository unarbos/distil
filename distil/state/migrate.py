"""One-shot migration: archive orphaned legacy state shards.

The rewrite drops several state files that nothing reads any more
(score_history.json, model_score_history.json, king_canary_streak.json,
permanently_bad_models.json, king_regression_streak.json,
finetune_probe_telemetry.jsonl, healthcheck.json, restart_budget.json,
private_pool_*.json, openclaw_config_health.json, …). Move them to
``state/_legacy/`` so we keep the audit trail without confusing readers.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from distil.settings import settings

# Files that are no longer written by the new validator.
ORPHANED_FILES = (
    "score_history.json",
    "model_score_history.json",
    "permanently_bad_models.json",
    "king_canary_streak.json",
    "king_regression_streak.json",
    "finetune_probe_telemetry.jsonl",
    "healthcheck.json",
    "restart_budget.json",
    "openclaw_config_health.json",
    "private_pool_commit.json",
    "private_pool_reveal.json",
    "model_content_hashes.json",
    "weight_hashes.json",
    "axis_correlation.json",
    "kimi_cutover_pod.json",
    "dq_reasons.json",  # superseded by reading disqualified.json directly
    "cumulative_scores.json",
    "h2h_tested_against_king.json.bak2",
    # backup files
)


def migrate(state_dir: Path | None = None, dry_run: bool = False) -> list[str]:
    """Move orphaned files to ``state/_legacy/``. Returns the list moved."""
    d = Path(state_dir or settings.state_dir)
    archive = d / "_legacy"
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for name in ORPHANED_FILES:
        src = d / name
        if src.exists():
            dst = archive / name
            if dry_run:
                moved.append(str(src))
                continue
            shutil.move(str(src), str(dst))
            moved.append(str(src))
    # Backup files (*.bak.*) — sweep root only
    for bak in d.glob("*.bak*"):
        if bak.is_file():
            if dry_run:
                moved.append(str(bak))
                continue
            shutil.move(str(bak), str(archive / bak.name))
            moved.append(str(bak))
    return moved


def run(*, dry_run: bool = False) -> int:
    """CLI entry — return 0 on success, prints what was (or would be) moved."""
    moved = migrate(Path(settings.state_dir), dry_run=dry_run)
    print(f"{'Would move' if dry_run else 'Moved'} {len(moved)} files to state/_legacy/:")
    for m in moved:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(dry_run=args.dry_run))
