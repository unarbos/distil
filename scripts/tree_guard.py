#!/usr/bin/env python3
"""Working-tree safety net for /opt/distil/repo.

Runs every 3 min via ``distil-tree-guard.timer``. Two responsibilities:

1.  **Tripwire** — detect mass deletion of tracked files and restore
    them via ``git checkout HEAD -- <path>`` BEFORE the next service
    restart catches the bare tree. On 2026-05-18 and 2026-05-19 the
    repo was wiped twice (87 and 103 tracked files removed from the
    working tree while ``HEAD`` was untouched); both times the running
    validator kept executing off cached imports and a restart would
    have crashed silently. This tripwire closes that gap.

2.  **WIP auto-checkpoint** — if there are stable working-tree changes
    (files unmodified for >5 min) AND no commit has landed in the last
    30 min, commit them as ``chore: auto-checkpoint`` and push. This
    means edits-in-flight never live ONLY on disk — a subsequent wipe
    can be restored from the remote.

Both actions are conservative + rate-limited via ``state/tree_guard.json``
so a flapping wipe loop can't self-DoS:

    {
      "last_restore_ts":  <unix epoch>,
      "last_restore_count": <int>,
      "last_checkpoint_ts": <unix epoch>,
      "last_checkpoint_sha": "<short sha>",
      "incidents": [...]   # rolling tail, capped at 20
    }

Discord alert is best-effort and uses the existing ``distil.eval.announce``
bot token; if the bot/channel is unreachable we still log to journal.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("DISTIL_REPO_ROOT", "/opt/distil/repo"))
STATE_FILE = REPO / "state" / "tree_guard.json"

# ── Tripwire knobs ───────────────────────────────────────────────────
# Number of tracked files missing on disk that constitutes a "mass
# deletion" worth auto-restoring. The two real incidents wiped 87 and
# 103 files; a single edit / stash typically deletes < 5. 25 is
# comfortably above the noise floor.
DELETION_THRESHOLD = 25
# Don't auto-restore more than once per hour. If a wipe loop is firing
# we want a human in the loop, not a thrashing restore-vs-wipe race.
RESTORE_COOLDOWN_S = 3600

# ── WIP checkpoint knobs ─────────────────────────────────────────────
# Only commit working-tree changes when ALL changed files have an
# mtime older than this. An agent actively editing files will keep
# bumping mtimes; we wait for the session to go quiet.
WIP_STABILITY_S = 300       # 5 min
# Don't auto-checkpoint more often than every 30 min. Human commits in
# between push the next checkpoint horizon out — no need to clutter
# history when the operator is committing themselves.
WIP_CHECKPOINT_COOLDOWN_S = 1800  # 30 min
# Hard cap on per-checkpoint scope so a runaway WIP can't push a
# 10,000-line "chore" commit. If the staged diff is larger than this,
# log a warning and skip the auto-commit — the operator should split
# the work into intentional commits.
WIP_MAX_FILES = 200

# ── Discord alert ────────────────────────────────────────────────────
_ANNOUNCE_DISABLED = os.environ.get("DISTIL_TREE_GUARD_QUIET", "0") == "1"

logger = logging.getLogger("tree_guard")


def _run(cmd: list[str], *, timeout: int = 60) -> tuple[int, str, str]:
    """Subprocess wrapper that returns (rc, stdout, stderr) and never raises."""
    try:
        r = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout running {cmd}"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"{type(e).__name__}: {e}"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"incidents": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {"incidents": []}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_FILE)


def _record_incident(state: dict, kind: str, details: dict) -> None:
    entry = {"ts": time.time(), "kind": kind, **details}
    state.setdefault("incidents", []).append(entry)
    # Keep tail bounded so the file doesn't grow without limit.
    state["incidents"] = state["incidents"][-20:]


def _alert_discord(title: str, body: str) -> None:
    if _ANNOUNCE_DISABLED:
        logger.info("discord alert suppressed (DISTIL_TREE_GUARD_QUIET=1)")
        return
    # Best-effort import — if the announce module is missing or the
    # bot token isn't reachable we still log the incident locally.
    try:
        sys.path.insert(0, str(REPO))
        from distil.eval.announce import _bot_token, _channel_id
        tok = _bot_token()
        ch = _channel_id()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"discord-alert: could not load bot creds: {e}")
        return
    if not tok or not ch:
        logger.info("discord-alert: bot token/channel not configured; skipping")
        return
    msg = f"## {title}\n\n{body}"
    url = f"https://discord.com/api/v10/channels/{ch}/messages"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"content": msg[:1900]}).encode("utf-8"),
            headers={
                "Authorization": f"Bot {tok}",
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot (distil-tree-guard 1.0)",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
    except urllib.error.HTTPError as e:
        logger.warning(f"discord-alert: HTTP {e.code}: {e.read()[:200]!r}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"discord-alert: {type(e).__name__}: {e}")


# ── Tripwire ─────────────────────────────────────────────────────────


def _tracked_at_head() -> list[str]:
    rc, out, _ = _run(["git", "ls-tree", "-r", "HEAD", "--name-only"])
    if rc != 0:
        return []
    return [p for p in out.splitlines() if p]


def _missing_on_disk(tracked: list[str]) -> list[str]:
    missing = []
    for p in tracked:
        if not (REPO / p).exists():
            missing.append(p)
    return missing


def _tripwire(state: dict, *, dry_run: bool) -> dict:
    tracked = _tracked_at_head()
    if not tracked:
        return {"action": "skip", "reason": "no HEAD tracked files"}
    missing = _missing_on_disk(tracked)
    if len(missing) < DELETION_THRESHOLD:
        return {"action": "ok", "missing_count": len(missing)}

    now = time.time()
    last = state.get("last_restore_ts", 0) or 0
    if now - last < RESTORE_COOLDOWN_S and not dry_run:
        return {
            "action": "cooldown",
            "missing_count": len(missing),
            "seconds_since_last_restore": int(now - last),
        }

    sample = sorted(missing)[:6]
    if dry_run:
        return {"action": "would-restore", "missing_count": len(missing), "sample": sample}

    rc, _, err = _run(["git", "checkout", "HEAD", "--", "."], timeout=180)
    success = rc == 0
    state["last_restore_ts"] = now
    state["last_restore_count"] = len(missing)
    _record_incident(state, "tripwire_restore", {
        "missing_count": len(missing),
        "sample": sample,
        "git_checkout_rc": rc,
        "git_checkout_err": err[:400] if err else None,
    })
    if success:
        logger.warning(
            f"tripwire: restored {len(missing)} tracked files "
            f"(sample={sample}); see {STATE_FILE}"
        )
        _alert_discord(
            "tree-guard: auto-restored after mass deletion",
            (
                f"Detected **{len(missing)} tracked files** missing from "
                f"`/opt/distil/repo` on disk while `HEAD` was untouched. "
                f"Auto-restored via `git checkout HEAD -- .`.\n\n"
                f"Sample: ```\n" + "\n".join(sample) + "\n```\n"
                f"Next restore allowed in {RESTORE_COOLDOWN_S // 60} min "
                f"(cooldown). Investigate `state/tree_guard.json` for the "
                f"incident trail."
            ),
        )
    else:
        logger.error(f"tripwire: restore FAILED rc={rc} err={err[:200]}")
        _alert_discord(
            "tree-guard: auto-restore FAILED",
            f"Detected {len(missing)} missing files but `git checkout` "
            f"returned rc={rc}. Manual intervention required.",
        )
    return {
        "action": "restored" if success else "restore-failed",
        "missing_count": len(missing),
        "sample": sample,
    }


# ── WIP checkpoint ───────────────────────────────────────────────────


def _porcelain_changes() -> list[tuple[str, str]]:
    """Return ``[(status, path)]`` for every entry in ``git status --porcelain``.

    Skips untracked-and-ignored entries (status `!!`) so a fresh
    ``__pycache__/`` or local ``state/`` write never triggers a
    checkpoint.
    """
    rc, out, _ = _run(["git", "status", "--porcelain=v1", "--no-renames"])
    if rc != 0:
        return []
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:].strip()
        if status.strip() == "!!":
            continue
        rows.append((status, path))
    return rows


def _all_stable(paths: list[str]) -> bool:
    """All listed paths have an mtime older than WIP_STABILITY_S."""
    cutoff = time.time() - WIP_STABILITY_S
    for p in paths:
        full = REPO / p
        try:
            if full.is_dir():
                # Use the most recently modified file in the dir.
                for f in full.rglob("*"):
                    try:
                        if f.stat().st_mtime > cutoff:
                            return False
                    except OSError:
                        continue
                continue
            if full.stat().st_mtime > cutoff:
                return False
        except OSError:
            continue
    return True


def _last_commit_age_s() -> float:
    rc, out, _ = _run(["git", "log", "-1", "--format=%ct"])
    if rc != 0 or not out.strip().isdigit():
        return float("inf")
    return time.time() - int(out.strip())


def _wip_checkpoint(state: dict, *, dry_run: bool) -> dict:
    changes = _porcelain_changes()
    # Only operate on TRACKED edits (modifications + deletions). Don't
    # auto-commit fresh untracked files — they're often local-only
    # scratch files (the operator can `git add` intentionally).
    tracked = [(s, p) for (s, p) in changes if s.strip() not in ("??",)]
    if not tracked:
        return {"action": "skip", "reason": "no tracked working-tree changes"}

    if len(tracked) > WIP_MAX_FILES:
        return {
            "action": "skip-too-large",
            "changed_count": len(tracked),
            "max": WIP_MAX_FILES,
        }

    last_age = _last_commit_age_s()
    if last_age < WIP_CHECKPOINT_COOLDOWN_S:
        return {
            "action": "skip-cooldown",
            "seconds_since_last_commit": int(last_age),
            "min_cooldown_s": WIP_CHECKPOINT_COOLDOWN_S,
        }

    paths = [p for (_, p) in tracked]
    if not _all_stable(paths):
        return {
            "action": "skip-unstable",
            "changed_count": len(tracked),
            "stability_window_s": WIP_STABILITY_S,
        }

    # Guard against the remote being ahead of us — auto-commits push,
    # and pushing on top of a divergent remote could be destructive.
    rc, _, _ = _run(["git", "fetch", "origin", "--quiet"], timeout=30)
    if rc == 0:
        rc2, ahead_behind, _ = _run(
            ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"]
        )
        if rc2 == 0:
            try:
                _ahead, behind = ahead_behind.strip().split()
                if int(behind) > 0:
                    return {
                        "action": "skip-divergent",
                        "behind_remote_by": int(behind),
                    }
            except (ValueError, IndexError):
                pass

    if dry_run:
        return {"action": "would-checkpoint", "files": paths[:10], "count": len(paths)}

    # Stage every TRACKED change. We deliberately skip ``git add -A``
    # because that would also stage untracked scratch files.
    _run(["git", "add", "-u"])

    rc, _, _ = _run(["git", "diff", "--cached", "--quiet"])
    if rc == 0:
        return {"action": "skip-empty-diff", "note": "after staging, no diff"}

    msg = (
        f"chore(auto-checkpoint): {len(paths)} stable working-tree file(s)\n\n"
        f"Captured by ``distil-tree-guard`` — stable for >{WIP_STABILITY_S // 60} min, "
        f"last human commit {int(last_age // 60)} min ago. Replace with a "
        f"meaningful commit once the session is ready to publish."
    )
    rc, _, err = _run(["git", "commit", "-m", msg], timeout=30)
    if rc != 0:
        logger.warning(f"wip-checkpoint: commit failed rc={rc} err={err[:200]}")
        return {"action": "commit-failed", "rc": rc, "err": err[:200]}

    rc, _, err = _run(["git", "push", "origin", "HEAD"], timeout=60)
    if rc != 0:
        # Roll back the commit so the next run can retry cleanly. The
        # mtimes don't change, so the next tick still sees the same
        # stable WIP.
        logger.warning(f"wip-checkpoint: push failed rc={rc} err={err[:200]}; rolling back")
        _run(["git", "reset", "--soft", "HEAD^"])
        return {"action": "push-failed", "rc": rc, "err": err[:200]}

    rc, out, _ = _run(["git", "rev-parse", "--short", "HEAD"])
    sha = out.strip() if rc == 0 else "?"
    state["last_checkpoint_ts"] = time.time()
    state["last_checkpoint_sha"] = sha
    _record_incident(state, "wip_checkpoint", {
        "sha": sha,
        "file_count": len(paths),
    })
    logger.info(f"wip-checkpoint: committed+pushed {sha} ({len(paths)} files)")
    return {"action": "checkpointed", "sha": sha, "count": len(paths)}


# ── Entry point ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be done; mutate nothing.")
    ap.add_argument(
        "--mode", choices=("all", "tripwire", "checkpoint"), default="all",
        help="Run only the named phase (default: both).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s tree_guard: %(message)s",
    )

    state = _load_state()
    report: dict[str, Any] = {"ts": time.time(), "dry_run": args.dry_run}

    if args.mode in ("all", "tripwire"):
        report["tripwire"] = _tripwire(state, dry_run=args.dry_run)
    if args.mode in ("all", "checkpoint"):
        report["checkpoint"] = _wip_checkpoint(state, dry_run=args.dry_run)

    if not args.dry_run:
        state["last_run_ts"] = report["ts"]
        state["last_run"] = report
        _save_state(state)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
