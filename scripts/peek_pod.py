#!/usr/bin/env python3
"""Peek at the live pod eval state without disturbing the running process.

Reads /home/distil_eval_*/eval_progress.json and tails /home/distil_eval_*/eval.log
on the remote pod via SSH (Lium pod manager).

Usage:
    python scripts/peek_pod.py            # progress + last 40 log lines
    python scripts/peek_pod.py --tail 200 # progress + last 200 log lines
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from eval.pod import PodManager  # noqa: E402

try:
    from lium import Config, Lium  # type: ignore
except Exception:
    Lium = None  # type: ignore
    Config = None  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default=os.environ.get("LIUM_POD_NAME", "distil-eval"))
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--api-key", default=os.environ.get("LIUM_API_KEY"))
    args = ap.parse_args()

    if Lium is None:
        print("[peek] lium SDK not available", file=sys.stderr)
        return 2

    if not args.api_key:
        print("[peek] LIUM_API_KEY not set", file=sys.stderr)
        return 2

    from pathlib import Path
    cfg = Config(api_key=args.api_key, ssh_key_path=Path.home() / ".ssh" / "id_ed25519")
    lium = Lium(config=cfg)
    pod = PodManager(lium, args.pod)
    pod.connect()

    res = pod.exec("ls -1dt /home/distil_eval_* 2>/dev/null | head -1")
    out = res.get("stdout") if isinstance(res, dict) else ""
    run_dir = (out or "").strip().splitlines()[0] if out else ""
    if not run_dir:
        print("[peek] no /home/distil_eval_* run dirs on pod")
        return 1
    print(f"[peek] run_dir={run_dir}")

    res = pod.exec(f"cat {run_dir}/eval_progress.json 2>/dev/null || echo '{{}}'")
    try:
        prog = json.loads(((res.get("stdout") if isinstance(res, dict) else "") or "{}").strip() or "{}")
    except Exception:
        prog = {}
    if prog:
        compact = {
            k: prog.get(k)
            for k in (
                "phase",
                "students_total",
                "students_done",
                "prompts_total",
                "prompts_done",
                "current",
                "completed",
                "started_at",
                "updated_at",
            )
            if k in prog
        }
        print(json.dumps(compact, indent=2, default=str))
    else:
        print("[peek] no eval_progress.json yet")

    print(f"\n--- last {args.tail} lines of {run_dir}/eval.log ---")
    res = pod.exec(f"tail -n {int(args.tail)} {run_dir}/eval.log 2>/dev/null")
    print((res.get("stdout") if isinstance(res, dict) else "") or "(empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
