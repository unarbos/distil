#!/usr/bin/env python3
"""Patch the running open-webui container so that frontend ``execute:python``
and ``execute:tool`` events always dispatch to the Pyodide / tool worker.

Why this exists
---------------
Open-WebUI 0.8.12's ``chatEventHandler`` (in ``+layout.svelte``) only fires
the Pyodide worker (``executePythonAsWorker``) when **both** of these are
true at the moment the backend event arrives:

    1. The user is currently viewing the chat that triggered the call.
    2. The browser tab is focused (``document.visibilityState === 'visible'``).
    3. The data.session_id matches the current ``$socket.id``.

If any of those drift (user navigates to another chat, minimises the tab,
the socket reconnects mid-completion and gets a new id) the worker never
runs, the backend's ``__event_call__`` ack times out, ``execute_code``
parses ``output`` as ``None`` and returns ``stdout=''``. To the user this
shows up as the "empty output" failure mode reported on chat.arbos.life
on 2026-05-03 — same bug that ate the "9^100th Fibonacci" debug session.

The fix routes ``execute:python`` and ``execute:tool`` BEFORE the
focus/chat/session checks, so they always run when the event arrives.
The remaining notification-only branches still gate on focus as before.

Idempotent: if the patch is already applied the script is a no-op.

Usage
-----
::

    python3 scripts/chat_pod/patch_owui_pyodide_dispatch.py

Designed to run inside the open-webui container, but can also be invoked
from the host via ``docker exec`` — both code paths are covered.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Directory under which all sveltekit-built immutable JS chunks live.
APP_BUILD_NODES = Path("/app/build/_app/immutable/nodes")

# This is the layout-level chunk that holds chatEventHandler. The hash in
# the filename (``0.<HASH>.js``) is content-derived and changes on every
# OWUI release; we therefore detect by glob + content fingerprint, not by
# fixed name.
LAYOUT_NODE_GLOB = "0.*.js"

# Fingerprint that uniquely identifies the buggy ``chatEventHandler`` body
# in the minified bundle. Present in 0.8.12.
OLD_FRAGMENT = (
    'if(t.chat_id!==ce()&&!_()||I)if(L==="chat:completion"){'
)

# Replacement: execute:python / execute:tool fire unconditionally; the
# remaining branches keep the old gating so notifications don't fire on
# the wrong tab.
NEW_FRAGMENT = (
    'if(L==="execute:python"){Re(g.id,g.code,n,g.files||[]);return}'
    'if(L==="execute:tool"){de(g,n);return}'
    'if(t.chat_id!==ce()&&!_()||I)if(L==="chat:completion"){'
)

# Marker we leave in the patched file so we can detect "already patched".
PATCHED_MARKER = '"execute:python"){Re(g.id,g.code,n,g.files||[]);return}'


def patch_file(path: Path) -> str:
    """Apply the patch in-place. Returns one of:
    ``"patched"`` - file was modified;
    ``"already-patched"`` - marker present, nothing to do;
    ``"miss"`` - neither the OLD pattern nor the marker were found.
    """
    src = path.read_text(encoding="utf-8")
    if PATCHED_MARKER in src:
        return "already-patched"
    if OLD_FRAGMENT not in src:
        return "miss"
    n = src.count(OLD_FRAGMENT)
    if n != 1:
        raise RuntimeError(
            f"OLD_FRAGMENT appears {n} times in {path} — refusing to patch "
            "a non-unique target."
        )
    backup = path.with_suffix(path.suffix + ".pre-pyodide-fix.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    new_src = src.replace(OLD_FRAGMENT, NEW_FRAGMENT, 1)
    path.write_text(new_src, encoding="utf-8")
    return "patched"


def find_target() -> Path | None:
    """Locate the layout chunk inside the OWUI build."""
    if not APP_BUILD_NODES.is_dir():
        return None
    for candidate in sorted(APP_BUILD_NODES.glob(LAYOUT_NODE_GLOB)):
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        if PATCHED_MARKER in text or OLD_FRAGMENT in text:
            return candidate
    return None


def in_container() -> bool:
    return APP_BUILD_NODES.is_dir()


def run_via_docker(container: str) -> int:
    """Re-exec this script inside the named container."""
    here = Path(__file__).resolve()
    cmd = [
        "docker", "exec", "-i", container, "python3", "-",
    ]
    print(f"[host] re-running self inside container '{container}'...")
    proc = subprocess.run(
        cmd, input=here.read_text(encoding="utf-8"), text=True, check=False,
    )
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--container", default="open-webui",
        help="Docker container name to exec into when run from the host.",
    )
    args = ap.parse_args()
    if not in_container():
        return run_via_docker(args.container)
    target = find_target()
    if target is None:
        print(
            f"[in-container] no candidate JS chunk found under {APP_BUILD_NODES}; "
            "either the path moved or this is a different OWUI build.",
            file=sys.stderr,
        )
        return 2
    status = patch_file(target)
    print(f"[in-container] {target}: {status}")
    return 0 if status in ("patched", "already-patched") else 3


if __name__ == "__main__":
    sys.exit(main())
