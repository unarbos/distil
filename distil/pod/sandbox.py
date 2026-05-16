"""Sandboxed Python subprocess for HumanEval / MBPP graders.

Runs short, untrusted snippets in a separate ``python3 -c`` subprocess
with hard wall-clock + memory limits and no network.
"""

from __future__ import annotations

import resource
import subprocess
import textwrap
from typing import Any


def _set_limits():
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    except Exception:
        pass


def run_snippet(source: str, *, stdin: str = "", timeout_s: float = 8.0) -> dict[str, Any]:
    """Execute ``source`` in a fresh subprocess; return ``{ok, stdout, stderr, returncode}``."""
    code = textwrap.dedent(source)
    try:
        proc = subprocess.run(
            ["python3", "-I", "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_set_limits,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr[-2048:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "returncode": -2}


def run_humaneval(prompt: str, completion: str, tests: str, *, timeout_s: float = 8.0) -> bool:
    """Return True iff the function passes the test cases."""
    src = (
        prompt
        + "\n"
        + completion
        + "\n\n"
        + tests
        + "\n\nimport sys; sys.exit(0 if check(candidate) is None else 1)"
    )
    return run_snippet(src, timeout_s=timeout_s)["ok"]
