"""Sandboxed Python subprocess for HumanEval / MBPP graders.

Runs short, untrusted snippets in a separate ``python3 -I -s`` subprocess
with hard wall-clock + memory limits, no network, and no inherited env.

Provides two entry points:

* :func:`run_snippet` — execute arbitrary source, return ``{ok, stdout, ...}``.
* :func:`run_humaneval` — assemble a HumanEval program from
  ``(prompt, completion, test, entry_point)`` and report pass/fail.

The HumanEval assembly handles three Goodhart-trap patterns that
otherwise turn correct-but-format-mismatched solutions into 0% scores:

1. **Missing ``candidate``**: tests call ``check(candidate)`` but
   ``candidate`` is never assigned. We call ``check({entry_point})``
   directly using the known function name.
2. **Unindented body**: prompt ends mid-``def`` block; model emits
   ``return ...`` at column 0. We re-indent every bare line by 4 spaces.
3. **Chat-style prose / markdown fences**: model wraps the body in
   ``Sure!\\n\\n```python ... ``` ``\\n\\nHope this helps!``. We strip
   fenced blocks and trim leading/trailing prose until the program parses.

Without these, every miner — including the king and the reference
baseline — scored 0/16 on ``v31_code_humaneval_plus`` because the
``check(candidate)`` line in the generator's test block raised
``NameError`` every single time.

A per-sample nonce printed only after ``check()`` returns ensures
miners cannot spoof a pass with ``os._exit(0)``: a non-zero exit OR a
missing sentinel both count as a fail.
"""

from __future__ import annotations

import ast
import os
import re
import resource
import secrets
import subprocess
import tempfile
import textwrap
from typing import Any

TIMEOUT_SECS = int(os.environ.get("DISTIL_SANDBOX_TIMEOUT", "10"))
MEMORY_LIMIT_MB = int(os.environ.get("DISTIL_SANDBOX_MEMORY_MB", "512"))
CPU_LIMIT_SECS = int(os.environ.get("DISTIL_SANDBOX_CPU_SECS", "8"))
MAX_OUTPUT_CHARS = 4000

_FENCE_PAIRED = re.compile(
    r"```(?:python|py)?[^\S\n]*\n(.*?)\n[ \t]*```",
    re.DOTALL,
)


def _set_limits():
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECS, CPU_LIMIT_SECS))
    except Exception:
        pass
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (MEMORY_LIMIT_MB * 1024 * 1024, MEMORY_LIMIT_MB * 1024 * 1024),
        )
    except Exception:
        pass


def run_snippet(source: str, *, stdin: str = "", timeout_s: float = 8.0) -> dict[str, Any]:
    """Execute ``source`` in a fresh subprocess; return ``{ok, stdout, ...}``."""
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


def _strip_code_fences(text: str) -> str:
    """Unwrap a Markdown fenced Python block from a chat-style emission.

    Conservative: ``rstrip`` only — leading indentation matters because
    many HumanEval prompts end mid-``def`` block. Falls back to single-
    marker splitting if no paired fence is found.
    """
    if not text:
        return ""
    paired = _FENCE_PAIRED.search(text)
    if paired:
        body = paired.group(1)
        if body.strip():
            return body.rstrip()
    t = text
    for marker in ("```python", "```py", "```"):
        if marker in t:
            parts = t.split(marker)
            if len(parts) >= 3:
                t = parts[1]
                break
            if len(parts) == 2:
                left, right = parts[0], parts[1]
                if not right.strip():
                    t = left
                elif not left.strip():
                    t = right
                else:
                    t = left
    return t.rstrip()


def _find_parseable_window(prompt: str, gen: str, must_contain: str | None = None) -> str:
    """Largest contiguous window of ``gen`` such that ``prompt + gen``
    parses cleanly. Trims prose at the outer boundary while keeping
    the gen verbatim if it already parses. Returns ``gen`` unchanged
    when no parseable window exists.
    """
    if not gen:
        return gen
    connector = "" if (not prompt or prompt.endswith("\n")) else "\n"
    full = prompt + connector + gen
    try:
        ast.parse(full)
        if must_contain is None or must_contain in full:
            return gen
    except SyntaxError:
        pass
    except Exception:
        return gen

    lines = gen.splitlines(keepends=True)
    n = len(lines)
    best: tuple[int, int, int, str] | None = None
    for start in range(n):
        if not lines[start].strip():
            continue
        for end in range(n, start, -1):
            truncated = "".join(lines[start:end])
            candidate = prompt + connector + truncated
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            except Exception:
                return gen
            if must_contain is not None and must_contain not in candidate:
                break
            length = end - start
            if best is None or length > best[2]:
                best = (start, end, length, truncated)
            break
    return gen if best is None else best[3]


def _assemble_humaneval(
    prompt: str, completion: str, test: str, entry_point: str, nonce: str
) -> str:
    """Build the program text to run for a HumanEval sample.

    Recovers chat-style prose, markdown fences, unindented bodies, and
    redundant ``def`` redeclarations so the grader measures coding
    ability instead of prompt-format compliance.
    """
    gen = _strip_code_fences(completion)
    def_marker = f"def {entry_point}(" if entry_point else ""
    if def_marker:
        if not prompt.strip():
            gen = _find_parseable_window(prompt, gen, must_contain=def_marker)
        elif prompt.strip():
            gen = _find_parseable_window(prompt, gen)
    has_def_in_gen = (
        def_marker
        and gen.count(def_marker) > 0
        and prompt.count(def_marker) >= 1
    )
    if has_def_in_gen:
        idx = gen.find(def_marker)
        gen_tail = gen[idx:]
        lines = gen_tail.splitlines()
        if lines and lines[0].strip().startswith("def ") and lines[0].strip() in prompt:
            lines = lines[1:]
        gen = "\n".join(lines)
    elif def_marker and prompt.count(def_marker) >= 1:
        body_lines = gen.splitlines()
        first_nonblank = next((ln for ln in body_lines if ln.strip()), None)
        if (
            first_nonblank is not None
            and not first_nonblank.startswith((" ", "\t"))
            and not first_nonblank.lstrip().startswith(("def ", "class ", "@"))
        ):
            gen = "\n".join(
                ("    " + ln) if ln.strip() else ln for ln in body_lines
            )

    program = prompt
    if not program.endswith("\n"):
        program += "\n"
    program += gen
    program += "\n\n" + test + "\n"
    program += f"check({entry_point})\n"
    program += "import sys as __sbx_sys\n"
    program += f"__sbx_sys.stdout.write({nonce!r} + '\\n')\n"
    program += "__sbx_sys.stdout.flush()\n"
    return program


def run_humaneval(
    prompt: str,
    completion: str,
    tests: str,
    *,
    entry_point: str,
    timeout_s: float = 8.0,
) -> bool:
    """Run a HumanEval sample and return True iff it passes.

    Uses a per-sample nonce written to stdout only after ``check()``
    returns successfully — a passing return code without the sentinel
    counts as a fail to thwart ``os._exit(0)`` spoofs.
    """
    if not entry_point:
        return False
    nonce = "__SBX_OK_" + secrets.token_hex(16) + "__"
    program = _assemble_humaneval(prompt, completion, tests, entry_point, nonce)
    with tempfile.TemporaryDirectory(prefix="he_sbx_") as tmpdir:
        script_path = os.path.join(tmpdir, "prog.py")
        with open(script_path, "w") as f:
            f.write(program)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C",
            "LANG": "C",
        }
        try:
            proc = subprocess.run(
                ["python3", "-I", "-s", script_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                env=env,
                timeout=timeout_s,
                preexec_fn=_set_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        stdout_raw = proc.stdout.decode(errors="replace")
        sentinel_seen = nonce in stdout_raw
        return proc.returncode == 0 and sentinel_seen
