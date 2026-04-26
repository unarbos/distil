"""HumanEval subprocess sandbox — SN97 edition.

The official ``human-eval`` library runs candidate code with raw
``exec()`` inside the evaluator process. That is not safe for our use
case: miner-trained models can emit arbitrary Python, and if it crashes
(segfault, OOM, infinite alloc, ``os._exit``) the whole eval loop dies.

This module runs each candidate in a fresh ``python3 -I -s``
subprocess with:

- hard wall-clock timeout (``timeout=TIMEOUT_SECS``), kill-tree on hit
- per-process RLIMITs: CPU time, address-space, file-descriptors, file-size
- CWD set to a fresh tempdir, removed after
- No inherited env vars except ``PATH`` and ``LC_ALL=C``
- No stdin
- stdout/stderr captured for logging

Exit code 0 (all asserts pass) → pass. Any non-zero (including timeout,
signal, limit exceeded) → fail, with the reason recorded.

Usage:

    from humaneval_sandbox import run_sample

    result = run_sample(prompt, generation, test, entry_point)
    # result.passed : bool
    # result.reason : str  ("pass" / "timeout" / "error:<first line>")
    # result.stderr : str  (captured stderr tail, for debug)

``run_batch(samples)`` runs a list concurrently via a ThreadPool; each
sample gets its own subprocess so they're already isolated from one
another. Concurrency is bounded by ``max_workers`` (default: 4).
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import secrets
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass

TIMEOUT_SECS = int(os.environ.get("HUMANEVAL_SANDBOX_TIMEOUT", "10"))
MEMORY_LIMIT_MB = int(os.environ.get("HUMANEVAL_SANDBOX_MEMORY_MB", "512"))
CPU_LIMIT_SECS = int(os.environ.get("HUMANEVAL_SANDBOX_CPU_SECS", "8"))
MAX_OUTPUT_CHARS = 4000


@dataclass
class SandboxResult:
    passed: bool
    reason: str
    stderr: str = ""
    stdout: str = ""


_RLIMIT_PREAMBLE = textwrap.dedent(f"""\
    import resource, sys
    try:
        resource.setrlimit(resource.RLIMIT_CPU, ({CPU_LIMIT_SECS}, {CPU_LIMIT_SECS}))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_LIMIT_MB} * 1024 * 1024, {MEMORY_LIMIT_MB} * 1024 * 1024))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    except Exception:
        pass
    import signal
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(99))
    signal.alarm({CPU_LIMIT_SECS})
""")


def _strip_code_fences(text: str) -> str:
    r"""Unwrap a Markdown fenced Python block from a chat-style emission.

    Handles three robust formats:

    1. Prose intro + paired ``` fences + prose outro (regex captures
       the inner body, drops both prose sides).
    2. Just an opening fence (no closing) — drop the fence prefix,
       keep the rest.
    3. Just a closing fence — drop the fence and anything after it,
       keep the code that came before it.

    Round 18 (2026-04-26 Goodhart hardening) hardened this in two ways:
    - Added the paired-fence regex so chatty models that wrap code in
      ``Sure!\n\n```python ... ```\nDone!`` extract the inner body
      cleanly. The older single-marker logic mishandled this case by
      re-treating the closing fence as another opening fence.
    - Disambiguated the single-marker fallback when both sides of the
      split are non-empty: prefer the side that *precedes* a bare
      ``\`\`\`\`\`\` `` (treated as a trailing fence + prose), since
      the paired regex already handled the symmetric case.

    We only ``rstrip`` the result — leading indentation matters because
    many HumanEval prompts end mid-function-body (closing docstring at
    4-space indent) and the generation must continue that indent level.
    """
    if not text:
        return ""
    t = text

    import re
    paired = re.search(
        r"```(?:python|py)?[^\S\n]*\n(.*?)\n[ \t]*```",
        t,
        re.DOTALL,
    )
    if paired:
        body = paired.group(1)
        if body.strip():
            return body.rstrip()

    for marker in ("```python", "```py", "```"):
        if marker in t:
            parts = t.split(marker)
            if len(parts) >= 3:
                t = parts[1]
                break
            elif len(parts) == 2:
                left, right = parts[0], parts[1]
                if not right.strip():
                    t = left
                elif not left.strip():
                    t = right
                else:
                    t = left
    return t.rstrip()


def _find_parseable_gen_window(
    prompt: str,
    gen: str,
    must_contain: str | None = None,
) -> str:
    """Find the largest contiguous line range of ``gen`` such that
    ``prompt + connector + gen[range]`` parses cleanly with ``ast.parse``.

    Goodhart hardening (2026-04-26 round 18): models occasionally wrap
    correct code in chat-style prose ("Sure, here's the function:" /
    "Hope this helps!"). When fed to the sandbox verbatim those
    prefixes/suffixes raise ``SyntaxError`` instead of letting the test
    harness exercise the actual function — confirmed in a synthetic
    repro where logically-correct ``def is_sorted(...)`` failed because
    of a leading "Sure! Here is..." line. That penalises real coding
    skill on the basis of pedantic instruction-following (which we
    already grade separately via ``ifeval_bench``), so it is a textbook
    Goodhart vector.

    Two modes:

    * Empty ``prompt`` (MBPP): the gen contains a complete function
      definition, possibly wrapped in prose. We look for the largest
      contiguous parseable window that contains ``must_contain``
      (e.g. ``def {entry_point}(``).

    * Non-empty ``prompt`` (HumanEval): the prompt ends mid-function
      body, so the gen alone cannot parse standalone. We look for the
      largest window of ``gen`` that, when concatenated to ``prompt``,
      produces a parseable program.

    Conservative: never invents code or rearranges statements; only
    trims prose at the outer boundary. If no parseable window exists,
    the original gen is returned unchanged.
    """
    if not gen:
        return gen
    try:
        import ast
    except Exception:
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
    if best is None:
        return gen
    return best[3]


def _extract_python_block(text: str, must_contain: str | None = None) -> str:
    """Backwards-compatible wrapper for the empty-prompt MBPP case.

    Equivalent to ``_find_parseable_gen_window("", text, must_contain)``.
    Kept as a separate name because the MBPP test suite imports it
    directly and the empty-prompt semantics are conceptually distinct
    from the HumanEval prose-trim."""
    return _find_parseable_gen_window("", text, must_contain=must_contain)


def _assemble_program(prompt: str, generation: str, test: str, entry_point: str,
                       nonce: str) -> str:
    """Build the full program text to run in the sandbox.

    HumanEval expects:
        <prompt (function signature + docstring, no body)>
        <generation (the body + any additional code)>
        <test (defines check(candidate))>
        check({entry_point})

    A per-sample ``nonce`` is printed only after ``check()`` returns
    successfully, so miners cannot spoof a pass with ``os._exit(0)``.

    Format-recovery (2026-04-26 Goodhart hardening): the bench prompt
    instructs models to output "only the function body". Many models
    comply but emit the body **without leading indentation** — when
    concatenated to a prompt that ends inside a ``def`` block, that's
    a SyntaxError instead of a real capability signal (Round 15 audit
    on Qwen/Qwen3.5-4B: 2 of 4 failures were ``return outside function``
    on bare ``return ...`` outputs). We add an auto-indent recovery so
    the grader measures coding ability, not prompt-format compliance.
    A bare ``return ...`` without indentation cannot mean anything else
    when continuing a HumanEval prompt that ends mid-``def`` block, so
    the recovery only ever fixes a format mistake — it never changes a
    semantically valid program.

    Prose-stripping (2026-04-26 round 18, both MBPP and HumanEval):
    chatty models write things like "Sure! Here is the function:\\n
    \\ndef foo():..." or "...\\n\\nHope this helps!" — both trip
    ``SyntaxError`` even though the code is correct, again penalising
    coding ability instead of measuring it. We run
    ``_find_parseable_gen_window`` once before the auto-indent path
    (HumanEval) and once for the empty-prompt path (MBPP) to drop
    prose at the outer boundary while keeping the gen verbatim if it
    already parses. The MBPP pass requires the entry-point ``def`` to
    survive the trim; the HumanEval pass relies on ``prompt + gen``
    parsing as a whole and is naturally guarded by the prompt's open
    ``def`` block.
    """
    gen = _strip_code_fences(generation)
    def_marker = f"def {entry_point}("
    if not prompt.strip() and entry_point:
        gen = _find_parseable_gen_window(prompt, gen, must_contain=def_marker)
    elif prompt.strip() and entry_point:
        gen = _find_parseable_gen_window(prompt, gen)
    has_def_in_gen = gen.count(def_marker) > 0 and prompt.count(def_marker) >= 1
    if has_def_in_gen:
        idx = gen.find(def_marker)
        gen_tail = gen[idx:]
        lines = gen_tail.splitlines()
        if lines and lines[0].strip().startswith("def ") and lines[0].strip() in prompt:
            lines = lines[1:]
        gen = "\n".join(lines)
    elif prompt.count(def_marker) >= 1:
        # Auto-indent recovery only fires when the prompt itself ends
        # mid-``def`` block (HumanEval-style "complete the body" prompts).
        # If the prompt is empty or the body is unindented, the gen is a
        # bare body that needs the indentation reattached. We skip this
        # branch when the prompt has no ``def {entry}(`` (e.g. MBPP-style
        # tests that pass ``prompt=""`` and a complete function in gen).
        body_lines = gen.splitlines()
        first_nonblank = next(
            (line for line in body_lines if line.strip()), None
        )
        if (
            first_nonblank is not None
            and not first_nonblank.startswith((" ", "\t"))
            and not first_nonblank.lstrip().startswith(("def ", "class ", "@"))
        ):
            gen = "\n".join(
                ("    " + line) if line.strip() else line
                for line in body_lines
            )
    program = _RLIMIT_PREAMBLE
    program += prompt
    if not prompt.endswith("\n"):
        program += "\n"
    program += gen
    program += "\n\n"
    program += test
    # The nonce is randomly generated per sample and is not present
    # anywhere in the prompt or test code. Miners can only emit it if
    # they actually run the test harness.
    program += (
        f"\n\ncheck({entry_point})\n"
        "import sys as __sbx_sys\n"
        f"__sbx_sys.stdout.write({nonce!r} + '\\n')\n"
        "__sbx_sys.stdout.flush()\n"
    )
    return program


def run_sample(prompt: str, generation: str, test: str, entry_point: str) -> SandboxResult:
    """Run one HumanEval sample in an isolated subprocess."""
    nonce = "__SBX_OK_" + secrets.token_hex(16) + "__"
    program = _assemble_program(prompt, generation, test, entry_point, nonce)
    with tempfile.TemporaryDirectory(prefix="he_sbx_") as tmpdir:
        script_path = os.path.join(tmpdir, "prog.py")
        with open(script_path, "w") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "LC_ALL": "C", "LANG": "C"}
        try:
            proc = subprocess.run(
                ["python3", "-I", "-s", script_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                env=env,
                timeout=TIMEOUT_SECS,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                passed=False,
                reason="timeout",
                stderr=(e.stderr or b"").decode(errors="replace")[-MAX_OUTPUT_CHARS:],
                stdout=(e.stdout or b"").decode(errors="replace")[-MAX_OUTPUT_CHARS:],
            )
        except Exception as e:
            return SandboxResult(passed=False, reason=f"spawn_error:{str(e)[:120]}")
        stdout_raw = proc.stdout.decode(errors="replace")
        stderr_raw = proc.stderr.decode(errors="replace")
        stdout = stdout_raw[-MAX_OUTPUT_CHARS:]
        stderr = stderr_raw[-MAX_OUTPUT_CHARS:]
        sentinel_seen = nonce in stdout_raw
        if proc.returncode == 0 and sentinel_seen:
            return SandboxResult(passed=True, reason="pass", stderr=stderr, stdout=stdout)
        if proc.returncode == 0 and not sentinel_seen:
            return SandboxResult(
                passed=False,
                reason="fail:no_sentinel (tests did not run to completion)",
                stderr=stderr,
                stdout=stdout,
            )
        first_err = ""
        if stderr:
            for line in reversed(stderr.splitlines()):
                line = line.strip()
                if line:
                    first_err = line[:200]
                    break
        return SandboxResult(
            passed=False,
            reason=f"fail:{first_err or f'exit={proc.returncode}'}",
            stderr=stderr,
            stdout=stdout,
        )


def run_batch(samples: list[tuple[str, str, str, str]],
              max_workers: int = 4) -> list[SandboxResult]:
    """Run a batch of (prompt, generation, test, entry_point) tuples."""
    if not samples:
        return []
    results: list[SandboxResult | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_sample, *s): i for i, s in enumerate(samples)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = SandboxResult(passed=False, reason=f"runner_error:{str(e)[:120]}")
    return [r if r is not None else SandboxResult(passed=False, reason="lost") for r in results]


if __name__ == "__main__":
    demo_prompt = (
        "from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Check if any two numbers are closer than threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    \"\"\"\n"
    )
    demo_gen = (
        "    for i, a in enumerate(numbers):\n"
        "        for b in numbers[i+1:]:\n"
        "            if abs(a - b) < threshold:\n"
        "                return True\n"
        "    return False\n"
    )
    demo_test = (
        "def check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
        "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
    )
    res = run_sample(demo_prompt, demo_gen, demo_test, "has_close_elements")
    print("passed:", res.passed, "reason:", res.reason)
    print("stderr:", res.stderr[:200])
