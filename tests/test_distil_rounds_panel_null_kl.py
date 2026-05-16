"""Regression test for the 2026-05-16 Rounds tab crash.

When every non-king student in a recent round fails Phase 2 (HF 404,
vLLM init crash, ``max_position_embeddings`` < ``max_model_len``, etc.)
the API returns ``kl=null`` AND ``composite.worst=null`` for those rows.

The Rounds tab's ``Side`` component (rendered for the king and the top
challenger in every ``BoutCard``) previously did:

    {worst != null
        ? `worst ${worst.toFixed(3)}`
        : `KL ${result.kl.toFixed(4)}`}

With both ``worst`` and ``result.kl`` null this hit
``null.toFixed(4)`` and crashed the entire RoundsPanel React tree —
which is exactly what the user reported as "the rounds tab is down".
The TypeScript ``kl: number`` interface was lying; at runtime it can
be null.

We pin two invariants:

1.  The ``H2hResult.kl`` field is declared nullable.
2.  Every ``.toFixed(`` call on a ``kl`` value in ``rounds-panel.tsx``
    is guarded by a ``typeof ... === "number"`` / ``Number.isFinite``
    check (or equivalent null-check) so a null KL renders a fallback
    instead of throwing.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROUNDS_PANEL = REPO / "frontend" / "src" / "components" / "v2" / "rounds-panel.tsx"


def test_h2h_result_kl_is_nullable():
    """The ``H2hResult`` interface MUST declare ``kl`` as nullable.

    A non-nullable ``kl: number`` was the proximate cause of the
    Rounds tab crash: it made callers feel safe writing
    ``result.kl.toFixed(...)`` without a guard.
    """
    src = ROUNDS_PANEL.read_text()
    m = re.search(r"interface H2hResult\s*\{([^}]+)\}", src)
    assert m, "could not locate H2hResult interface in rounds-panel.tsx"
    body = m.group(1)
    kl_decl = re.search(r"kl:\s*([^;]+);", body)
    assert kl_decl, f"H2hResult.kl declaration missing in:\n{body}"
    kl_type = kl_decl.group(1).strip()
    assert "null" in kl_type, (
        f"H2hResult.kl MUST be nullable to reflect the API surface "
        f"(failed students return ``kl=null``); got: {kl_type!r}"
    )


def test_kl_tofixed_calls_are_null_guarded():
    """Every ``result.kl.toFixed(...)`` call in the Side component
    branch MUST be inside a ``typeof ... === "number"`` /
    ``Number.isFinite`` ternary so null KL falls through to a literal
    placeholder rather than crashing the React tree.
    """
    raw = ROUNDS_PANEL.read_text()
    # Strip ``//`` line comments and ``/* */`` block comments before
    # scanning so the test doesn't trip on doc-strings that legitimately
    # mention the bug pattern (e.g. the H2hResult interface comment
    # describing the 2026-05-16 crash).
    no_block = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", no_block)
    for m in re.finditer(r"(\w+)\.kl\.toFixed", src):
        var = m.group(1)
        start = max(0, m.start() - 200)
        window = src[start : m.end() + 50]
        guarded = (
            f"typeof {var}.kl" in window
            or f"Number.isFinite({var}.kl)" in window
            or f"{var}.kl != null" in window
            or f"{var}.kl !== null" in window
        )
        assert guarded, (
            f"unguarded ``{var}.kl.toFixed`` near offset {m.start()} — "
            f"a null kl will crash the React tree. Context:\n"
            f"---\n{window}\n---"
        )


def test_side_renders_fallback_for_null_kl_and_null_worst():
    """The exact branch that crashed must now render a literal
    fallback string when both ``worst`` and ``kl`` are null.
    """
    src = ROUNDS_PANEL.read_text()
    # The ``Side`` component renders the worst/KL line. We assert the
    # else-branch has a null-guard ternary that yields a literal
    # placeholder (e.g. "— no score") instead of an unguarded toFixed.
    side_block = re.search(
        r"function Side\(.*?\n\}\n", src, re.DOTALL
    )
    assert side_block, "could not locate Side(...) function body"
    body = side_block.group(0)
    assert "Number.isFinite(result.kl)" in body or "typeof result.kl" in body, (
        "Side(...) must null-guard result.kl before calling toFixed; "
        "missing isFinite/typeof check in body:\n" + body
    )
