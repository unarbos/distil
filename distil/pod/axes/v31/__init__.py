"""v31 procedural axis generators (verbatim copies of ``scripts/v31/``).

Each module is a self-contained pair (generator + grader):

* ``generate_items(block_seed, n_items) -> list[dict]`` — pure-Python
  procedural items keyed on ``block_seed`` so every validator sees the
  same items this round but no two rounds share an item.
* ``grade_response(text, gold[, all_correct]) -> bool`` — pure-Python
  scorer (except for the math axes which share ``distil.pod.axes._math``
  and ``code_humaneval_plus`` which delegates to the sandbox).

These modules are deliberately decoupled from vLLM / the engine API.
The thin scoring wrappers live one level up in ``distil.pod.axes.<axis>``.

Reference: ``reports/2026-05-09-v31-procedural-redesign.md``.
"""

from __future__ import annotations
