"""consistency_paraphrase - v31 paraphrase-pair consistency axis.

Each item is a (base, paraphrased) pair built from M1 GSM-Symbolic
templates so the gold is mechanically computable. Score: 1.0 if the
model answers both correctly, 0.5 if only one, 0.0 if neither.

Paraphrase generator combines (randomized):
* **Isomorphic name rotation** (IPT defence, RLVR reward-hacking paper
  arXiv 2604.15149) — swap each first name with another from the same
  gender inventory.
* Unit relabeling (dollars -> euros, miles -> km).
* Connector swaps ("then" -> "after that").
* Irrelevant context prepended.
* Word-order changes on the final question.
* arXiv:2603.16197 (Mar 2026): "Lexical and behavioural
  contamination signatures across SOTA frontier models" -
  finding paraphrase-induced accuracy drops as evidence of
  memorization.
* arXiv:2604.15149 (Apr 2026): "LLMs Gaming Verifiers: RLVR can
  Lead to Reward Hacking" — shows RLVR-trained frontier models
  do "instance enumeration over rule learning" and fail under
  isomorphic perturbation; recommends IPT as a defence.
"""

from __future__ import annotations

import random
import re

from distil.pod.axes.v31.math_gsm_symbolic import (
    TEMPLATES as _M1_TEMPLATES,
    _NAMES_M as _M1_NAMES_M,
    _NAMES_F as _M1_NAMES_F,
)


_BENCH_STREAM_OFFSET = 0x563A  # "VA"


# ─────────────────────────────────────────────────────────────────────
#  Paraphrase transforms.
# ─────────────────────────────────────────────────────────────────────


_CONNECTOR_SUBS = (
    (" then ", " after that "),
    (" but ", " however "),
    (" and ", " ; in addition "),
    (" because ", " since "),
    (" so ", " thus "),
)

_PHRASING_SUBS = (
    ("How many", "What is the total number of"),
    ("How much", "What is the total amount of"),
    ("What is", "Find"),
    ("Find", "Compute"),
    ("does", "would"),
)

_UNIT_SUBS = (
    ("$", "€"),
    (" dollars", " euros"),
    (" mph", " km/h"),
    (" miles", " kilometres"),
)

_PREFIX_FILLER = (
    "On a quiet morning,",
    "Earlier in the day,",
    "After running a few errands,",
    "Just before noon,",
    "In a small town nearby,",
)


def _rotate_names(question: str, rng: random.Random) -> str:
    """Replace each first-name occurrence in ``question`` with a
    different name from the *same* gender inventory.

    This is the IPT (Isomorphic Perturbation Test) defence advocated
    by the 2026 RLVR reward-hacking literature: a model that has
    learned the underlying procedure should be invariant to the
    actor's identity, while a model that has memorised template
    instances will degrade. Critically, the gold answer never
    depends on the name — the rotation is a strict surface change.

    Implementation notes:

    * We word-tokenise on ``\\b`` boundaries so we don't replace
      partial matches (no substring like "Iris" inside "Iriserved").
    * Each appearance of the same source name maps consistently to
      the same target name within one paraphrase pass — so
      "Alice ... Alice" stays a single actor.
    * Replacement target is sampled disjoint from the names
      currently in the question, to avoid degenerate "Alice → Alice".
    """
    src_names_m = set(_M1_NAMES_M)
    src_names_f = set(_M1_NAMES_F)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for m in re.finditer(r"\b([A-Z][a-z]+)\b", question):
        nm = m.group(1)
        if nm in mapping:
            continue
        if nm in src_names_m:
            pool = [n for n in _M1_NAMES_M if n != nm and n not in used]
        elif nm in src_names_f:
            pool = [n for n in _M1_NAMES_F if n != nm and n not in used]
        else:
            continue
        if not pool:
            continue
        repl = rng.choice(pool)
        mapping[nm] = repl
        used.add(repl)
    if not mapping:
        return question
    # Apply all substitutions in a single pass so chained mappings
    # like {Alice: Bella, Bella: Quinn} can't collide. We compile one
    # alternation and rewrite each match via a lookup.
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in mapping.keys()) + r")\b"
    )
    return pattern.sub(lambda m: mapping[m.group(1)], question)


def _paraphrase(question: str, rng: random.Random) -> str:
    """Apply a procedural paraphrase to ``question`` while preserving
    semantic content (and gold answer).

    Always produces a non-identity output: if all the random gates
    happen to be no-ops, we fall back to forcing a prefix insertion
    so the surface form is guaranteed to differ from the input.
    """
    out = question
    if rng.random() < 0.85:
        out = _rotate_names(out, rng)
    for pat, repl in rng.sample(_CONNECTOR_SUBS, len(_CONNECTOR_SUBS)):
        if pat in out and rng.random() < 0.5:
            out = out.replace(pat, repl, 1)
    if rng.random() < 0.7:
        for pat, repl in rng.sample(_PHRASING_SUBS, len(_PHRASING_SUBS)):
            if pat in out:
                out = out.replace(pat, repl, 1)
                break
    if rng.random() < 0.5:
        for pat, repl in _UNIT_SUBS:
            if pat in out:
                out = out.replace(pat, repl)
                break
    if rng.random() < 0.6 or out == question:
        prefix = rng.choice(_PREFIX_FILLER)
        parts = out.split("\n\nSolve step by step", 1)
        if len(parts) == 2:
            out = prefix + " " + parts[0].lstrip() + "\n\nSolve step by step" + parts[1]
    if out == question:
        prefix = rng.choice(_PREFIX_FILLER)
        out = prefix + " " + question
    return out


# ─────────────────────────────────────────────────────────────────────
#  Generator.
# ─────────────────────────────────────────────────────────────────────


def generate_items(block_seed, n_items: int) -> list[dict]:
    """Generate ``n_items`` paraphrase-pair items.

    Each item carries TWO questions (`question_a`, `question_b`) and
    one gold value. Downstream the validator generates one response
    for each variant, then computes a 3-way consistency score.

    For backward compatibility with the existing single-question
    bench probe scaffold, we emit items where ``question`` is
    ``question_a`` and ``question_b`` is stored in metadata. The
    bench probe (``v31_consistency_paraphrase_bench_probe``) reads
    both and scores accordingly.
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        spec = item_rng.choice(_M1_TEMPLATES)
        difficulty = item_rng.choices([0, 1], weights=[0.65, 0.35])[0]
        question_a, gold = spec.fn(item_rng, difficulty)
        question_b = _paraphrase(question_a, item_rng)
        out.append(
            {
                "src": f"v31_consistency_paraphrase/{spec.name}/p{difficulty}",
                "question": question_a,
                "question_b": question_b,
                "gold": str(int(gold)),
                "template": spec.name,
                "difficulty": difficulty,
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
#  Consistency scorer.
#
#  Per-item score:
#    1.0 - both correct
#    0.5 - exactly one correct
#    0.0 - neither correct
#
#  Note: this differs from the pure pass_frac metric used by the
#  other v31 axes - we explicitly want to penalize models that get
#  the right answer on one phrasing and wrong on the other (since
#  that's evidence of surface-form fragility / partial memorization).
# ─────────────────────────────────────────────────────────────────────


def consistency_score(response_a: str, response_b: str, gold: str) -> float:
    """Score a paraphrase pair. Uses the math-style answer extractor
    from the existing pipeline (re-implemented locally to keep this
    module standalone)."""
    a_correct = _math_match(response_a, gold)
    b_correct = _math_match(response_b, gold)
    if a_correct and b_correct:
        return 1.0
    if a_correct or b_correct:
        return 0.5
    return 0.0


def _math_match(response: str, gold: str) -> bool:
    if not response:
        return False
    import re
    target = str(gold).strip()
    text = response.strip()
    m = re.search(r"####\s*(-?\d+)", text)
    if m and m.group(1) == target:
        return True
    m = re.search(r"final\s+answer\s*[:\-]?\s*(-?\d+)", text, re.IGNORECASE)
    if m and m.group(1) == target:
        return True
    nums = re.findall(r"-?\d+", text)
    if nums and nums[-1] == target:
        return True
    return False


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=4)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']!r}")
        print("VARIANT A:")
        print(it["question"])
        print("VARIANT B:")
        print(it["question_b"])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
