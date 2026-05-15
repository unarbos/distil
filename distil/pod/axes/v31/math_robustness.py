"""math_robustness - v31 GSM-Plus + GSM-NoOp axis.

Five mechanically-graded perturbations layered on top of
``v31_math_gsm_symbolic`` so a memoriser of canonical GSM wording
fails here. References: GSM-Plus (Li et al., ACL 2024) and GSM-NoOp
(Mirzadeh et al., Apple 2024).

Perturbations (all with mechanically derivable gold):
* ``numerical_swap``: re-sample variable values; template recomputes
  gold. Tests robustness to specific number choices.
* ``digit_expand``: multiply all numbers by 10x/100x; gold scales.
  Tests larger-magnitude digits.
* ``context_pad``: prepend irrelevant story sentences. Gold unchanged.
* ``unit_swap``: relabel units (dollars->euros, miles->km, ...).
  Gold unchanged.
* ``topical_distractor``: inject a topically-relevant but math-
  irrelevant clause carrying a number that should NOT participate
  (the GSM-NoOp signal — up to 65 pp drop on frontier models).

GSM-Plus "critical thinking" lives in ``v31_truthfulness_calibration``
(needs hand-annotation per template); "problem reversal" excluded as
high-effort / low-yield.

References:
* Li, Q., et al. (2024). "GSM-Plus: A Comprehensive Benchmark
  for Evaluating the Robustness of LLMs as Mathematical Problem
  Solvers." ACL 2024. arXiv:2402.19255.
* Mirzadeh, I., et al. (2024). "GSM-Symbolic / GSM-NoOp"
  (Apple). arXiv:2410.05229.
* Cobbe, K., et al. (2021). GSM8K original.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from distil.pod.axes.v31.math_gsm_symbolic import (
    TEMPLATES as _M1_TEMPLATES,
    _format_question as _m1_format_question,
)


# Per-round perturbation distribution. The GSM-NoOp topical
# distractor gets the largest individual share (0.30) because the
# Apple paper documents it as the single largest robustness
# differentiator (up to 65 pp drop on frontier models); the other
# perturbations are 0.10–0.20 weight.
_PERTURBATIONS: tuple[str, ...] = (
    "numerical_swap",
    "digit_expand",
    "context_pad",
    "unit_swap",
    "topical_distractor",
)
_PERTURB_RATIO: tuple[float, ...] = (0.20, 0.15, 0.20, 0.15, 0.30)


_BENCH_STREAM_OFFSET = 0x5634  # "V4"


def _sample_perturbation(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for p, w in zip(_PERTURBATIONS, _PERTURB_RATIO):
        cum += w
        if r < cum:
            return p
    return _PERTURBATIONS[-1]


# ─────────────────────────────────────────────────────────────────────
#  Perturbation 1: numerical_swap
#
#  We just re-call the underlying M1 template. The template's RNG
#  produces fresh number samples so gold is mechanically recomputed.
#  This is operationally identical to "another GSM-Symbolic item"
#  but keeping the namespace helps us sliced telemetry: if the M1
#  axis pass-rate is higher than this perturbation, that means the
#  model has "preferred" number ranges (a weak form of memorisation).
# ─────────────────────────────────────────────────────────────────────


def _gen_numerical_swap(rng: random.Random) -> tuple[str, int, str]:
    spec = rng.choice(_M1_TEMPLATES)
    difficulty = rng.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
    question, gold = spec.fn(rng, difficulty)
    return question, int(gold), spec.name


# ─────────────────────────────────────────────────────────────────────
#  Perturbation 2: digit_expand
#
#  Multiply every standalone integer in the question by a fixed
#  scale factor (10 or 100) and re-add the answer-marker boilerplate.
#  Most M1 templates compute gold as a sum of (count × price) terms
#  with optional percentage rescaling. Those templates are linear in
#  counts AND prices, but percentage-rescaling is *not* linear when
#  numbers grow; so we restrict digit_expand to scaling counts only,
#  not percentages or per-unit rates that are explicitly multiplied
#  together. To keep this simple and provably gold-correct, we
#  re-run the underlying template with a "scale" knob:
#  * scale = 10 multiplies counts (n_a, n_b) by 10 and gold by 10
#    while keeping prices fixed.
#  This is implemented by post-processing: we run the template
#  normally, then surface-substitute "{n} <noun>" patterns in the
#  question with "{n*scale} <noun>" and gold *= scale. We only do
#  this for counts associated with plural-item nouns (the most
#  reliably scalable quantity in our templates).
#
#  When ambiguous, we fall back to numerical_swap to guarantee gold
#  remains correct.
# ─────────────────────────────────────────────────────────────────────


_SCALABLE_NOUNS = {
    "apples", "bagels", "bananas", "books", "candles", "cookies", "cupcakes",
    "donuts", "erasers", "magazines", "muffins", "notebooks", "oranges",
    "pencils", "pens", "postcards", "scones", "stickers", "tickets",
    "envelopes", "boxes", "widgets", "flyers", "labels", "marbles",
    "tomatoes", "peppers", "cucumbers", "zucchini", "onions",
    "loaves", "cakes", "pies", "tarts", "students", "supplies",
}


def _scale_counts_in_question(q: str, scale: int) -> tuple[str, int]:
    """Replace patterns like '<int> <scalable_noun>' by scaling the int.

    Returns ``(new_question, multiplier)``. If multiplier == 1, the
    perturbation didn't change anything and the caller should fall
    back to numerical_swap.
    """
    # Match an integer followed by one of the scalable nouns
    # (case-insensitive, word-boundary).
    nouns_alt = "|".join(re.escape(n) for n in sorted(_SCALABLE_NOUNS, key=len, reverse=True))
    pattern = re.compile(rf"\b(\d+)\s+({nouns_alt})\b", re.IGNORECASE)
    matches = list(pattern.finditer(q))
    if not matches:
        return q, 1
    # Replace all in one pass (right-to-left to avoid offset shift).
    out = q
    for m in reversed(matches):
        n = int(m.group(1))
        new_n = n * scale
        out = out[:m.start(1)] + str(new_n) + out[m.end(1):]
    return out, scale


def _gen_digit_expand(rng: random.Random) -> tuple[str, int, str] | None:
    spec = rng.choice(_M1_TEMPLATES)
    difficulty = 0  # keep p0 only; p1/p2 layer extra clauses that may not scale linearly
    question, gold = spec.fn(rng, difficulty)
    scale = rng.choice([10, 100])
    new_q, mult = _scale_counts_in_question(question, scale)
    if mult == 1:
        return None  # caller should re-sample
    new_gold = int(gold) * mult
    if "How many" not in new_q and "how many" not in new_q:
        return None  # only count-style questions scale linearly
    # Sanity: only templates whose gold is dominated by the count term
    # actually scale linearly. Restrict to a known-safe subset.
    if spec.name not in {
        "classroom_supplies", "garden_harvest", "bakery_orders",
        "library_books", "work_rate",
    }:
        return None
    return new_q, new_gold, spec.name + "/scaled"


# ─────────────────────────────────────────────────────────────────────
#  Perturbation 3: context_pad
#
#  Prepend one or two irrelevant narrative sentences. Gold unchanged.
#  This tests whether the model can ignore irrelevant context (a
#  short-context 4B failure mode well-documented in RULER, NIAH, and
#  GSM-Plus).
# ─────────────────────────────────────────────────────────────────────


_CONTEXT_PAD_SNIPPETS = (
    "It is a sunny Saturday afternoon and the streets are crowded.",
    "Earlier in the day there was a small parade in the town square.",
    "The weather has been unusually warm for this time of year.",
    "The bus route between downtown and the suburbs runs every fifteen minutes.",
    "A documentary about local history was filmed nearby last week.",
    "A neighbourhood dog walker has just passed by the door.",
    "The news mentioned that several construction projects are wrapping up.",
)


def _gen_context_pad(rng: random.Random) -> tuple[str, int, str]:
    spec = rng.choice(_M1_TEMPLATES)
    difficulty = rng.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
    question, gold = spec.fn(rng, difficulty)
    pad_count = rng.choice([1, 2])
    pad = " ".join(rng.sample(_CONTEXT_PAD_SNIPPETS, pad_count))
    # Splice before the existing problem text.
    parts = question.split("\n\nSolve step by step", 1)
    if len(parts) != 2:
        return question, int(gold), spec.name + "/padded"
    new_q = pad + "\n\n" + parts[0] + "\n\nSolve step by step" + parts[1]
    return new_q, int(gold), spec.name + "/padded"


# ─────────────────────────────────────────────────────────────────────
#  Perturbation 4: unit_swap
#
#  Replace "dollars / $" -> "euros / €" or "pounds / £", and
#  "miles / mph" -> "kilometres / kph". Gold unchanged because we're
#  re-labeling units, not changing values.
# ─────────────────────────────────────────────────────────────────────


_UNIT_SWAPS: tuple[tuple[Callable[[str], str], str], ...] = (
    (
        lambda q: q.replace("$", "€").replace(" dollars", " euros"),
        "euros",
    ),
    (
        lambda q: q.replace("$", "£").replace(" dollars", " pounds"),
        "pounds",
    ),
    (
        lambda q: q.replace(" mph ", " km/h ").replace(" miles", " kilometres"),
        "metric",
    ),
)


def _gen_unit_swap(rng: random.Random) -> tuple[str, int, str]:
    spec = rng.choice(_M1_TEMPLATES)
    difficulty = rng.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
    question, gold = spec.fn(rng, difficulty)
    swap_fn, tag = rng.choice(_UNIT_SWAPS)
    new_q = swap_fn(question)
    return new_q, int(gold), spec.name + "/" + tag


# ─────────────────────────────────────────────────────────────────────
#  Perturbation 5: topical_distractor (GSM-NoOp / Apple 2024).
#
#  Inject a topically-relevant clause that contains a number which
#  is **not** a problem variable. The Apple GSM-NoOp paper shows
#  that frontier models drop up to 65 pp on this perturbation
#  because they treat any number in the prompt as a calculation
#  input. We extract one of the scalable nouns the question already
#  uses (e.g. "bagels", "books") and append a clause referencing
#  that noun and a number — but the clause is grammatically a
#  general statement, not a constraint.
#
#  The injected number is intentionally small and not a multiple of
#  the gold so the model can't accidentally fold it in correctly.
# ─────────────────────────────────────────────────────────────────────


_NOOP_TEMPLATES: tuple[str, ...] = (
    "{noun} are typically sold in packs of {n}.",
    "Each {noun_sing} usually weighs about {n} grams.",
    "A nearby competitor charges about {n} cents extra for {noun}.",
    "Last year the price of {noun} was {n} percent higher.",
    "Most stores keep their {noun} on the {n}th shelf from the floor.",
    "The store has been selling {noun} for {n} years.",
    "Standard delivery for {noun} normally takes {n} business days.",
)


def _singularize(noun: str) -> str:
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith("oes"):
        return noun[:-2]
    if noun.endswith("ves"):
        return noun[:-3] + "f"
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def _gen_topical_distractor(rng: random.Random) -> tuple[str, int, str] | None:
    spec = rng.choice(_M1_TEMPLATES)
    difficulty = rng.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
    question, gold = spec.fn(rng, difficulty)
    nouns_alt = "|".join(re.escape(n) for n in sorted(_SCALABLE_NOUNS, key=len, reverse=True))
    matches = re.findall(rf"\b({nouns_alt})\b", question, re.IGNORECASE)
    if not matches:
        return None
    noun = matches[0].lower()
    distractor_n = rng.randint(2, 9)
    if distractor_n == int(gold) or distractor_n * 10 == int(gold):
        distractor_n = (distractor_n % 7) + 11
    template = rng.choice(_NOOP_TEMPLATES)
    distractor = template.format(noun=noun, noun_sing=_singularize(noun), n=distractor_n)
    parts = question.split("\n\nSolve step by step", 1)
    if len(parts) != 2:
        return None
    body, tail = parts
    body = body.rstrip()
    if not body.endswith((".", "!", "?")):
        body = body + "."
    new_q = body + " " + distractor + "\n\nSolve step by step" + tail
    return new_q, int(gold), spec.name + "/noop"


# ─────────────────────────────────────────────────────────────────────
#  Master generator.
# ─────────────────────────────────────────────────────────────────────


_PERTURB_GENERATORS = {
    "numerical_swap": _gen_numerical_swap,
    "digit_expand": _gen_digit_expand,
    "context_pad": _gen_context_pad,
    "unit_swap": _gen_unit_swap,
    "topical_distractor": _gen_topical_distractor,
}


def generate_items(block_seed, n_items: int) -> list[dict]:
    """Generate ``n_items`` math-robustness items.

    Each item carries a ``perturbation`` field naming which family
    it came from. ``src`` is namespaced
    ``v31_math_robustness/<perturbation>/<template>`` so the
    per-source telemetry surfaces per-perturbation pass rates - and
    the *gap* between the underlying M1 axis and these perturbations
    is itself a robustness signal.
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    target = max(1, int(n_items))
    misses = 0
    while len(out) < target and misses < target * 4:
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        perturbation = _sample_perturbation(item_rng)
        gen = _PERTURB_GENERATORS[perturbation]
        result = gen(item_rng)
        if result is None:
            # digit_expand and topical_distractor can return None when
            # the chosen template doesn't satisfy their preconditions
            # (e.g. no scalable noun present); fall back to
            # numerical_swap so we still emit n_items items.
            misses += 1
            result = _gen_numerical_swap(item_rng)
            perturbation = "numerical_swap"
        question, gold, tpl_tag = result
        out.append(
            {
                "src": f"v31_math_robustness/{perturbation}/{tpl_tag}",
                "question": question,
                "gold": str(int(gold)),
                "perturbation": perturbation,
                "template": tpl_tag,
            }
        )
    return out


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=8)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']}")
        print(it["question"])
    print("-" * 60)


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
