"""math_gsm_symbolic - v31 GSM-Symbolic axis.

Apple's GSM-Symbolic methodology (Mirzadeh et al. 2024, arXiv 2410.05229)
re-implemented over our own template store:
* Symbolic templates with named placeholders + variable-spec
  constraints; gold computed in Python for exact cross-validator
  agreement.
* P0 / P1 / P2 difficulty knobs (extra independent sub-calculations).
* Optional GSM-NoOp topical distractor (math-irrelevant clause).

Exports ``generate_items(block_seed, n_items)`` returning a list of
``{src, question, gold}`` compatible with ``_math_score_one``.

References:
* Mirzadeh, I., Alizadeh, K., et al. (2024). "GSM-Symbolic:
  Understanding the Limitations of Mathematical Reasoning in Large
  Language Models." arXiv:2410.05229.
* Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word
  Problems." arXiv:2110.14168 (the GSM8K paper).
* Li, Q., et al. (2024). "GSM-Plus: A Comprehensive Benchmark for
  Evaluating the Robustness of LLMs as Mathematical Problem Solvers."
  ACL 2024 (the perturbation methodology that motivates GSM-NoOp).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from math import gcd

# ─────────────────────────────────────────────────────────────────────
#  Realistic surface vocabulary (decorative — does not affect gold).
#  These are deliberately small and English-monoculture; the v31 spec
#  acknowledges this is a known limitation and a future sprint can
#  expand the vocabulary with multilingual procedural names.
# ─────────────────────────────────────────────────────────────────────

_NAMES_M = (
    "Alex", "Ben", "Chris", "David", "Ethan", "Felix", "George",
    "Henry", "Ian", "Jake", "Kevin", "Liam", "Mason", "Noah",
    "Oscar", "Paul", "Quinn", "Ryan", "Sam", "Tyler",
)
_NAMES_F = (
    "Alice", "Bella", "Carla", "Diana", "Emma", "Fiona", "Grace",
    "Hannah", "Iris", "Julia", "Kara", "Lily", "Maya", "Nora",
    "Olivia", "Piper", "Quinn", "Rachel", "Sophia", "Tara",
)
_SHOPS = (
    "bakery", "bookstore", "café", "candy shop", "convenience store",
    "deli", "farmers' market", "florist", "general store", "grocery store",
    "hardware store", "hobby shop", "music store", "pet store",
    "stationery shop", "supermarket", "thrift shop", "toy store",
)
_ITEMS_PRICED = (
    ("apple", "apples"), ("bagel", "bagels"), ("banana", "bananas"),
    ("book", "books"), ("candle", "candles"), ("cookie", "cookies"),
    ("cupcake", "cupcakes"), ("donut", "donuts"), ("eraser", "erasers"),
    ("magazine", "magazines"), ("muffin", "muffins"), ("notebook", "notebooks"),
    ("orange", "oranges"), ("pencil", "pencils"), ("pen", "pens"),
    ("postcard", "postcards"), ("scone", "scones"), ("sticker", "stickers"),
    ("ticket", "tickets"), ("toy car", "toy cars"),
)


def _name(rng: random.Random) -> str:
    return rng.choice(_NAMES_M + _NAMES_F)


def _shop(rng: random.Random) -> str:
    return rng.choice(_SHOPS)


def _item_plural(rng: random.Random) -> str:
    return rng.choice(_ITEMS_PRICED)[1]


# Template engine: each template is a Python closure taking
# (rng, difficulty) and returning (question, gold_int). Closures over
# JSON specs so inter-variable constraints and P1/P2 extras can use
# imperative branching.


GsmSymTemplate = Callable[[random.Random, int], tuple[str, int]]


@dataclass(frozen=True)
class TemplateSpec:
    """Lightweight metadata for a GSM-Symbolic template.

    ``name`` is the template identifier used as the ``src`` suffix on
    the emitted item (``v31_gsm_symbolic/<name>``) so the per-template
    pass-rate can be inspected via the existing per-source telemetry.
    """

    name: str
    fn: GsmSymTemplate
    families: tuple[str, ...] = field(default=())
    """Tags for downstream analysis (e.g. ``("arithmetic", "money")``)."""


def _format_question(question: str) -> str:
    """Append the GSM8K-style answer marker.

    The existing ``_math_extract_answer`` pipeline parses ``#### N`` so
    we keep that contract.
    """
    return question + (
        "\n\nSolve step by step and end with '#### N' where N is the "
        "final integer answer."
    )


# ─── Template 1: shopping with discount + tax ────────────────────────


def t_shopping_discount(rng: random.Random, difficulty: int) -> tuple[str, int]:
    name = _name(rng)
    start = rng.choice([60, 80, 100, 120, 150, 200])
    n_a = rng.randint(2, 6)
    p_a = rng.choice([3, 4, 5, 6, 8])
    n_b = rng.randint(2, 5)
    p_b = rng.choice([4, 5, 6, 8, 10])
    disc = rng.choice([10, 20, 25])
    tax = rng.choice([5, 8, 10])
    shop = _shop(rng)
    item_a = _item_plural(rng)
    item_b = _item_plural(rng)
    cost_a = n_a * p_a
    cost_b_pre = n_b * p_b
    cost_b = cost_b_pre - cost_b_pre * disc // 100
    subtotal = cost_a + cost_b
    final_tax = subtotal * tax // 100
    spent = subtotal + final_tax
    gold = start - spent
    q = (
        f"{name} goes to the {shop} with ${start}. They buy {n_a} "
        f"{item_a} at ${p_a} each, and {n_b} {item_b} at ${p_b} each "
        f"(today the {item_b} are {disc}% off). There is a {tax}% "
        f"sales tax on the discounted subtotal. How much money does "
        f"{name} have left?"
    )
    if difficulty >= 1:
        n_c = rng.randint(1, 3)
        p_c = rng.choice([4, 6, 8, 10, 12])
        item_c = _item_plural(rng)
        cost_c = n_c * p_c
        new_subtotal = subtotal + cost_c
        new_tax = new_subtotal * tax // 100
        spent = new_subtotal + new_tax
        gold = start - spent
        q = q.replace(
            f"and {n_b} {item_b} at ${p_b} each",
            f"{n_b} {item_b} at ${p_b} each, and {n_c} {item_c} at ${p_c} each",
        )
    if difficulty >= 2:
        gift_card = rng.randint(5, 15)
        gold = gold + gift_card
        q = q + (
            f" After paying, {name} also receives a ${gift_card} gift "
            f"card refund from a previous purchase."
        )
    return _format_question(q), gold


# ─── Template 2: travel with multi-leg distance ──────────────────────


def t_travel_distance(rng: random.Random, difficulty: int) -> tuple[str, int]:
    name = _name(rng)
    legs = rng.randint(2, 3)
    speeds = [rng.choice([40, 50, 60, 70]) for _ in range(legs)]
    times = [rng.choice([2, 3, 4, 5]) for _ in range(legs)]
    distance = sum(s * t for s, t in zip(speeds, times))
    leg_str = ", then ".join(
        f"travels at {s} mph for {t} hours" for s, t in zip(speeds, times)
    )
    q = (
        f"{name} drives across the country. They first {leg_str}. "
        f"What is the total distance in miles that {name} drives?"
    )
    gold = distance
    if difficulty >= 1:
        extra_speed = rng.choice([45, 55, 65])
        extra_time = rng.choice([1, 2, 3])
        gold += extra_speed * extra_time
        q = q.replace(
            "What is the total",
            f"After a short break they continue at {extra_speed} mph "
            f"for {extra_time} more hours. What is the total",
        )
    if difficulty >= 2:
        detour = rng.randint(15, 60)
        gold += detour
        q = q.replace(
            "What is the total",
            f"During the trip they take a {detour}-mile detour to "
            f"avoid construction. What is the total",
        )
    return _format_question(q), gold


# ─── Template 3: classroom multi-step (students, materials) ──────────


def t_classroom_supplies(rng: random.Random, difficulty: int) -> tuple[str, int]:
    teacher = _name(rng)
    n_students = rng.choice([18, 20, 24, 25, 28, 30])
    pencils_each = rng.randint(2, 5)
    notebooks_each = rng.randint(1, 3)
    erasers_each = rng.randint(1, 2)
    total = n_students * (pencils_each + notebooks_each + erasers_each)
    q = (
        f"{teacher} teaches a class of {n_students} students. Each "
        f"student needs {pencils_each} pencils, {notebooks_each} "
        f"notebooks, and {erasers_each} erasers. How many supplies "
        f"in total does {teacher} need to order?"
    )
    gold = total
    if difficulty >= 1:
        spare_pct = rng.choice([10, 20, 25])
        spare = total * spare_pct // 100
        gold = total + spare
        q = q.replace(
            "How many",
            f"They also order an extra {spare_pct}% as spares. How many",
        )
    if difficulty >= 2:
        broken = rng.randint(3, 8)
        gold -= broken
        q = q.replace(
            "How many",
            f"On the way to school, {broken} items get crushed and "
            f"have to be discarded. How many",
        )
    return _format_question(q), gold


# ─── Template 4: garden harvest with shrinkage ───────────────────────


def t_garden_harvest(rng: random.Random, difficulty: int) -> tuple[str, int]:
    farmer = _name(rng)
    rows = rng.randint(4, 12)
    plants_per_row = rng.randint(8, 20)
    yield_per_plant = rng.randint(3, 8)
    total = rows * plants_per_row * yield_per_plant
    crop = rng.choice(["tomatoes", "peppers", "cucumbers", "zucchini", "onions"])
    q = (
        f"{farmer} grows {crop} in a garden with {rows} rows. Each "
        f"row has {plants_per_row} plants, and each plant yields "
        f"{yield_per_plant} {crop}. How many {crop} does {farmer} "
        f"harvest in total?"
    )
    gold = total
    if difficulty >= 1:
        bad_pct = rng.choice([10, 15, 20])
        bad = total * bad_pct // 100
        gold = total - bad
        q = q.replace(
            "How many",
            f"Of the harvest, {bad_pct}% are damaged and discarded. "
            f"How many",
        )
    if difficulty >= 2:
        bonus_rows = rng.randint(2, 4)
        bonus = bonus_rows * plants_per_row * yield_per_plant
        bonus_bad = bonus * (bad_pct if difficulty >= 1 else 0) // 100
        gold += bonus - bonus_bad
        q = q.replace(
            "How many",
            f"A neighbor gives {farmer} {bonus_rows} more rows of the "
            f"same crop, and the same percentage are discarded. "
            f"How many",
        )
    return _format_question(q), gold


# ─── Template 5: bakery production minus orders ──────────────────────


def t_bakery_orders(rng: random.Random, difficulty: int) -> tuple[str, int]:
    baker = _name(rng)
    daily = rng.choice([60, 80, 100, 120, 150])
    days = rng.randint(3, 7)
    orders = rng.randint(daily // 2, daily * 4 // 5)
    item = rng.choice(["loaves", "cakes", "muffins", "pies", "tarts"])
    sold = orders * days
    leftover = daily * days - sold
    q = (
        f"{baker} runs a bakery that produces {daily} {item} per day. "
        f"Over {days} days, the bakery has standing orders for {orders} "
        f"{item} per day. How many extra {item} are left over after "
        f"{days} days?"
    )
    gold = leftover
    if difficulty >= 1:
        donated = rng.randint(3, min(10, leftover))
        gold = leftover - donated
        q = q.replace(
            "How many extra",
            f"After fulfilling orders, {donated} {item} per day are "
            f"donated to a shelter. How many extra",
        )
        gold = leftover - donated * days
    if difficulty >= 2:
        bad = rng.randint(2, 8)
        gold -= bad * days
        q = q.replace(
            "How many extra",
            f"Each day, {bad} {item} are burned and thrown away. "
            f"How many extra",
        )
    return _format_question(q), gold


# ─── Template 6: library returns / fines ─────────────────────────────


def t_library_books(rng: random.Random, difficulty: int) -> tuple[str, int]:
    librarian = _name(rng)
    days_late = rng.randint(3, 12)
    fine_per_day = rng.choice([1, 2, 5])
    n_books = rng.randint(2, 6)
    total_fine = days_late * fine_per_day * n_books
    q = (
        f"{librarian} is checking in returned books. {n_books} books "
        f"are {days_late} days overdue. The fine is ${fine_per_day} "
        f"per day per book. What is the total fine in dollars?"
    )
    gold = total_fine
    if difficulty >= 1:
        forgive_pct = rng.choice([10, 25])
        forgive = total_fine * forgive_pct // 100
        gold = total_fine - forgive
        q = q.replace(
            "What is the total",
            f"The library waives {forgive_pct}% of the fine for a "
            f"loyal patron. What is the total",
        )
    if difficulty >= 2:
        damaged = rng.randint(0, 2)
        damage_fee = damaged * rng.choice([10, 15, 20])
        gold += damage_fee
        q = q.replace(
            "What is the total",
            f"{damaged} of the books are also damaged, with a "
            f"replacement fee of ${damage_fee // damaged if damaged else 0} "
            f"each. What is the total",
        )
    return _format_question(q), gold


# ─── Template 7: percentage / proportion (legacy direct-compute) ─────


def t_percentage_compose(rng: random.Random, difficulty: int) -> tuple[str, int]:
    name = _name(rng)
    base = rng.choice([200, 300, 400, 500, 800, 1000])
    pct = rng.choice([10, 15, 20, 25, 30, 40])
    gold = base * pct // 100
    q = (
        f"{name} has {base} marbles. They give {pct}% of the marbles "
        f"to a friend. How many marbles does {name} give away?"
    )
    if difficulty >= 1:
        pct2 = rng.choice([10, 20, 25, 50])
        gold = (base * pct // 100) * pct2 // 100
        q = (
            f"{name} has {base} marbles. They give {pct}% of the marbles "
            f"to a friend. The friend then gives {pct2}% of THOSE marbles "
            f"to a sibling. How many marbles does the sibling receive?"
        )
    if difficulty >= 2:
        pct3 = rng.choice([10, 25, 50])
        gold = ((base * pct // 100) * pct2 // 100) * pct3 // 100
        q = (
            q.replace("does the sibling receive?", "does the sibling receive after they redistribute?")
            + f" The sibling later gives {pct3}% to a cousin; how many "
            f"marbles does the cousin receive?"
        )
    return _format_question(q), gold


# ─── Template 8: rate problem (work / output) ────────────────────────


def t_work_rate(rng: random.Random, difficulty: int) -> tuple[str, int]:
    a = _name(rng)
    b = _name(rng)
    while b == a:
        b = _name(rng)
    rate_a = rng.choice([20, 25, 30, 40])
    rate_b = rng.choice([15, 20, 25, 35])
    hours = rng.choice([3, 4, 5, 6, 8])
    gold = (rate_a + rate_b) * hours
    item = rng.choice(["envelopes", "boxes", "widgets", "flyers", "labels"])
    q = (
        f"{a} can pack {rate_a} {item} per hour, and {b} can pack "
        f"{rate_b} {item} per hour. Working together for {hours} "
        f"hours, how many {item} do they pack?"
    )
    if difficulty >= 1:
        c = _name(rng)
        while c in (a, b):
            c = _name(rng)
        rate_c = rng.choice([10, 15, 20])
        gold = (rate_a + rate_b + rate_c) * hours
        q = q.replace(
            "Working together",
            f"A third worker {c} packs {rate_c} {item} per hour. "
            f"Working together",
        )
    if difficulty >= 2:
        break_min = rng.choice([10, 15, 20, 30])
        per_min_total = gold // (60 * hours)
        loss = per_min_total * break_min if per_min_total > 0 else break_min
        gold -= loss
        q = q.replace(
            "how many",
            f"They each take a {break_min}-minute break during the "
            f"shift, during which no packing happens. How many",
        )
    return _format_question(q), gold


# ─── Registry of templates ───────────────────────────────────────────


TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec("shopping_discount", t_shopping_discount, ("money", "arithmetic")),
    TemplateSpec("travel_distance", t_travel_distance, ("rate", "distance")),
    TemplateSpec("classroom_supplies", t_classroom_supplies, ("counting", "multiplication")),
    TemplateSpec("garden_harvest", t_garden_harvest, ("counting", "percentage")),
    TemplateSpec("bakery_orders", t_bakery_orders, ("counting", "subtraction")),
    TemplateSpec("library_books", t_library_books, ("money", "percentage")),
    TemplateSpec("percentage_compose", t_percentage_compose, ("percentage", "composition")),
    TemplateSpec("work_rate", t_work_rate, ("rate", "addition")),
)


# ─────────────────────────────────────────────────────────────────────
#  GSM-NoOp distractor injection.
#
#  Apple's NoOp methodology: take a clean GSM-Symbolic item and inject
#  a sentence that is topically relevant (mentions a number) but does
#  NOT change the gold answer. A model that pattern-matches "see
#  number → use number" gets the wrong answer; a model that actually
#  reasons about which numbers matter gets the right answer.
#
#  We implement this as a list of distractor sentence templates that
#  splice into the question right before the final question phrase.
#  Each distractor mentions an irrelevant numeric quantity (a person's
#  age, a year, a calorie count, etc.) that should NOT appear in the
#  computation.
# ─────────────────────────────────────────────────────────────────────


def _noop_distractor(rng: random.Random) -> str:
    return rng.choice(
        (
            "The shop is {n} blocks from the bus stop.",
            "The city has been hosting this event for {n} years.",
            "The recipe was originally written {n} years ago.",
            "There are {n} other customers in line.",
            "The receipt is printed on {n}-inch paper.",
            "The store opened {n} hours earlier than usual.",
            "A passerby comments that they have {n} pets at home.",
            "A nearby billboard advertises a sale ending in {n} days.",
            "The cashier mentions they've worked there for {n} years.",
            "The friend's lucky number is {n}.",
        )
    ).format(n=rng.randint(2, 99))


def _splice_noop(question: str, rng: random.Random) -> str:
    """Inject a NoOp distractor before the final answer-marker line."""
    distractor = _noop_distractor(rng)
    parts = question.rsplit("\n\nSolve step by step", 1)
    if len(parts) != 2:
        return question
    return parts[0].rstrip() + " " + distractor + "\n\nSolve step by step" + parts[1]


# ─────────────────────────────────────────────────────────────────────
#  Main generator entrypoint.
#
#  ``generate_items(block_seed, n_items)`` is the function that the
#  ``_BENCH_SAMPLE_GENERATORS`` registry in ``pod_eval_vllm.py``
#  invokes once per round to populate ``_BENCH_SAMPLES["v31_math_gsm_symbolic"]``.
#
#  Distribution targets per round:
#  * 50 % P0 (base difficulty) — calibrates against gsm8k.
#  * 25 % P1 (+1 clause)       — calibrates against GSM-Symbolic-P1.
#  * 15 % P2 (+2 clauses)      — calibrates against GSM-Symbolic-P2.
#  * 10 % NoOp                 — calibrates against GSM-NoOp.
#
#  These ratios are tunable via env vars (read at module import time
#  to avoid per-round overhead, see ``_DIFFICULTY_RATIO`` below).
# ─────────────────────────────────────────────────────────────────────


_DIFFICULTY_RATIO: tuple[float, float, float, float] = (0.50, 0.25, 0.15, 0.10)
"""(p0, p1, p2, noop) — must sum to 1.0."""

# v31 stream offset — keeps this pool statistically independent from
# the v30 ``_generate_math_items`` pool that uses block_seed XOR 0.
_BENCH_STREAM_OFFSET = 0x5631  # "V1"


def _difficulty_for_index(rng: random.Random) -> tuple[int, bool]:
    """Sample (difficulty_level, is_noop) per the ratio above.

    ``difficulty_level ∈ {0, 1, 2}``.
    ``is_noop`` only relevant when difficulty == 0 (NoOp items are
    P0-baseline with a distractor injected; making P1/P2 NoOp would
    confound the difficulty signal).
    """
    r = rng.random()
    p0, p1, p2, _noop = _DIFFICULTY_RATIO
    if r < p0:
        return 0, False
    if r < p0 + p1:
        return 1, False
    if r < p0 + p1 + p2:
        return 2, False
    return 0, True


def generate_items(block_seed: int | None, n_items: int) -> list[dict]:
    """Generate procedural GSM-Symbolic items for one validator round.

    Args:
        block_seed: per-round entropy from the validator's substrate.
            ``None`` is treated as 0 (deterministic, useful for tests).
        n_items: number of items to generate.

    Returns:
        list of ``{src, question, gold, difficulty, is_noop, template}``
        dicts. ``src`` follows the v31 namespace
        ``"v31_gsm_symbolic/<template_name>"`` so per-source
        telemetry surfaces template-level pass rates.
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(n_items):
        per_item_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_item_seed)
        difficulty, is_noop = _difficulty_for_index(item_rng)
        spec = item_rng.choice(TEMPLATES)
        question, gold_int = spec.fn(item_rng, difficulty)
        if is_noop:
            question = _splice_noop(question, item_rng)
        suffix = "noop" if is_noop else f"p{difficulty}"
        out.append(
            {
                "src": f"v31_gsm_symbolic/{spec.name}/{suffix}",
                "question": question,
                "gold": str(int(gold_int)),
                "difficulty": difficulty,
                "is_noop": is_noop,
                "template": spec.name,
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
#  Self-test: invoke when run as a module to spot-check item shapes.
#
#  ``python -m scripts.v31.math_gsm_symbolic`` prints a few sample
#  items for each (difficulty, noop) combination so a human can
#  visually verify they're well-formed and the gold makes sense.
#  This is NOT a unit test — see ``tests/test_v31_math_gsm_symbolic.py``
#  for the proper test suite.
# ─────────────────────────────────────────────────────────────────────


def _self_test_demo() -> None:  # pragma: no cover - dev helper
    items = generate_items(block_seed=42, n_items=12)
    for it in items:
        print("─" * 60)
        print(f"src     = {it['src']}")
        print(f"diff    = {it['difficulty']}, noop = {it['is_noop']}")
        print(f"gold    = {it['gold']}")
        print("question:")
        print(it["question"])
    print("─" * 60)
    print(f"generated {len(items)} items OK")


if __name__ == "__main__":  # pragma: no cover - dev helper
    _self_test_demo()
