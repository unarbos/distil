"""math_competition - v31 competition-math axis.

Procedural items in the AMPS / LiveBench-math / MathArena spirit
(real AIME/Putnam problems are public and contaminated, so we
generate skill-family items instead). Skill families:
algebra (quadratic, 2x2 system), number theory (divisibility,
gcd/lcm), combinatorics (permutations, binomial), geometry
(area/perimeter), probability (dice/coin). Gold computed in Python,
integer or small-denominator rationals (p/q). AIME-style answers in
[0, 999] are respected where natural.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from math import comb, gcd


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _format_question(q: str) -> str:
    return q + (
        "\n\nSolve step by step and end with '#### N' where N is the "
        "final integer (or simplified fraction p/q) answer."
    )


# ─────────────────────────────────────────────────────────────────────
#  Templates
# ─────────────────────────────────────────────────────────────────────


def t_quadratic_sum_roots(rng: random.Random) -> tuple[str, str]:
    r1 = rng.randint(-10, 10)
    r2 = rng.randint(-10, 10)
    while r2 == r1 or r1 == 0 or r2 == 0:
        r1 = rng.randint(-10, 10)
        r2 = rng.randint(-10, 10)
    a = rng.choice([1, 2])
    b = -a * (r1 + r2)
    c = a * r1 * r2
    target = rng.choice(["sum", "product"])
    gold = (r1 + r2) if target == "sum" else (r1 * r2)
    sign_b = "+" if b >= 0 else "-"
    sign_c = "+" if c >= 0 else "-"
    q = (
        f"Find the {target} of the roots of the quadratic equation "
        f"{a}x^2 {sign_b} {abs(b)}x {sign_c} {abs(c)} = 0."
    )
    return _format_question(q), str(gold)


def t_linear_system_2x2(rng: random.Random) -> tuple[str, str]:
    x = rng.randint(-7, 7)
    y = rng.randint(-7, 7)
    while x == 0 or y == 0:
        x = rng.randint(-7, 7)
        y = rng.randint(-7, 7)
    a, b = rng.randint(1, 6), rng.randint(1, 6)
    c, d = rng.randint(1, 6), rng.randint(1, 6)
    while a * d - b * c == 0:
        c, d = rng.randint(1, 6), rng.randint(1, 6)
    e = a * x + b * y
    f = c * x + d * y
    target = rng.choice(["x", "y", "x+y", "xy"])
    if target == "x":
        gold = x
    elif target == "y":
        gold = y
    elif target == "x+y":
        gold = x + y
    else:
        gold = x * y
    target_str = {"x": "x", "y": "y", "x+y": "x + y", "xy": "xy"}[target]
    q = (
        f"Solve the system of equations:\n"
        f"  {a}x + {b}y = {e}\n"
        f"  {c}x + {d}y = {f}\n"
        f"Find {target_str}."
    )
    return _format_question(q), str(gold)


def t_smallest_k_divisible(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(50, 999)
    d = rng.randint(7, 23)
    while d == 1:
        d = rng.randint(7, 23)
    rem = n % d
    k = (d - rem) % d
    if k == 0:
        k = d
    q = (
        f"Find the smallest positive integer k such that {n} + k is "
        f"divisible by {d}."
    )
    return _format_question(q), str(k)


def t_gcd_three(rng: random.Random) -> tuple[str, str]:
    base = rng.randint(2, 9)
    m1 = rng.randint(2, 8)
    m2 = rng.randint(2, 8)
    m3 = rng.randint(2, 8)
    a = base * m1
    b = base * m2
    c = base * m3
    target = rng.choice(["gcd", "lcm"])
    g = gcd(gcd(a, b), c)
    l = _lcm(_lcm(a, b), c)
    gold = g if target == "gcd" else l
    q = f"Find the {target} of the three integers {a}, {b}, and {c}."
    return _format_question(q), str(gold)


def t_arrangements_constraint(rng: random.Random) -> tuple[str, str]:
    n_letters = rng.choice([5, 6, 7])
    constraint = rng.choice(["adjacent", "not_adjacent"])
    from math import factorial as fact
    if constraint == "adjacent":
        gold = fact(n_letters - 1) * 2
        q = (
            f"In how many ways can the letters A, B, C, ..., be "
            f"arranged in a row of {n_letters} positions such that "
            f"the letters A and B are adjacent? "
            f"(Use the first {n_letters} letters of the alphabet.)"
        )
    else:
        gold = fact(n_letters) - fact(n_letters - 1) * 2
        q = (
            f"In how many ways can the letters A, B, C, ..., be "
            f"arranged in a row of {n_letters} positions such that "
            f"the letters A and B are NOT adjacent? "
            f"(Use the first {n_letters} letters of the alphabet.)"
        )
    return _format_question(q), str(gold)


def t_binomial_compute(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(5, 12)
    k = rng.randint(2, n - 1)
    gold = comb(n, k)
    q = f"Compute the value of C({n}, {k}), the binomial coefficient."
    return _format_question(q), str(gold)


def t_composite_area(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    c = rng.randint(2, min(a, b) - 1)
    target = rng.choice(["area", "perimeter"])
    if target == "area":
        gold = a * b - c * c
    else:
        gold = 2 * a + 2 * b
    q = (
        f"A rectangle has dimensions {a} by {b}. A small square of "
        f"side length {c} is cut out from one corner. What is the "
        f"{target} of the resulting L-shaped figure?"
    )
    return _format_question(q), str(gold)


def t_probability_fraction(rng: random.Random) -> tuple[str, str]:
    setup = rng.choice(["dice", "coin", "ball"])
    if setup == "dice":
        n = rng.choice([2, 3])
        target_sum = rng.randint(2, 6 * n)
        favourable = sum(
            1 for combo in _all_dice_outcomes(n) if sum(combo) == target_sum
        )
        total = 6 ** n
        g = gcd(favourable, total)
        if favourable == 0:
            return t_binomial_compute(rng)
        gold_str = f"{favourable // g}/{total // g}"
        q = (
            f"Two fair six-sided dice are rolled. " if n == 2
            else f"Three fair six-sided dice are rolled. "
        ) + (
            f"What is the probability that the sum equals {target_sum}? "
            f"Express your answer as a simplified fraction p/q."
        )
        return _format_question(q), gold_str
    if setup == "coin":
        n = rng.randint(3, 6)
        k = rng.randint(0, n)
        favourable = comb(n, k)
        total = 2 ** n
        g = gcd(favourable, total)
        gold_str = f"{favourable // g}/{total // g}"
        q = (
            f"A fair coin is flipped {n} times. What is the probability "
            f"that exactly {k} flips come up heads? Express your answer "
            f"as a simplified fraction p/q."
        )
        return _format_question(q), gold_str
    n_red = rng.randint(2, 8)
    n_blue = rng.randint(2, 8)
    total_balls = n_red + n_blue
    favourable = n_red
    g = gcd(favourable, total_balls)
    gold_str = f"{favourable // g}/{total_balls // g}"
    q = (
        f"A bag contains {n_red} red balls and {n_blue} blue balls. "
        f"One ball is drawn at random. What is the probability that "
        f"it is red? Express your answer as a simplified fraction p/q."
    )
    return _format_question(q), gold_str


def _all_dice_outcomes(n):
    if n == 1:
        return [(i,) for i in range(1, 7)]
    smaller = _all_dice_outcomes(n - 1)
    out = []
    for s in smaller:
        for i in range(1, 7):
            out.append(s + (i,))
    return out


# ─────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CompTpl:
    name: str
    fn: Callable[[random.Random], tuple[str, str]]
    family: str


TEMPLATES: tuple[_CompTpl, ...] = (
    _CompTpl("quadratic_sum_roots", t_quadratic_sum_roots, "algebra"),
    _CompTpl("linear_system_2x2", t_linear_system_2x2, "algebra"),
    _CompTpl("smallest_k_divisible", t_smallest_k_divisible, "number_theory"),
    _CompTpl("gcd_three", t_gcd_three, "number_theory"),
    _CompTpl("arrangements_constraint", t_arrangements_constraint, "combinatorics"),
    _CompTpl("binomial_compute", t_binomial_compute, "combinatorics"),
    _CompTpl("composite_area", t_composite_area, "geometry"),
    _CompTpl("probability_fraction", t_probability_fraction, "probability"),
)


_BENCH_STREAM_OFFSET = 0x5635  # "V5"


def generate_items(block_seed, n_items: int) -> list[dict]:
    """Generate ``n_items`` competition-math items.

    Items are uniformly sampled across families; the ``src``
    namespace exposes ``family/template`` so per-family pass rate
    is observable.
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        spec = item_rng.choice(TEMPLATES)
        question, gold = spec.fn(item_rng)
        out.append(
            {
                "src": f"v31_math_competition/{spec.family}/{spec.name}",
                "question": question,
                "gold": str(gold),
                "family": spec.family,
                "template": spec.name,
            }
        )
    return out


def grade_response(response: str, gold: str) -> bool:
    """Grader compatible with the existing math pipeline.

    Accepts ``#### <gold>``, integer matches, and ``p/q``
    fractional answers (canonicalized to lowest terms).
    """
    if not response:
        return False
    import re

    text = response.strip()
    target = str(gold).strip()
    # If gold is a fraction p/q, canonicalize both sides.
    if "/" in target:
        try:
            tp, tq = map(int, target.split("/"))
            g = gcd(tp, tq)
            tp //= g; tq //= g
        except (ValueError, ZeroDivisionError):
            return False
        # Look for any p/q in response.
        m = re.findall(r"(-?\d+)\s*/\s*(\d+)", text)
        for p, q in m[::-1]:
            try:
                p, q = int(p), int(q)
                if q == 0:
                    continue
                gg = gcd(p, q)
                if (p // gg, q // gg) == (tp, tq):
                    return True
            except ValueError:
                continue
        return False
    # Otherwise integer.
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
    items = generate_items(block_seed=42, n_items=8)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']}")
        print(it["question"])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
