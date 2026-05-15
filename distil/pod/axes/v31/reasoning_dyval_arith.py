"""reasoning_dyval_arith - v31 DyVal arithmetic-DAG axis.

DyVal methodology (Zhu et al. ICLR 2024 spotlight; arXiv 2309.17167):
random DAGs of integer ops with leaf values; interior nodes apply an
op to their children. Depth / width control difficulty. Gold is
computed programmatically over the DAG; problems are rendered in both
math-expression and natural-language form. Memorisation-impossible:
DAG space grows ~10^12 at d=4/width=2 and >10^20 at d=6.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────
#  Operation registry. Each op declares: name, arity, applier,
#  formatter (for math + nl). We restrict to integer-valued safe ops
#  to keep ground-truth grading exact.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _Op:
    name: str
    arity: int
    apply: Callable[..., int]
    math_fmt: Callable[..., str]
    nl_fmt: Callable[..., str]


def _safe_div(a, b):
    if b == 0:
        return 0
    return a // b


_OPS: tuple[_Op, ...] = (
    _Op(
        "add", 2,
        lambda a, b: a + b,
        lambda a, b: f"({a} + {b})",
        lambda a, b: f"the sum of {a} and {b}",
    ),
    _Op(
        "sub", 2,
        lambda a, b: a - b,
        lambda a, b: f"({a} - {b})",
        lambda a, b: f"{a} minus {b}",
    ),
    _Op(
        "mul", 2,
        lambda a, b: a * b,
        lambda a, b: f"({a} * {b})",
        lambda a, b: f"the product of {a} and {b}",
    ),
    _Op(
        "min", 2,
        lambda a, b: min(a, b),
        lambda a, b: f"min({a}, {b})",
        lambda a, b: f"the smaller of {a} and {b}",
    ),
    _Op(
        "max", 2,
        lambda a, b: max(a, b),
        lambda a, b: f"max({a}, {b})",
        lambda a, b: f"the larger of {a} and {b}",
    ),
)


# ─────────────────────────────────────────────────────────────────────
#  DAG node + builder.
#
#  Each node is either a Leaf (carries an int) or an Op (carries an
#  Op + list of child nodes). Eval is naive recursion. Render goes
#  through one of two formatters; both yield a human-readable
#  expression with intermediate variable names so the model can
#  benefit from chain-of-thought, mirroring DyVal's NL formulation.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _Node:
    op: _Op | None  # None == leaf
    value: int | None = None
    children: list["_Node"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.op is None


def _build_dag(rng: random.Random, depth: int, max_width: int = 2,
               leaf_range: tuple[int, int] = (1, 9)) -> _Node:
    """Build a DAG of given depth. width=2 (binary) is the default
    DyVal config; we also allow width=3 trees for extra difficulty.
    """
    if depth == 0:
        return _Node(op=None, value=rng.randint(*leaf_range))
    op = rng.choice(_OPS)
    children = [
        _build_dag(rng, depth - 1, max_width, leaf_range)
        for _ in range(op.arity)
    ]
    return _Node(op=op, children=children)


def _evaluate(node: _Node) -> int:
    if node.is_leaf():
        return node.value
    args = [_evaluate(c) for c in node.children]
    return node.op.apply(*args)


def _render_math(node: _Node) -> str:
    if node.is_leaf():
        return str(node.value)
    return node.op.math_fmt(*[_render_math(c) for c in node.children])


def _render_nl_with_vars(node: _Node, var_lines: list[str], counter: list[int]) -> str:
    """Render as a sequence of variable assignments.

    Returns the final variable name. This is closer to how a CoT
    reasoner thinks about the DAG step-by-step: each interior node
    gets its own line ``vK = expr_in_terms_of_smaller_vars``.
    """
    if node.is_leaf():
        return str(node.value)
    child_refs = [
        _render_nl_with_vars(c, var_lines, counter) for c in node.children
    ]
    counter[0] += 1
    var = f"v{counter[0]}"
    var_lines.append(f"{var} = {node.op.math_fmt(*child_refs)}")
    return var


# ─────────────────────────────────────────────────────────────────────
#  Difficulty schedule.
# ─────────────────────────────────────────────────────────────────────


_BENCH_STREAM_OFFSET = 0xDA9A  # "Dy" style

_DIFFICULTY_DEPTHS: tuple[int, ...] = (2, 3, 4, 5, 6)
_DIFFICULTY_RATIO: tuple[float, ...] = (0.20, 0.25, 0.25, 0.20, 0.10)


def _sample_depth(rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for d, w in zip(_DIFFICULTY_DEPTHS, _DIFFICULTY_RATIO):
        cum += w
        if r < cum:
            return d
    return _DIFFICULTY_DEPTHS[-1]


# ─────────────────────────────────────────────────────────────────────
#  Surface formatter.
# ─────────────────────────────────────────────────────────────────────


def _format_question(node: _Node, mode: str) -> str:
    if mode == "math":
        expr = _render_math(node)
        return (
            "Compute the integer value of the following expression "
            "(use integer division for any division). Show your work "
            "if helpful, then give the final answer in the format "
            "'Final answer: <integer>'.\n\n"
            f"{expr} = ?"
        )
    var_lines: list[str] = []
    counter = [0]
    final_var = _render_nl_with_vars(node, var_lines, counter)
    body = "\n".join(var_lines)
    return (
        "We have the following arithmetic computation. The variables "
        "v1, v2, ... are intermediate results. Compute the final value of "
        f"{final_var}.\n\n{body}\n\nGive the final answer in the format "
        "'Final answer: <integer>'."
    )


# ─────────────────────────────────────────────────────────────────────
#  Generator entrypoint.
# ─────────────────────────────────────────────────────────────────────


def generate_items(block_seed, n_items: int) -> list[dict]:
    """Generate ``n_items`` DyVal-style arithmetic DAG items.

    Returns ``{src, question, gold, depth, mode}`` dicts. Gold is an
    integer cast to string (consistent with rest of math pipeline).
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        depth = _sample_depth(rng)
        node = _build_dag(rng, depth, max_width=2, leaf_range=(1, 9))
        gold = _evaluate(node)
        # 50/50 split between math-expression and var-decomposed NL forms.
        mode = "math" if rng.random() < 0.5 else "nl_vars"
        question = _format_question(node, mode)
        out.append(
            {
                "src": f"v31_dyval_arith/d{depth}/{mode}",
                "question": question,
                "gold": str(gold),
                "depth": depth,
                "mode": mode,
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
#  Grader. Re-uses the math answer extractor to stay consistent with
#  the rest of the math pipeline.
# ─────────────────────────────────────────────────────────────────────


def grade_response(response: str, gold: str) -> bool:
    """Forgiving integer grader. Accepts ``Final answer: <n>``,
    ``\\boxed{<n>}``, or any standalone integer at the end of the
    response that equals ``gold``.
    """
    if not response:
        return False
    import re

    text = response.strip()
    target = str(gold).strip()
    # 'Final answer: <int>'.
    m = re.search(r"final\s+answer\s*[:\-]?\s*(-?\d+)", text, re.IGNORECASE)
    if m and m.group(1) == target:
        return True
    # boxed notation.
    m = re.search(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    if m and m.group(1) == target:
        return True
    # Last integer in the response.
    nums = re.findall(r"-?\d+", text)
    if nums and nums[-1] == target:
        return True
    return False


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=4)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']}")
        print(it["question"])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
