"""code_humaneval_plus - v31 code-correctness axis.

EvalPlus methodology (Liu et al. NeurIPS 2023; arXiv 2305.01210)
adopted as the *test-augmentation* approach (not the public dataset,
which is contaminated). Per item:

* Pick an algorithm template; rename the function + args + vars per
  round so a memoriser of HumanEval/MBPP can't pattern-match.
* Emit 30-60 procedural tests including aggressive edge cases (empty,
  singleton, max-size, boundary numerics).
* Run the canonical reference implementation in-process to compute
  expected outputs; render the test block as ``assert candidate(args)
  == expected``.

Items emit ``{prompt, test, entry_point, task_id}`` for the existing
``_run_humaneval_sandbox_bench`` grader.
* Chen, M., et al. (2021). "Evaluating Large Language Models
  Trained on Code." (HumanEval paper.) arXiv:2107.03374.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable


_BENCH_STREAM_OFFSET = 0x5639  # "V9"


# ─────────────────────────────────────────────────────────────────────
#  Helpers for procedural function naming.
# ─────────────────────────────────────────────────────────────────────


def _hash_suffix(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


# ─────────────────────────────────────────────────────────────────────
#  Template 1: count_in_list
#  Reference: count occurrences of a target value in a list.
# ─────────────────────────────────────────────────────────────────────


def _t_count_in_list(rng: random.Random) -> dict:
    fname = "count_target_" + _hash_suffix(rng)
    arg1 = rng.choice(["nums", "values", "items", "data"])
    arg2 = "target"

    def _ref(items, target):
        return sum(1 for x in items if x == target)

    test_cases = []
    for _ in range(rng.randint(40, 60)):
        size = rng.randint(0, 12)
        target = rng.randint(-5, 5)
        items = [rng.randint(-5, 5) for _ in range(size)]
        test_cases.append((items, target, _ref(items, target)))
    # Edge cases.
    test_cases.append(([], 0, 0))
    test_cases.append(([1], 1, 1))
    test_cases.append(([1, 1, 1], 1, 3))
    test_cases.append(([1, 2, 3], 4, 0))
    docstring = (
        f"\"\"\"Return the number of occurrences of {arg2} in the "
        f"list {arg1}.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}([1, 2, 1, 3, 1], 1)\n"
        f"    3\n"
        f"    >>> {fname}([], 5)\n"
        f"    0\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}, {arg2}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1, arg2])


# ─────────────────────────────────────────────────────────────────────
#  Template 2: reverse_words
#  Reference: reverse each word in a string while preserving spaces.
# ─────────────────────────────────────────────────────────────────────


def _t_reverse_words(rng: random.Random) -> dict:
    fname = "reverse_each_word_" + _hash_suffix(rng)
    arg1 = rng.choice(["text", "sentence", "s", "phrase"])

    def _ref(text):
        return " ".join(word[::-1] for word in text.split(" "))

    test_cases = []
    word_pool = ["hello", "world", "code", "test", "abc", "x", "ab", "racecar", "z"]
    for _ in range(rng.randint(35, 50)):
        n = rng.randint(0, 5)
        if n == 0:
            text = ""
        else:
            text = " ".join(rng.choice(word_pool) for _ in range(n))
        test_cases.append((text, _ref(text)))
    test_cases.append(("", ""))
    test_cases.append(("a", "a"))
    test_cases.append(("ab cd", "ba dc"))
    docstring = (
        f"\"\"\"Reverse each space-separated word in {arg1} while "
        f"keeping the spaces in place.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}('hello world')\n"
        f"    'olleh dlrow'\n"
        f"    >>> {fname}('a')\n"
        f"    'a'\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1])


# ─────────────────────────────────────────────────────────────────────
#  Template 3: filter_above_threshold
# ─────────────────────────────────────────────────────────────────────


def _t_filter_above(rng: random.Random) -> dict:
    fname = "filter_above_" + _hash_suffix(rng)
    arg1 = rng.choice(["nums", "values", "items"])
    arg2 = rng.choice(["threshold", "limit", "min_val"])

    def _ref(items, threshold):
        return [x for x in items if x > threshold]

    test_cases = []
    for _ in range(rng.randint(40, 60)):
        size = rng.randint(0, 10)
        items = [rng.randint(-10, 10) for _ in range(size)]
        threshold = rng.randint(-10, 10)
        test_cases.append((items, threshold, _ref(items, threshold)))
    test_cases.append(([], 0, []))
    test_cases.append(([1], 0, [1]))
    test_cases.append(([1], 1, []))
    test_cases.append(([3, 3, 3], 3, []))
    docstring = (
        f"\"\"\"Return a list containing only elements of {arg1} that "
        f"are STRICTLY GREATER than {arg2}.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}([1, 5, 3, 8, 2], 3)\n"
        f"    [5, 8]\n"
        f"    >>> {fname}([], 0)\n"
        f"    []\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}, {arg2}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1, arg2])


# ─────────────────────────────────────────────────────────────────────
#  Template 4: dict_value_sum
# ─────────────────────────────────────────────────────────────────────


def _t_dict_value_sum(rng: random.Random) -> dict:
    fname = "sum_dict_values_" + _hash_suffix(rng)
    arg1 = rng.choice(["d", "data", "mapping", "kv"])

    def _ref(d):
        return sum(d.values())

    test_cases = []
    for _ in range(rng.randint(35, 50)):
        size = rng.randint(0, 8)
        d = {f"k{i}": rng.randint(-10, 10) for i in range(size)}
        test_cases.append((d, _ref(d)))
    test_cases.append(({}, 0))
    test_cases.append(({"a": 1}, 1))
    test_cases.append(({"a": -1, "b": 1}, 0))
    docstring = (
        f"\"\"\"Return the sum of all integer values in the "
        f"dictionary {arg1}.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}({{'a': 1, 'b': 2, 'c': 3}})\n"
        f"    6\n"
        f"    >>> {fname}({{}})\n"
        f"    0\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1])


# ─────────────────────────────────────────────────────────────────────
#  Template 5: is_palindrome (string)
# ─────────────────────────────────────────────────────────────────────


def _t_is_palindrome(rng: random.Random) -> dict:
    fname = "is_palindrome_" + _hash_suffix(rng)
    arg1 = rng.choice(["s", "text", "word"])

    def _ref(s):
        return s == s[::-1]

    test_cases = []
    pool = ["abc", "aba", "abba", "racecar", "x", "", "noon", "level", "test", "hello"]
    for _ in range(rng.randint(35, 50)):
        s = rng.choice(pool)
        test_cases.append((s,  _ref(s)))
    # Edge cases.
    test_cases.extend(
        [
            ("", True),
            ("a", True),
            ("ab", False),
            ("aa", True),
            ("racecar", True),
            ("racecars", False),
        ]
    )
    docstring = (
        f"\"\"\"Return True if {arg1} reads the same forwards and "
        f"backwards, False otherwise. Empty string is a palindrome.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}('racecar')\n"
        f"    True\n"
        f"    >>> {fname}('hello')\n"
        f"    False\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1])


# ─────────────────────────────────────────────────────────────────────
#  Template 6: max_consecutive_run
# ─────────────────────────────────────────────────────────────────────


def _t_max_run(rng: random.Random) -> dict:
    fname = "max_consecutive_run_" + _hash_suffix(rng)
    arg1 = rng.choice(["nums", "values", "items"])

    def _ref(items):
        if not items:
            return 0
        best = cur = 1
        for i in range(1, len(items)):
            if items[i] == items[i - 1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    test_cases = []
    for _ in range(rng.randint(35, 50)):
        size = rng.randint(0, 12)
        items = [rng.randint(0, 4) for _ in range(size)]
        test_cases.append((items, _ref(items)))
    test_cases.extend(
        [
            ([], 0),
            ([1], 1),
            ([1, 1, 1], 3),
            ([1, 2, 1], 1),
        ]
    )
    docstring = (
        f"\"\"\"Return the length of the longest run of consecutive "
        f"equal values in {arg1}. Returns 0 for an empty list.\n\n"
        f"    Examples:\n"
        f"    >>> {fname}([1, 1, 2, 2, 2, 1])\n"
        f"    3\n"
        f"    >>> {fname}([])\n"
        f"    0\n"
        f"    \"\"\""
    )
    prompt = f"def {fname}({arg1}):\n    {docstring}\n"
    return _build_item(fname, prompt, test_cases, [arg1])


# ─────────────────────────────────────────────────────────────────────
#  Item builder.
#
#  Renders the {prompt, test, entry_point, task_id} item shape that
#  ``_run_humaneval_sandbox_bench`` expects. The ``test`` function
#  is a single ``def check(candidate):`` containing one ``assert
#  candidate(args) == expected`` per test case.
# ─────────────────────────────────────────────────────────────────────


def _build_item(fname: str, prompt: str, test_cases: list, arg_names: list[str]) -> dict:
    n_args = len(arg_names)

    def _arg_repr(args):
        if n_args == 1:
            return repr(args)
        return ", ".join(repr(a) for a in args)

    asserts = []
    for tc in test_cases:
        if n_args == 1:
            args, expected = tc[0], tc[1]
            asserts.append(
                f"    assert candidate({_arg_repr(args)}) == {expected!r}"
            )
        else:
            *args, expected = tc
            asserts.append(
                f"    assert candidate({_arg_repr(tuple(args))}) == {expected!r}"
            )
    test_block = "def check(candidate):\n" + "\n".join(asserts) + "\n"
    return {
        "task_id": "v31_codeplus/" + fname,
        "entry_point": fname,
        "prompt": prompt,
        "test": test_block,
        "n_test_cases": len(test_cases),
    }


# ─────────────────────────────────────────────────────────────────────
#  Master generator.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CodeTpl:
    name: str
    fn: Callable[[random.Random], dict]


TEMPLATES: tuple[_CodeTpl, ...] = (
    _CodeTpl("count_in_list", _t_count_in_list),
    _CodeTpl("reverse_words", _t_reverse_words),
    _CodeTpl("filter_above", _t_filter_above),
    _CodeTpl("dict_value_sum", _t_dict_value_sum),
    _CodeTpl("is_palindrome", _t_is_palindrome),
    _CodeTpl("max_consecutive_run", _t_max_run),
)


def generate_items(block_seed, n_items: int) -> list[dict]:
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        spec = item_rng.choice(TEMPLATES)
        item = spec.fn(item_rng)
        item["src"] = f"v31_code_humaneval_plus/{spec.name}"
        item["template"] = spec.name
        out.append(item)
    return out


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=2)
    for it in items:
        print("-" * 60)
        print(f"task_id={it['task_id']} entry_point={it['entry_point']}")
        print(f"n_test_cases={it['n_test_cases']}")
        print("PROMPT:")
        print(it["prompt"])
        print("TEST (head):")
        print(it["test"][:300])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
