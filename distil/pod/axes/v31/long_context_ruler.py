"""long_context_ruler - v31 long-context axis (RULER subset).

Four RULER tasks (Hsieh et al. NVIDIA, ICLR 2024; arXiv 2404.06654)
chosen because they fit the 4-8K context budget our distilled models
operate in:

* ``niah_single``     - 1 (key, value) pair hidden in distractors.
* ``niah_multikey``   - N pairs; ask for the value of one key.
* ``multihop_var``    - variable-assignment chain; final value.
* ``aggregation_count`` - count occurrences of a keyword.

Goodhart-resistant: haystacks/keys/values/chains are freshly sampled
per round from a bland distractor pool, so memorisation is impossible
and pattern-matching can't cheat.

References:
* Hsieh, C.-P., et al. (2024). "RULER: What's the Real Context
  Size of Your Long-Context Language Models?" arXiv:2404.06654.
  ICLR 2024 spotlight.
"""

from __future__ import annotations

import random
import re
import string


_BENCH_STREAM_OFFSET = 0x5637  # "V7"


# ─────────────────────────────────────────────────────────────────────
#  Distractor sentence pool (procedural).
#
#  We deliberately keep these factually bland and topic-stable so
#  there's no signal a model can pattern-match besides the inserted
#  needles. The variety is achieved by combining random fragments.
# ─────────────────────────────────────────────────────────────────────


_PEOPLE_NAMES = (
    "Alex", "Sam", "Casey", "Jordan", "Morgan", "Taylor", "Riley",
    "Drew", "Quinn", "Avery", "Jamie", "Reese", "Skyler", "Parker",
)
_VERBS_NEUTRAL = (
    "walked", "looked", "noticed", "remembered", "considered",
    "watched", "thought about", "ignored", "described",
)
_NOUNS_NEUTRAL = (
    "the table", "the window", "the sky", "the corner",
    "a passing car", "the wallpaper", "an old book",
    "a cup of tea", "the calendar", "a small lamp",
)
_CONNECTORS = (
    "Meanwhile,", "Later,", "After a while,", "On a related note,",
    "In another part of the story,", "Earlier that day,",
    "Around the same time,", "Suddenly,", "Quietly,", "Then,",
)


def _random_sentence(rng: random.Random) -> str:
    """Generate one fully-procedural distractor sentence.

    Six tokens minimum so 200 sentences = ~1200 tokens; tunable.
    """
    use_connector = rng.random() < 0.5
    parts = []
    if use_connector:
        parts.append(rng.choice(_CONNECTORS))
    parts.append(rng.choice(_PEOPLE_NAMES))
    parts.append(rng.choice(_VERBS_NEUTRAL))
    parts.append(rng.choice(_NOUNS_NEUTRAL))
    parts.append("for a few moments.")
    return " ".join(parts)


def _make_haystack(rng: random.Random, n_sentences: int) -> list[str]:
    return [_random_sentence(rng) for _ in range(n_sentences)]


# ─────────────────────────────────────────────────────────────────────
#  Helpers for needles.
# ─────────────────────────────────────────────────────────────────────


def _random_token(rng: random.Random, length: int = 6) -> str:
    """Random alphanumeric token used as needle key or value."""
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def _insert_needle(haystack: list[str], position: int, sentence: str) -> list[str]:
    return haystack[:position] + [sentence] + haystack[position:]


# ─────────────────────────────────────────────────────────────────────
#  Task: niah_single
# ─────────────────────────────────────────────────────────────────────


def _gen_niah_single(rng: random.Random, n_distractors: int) -> tuple[str, str, dict]:
    key = _random_token(rng, 7)
    value = _random_token(rng, 6)
    needle = f"The secret code for {key} is {value}."
    haystack = _make_haystack(rng, n_distractors)
    pos = rng.randint(0, len(haystack))
    haystack = _insert_needle(haystack, pos, needle)
    body = " ".join(haystack)
    question = (
        f"Read the following passage carefully.\n\n"
        f"{body}\n\n"
        f"Question: What is the secret code for {key}?\n"
        f"Answer with a single token in the form 'Answer: <token>'."
    )
    return question, value, {
        "task": "niah_single", "needle_pos": pos, "n_distractors": n_distractors,
    }


# ─────────────────────────────────────────────────────────────────────
#  Task: niah_multikey
# ─────────────────────────────────────────────────────────────────────


def _gen_niah_multikey(rng: random.Random, n_distractors: int) -> tuple[str, str, dict]:
    n_needles = rng.choice([3, 4, 5])
    keys = [_random_token(rng, 6) for _ in range(n_needles)]
    values = [_random_token(rng, 5) for _ in range(n_needles)]
    needles = [f"The code {k} maps to {v}." for k, v in zip(keys, values)]
    haystack = _make_haystack(rng, n_distractors)
    positions = sorted(rng.sample(range(len(haystack) + 1), n_needles))
    for ofs, (p, n) in enumerate(zip(positions, needles)):
        haystack = _insert_needle(haystack, p + ofs, n)
    body = " ".join(haystack)
    target_idx = rng.randint(0, n_needles - 1)
    target_key = keys[target_idx]
    target_value = values[target_idx]
    question = (
        f"Read the following passage carefully.\n\n"
        f"{body}\n\n"
        f"Question: What does code {target_key} map to?\n"
        f"Answer with a single token in the form 'Answer: <token>'."
    )
    return question, target_value, {
        "task": "niah_multikey", "n_needles": n_needles,
        "n_distractors": n_distractors,
    }


# ─────────────────────────────────────────────────────────────────────
#  Task: multihop_var
# ─────────────────────────────────────────────────────────────────────


def _gen_multihop_var(rng: random.Random, n_distractors: int) -> tuple[str, str, dict]:
    n_hops = rng.choice([3, 4, 5])
    var_names = []
    used = set()
    while len(var_names) < n_hops:
        n = "v_" + _random_token(rng, 3).lower()
        if n in used:
            continue
        used.add(n)
        var_names.append(n)
    initial_value = rng.randint(100, 999)
    assignments = [f"Variable {var_names[0]} is set to {initial_value}."]
    for i in range(1, n_hops):
        assignments.append(
            f"Variable {var_names[i]} takes the same value as {var_names[i - 1]}."
        )
    haystack = _make_haystack(rng, n_distractors)
    positions = sorted(rng.sample(range(len(haystack) + 1), n_hops))
    for ofs, (p, a) in enumerate(zip(positions, assignments)):
        haystack = _insert_needle(haystack, p + ofs, a)
    body = " ".join(haystack)
    target_var = var_names[-1]
    question = (
        f"Read the following passage carefully.\n\n"
        f"{body}\n\n"
        f"Question: What is the value of variable {target_var}?\n"
        f"Answer with the integer in the form 'Answer: <integer>'."
    )
    return question, str(initial_value), {
        "task": "multihop_var", "n_hops": n_hops,
        "n_distractors": n_distractors,
    }


# ─────────────────────────────────────────────────────────────────────
#  Task: aggregation_count
# ─────────────────────────────────────────────────────────────────────


_RARE_KEYWORDS = (
    "octopus", "lighthouse", "porcelain", "compass", "tornado",
    "obsidian", "marigold", "telegraph", "saxophone", "comet",
)


def _gen_aggregation_count(rng: random.Random, n_distractors: int) -> tuple[str, str, dict]:
    target_word = rng.choice(_RARE_KEYWORDS)
    count = rng.randint(3, 8)
    haystack = _make_haystack(rng, n_distractors)
    insert_positions = sorted(rng.sample(range(len(haystack) + 1), count))
    insertion_sentences = [
        f"Someone briefly mentioned the word {target_word} in passing."
        for _ in range(count)
    ]
    for ofs, (p, s) in enumerate(zip(insert_positions, insertion_sentences)):
        haystack = _insert_needle(haystack, p + ofs, s)
    body = " ".join(haystack)
    question = (
        f"Read the following passage carefully.\n\n"
        f"{body}\n\n"
        f"Question: How many times does the word '{target_word}' appear in the "
        f"passage above? Count exact case-sensitive matches.\n"
        f"Answer with the integer in the form 'Answer: <integer>'."
    )
    return question, str(count), {
        "task": "aggregation_count", "target_word": target_word,
        "n_distractors": n_distractors,
    }


# ─────────────────────────────────────────────────────────────────────
#  Master generator.
# ─────────────────────────────────────────────────────────────────────


_TASKS = (
    ("niah_single", _gen_niah_single),
    ("niah_multikey", _gen_niah_multikey),
    ("multihop_var", _gen_multihop_var),
    ("aggregation_count", _gen_aggregation_count),
)
_TASK_RATIO = (0.30, 0.30, 0.20, 0.20)

_DISTRACTOR_RANGE = (60, 200)  # 60-200 sentences => ~400-1500 tokens


def _sample_task(rng: random.Random):
    r = rng.random()
    cum = 0.0
    for (name, fn), w in zip(_TASKS, _TASK_RATIO):
        cum += w
        if r < cum:
            return name, fn
    return _TASKS[-1]


def generate_items(block_seed, n_items: int) -> list[dict]:
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        name, fn = _sample_task(item_rng)
        n_dist = item_rng.randint(*_DISTRACTOR_RANGE)
        question, gold, meta = fn(item_rng, n_dist)
        out.append(
            {
                "src": f"v31_long_context_ruler/{name}",
                "question": question,
                "gold": gold,
                **meta,
            }
        )
    return out


def grade_response(response: str, gold: str) -> bool:
    """Forgiving grader; accepts ``Answer: <token>`` or last token /
    integer in the response.
    """
    if not response:
        return False
    text = response.strip()
    target = str(gold).strip()
    m = re.search(r"answer\s*[:\-]?\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
    if m and m.group(1) == target:
        return True
    # Look for the exact token anywhere in the response (case-sensitive).
    if re.search(rf"\b{re.escape(target)}\b", text):
        return True
    # If gold is integer, allow last integer in response.
    if target.lstrip("-").isdigit():
        nums = re.findall(r"-?\d+", text)
        if nums and nums[-1] == target:
            return True
    return False


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=4)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']!r}")
        print(it["question"][:300])
        print("...")
        print(it["question"][-300:])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
