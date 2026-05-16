"""knowledge_multi_hop_kg - v31 multi-hop KG axis.

Each item contains its own synthetic knowledge graph rendered as
inline facts; the question can only be answered by chaining through
those facts (no reliance on parametric world knowledge). Inspired by
MuSiQue (Trivedi et al. TACL 2022) bottom-up chain composition and
RULER's variable tracking generalised to relation names.

Per item:
1. 12-20 entities with synthetic names.
2. Pick a relation schema (family / location / employment).
3. Build a connected fact graph (acyclic for parent_of, symmetric for
   sibling_of, etc.).
4. Walk a 2-5 hop path; the endpoint is the gold.
5. Add 5-15 off-path distractor facts.
6. Render facts in random order; ask the chained question.
"""

from __future__ import annotations

import random


_BENCH_STREAM_OFFSET = 0x5638  # "V8"


# ─────────────────────────────────────────────────────────────────────
#  Synthetic entity name pools.
#
#  We use procedural compounded names ("Person_AB12") so the model
#  has zero parametric prior on these entities. This is the
#  defensive answer to the contamination audit (arXiv 2603.16197):
#  if the model has never seen the entity name, it can't answer
#  except by chaining through inline facts.
# ─────────────────────────────────────────────────────────────────────


def _make_person_name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = "Person_" + "".join(
            rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(4)
        )
        if n not in used:
            used.add(n)
            return n


def _make_place_name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = "Place_" + "".join(
            rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(4)
        )
        if n not in used:
            used.add(n)
            return n


def _make_org_name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = "Org_" + "".join(
            rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(4)
        )
        if n not in used:
            used.add(n)
            return n


# ─────────────────────────────────────────────────────────────────────
#  Schema 1: family tree (multi-hop parent_of / sibling_of)
# ─────────────────────────────────────────────────────────────────────


def _gen_family(rng: random.Random) -> tuple[str, str, dict]:
    """Build a binary tree of N=8-12 people. Then pick a random
    multi-hop path: ancestor, descendant, sibling-of-ancestor, etc.
    """
    n_gens = rng.choice([3, 4])  # 3-gen tree: 7 people; 4-gen: 15
    used = set()
    # Build a tree by levels.
    levels: list[list[str]] = []
    levels.append([_make_person_name(rng, used)])
    parent_of: dict[str, str] = {}  # child -> parent
    children_of: dict[str, list[str]] = {}
    for _ in range(n_gens - 1):
        new_level = []
        for parent in levels[-1]:
            n_kids = rng.choice([1, 2, 2])
            kids = [_make_person_name(rng, used) for _ in range(n_kids)]
            children_of[parent] = kids
            for k in kids:
                parent_of[k] = parent
            new_level.extend(kids)
        levels.append(new_level)
    # Render facts (parent_of edges) - one per child.
    facts = []
    for child, parent in parent_of.items():
        facts.append(f"{parent} is the parent of {child}.")
    # Add a few sibling-of facts (redundant but reinforces graph).
    sibling_facts_added = 0
    target_siblings = rng.randint(1, max(1, n_gens))
    for parent, kids in children_of.items():
        if sibling_facts_added >= target_siblings:
            break
        if len(kids) >= 2:
            facts.append(f"{kids[0]} and {kids[1]} are siblings.")
            sibling_facts_added += 1
    rng.shuffle(facts)
    # Pick a multi-hop question: pattern is one of:
    # 1. "Who is the grandparent of X?" (2-hop up)
    # 2. "Who is the grandchild of X (via Y)?" (2-hop down)
    # 3. "Who is the great-grandparent of X?" (3-hop up; only if
    #    n_gens >= 4)
    pattern = rng.choice(["grandparent", "great_grandparent", "grandchild"])
    if pattern == "great_grandparent" and n_gens < 4:
        pattern = "grandparent"
    if pattern == "grandparent":
        # has a grandparent ⇔ their parent is itself a child of someone
        candidates = [p for p in parent_of if parent_of[p] in parent_of]
        if not candidates:
            return _gen_family(rng)
        target = rng.choice(candidates)
        gold = parent_of[parent_of[target]]
        question = f"Who is the grandparent of {target}?"
    elif pattern == "great_grandparent":
        candidates = [
            p for p in parent_of
            if parent_of[p] in parent_of
            and parent_of[parent_of[p]] in parent_of
        ]
        if not candidates:
            return _gen_family(rng)
        target = rng.choice(candidates)
        gold = parent_of[parent_of[parent_of[target]]]
        question = f"Who is the great-grandparent of {target}?"
    else:  # grandchild (we ask which level-2 is a grandchild of root)
        if n_gens < 3:
            return _gen_family(rng)
        # pick a level-0 person (the root of the tree) and ask for one of
        # their grandchildren.
        root = levels[0][0]
        grandchildren = []
        for kid in children_of.get(root, []):
            grandchildren.extend(children_of.get(kid, []))
        if not grandchildren:
            return _gen_family(rng)
        gold = rng.choice(grandchildren)
        # Disambiguate: ask "name one grandchild"; we accept any
        # listed grandchild as correct.
        question = (
            f"Name one grandchild of {root}. (Any grandchild is "
            f"accepted as correct.)"
        )
        # We store all grandchildren for grading.
        return (
            _format_kg_question(facts, question), gold,
            {"task": "kg_family", "pattern": pattern,
             "all_correct_answers": grandchildren},
        )
    return (
        _format_kg_question(facts, question), gold,
        {"task": "kg_family", "pattern": pattern, "all_correct_answers": [gold]},
    )


# ─────────────────────────────────────────────────────────────────────
#  Schema 2: location chain (multi-hop located_in)
# ─────────────────────────────────────────────────────────────────────


def _gen_location(rng: random.Random) -> tuple[str, str, dict]:
    """Build a chain of nested locations and ask multi-hop containment."""
    chain_len = rng.choice([3, 4, 5])
    used = set()
    chain = [_make_place_name(rng, used) for _ in range(chain_len)]
    facts = []
    for i in range(chain_len - 1):
        facts.append(f"{chain[i]} is located inside {chain[i + 1]}.")
    # Distractor: add a parallel chain.
    parallel = [_make_place_name(rng, used) for _ in range(rng.randint(2, 3))]
    for i in range(len(parallel) - 1):
        facts.append(f"{parallel[i]} is located inside {parallel[i + 1]}.")
    # A few unrelated facts.
    for _ in range(rng.randint(2, 4)):
        a = _make_place_name(rng, used)
        b = _make_place_name(rng, used)
        facts.append(f"{a} is located inside {b}.")
    rng.shuffle(facts)
    # Question: outermost containing region of innermost place.
    target = chain[0]
    hops = rng.randint(2, chain_len - 1)
    gold = chain[hops]
    question = (
        f"Starting from {target} and going up the location hierarchy "
        f"{hops} step{'s' if hops > 1 else ''}, which place do we "
        f"reach?"
    )
    return (
        _format_kg_question(facts, question), gold,
        {"task": "kg_location", "hops": hops, "all_correct_answers": [gold]},
    )


# ─────────────────────────────────────────────────────────────────────
#  Schema 3: organisation employment chain
# ─────────────────────────────────────────────────────────────────────


def _gen_employment(rng: random.Random) -> tuple[str, str, dict]:
    """Build org -> employee -> role + reports_to chains."""
    used = set()
    n_orgs = rng.choice([2, 3])
    orgs = [_make_org_name(rng, used) for _ in range(n_orgs)]
    n_people = rng.choice([6, 8, 10])
    people = [_make_person_name(rng, used) for _ in range(n_people)]
    # Each person works for one org.
    works_at = {p: rng.choice(orgs) for p in people}
    # Build a reports_to chain within each org.
    reports_to: dict[str, str] = {}
    for org in orgs:
        emps = [p for p in people if works_at[p] == org]
        if len(emps) < 2:
            continue
        # Linear chain p[0] -> p[1] -> ... -> p[-1] (top).
        for i in range(len(emps) - 1):
            reports_to[emps[i]] = emps[i + 1]
    # Render facts.
    facts = []
    for p, o in works_at.items():
        facts.append(f"{p} works at {o}.")
    for r, m in reports_to.items():
        facts.append(f"{r} reports to {m}.")
    rng.shuffle(facts)
    # Question: who is the CEO (top of chain) of org X?
    org_target = rng.choice(orgs)
    emps = [p for p in people if works_at[p] == org_target]
    if len(emps) < 2:
        return _gen_employment(rng)
    # Walk reports_to chain to the top.
    top = emps[0]
    visited = {top}
    while top in reports_to and reports_to[top] not in visited:
        top = reports_to[top]
        visited.add(top)
    gold = top
    question = f"Who is the most senior employee (the one nobody reports to) at {org_target}?"
    return (
        _format_kg_question(facts, question), gold,
        {"task": "kg_employment", "all_correct_answers": [gold]},
    )


# ─────────────────────────────────────────────────────────────────────
#  Question formatter.
# ─────────────────────────────────────────────────────────────────────


def _format_kg_question(facts: list[str], question: str) -> str:
    fact_block = "\n".join(f"- {f}" for f in facts)
    return (
        "Use ONLY the facts listed below. Do not use any outside "
        "knowledge.\n\nFACTS:\n"
        f"{fact_block}\n\nQUESTION: {question}\n"
        "Answer with the entity name in the form 'Answer: <name>'."
    )


# ─────────────────────────────────────────────────────────────────────
#  Master generator.
# ─────────────────────────────────────────────────────────────────────


_SCHEMAS = (
    ("kg_family", _gen_family, 0.4),
    ("kg_location", _gen_location, 0.3),
    ("kg_employment", _gen_employment, 0.3),
)


def _sample_schema(rng: random.Random):
    r = rng.random()
    cum = 0.0
    for name, fn, w in _SCHEMAS:
        cum += w
        if r < cum:
            return name, fn
    return _SCHEMAS[-1][0], _SCHEMAS[-1][1]


def generate_items(block_seed, n_items: int) -> list[dict]:
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(max(1, int(n_items))):
        per_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_seed)
        name, fn = _sample_schema(item_rng)
        question, gold, meta = fn(item_rng)
        out.append(
            {
                "src": f"v31_knowledge_multi_hop_kg/{name}",
                "question": question,
                "gold": gold,
                **meta,
            }
        )
    return out


def grade_response(response: str, gold: str, all_correct=None) -> bool:
    """Forgiving grader; accepts any 'Answer: <name>' or any
    standalone occurrence of the gold (or any entity in
    ``all_correct``).
    """
    if not response:
        return False
    import re
    text = response.strip()
    targets = [str(gold).strip()]
    if all_correct:
        targets.extend(str(t).strip() for t in all_correct)
    targets = list({t for t in targets if t})  # de-dup
    m = re.search(r"answer\s*[:\-]?\s*([\w_]+)", text, re.IGNORECASE)
    if m:
        if any(m.group(1) == t for t in targets):
            return True
    for t in targets:
        if re.search(rf"\b{re.escape(t)}\b", text):
            return True
    return False


def _self_test_demo():  # pragma: no cover
    items = generate_items(block_seed=42, n_items=4)
    for it in items:
        print("-" * 60)
        print(f"src={it['src']} gold={it['gold']!r}")
        print(it["question"])


if __name__ == "__main__":  # pragma: no cover
    _self_test_demo()
