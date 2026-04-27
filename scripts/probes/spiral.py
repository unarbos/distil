"""Reasoning-spiral probe — miner-friendly version.

Runs three trivial prompts greedily with a 1024-token budget against
the candidate model. A model passes when:

  * It terminates (hits EOS) on at least 2/3 of the prompts within the
    budget.
  * No 6-word phrase repeats >= 15 times in any single output.
  * Mean output length is reasonable (< 600 tokens averaged across the
    three prompts — if you're emitting 600+ tokens of `<think>` to
    answer "Hi" you're spiraling).

The 6-word/15-rep heuristic comes from the post-mortem of the
2026-04-17 reasoning-spiral king (UID 107). Its ``"Hi"`` output
contained the 6-word phrase ``I'll write:* "Hello! How are you``
repeated 102 times.

This is NOT the validator's probe — it's a lighter sanity check
miners can run before submitting. The validator's stricter probe
lives at ``scripts/pod_eval_vllm.py::thinking_collapse_probe`` and
runs against the full per-round teacher distribution.

Usage:
    from scripts.probes.spiral import spiral_probe
    result = spiral_probe(model, tokenizer, device="cuda")
    if not result.passed:
        print(f"FAIL: {result.summary}")
        for sample in result.samples:
            print(f"  prompt={sample.prompt!r} terminated={sample.terminated}")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# Three trivial prompts the teacher answers in <50 tokens. A 4B
# student that takes >1024 tokens to answer "Hi" is broken. Order is
# stable (no block-seed rotation here — this is a miner pre-flight
# check, not a per-round eval; reproducibility > rotation).
PROBE_PROMPTS: tuple[str, ...] = (
    "Hi",
    "What is the largest planet? Answer in one word.",
    "Say the word: done",
)

# A model fails if any 6-word phrase repeats this many times in a
# single output. Calibrated against UID 107's 102-rep `"I'll write:*
# Hello! How are you"` failure mode. 15 catches obvious spirals while
# leaving room for legitimate emphasis ("yes yes yes" etc.).
NGRAM_LEN: int = 6
NGRAM_MAX_REPEAT: int = 15

# Generation budget per prompt. The teacher answers each in <50 tokens.
# 1024 is plenty of headroom for legitimate CoT; anything longer is
# almost certainly a loop.
MAX_NEW_TOKENS: int = 1024

# Pass threshold: at least this many of the three prompts must
# terminate (hit EOS) within the budget.
MIN_TERMINATED: int = 2

# A model that emits >MAX_MEAN_GEN_TOKENS averaged across three trivial
# prompts is almost certainly verbose-as-default — likely to fail the
# `length` and `degeneracy` axes too.
MAX_MEAN_GEN_TOKENS: float = 600.0


@dataclass
class SpiralSample:
    prompt: str
    output_text: str
    output_tokens: int
    terminated: bool
    # Largest n-gram-repeat-count (e.g. 102 for the UID 107 case).
    max_ngram_repeat: int
    repeated_phrase: Optional[str] = None


@dataclass
class SpiralResult:
    passed: bool
    summary: str
    n_terminated: int
    mean_gen_tokens: float
    samples: List[SpiralSample] = field(default_factory=list)


def count_ngram_repeats(text: str, n: int = NGRAM_LEN) -> tuple[int, Optional[str]]:
    """Return (max_count, phrase) for the most-repeated n-word phrase
    in ``text``. Returns (0, None) for short inputs."""
    words = re.findall(r"\S+", text)
    if len(words) < n:
        return 0, None
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        ng = tuple(words[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    if not counts:
        return 0, None
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[1], " ".join(best[0])


def spiral_probe(
    model,
    tokenizer,
    *,
    device: str = "cuda",
    prompts: tuple[str, ...] = PROBE_PROMPTS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> SpiralResult:
    """Run the spiral probe. Returns a :class:`SpiralResult`.

    Imports torch lazily so this module is import-safe in environments
    where torch isn't installed (CPU-only docs builds, etc).

    Args:
        model: a HuggingFace AutoModelForCausalLM instance. Must be
            already moved to ``device`` and put in eval mode.
        tokenizer: matching AutoTokenizer.
        device: forwarded to ``input_ids.to(device)``.
        prompts: override the default trivial-prompt set.
        max_new_tokens: per-prompt generation budget.

    The probe is greedy (do_sample=False, temperature=1.0, top_p=1.0)
    so the result is deterministic for a given (model, weights) pair.
    """
    import torch  # local import — keep module import-safe

    if not getattr(tokenizer, "chat_template", None):
        return SpiralResult(
            passed=False,
            summary="tokenizer has no chat_template — can't run probe",
            n_terminated=0,
            mean_gen_tokens=0.0,
        )

    eos_ids: list[int] = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        except Exception:
            pass
    if getattr(tokenizer, "eos_token_id", None) is not None:
        try:
            eos_ids.append(int(tokenizer.eos_token_id))
        except Exception:
            pass
    eos_ids = list(dict.fromkeys(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0

    samples: list[SpiralSample] = []
    n_terminated = 0
    total_tokens = 0

    with torch.no_grad():
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            try:
                rendered = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
            try:
                gen = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=pad_id,
                    eos_token_id=eos_ids,
                    use_cache=True,
                )
            except Exception as exc:
                samples.append(
                    SpiralSample(
                        prompt=prompt,
                        output_text=f"<error: {exc}>",
                        output_tokens=0,
                        terminated=False,
                        max_ngram_repeat=0,
                    )
                )
                continue
            new_ids = gen[0, ids.shape[1] :]
            gen_len = int(new_ids.shape[0])
            terminated = (gen_len < max_new_tokens) or (
                eos_ids is not None and int(new_ids[-1].item()) in eos_ids
            )
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            max_rep, phrase = count_ngram_repeats(text, NGRAM_LEN)
            sample = SpiralSample(
                prompt=prompt,
                output_text=text,
                output_tokens=gen_len,
                terminated=terminated,
                max_ngram_repeat=max_rep,
                repeated_phrase=phrase if max_rep >= NGRAM_MAX_REPEAT else None,
            )
            samples.append(sample)
            if terminated:
                n_terminated += 1
            total_tokens += gen_len

    mean_gen = total_tokens / max(1, len(samples))

    failures: list[str] = []
    if n_terminated < MIN_TERMINATED:
        failures.append(
            f"only {n_terminated}/{len(samples)} prompts terminated "
            f"(need ≥{MIN_TERMINATED})"
        )
    bad_phrases = [s for s in samples if s.max_ngram_repeat >= NGRAM_MAX_REPEAT]
    for s in bad_phrases:
        failures.append(
            f'on prompt "{s.prompt[:30]}…": {NGRAM_LEN}-word phrase '
            f'"{(s.repeated_phrase or "")[:50]}…" repeated '
            f"{s.max_ngram_repeat}× (max {NGRAM_MAX_REPEAT - 1})"
        )
    if mean_gen > MAX_MEAN_GEN_TOKENS:
        failures.append(
            f"mean output length {mean_gen:.0f} tok > {MAX_MEAN_GEN_TOKENS:.0f} "
            f"on trivial prompts (likely verbose-by-default — will fail length axis)"
        )

    if failures:
        return SpiralResult(
            passed=False,
            summary="; ".join(failures),
            n_terminated=n_terminated,
            mean_gen_tokens=mean_gen,
            samples=samples,
        )

    return SpiralResult(
        passed=True,
        summary=(
            f"{n_terminated}/{len(samples)} terminated, "
            f"mean {mean_gen:.0f} tok, "
            f"max repeat {max((s.max_ngram_repeat for s in samples), default=0)}× "
            f"(threshold {NGRAM_MAX_REPEAT})"
        ),
        n_terminated=n_terminated,
        mean_gen_tokens=mean_gen,
        samples=samples,
    )


__all__ = [
    "PROBE_PROMPTS",
    "NGRAM_LEN",
    "NGRAM_MAX_REPEAT",
    "MAX_NEW_TOKENS",
    "MIN_TERMINATED",
    "MAX_MEAN_GEN_TOKENS",
    "SpiralSample",
    "SpiralResult",
    "count_ngram_repeats",
    "spiral_probe",
]
