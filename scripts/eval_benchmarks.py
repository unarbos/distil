"""Small benchmark helper utilities shared by the pod eval runner.

Keep this module dependency-light: it is uploaded beside ``pod_eval.py`` on
remote eval pods.
"""
from __future__ import annotations

import random
import re
from typing import Callable


NoisePerturbation = tuple[str, Callable[[str, int], str]]


def _noise_safe_letter_swap(text: str, rate: float, rng_seed: int) -> str:
    """Substitute alpha chars with adjacent QWERTY keys at ``rate``."""
    rng = random.Random(rng_seed)
    qwerty = {
        "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg",
        "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "o",
        "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc", "g": "fthv",
        "h": "gybn", "j": "hkun", "k": "jlim", "l": "ko",
        "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
        "n": "bhjm", "m": "njk",
    }
    out_chars = []
    for ch in text:
        if ch.isalpha() and ch.isascii() and rng.random() < rate:
            low = ch.lower()
            if low in qwerty:
                sub = rng.choice(qwerty[low])
                out_chars.append(sub.upper() if ch.isupper() else sub)
                continue
        out_chars.append(ch)
    return "".join(out_chars)


def _noise_case_jitter(text: str, rate: float, rng_seed: int) -> str:
    rng = random.Random(rng_seed)
    return "".join(
        (ch.swapcase() if ch.isalpha() and ch.isascii() and rng.random() < rate else ch)
        for ch in text
    )


def _noise_extra_whitespace(text: str, rng_seed: int) -> str:
    rng = random.Random(rng_seed)
    out = []
    for ch in text:
        if ch == " " and rng.random() < 0.10:
            out.append(" " * rng.randint(2, 3))
        elif ch == "\n" and rng.random() < 0.15:
            out.append("\n\n")
        else:
            out.append(ch)
    return "".join(out)


def _noise_common_misspellings(text: str) -> str:
    table = [
        (r"\bthe\b", "teh"),
        (r"\byour\b", "youre"),
        (r"\bbecause\b", "becuase"),
        (r"\bdefinitely\b", "definately"),
        (r"\bseparate\b", "seperate"),
        (r"\bachieve\b", "acheive"),
        (r"\boccur\b", "occure"),
        (r"\bweird\b", "wierd"),
        (r"\breceive\b", "recieve"),
    ]
    for pat, rep in table:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def _noise_drop_sentence_periods(text: str, rng_seed: int) -> str:
    rng = random.Random(rng_seed)

    def _maybe_drop(match):
        return "" if rng.random() < 0.5 else match.group(0)

    return re.sub(r"\.(?=\s|$|[A-Z])", _maybe_drop, text)


NOISE_PERTURBATION_TEMPLATES: tuple[NoisePerturbation, ...] = (
    ("light_typos", lambda p, s: _noise_safe_letter_swap(p, rate=0.025, rng_seed=s)),
    ("case_jitter", lambda p, s: _noise_case_jitter(p, rate=0.04, rng_seed=s)),
    (
        "chatter_prefix",
        lambda p, s: (
            "Hey! I'm working through some practice problems - "
            "could you take a look at this one?\n\n" + p
        ),
    ),
    ("chatter_suffix", lambda p, s: p.rstrip() + "\n\nThanks in advance, really appreciate it!"),
    ("extra_whitespace", lambda p, s: _noise_extra_whitespace(p, rng_seed=s)),
    ("common_misspellings", lambda p, s: _noise_common_misspellings(p)),
    ("drop_periods", lambda p, s: _noise_drop_sentence_periods(p, rng_seed=s)),
    (
        "polite_distractor",
        lambda p, s: (
            "(My cat just walked across the keyboard, sorry if anything "
            "looks weird.)\n\n" + p
        ),
    ),
)
