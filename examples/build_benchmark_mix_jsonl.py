#!/usr/bin/env python3
"""
Build a local JSONL training mix for the Arena-v3 style validator axes.

The output has one field per row:
  {"text": "..."}

It can be used directly with examples/distil_kl_train_prebuilt.py after the
local JSONL loader support in that script:

  python distil/examples/build_benchmark_mix_jsonl.py \
    --output ./distil-checkpoints/arena_mix.jsonl \
    --max_examples 50000

  python distil/examples/distil_kl_train_prebuilt.py train \
    --dataset ./distil-checkpoints/arena_mix.jsonl \
    --dataset_split train \
    --min_chars 0 \
    --max_seq_len 1536 \
    --kl_start_pos 64 \
    --lr 2e-6 \
    --samples_per_step 24
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "Qwen/Qwen3.5-35B-A3B"


HF_SOURCES = [
    # Math / AIME-like reasoning.
    ("openai/gsm8k", "main", "train", 0.12, "math"),
    ("meta-math/MetaMathQA", None, "train", 0.12, "math"),
    ("TIGER-Lab/MathInstruct", None, "train", 0.10, "math"),
    # Code writing / repair style data.
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", 0.12, "code"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train", 0.10, "code"),
    # General instruction following and chat quality.
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", 0.12, "chat"),
    ("Open-Orca/OpenOrca", None, "train", 0.10, "chat"),
    ("garage-bAInd/Open-Platypus", None, "train", 0.08, "reasoning"),
]


def chat_text(tokenizer, user: str, assistant: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user.strip()})
    messages.append({"role": "assistant", "content": assistant.strip()})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = []
        if system:
            parts.append(f"System: {system.strip()}")
        parts.append(f"User: {user.strip()}")
        parts.append(f"Assistant: {assistant.strip()}")
        return "\n\n".join(parts)


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def row_to_text(tokenizer, row: dict[str, Any], kind: str) -> str | None:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list) and messages:
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value")
            if not isinstance(content, str) or not content.strip():
                continue
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            else:
                role = "system" if role == "system" else "user"
            normalized.append({"role": role, "content": content.strip()})
        if len(normalized) >= 2:
            try:
                return tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                return "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in normalized)

    user = first_text(
        row,
        (
            "question", "query", "instruction", "prompt", "problem", "input",
            "task", "text",
        ),
    )
    assistant = first_text(
        row,
        (
            "answer", "response", "output", "solution", "completion",
            "chosen", "target",
        ),
    )
    if not user or not assistant or user == assistant:
        return None

    system_by_kind = {
        "math": "Solve carefully. Give the final answer clearly and concisely.",
        "code": "Write correct Python. Prefer simple, testable functions.",
        "reasoning": "Reason step by step only as much as needed, then answer.",
        "chat": "Follow the user's instruction exactly.",
    }
    return chat_text(tokenizer, user, assistant, system_by_kind.get(kind))


def load_iter(name: str, config: str | None, split: str):
    if config:
        return iter(load_dataset(name, config, split=split, streaming=True))
    return iter(load_dataset(name, split=split, streaming=True))


def synthetic_examples(rng: random.Random, tokenizer, n: int):
    for _ in range(n):
        kind = rng.choice(["calibration", "long_context", "multi_doc", "refactor"])
        if kind == "calibration":
            a, b = rng.randint(10, 90), rng.randint(10, 90)
            if rng.random() < 0.5:
                user = f"What is {a} + {b}? Return only the integer."
                assistant = str(a + b)
            else:
                user = (
                    "A sealed report contains a secret integer code that is not shown. "
                    "What is the exact code? Return only the integer if knowable."
                )
                assistant = "The exact code is not knowable from the information provided."
        elif kind == "long_context":
            target = rng.randint(1000, 9999)
            filler = " ".join(
                f"Document note {i}: value {rng.randint(1000, 9999)} belongs to item {i}."
                for i in range(20)
            )
            user = (
                f"{filler}\n\nImportant: the blue lantern code is {target}. "
                "Question: what is the blue lantern code? Return only the code."
            )
            assistant = str(target)
        elif kind == "multi_doc":
            x, y, z = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
            user = (
                f"Doc A: Mira has {x} red tokens.\n"
                f"Doc B: Sol has {y} more red tokens than Mira.\n"
                f"Doc C: Ivo has {z} fewer red tokens than Sol.\n"
                "How many red tokens does Ivo have? Return only the integer."
            )
            assistant = str(x + y - z)
        else:
            n1, n2 = rng.randint(1, 9), rng.randint(1, 9)
            user = (
                "Refactor this Python function to be clearer while preserving behavior:\n\n"
                f"def f(x):\n    y=[]\n    for i in x:\n        y.append(i*{n1}+{n2})\n    return y\n"
            )
            assistant = (
                f"def f(x):\n"
                f"    return [item * {n1} + {n2} for item in x]\n"
            )
        yield chat_text(tokenizer, user, assistant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max_examples", type=int, default=50000)
    parser.add_argument("--synthetic_frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_n = int(args.max_examples * args.synthetic_frac)
    hf_n = max(0, args.max_examples - synthetic_n)
    weights = [src[3] for src in HF_SOURCES]
    total_w = sum(weights)
    quotas = [int(hf_n * w / total_w) for w in weights]
    quotas[0] += hf_n - sum(quotas)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for text in synthetic_examples(rng, tokenizer, synthetic_n):
            f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
            written += 1

        for (name, config, split, _weight, kind), quota in zip(HF_SOURCES, quotas):
            if quota <= 0:
                continue
            try:
                it = load_iter(name, config, split)
            except Exception as exc:
                print(f"skip {name}: {exc}")
                continue
            kept = 0
            scanned = 0
            while kept < quota and scanned < quota * 30:
                scanned += 1
                try:
                    row = next(it)
                except StopIteration:
                    break
                except Exception as exc:
                    print(f"stop {name}: {exc}")
                    break
                text = row_to_text(tokenizer, row, kind)
                if not text or len(text) < 80:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
                kept += 1
                written += 1
            print(f"{name}: wrote {kept}/{quota}")

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
