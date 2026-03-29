"""Verify tokenizer compatibility between teacher and student models."""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

TEACHER_TOKENIZER = "zai-org/GLM-5"

# Diverse test strings covering code, prose, and special chars
_TEST_STRINGS = [
    "def hello_world():",
    "import numpy as np",
    "The quick brown fox jumps over the lazy dog",
    "class MyClass(BaseClass):\n    def __init__(self):\n        pass",
    '{"key": "value", "numbers": [1, 2, 3]}',
    "# Comment with special chars: αβγ ñ ü",
]


def check_tokenizer_compatibility(
    student_model: str,
    teacher_tokenizer_name: str = TEACHER_TOKENIZER,
) -> bool:
    """
    Check that the student model uses the same tokenizer as the teacher.

    Compares:
        1. Vocabulary size.
        2. Encoded token IDs for a set of representative strings.

    Returns True if the tokenizer is compatible, False otherwise.
    """
    try:
        teacher_tok = AutoTokenizer.from_pretrained(
            teacher_tokenizer_name, trust_remote_code=True
        )
        student_tok = AutoTokenizer.from_pretrained(
            student_model, trust_remote_code=True
        )
    except Exception as exc:
        logger.error("Failed to load tokenizer(s): %s", exc)
        return False

    if teacher_tok.vocab_size != student_tok.vocab_size:
        logger.warning(
            "Vocab size mismatch: teacher=%d, student=%d",
            teacher_tok.vocab_size,
            student_tok.vocab_size,
        )
        return False

    for s in _TEST_STRINGS:
        t_ids = teacher_tok.encode(s)
        s_ids = student_tok.encode(s)
        if t_ids != s_ids:
            logger.warning("Encoding mismatch for: %r", s)
            return False

    return True
