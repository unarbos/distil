"""Unit tests for the chat-transcript splitter in distil_kl_train_prebuilt.py.
Uses a minimal torch stub so the tests run without a GPU or real torch installed.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "examples" / "distil_kl_train_prebuilt.py"


def _stub_torch():
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    fake.Tensor = object
    fake.device = object
    fake.int32 = object()
    fake.float32 = object()
    fake.long = object()
    fake.bfloat16 = object()
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    nn_func = types.ModuleType("torch.nn.functional")
    nn_mod.functional = nn_func
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.AdamW = object
    fake.nn = nn_mod
    fake.optim = optim_mod
    sys.modules["torch"] = fake
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_func
    sys.modules["torch.optim"] = optim_mod


_stub_torch()


def _load_module():
    spec = importlib.util.spec_from_file_location("distil_kl_train_prebuilt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["distil_kl_train_prebuilt"] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class TranscriptSplitterTests(unittest.TestCase):

    def test_splits_last_assistant_turn_qwen_markers(self):
        text = (
            "<|im_start|>system\nBe terse.<|im_end|>\n"
            "<|im_start|>user\nFirst question<|im_end|>\n"
            "<|im_start|>assistant\nFirst answer<|im_end|>\n"
            "<|im_start|>user\nSecond question<|im_end|>\n"
            "<|im_start|>assistant\nSecond answer<|im_end|>"
        )
        prompt, completion = mod._split_chat_transcript_text(text)
        self.assertTrue(
            prompt.endswith("<|im_start|>assistant\n"),
            f"expected prompt to end with assistant header, got: {prompt!r}",
        )
        self.assertIn("Second answer", completion)

    def test_splits_plain_assistant_label(self):
        text = "System: rules\n\nUser: question\n\nAssistant: answer text here"
        prompt, completion = mod._split_chat_transcript_text(text)
        self.assertIn("Assistant:", prompt)
        self.assertEqual(completion, "answer text here")

    def test_returns_none_for_plaintext_no_markers(self):
        text = "Just some plain text without any role markers."
        self.assertIsNone(mod._split_chat_transcript_text(text))

    def test_skips_empty_completion(self):
        text = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
        self.assertIsNone(mod._split_chat_transcript_text(text))

    def test_extract_from_prompt_completion_fields(self):
        item = {"question": "What is 2+2?", "answer": "4"}
        result = mod._extract_prompt_completion_text(item, None)
        self.assertIsNotNone(result)
        prompt, completion = result
        self.assertIn("2+2", prompt)
        self.assertEqual(completion, "4")


if __name__ == "__main__":
    unittest.main()
