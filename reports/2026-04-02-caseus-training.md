# Report: Merge caseus's training script + Update README

**Date:** 2026-04-02  
**Branch:** `improvements/validator-fixes-v2`  
**Repo:** `/home/openclaw/distillation`

## What Was Done

1. **Added `examples/distil_kl_train.py`** — Community-contributed KL distillation training script from caseus (GitHub: @winglian), originally submitted as PR #1. The script trains a student model to match the teacher's (Qwen3.5-35B-A3B) output distribution using forward KL divergence on text from `karpathy/climbmix-400b-shuffle`.

2. **Updated `README.md`** with two new sections:
   - **Training Guide** (after Mining Guide) — References the training script, documents GPU requirements (A100 80GB+ for unquantized teacher), shows 3 usage modes, lists key hyperparameters, credits caseus/@winglian, and links to the original gist.
   - **Community Contributions** (before License) — Acknowledges caseus's training script and top-k=128 shadow KL suggestion, notes that PRs are welcome.

## Files Changed

- `examples/distil_kl_train.py` — New file (training script)
- `README.md` — Added Training Guide and Community Contributions sections
- `reports/2026-04-02-caseus-training.md` — This report

## Credit

- Training script by caseus ([@winglian](https://github.com/winglian))
- Original gist: https://gist.github.com/winglian/a8fe6b859ca1f23abcdd550fd5cfa0c5
