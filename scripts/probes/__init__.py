"""Standalone probes shared between miner-side check tools and the
validator pod.

The pod's full ``thinking_collapse_probe`` (in
``scripts/pod_eval_vllm.py``) is heavyweight: it depends on per-round
teacher samples, MAD-z-score thresholds tuned against the teacher's
own degeneracy distribution, and ~30+ helper functions. We do NOT
import that for miner tools — too much surface, too much risk of
import-time side effects.

Instead, this module provides a lighter, self-contained spiral
detector that captures the *spirit* of the validator's probe: trivial
prompts that the teacher answers in a few tokens, run greedily on the
candidate, with simple repetition + termination thresholds.

A model that fails this miner-side probe will almost certainly fail
the validator's probe too. A model that passes this might still fail
the stricter validator-side test, but at least catches the obvious
cases (UID 107-style 4096-token loops on ``"Hi"``).
"""
