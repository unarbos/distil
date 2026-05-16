"""Distil SN97 — competitive Kimi-K2.6 distillation on Bittensor.

This is the rewrite-v2 package. Layout:

- :mod:`distil.settings`     — typed runtime configuration
- :mod:`distil.state`        — atomic JSON R/W + drop-in legacy compat
- :mod:`distil.chain`        — Bittensor metagraph / commitments / weights
- :mod:`distil.eval`         — validator-host orchestration
- :mod:`distil.pod`          — GPU-pod runner (uploaded to Lium per round)
- :mod:`distil.api`          — FastAPI dashboard + chat surface
- :mod:`distil.chat_pod`     — vLLM bootstrap on the king's chat pod
- :mod:`distil.cli`          — operator CLI (``distil ...``)
"""

__version__ = "2.0.0"
