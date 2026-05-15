"""Operator CLI dispatch — single ``distil`` entrypoint.

Subcommands:

- ``distil miner``         — submit a model commitment on-chain
- ``distil check``         — pre-submission precheck on a HF model
- ``distil validate``      — run the validator service loop
- ``distil verify-kl``     — one-shot HF↔vLLM scorer agreement test
- ``distil migrate-state`` — archive orphaned state shards to ``state/_legacy/``
- ``distil api``           — run the FastAPI dashboard server
"""
