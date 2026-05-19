"""Internal sub-package: chat-completions plumbing for the SN97 king.

Split out of the legacy ``api/agent_runner.py`` mega-module on 2026-05-19
so each concern (sanitizer, vLLM model wrapper, streaming bridge, etc.)
can be read and tested in isolation. The public surface is still the
``agent_runner`` module — these are implementation details and are
re-exported there for backwards compatibility with the test suite and
the ``api/routes/chat.py`` caller.
"""
