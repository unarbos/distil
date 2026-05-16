"""FastAPI dashboard backend + chat-with-king surface.

Run with ``distil api`` (foreground) or via the systemd unit
``deploy/systemd/distil-api.service``. State files are read through
:class:`distil.state.StateStore` (mtime-invalidated). Chat traffic is
forwarded to the king's chat pod via :mod:`distil.api.chat`.
"""
