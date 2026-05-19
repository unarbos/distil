"""Dethrone / king-change announcer for the rewrite-v2 validator.

When :func:`distil.eval.service._round` decides that the dethrone gate
fired (i.e. ``king_changed=True``), it calls :func:`announce_new_king`
here. We do two things:

1. Write a pending announcement to ``state/announcement.json`` so the
   dashboard banner (``GET /api/announcement``) and any external
   poller (e.g. the ``Arbos`` Discord bot) can pick it up via the
   existing claim endpoint (``POST /api/announcement/claim``).
2. POST the message DIRECTLY to the configured Discord channel using
   the bot token. The direct post is what users in the channel
   actually see; the state file is the backup path and powers the
   dashboard banner.

Both paths are best-effort and never raise — a dethrone announcement
failure must not crash the round or block ``set_weights``. We log the
failure and move on.

Configuration (env vars, all optional with sensible production defaults
baked into :mod:`distil.settings`):

- ``DISTIL_DISCORD_BOT_TOKEN`` — bot token of the account that owns
  the channel. Without it we still write ``state/announcement.json``
  for the dashboard banner; only the Discord post is skipped.
- ``DISTIL_DISCORD_CHANNEL_ID`` — numeric Discord channel id where
  king-change posts should land. Default: the main ``#distil``
  channel (``1482026267392868583``).
- ``DISTIL_DISCORD_ROLE_ID`` — role id pinged in the message body.
  Default: the ``distil`` notification role (``1482026585358991571``).
- ``DISTIL_ANNOUNCE_DISABLED=1`` — kill-switch for both paths
  (state-file write + Discord post), useful for dry-run / dev.

The legacy ``scripts/validator/announcements.py`` is left in place for
historical reference but is not imported here — its 200-line composite-
formula text is tied to the legacy ``ValidatorState`` and ``WORST_3_MEAN_K``
plumbing that the rewrite-v2 dropped. Producing a NEW, smaller message
keeps the announcer self-contained and unblocks the rewrite.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Production defaults — see frontend/src/lib/subnet-config.json
# (``distilRoleId``) and ``/root/.openclaw/openclaw.json``
# (channel ``1482026267392868583``) for provenance.
_DEFAULT_CHANNEL_ID = "1482026267392868583"
_DEFAULT_ROLE_ID = "1482026585358991571"
_DASHBOARD_URL = "https://distil.arbos.life"
_MINING_GUIDE_URL = "https://github.com/unarbos/distil#mining-guide"


def _is_disabled() -> bool:
    return bool(int(os.environ.get("DISTIL_ANNOUNCE_DISABLED", "0") or 0))


def _channel_id() -> str:
    return os.environ.get("DISTIL_DISCORD_CHANNEL_ID") or _DEFAULT_CHANNEL_ID


def _role_id() -> str:
    return os.environ.get("DISTIL_DISCORD_ROLE_ID") or _DEFAULT_ROLE_ID


def _bot_token() -> str | None:
    tok = os.environ.get("DISTIL_DISCORD_BOT_TOKEN")
    if tok:
        return tok.strip() or None
    # Fallback: read from OpenClaw config so a fresh validator host
    # without the env var still posts (the ``Arbos`` bot account
    # is provisioned there and credentials live alongside the agent
    # workspace). Best-effort: silently skip on permission errors.
    for path in (
        "/root/.openclaw/openclaw.json",
        "/home/distil/.openclaw/openclaw.json",
    ):
        try:
            with open(path) as f:
                cfg = json.load(f)
            tok = (
                cfg.get("channels", {})
                .get("discord", {})
                .get("accounts", {})
                .get("arbos", {})
                .get("token")
            )
            if tok:
                return str(tok).strip()
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            continue
    return None


def _format_message(
    *,
    new_uid: int,
    new_model: str | None,
    prev_uid: int | None,
    prev_model: str | None,
    new_composite_final: float | None,
    prev_composite_final: float | None,
    dethrone_method: str | None,
    block: int | None,
) -> str:
    role_ping = f"<@&{_role_id()}>"
    is_cold_start = prev_uid is None or prev_uid == new_uid

    if is_cold_start:
        title = "## 🏆 King Seated"
        action = f"**UID {new_uid}** is the seated king of Distil SN97."
    else:
        title = "## 🏆 New King of Distil SN97!"
        action = f"**UID {new_uid}** has dethroned **UID {prev_uid}**."

    score_line = ""
    if new_composite_final is not None:
        score_line = f"📊 **Composite final: {new_composite_final:.3f}**"
        if prev_composite_final is not None and not is_cold_start:
            score_line += f" — previous king: {prev_composite_final:.3f}"
        score_line += "\n"

    model_line = ""
    if new_model:
        model_line = (
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
        )

    prev_line = ""
    if not is_cold_start and prev_model:
        prev_line = (
            f"👑 Previous king: [{prev_model}]"
            f"(<https://huggingface.co/{prev_model}>)\n"
        )

    method_line = ""
    if dethrone_method and not is_cold_start:
        method_line = f"⚖️ Dethrone gate: ``{dethrone_method}``\n"

    block_line = f"📦 Block: ``{block}``\n" if block is not None else ""

    return (
        f"{role_ping}\n"
        f"{title}\n\n"
        f"{action}\n\n"
        f"{score_line}"
        f"{model_line}"
        f"{prev_line}"
        f"{method_line}"
        f"{block_line}"
        f"\n"
        f"Check the [mining guide](<{_MINING_GUIDE_URL}>) to compete. "
        f"📈 [Live Dashboard](<{_DASHBOARD_URL}>)"
    )


def _write_state_announcement(state_dir: str, payload: dict[str, Any]) -> None:
    """Drop the pending announcement into ``state/announcement.json``.

    Atomic write via ``<path>.tmp`` + ``os.replace`` so a concurrent
    reader (the API claim endpoint) never sees a half-written file.
    """
    path = os.path.join(state_dir, "announcement.json")
    tmp = path + ".tmp"
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("announce: failed to write %s: %s", path, exc)


def _post_to_discord(message: str) -> str | None:
    """POST the announcement to Discord. Returns the Discord message
    id on success, ``None`` on any failure (we never raise).
    """
    tok = _bot_token()
    if not tok:
        logger.info(
            "announce: DISTIL_DISCORD_BOT_TOKEN not configured; "
            "skipping direct Discord post (state file still written)"
        )
        return None
    channel_id = _channel_id()
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    body = json.dumps({
        "content": message,
        # Allow the role mention so subscribers actually get pinged;
        # everything else is suppressed so a stray @everyone never
        # escapes the model-generated text.
        "allowed_mentions": {"parse": [], "roles": [_role_id()]},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bot {tok}",
            "Content-Type": "application/json",
            "User-Agent": "distil-sn97-validator (announcer)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("id") or "")
    except urllib.error.HTTPError as exc:
        body_excerpt = ""
        try:
            body_excerpt = exc.read().decode("utf-8")[:400]
        except Exception:
            pass
        logger.warning(
            "announce: Discord POST failed (status=%s): %s",
            exc.code, body_excerpt,
        )
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("announce: Discord POST network error: %s", exc)
        return None


def announce_new_king(
    *,
    new_uid: int,
    new_model: str | None,
    prev_uid: int | None,
    prev_model: str | None,
    new_composite_final: float | None = None,
    prev_composite_final: float | None = None,
    dethrone_method: str | None = None,
    block: int | None = None,
    state_dir: str | None = None,
) -> None:
    """Publish a king-change announcement.

    Always best-effort: never raises, never blocks the round on a
    network hiccup. The two side effects (state file write + Discord
    POST) are independent; either can fail without affecting the other.
    """
    if _is_disabled():
        logger.info("announce: DISTIL_ANNOUNCE_DISABLED=1; skipping")
        return

    msg = _format_message(
        new_uid=new_uid,
        new_model=new_model,
        prev_uid=prev_uid,
        prev_model=prev_model,
        new_composite_final=new_composite_final,
        prev_composite_final=prev_composite_final,
        dethrone_method=dethrone_method,
        block=block,
    )

    discord_id = _post_to_discord(msg)

    payload: dict[str, Any] = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": bool(discord_id),
        "message": msg,
        "data": {
            "new_uid": new_uid,
            "new_model": new_model,
            "prev_uid": prev_uid,
            "prev_model": prev_model,
            "new_composite_final": new_composite_final,
            "prev_composite_final": prev_composite_final,
            "dethrone_method": dethrone_method,
            "block": block,
        },
    }
    if discord_id:
        payload["posted_at"] = time.time()
        payload["posted_via"] = "validator-direct"
        payload["discord_message_id"] = discord_id

    if state_dir:
        _write_state_announcement(state_dir, payload)

    if discord_id:
        logger.info(
            "announce: posted new-king to Discord (msg_id=%s, uid=%s)",
            discord_id, new_uid,
        )
    else:
        logger.info(
            "announce: new-king queued to state file only (uid=%s)",
            new_uid,
        )


__all__ = ["announce_new_king"]
