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


def _format_defense_message(
    *,
    king_uid: int,
    king_model: str | None,
    top_challenger_uid: int | None,
    top_challenger_model: str | None,
    king_composite_final: float | None,
    top_challenger_final: float | None,
    block: int | None,
    n_challengers: int,
) -> str:
    """Build the king-defense message body.

    Title + framing adapt to the score relationship:

      * king > challenger  →  "🛡️ King Defends the Crown" + positive gap
        (clean victory: king is genuinely the strongest model this round)
      * king < challenger  →  "⚖️ King Holds via 5% Margin Gate"
        (narrow hold: challenger scored higher but didn't clear the
         5% dethrone margin, so the seated king stays. We surface this
         honestly rather than claiming victory.)
      * king == challenger →  same as the clean-win path (tie goes to
        the king under the gate)

    Pre-fix the message always read "King Defends the Crown" with a
    ``(gap +-0.014)`` rendering when the challenger actually scored
    higher — observed live on 2026-05-20 12:20 UTC for UID 21 vs UID
    231 (msg_id 1506627609977684111). The negative-gap rendering came
    from ``f" (gap +{gap:.3f})"`` always prepending ``+``; semantically
    the message was misleading because it framed a narrow margin-gate
    hold as a clean defense.
    """
    is_clean_win = (
        king_composite_final is not None
        and top_challenger_final is not None
        and king_composite_final >= top_challenger_final
    )
    if is_clean_win:
        title = "## 🛡️ King Defends the Crown"
        verb = "held off"
    else:
        title = "## ⚖️ King Holds via 5% Margin Gate"
        verb = "narrowly held off"

    line_king = ""
    if king_model:
        line_king = (
            f"👑 **UID {king_uid}** ([{king_model}]"
            f"(<https://huggingface.co/{king_model}>)) {verb} "
        )
    else:
        line_king = f"👑 **UID {king_uid}** {verb} "
    if top_challenger_uid is not None:
        if top_challenger_model:
            line_king += (
                f"**{n_challengers} challenger"
                f"{'s' if n_challengers != 1 else ''}** — top contender: "
                f"**UID {top_challenger_uid}** "
                f"([{top_challenger_model}]"
                f"(<https://huggingface.co/{top_challenger_model}>))."
            )
        else:
            line_king += (
                f"**{n_challengers} challenger"
                f"{'s' if n_challengers != 1 else ''}** — top contender: "
                f"**UID {top_challenger_uid}**."
            )
    else:
        line_king += (
            f"**{n_challengers} challenger"
            f"{'s' if n_challengers != 1 else ''}**."
        )

    score_line = ""
    if king_composite_final is not None:
        score_line = f"📊 King: **{king_composite_final:.3f}**"
        if top_challenger_final is not None:
            score_line += f" vs top challenger {top_challenger_final:.3f}"
            try:
                gap = king_composite_final - top_challenger_final
                # ``{gap:+.3f}`` renders a signed format (``+0.014`` or
                # ``-0.014``) so we never produce a malformed ``+-0.014``.
                # Label the gap as "lead" when positive, "deficit" when
                # negative — gives the reader an immediate read on
                # whether the king is actually winning.
                if gap >= 0:
                    score_line += f" (lead {gap:+.3f})"
                else:
                    score_line += f" (deficit {gap:+.3f}; held by margin gate)"
            except Exception:  # noqa: BLE001
                pass
        score_line += "\n"

    block_line = f"📦 Block: ``{block}``\n" if block is not None else ""

    return (
        f"{title}\n\n"
        f"{line_king}\n\n"
        f"{score_line}"
        f"{block_line}"
        f"\n"
        f"📈 [Live Dashboard](<{_DASHBOARD_URL}>)"
    )


def announce_king_defense(
    *,
    king_uid: int,
    king_model: str | None,
    top_challenger_uid: int | None,
    top_challenger_model: str | None,
    king_composite_final: float | None = None,
    top_challenger_final: float | None = None,
    block: int | None = None,
    n_challengers: int = 0,
    state_dir: str | None = None,
) -> None:
    """Publish a king-defense announcement (king held the crown).

    Called by :func:`distil.eval.service._emit_round_announcement` when
    a round closes with ``king_changed=False`` AND there was at least
    one real challenger that finished scoring. Per-round, never raises,
    never blocks ``set_weights``.

    ``state/announcement.json`` is overwritten with the defense payload
    just like ``announce_new_king`` does for dethrones; the dashboard
    banner reads the same field and renders either flavour based on
    ``type``.
    """
    if _is_disabled():
        logger.info("announce: DISTIL_ANNOUNCE_DISABLED=1; skipping defense")
        return
    if n_challengers <= 0:
        # No real challengers (king-re-eval-only round, or all
        # challengers DQ'd before scoring). No defense to announce.
        return

    msg = _format_defense_message(
        king_uid=king_uid,
        king_model=king_model,
        top_challenger_uid=top_challenger_uid,
        top_challenger_model=top_challenger_model,
        king_composite_final=king_composite_final,
        top_challenger_final=top_challenger_final,
        block=block,
        n_challengers=n_challengers,
    )

    discord_id = _post_to_discord(msg)

    payload: dict[str, Any] = {
        "type": "king_defense",
        "timestamp": time.time(),
        "posted": bool(discord_id),
        "message": msg,
        "data": {
            "king_uid": king_uid,
            "king_model": king_model,
            "top_challenger_uid": top_challenger_uid,
            "top_challenger_model": top_challenger_model,
            "king_composite_final": king_composite_final,
            "top_challenger_final": top_challenger_final,
            "block": block,
            "n_challengers": n_challengers,
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
            "announce: posted king-defense to Discord (msg_id=%s, uid=%s, n=%d)",
            discord_id, king_uid, n_challengers,
        )
    else:
        logger.info(
            "announce: king-defense queued to state file only (uid=%s, n=%d)",
            king_uid, n_challengers,
        )


__all__ = ["announce_new_king", "announce_king_defense"]
