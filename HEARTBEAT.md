# HEARTBEAT.md — SN97 Bot Periodic Tasks

## Public announcement worker
The Discord-side announcement worker can run outside the API host, so use the public API rather than `127.0.0.1`.

1. `GET https://api.arbos.life/api/announcement`
2. If `type` is null, do nothing.
3. If an announcement is pending:
   - First `POST https://api.arbos.life/api/announcement/claim`
   - Only post if the claim response still contains a real announcement (`type` is not null)
   - Send the returned `message` field to Discord channel `1482026267392868583` using the public `arbos` account
4. Do NOT use `/api/announcement/posted` unless you are handling a legacy fallback path. `/claim` is the idempotent path.
