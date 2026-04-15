# Distil OpenClaw Pipeline

This project uses two Discord-facing agents plus one hourly ops loop:

- `sn97-bot`: public/community bot on the `arbos` Discord account. It answers public SN97 questions and posts king-change announcements.
- `distil`: private/internal ops bot on the default Discord account. It monitors the Distil channel, can SSH to infrastructure, and handles maintenance work.
- `sn97-hourly-healthcheck`: OpenClaw cron job that runs on the OpenClaw host, checks Distil end to end, applies safe repairs, and posts a short summary to the internal Distil channel.

## Host split

- `distil`: the only machine that runs Distil services and state.
- `remote_vm` / OpenClaw host: Discord ingress, agent orchestration, hourly maintenance logic.
- External dependencies remain external:
  - Lium eval pod
  - chat host
  - benchmark sync host

OpenClaw should not run the validator, API, dashboard, or state sync locally. It should orchestrate and repair them remotely over SSH on `distil`.

## Announcement flow

1. Validator writes `state/announcement.json`.
2. API exposes `/api/announcement` and `/api/announcement/claim`.
3. OpenClaw polls `https://api.arbos.life/api/announcement`.
4. If an announcement exists, OpenClaw claims it first via `/api/announcement/claim`.
5. OpenClaw posts the claimed `message` to Discord on the public `arbos` account.

## Hourly healthcheck flow

1. OpenClaw runs the `distil` agent hourly.
2. The agent SSHes to `distil` and runs:
   - `python3 /opt/distil/repo/scripts/sn97_healthcheck.py --repair --format json`
3. The script performs deterministic checks and safe repairs for:
   - `distil-validator`
   - `distil-api`
   - `distil-dashboard`
   - `distil-benchmark-sync.timer`
   - `chat-tunnel`
   - `caddy`
   - `open-webui`
   - local and public HTTP health
4. The agent then reads new Discord messages, handles bug reports, and escalates to code changes only when deterministic repair is not enough.

## Why this split

- Deterministic infrastructure repair is faster and safer in a script than in a free-form agent prompt.
- The agent is still useful for triage, explanation, code fixes, and deployment.
- Public SN97 replies stay low-privilege, while private Distil ops retain full tooling.
