# deploy/

Production deployment artefacts for SN97 (rewrite-v2). Everything here is
host-local; nothing here is imported by the runtime.

## Layout

```
systemd/
  distil-validator.service   validator service loop (eval/service.py)
  distil-api.service         FastAPI + dashboard backend (api/server.py)
  distil-chat-tunnel.service ssh -L 127.0.0.1:8100 → chat pod
caddy.conf                   reverse proxy for api / api-v2 / dashboard / chat
openwebui/
  configure_tools.py         idempotent OWUI tool wiring (one-shot)
  patch_pyodide.py           OWUI Pyodide dispatch patch (one-shot)
  tools/sn97_status.py       OWUI tool plug-in (live SN97 state)
  filters/sn97_grounding.py  OWUI filter (grounds answers in SN97 state)
```

## Cutover order

1. Install the three systemd units; reload + enable; do **not** start
   `distil-validator` yet.
2. Copy `caddy.conf` into `/etc/caddy/Caddyfile`, run
   `caddy validate && systemctl reload caddy`.
3. Run `distil migrate-state --dry-run` and confirm the orphan list,
   then re-run without `--dry-run`.
4. `systemctl start distil-api`; smoke-test `https://api-v2.arbos.life/api/health`.
5. `systemctl start distil-validator`; watch logs for one full round.
6. Once a clean round has set weights, edit `caddy.conf` so
   `api.arbos.life` upstream points to `127.0.0.1:3710` and reload caddy.
7. Stop + disable the legacy validator + API; archive `state/_legacy/`;
   delete `legacy/` from git on a separate PR.
