# Distil v2 — Cutover Runbook

This is a one-shot procedure. It is reversible at every step until step 9.

## 0. Prerequisites

- `rewrite-v2` is merged into `main` (or you are running it from the branch).
- `~/.secrets/distil.env` already has the live secrets the legacy validator
  uses (HF_TOKEN, OPENROUTER_API_KEY, etc.). The new code reads the same
  file via `pydantic-settings`.
- Caddy is up; the legacy backend is running on `127.0.0.1:3700`.

## 1. Install the new package

```bash
cd /opt/distil/repo
git checkout rewrite-v2
python3 -m pip install --upgrade -e .
```

The new `distil` console script will be on `$PATH`.

## 2. Smoke-test offline

```bash
python3 -m compileall -q distil
distil check --model-repo moonshotai/Kimi-K2.6     # known-good
distil check --model-repo Qwen/Qwen3.5-4B          # baseline
```

## 3. Migrate state (dry-run first)

```bash
distil migrate-state --dry-run
# review the output; if it looks right, repeat without --dry-run:
distil migrate-state
```

This moves orphaned state shards (anything no longer read by the live
contract) into `state/_legacy/`. **All 15 live shards stay in place** so
the new validator + API see the same thing the old code was writing.

## 4. Bring up the new API on a side port

```bash
sudo cp deploy/systemd/distil-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now distil-api
curl -fsS http://127.0.0.1:3710/api/health | jq .
```

Add `api-v2.arbos.life` to Caddy (snippet already in `deploy/caddy.conf`)
and reload:

```bash
sudo cp deploy/caddy.conf /etc/caddy/Caddyfile
sudo caddy validate
sudo systemctl reload caddy
curl -fsS https://api-v2.arbos.life/api/health | jq .
```

The dashboard / chat still point at `api.arbos.life` (legacy). Nothing
user-visible has changed.

## 5. Verify KL parity (optional, recommended)

If you've changed the scorer (improvement #1: vLLM `prompt_logprobs`),
prove it agrees with the HF reference path on one production model:

```bash
distil verify-kl --model-repo moonshotai/Kimi-K2.6 --n-prompts 100
# expects: "OK — delta ~1e-5 <= threshold 1e-04"
```

## 6. Shadow-run one round (no set_weights)

```bash
sudo cp deploy/systemd/distil-validator.service /etc/systemd/system/
sudo systemctl daemon-reload
# DO NOT enable yet. Run once interactively:
sudo -u distil distil validate \
    --wallet-name "$WALLET_NAME" \
    --hotkey-name "$HOTKEY_NAME" \
    --once --dry-run --log-level INFO
```

Watch the log; expect:
- `precheck pass: ~50/N`
- `pod eval done in ~30-50m`
- `composite scored N rows`
- `dry-run set_weights skipped`

Inspect the round record:
```bash
jq '.king_uid, .results | length' state/h2h_latest.json
jq '.[]' state/composite_scores.json | head -40
```

## 7. Compare with the legacy round

The legacy validator writes the same shards. Diff the live and new outputs:

```bash
diff <(jq -S . state/composite_scores.json) <(jq -S . state/composite_scores.json.bak)
```

Acceptable deltas: per-axis values < 0.005, identical `final_alpha` /
`worst_3_mean`, identical king selection.

## 8. Cut traffic over

Stop the legacy validator + API:

```bash
sudo systemctl stop distil-validator-legacy distil-api-legacy
```

Edit `deploy/caddy.conf` so `api.arbos.life` upstream points at
`127.0.0.1:3710`, reload caddy:

```bash
sudo systemctl reload caddy
```

Start the new validator (no `--dry-run`):

```bash
sudo systemctl enable --now distil-validator
```

Watch:
```bash
journalctl -fu distil-validator -n 200
```

## 9. Burn the legacy

Once a clean round has set on-chain weights from the new validator:

```bash
git rm -r legacy/
git commit -m "chore: delete legacy/ — rewrite-v2 cutover complete"
```

Archive `state/_legacy/` under `state-archive-YYYYMMDD.tar.gz` and remove
the working copy.

## Rollback

If anything goes sideways before step 8:
```bash
sudo systemctl stop distil-validator distil-api
# legacy backend was never stopped; nothing to do.
```

If after step 8: revert the `caddy.conf` edit, reload caddy, restart the
legacy validator + API. State files are append-only-with-replace so the
legacy code will resume from the last round it owns.
