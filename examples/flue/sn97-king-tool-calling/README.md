# SN97 King × Flue — Tool-Calling Examples

This directory ships ready-to-run [Flue](https://flueframework.com) agents
that use the **Bittensor SN97 king** as their model, with full
OpenAI-compatible tool calling. The king is the live top-of-leaderboard
student model on subnet 97, served at
[`https://api.arbos.life/v1`](https://api.arbos.life/v1).

## What's in here

| File                                       | What it shows                                                  |
| ------------------------------------------ | -------------------------------------------------------------- |
| `scripts/install-king-model.mjs`           | Postinstall hook — registers `distil/sn97-king` with pi-ai     |
| `.flue/agents/king-chat.ts`                | Bare chat, no tools                                            |
| `.flue/agents/king-weather.ts`             | One-shot tool call (synthetic `get_weather`)                   |
| `.flue/agents/king-research.ts`            | Multi-turn loop with `http_get` + sandbox FS                   |
| `.flue/lib/register-king.ts`               | Doc file explaining how registration works (no runtime code)   |
| `AGENTS.md`                                | Agent context auto-discovered by Flue                          |

## Setup

```bash
cd examples/flue/sn97-king-tool-calling
npm install         # Runs scripts/install-king-model.mjs as postinstall
cp .env.example .env
```

The endpoint at `https://api.arbos.life/v1` is **open** — no API key is
required. The `OPENAI_API_KEY` in `.env` is a sentinel value that satisfies
pi-ai's OpenAI-client constructor; it is never transmitted to the king.

## Run

### One-shot CLI run

```bash
# Bare chat
npm run run:chat

# Synthetic weather tool (single tool call → final answer)
npm run run:weather

# Web-research agent with sandbox FS + http_get tool
npm run run:research
```

Or invoke directly:

```bash
npx flue run king-weather --target node --id demo --env .env \
  --payload '{"city": "San Francisco"}'
```

### Long-running dev server

```bash
npm run dev
# Default port 3583. Hot-reloads on file changes.

# In another shell:
curl -sS http://localhost:3583/agents/king-weather/demo \
  -H 'content-type: application/json' \
  -d '{"city": "San Francisco"}' | jq .
```

## Why this works

The chat pod's vLLM is launched with:

```
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
```

Qwen 3.5 / 3.6 family models emit tool calls as XML
(`<tool_call><function=...><parameter=k>v</parameter></function></tool_call>`).
The `qwen3_xml` parser converts this to the standard OpenAI `tool_calls`
array, so any framework that talks OpenAI-compatible chat completions
(Flue, the Vercel AI SDK, the OpenAI Python SDK, LangChain, LlamaIndex,
…) gets native function calls without any client-side parsing.

## Realistic expectations

The current king is a **4 B-parameter** student model. It handles
single-turn tool calls (`king-weather`) perfectly, but multi-turn
research loops can wander on hard topics — the harness loop terminates,
but the model may not always reach a useful conclusion in 4-6 rounds.

When the staged 40 B / Kimi K2.6-distilled king crowns
(scheduled 2026-05-03), all three demos will start producing
production-quality answers without any code changes here.

You can always check what model is live:

```bash
curl -s https://api.arbos.life/v1/models | jq .data[0]
```

## Adapting this to your own student / king

Edit `scripts/install-king-model.mjs` and change `baseUrl` to your own
OpenAI-compatible endpoint. As long as the upstream serves
Qwen3-XML tool calls and exposes `/v1/chat/completions`, no other
changes are needed. Re-run `npm run register-king` after each `npm
install` to re-apply the patch.

## How registration works under the hood

Flue resolves `init({ model: 'distil/sn97-king' })` against pi-ai's
static `MODELS` dictionary, which is sealed at module load. There is no
public `registerModel()` API and pi-ai's package `exports` map blocks
deep imports of `models.generated.js`.

`scripts/install-king-model.mjs` runs as a `postinstall` hook and
appends two idempotent patches:

1. To `node_modules/@mariozechner/pi-ai/dist/models.generated.js` —
   adds the `distil/sn97-king` entry to the `MODELS` object.
2. To `node_modules/@mariozechner/pi-ai/dist/env-api-keys.js` —
   maps the `distil` provider to the `OPENAI_API_KEY` env var.

Both patches are guarded by a `// registered: distil/sn97-king` marker
so they never double-write.

## Public endpoint reference

```http
GET /v1/models  HTTP/1.1
Host: api.arbos.life

POST /v1/chat/completions  HTTP/1.1
Host: api.arbos.life
Content-Type: application/json

{
  "model": "sn97-king",
  "messages": [...],
  "tools": [...],
  "tool_choice": "auto"
}
```

Returns the live king's HuggingFace repo id under `data[0].id` (e.g.
`talent-richer/help`). The served-model name `sn97-king` is stable; the
underlying weights are swapped automatically each time a new king is
crowned.

The OpenAI-compatible endpoint has its own rate limit (240 requests/
minute/IP) separate from human chat (10/min/IP). For higher-throughput
deployments, host your own client of `chat.arbos.life` or run a fork of
the SN97 student off-mainnet.
