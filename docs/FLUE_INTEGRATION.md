# Flue Integration

The SN97 king model is now usable from any framework that speaks
OpenAI-compatible chat completions, including
[**Flue**](https://flueframework.com), the agent-harness framework. This
doc explains the integration, the public endpoint, and the
ready-to-run example agents in
[`examples/flue/sn97-king-tool-calling/`](../examples/flue/sn97-king-tool-calling/).

## Public endpoint

```
https://api.arbos.life/v1
```

Two routes are exposed:

```http
GET  /v1/models             # the live king's HF repo id
POST /v1/chat/completions   # standard OpenAI chat completion shape
```

The endpoint is open (no auth required). Tool calls are supported
natively — the chat pod's vLLM is launched with:

```
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
```

Qwen 3.5 / 3.6 student models emit tool calls in XML
(`<tool_call><function=...><parameter=k>v</parameter></function></tool_call>`),
which the `qwen3_xml` parser converts to the standard OpenAI
`tool_calls` array. Clients see native function calls, no client-side
parsing required.

### Rate limits

| Endpoint                       | Limit (per source IP)         |
| ------------------------------ | ----------------------------- |
| `/v1/chat/completions`         | 240 requests / minute         |
| `/api/chat` (human chat proxy) | 10 requests / minute          |

The OpenAI-compatible endpoint has the larger budget because agent
harnesses (Flue, OpenAI Agents SDK, Vercel AI SDK, LangChain, …) loop
the king many times for a single conceptual task, while the human-chat
endpoint never sees more than ~1 req/sec from a real browser.

For higher-throughput needs, run your own SN97 student fork off-mainnet.

## Quickstart with Flue

```bash
cd examples/flue/sn97-king-tool-calling
npm install         # runs scripts/install-king-model.mjs (postinstall)
cp .env.example .env
npm run run:weather
```

In Flue agent code, the king is just a model string:

```ts
import type { FlueContext } from '@flue/sdk/client';

export const triggers = { webhook: true };

export default async function ({ init, payload }: FlueContext) {
  const agent = await init({ model: 'distil/sn97-king' });
  const session = await agent.session();
  return await session.prompt(payload.message);
}
```

With tools:

```ts
import { Type, type FlueContext, type ToolDef } from '@flue/sdk/client';

const search: ToolDef = {
  name: 'search',
  description: 'Search the company knowledge base.',
  parameters: Type.Object({ query: Type.String() }),
  async execute({ query }) {
    const res = await fetch(`https://api.example.com/search?q=${encodeURIComponent(query)}`);
    return await res.text();
  },
};

export default async function ({ init, payload }: FlueContext) {
  const agent = await init({
    model: 'distil/sn97-king',
    tools: [search],
  });
  const session = await agent.session();
  return await session.prompt(payload.message);
}
```

## Why a postinstall patch?

Flue's `init({ model: 'provider/model-id' })` looks the string up in
[`@mariozechner/pi-ai`](https://www.npmjs.com/package/@mariozechner/pi-ai)'s
static `MODELS` dictionary. The dictionary is sealed at module load
time and there is no public `registerModel()` API. pi-ai's package
`exports` map also blocks deep imports of `models.generated.js`, so a
runtime injection module can't reach the relevant object.

`scripts/install-king-model.mjs` writes two idempotent patches into
`node_modules/@mariozechner/pi-ai/dist/`:

| File                    | Patch                                                                |
| ----------------------- | -------------------------------------------------------------------- |
| `models.generated.js`   | Adds `MODELS.distil = { 'sn97-king': { … openai-completions … } }`   |
| `env-api-keys.js`       | Adds `distil: 'OPENAI_API_KEY'` to the provider → env-var map        |

Both patches are guarded by a `// registered: distil/sn97-king` marker
so they never double-write. They re-apply on every `npm install` /
`npm ci` because the dist files are restored from the registry.

## Other frameworks

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.arbos.life/v1",
    api_key="sn97-public-no-auth-required",  # ignored
)
resp = client.chat.completions.create(
    model="sn97-king",
    messages=[{"role": "user", "content": "Hello, king!"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(resp.choices[0].message)
```

### Vercel AI SDK

```ts
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { generateText } from 'ai';

const sn97 = createOpenAICompatible({
  name: 'sn97',
  baseURL: 'https://api.arbos.life/v1',
  apiKey: 'sn97-public-no-auth-required',
});

const { text } = await generateText({
  model: sn97('sn97-king'),
  prompt: 'Hello, king!',
});
```

### LangChain (Python)

```python
from langchain_openai import ChatOpenAI

king = ChatOpenAI(
    model="sn97-king",
    base_url="https://api.arbos.life/v1",
    api_key="sn97-public-no-auth-required",
)
king.invoke("Hello, king!")
```

## Live model lookup

The HuggingFace repo id of the current king changes whenever a new king
is crowned. Clients that need to know exactly which weights answered
their request can read the OpenAI-compatible models list:

```bash
curl -s https://api.arbos.life/v1/models | jq '.data[0]'
```

```json
{
  "id": "talent-richer/help",
  "object": "model",
  "created": 1777741178,
  "owned_by": "distil-sn97-uid221"
}
```

The served-model name `sn97-king` is stable across king rotations — the
underlying weights are hot-swapped under the hood without breaking
existing clients.
