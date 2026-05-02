/**
 * Research agent: the SN97 king plans research, uses ``http_get`` to fetch
 * URLs, and uses the Flue virtual-sandbox tools (bash, read, write, grep,
 * glob) for working notes — then delivers a markdown summary.
 *
 * Run:
 *   npm run dev
 *   curl -sS http://localhost:3583/agents/king-research/demo \
 *     -H 'content-type: application/json' \
 *     -d '{"topic": "How does Bittensor SN97 prevent Goodhart-style overfitting?"}'
 *
 * The king has access to:
 *   - The default Flue virtual-sandbox tools (read, write, grep, glob, bash)
 *   - Our custom ``http_get`` tool below
 *
 * The model decides which tools to call. Tool calls are extracted from
 * Qwen3-style XML by vLLM's ``qwen3_xml`` parser and surfaced as native
 * OpenAI ``tool_calls`` to pi-ai → Flue.
 *
 * NOTE — Why no Valibot result schema?
 * Flue's structured-output mechanism asks the model to wrap its answer in
 * ``---RESULT_START---``/``---RESULT_END---`` markers. The current 4 B
 * king is a small student model and doesn't follow that protocol
 * reliably; it frequently returns the answer as plain markdown instead
 * of the expected envelope. Bigger student models (40 B with Kimi K2.6
 * as teacher, rolling out 2026-05-03) will, but for now the demo
 * returns plain markdown so it works on every king crowned today.
 */
// distil/sn97-king is registered with pi-ai via the postinstall hook in
// scripts/install-king-model.mjs. See .flue/lib/register-king.ts for why.
import { Type, type FlueContext, type ToolDef } from '@flue/sdk/client';

export const triggers = { webhook: true };

const httpGet: ToolDef = {
  name: 'http_get',
  description:
    'Fetch a URL and return the response body as text. Use to research a public web page or JSON API. Truncates responses to ~32 KB.',
  parameters: Type.Object({
    url: Type.String({ description: 'A fully-qualified https:// URL.' }),
    headers: Type.Optional(
      Type.Record(Type.String(), Type.String(), {
        description: 'Optional HTTP headers (e.g., Accept, User-Agent).',
      }),
    ),
  }),
  async execute({ url, headers }) {
    if (!url.startsWith('https://') && !url.startsWith('http://')) {
      return JSON.stringify({ error: 'url must start with http:// or https://' });
    }
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const res = await fetch(url, {
        method: 'GET',
        headers: { 'user-agent': 'sn97-king-flue-demo/0.1', ...(headers ?? {}) },
        signal: controller.signal,
      });
      const body = await res.text();
      const truncated = body.length > 32000;
      return JSON.stringify({
        status: res.status,
        content_type: res.headers.get('content-type'),
        body: truncated ? body.slice(0, 32000) : body,
        truncated,
        bytes: body.length,
      });
    } catch (err: any) {
      return JSON.stringify({ error: String(err?.message ?? err) });
    } finally {
      clearTimeout(timeout);
    }
  },
};

export default async function ({ init, payload }: FlueContext<{ topic: string }>) {
  const agent = await init({
    model: 'distil/sn97-king',
    tools: [httpGet],
  });
  const session = await agent.session();
  const answer = await session.prompt(
    `Research the following topic and write a concise, well-sourced ` +
      `markdown summary (<= 200 words, with bullet points).

Topic: ${payload.topic}

Use http_get if you need to fetch a public URL. Cite sources you use as
plain URLs at the end. Do not invent sources. If you don't have enough
information, say so plainly.`,
  );
  return { topic: payload.topic, answer: answer.text };
}
