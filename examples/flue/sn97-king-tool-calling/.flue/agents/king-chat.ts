/**
 * Bare chat agent — the king with no extra tools. Use this to compare the
 * raw chat experience against the tool-equipped agents in this folder.
 *
 * Run:
 *   npm run dev
 *   curl -sS http://localhost:3583/agents/king-chat/demo \
 *     -H 'content-type: application/json' \
 *     -d '{"message": "Explain Goodhart'\''s Law in one paragraph."}'
 */
// distil/sn97-king is registered with pi-ai via the postinstall hook in
// scripts/install-king-model.mjs. See .flue/lib/register-king.ts for why.
import type { FlueContext } from '@flue/sdk/client';

export const triggers = { webhook: true };

export default async function ({ init, payload }: FlueContext<{ message: string }>) {
  const agent = await init({ model: 'distil/sn97-king' });
  const session = await agent.session();
  const answer = await session.prompt(payload.message);
  return { answer: answer.text };
}
