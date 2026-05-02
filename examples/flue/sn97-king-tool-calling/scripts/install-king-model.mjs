#!/usr/bin/env node
/**
 * Idempotent patch that registers ``distil/sn97-king`` with the
 * @mariozechner/pi-ai model registry that ships inside Flue.
 *
 * ─── Why a postinstall patch? ────────────────────────────────────────────
 * Flue's ``init({ model: 'provider/model-id' })`` looks up models in
 * pi-ai's ``MODELS`` dict, which is populated ONCE at module load. There
 * is no runtime ``registerModel()`` API, and pi-ai's package ``exports``
 * map blocks deep imports of ``models.generated.js``.
 *
 * The cleanest way around that is to write our entry into the file on
 * disk. esbuild and Flue's build pipeline pick it up automatically
 * because pi-ai is loaded externally at runtime.
 *
 * ─── What it does ────────────────────────────────────────────────────────
 * 1. Locate ``node_modules/@mariozechner/pi-ai/dist/models.generated.js``.
 * 2. If our marker line ``// registered: distil/sn97-king`` is already
 *    present, exit (idempotent).
 * 3. Append a JS literal that mutates the exported ``MODELS`` object to
 *    add the ``distil`` provider with the ``sn97-king`` model spec.
 *
 * ─── Re-run safety ───────────────────────────────────────────────────────
 * The marker line guarantees we never double-write. ``npm ci``,
 * ``npm install``, and any ``npm dedupe`` will overwrite the dist files
 * with a fresh copy from the registry, so the postinstall hook reapplies
 * the patch each time.
 *
 * ─── Endpoint ────────────────────────────────────────────────────────────
 * https://api.arbos.life/v1 — the live SN97 king served by vLLM with
 * ``--enable-auto-tool-choice --tool-call-parser qwen3_xml``.
 */
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const piAiDist = join(
  __dirname,
  '..',
  'node_modules',
  '@mariozechner',
  'pi-ai',
  'dist',
);

const MARKER = '// registered: distil/sn97-king';

// 1. Add ``distil/sn97-king`` to the model registry.
function patchModelsGenerated() {
  const file = join(piAiDist, 'models.generated.js');
  if (!existsSync(file)) {
    console.warn(`[register-king] skipped models.generated.js — ${file} not found.`);
    return;
  }
  const original = readFileSync(file, 'utf8');
  if (original.includes(MARKER)) {
    return;
  }
  const patch = `

${MARKER}
MODELS.distil = {
  "sn97-king": {
    id: "sn97-king",
    name: "SN97 King (chat.arbos.life)",
    api: "openai-completions",
    provider: "distil",
    baseUrl: "https://api.arbos.life/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens: 24576,
  },
};
`;
  writeFileSync(file, original + patch, 'utf8');
  console.log('[register-king] patched ' + file);
}

// 2. Add ``distil`` -> ``OPENAI_API_KEY`` to the env-api-keys map.
//    Without this, pi-ai throws "No API key for provider: distil" before it
//    ever calls our endpoint. Our endpoint is open and ignores the key, but
//    pi-ai requires *some* string to construct the OpenAI client.
function patchEnvApiKeys() {
  const file = join(piAiDist, 'env-api-keys.js');
  if (!existsSync(file)) {
    console.warn(`[register-king] skipped env-api-keys.js — ${file} not found.`);
    return;
  }
  const original = readFileSync(file, 'utf8');
  if (original.includes(MARKER)) {
    return;
  }
  const needle = '        openai: "OPENAI_API_KEY",';
  if (!original.includes(needle)) {
    console.warn(
      '[register-king] env-api-keys.js layout changed — could not find the ' +
        '``openai: "OPENAI_API_KEY"`` line. Skipping env patch; tool calls ' +
        'against distil/sn97-king will fail until this script is updated.',
    );
    return;
  }
  // Inject our mapping right after the `openai` entry so the diff stays
  // small and obvious.
  const patched = original.replace(
    needle,
    `${needle}\n        ${MARKER}\n        distil: "OPENAI_API_KEY",`,
  );
  writeFileSync(file, patched, 'utf8');
  console.log('[register-king] patched ' + file);
}

patchModelsGenerated();
patchEnvApiKeys();
