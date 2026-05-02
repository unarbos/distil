/**
 * Marker file documenting how the SN97 king is registered with Flue.
 *
 * The actual registration happens in
 * ``scripts/install-king-model.mjs``, which runs as an ``npm postinstall``
 * hook and patches ``node_modules/@mariozechner/pi-ai/dist/models.generated.js``
 * in place.
 *
 * Why a postinstall patch instead of a runtime import?
 *
 *   - Flue's ``init({ model: 'distil/sn97-king' })`` looks the string up in
 *     pi-ai's ``MODELS`` dict, which is populated ONCE at module load.
 *     There is no runtime ``registerModel()`` API.
 *   - Flue's CLI builds the agent into a single bundled ``server.mjs`` that
 *     externalizes ``@flue/sdk/internal``. That external pi-ai loads its
 *     own ``MODELS`` at runtime, separate from any inlined copy in the
 *     bundle. Mutating an inlined ``MODELS`` from inside the bundle has no
 *     effect on the runtime registry.
 *   - pi-ai's package ``exports`` map blocks deep imports of
 *     ``models.generated.js``, so even ``await import(...)`` from a runtime
 *     module hits ``ERR_PACKAGE_PATH_NOT_EXPORTED``.
 *
 * Patching the file on disk side-steps all of this and keeps the agent
 * code idiomatic — every agent just does
 *
 *     await init({ model: 'distil/sn97-king' });
 *
 * and Flue's standard resolver finds the entry without any glue code.
 *
 * If you ever update ``@mariozechner/pi-ai`` and the patch falls off,
 * re-run ``npm run register-king``.
 */
export const KING = { provider: 'distil', modelId: 'sn97-king' } as const;
