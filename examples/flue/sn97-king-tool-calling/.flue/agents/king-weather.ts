/**
 * Minimal Flue agent that demonstrates the SN97 king using a custom tool.
 *
 * Flow:
 *   1. User asks for the weather in some city.
 *   2. King emits a tool_call to ``get_weather``.
 *   3. Our (mocked) tool returns a weather payload.
 *   4. King consumes the result and writes a one-sentence summary.
 *
 * Run:
 *   npm run dev
 *   curl -sS http://localhost:3583/agents/king-weather/demo \
 *     -H 'content-type: application/json' \
 *     -d '{"city": "San Francisco"}'
 */
// distil/sn97-king is registered with pi-ai via the postinstall hook in
// scripts/install-king-model.mjs. See .flue/lib/register-king.ts for why.
import { Type, type FlueContext, type ToolDef } from '@flue/sdk/client';

export const triggers = { webhook: true };

const getWeather: ToolDef = {
  name: 'get_weather',
  description: 'Get the current weather for a named city.',
  parameters: Type.Object({
    city: Type.String({ description: "The city name, e.g. 'San Francisco'." }),
    unit: Type.Optional(
      Type.Union([Type.Literal('celsius'), Type.Literal('fahrenheit')], {
        description: 'Preferred temperature unit. Defaults to celsius.',
      }),
    ),
  }),
  async execute({ city, unit }) {
    const tempC = 11 + Math.floor(Math.random() * 14);
    const conditions = ['sunny', 'partly cloudy', 'overcast', 'foggy'][
      Math.floor(Math.random() * 4)
    ];
    return JSON.stringify({
      city,
      conditions,
      temperature_c: tempC,
      temperature_f: Math.round(tempC * 1.8 + 32),
      wind_kph: 8 + Math.floor(Math.random() * 25),
      requested_unit: unit ?? 'celsius',
      mocked: true,
    });
  },
};

export default async function ({ init, payload }: FlueContext<{ city: string }>) {
  const agent = await init({
    model: 'distil/sn97-king',
    tools: [getWeather],
  });
  const session = await agent.session();
  const answer = await session.prompt(
    `What's the weather in ${payload.city} right now? ` +
      'Use the get_weather tool, then answer in one short sentence.',
  );
  return { answer: answer.text };
}
