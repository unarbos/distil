"use client";

import { useState } from "react";

type DocKey =
  | "overview"
  | "protocol"
  | "miner"
  | "validator"
  | "scoring"
  | "antigaming"
  | "api"
  | "constants"
  | "links";

interface DocItem {
  key: DocKey;
  label: string;
}

const NAV: DocItem[] = [
  { key: "overview", label: "Overview" },
  { key: "protocol", label: "Protocol" },
  { key: "miner", label: "Miner quickstart" },
  { key: "validator", label: "Validator setup" },
  { key: "scoring", label: "Scoring & H2H" },
  { key: "antigaming", label: "Anti-spiral" },
  { key: "api", label: "API reference" },
  { key: "constants", label: "Constants" },
  { key: "links", label: "Links" },
];

/**
 * In-dashboard docs. Sidebar nav + body. Idiom from the v2 reference.
 *
 * Body content is composite-first: the eval is 17 axes, KL is one of
 * them, the ranking key is composite.worst, the anti-spiral
 * countermeasures are concrete (reasoning_density,
 * thinking_collapse_probe). Cites composite.py line refs where
 * useful so a reader can verify the claims.
 */
export function DocsPanel() {
  const [active, setActive] = useState<DocKey>("overview");
  return (
    <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      <aside className="border-b md:border-b-0 md:border-r border-border bg-[var(--surface-soft)] px-6 py-8 overflow-y-auto">
        <h6 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium mb-3.5">
          Distil
        </h6>
        <ul className="space-y-1">
          {NAV.map((item) => (
            <li key={item.key}>
              <button
                onClick={() => setActive(item.key)}
                className={[
                  "block w-full text-left text-[13px] px-2 py-1.5 transition-colors",
                  active === item.key
                    ? "bg-foreground text-white"
                    : "text-foreground hover:bg-[var(--surface-elevated)]",
                ].join(" ")}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <div className="px-8 sm:px-12 py-12 max-w-3xl overflow-y-auto">
        <DocsBody active={active} />
      </div>
    </div>
  );
}

function DocsBody({ active }: { active: DocKey }) {
  switch (active) {
    case "overview":
      return (
        <Article>
          <h2>Distil · SN97</h2>
          <p className="lead">
            Bittensor subnet 97. Miners distil a frozen teacher into smaller
            students. Validators score them on a 17-axis composite covering
            distribution match, capability against ground truth, conversational
            quality, generation discipline, and robustness. The highest
            <code> composite.worst </code>wears the crown.
          </p>
          <h3>Teacher</h3>
          <p>
            <strong>Qwen/Qwen3.5-35B-A3B</strong> — 35B-parameter MoE, 3B active.
            Vocab 248,044. Frozen.
          </p>
          <h3>Reference</h3>
          <p>
            <strong>Qwen/Qwen3.5-4B</strong> — the unfine-tuned baseline. The
            reference-broken-axes filter uses it to drop axes the base model
            itself can&apos;t pass under our eval setup.
          </p>
          <h3>Constraint</h3>
          <p>
            Students must stay under <strong>5.25B</strong> total parameters. No
            teacher-activation copies. Identical safetensors hashes are blacklisted
            per commit.
          </p>
        </Article>
      );
    case "protocol":
      return (
        <Article>
          <h2>Protocol</h2>
          <p className="lead">Three steps. Public, deterministic, on-chain.</p>
          <h3>1 · Commit</h3>
          <p>
            Each UID commits a Hugging Face repo + revision on-chain. Weights
            are pinned, sha-verified, and pulled by validators at the next eval
            boundary.
          </p>
          <h3>2 · Score</h3>
          <p>
            The validator runs <strong>300 prompts</strong> through the teacher
            each round (vLLM, fp8, concurrency 32). The student is then scored
            on 17 axes covering distribution match (KL, on-policy RKL, capability,
            length, degeneracy), capability against ground truth (math, code,
            reasoning, IFEval, AIME, MBPP, tool-use, long-context, robustness),
            conversational quality (judge-probe, chat-turns), and generation
            discipline (reasoning-density). Each axis lands in [0, 1].
          </p>
          <h3>3 · Crown</h3>
          <p>
            The king is whoever has the highest <code>composite.worst</code> in{" "}
            <code>composite_scores.json</code>. A challenger dethrones the
            incumbent only when their <code>worst</code> beats the king&apos;s by
            ≥3% (with weighted-mean as tiebreaker in the saturated regime).
          </p>
        </Article>
      );
    case "miner":
      return (
        <Article>
          <h2>Miner quickstart</h2>
          <h3>Need</h3>
          <ul>
            <li>Hugging Face account + writable repo</li>
            <li>An H100 (or equivalent) for training</li>
            <li>Bittensor wallet on netuid 97</li>
          </ul>
          <h3>Steps</h3>
          <ol>
            <li>
              Clone <code>github.com/unarbos/distil</code>
            </li>
            <li>
              Train your student against{" "}
              <code>Qwen/Qwen3.5-35B-A3B</code> · stay under{" "}
              <strong>5.25B params</strong>
            </li>
            <li>Push weights to a HF repo · note the revision sha</li>
            <li>
              Commit on-chain:{" "}
              <code>btcli s commit --netuid 97 --hf_repo &lt;repo&gt; --rev &lt;sha&gt;</code>
            </li>
            <li>
              Wait for the next eval boundary · watch the <em>Live</em> tab
            </li>
          </ol>
          <h3>Constraints</h3>
          <dl className="kv">
            <dt>Max params</dt>
            <dd>5.25B</dd>
            <dt>Max new tokens</dt>
            <dd>8192</dd>
            <dt>Max prompt tokens</dt>
            <dd>1024</dd>
            <dt>Activation copy threshold</dt>
            <dd>0.99999</dd>
          </dl>
          <h3>What to optimise</h3>
          <p>
            <strong>composite.worst</strong>, not KL. KL is one of 17 axes
            (0.15 of the relative tier). A pure-KL model that loops on
            <code> &quot;Hi&quot;</code> or fails grade-school math cannot take
            the crown. Read the axis playbook in{" "}
            <code>docs/MINER_FAQ.md</code> for what each axis rewards and what
            data to mix in for it.
          </p>
        </Article>
      );
    case "validator":
      return (
        <Article>
          <h2>Validator setup</h2>
          <h3>Hardware</h3>
          <ul>
            <li>Multi-GPU node, 80GB+ each</li>
            <li>vLLM concurrency: 32</li>
            <li>NVMe scratch for teacher logits</li>
          </ul>
          <h3>Run</h3>
          <pre>{`git clone https://github.com/unarbos/distil
cd distil
uv sync
python -m distil.validator --netuid 97`}</pre>
          <p>
            Logs surface in the <em>Live</em> tab via{" "}
            <code>/api/eval-progress</code> and <code>/api/gpu-logs</code>.
          </p>
        </Article>
      );
    case "scoring":
      return (
        <Article>
          <h2>Scoring &amp; H2H</h2>
          <h3>Per-round (single-eval mode)</h3>
          <p>
            Each commitment is scored exactly once on its own block-seeded
            300-prompt set. The teacher runs first; its top-128 logprobs are
            cached. Each student is loaded sequentially on the pod (vLLM
            teacher → student forward pass → bench battery). The reference
            baseline (Qwen3.5-4B) runs every round so the
            reference-broken-axes filter can drop axes the base model itself
            can&apos;t pass under our eval setup.
          </p>
          <h3>The 17 axes</h3>
          <p>
            All in [0, 1]. Higher-is-better. Live weights from
            <code> composite.py:AXIS_WEIGHTS </code>+
            <code> BENCH_AXIS_WEIGHTS </code>+
            <code> ARENA_V3_AXIS_WEIGHTS </code>+ probe weights:
          </p>
          <ul>
            <li>
              <strong>Distribution match (relative tier):</strong>{" "}
              <code>on_policy_rkl 0.35</code> · <code>kl 0.15</code> ·{" "}
              <code>capability 0.25</code> · <code>length 0.10</code> ·{" "}
              <code>degeneracy 0.15</code>.
            </li>
            <li>
              <strong>Capability vs ground truth:</strong>{" "}
              <code>math 0.14</code> · <code>code 0.14</code> ·{" "}
              <code>reasoning 0.10</code> · <code>ifeval 0.07</code> ·{" "}
              <code>aime 0.10</code> · <code>mbpp 0.08</code> ·{" "}
              <code>tool_use 0.06</code> · <code>long_context 0.04</code> ·{" "}
              <code>robustness 0.07</code>.
            </li>
            <li>
              <strong>Conversational:</strong>{" "}
              <code>judge_probe 0.15</code> · <code>chat_turns 0.08</code>.
            </li>
            <li>
              <strong>Discipline:</strong>{" "}
              <code>reasoning_density 0.05</code>.
            </li>
          </ul>
          <h3>Dethrone gates (all must pass)</h3>
          <ol>
            <li>
              <strong>Composite-worst margin.</strong>{" "}
              <code>challenger.worst &gt; king.worst × 1.03</code>.
            </li>
            <li>
              <strong>Worst-axis floor.</strong> <code>composite.worst &lt; 0.20</code>{" "}
              vetoes the dethrone even if the margin passes.
            </li>
            <li>
              <strong>Pareto-dominance.</strong> Soft majority: <code>n_wins ≥ n_losses</code>{" "}
              with a 2% noise margin. Insufficient comparable axes fails open.
            </li>
          </ol>
        </Article>
      );
    case "antigaming":
      return (
        <Article>
          <h2>Anti-spiral &amp; anti-gaming</h2>
          <p className="lead">
            Why the eval isn&apos;t just KL.
          </p>
          <h3>The 2026-04-17 reasoning-spiral king</h3>
          <p>
            UID 107 (<code>gtensorapp/prime-dusk-4260</code>) topped the
            KL-only leaderboard with KL=0.049. On the literal prompt{" "}
            <code>&quot;Hi&quot;</code> it generated 4096 tokens — the cap —
            mostly the 6-word phrase{" "}
            <code>I&apos;ll write:* &quot;Hello! How are you</code> repeated
            102 times. It never produced a final answer. On 5/5 reasoning
            benchmarks it was <strong>strictly worse than the unfine-tuned 4B base</strong>.
          </p>
          <p>
            That failure mode is documented in{" "}
            <code>paper/off_policy_cot_collapse.md</code>: pure forward-KL
            on teacher continuations rewards token-level surface match. A
            4B student that mimics the teacher&apos;s &ldquo;wait, let me
            reconsider&rdquo; filler can win KL while never delivering an
            answer.
          </p>
          <h3>Countermeasures shipped</h3>
          <ul>
            <li>
              <code>thinking_collapse_probe</code> in{" "}
              <code>scripts/pod_eval_vllm.py</code>: greedy 1024-token
              budget on three trivial prompts (<code>Hi</code>,{" "}
              <code>largest planet one word</code>,{" "}
              <code>say the word: done</code>). Flags any 6-word phrase
              repeated ≥15× or &lt;2/3 prompts hitting EOS. Sets{" "}
              <code>kl_global_avg = inf</code> so the model never wins H2H.
            </li>
            <li>
              <code>reasoning_density</code> axis (weight 0.05): mean
              generation tokens normalised by per-task target × pass_frac.
              Penalises both over-thinking trivia AND verbose-but-wrong
              answers. Cannot be gamed by short-wrong: pass_frac=0 → axis=0.
            </li>
            <li>
              <code>on_policy_rkl</code> axis (weight 0.35): reverse-KL
              under the student&apos;s own sampling. Catches &ldquo;matches
              teacher logits but collapses under free generation&rdquo;.
            </li>
            <li>
              <code>robustness_bench</code> axis (weight 0.07): re-asks math
              items under K block-rotated paraphrases + noise wrappers.
              Catches models that memorised canonical wordings.
            </li>
            <li>
              Procedural item generation: math, code, reasoning, AIME, MBPP,
              tool-use, long-context, and robustness axes all generate
              items per round from a block-seed. There is no static answer
              key to memorise.
            </li>
          </ul>
          <h3>Anti-copy</h3>
          <ul>
            <li>SHA256 hash duplicate detection — first committer owns the hash.</li>
            <li>
              Logit fingerprinting — cosine similarity &gt; 0.99999 on
              activation vectors flags functional copies even when hashes
              differ.
            </li>
          </ul>
        </Article>
      );
    case "api":
      return (
        <Article>
          <h2>API reference</h2>
          <p className="lead">
            Public read-only API. Cached 60s. Base:{" "}
            <code>https://api.arbos.life</code>.
          </p>
          <p>
            <strong>Live OpenAPI / Swagger UI:</strong>{" "}
            <a
              href="https://api.arbos.life/docs"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              api.arbos.life/docs
            </a>{" "}
            — every endpoint, every parameter, runnable in the browser.
            Bookmark it; this page is just the cheatsheet.
          </p>
          <h3>Endpoint cheatsheet</h3>
          <dl className="kv">
            <dt>GET /api/health</dt>
            <dd>service health, current king, code revision</dd>
            <dt>GET /api/metagraph</dt>
            <dd>chain block + per-UID stake / hotkey / commitment</dd>
            <dt>GET /api/scores</dt>
            <dd>latest KL per UID (one of 17 axes)</dd>
            <dt>GET /api/leaderboard</dt>
            <dd>top-N with composite worst + weighted breakdown</dd>
            <dt>GET /api/miner/{`{uid}`}</dt>
            <dd>per-UID card: model, KL, full composite axes, H2H tail</dd>
            <dt>GET /api/h2h-latest</dt>
            <dd>last bout, full composite per UID, king flag</dd>
            <dt>GET /api/h2h-history?limit=N</dt>
            <dd>past N rounds with composite per result</dd>
            <dt>GET /api/king-history</dt>
            <dd>king flips with reign_blocks</dd>
            <dt>GET /api/history</dt>
            <dd>KL axis time series</dd>
            <dt>GET /api/eval-progress</dt>
            <dd>validator phase + progress (active eval)</dd>
            <dt>GET /api/eval-stream</dt>
            <dd>SSE stream of live eval events</dd>
            <dt>GET /api/benchmarks</dt>
            <dd>
              held-out evalscope reports for the king, teacher, reference (NOT
              the validator&apos;s composite — see Bench tab footer)
            </dd>
            <dt>GET /api/queue</dt>
            <dd>pending challengers in the next eval round</dd>
            <dt>GET /api/incidents</dt>
            <dd>recent ops events (king flips, restarts, DQ events)</dd>
            <dt>GET /api/price</dt>
            <dd>α / τ / USD</dd>
            <dt>GET /api/gpu-logs?limit=N</dt>
            <dd>recent GPU pod log lines (sanitised)</dd>
          </dl>
          <h3>Where to ask questions</h3>
          <p>
            Discord <code>#ა・distil・97</code> in the Bittensor server. Open
            issues / PRs at{" "}
            <a
              href="https://github.com/unarbos/distil"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              github.com/unarbos/distil
            </a>
            .
          </p>
        </Article>
      );
    case "constants":
      return (
        <Article>
          <h2>Constants</h2>
          <dl className="kv">
            <dt>netuid</dt>
            <dd>97</dd>
            <dt>teacher</dt>
            <dd>Qwen/Qwen3.5-35B-A3B</dd>
            <dt>reference</dt>
            <dd>Qwen/Qwen3.5-4B</dd>
            <dt>max student params</dt>
            <dd>5.25B</dd>
            <dt>vocab size</dt>
            <dd>248,044</dd>
            <dt>eval prompts (single-eval)</dt>
            <dd>300</dd>
            <dt>vLLM concurrency</dt>
            <dd>32</dd>
            <dt>composite schema version</dt>
            <dd>28</dd>
            <dt>weighted axes (live)</dt>
            <dd>17</dd>
            <dt>SINGLE_EVAL_DETHRONE_MARGIN</dt>
            <dd>0.03</dd>
            <dt>COMPOSITE_DETHRONE_FLOOR</dt>
            <dd>0.20</dd>
            <dt>top-N always</dt>
            <dd>n/a (single-eval)</dd>
            <dt>max KL axis threshold</dt>
            <dd>2.0</dd>
            <dt>activation copy threshold</dt>
            <dd>0.99999</dd>
          </dl>
        </Article>
      );
    case "links":
      return (
        <Article>
          <h2>Links</h2>
          <ul>
            <li>
              <a href="https://github.com/unarbos/distil">
                GitHub · unarbos/distil
              </a>
            </li>
            <li>
              <a href="https://chat.arbos.life">Chat with the king</a>
            </li>
            <li>
              <a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">
                Teacher on Hugging Face
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/Qwen/Qwen3.5-4B">
                Reference on Hugging Face
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/paper/off_policy_cot_collapse.md">
                Paper · off-policy CoT collapse
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/paper/mechanism_hardening.md">
                Paper · mechanism hardening
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/scripts/validator/composite.py">
                composite.py — axis weights live here
              </a>
            </li>
            <li>
              <a href="https://taomarketcap.com/subnets/97">
                SN97 on TaoMarketCap
              </a>
            </li>
          </ul>
        </Article>
      );
  }
}

function Article({ children }: { children: React.ReactNode }) {
  return <article className="docs-body space-y-4">{children}</article>;
}
