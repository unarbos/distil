"use client";

import { Card, CardContent } from "@/components/ui/card";
import { formatParams } from "@/lib/utils";
import { TEACHER } from "@/lib/subnet";

interface DocsTabProps {
  scoreToBeat: number | null;
  kingKl: number | null;
}

export function DocsTab({ scoreToBeat, kingKl }: DocsTabProps) {
  return (
    <div className="max-w-3xl space-y-6">
      {/* Score to beat */}
      <Card className="border-orange-400/30 bg-orange-400/5">
        <CardContent className="p-4 flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-orange-400">Score to Beat</p>
            <p className="text-xs text-muted-foreground">
              KL divergence <strong className="text-foreground">&gt;1% lower</strong> than
              the king to claim emissions. Models with KL &gt; 2.0 are disqualified.
            </p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-mono font-bold text-orange-400">
              &lt;{scoreToBeat != null ? scoreToBeat.toFixed(4) : "—"}
            </p>
            <p className="text-[10px] text-muted-foreground">
              {kingKl != null ? `king: ${kingKl.toFixed(4)} − 1%` : "target KL"}
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-6 text-sm leading-relaxed text-muted-foreground">
        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">The Goal</h2>
          <p>
            Distill{" "}
            <a href={`https://huggingface.co/${TEACHER.model}`} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-mono">
              {TEACHER.model}
            </a>{" "}
            ({formatParams(TEACHER.totalParams)} total, {formatParams(TEACHER.activeParams)} active MoE)
            into a smaller model. The miner whose model most closely matches the
            teacher&apos;s output distribution wins 100% of emissions.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">Scoring</h2>
          <p>
            The validator generates 512-token continuations with vLLM-accelerated inference (5–10× faster
            than HF), then extracts full-vocab logits for scoring.
            Your model&apos;s distribution is compared using{" "}
            <strong className="text-foreground">KL divergence</strong> across all{" "}
            {TEACHER.vocabSize.toLocaleString()} tokens. Lower KL = better.
          </p>
          <p>
            Each eval uses 60 prompts from{" "}
            <a href="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              ClimbMix-400B
            </a>
            . Prompts are seeded by on-chain block hash — unpredictable before finalization.
          </p>
          <p>
            <strong className="text-foreground">Winner-take-all:</strong> Lowest KL gets weight 1.0, everyone else gets 0.
          </p>
          <p className="text-xs italic text-muted-foreground/60">
            KL scores from different rounds use different prompts and aren&apos;t directly comparable.
            King is determined by head-to-head on identical prompts.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">King-of-the-Hill</h2>
          <p>
            The current best model (&quot;king&quot;) holds the crown until a challenger beats it by &gt;1%.
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong className="text-foreground">Pre-checks first</strong> — architecture, hash, integrity verified before any GPU time</li>
            <li><strong className="text-foreground">Only new challengers evaluated</strong> — already-scored models keep their scores</li>
            <li><strong className="text-foreground">60 prompts, 95% CI</strong> — tight confidence intervals</li>
            <li><strong className="text-foreground">Same prompts, fair comparison</strong> — king and challenger scored on identical continuations</li>
            <li><strong className="text-foreground">Early stopping</strong> — clearly worse models stopped early to save GPU time</li>
          </ul>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">Model Requirements</h2>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong className="text-foreground">≤{formatParams(TEACHER.maxStudentParams)} total params</strong> — from safetensors metadata</li>
            <li><strong className="text-foreground">Same tokenizer</strong> as teacher (vocab_size={TEACHER.vocabSize.toLocaleString()})</li>
            <li><strong className="text-foreground">No quantization</strong> — GPTQ/AWQ/FP8 rejected</li>
            <li><strong className="text-foreground">Unique weights</strong> — SHA256 duplicate detection</li>
            <li><strong className="text-foreground">Safetensors format</strong> — bf16/fp16</li>
            <li><strong className="text-foreground">One commit per hotkey</strong> — permanent. If DQ&apos;d, register a new hotkey.</li>
          </ul>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">How to Mine</h2>
          <div className="space-y-2">
            <p><strong className="text-foreground">1.</strong> Train a distilled model from <span className="font-mono">{TEACHER.model}</span></p>
            <p><strong className="text-foreground">2.</strong> Upload to HuggingFace as a public repo with safetensors weights</p>
            <p><strong className="text-foreground">3.</strong> Register on subnet 97 and commit with the miner script</p>
          </div>
          <Card className="border-border/50 bg-card/50 mt-3">
            <CardContent className="p-3">
              <pre className="text-xs font-mono overflow-x-auto whitespace-pre text-foreground">{`git clone https://github.com/unarbos/distil.git && cd distil
pip install .

# Pre-check your model (recommended)
python check_model.py --model-repo you/your-model --eval

# Commit (PERMANENT)
python miner.py --wallet-name mywallet --hotkey-name myhotkey \\
  --model-repo you/your-model --netuid 97 --network finney`}</pre>
            </CardContent>
          </Card>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">Anti-Gaming</h2>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong className="text-foreground">Copy detection</strong> — SHA256 hash tracking; first committer owns the weights</li>
            <li><strong className="text-foreground">Block-hash seeded prompts</strong> — unpredictable before finalization</li>
            <li><strong className="text-foreground">Full-distribution KL</strong> — all {TEACHER.vocabSize.toLocaleString()} tokens, not top-k</li>
            <li><strong className="text-foreground">Integrity checks</strong> — models must stay public and unchanged</li>
          </ul>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">Teacher Model</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs rounded-lg border border-border/30 p-3">
            <div>
              <span className="text-muted-foreground/60">Model</span>
              <p className="font-mono text-foreground">{TEACHER.model}</p>
            </div>
            <div>
              <span className="text-muted-foreground/60">Total</span>
              <p className="font-mono text-foreground">{formatParams(TEACHER.totalParams)}</p>
            </div>
            <div>
              <span className="text-muted-foreground/60">Active</span>
              <p className="font-mono text-foreground">{formatParams(TEACHER.activeParams)}</p>
            </div>
            <div>
              <span className="text-muted-foreground/60">Arch</span>
              <p className="font-mono text-foreground">{TEACHER.architecture}</p>
            </div>
            <div>
              <span className="text-muted-foreground/60">Vocab</span>
              <p className="font-mono text-foreground">{TEACHER.vocabSize.toLocaleString()}</p>
            </div>
            <div>
              <span className="text-muted-foreground/60">Max Student</span>
              <p className="font-mono text-foreground">{formatParams(TEACHER.maxStudentParams)}</p>
            </div>
          </div>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-foreground">Roadmap</h2>
          <p className="text-xs text-muted-foreground/70">
            Next teacher candidates under discussion (subject to governance + community feedback).
            No change is scheduled until announced.
          </p>
          <ul className="list-disc pl-5 space-y-1 text-xs">
            <li>
              <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-mono">
                Qwen/Qwen3-Next-80B-A3B-Instruct
              </a>{" "}
              — current teacher.
            </li>
            <li>
              Future MoE candidates with comparable active-param budgets.
            </li>
            <li>
              Next-gen dense 7B-12B teachers if a strong open release appears.
            </li>
          </ul>
          <p className="text-[11px] italic text-muted-foreground/50">
            Want to propose a teacher? Open an issue on{" "}
            <a href="https://github.com/unarbos/distil/issues" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              GitHub
            </a>
            .
          </p>
        </section>
      </div>
    </div>
  );
}
