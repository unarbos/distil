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
            than HF), then extracts the teacher&apos;s top-128 logprobs per position
            (<code className="font-mono text-muted-foreground">--max-logprobs 128</code>).
            Your model computes a full-vocab softmax over all {TEACHER.vocabSize.toLocaleString()} tokens,
            gathers + renormalizes over the teacher&apos;s top-128 support, and is compared using{" "}
            <strong className="text-foreground">sparse top-128 KL divergence</strong>.
            Lower KL = better. (Full-vocab dense path exists for reference; disabled in prod — ~150 GB/round at vocab size.)
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
            <li><strong className="text-foreground">Sparse top-128 KL</strong> — teacher returns top-128 logprobs per position via vLLM (<code className="font-mono text-muted-foreground">--max-logprobs 128</code>); student computes full-vocab softmax over all {TEACHER.vocabSize.toLocaleString()} tokens then gathers + renormalizes over the teacher&apos;s top-128 support. Proper KL over the shared 128-token support. Full-vocab path (dense teacher logits) is available in-code for reference but disabled in prod for bandwidth reasons.</li>
            <li><strong className="text-foreground">Integrity checks</strong> — models must stay public and unchanged</li>
          </ul>
        </section>

        <section id="chat-collapse" className="space-y-2 scroll-mt-20">
          <h2 className="text-base font-semibold text-foreground">Chat & Thinking Collapse</h2>
          <p className="text-xs text-muted-foreground/80">
            Pure token-level KL on web text does not punish a student that has forgotten how to stop
            generating. Off-policy distillation on long teacher chains-of-thought produces students
            that fill the thinking block with repeated filler phrases (<code className="font-mono text-foreground">*Wait, I&apos;ll write:*</code>)
            and never emit a final answer — matches the teacher&apos;s short-range token statistics
            perfectly, fails every real chat interaction. See{" "}
            <a className="underline" href="https://thinkingmachines.ai/blog/on-policy-distillation/" target="_blank" rel="noreferrer">Thinking Machines on-policy distillation</a>{" "}
            and{" "}
            <a className="underline" href="https://arxiv.org/abs/2502.07266" target="_blank" rel="noreferrer">arXiv:2502.07266</a>{" "}
            on CoT-complexity mismatch between teacher and student.
          </p>
          <div className="rounded-lg border border-border/30 p-3 text-xs font-mono space-y-1">
            <div><span className="text-muted-foreground/60">chat_response_probe</span> → greedy gen, <code>enable_thinking=False</code> on 4 trivial prompts</div>
            <div><span className="text-muted-foreground/60">DQ if</span> &lt; 50% terminate within <code>768</code> tokens</div>
            <div><span className="text-muted-foreground/60">DQ if</span> &lt; 50% emit non-empty content after stripping <code>&lt;think&gt;…&lt;/think&gt;</code></div>
            <div className="pt-1"><span className="text-muted-foreground/60">thinking_collapse_probe</span> → <code>enable_thinking=True</code> on 5 trivial prompts</div>
            <div><span className="text-muted-foreground/60">per-sample stats</span> gzip ratio, distinct-{'{'}1,2,4{'}'}-gram rate, top-6-gram rate, byte-entropy</div>
            <div><span className="text-muted-foreground/60">DQ if</span> &gt; 33% degenerate: gzip ratio robust z-score &lt; <code>−4σ</code> vs teacher on same prompts (or gzip ratio &lt; <code>0.25</code> when no teacher reference)</div>
            <div><span className="text-muted-foreground/60">DQ if</span> &lt; 66% terminate within <code>1024</code> tokens</div>
          </div>
          <p className="text-xs text-muted-foreground/70">
            The threshold is statistical, not hand-picked: the teacher&apos;s own distribution on identical
            prompts defines the &quot;natural&quot; band via robust median/MAD z-scores (Iglewicz &amp; Hoaglin 1993);
            a student only fails when its compression or top-n-gram rate is outside a <code>4σ</code>
            robust band from the teacher&apos;s empirical distribution.
          </p>
          <p className="text-xs text-muted-foreground/70">
            Refs: Holtzman et al. <a className="underline" href="https://arxiv.org/abs/1904.09751" target="_blank" rel="noreferrer">arXiv:1904.09751</a>{" "}
            (repetition/entropy/Zipfian degeneracy axes); Pillutla et al. MAUVE{" "}
            <a className="underline" href="https://arxiv.org/abs/2102.01454" target="_blank" rel="noreferrer">arXiv:2102.01454</a>{" "}
            (distributional-gap formalism);{" "}
            <a className="underline" href="https://thinkingmachines.ai/blog/on-policy-distillation/" target="_blank" rel="noreferrer">Thinking Machines on-policy distillation</a>{" "}
            and <a className="underline" href="https://arxiv.org/abs/2502.07266" target="_blank" rel="noreferrer">arXiv:2502.07266</a>{" "}
            (why off-policy distillation causes CoT collapse).
          </p>
          <p className="text-xs text-muted-foreground/70">
            Env knobs: <code>THINK_PROBE_MAX_TOKENS</code>, <code>THINK_PROBE_DEGEN_SIGMA</code>,{" "}
            <code>THINK_PROBE_GZIP_FLOOR</code>, <code>THINK_PROBE_TERMINATE_THRESHOLD</code>. Set{" "}
            <code>THINK_COLLAPSE_PROBE=0</code> to skip.
          </p>
        </section>

        <section id="anti-finetune" className="space-y-2 scroll-mt-20">
          <h2 className="text-base font-semibold text-foreground">Fine-Tunability Requirement</h2>
          <p className="text-xs text-muted-foreground/80">
            Submitted models must be continuable-pretraining targets, not dead ends. Every student is
            probed before scoring with a simple cross-entropy pass; models whose gradients explode or
            whose layer norms have been inflated out of range are disqualified (DQ reason prefixed{" "}
            <code className="font-mono text-foreground">anti-finetune:</code>).
          </p>
          <div className="rounded-lg border border-border/30 p-3 text-xs font-mono space-y-1">
            <div><span className="text-muted-foreground/60">Probe input</span> → <span className="text-foreground">&quot;The capital of France is Paris. The capital of Germany is Berlin.&quot;</span></div>
            <div><span className="text-muted-foreground/60">Probe mode</span> → <span className="text-foreground">forward + backward on labels=input_ids (standard causal CE loss)</span></div>
            <div><span className="text-muted-foreground/60">DQ if</span> loss is <code>NaN</code>/<code>Inf</code></div>
            <div><span className="text-muted-foreground/60">DQ if</span> any grad is <code>NaN</code>/<code>Inf</code></div>
            <div><span className="text-muted-foreground/60">DQ if</span> global grad-norm &gt; <code className="text-danger">500</code></div>
            <div><span className="text-muted-foreground/60">DQ if</span> any per-param-type grad-norm &gt; <code className="text-danger">500</code> (norm / embed / lm_head / attn / ffn / bias)</div>
            <div><span className="text-muted-foreground/60">DQ if</span> any LayerNorm/RMSNorm weight <code>|w|</code><sub>max</sub> &gt; <code className="text-danger">30</code></div>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Thresholds are env-tunable (<code>FINETUNE_GRAD_NORM_MAX</code>,{" "}
            <code>FINETUNE_NORM_WEIGHT_MAX</code>). Initial values are heuristic — per mantaLLM&apos;s
            suggestion we will continue to tighten them as we observe probe outputs across the
            miner population. Well-behaved distilled models sit 1–3 orders of magnitude below the
            current caps. If you hit a false DQ, open a thread with your probe numbers from the
            round card (<code>loss</code>, <code>global_grad_norm</code>, <code>worst_param_type</code>,{" "}
            <code>worst_norm_weight</code>).
          </p>
        </section>

        <section id="local-eval" className="space-y-2 scroll-mt-20">
          <h2 className="text-base font-semibold text-foreground">Local Eval (Parity Recipe)</h2>
          <p className="text-xs text-muted-foreground/80">
            Want to see your KL against the current king before you submit? Run the same prompt
            sampling + KL math the validator does, locally, against any HF model.
          </p>
          <div className="rounded-lg border border-border/30 p-3 text-xs font-mono space-y-1">
            <div><span className="text-muted-foreground/60"># start a teacher vLLM on port 8000 (Qwen3.5-35B-A3B)</span></div>
            <div><span className="text-muted-foreground/60"># then from repo root:</span></div>
            <div>STUDENT_HF=<span className="text-foreground">your-org/your-model</span> \</div>
            <div className="pl-4">python scripts/local_eval.py</div>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Env vars: <code>STUDENT_HF</code> (required), <code>BLOCK_NUMBER</code>/<code>BLOCK_HASH</code>{" "}
            (optional — defaults to the latest public round), <code>KING_KL</code> (optional),{" "}
            <code>TEACHER_BACKEND</code> (<code>vllm</code> default, or <code>hf</code> for pure
            transformers), <code>N_PROMPTS</code> (default 300), <code>DEVICE</code>,{" "}
            <code>DTYPE</code>. Results are written to <code>state/local_eval/</code>.
          </p>
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
