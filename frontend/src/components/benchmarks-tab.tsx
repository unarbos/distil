"use client";

import { useEffect, useState } from "react";

interface BenchmarkResult {
  task: string;
  score: number | null;
  baseline_score: number | null;
}

interface BenchmarkData {
  timestamp: string;
  model: {
    uid: number;
    kl: number;
    name: string;
  };
  baseline_model: string;
  limit: number;
  results: BenchmarkResult[];
}

// Latest benchmark data — updated when benchmark.py runs
// TODO: serve this from an API endpoint once we have automated benchmark runs
const LATEST_BENCHMARK: BenchmarkData = {
  timestamp: "2026-03-31T17:00:00Z",
  model: {
    uid: 204,
    kl: 0.0834,
    name: "EnvyIrys/moonshine_distillation",
  },
  baseline_model: "Qwen/Qwen3.5-4B",
  limit: 100,
  results: [
    { task: "ARC-Challenge", score: 0.58, baseline_score: 0.53 },
    { task: "HellaSwag", score: 0.70, baseline_score: 0.68 },
    { task: "WinoGrande", score: 0.77, baseline_score: 0.75 },
    { task: "TruthfulQA MC2", score: 0.512, baseline_score: 0.492 },
    { task: "IFEval", score: 0.66, baseline_score: 0.47 },
    { task: "GSM8K", score: null, baseline_score: 0.81 },
    { task: "MMLU-Pro", score: null, baseline_score: null },
  ],
};

function formatScore(score: number | null): string {
  if (score === null) return "—";
  return (score * 100).toFixed(1);
}

function getDelta(score: number | null, baseline: number | null): { text: string; className: string } {
  if (score === null || baseline === null) return { text: "—", className: "text-muted-foreground/50" };
  const delta = (score - baseline) * 100;
  if (delta > 0) return { text: `+${delta.toFixed(1)}`, className: "text-green-400" };
  if (delta < 0) return { text: delta.toFixed(1), className: "text-red-400" };
  return { text: "0.0", className: "text-muted-foreground/50" };
}

export function BenchmarksTab() {
  const data = LATEST_BENCHMARK;
  const comparableResults = data.results.filter(r => r.score !== null && r.baseline_score !== null);
  const wins = comparableResults.filter(r => (r.score ?? 0) > (r.baseline_score ?? 0)).length;
  const totalComparable = comparableResults.length;
  const date = new Date(data.timestamp);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
          🏆 Benchmark Comparison
        </h2>
        <p className="text-xs text-muted-foreground/60 mt-1">
          SN97 Distilled Model vs Qwen&apos;s Own Baseline · Last run:{" "}
          {date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground/50 mb-1">Benchmarked Model</div>
          <div className="text-sm font-mono font-medium truncate" title={data.model.name}>
            {data.model.name}
          </div>
          <div className="text-xs text-muted-foreground/60 mt-0.5">
            UID {data.model.uid} · KL {typeof data.model.kl === "number" ? data.model.kl.toFixed(4) : "—"}
          </div>
        </div>

        <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground/50 mb-1">Baseline</div>
          <div className="text-sm font-mono font-medium truncate">{data.baseline_model}</div>
          <div className="text-xs text-muted-foreground/60 mt-0.5">Qwen&apos;s official 4B distillation</div>
        </div>

        <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground/50 mb-1">Result</div>
          <div className="text-lg font-bold text-green-400">
            Wins {wins}/{totalComparable}
          </div>
          <div className="text-xs text-muted-foreground/60 mt-0.5">
            {data.limit} samples per benchmark
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/20">
              <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
                Benchmark
              </th>
              <th className="text-right px-4 py-3 text-xs font-medium text-blue-400/70 uppercase tracking-wider">
                Distilled
              </th>
              <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
                Baseline
              </th>
              <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
                Delta
              </th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((result) => {
              const delta = getDelta(result.score, result.baseline_score);
              const isWin = result.score !== null && result.baseline_score !== null && result.score > result.baseline_score;
              return (
                <tr key={result.task} className="border-b border-border/10 hover:bg-card/20 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-foreground/80">{result.task}</td>
                  <td className={`px-4 py-2.5 text-right font-mono tabular-nums ${isWin ? "text-green-400 font-semibold" : "text-foreground/70"}`}>
                    {formatScore(result.score)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-foreground/70">
                    {formatScore(result.baseline_score)}
                  </td>
                  <td className={`px-4 py-2.5 text-right font-mono tabular-nums ${delta.className}`}>
                    {delta.text}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Methodology Note */}
      <div className="rounded-xl border border-border/20 bg-card/5 p-4 text-xs text-muted-foreground/50 space-y-1">
        <p className="font-medium text-muted-foreground/70">Methodology</p>
        <p>
          Benchmarks run via{" "}
          <a href="https://github.com/unarbos/distil/blob/main/benchmark.py" target="_blank" rel="noopener noreferrer" className="text-blue-400/70 hover:text-blue-400 underline">
            benchmark.py
          </a>{" "}
          using{" "}
          <a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank" rel="noopener noreferrer" className="text-blue-400/70 hover:text-blue-400 underline">
            lm-eval-harness
          </a>{" "}
          (HF backend) on Vast.ai A100 GPUs.
        </p>
        <p>
          Loglikelihood tasks (ARC, HellaSwag, TruthfulQA, WinoGrande): 0-shot, acc_norm where available.
          Generation tasks (GSM8K, IFEval): 0-shot with chat template, max_gen_toks=512.
        </p>
        <p>
          Full paper:{" "}
          <a href="https://github.com/unarbos/distil/blob/main/paper/benchmark_kl_vs_performance.md" target="_blank" rel="noopener noreferrer" className="text-blue-400/70 hover:text-blue-400 underline">
            KL vs Performance Analysis
          </a>
        </p>
      </div>
    </div>
  );
}
