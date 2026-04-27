"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface BenchmarkPayload {
  uid?: number | null;
  model: string;
  kl?: number | null;
  is_baseline?: boolean;
  is_king?: boolean;
  completed?: boolean;
  benchmarks: Record<string, number | null>;
  counts?: Record<string, number | null>;
  timestamp?: string | number;
  fetched_at?: number;
  limit?: number | null;
  eval_seconds?: number;
}

interface BenchmarksResponse {
  models: BenchmarkPayload[];
  baseline: BenchmarkPayload | null;
}

const TASK_LABELS: Record<string, string> = {
  mmlu: "MMLU",
  mmlu_pro: "MMLU-Pro",
  gsm8k: "GSM8K",
  bbh: "BBH",
  bbh_cot_fewshot: "BBH (CoT)",
  hellaswag: "HellaSwag",
  winogrande: "WinoGrande",
  arc: "ARC-Challenge",
  arc_challenge: "ARC-Challenge",
  truthfulqa_mc2: "TruthfulQA MC2",
  ifeval: "IFEval",
  humaneval: "HumanEval",
};

function formatScore(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return "—";
  return (score * 100).toFixed(1);
}

function formatRelative(epochSec: number): string {
  const deltaMs = Date.now() - epochSec * 1000;
  const s = Math.max(0, Math.floor(deltaMs / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function getDelta(score: number | null | undefined, baseline: number | null | undefined) {
  if (score == null || baseline == null || !Number.isFinite(score) || !Number.isFinite(baseline)) {
    return { text: "—", className: "text-muted-foreground/50" };
  }
  const delta = (score - baseline) * 100;
  if (delta > 0.05) return { text: `+${delta.toFixed(1)}`, className: "text-ok" };
  if (delta < -0.05) return { text: delta.toFixed(1), className: "text-danger" };
  return { text: "0.0", className: "text-muted-foreground/50" };
}

export function BenchmarksTab() {
  const [payload, setPayload] = useState<BenchmarksResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/benchmarks`, { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const json = (await res.json()) as BenchmarksResponse;
        if (!cancelled) { setPayload(json); setError(null); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "load failed");
      }
    };
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-sm text-muted-foreground">
        Failed to load benchmarks ({error}). Retrying…
      </div>
    );
  }

  if (!payload) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-sm text-muted-foreground">
        Loading benchmarks…
      </div>
    );
  }

  const { models, baseline } = payload;
  const latest = models[0] ?? null;

  if (!latest) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-sm text-muted-foreground">
        No benchmark runs yet. Benchmarks run automatically when a new king is crowned.
      </div>
    );
  }

  const taskKeys = Array.from(
    new Set([...(baseline ? Object.keys(baseline.benchmarks) : []), ...Object.keys(latest.benchmarks)])
  );
  const comparable = taskKeys.filter((k) => {
    const a = latest.benchmarks[k];
    const b = baseline?.benchmarks?.[k];
    return a != null && b != null;
  });
  const wins = comparable.filter((k) => (latest.benchmarks[k] ?? 0) > (baseline?.benchmarks?.[k] ?? 0)).length;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Benchmarks</h2>
        <p className="text-xs text-muted-foreground mt-1">
          Current king evaluated with evalscope against the live chat vLLM{baseline ? ` · baseline ${baseline.model}` : ""}.
          {latest.fetched_at && (
            <span className="ml-1 text-muted-foreground/50">
              · refreshed {formatRelative(latest.fetched_at)}
            </span>
          )}
          {latest.limit ? (
            <span className="ml-1 text-muted-foreground/50">· {latest.limit} samples/task</span>
          ) : (
            <span className="ml-1 text-muted-foreground/50">· full eval sets</span>
          )}
        </p>
      </div>

      <div className={`grid grid-cols-1 gap-3 ${baseline ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
        <div className="rounded-xl border border-border/20 bg-card/10 p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">King</div>
          <div className="text-sm font-mono font-medium truncate" title={latest.model}>
            {latest.model}
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">
            {latest.uid != null ? `UID ${latest.uid}` : "—"}
            {latest.kl != null && <> · KL {latest.kl.toFixed(4)}</>}
          </div>
        </div>

        {baseline && (
          <div className="rounded-xl border border-border/20 bg-card/10 p-4">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Baseline</div>
            <div className="text-sm font-mono font-medium truncate">{baseline.model}</div>
            <div className="text-xs text-muted-foreground mt-0.5">Reference model</div>
          </div>
        )}

        <div className="rounded-xl border border-border/20 bg-card/10 p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Result</div>
          {baseline ? (
            <>
              <div className="text-lg font-bold text-ok">
                Wins {wins}/{comparable.length}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {taskKeys.length} tasks
              </div>
            </>
          ) : (
            <>
              <div className="text-lg font-bold text-foreground/80">
                {taskKeys.length} tasks
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">
                Baseline pending
              </div>
            </>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-border/20 bg-card/10 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/20">
              <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Benchmark</th>
              <th className="text-right px-4 py-3 text-xs font-medium text-eval uppercase tracking-wider">King</th>
              {baseline && (
                <>
                  <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Baseline</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Delta</th>
                </>
              )}
              <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">n</th>
            </tr>
          </thead>
          <tbody>
            {taskKeys.map((k) => {
              const a = latest.benchmarks[k];
              const b = baseline?.benchmarks?.[k];
              const d = getDelta(a, b);
              const win = a != null && b != null && a > b;
              return (
                <tr key={k} className="border-b border-border/10 hover:bg-card/20 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-foreground/80">{TASK_LABELS[k] ?? k}</td>
                  <td className={`px-4 py-2.5 text-right font-mono tabular-nums ${win ? "text-ok font-semibold" : "text-foreground/70"}`}>
                    {formatScore(a)}
                  </td>
                  {baseline && (
                    <>
                      <td className="px-4 py-2.5 text-right font-mono tabular-nums text-foreground/70">
                        {formatScore(b)}
                      </td>
                      <td className={`px-4 py-2.5 text-right font-mono tabular-nums ${d.className}`}>
                        {d.text}
                      </td>
                    </>
                  )}
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-muted-foreground/70">
                    {latest.counts?.[k] ?? latest.limit ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="rounded-xl border border-border/20 bg-card/5 p-4 text-xs text-muted-foreground space-y-1">
        <p className="font-medium text-foreground/80">Methodology</p>
        <p>
          Run via{" "}
          <a href="https://github.com/unarbos/distil/blob/main/scripts/run_king_benchmark.py" target="_blank" rel="noopener noreferrer" className="text-eval hover:underline">
            run_king_benchmark.py
          </a>{" "}
          using{" "}
          <a href="https://github.com/modelscope/evalscope" target="_blank" rel="noopener noreferrer" className="text-eval hover:underline">
            evalscope
          </a>{" "}
          against the live king vLLM (chat-king Lium pod). Refreshed nightly at 02:00 UTC and on every new king.
        </p>
      </div>
    </div>
  );
}
