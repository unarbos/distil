"use client";

import { useEffect, useState } from "react";
import type { H2hComposite, H2hHistoryResponse, H2hLatestResponse } from "@/lib/api";
import { useRefreshKey } from "@/components/auto-refresh";
import { CLIENT_API_BASE } from "@/lib/subnet";
import { formatFixed, formatPromptCount, timeAgo } from "@/lib/utils";

// Axis display order + short labels — stable across the UI so miners
// can build mental models around what each abbreviation means. Grouped
// into KL-like (relative), capability, judge/probe, benchmark, and
// live v3 extension axes.
const AXIS_DISPLAY: Array<{ key: keyof NonNullable<H2hComposite["axes"]>; label: string; group: string }> = [
  { key: "kl", label: "KL", group: "rel" },
  { key: "on_policy_rkl", label: "RKL", group: "rel" },
  { key: "capability", label: "cap", group: "rel" },
  { key: "length", label: "len", group: "rel" },
  { key: "degeneracy", label: "deg", group: "rel" },
  { key: "judge_probe", label: "judge", group: "probe" },
  { key: "math_bench", label: "math", group: "bench" },
  { key: "code_bench", label: "code", group: "bench" },
  { key: "reasoning_bench", label: "reas", group: "bench" },
  { key: "knowledge_bench", label: "know", group: "bench" },
  { key: "ifeval_bench", label: "ifev", group: "bench" },
  { key: "aime_bench", label: "aime", group: "bench" },
  { key: "mbpp_bench", label: "mbpp", group: "bench" },
  { key: "tool_use_bench", label: "tool", group: "bench" },
  { key: "self_consistency_bench", label: "scon", group: "bench" },
  { key: "arc_bench", label: "arc", group: "bench" },
  { key: "truthful_bench", label: "truth", group: "bench" },
  { key: "long_context_bench", label: "long", group: "bench" },
  { key: "procedural_bench", label: "proc", group: "bench" },
  { key: "robustness_bench", label: "robust", group: "bench" },
  { key: "noise_resistance_bench", label: "noise", group: "bench" },
  { key: "reasoning_density", label: "rd", group: "live" },
  { key: "chat_turns_probe", label: "chat", group: "live" },
];

// Colour-code an axis value on a 0–1 scale. A floor of 0.8 is where
// composite-dethrone starts to feel comfortable; we mirror the
// dashboard convention used by the standalone telemetry panel.
function axisColorClass(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "text-muted-foreground/30";
  if (v >= 0.85) return "text-emerald-400/90";
  if (v >= 0.70) return "text-lime-400/80";
  if (v >= 0.55) return "text-amber-400/80";
  if (v >= 0.40) return "text-orange-400/80";
  return "text-rose-400/80";
}

function CompositeAxesRow({ composite, isKing }: { composite: H2hComposite; isKing: boolean }) {
  const axes = composite.axes || {};
  const pareto = composite.pareto || null;
  const worst = composite.worst;
  const weighted = composite.weighted;
  const broken = new Set(composite.broken_axes || []);
  const hasAny = AXIS_DISPLAY.some(({ key }) => {
    const v = (axes as Record<string, unknown>)[key as string];
    return typeof v === "number" && Number.isFinite(v);
  });
  if (!hasAny && !pareto) return null;
  const axisItems = AXIS_DISPLAY.filter(({ key }) => {
    const v = (axes as Record<string, unknown>)[key as string];
    return typeof v === "number" && Number.isFinite(v);
  });
  return (
    <div className="col-span-full px-5 pb-2 pt-0 text-[10px] font-mono text-muted-foreground/70 flex flex-wrap items-center gap-x-3 gap-y-1">
      {/* Worst-axis + weighted chips — these are the PRIMARY ranking keys
       * on a composite dethrone attempt. */}
      {typeof worst === "number" && (
        <span className="inline-flex items-center gap-1">
          <span className="text-muted-foreground/40">worst</span>
          <span className={axisColorClass(worst)}>{worst.toFixed(3)}</span>
        </span>
      )}
      {typeof weighted === "number" && (
        <span className="inline-flex items-center gap-1">
          <span className="text-muted-foreground/40">wavg</span>
          <span className={axisColorClass(weighted)}>{weighted.toFixed(3)}</span>
        </span>
      )}
      {/* Pareto dominance vs king — shadow for now, but useful for miners
       * to see which axes they win / lose vs the current king. */}
      {!isKing && pareto && (
        <span
          className="inline-flex items-center gap-1"
          title={`Pareto vs king: wins ${pareto.wins?.join(", ") || "—"}; losses ${pareto.losses?.join(", ") || "—"}; ties ${pareto.ties?.join(", ") || "—"}. margin=${pareto.margin ?? "—"} of min_comparable=${pareto.min_comparable ?? "—"}.`}
        >
          <span className="text-muted-foreground/40">pareto</span>
          <span className={
            pareto.pareto_wins
              ? "text-emerald-400/80"
              : (pareto.n_wins ?? 0) > (pareto.n_losses ?? 0)
                ? "text-lime-400/70"
                : "text-muted-foreground/50"
          }>
            {pareto.n_wins ?? 0}W / {pareto.n_losses ?? 0}L / {pareto.n_ties ?? 0}T
          </span>
        </span>
      )}
      {/* Per-axis chips. Broken axes (teacher failed this round) are
       * rendered struck-through so miners know not to optimise to
       * noise. */}
      <span className="mx-1 text-muted-foreground/20">·</span>
      {axisItems.map(({ key, label }) => {
        const v = (axes as Record<string, number | undefined>)[key as string];
        const cls = axisColorClass(v);
        const isBroken = broken.has(key as string);
        return (
          <span
            key={key as string}
            title={`${key} = ${v?.toFixed(4) ?? "—"}${isBroken ? " (eval-broken this round — dropped from composite worst/weighted because the teacher or reference base scored below the sanity floor on this axis)" : ""}`}
            className={`inline-flex items-center gap-1 ${isBroken ? "line-through opacity-50" : ""}`}
          >
            <span className="text-muted-foreground/40">{label}</span>
            <span className={cls}>{v != null ? v.toFixed(2) : "—"}</span>
          </span>
        );
      })}
    </div>
  );
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }) + " UTC";
}

function RoundRow({ round, defaultOpen, isLatest }: { round: H2hLatestResponse; defaultOpen: boolean; isLatest: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const results = Array.isArray(round.results) ? round.results : [];
  const epsilon = typeof round.epsilon === "number" && Number.isFinite(round.epsilon) ? round.epsilon : 0.01;
  const epsilonPct = formatFixed(epsilon * 100, 0, "1");
  const king = results.find((r) => r.is_king);
  const bestChallenger = results.find((r) => !r.is_king);
  const kingH2hKl = typeof round.king_h2h_kl === "number" && Number.isFinite(round.king_h2h_kl)
    ? round.king_h2h_kl
    : (typeof king?.kl === "number" && Number.isFinite(king.kl) ? king.kl : null);
  const kingGlobalKl = typeof round.king_global_kl === "number" && Number.isFinite(round.king_global_kl)
    ? round.king_global_kl
    : null;
  const epsilonThreshold = typeof round.epsilon_threshold === "number" && Number.isFinite(round.epsilon_threshold)
    ? round.epsilon_threshold
    : (kingH2hKl != null ? kingH2hKl * (1 - epsilon) : null);
  const nModels = results.length;

  return (
    <div className={`rounded-xl border backdrop-blur-sm overflow-hidden transition-all ${
      isLatest
        ? "border-blue-400/30 bg-blue-400/[0.03]"
        : "border-border/20 bg-card/10"
    }`}>
      {/* Accordion header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-5 py-3.5 text-left hover:bg-card/30 transition-colors"
      >
        <span className={`text-[10px] transition-transform duration-200 text-muted-foreground/40 ${open ? "rotate-90" : ""}`}>
          ▶
        </span>

        {/* Left: block + time */}
        <div className="flex-1 min-w-0 flex flex-wrap items-center gap-x-3 gap-y-1">
          {isLatest && (
            <span className="inline-flex items-center rounded-full bg-blue-400/10 border border-blue-400/20 px-2 py-0.5 text-[10px] text-blue-400 font-mono">
              LATEST
            </span>
          )}
          <span className="text-sm font-mono font-semibold text-foreground">
            Block #{round.block?.toLocaleString() ?? "—"}
          </span>
          {typeof round.shard_idx === "number" && (
            <span
              className="text-[10px] font-mono text-muted-foreground/60 rounded border border-border/30 px-1.5 py-0.5"
              title={`Climbmix shard ${round.shard_idx} of 6542 (picked from block hash)`}
            >
              shard {round.shard_idx}
            </span>
          )}
          <span className="text-[11px] font-mono text-muted-foreground/50">
            {formatTimestamp(round.timestamp)}
          </span>
          <span className="text-[11px] font-mono text-muted-foreground/40">
            ({timeAgo(round.timestamp)})
          </span>
        </div>

        {/* Right: summary chips */}
        <div className="hidden sm:flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-mono text-muted-foreground/40">
            {nModels} models · {round.n_prompts}p
          </span>
          {king && (
            <span className="text-[11px] font-mono text-yellow-400/70">
              👑 {formatFixed(king.kl, 6)}
            </span>
          )}
          {round.king_changed && (
            <span className="inline-flex items-center rounded-full bg-emerald-400/10 border border-emerald-400/20 px-2 py-0.5 text-[10px] text-emerald-400 font-mono">
              👑 DETHRONED
            </span>
          )}
          {!round.king_changed && bestChallenger && (
            <span className="inline-flex items-center rounded-full bg-muted/30 border border-border/20 px-2 py-0.5 text-[10px] text-muted-foreground/50 font-mono">
              king holds
            </span>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {open && (
        <div className="border-t border-border/20">
          {/* King changed banner */}
          {round.king_changed && round.new_king_uid != null && (
            <div className="mx-4 mt-3 rounded-lg border border-emerald-400/30 bg-emerald-400/[0.06] px-4 py-2.5 text-sm text-emerald-400 font-mono">
              🎉 UID {round.new_king_uid} dethroned the previous king this round
            </div>
          )}
          {!round.king_changed && round.king_retained_reason && (
            <div className="mx-4 mt-3 rounded-lg border border-amber-400/30 bg-amber-400/[0.06] px-4 py-2.5 text-sm text-amber-400 font-mono">
              King retained — {round.king_retained_reason}
            </div>
          )}

          {/* Info bar */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-5 py-2.5 text-[11px] font-mono text-muted-foreground/50">
            <span>ε = {epsilonPct}% (need &gt;{epsilonPct}% better to dethrone)</span>
            <span className="hidden sm:inline">·</span>
            <span className="hidden sm:inline">
              King global: <span className="text-foreground/60">{formatFixed(kingGlobalKl, 6)}</span>
            </span>
            <span className="hidden sm:inline">·</span>
            <span className="hidden sm:inline">
              King H2H: <span className="text-foreground/60">{formatFixed(kingH2hKl, 6)}</span>
            </span>
            {epsilonThreshold != null && (
              <>
                <span className="hidden sm:inline">·</span>
                <span className="hidden sm:inline">
                  Threshold: <span className="text-orange-400/70">&lt;{formatFixed(epsilonThreshold, 6)}</span>
                </span>
              </>
            )}
          </div>

          {/* Table header */}
          <div className="hidden sm:grid grid-cols-[2rem_1fr_7rem_10rem] gap-4 px-5 py-2 text-[10px] text-muted-foreground/40 font-mono uppercase tracking-wider border-y border-border/10">
            <span>#</span>
            <span>Model</span>
            <span className="text-right" title="Head-to-head KL score — all models scored on identical prompts">H2H KL Score</span>
            <span className="text-right">vs King</span>
          </div>

          {/* Rows */}
          {results.map((result, idx) => {
            const isKing = result.is_king;
            const isDq = !isKing && Boolean(result.disqualified);
            const vsKing = result.vs_king ?? "";
            const promptCount = formatPromptCount(result.paired_prompts ?? result.prompts_scored, result.prompts_total ?? round.n_prompts);
            const needsMorePrompts = !isKing && !isDq && (result.dethrone_eligible === false || /need\s+100p/i.test(vsKing));
            const isClose = !isKing && !isDq && vsKing.includes("not enough");
            const isWorse = !isKing && !isDq && (vsKing === "worse" || (kingH2hKl != null && result.kl >= kingH2hKl));
            const isDethroner = round.king_changed && !isKing && !isDq && (
              (result.uid != null && round.new_king_uid != null && result.uid === round.new_king_uid)
              || /dethroned/i.test(vsKing)
            );

            // Parse the percentage from vs_king
            const pctMatch = vsKing.match(/-(\d+\.\d+)%/);
            const pctText = pctMatch ? `-${pctMatch[1]}%` : null;

            const tt = result.t_test;
            const tooltip = tt
              ? `n=${tt.n ?? "?"} paired prompts; KL Δ=${tt.mean_delta?.toFixed(6) ?? "?"}; t=${tt.t?.toFixed(3) ?? "?"}; p=${tt.p != null ? tt.p.toExponential(2) : "?"} (dethrone requires p<0.05 & ε>1%).`
              : undefined;
            return (
              <div
                key={result.model}
                title={tooltip}
                className={`grid grid-cols-1 sm:grid-cols-[2rem_1fr_7rem_10rem] gap-2 sm:gap-4 items-center px-5 py-2 ${
                  isKing
                    ? "bg-yellow-400/[0.04]"
                    : isDethroner
                    ? "bg-emerald-400/[0.04]"
                    : ""
                } ${idx < results.length - 1 ? "border-b border-border/[0.06]" : ""}`}
              >
                {/* Rank */}
                <span className={`text-sm font-mono tabular-nums hidden sm:block ${
                  isKing ? "text-yellow-400" : "text-muted-foreground/30"
                }`}>
                  {idx + 1}
                </span>

                {/* Model */}
                <div className="min-w-0 flex items-center gap-2">
                  <span className="sm:hidden text-xs text-muted-foreground/30 font-mono">
                    #{idx + 1}
                  </span>
                  <a
                    href={`https://huggingface.co/${result.model}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`font-mono text-[13px] truncate transition-colors ${
                      isKing
                        ? "text-yellow-400 hover:text-yellow-300"
                        : isDethroner
                        ? "text-emerald-400 hover:text-emerald-300"
                        : "text-blue-400/80 hover:text-blue-300"
                    } hover:underline`}
                  >
                    {result.model}
                  </a>
                  {isKing && (
                    <span className="text-yellow-400 text-xs shrink-0">👑</span>
                  )}
                  {isDethroner && (
                    <span className="inline-flex items-center rounded-full bg-emerald-400/10 border border-emerald-400/20 px-1.5 py-0.5 text-[9px] text-emerald-400 font-mono shrink-0">
                      NEW KING
                    </span>
                  )}
                  {!isKing && needsMorePrompts && (
                    <span className="inline-flex items-center rounded-full bg-amber-400/10 border border-amber-400/20 px-1.5 py-0.5 text-[9px] text-amber-400 font-mono shrink-0">
                      more prompts needed
                    </span>
                  )}
                  {isDq && (
                    <span
                      title={result.dq_reason ?? ""}
                      className="inline-flex items-center rounded-full bg-rose-400/10 border border-rose-400/30 px-1.5 py-0.5 text-[9px] text-rose-400 font-mono shrink-0"
                    >
                      DQ
                    </span>
                  )}
                </div>

                {/* KL Score */}
                <div className="text-right">
                  <span className={`font-mono text-[13px] font-semibold tabular-nums ${
                    isKing ? "text-yellow-400" : isDethroner ? "text-emerald-400" : "text-foreground/80"
                  }`}>
                    {formatFixed(result.kl, 6)}
                  </span>
                  {promptCount && (
                    <div className="text-[9px] text-muted-foreground/35 font-mono">
                      {promptCount}{result.early_stopped ? " early" : ""}
                    </div>
                  )}
                </div>

                {/* vs King */}
                <div className="text-right font-mono">
                  {isKing ? (
                    <span className="text-[11px] text-yellow-400/40">king</span>
                  ) : isDq ? (
                    <span
                      title={result.dq_reason ?? ""}
                      className="text-[11px] text-rose-400"
                    >
                      DQ — not crowned
                    </span>
                  ) : isDethroner ? (
                    <span className="text-[11px] text-emerald-400 font-semibold">
                      {pctText} ✓ beat ε
                    </span>
                  ) : needsMorePrompts ? (
                    <span className="text-[11px] text-amber-400/80">
                      {pctText ?? "better"} <span className="text-amber-400/50">(need 100p)</span>
                    </span>
                  ) : isClose ? (
                    <span className="text-[11px] text-amber-400/80">
                      {pctText} <span className="text-amber-400/50">(need &gt;{epsilonPct}%)</span>
                    </span>
                  ) : isWorse ? (
                    <span className="text-[11px] text-muted-foreground/30">
                      worse
                    </span>
                  ) : (
                    <span className="text-[11px] text-muted-foreground/40">
                      {vsKing || "—"}
                    </span>
                  )}
                  {!isKing && tt && (tt.p != null || tt.n != null) && (
                    <div className="text-[9px] text-muted-foreground/40 mt-0.5">
                      {tt.p != null && (
                        <span className={
                          tt.p < 0.03
                            ? "text-emerald-400/80"
                            : tt.p < 0.1
                            ? "text-amber-400/70"
                            : "text-muted-foreground/40"
                        }>p={tt.p < 1e-3 ? tt.p.toExponential(1) : tt.p.toFixed(3)}</span>
                      )}
                      {tt.t != null && <span className="ml-1">t={tt.t.toFixed(2)}</span>}
                      {tt.n != null && <span className="ml-1 text-muted-foreground/30">n={tt.n}</span>}
                    </div>
                  )}
                </div>

                {/* Composite axes (Arena v3) — shown below the main row so
                 * the dashboard reflects the worst-axis + pareto gates, not
                 * just raw KL. This is the difference between "model wins
                 * KL" and "model is SOTA across the board". */}
                {result.composite && (result.composite.axes || result.composite.pareto) && (
                  <CompositeAxesRow composite={result.composite} isKing={isKing} />
                )}
              </div>
            );
          })}

          {/* Footer */}
          <div className="px-5 py-2.5 border-t border-border/10 text-[10px] text-muted-foreground/40 font-mono">
            KL variance is normal across prompt sets — king&apos;s global score ({formatFixed(kingGlobalKl, 6)}) may differ from this round&apos;s H2H score ({formatFixed(kingH2hKl, 6)}).
          </div>
        </div>
      )}
    </div>
  );
}

export function H2hHistory() {
  const [rounds, setRounds] = useState<H2hLatestResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const refreshKey = useRefreshKey();

  useEffect(() => {
    let cancelled = false;

    async function fetchRounds() {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/h2h-history`, { cache: "no-store" });
        if (!res.ok) { if (!cancelled) setError(true); return; }
        const data: H2hHistoryResponse | H2hLatestResponse[] = await res.json();
        const fetchedRounds = Array.isArray(data)
          ? [...data].reverse()
          : Array.isArray(data.rounds)
            ? data.rounds
            : [];
        if (!cancelled) {
          setRounds(fetchedRounds);
          setError(false);
        }
      } catch {
        if (!cancelled) setError(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchRounds();
    const id = setInterval(fetchRounds, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, [refreshKey]);

  if (loading) {
    return (
      <div className="space-y-3">
        <h2 className="text-xl font-semibold tracking-tight bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
          ⚔️ Eval Rounds
        </h2>
        <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-8 text-center">
          <span className="text-sm text-muted-foreground/40 font-mono">Loading eval history…</span>
        </div>
      </div>
    );
  }

  if (error || rounds.length === 0) {
    return (
      <div className="space-y-3">
        <h2 className="text-xl font-semibold tracking-tight bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
          ⚔️ Eval Rounds
        </h2>
        <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-8 text-center">
          <p className="text-sm text-muted-foreground/60 font-mono">
            {error ? "Failed to load eval history." : "No eval rounds yet."}
          </p>
          <p className="text-xs text-muted-foreground/40 font-mono mt-2">
            {error ? "Retrying every 30s…" : "Rounds will appear here after the next evaluation completes."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Section header */}
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
          ⚔️ Eval Rounds
        </h2>
        <p className="text-[12px] font-mono text-muted-foreground/50">
          Full scoring data for every head-to-head evaluation. King must be beaten by &gt;1% (epsilon) on the same prompts.
        </p>
      </div>

      {/* Rounds list */}
      <div className="space-y-2">
        {rounds.map((round, idx) => (
          <RoundRow key={`${round.block}-${round.timestamp}`} round={round} defaultOpen={idx === 0} isLatest={idx === 0} />
        ))}
      </div>
    </div>
  );
}
