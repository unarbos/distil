"use client";

import { useEffect, useState } from "react";
import type { H2hLatestResponse } from "@/lib/api";
import { useRefreshKey } from "@/components/auto-refresh";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.arbos.life";

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

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

function formatFixed(value: number | null | undefined, digits: number, fallback = "—"): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

function RoundRow({ round, defaultOpen, isLatest }: { round: H2hLatestResponse; defaultOpen: boolean; isLatest: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const epsilon = typeof round.epsilon === "number" && Number.isFinite(round.epsilon) ? round.epsilon : 0.01;
  const epsilonPct = formatFixed(epsilon * 100, 0, "1");
  const king = Array.isArray(round.results) ? round.results.find((r) => r.is_king) : undefined;
  const bestChallenger = Array.isArray(round.results) ? round.results.find((r) => !r.is_king) : undefined;
  const kingH2hKl = typeof round.king_h2h_kl === "number" && Number.isFinite(round.king_h2h_kl)
    ? round.king_h2h_kl
    : (typeof king?.kl === "number" && Number.isFinite(king.kl) ? king.kl : null);
  const kingGlobalKl = typeof round.king_global_kl === "number" && Number.isFinite(round.king_global_kl)
    ? round.king_global_kl
    : null;
  const epsilonThreshold = typeof round.epsilon_threshold === "number" && Number.isFinite(round.epsilon_threshold)
    ? round.epsilon_threshold
    : (kingH2hKl != null ? kingH2hKl * (1 - epsilon) : null);
  const nModels = Array.isArray(round.results) ? round.results.length : 0;

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
          {round.results.map((result, idx) => {
            const isKing = result.is_king;
            const isClose = !isKing && result.vs_king.includes("not enough");
            const isWorse = !isKing && (result.vs_king === "worse" || (kingH2hKl != null && result.kl >= kingH2hKl));
            const isDethroner = round.king_changed && !isKing && epsilonThreshold != null && result.kl < epsilonThreshold;

            // Parse the percentage from vs_king
            const pctMatch = result.vs_king.match(/-(\d+\.\d+)%/);
            const pctText = pctMatch ? `-${pctMatch[1]}%` : null;

            return (
              <div
                key={result.model}
                className={`grid grid-cols-1 sm:grid-cols-[2rem_1fr_7rem_10rem] gap-2 sm:gap-4 items-center px-5 py-2 ${
                  isKing
                    ? "bg-yellow-400/[0.04]"
                    : isDethroner
                    ? "bg-emerald-400/[0.04]"
                    : ""
                } ${idx < round.results.length - 1 ? "border-b border-border/[0.06]" : ""}`}
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
                </div>

                {/* KL Score */}
                <span className={`font-mono text-[13px] font-semibold tabular-nums text-right ${
                  isKing ? "text-yellow-400" : isDethroner ? "text-emerald-400" : "text-foreground/80"
                }`}>
                  {formatFixed(result.kl, 6)}
                </span>

                {/* vs King */}
                <div className="text-right font-mono">
                  {isKing ? (
                    <span className="text-[11px] text-yellow-400/40">king</span>
                  ) : isDethroner ? (
                    <span className="text-[11px] text-emerald-400 font-semibold">
                      {pctText} ✓ beat ε
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
                      {result.vs_king}
                    </span>
                  )}
                </div>
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
        const res = await fetch(`${API_BASE}/api/h2h-history`, { cache: "no-store" });
        if (!res.ok) { if (!cancelled) setError(true); return; }
        const data: H2hLatestResponse[] = await res.json();
        if (!cancelled) {
          // Show newest first
          setRounds([...data].reverse());
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

  if (error || rounds.length === 0) return null;

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
