"use client";

import { useState } from "react";
import Link from "next/link";
import type { MinerEntry, ModelInfo, ScoresResponse } from "@/lib/api";
import { TEACHER } from "@/lib/api";
import { formatParams } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { CopyableHotkey } from "@/components/copyable-hotkey";

interface MinersTabProps {
  miners: MinerEntry[];
  scores: ScoresResponse | null;
  modelInfoMap: Record<string, ModelInfo>;
  currentBlock: number;
  taoUsd: number;
  minersTaoDay: number;
}

type FilterMode = "all" | "scored" | "queued" | "dq";

function timeAgo(block: number, currentBlock: number): string {
  const blockDiff = currentBlock - block;
  const hoursAgo = (blockDiff * 12) / 3600;
  if (hoursAgo < 1) return `${Math.round(hoursAgo * 60)}m ago`;
  if (hoursAgo < 24) return `${Math.round(hoursAgo)}h ago`;
  return `${Math.round(hoursAgo / 24)}d ago`;
}

function formatDqReason(reason: string | null): string {
  if (!reason) return "DQ";
  if (reason.includes("429") || reason.includes("rate limit")) return "HF rate limit — will retry";
  if (reason.startsWith("copy:")) return reason.replace("copy: ", "Copy: ");
  if (reason.startsWith("arch:")) return reason.replace("arch: ", "");
  if (reason.startsWith("integrity:")) return reason.replace("integrity: ", "");
  if (reason.startsWith("check_failed:")) return reason.replace("check_failed:", "Check: ");
  if (reason.length > 80) return reason.slice(0, 77) + "...";
  return reason;
}

export function MinersTab({
  miners,
  scores,
  modelInfoMap,
  currentBlock,
  taoUsd,
  minersTaoDay,
}: MinersTabProps) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sortBy, setSortBy] = useState<"score" | "newest">("score");
  const [search, setSearch] = useState("");

  const totalPrompts = scores?.last_eval?.n_prompts ?? 0;

  const scored = miners.filter(m => !m.isDisqualified && m.klScore != null);
  const queued = miners.filter(m => !m.isDisqualified && m.klScore == null);
  const dqd = miners.filter(m => m.isDisqualified);

  let filtered = miners;
  if (filter === "scored") filtered = scored;
  else if (filter === "queued") filtered = queued;
  else if (filter === "dq") filtered = dqd;

  // Search
  if (search.trim()) {
    const q = search.trim().toLowerCase();
    filtered = filtered.filter(m =>
      m.model.toLowerCase().includes(q) ||
      m.hotkey.toLowerCase().includes(q) ||
      String(m.uid) === q
    );
  }

  // Sort
  if (sortBy === "newest") {
    filtered = [...filtered].sort((a, b) => b.commitBlock - a.commitBlock);
  }
  // "score" sort is the default from buildMinerList (DQ last, then by KL)

  return (
    <div className="space-y-3">
      {/* Search + Filter bar */}
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search UID, model, or hotkey..."
          className="h-7 w-48 rounded-md border border-border/30 bg-card/20 px-2.5 text-[11px] font-mono text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-blue-400/40"
        />
        <div className="flex items-center gap-1 text-[11px] font-mono">
          {(
            [
              ["all", `All (${miners.length})`],
              ["scored", `Scored (${scored.length})`],
              ["queued", `Queued (${queued.length})`],
              ["dq", `DQ (${dqd.length})`],
            ] as [FilterMode, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                filter === key
                  ? "bg-blue-400/15 text-blue-300 border border-blue-400/30"
                  : "text-muted-foreground/50 hover:text-muted-foreground/80 border border-transparent"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-1 text-[10px] font-mono text-muted-foreground/40">
          <button
            onClick={() => setSortBy("score")}
            className={sortBy === "score" ? "text-foreground/60" : "hover:text-muted-foreground/60"}
          >
            by score
          </button>
          <span>·</span>
          <button
            onClick={() => setSortBy("newest")}
            className={sortBy === "newest" ? "text-foreground/60" : "hover:text-muted-foreground/60"}
          >
            newest
          </button>
        </div>
      </div>

      {/* Score type explanation */}
      <div className="rounded-lg border border-amber-400/15 bg-amber-400/[0.03] px-3 py-2 space-y-1">
        <div className="flex items-start gap-2">
          <span className="text-amber-400/60 text-[11px] mt-px">ℹ️</span>
          <div className="text-[10px] font-mono text-muted-foreground/50 space-y-1">
            <p>
              <span className="text-yellow-400/80 font-semibold">H2H Score</span>
              {" "}— Head-to-head KL on the <em>same prompts</em> as the king. This is the fair comparison used to determine the king.
            </p>
            <p>
              <span className="text-blue-400/80 font-semibold">Global Score</span>
              {" "}— KL from the last global eval using <em>different random prompts</em>. These vary between rounds and are <strong>not directly comparable</strong> across models.
            </p>
          </div>
        </div>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <div className="rounded-xl border border-border/20 bg-card/10 p-8 text-center">
          <p className="text-sm text-muted-foreground/50 font-mono">
            {filter === "queued" ? "No models queued" : filter === "dq" ? "No disqualified models" : "No submissions yet"}
          </p>
          {miners.length === 0 && (
            <div className="mt-3 space-y-2">
              <p className="text-sm text-muted-foreground max-w-lg mx-auto">
                Distill <span className="font-mono text-blue-400">{TEACHER.model}</span> into
                ≤{formatParams(5_250_000_000)} params and earn emissions.
              </p>
              <Link
                href="/about"
                className="inline-block rounded-lg bg-blue-500/10 border border-blue-500/20 px-4 py-2 text-sm text-blue-400 hover:bg-blue-500/20 transition-colors"
              >
                How to mine →
              </Link>
            </div>
          )}
        </div>
      )}

      {/* Miner rows */}
      <div className="space-y-1.5 max-h-[60vh] overflow-y-auto pr-1">
        {filtered.map((miner, idx) => {
          const dailyTao = minersTaoDay * miner.incentive;
          const dailyUsd = dailyTao * taoUsd;

          const studentData = scores?.last_eval?.students?.[miner.model];
          const tokensPerSec = studentData?.tokens_per_sec ?? null;
          const minerPromptCount = studentData?.kl_per_prompt?.length ?? 0;
          const minerEarlyStopped = minerPromptCount > 0 && minerPromptCount < totalPrompts;

          const mInfo = modelInfoMap[miner.model];
          const dqCategory = miner.dqReason?.startsWith("copy:")
            ? "COPY"
            : miner.dqReason?.startsWith("integrity:")
            ? "REMOVED"
            : miner.dqReason?.startsWith("arch:")
            ? "INVALID"
            : miner.isDisqualified
            ? "DQ"
            : null;

          const rank = sortBy === "score" && !miner.isDisqualified && miner.klScore != null
            ? scored.indexOf(miner) + 1
            : null;

          return (
            <Link
              key={`${miner.uid}-${miner.commitBlock}`}
              href={`/miner/${miner.uid}`}
              className="block group"
            >
              <div
                className={`rounded-xl border px-4 py-3 transition-all duration-200 ${
                  miner.isDisqualified
                    ? "border-red-400/15 bg-red-400/[0.02] opacity-60 hover:opacity-80"
                    : miner.isWinner
                    ? "border-yellow-400/25 bg-yellow-400/[0.03] hover:bg-yellow-400/[0.06]"
                    : miner.klScore == null
                    ? "border-cyan-400/15 bg-cyan-400/[0.02] hover:bg-cyan-400/[0.04]"
                    : "border-border/25 bg-card/15 hover:bg-card/30"
                }`}
              >
                <div className="flex items-center gap-3">
                  {/* Rank / Status */}
                  <div className="w-8 shrink-0 text-center">
                    {miner.isDisqualified ? (
                      <span className="text-red-400/40 text-sm">✗</span>
                    ) : miner.klScore == null ? (
                      <span className="text-cyan-400/50 text-sm">⏳</span>
                    ) : rank ? (
                      <span className={`text-lg font-bold tabular-nums ${
                        miner.isWinner ? "text-yellow-400" : "text-muted-foreground/30"
                      }`}>
                        {rank}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/20 text-sm">·</span>
                    )}
                  </div>

                  {/* Model + metadata */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`font-mono text-sm truncate ${
                        miner.isDisqualified
                          ? "text-muted-foreground/40 line-through"
                          : "text-blue-400/90"
                      }`}>
                        {miner.model}
                      </span>
                      <span className="text-[10px] text-muted-foreground/40 font-mono">
                        UID {miner.uid}
                      </span>
                      {miner.isWinner && (
                        <Badge className="bg-yellow-400/10 text-yellow-400 border-yellow-400/20 text-[9px] px-1 font-normal">
                          👑 KING
                        </Badge>
                      )}
                      {dqCategory && (
                        <Badge className="bg-red-400/10 text-red-400 border-red-400/20 text-[9px] px-1 font-normal">
                          {dqCategory}
                        </Badge>
                      )}
                      {miner.klScore == null && !miner.isDisqualified && (
                        <Badge className="bg-cyan-400/10 text-cyan-400 border-cyan-400/20 text-[9px] px-1 font-normal">
                          QUEUED
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 text-[10px] text-muted-foreground/50">
                      <CopyableHotkey hotkey={miner.hotkey} chars={5} />
                      {mInfo?.params_b && (
                        <span>
                          {mInfo.is_moe ? `${mInfo.active_params_b}B active` : `${mInfo.params_b}B`}
                        </span>
                      )}
                      {tokensPerSec != null && (
                        <span className="text-emerald-400/60">⚡{Math.round(tokensPerSec)}</span>
                      )}
                      <span>{timeAgo(miner.commitBlock, currentBlock)}</span>
                      <span className="text-muted-foreground/30">#{miner.commitBlock.toLocaleString()}</span>
                    </div>
                    {miner.isDisqualified && miner.dqReason && (
                      <div className="text-[9px] text-red-400/50 mt-0.5 truncate max-w-md" title={miner.dqReason}>
                        {formatDqReason(miner.dqReason)}
                      </div>
                    )}
                  </div>

                  {/* Score / Earnings */}
                  <div className="shrink-0 text-right">
                    {miner.isDisqualified ? (
                      <span className="font-mono text-sm text-red-400/30">—</span>
                    ) : miner.klScore != null ? (
                      <div>
                        <span className={`font-mono text-base font-bold tabular-nums ${
                          miner.isWinner ? "text-yellow-400" : "text-foreground/80"
                        }`}>
                          {miner.klScore.toFixed(4)}
                        </span>
                        <div className="text-[9px] text-blue-400/50 font-mono">
                          Global
                        </div>
                        {miner.ci95 && (
                          <div className="text-[9px] text-muted-foreground/40 font-mono">
                            [{Number.isFinite(miner.ci95[0]) ? miner.ci95[0].toFixed(3) : "?"},{Number.isFinite(miner.ci95[1]) ? miner.ci95[1].toFixed(3) : "?"}]
                          </div>
                        )}
                        {minerPromptCount > 0 && (
                          <div className="text-[9px] text-muted-foreground/30 font-mono">
                            {minerPromptCount}/{totalPrompts}p
                            {minerEarlyStopped && <span className="text-yellow-400/50"> early</span>}
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-[11px] text-cyan-400/50 italic font-mono">queued</span>
                    )}
                    {dailyUsd > 0.01 && !miner.isDisqualified && (
                      <div className="text-[10px] text-emerald-400/70 font-mono mt-0.5">
                        ${dailyUsd > 1000 ? `${(dailyUsd/1000).toFixed(1)}k` : dailyUsd.toFixed(0)}/d
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
