"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import type { MinerEntry, ModelInfo } from "@/lib/api";
import { shortRevision } from "@/lib/utils";

/**
 * MinersPanelProps still receives `taoUsd` and `minersTaoDay` from the
 * SSR layer — the leaderboard table itself no longer renders the τ/day
 * or incentive columns (per user feedback 2026-04-27: those columns
 * were noise; you can read them from the wallet UI). We keep the
 * props so the SSR layer doesn't have to be rewired, and so a future
 * "show emissions" toggle can opt-in.
 */
export interface MinersPanelProps {
  miners: MinerEntry[];
  modelInfoMap: Record<string, ModelInfo>;
  taoUsd: number;
  minersTaoDay: number;
}

type FilterMode = "all" | "scored" | "queued" | "dq";

/**
 * Miners tab — dense leaderboard.
 *
 * Columns: rank, UID, model · revision, τ/day, τ stake (incentive),
 * worst, weighted, age. KL is *not* a column; it's accessible per-row
 * via the link to /miner/[uid] for the per-axis drill-down.
 *
 * Default sort: composite.worst desc (the ranking key). Falls back to
 * KL only when no v27+ composite is available for that row.
 *
 * Visual idiom: hairline borders, sticky thead, monospace numbers,
 * ♛ glyph for the king. No badges, no chips, no gradients.
 */
export function MinersPanel({
  miners,
  modelInfoMap,
}: MinersPanelProps) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"score" | "newest">("score");

  const scored = miners.filter((m) => !m.isDisqualified && (m.compositeWorst != null || m.klScore != null));
  const queued = miners.filter((m) => !m.isDisqualified && m.compositeWorst == null && m.klScore == null);
  const dqd = miners.filter((m) => m.isDisqualified);

  const filtered = useMemo(() => {
    let list = miners;
    if (filter === "scored") list = scored;
    else if (filter === "queued") list = queued;
    else if (filter === "dq") list = dqd;

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (m) =>
          m.model.toLowerCase().includes(q) ||
          m.hotkey.toLowerCase().includes(q) ||
          String(m.uid) === q
      );
    }
    if (sortBy === "newest") {
      list = [...list].sort((a, b) => b.commitBlock - a.commitBlock);
    } else {
      // score: composite.worst desc, fallback to (lower) KL
      list = [...list].sort((a, b) => {
        if (a.isDisqualified && !b.isDisqualified) return 1;
        if (!a.isDisqualified && b.isDisqualified) return -1;
        const aw = a.compositeWorst ?? -Infinity;
        const bw = b.compositeWorst ?? -Infinity;
        if (aw !== bw) return bw - aw;
        const ak = a.klScore ?? Infinity;
        const bk = b.klScore ?? Infinity;
        return ak - bk;
      });
    }
    return list;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [miners, filter, search, sortBy]);

  return (
    <div className="px-6 sm:px-9 py-8 min-h-[calc(100vh-3.5rem-3rem)] overflow-y-auto">
      <div className="flex items-baseline justify-between gap-3 flex-wrap mb-4">
        <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
          Leaderboard · ranked by composite-worst
        </h2>
        <span className="text-[11px] text-meta num">
          <span className="text-foreground font-medium">{scored.length}</span> active ·{" "}
          <span className="text-foreground font-medium">{queued.length}</span> queued ·{" "}
          <span className="text-foreground font-medium">{dqd.length}</span> dq
        </span>
      </div>

      <div className="flex items-center gap-2 flex-wrap mb-4">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="search uid, model, or hotkey…"
          className="h-7 w-56 border border-border bg-white px-2.5 text-[12px] num placeholder:text-meta focus:outline-none focus:border-foreground"
        />
        <div className="flex items-center gap-1 text-[11px]">
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
              className={[
                "h-7 px-3 border transition-colors",
                filter === key
                  ? "bg-foreground text-white border-foreground"
                  : "border-border text-meta hover:text-foreground",
              ].join(" ")}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-1 text-[10px] text-meta">
          <span>sort:</span>
          <button
            onClick={() => setSortBy("score")}
            className={sortBy === "score" ? "text-foreground" : "hover:text-foreground"}
          >
            worst-axis
          </button>
          <span>·</span>
          <button
            onClick={() => setSortBy("newest")}
            className={sortBy === "newest" ? "text-foreground" : "hover:text-foreground"}
          >
            newest
          </button>
        </div>
      </div>

      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            <Th>#</Th>
            <Th>UID</Th>
            <Th>Model · revision</Th>
            <Th align="right" title="composite.worst — the ranking key (lowest of 17 axes)">
              Worst
            </Th>
            <Th align="right" title="composite.weighted — the soft tiebreaker (Σ wᵢ·axisᵢ / Σ wᵢ)">
              Weighted
            </Th>
            <Th title="The single axis bottlenecking this miner's composite.worst. Train data here to push it up.">
              Limiting axis
            </Th>
            <Th align="right" title="KL axis only — one of 17, never the gate">
              KL <span className="opacity-60">(1/17)</span>
            </Th>
            <Th align="right">Age</Th>
          </tr>
        </thead>
        <tbody>
          {filtered.length === 0 && (
            <tr>
              <td colSpan={8} className="text-center text-meta py-8">
                no rows
              </td>
            </tr>
          )}
          {filtered.map((m, i) => {
            const info = modelInfoMap[m.model];
            const params =
              typeof info?.params_b === "number"
                ? `${info.params_b.toFixed(2)}B`
                : null;
            const isKing = m.isWinner;
            const rev = shortRevision(m.revision);
            const rank =
              m.isDisqualified
                ? null
                : isKing
                  ? "♛"
                  : sortBy === "score"
                    ? String(scored.indexOf(m) + 1).padStart(2, "0")
                    : String(i + 1).padStart(2, "0");
            return (
              <tr
                key={`${m.uid}-${m.commitBlock}`}
                className={[
                  "border-b border-border",
                  isKing
                    ? "bg-[var(--surface-soft)]"
                    : "hover:bg-[var(--surface-soft)]",
                  m.isDisqualified ? "opacity-60" : "",
                ].join(" ")}
              >
                <Td>{rank ?? "—"}</Td>
                <Td>
                  <Link
                    href={`/miner/${m.uid}`}
                    className="font-medium hover:underline"
                  >
                    #{m.uid}
                  </Link>
                </Td>
                <Td>
                  <Link
                    href={`/miner/${m.uid}`}
                    className="hover:underline"
                  >
                    <span className="text-foreground">{m.model}</span>
                    {rev && (
                      <span className="text-meta text-[11px] ml-1.5">@ {rev}</span>
                    )}
                    {params && (
                      <span className="text-meta text-[11px] ml-1.5">· {params}</span>
                    )}
                  </Link>
                  {m.dqReason && (
                    <div className="text-[10px] text-danger mt-0.5">
                      {m.dqReason.slice(0, 80)}
                    </div>
                  )}
                </Td>
                <Td align="right" className="num font-medium">
                  {m.compositeWorst != null ? m.compositeWorst.toFixed(3) : "—"}
                </Td>
                <Td align="right" className="num text-meta">
                  {m.compositeWeighted != null
                    ? m.compositeWeighted.toFixed(3)
                    : "—"}
                </Td>
                <Td className="text-[12px]">
                  {m.limitingAxis ? (
                    <span
                      className="text-foreground"
                      title={`limiting axis: ${m.limitingAxis} — train data targeting this axis to improve the ranking key`}
                    >
                      {m.limitingAxis.replace(/_bench$/, "").replace(/_/g, " ")}
                    </span>
                  ) : (
                    <span className="text-meta">—</span>
                  )}
                </Td>
                <Td align="right" className="num text-meta">
                  {m.klScore != null ? m.klScore.toFixed(4) : "—"}
                </Td>
                <Td align="right" className="num text-meta text-[11px]">
                  {/* commit_block as proxy for age */}
                  {m.commitBlock > 0 ? `#${(m.commitBlock / 1000).toFixed(0)}k` : "—"}
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="text-[10px] text-meta mt-4 leading-relaxed max-w-2xl">
        <strong className="text-foreground">Worst</strong> is the ranking key
        (lowest of the 17 axes after dropping reference-broken axes).{" "}
        <strong className="text-foreground">KL (1/17)</strong> is the
        forward-KL axis only — one of seventeen, useful for sanity-checking
        distillation but never the gate. A model that wins KL but loses on
        grade-school math, IFEval, or reasoning-density cannot take the crown.
        Click any UID for the per-axis drill-down.
      </p>
    </div>
  );
}

function Th({
  children,
  align,
  title,
}: {
  children: React.ReactNode;
  align?: "right";
  title?: string;
}) {
  return (
    <th
      title={title}
      className={[
        "text-[10px] uppercase tracking-[0.18em] text-meta font-medium px-2.5 py-2.5",
        "border-b border-border bg-white sticky top-0",
        align === "right" ? "text-right" : "text-left",
      ].join(" ")}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align,
  className,
}: {
  children: React.ReactNode;
  align?: "right";
  className?: string;
}) {
  return (
    <td
      className={[
        "px-2.5 py-2.5",
        align === "right" ? "text-right" : "",
        className ?? "",
      ].join(" ")}
    >
      {children}
    </td>
  );
}
