"use client";

import { useEffect, useMemo, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface H2hResult {
  uid?: number;
  model: string;
  kl: number;
  is_king?: boolean;
  vs_king?: string;
  disqualified?: boolean;
  composite?: {
    worst?: number | null;
    weighted?: number | null;
    axes?: Record<string, number | null | undefined>;
  } | null;
}

function findLimitingAxis(r: H2hResult | null | undefined): string | null {
  if (!r?.composite?.axes) return null;
  const entries = Object.entries(r.composite.axes).filter(
    (e): e is [string, number] => typeof e[1] === "number" && Number.isFinite(e[1]),
  );
  if (entries.length === 0) return null;
  return entries.reduce((best, cur) => (cur[1] < best[1] ? cur : best))[0];
}

function prettyAxis(s: string): string {
  return s.replace(/_bench$/, "").replace(/_/g, " ");
}

interface H2hRound {
  block: number;
  timestamp: number;
  king_uid: number;
  king_h2h_kl?: number;
  king_changed?: boolean;
  new_king_uid?: number | null;
  results: H2hResult[];
}

interface KingHistoryRound {
  block: number;
  timestamp: number;
  old_king_uid?: number | null;
  new_king_uid?: number | null;
  reign_blocks?: number | null;
}

/**
 * Rounds tab — head-to-head bouts + composite-worst trend.
 *
 * Layout: two-column. Left = bout cards stacked, one per round from
 * /api/h2h-history. Right = composite-worst trend SVG over the last
 * dozen epochs + reign-bar chart of past kings (length of each reign
 * in blocks).
 *
 * The trend is composite-worst (or its KL fallback if the round
 * predates v27 composite storage) — we deliberately don't headline KL
 * here; the entire framing is "the worst axis is the ranking key".
 */
export function RoundsPanel() {
  const [rounds, setRounds] = useState<H2hRound[]>([]);
  const [kingHistory, setKingHistory] = useState<KingHistoryRound[]>([]);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/h2h-history?limit=12`, { cache: "no-store" });
        if (res.ok && !cancel) {
          const json = await res.json();
          if (Array.isArray(json?.rounds)) setRounds(json.rounds.slice(0, 12));
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 30_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/king-history?limit=10`, { cache: "no-store" });
        if (res.ok && !cancel) {
          const json = await res.json();
          if (Array.isArray(json?.changes)) {
            setKingHistory(json.changes.slice(0, 10));
          } else if (Array.isArray(json)) {
            setKingHistory(json.slice(0, 10));
          }
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 60_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.2fr] min-h-[calc(100vh-3.5rem-3rem)]">
      <div className="px-6 sm:px-9 py-8 border-b lg:border-b-0 lg:border-r border-border overflow-y-auto">
        <HeadRow title="Recent bouts" meta={`last ${rounds.length}`} />
        <div className="mt-4 flex flex-col">
          {rounds.length === 0 && (
            <p className="text-[12px] text-meta py-6">No H2H history available yet.</p>
          )}
          {rounds.slice(0, 7).map((r, i) => (
            <BoutCard key={`${r.block}-${i}`} round={r} />
          ))}
        </div>
      </div>

      <div className="px-6 sm:px-9 py-8 bg-[var(--surface-soft)] overflow-y-auto flex flex-col gap-8">
        <div>
          <HeadRow
            title="King composite-worst · last 12 epochs"
            meta="higher better"
          />
          <CompositeTrend rounds={rounds} />
        </div>
        <div>
          <HeadRow title="Past reigns" meta="blocks" />
          <ReignChart entries={kingHistory} />
        </div>
      </div>
    </div>
  );
}

function HeadRow({ title, meta }: { title: string; meta: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 flex-wrap">
      <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
        {title}
      </h2>
      <span className="text-[11px] text-meta num">{meta}</span>
    </div>
  );
}

function BoutCard({ round }: { round: H2hRound }) {
  const [expanded, setExpanded] = useState(false);

  // Two interesting sides per round: the king and the top challenger
  // (the new king if dethroned, otherwise the closest challenger).
  const king = round.results.find((r) => r.is_king);
  const others = round.results.filter((r) => !r.is_king && !r.disqualified);
  const sortedByWorst = [...others].sort((a, b) => {
    const aw = a.composite?.worst ?? -Infinity;
    const bw = b.composite?.worst ?? -Infinity;
    return bw - aw;
  });
  const challenger = sortedByWorst[0] ?? others[0] ?? null;
  const dethroned = !!round.king_changed && challenger?.uid != null && challenger.uid === round.new_king_uid;

  const kingWorst = king?.composite?.worst ?? null;
  const challengerWorst = challenger?.composite?.worst ?? null;
  const margin =
    kingWorst != null && challengerWorst != null
      ? ((challengerWorst - kingWorst) / Math.max(kingWorst, 0.001)) * 100
      : null;

  return (
    <div
      className={[
        "py-3.5 border-b border-border",
        dethroned ? "border-l-2 border-l-foreground pl-3 -ml-3" : "",
      ].join(" ")}
    >
      <button
        className="grid grid-cols-[1fr_28px_1fr_18px] gap-2.5 items-center w-full text-left hover:bg-white/40 -mx-2 px-2 py-1 transition-colors"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <Side
          side="left"
          loser={dethroned}
          winner={!dethroned}
          result={king}
          showAnnotation={!dethroned}
          annotation="defends"
          margin={null}
        />
        <span className="serif text-meta text-[13px] text-center">vs</span>
        <Side
          side="right"
          loser={!dethroned}
          winner={dethroned}
          result={challenger}
          showAnnotation={dethroned}
          annotation={dethroned ? "dethrone" : "challenger"}
          margin={dethroned ? margin : null}
        />
        <span
          className="text-meta text-[12px] num"
          aria-hidden
          title={expanded ? "collapse" : "expand round detail"}
        >
          {expanded ? "−" : "+"}
        </span>
      </button>

      {expanded && (
        <div className="mt-3 -mx-1 px-2 py-3 border-t border-border bg-[var(--surface-soft)]/60">
          <div className="text-[10px] uppercase tracking-[0.18em] text-meta mb-2">
            Round detail · block #{round.block} ·{" "}
            {round.results.filter((r) => !r.disqualified).length} live ·{" "}
            {round.results.filter((r) => r.disqualified).length} dq
          </div>
          <RoundAxisGrid round={round} />
        </div>
      )}
    </div>
  );
}

/**
 * Per-round per-axis grid. Rows = miners (king first, then by worst
 * desc, then DQ at bottom). Columns = the headline 12 axes (we don't
 * show all 17 here — too wide; the full 17 lives on /miner/[uid] and
 * the Axes tab). Cell = score, colour-coded.
 */
const ROUND_AXIS_COLS: { key: string; label: string }[] = [
  { key: "kl", label: "kl" },
  { key: "on_policy_rkl", label: "rkl" },
  { key: "capability", label: "cap" },
  { key: "math_bench", label: "math" },
  { key: "code_bench", label: "code" },
  { key: "reasoning_bench", label: "reas" },
  { key: "ifeval_bench", label: "ifev" },
  { key: "aime_bench", label: "aime" },
  { key: "judge_probe", label: "judg" },
  { key: "chat_turns_probe", label: "chat" },
  { key: "length", label: "len" },
  { key: "degeneracy", label: "deg" },
];

function RoundAxisGrid({ round }: { round: H2hRound }) {
  const sorted = [...round.results].sort((a, b) => {
    if (a.is_king && !b.is_king) return -1;
    if (!a.is_king && b.is_king) return 1;
    if (a.disqualified && !b.disqualified) return 1;
    if (!a.disqualified && b.disqualified) return -1;
    const aw = a.composite?.worst ?? -Infinity;
    const bw = b.composite?.worst ?? -Infinity;
    return bw - aw;
  });
  return (
    <div className="overflow-x-auto -mx-2">
      <table className="text-[11px] num border-collapse w-full">
        <thead>
          <tr>
            <th className="text-left text-meta font-medium px-2 py-1.5 sticky left-0 bg-[var(--surface-soft)]/80">
              UID
            </th>
            <th className="text-right text-meta font-medium px-2 py-1.5">worst</th>
            <th className="text-right text-meta font-medium px-2 py-1.5">wgt</th>
            {ROUND_AXIS_COLS.map((c) => (
              <th
                key={c.key}
                className="text-right text-meta font-medium px-2 py-1.5"
                title={c.key}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const ax = r.composite?.axes ?? {};
            return (
              <tr
                key={`${r.uid}-${r.model}`}
                className={[
                  "border-t border-border",
                  r.disqualified ? "opacity-50" : "",
                  r.is_king ? "bg-white" : "",
                ].join(" ")}
              >
                <td className="text-left px-2 py-1.5 sticky left-0 bg-inherit whitespace-nowrap">
                  {r.is_king && "♛ "}
                  {r.uid != null ? `#${r.uid}` : "—"}
                  {r.disqualified && (
                    <span className="text-danger ml-1.5 text-[9px] uppercase">dq</span>
                  )}
                </td>
                <td className="text-right px-2 py-1.5 font-medium">
                  {r.composite?.worst != null ? r.composite.worst.toFixed(3) : "—"}
                </td>
                <td className="text-right px-2 py-1.5 text-meta">
                  {r.composite?.weighted != null ? r.composite.weighted.toFixed(3) : "—"}
                </td>
                {ROUND_AXIS_COLS.map((c) => {
                  const v = ax[c.key];
                  if (typeof v !== "number" || !Number.isFinite(v)) {
                    return (
                      <td key={c.key} className="text-right px-2 py-1.5 text-meta">
                        —
                      </td>
                    );
                  }
                  // Heat: <0.3 red, 0.3-0.6 amber, 0.6-0.85 fg, ≥0.85 strong
                  const color =
                    v < 0.3
                      ? "text-danger"
                      : v < 0.6
                        ? "text-warning"
                        : v < 0.85
                          ? "text-foreground"
                          : "text-foreground font-medium";
                  return (
                    <td
                      key={c.key}
                      className={["text-right px-2 py-1.5", color].join(" ")}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

interface SideProps {
  side: "left" | "right";
  winner: boolean;
  loser: boolean;
  result: H2hResult | null | undefined;
  showAnnotation: boolean;
  annotation: string;
  margin: number | null;
}

function Side({ side, loser, result, showAnnotation, annotation, margin }: SideProps) {
  if (!result) {
    return <div className={side === "right" ? "text-right" : ""}>—</div>;
  }
  const worst = result.composite?.worst ?? null;
  const limiting = findLimitingAxis(result);
  const align = side === "right" ? "items-end text-right" : "items-start";
  return (
    <div className={["flex flex-col gap-0.5 min-w-0", align].join(" ")}>
      <div
        className={[
          "text-[13px] font-medium truncate max-w-full",
          loser ? "text-meta line-through" : "",
        ].join(" ")}
      >
        uid {result.uid ?? "—"}
      </div>
      <div className="text-[10px] text-meta num tracking-wider">
        {worst != null
          ? `worst ${worst.toFixed(3)}`
          : `KL ${result.kl.toFixed(4)}`}
      </div>
      {limiting && worst != null && (
        <div
          className="text-[9px] text-meta tracking-[0.05em] truncate max-w-full"
          title={`limiting axis (lowest non-broken): ${limiting}`}
        >
          ↳ {prettyAxis(limiting)}
        </div>
      )}
      {showAnnotation && (
        <div className="text-[9px] text-meta uppercase tracking-[0.12em] num mt-0.5">
          <strong className="text-foreground font-medium">{annotation}</strong>
          {margin != null && (
            <>
              {" · "}
              <strong className="text-foreground font-medium">
                {margin >= 0 ? "+" : ""}
                {margin.toFixed(1)}%
              </strong>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function CompositeTrend({ rounds }: { rounds: H2hRound[] }) {
  const points = useMemo(() => {
    if (rounds.length === 0) return [];
    const pts: { block: number; worst: number; kingUid: number }[] = [];
    // h2h-history is newest-first; the trend should read left-to-right.
    for (const r of [...rounds].reverse()) {
      const king = r.results.find((x) => x.is_king);
      const worst = king?.composite?.worst;
      if (typeof worst === "number" && worst >= 0) {
        pts.push({ block: r.block, worst, kingUid: r.king_uid });
      }
    }
    return pts;
  }, [rounds]);

  if (points.length < 2) {
    return (
      <p className="text-[11px] text-meta mt-3">
        Composite-worst trend will populate as more rounds commit (need ≥2
        v27+ records).
      </p>
    );
  }

  const W = 800;
  const H = 240;
  const P = 24;
  const min = Math.min(0, ...points.map((p) => p.worst));
  const max = Math.max(1, ...points.map((p) => p.worst));
  const dx = (W - P * 2) / (points.length - 1);
  const yof = (v: number) => P + ((max - v) / (max - min)) * (H - P * 2);

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${P + i * dx},${yof(p.worst)}`)
    .join(" ");
  const areaPath = `${linePath} L ${P + (points.length - 1) * dx} ${H - P} L ${P} ${H - P} Z`;
  const last = points[points.length - 1];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="w-full h-auto mt-3"
    >
      {/* Gridlines */}
      {[0, 1, 2, 3].map((i) => {
        const y = P + (i * (H - P * 2)) / 3;
        return (
          <line key={i} x1={P} x2={W - P} y1={y} y2={y} stroke="#f1f1f1" />
        );
      })}
      <path d={areaPath} fill="#f7f7f7" />
      <path d={linePath} stroke="#0a0a0a" strokeWidth={1.5} fill="none" />
      {points.map((p, i) => {
        const isLast = i === points.length - 1;
        return (
          <rect
            key={i}
            x={P + i * dx - 2.5}
            y={yof(p.worst) - 2.5}
            width={5}
            height={5}
            fill={isLast ? "#0a0a0a" : "#bbb"}
          />
        );
      })}
      <text
        x={W - P - 8}
        y={yof(last.worst) - 12}
        fontFamily="Inter, sans-serif"
        fontSize={11}
        textAnchor="end"
        fill="#0a0a0a"
        fontWeight={500}
      >
        worst {last.worst.toFixed(3)}
      </text>
    </svg>
  );
}

function ReignChart({ entries }: { entries: KingHistoryRound[] }) {
  const reigns = useMemo(() => {
    return entries
      .map((e, idx) => {
        const blocks =
          typeof e.reign_blocks === "number" && e.reign_blocks > 0
            ? e.reign_blocks
            : null;
        return {
          uid: e.old_king_uid ?? null,
          blocks,
          isCurrent: idx === entries.length - 1 && e.new_king_uid != null,
        };
      })
      .filter((r) => r.uid != null && r.blocks != null) as {
        uid: number;
        blocks: number;
        isCurrent: boolean;
      }[];
  }, [entries]);

  if (reigns.length === 0) {
    return (
      <p className="text-[11px] text-meta mt-3">
        Past reigns will populate after a king flip lands a reign_blocks value.
      </p>
    );
  }

  const W = 800;
  const H = 200;
  const P = 24;
  const max = Math.max(...reigns.map((r) => r.blocks));
  const slot = (W - P * 2) / reigns.length;
  const bw = slot - 8;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="w-full h-auto mt-3"
    >
      {reigns.map((r, i) => {
        const h = (r.blocks / max) * (H - P * 2 - 22);
        const x = P + i * slot + 4;
        const y = H - P - h;
        return (
          <g key={`${r.uid}-${i}`}>
            <rect
              x={x}
              y={y}
              width={bw}
              height={h}
              fill={r.isCurrent ? "#0a0a0a" : "#cfcfcf"}
            />
            <text
              x={x + bw / 2}
              y={H - 8}
              fontFamily="Inter, sans-serif"
              fontSize={10}
              textAnchor="middle"
              fill="#8a8a8a"
            >
              {r.uid}
            </text>
            <text
              x={x + bw / 2}
              y={y - 6}
              fontFamily="Inter, sans-serif"
              fontSize={10}
              textAnchor="middle"
              fill="#0a0a0a"
            >
              {r.blocks}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
