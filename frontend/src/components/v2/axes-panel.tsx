"use client";

import { useEffect, useMemo, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface H2hResult {
  uid?: number;
  model: string;
  kl: number;
  is_king?: boolean;
  disqualified?: boolean;
  composite?: {
    worst?: number | null;
    weighted?: number | null;
    axes?: Record<string, number | null | undefined>;
  } | null;
}

interface H2hLatest {
  results: H2hResult[];
  king_uid?: number;
}

/**
 * Axes tab — Pareto plot + composite breakdown bars.
 *
 * Pareto: worst-axis (x) vs weighted-mean (y). Both are composite
 * outputs; this is the honest 2-D view of "how good is this miner
 * across the 17 axes". A model in the upper-right is good on its
 * weakest axis AND good on average. The Pareto frontier traces the
 * staircase of non-dominated points.
 *
 * Composite list: per-UID stacked bar showing worst (foreground) +
 * weighted (background) so you can read "this miner has a 0.5 worst
 * but a 0.85 weighted — they're being held back by one specific
 * axis".
 *
 * Data: /api/h2h-latest. Already exposes composite.axes per row.
 */
export function AxesPanel() {
  const [latest, setLatest] = useState<H2hLatest | null>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/h2h-latest`, { cache: "no-store" });
        if (res.ok && !cancel) {
          const json = await res.json();
          setLatest(json);
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

  const points = useMemo(() => {
    if (!latest) return [];
    const pts: { uid: number; worst: number; weighted: number; isKing: boolean }[] = [];
    for (const r of latest.results) {
      if (r.disqualified) continue;
      if (r.uid == null) continue;
      const w = r.composite?.worst;
      const wm = r.composite?.weighted;
      if (typeof w !== "number" || typeof wm !== "number") continue;
      pts.push({
        uid: r.uid,
        worst: w,
        weighted: wm,
        isKing: !!r.is_king,
      });
    }
    return pts;
  }, [latest]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      <div className="px-6 sm:px-9 py-8 border-b lg:border-b-0 lg:border-r border-border overflow-y-auto">
        <HeadRow
          title="Pareto · worst × weighted"
          meta={`${points.length} active`}
        />
        <ParetoChart points={points} />
        <p className="text-[11px] text-meta mt-4 leading-relaxed">
          <strong className="text-foreground">x</strong> = composite.worst (the
          ranking key — your weakest axis after dropping reference-broken
          axes).{" "}
          <strong className="text-foreground">y</strong> = composite.weighted
          (Σ wᵢ · axisᵢ / Σ wᵢ across all 17). Upper-right is the goal:
          strong on every axis. The dashed line traces the Pareto frontier.
        </p>
      </div>

      <div className="px-6 sm:px-9 py-8 bg-[var(--surface-soft)] overflow-y-auto">
        <HeadRow title="Composite breakdown" meta="worst / weighted per UID" />
        <CompositeBreakdown results={latest?.results ?? []} kingUid={latest?.king_uid} />
        <p className="text-[10px] text-meta mt-3 leading-relaxed">
          Black bar = composite.worst (ranking key). Grey bar = composite.weighted
          (mean over surviving axes). KL is one of 17 axes; expand the row
          (link in /miner/[uid]) for the per-axis breakdown.
        </p>
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

function ParetoChart({
  points,
}: {
  points: { uid: number; worst: number; weighted: number; isKing: boolean }[];
}) {
  if (points.length === 0) {
    return (
      <p className="text-[11px] text-meta mt-6">
        Pareto chart will populate once ≥1 v27+ composite record lands.
      </p>
    );
  }
  const W = 800;
  const H = 480;
  const P = 44;

  // Both axes are 0..1 (composite is normalised per axis).
  const xof = (k: number) => P + Math.min(1, Math.max(0, k)) * (W - P * 2);
  const yof = (k: number) =>
    H - P - Math.min(1, Math.max(0, k)) * (H - P * 2);

  // Pareto frontier: scan in order of increasing worst, keep highest weighted.
  const sortedByWorst = [...points].sort((a, b) => a.worst - b.worst);
  let bestWeighted = -Infinity;
  const front: typeof points = [];
  // We want the upper-right frontier — iterate from the *highest worst*
  // downward and keep monotonically rising weighted; that gives us the
  // "no other point dominates" set.
  for (const p of [...sortedByWorst].reverse()) {
    if (p.weighted > bestWeighted) {
      front.unshift(p);
      bestWeighted = p.weighted;
    }
  }

  const frontPath = front
    .map((p, i) => `${i === 0 ? "M" : "L"}${xof(p.worst)},${yof(p.weighted)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="w-full h-auto mt-4"
    >
      <rect
        x={P}
        y={P}
        width={W - P * 2}
        height={H - P * 2}
        fill="#fff"
        stroke="#ebebeb"
      />
      {/* Gridlines at quartiles */}
      {[1, 2, 3].map((i) => {
        const x = P + (i * (W - P * 2)) / 4;
        return <line key={`vx${i}`} x1={x} x2={x} y1={P} y2={H - P} stroke="#f4f4f4" />;
      })}
      {[1, 2, 3].map((i) => {
        const y = P + (i * (H - P * 2)) / 4;
        return <line key={`hy${i}`} x1={P} x2={W - P} y1={y} y2={y} stroke="#f4f4f4" />;
      })}

      {front.length > 1 && (
        <path
          d={frontPath}
          stroke="#0a0a0a"
          strokeWidth={1}
          fill="none"
          strokeDasharray="4 3"
        />
      )}

      {points.map((p) => {
        const x = xof(p.worst);
        const y = yof(p.weighted);
        return (
          <g key={p.uid}>
            <rect
              x={x - 4}
              y={y - 4}
              width={8}
              height={8}
              fill={p.isKing ? "#0a0a0a" : "#bdbdbd"}
            />
            <text
              x={x + 10}
              y={y + 4}
              fontFamily="Inter, sans-serif"
              fontSize={11}
              fill={p.isKing ? "#0a0a0a" : "#8a8a8a"}
              fontWeight={p.isKing ? 500 : 400}
            >
              {p.uid}
            </text>
          </g>
        );
      })}

      <text
        x={P}
        y={H - 14}
        fontFamily="Inter, sans-serif"
        fontSize={10}
        fill="#8a8a8a"
        letterSpacing={2}
      >
        WORST → HIGHER
      </text>
      <text
        x={P - 8}
        y={P - 12}
        fontFamily="Inter, sans-serif"
        fontSize={10}
        fill="#8a8a8a"
        letterSpacing={2}
      >
        ↑ WEIGHTED
      </text>
    </svg>
  );
}

function CompositeBreakdown({
  results,
  kingUid,
}: {
  results: H2hResult[];
  kingUid?: number;
}) {
  // Sort by worst desc. Skip DQs.
  const sorted = useMemo(() => {
    return [...results]
      .filter((r) => !r.disqualified && r.composite)
      .sort((a, b) => {
        const aw = a.composite?.worst ?? -Infinity;
        const bw = b.composite?.worst ?? -Infinity;
        return bw - aw;
      });
  }, [results]);

  if (sorted.length === 0) {
    return (
      <p className="text-[11px] text-meta mt-3">
        No composite breakdown available for the latest round yet.
      </p>
    );
  }

  return (
    <div className="mt-4 flex flex-col">
      {sorted.map((r) => {
        const worst = r.composite?.worst ?? 0;
        const weighted = r.composite?.weighted ?? 0;
        const isKing = r.uid != null && r.uid === kingUid;
        return (
          <div
            key={`${r.uid}-${r.model}`}
            className="grid grid-cols-[54px_1fr_64px_64px] items-center gap-3.5 text-[13px] py-2.5 border-b border-border last:border-b-0"
          >
            <span className="font-medium num">
              #{r.uid ?? "—"}
              {isKing && <span className="text-meta text-[10px] ml-1">♛</span>}
            </span>
            <div className="relative h-1.5 bg-[#f1f1f1]">
              <span
                className="absolute inset-y-0 left-0 bg-[#bdbdbd]"
                style={{ width: `${Math.max(0, Math.min(1, weighted)) * 100}%` }}
              />
              <span
                className="absolute inset-y-0 left-0 bg-foreground"
                style={{ width: `${Math.max(0, Math.min(1, worst)) * 100}%` }}
              />
            </div>
            <span className="text-right num font-medium">{worst.toFixed(3)}</span>
            <span className="text-right num text-meta">{weighted.toFixed(3)}</span>
          </div>
        );
      })}
      <div className="grid grid-cols-[54px_1fr_64px_64px] gap-3.5 text-[10px] uppercase tracking-[0.16em] text-meta py-2 border-t border-border mt-1">
        <span></span>
        <span></span>
        <span className="text-right">Worst</span>
        <span className="text-right">Weighted</span>
      </div>
    </div>
  );
}
