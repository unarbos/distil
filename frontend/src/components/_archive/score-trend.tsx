"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ScoreHistoryEntry } from "@/lib/api";

// Muted color palette for dark theme
const COLORS = [
  "#6366f1", // indigo
  "#22d3ee", // cyan
  "#a78bfa", // violet
  "#f472b6", // pink
  "#34d399", // emerald
  "#fb923c", // orange
  "#38bdf8", // sky
  "#e879f9", // fuchsia
];
const KING_COLOR = "#facc15"; // gold

interface ScoreTrendProps {
  history: ScoreHistoryEntry[];
}

export function ScoreTrend({ history }: ScoreTrendProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted || history.length === 0) return null;

  // Build chart data: best (lowest) KL score at each eval point
  const data = history.map((entry) => {
    const scores = Object.values(entry.scores).filter(
      (v) => typeof v === "number" && v > 0 && v < 10
    );
    const bestKl = scores.length > 0 ? Math.min(...scores) : null;
    // Find which UID had the best score
    let bestUid: string | null = null;
    if (bestKl != null) {
      for (const [uid, kl] of Object.entries(entry.scores)) {
        if (kl === bestKl) {
          bestUid = uid;
          break;
        }
      }
    }
    return {
      block: entry.block,
      blockLabel: `#${(entry.block / 1000).toFixed(0)}k`,
      bestKl,
      bestUid,
      king_uid: entry.king_uid,
    };
  });

  return (
    <div
      className="animate-fade-in rounded-xl border border-border/30 bg-card/20 backdrop-blur-sm p-4"
      style={{ animationDelay: "150ms" }}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-xs font-mono text-muted-foreground/70 uppercase tracking-wider">
          KL axis · best per round
        </h3>
        <span className="text-[10px] text-muted-foreground/40 font-mono">
          {history.length} eval{history.length !== 1 ? "s" : ""}
        </span>
      </div>
      <p className="text-[10px] text-muted-foreground/40 mb-3 leading-relaxed">
        This chart shows the KL axis only — <strong>one of 17 axes</strong> in the composite. KL is not the ranking key; <code>composite.worst</code> is. KL also varies between rounds due to different random prompts; increases don&apos;t indicate worse performance.
      </p>
      <div className="flex items-start gap-2 mb-3 px-2 py-1.5 rounded-md bg-amber-500/5 border border-amber-500/10">
        <span className="text-amber-500/70 text-[10px] mt-px">ℹ️</span>
        <p className="text-[10px] text-amber-500/60 leading-relaxed">
          <strong className="text-amber-500/80">Dataset change at block ~7864k:</strong> Switched from FineWeb (10k samples, streaming) to <strong>ClimbMix-400B</strong> (6,542 shards, ~400B tokens). The larger, more diverse dataset prevents overfitting to a small sample pool and produces more representative KL scores — which is why scores appear higher after the transition. This is expected and reflects more robust evaluation.
        </p>
      </div>
      <div className="h-[220px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
            <XAxis
              dataKey="blockLabel"
              stroke="hsl(0 0% 40%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="hsl(0 0% 40%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={[0, "auto"]}
              tickFormatter={(v: number) => v.toFixed(3)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(0 0% 8%)",
                border: "1px solid hsl(0 0% 20%)",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              labelStyle={{ color: "hsl(0 0% 60%)", fontSize: "10px" }}
              formatter={(value: unknown) => [
                typeof value === "number" ? value.toFixed(6) : String(value ?? ""),
                "KL axis (best)",
              ]}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              labelFormatter={(label: any, payload: any) => {
                const entry = payload?.[0]?.payload as { blockLabel?: string; bestUid?: string } | undefined;
                const uid = entry?.bestUid;
                return `${String(label ?? "")}${uid ? ` · UID ${uid} 👑` : ""}`;
              }}
            />
            {/* Dataset change marker — ClimbMix transition */}
            <ReferenceLine
              x="#7865k"
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
              label={{
                value: "ClimbMix 400B",
                position: "top",
                fill: "#f59e0b",
                fontSize: 9,
                opacity: 0.6,
              }}
            />
            <Line
              type="monotone"
              dataKey="bestKl"
              name="KL axis (best)"
              stroke={KING_COLOR}
              strokeWidth={2.5}
              dot={{ r: 3, fill: KING_COLOR }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
