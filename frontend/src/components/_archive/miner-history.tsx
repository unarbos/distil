"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface HistoryPoint {
  block: number;
  blockLabel: string;
  score: number;
}

interface Props {
  uid: number;
  history: HistoryPoint[];
}

export function MinerHistory({ uid, history }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted || history.length < 2) return null;

  return (
    <div className="h-[160px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={history}
          margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
        >
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
            domain={["auto", "auto"]}
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
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(value: any) => [typeof value === "number" ? value.toFixed(4) : String(value ?? ""), `UID ${uid}`]}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#6366f1"
            strokeWidth={2}
            dot={{ r: 3, fill: "#6366f1" }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
