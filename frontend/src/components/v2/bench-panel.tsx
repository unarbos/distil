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
}

interface BenchmarksResponse {
  models: BenchmarkPayload[];
  baseline: BenchmarkPayload | null;
}

// Six benches we headline. Names match what evalscope uses.
const HEADLINE = [
  { key: "gsm8k", label: "GSM8K", desc: "grade-school multi-step math" },
  { key: "humaneval", label: "HumanEval", desc: "pass@1 hand-written Python" },
  { key: "math_500", label: "MATH-500", desc: "competition-style math" },
  { key: "bbh", label: "BBH", desc: "Big-Bench Hard, 27-task suite" },
  { key: "mmlu_pro", label: "MMLU-Pro", desc: "extended professional knowledge" },
  { key: "ifeval", label: "IFEval", desc: "instruction-following strict" },
];

const ALIASES: Record<string, string[]> = {
  bbh: ["bbh", "bbh_cot_fewshot"],
  arc: ["arc_challenge", "arc"],
  truthfulqa_mc2: ["truthfulqa_mc2", "truthfulqa"],
  mmlu_pro: ["mmlu_pro", "mmlupro"],
  math_500: ["math_500", "math500", "math-500"],
};

function pickScore(map: Record<string, number | null> | undefined, key: string): number | null {
  if (!map) return null;
  const candidates = ALIASES[key] ?? [key];
  for (const c of candidates) {
    const v = map[c];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

/**
 * Bench tab. Six-up grid of held-out evalscope benches. Each card shows
 * three bars: King, Teacher (Qwen3.5-35B-A3B), Reference (Qwen3.5-4B).
 * The footer line gives "retain %" (king / teacher) and "lift" (king
 * minus reference) in pp.
 *
 * These benches are NEVER inside the validator. Pure transfer
 * measurement — they're the answer to "did the composite eval produce
 * a model that's actually better, or just better at the composite?"
 */
export function BenchPanel() {
  const [data, setData] = useState<BenchmarksResponse | null>(null);

  useEffect(() => {
    let cancel = false;
    const load = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/benchmarks`, {
          cache: "no-store",
        });
        if (res.ok && !cancel) {
          setData((await res.json()) as BenchmarksResponse);
        }
      } catch {}
    };
    load();
    const id = setInterval(load, 120_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const king = data?.models.find((m) => m.is_king) ?? data?.models[0] ?? null;
  // Try to find a teacher entry. Otherwise teacher is the reference baseline.
  const teacher = data?.models.find((m) =>
    m.model.toLowerCase().includes("35b")
  );
  const reference = data?.baseline ?? null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 grid-rows-[1fr_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      {HEADLINE.map((b, i) => {
        const k = pickScore(king?.benchmarks, b.key);
        const t = pickScore(teacher?.benchmarks, b.key);
        const r = pickScore(reference?.benchmarks, b.key);
        const isLastCol = (i + 1) % 3 === 0;
        const isLastRow = i >= HEADLINE.length - 3;
        return (
          <BenchCard
            key={b.key}
            label={b.label}
            desc={b.desc}
            king={k}
            teacher={t}
            reference={r}
            noRightBorder={isLastCol}
            noBottomBorder={isLastRow}
          />
        );
      })}
      <div className="col-span-2 sm:col-span-3 px-6 sm:px-9 py-4 border-t border-border text-[10px] text-meta">
        Held-out evalscope on the live king. <strong className="text-foreground">None of these benches are inside the validator</strong> — pure transfer measurement. The composite eval that crowns the king runs different items (procedurally generated, block-seeded). If those two sets disagree, that&apos;s the signal that the composite is being gamed; if they agree, the composite is producing real models.
      </div>
    </div>
  );
}

interface BenchCardProps {
  label: string;
  desc: string;
  king: number | null;
  teacher: number | null;
  reference: number | null;
  noRightBorder?: boolean;
  noBottomBorder?: boolean;
}

function BenchCard({
  label,
  desc,
  king,
  teacher,
  reference,
  noRightBorder,
  noBottomBorder,
}: BenchCardProps) {
  const retain =
    king != null && teacher != null && teacher > 0 ? (king / teacher) * 100 : null;
  const lift =
    king != null && reference != null ? (king - reference) * 100 : null;

  // Bars are 0–1; values come in as fractions already.
  const teacherPct = teacher != null ? Math.max(0, Math.min(1, teacher)) * 100 : 0;
  const kingPct = king != null ? Math.max(0, Math.min(1, king)) * 100 : 0;
  const refPct = reference != null ? Math.max(0, Math.min(1, reference)) * 100 : 0;

  return (
    <div
      className={[
        "px-6 sm:px-7 py-6 flex flex-col",
        noRightBorder ? "" : "border-r border-border",
        noBottomBorder ? "" : "border-b border-border",
      ].join(" ")}
    >
      <h5 className="text-[22px] font-medium tracking-[-0.025em]">{label}</h5>
      <div className="text-[10px] text-meta uppercase tracking-[0.18em] mt-0.5 mb-4">
        {desc}
      </div>

      <BenchRow label="Teacher" pct={teacherPct} value={teacher} />
      <BenchRow label="King" pct={kingPct} value={king} highlight />
      <BenchRow label="Ref-4B" pct={refPct} value={reference} />

      <div className="mt-auto pt-4 text-[11px] text-meta num flex gap-4">
        {retain != null && (
          <span>
            retain <strong className="text-foreground">{retain.toFixed(1)}%</strong>
          </span>
        )}
        {lift != null && (
          <span>
            lift{" "}
            <strong className="text-foreground">
              {lift >= 0 ? "+" : ""}
              {lift.toFixed(1)}pp
            </strong>
          </span>
        )}
      </div>
    </div>
  );
}

function BenchRow({
  label,
  pct,
  value,
  highlight,
}: {
  label: string;
  pct: number;
  value: number | null;
  highlight?: boolean;
}) {
  return (
    <div className="grid grid-cols-[60px_1fr_44px] gap-3 items-center text-[11px] mb-2.5">
      <span
        className={[
          "text-[10px] uppercase tracking-[0.16em]",
          highlight ? "text-foreground font-medium" : "text-meta",
        ].join(" ")}
      >
        {label}
      </span>
      <div className="h-1.5 bg-[#f1f1f1] relative">
        <div
          className={[
            "absolute inset-y-0 left-0",
            highlight ? "bg-foreground" : "bg-[#dcdcdc]",
          ].join(" ")}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-right text-[12px] num">
        {value != null ? (value * 100).toFixed(1) : "—"}
      </span>
    </div>
  );
}
