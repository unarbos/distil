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
}

interface BenchmarksResponse {
  models: BenchmarkPayload[];
  baseline: BenchmarkPayload | null;
}

// Six benches we headline. The auto-bench backend (lm-evaluation-
// harness) actually runs these six on every king flip; the held-out
// evalscope (run separately by scripts/run_king_benchmark.py) covers
// math_500 in addition. We surface the auto-bench set here and link
// to evalscope reports when available.
const HEADLINE = [
  { key: "gsm8k", label: "GSM8K", desc: "grade-school multi-step math" },
  { key: "humaneval", label: "HumanEval", desc: "pass@1 hand-written Python" },
  { key: "ifeval", label: "IFEval", desc: "instruction-following strict" },
  { key: "bbh", label: "BBH", desc: "Big-Bench Hard, 27-task suite" },
  { key: "mmlu_pro", label: "MMLU-Pro", desc: "extended professional knowledge" },
  { key: "arc", label: "ARC", desc: "AI2 reasoning challenge" },
];

const ALIASES: Record<string, string[]> = {
  bbh: ["bbh", "bbh_cot_fewshot"],
  arc: ["arc_challenge", "arc", "arc_easy"],
  truthfulqa_mc2: ["truthfulqa_mc2", "truthfulqa"],
  mmlu_pro: ["mmlu_pro", "mmlupro"],
  math_500: ["math_500", "math500", "math-500"],
};

interface ScoreCount {
  score: number | null;
  count: number | null;
}

function pickScoreAndCount(
  scoreMap: Record<string, number | null> | undefined,
  countMap: Record<string, number | null> | undefined,
  key: string,
): ScoreCount {
  if (!scoreMap) return { score: null, count: null };
  const candidates = ALIASES[key] ?? [key];
  for (const c of candidates) {
    const v = scoreMap[c];
    if (typeof v === "number" && Number.isFinite(v)) {
      const cnt = countMap?.[c];
      return {
        score: v,
        count: typeof cnt === "number" && Number.isFinite(cnt) ? cnt : null,
      };
    }
  }
  return { score: null, count: null };
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

  // Surface limit (e.g. 50 items) so the user knows this is the
  // auto-bench cut, not the full evalscope run.
  const limit =
    typeof king?.limit === "number" && Number.isFinite(king.limit)
      ? king.limit
      : null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 grid-rows-[1fr_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      {HEADLINE.map((b, i) => {
        const k = pickScoreAndCount(king?.benchmarks, king?.counts, b.key);
        const t = pickScoreAndCount(teacher?.benchmarks, teacher?.counts, b.key);
        const r = pickScoreAndCount(reference?.benchmarks, reference?.counts, b.key);
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
      <div className="col-span-2 sm:col-span-3 px-6 sm:px-9 py-4 border-t border-border text-[10px] text-meta leading-relaxed">
        Auto-bench (lm-evaluation-harness, {limit ? `${limit}-item` : "subset"} cut)
        on the current king. <strong className="text-foreground">None of these benches are inside the validator</strong> — pure transfer measurement. The composite eval runs different items (procedurally generated, block-seeded).
        {king?.uid != null && (
          <>
            {" "}King: <strong className="text-foreground">UID {king.uid}</strong>{" "}
            <span className="num">{king.model}</span>.
          </>
        )}
        {" "}A bench shown as <strong className="text-foreground">n/a</strong> means
        the auto-bench backend didn&apos;t complete that task for this UID — not
        that the model failed. The full held-out evalscope run is at{" "}
        <code className="font-mono">benchmark_results/v28-full/</code>.
      </div>
    </div>
  );
}

interface BenchCardProps {
  label: string;
  desc: string;
  king: ScoreCount;
  teacher: ScoreCount;
  reference: ScoreCount;
  noRightBorder?: boolean;
  noBottomBorder?: boolean;
}

/**
 * `score` is meaningful only when `count > 0`. The auto-bench backend
 * sometimes records 0.0 with count=0 for a task it didn't actually
 * run; the dashboard treated that as a real 0% and drew a 0% bar
 * (the rendering bug miners called out on Discord 2026-04-27).
 *
 * Treat anything with count == 0 as "not run" — render n/a, no bar.
 */
function isMeasured(s: ScoreCount): boolean {
  if (s.score == null) return false;
  // Accept null count as "score-only" (e.g. evalscope reports without
  // counts), but reject zero counts as "didn't run".
  if (s.count == null) return true;
  return s.count > 0;
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
  const kingMeasured = isMeasured(king);
  const teacherMeasured = isMeasured(teacher);
  const refMeasured = isMeasured(reference);

  const retain =
    kingMeasured && teacherMeasured && teacher.score! > 0
      ? (king.score! / teacher.score!) * 100
      : null;
  const lift =
    kingMeasured && refMeasured ? (king.score! - reference.score!) * 100 : null;

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

      <BenchRow label="Teacher" entry={teacher} measured={teacherMeasured} />
      <BenchRow label="King" entry={king} measured={kingMeasured} highlight />
      <BenchRow label="Ref-4B" entry={reference} measured={refMeasured} />

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
        {!kingMeasured && (
          <span
            className="text-meta italic"
            title="Auto-bench backend didn't complete this task for the current king. Run scripts/run_king_benchmark.py to produce held-out evalscope numbers."
          >
            not run for current king
          </span>
        )}
      </div>
    </div>
  );
}

function BenchRow({
  label,
  entry,
  measured,
  highlight,
}: {
  label: string;
  entry: ScoreCount;
  measured: boolean;
  highlight?: boolean;
}) {
  const pct = measured ? Math.max(0, Math.min(1, entry.score!)) * 100 : 0;
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
        {measured && (
          <div
            className={[
              "absolute inset-y-0 left-0",
              highlight ? "bg-foreground" : "bg-[#dcdcdc]",
            ].join(" ")}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span
        className={[
          "text-right text-[12px] num",
          measured ? "" : "text-meta italic",
        ].join(" ")}
        title={
          measured
            ? entry.count != null
              ? `${entry.count} items`
              : undefined
            : "task not run for this UID"
        }
      >
        {measured ? (entry.score! * 100).toFixed(1) : "n/a"}
      </span>
    </div>
  );
}
