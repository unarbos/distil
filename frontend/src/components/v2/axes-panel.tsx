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
  block?: number;
  timestamp?: number;
}

/**
 * 6 macro-axes, each = arithmetic mean of its constituent v28 axes
 * that are non-null for the miner. Heuristic groupings based on what
 * the axis measures:
 *
 *   DISTILL    — distribution match to teacher (kl + on_policy_rkl + capability)
 *   REASONING  — multi-step + olympiad math (math + reasoning + aime)
 *   CODING     — programming (code + mbpp)
 *   DISCIPLINE — generation discipline (length + degeneracy + reasoning_density)
 *   DIALOGUE   — conversational (judge_probe + chat_turns_probe)
 *   ROBUSTNESS — instruction-following + adversarial (ifeval + tool_use + long_context + robustness)
 *
 * The single 17-axis radial would be unreadable; rolling up to 6 axes
 * keeps the "where is this miner strong/weak" gestalt readable in
 * <1s. The SCORES sub-tab still shows all 17 raw axes for the
 * detail-oriented.
 */
const MACRO_AXES: { label: string; members: string[] }[] = [
  {
    label: "DISTILL",
    members: ["kl", "on_policy_rkl", "capability"],
  },
  {
    label: "REASONING",
    members: ["math_bench", "reasoning_bench", "aime_bench"],
  },
  {
    label: "CODING",
    members: ["code_bench", "mbpp_bench"],
  },
  {
    label: "DISCIPLINE",
    members: ["length", "degeneracy", "reasoning_density"],
  },
  {
    label: "DIALOGUE",
    members: ["judge_probe", "chat_turns_probe"],
  },
  {
    label: "ROBUSTNESS",
    members: ["ifeval_bench", "tool_use_bench", "long_context_bench", "robustness_bench"],
  },
];

type SubTab = "radial" | "scores" | "samples" | "benchmarks";

export function AxesPanel() {
  const [latest, setLatest] = useState<H2hLatest | null>(null);
  const [subtab, setSubtab] = useState<SubTab>("radial");
  const [primaryUid, setPrimaryUid] = useState<number | null>(null);
  const [compareUid, setCompareUid] = useState<number | null>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/h2h-latest`, {
          cache: "no-store",
        });
        if (res.ok && !cancel) {
          const json = await res.json();
          setLatest(json);
          // Auto-select the king as the primary on first load.
          if (primaryUid == null && typeof json?.king_uid === "number") {
            setPrimaryUid(json.king_uid);
          }
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 30_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const eligible = (latest?.results ?? []).filter(
    (r) => r.uid != null && r.composite?.worst != null,
  );
  const primary = eligible.find((r) => r.uid === primaryUid) ?? null;
  const compare = compareUid != null ? eligible.find((r) => r.uid === compareUid) ?? null : null;

  return (
    <div className="px-6 sm:px-9 py-8 min-h-[calc(100vh-3.5rem-3rem)] overflow-y-auto">
      {/* Sub-tab strip + COMPARE */}
      <div className="flex items-baseline justify-between gap-3 flex-wrap mb-6">
        <div className="flex items-center gap-0">
          {(
            [
              ["radial", "Radial"],
              ["scores", "Scores"],
              ["samples", "Samples"],
              ["benchmarks", "Benchmarks"],
            ] as [SubTab, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSubtab(key)}
              className={[
                "text-[11px] uppercase tracking-[0.18em] py-2 px-4 transition-colors",
                "border-b-2",
                subtab === key
                  ? "border-foreground text-foreground"
                  : "border-transparent text-meta hover:text-foreground",
              ].join(" ")}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <UidPicker
            label="Primary"
            value={primaryUid}
            onChange={setPrimaryUid}
            results={eligible}
          />
          <UidPicker
            label="Compare"
            value={compareUid}
            onChange={setCompareUid}
            results={eligible.filter((r) => r.uid !== primaryUid)}
            allowClear
          />
        </div>
      </div>

      {/* Body */}
      {subtab === "radial" && (
        <RadialView primary={primary} compare={compare} />
      )}
      {subtab === "scores" && (
        <ScoresView primary={primary} compare={compare} />
      )}
      {subtab === "samples" && <SamplesView primary={primary} />}
      {subtab === "benchmarks" && <BenchmarksView primary={primary} />}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  UID picker
// ────────────────────────────────────────────────────────────────────

function UidPicker({
  label,
  value,
  onChange,
  results,
  allowClear,
}: {
  label: string;
  value: number | null;
  onChange: (uid: number | null) => void;
  results: H2hResult[];
  allowClear?: boolean;
}) {
  return (
    <label className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-meta">
      <span>{label}</span>
      <select
        value={value ?? ""}
        onChange={(e) =>
          onChange(e.target.value === "" ? null : Number(e.target.value))
        }
        className="h-7 border border-border bg-white px-2 text-[12px] num focus:outline-none focus:border-foreground"
      >
        {allowClear && <option value="">— none —</option>}
        {!allowClear && value == null && <option value="">— pick —</option>}
        {[...results]
          .sort((a, b) => (b.composite?.worst ?? 0) - (a.composite?.worst ?? 0))
          .map((r) => (
            <option key={r.uid} value={r.uid}>
              {r.is_king ? "♛ " : ""}#{r.uid} · {r.model.split("/").pop()?.slice(0, 24)}
            </option>
          ))}
      </select>
    </label>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Radial sub-tab
// ────────────────────────────────────────────────────────────────────

function macroValues(r: H2hResult | null): number[] {
  if (!r?.composite?.axes) return MACRO_AXES.map(() => 0);
  const axes = r.composite.axes;
  return MACRO_AXES.map((m) => {
    const present = m.members
      .map((k) => axes[k])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (present.length === 0) return 0;
    return present.reduce((a, b) => a + b, 0) / present.length;
  });
}

function RadialView({
  primary,
  compare,
}: {
  primary: H2hResult | null;
  compare: H2hResult | null;
}) {
  if (!primary) {
    return (
      <p className="text-[12px] text-meta">
        Pick a UID to see its composite radial.
      </p>
    );
  }
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-10 items-start">
      <RadialChart primary={primary} compare={compare} />
      <RadialLegend primary={primary} compare={compare} />
    </div>
  );
}

function RadialChart({
  primary,
  compare,
}: {
  primary: H2hResult;
  compare: H2hResult | null;
}) {
  const W = 720;
  const H = 720;
  const cx = W / 2;
  const cy = H / 2;
  const R = 260;

  const N = MACRO_AXES.length;
  // Angles: top = 0, clockwise. -pi/2 puts the first axis at 12 o'clock.
  const angle = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / N;
  const ringPath = (frac: number): string => {
    const r = R * frac;
    const pts = MACRO_AXES.map((_, i) => {
      const a = angle(i);
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number];
    });
    return (
      pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ") + " Z"
    );
  };

  const polygonPath = (vals: number[]): string => {
    const pts = vals.map((v, i) => {
      const a = angle(i);
      const r = R * Math.max(0, Math.min(1, v));
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number];
    });
    return (
      pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ") + " Z"
    );
  };

  const primaryVals = macroValues(primary);
  const compareVals = compare ? macroValues(compare) : null;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="w-full h-auto max-h-[640px]"
    >
      {/* Concentric reference rings at 0.25 / 0.5 / 0.75 / 1.0 */}
      {[0.25, 0.5, 0.75, 1].map((frac) => (
        <path
          key={frac}
          d={ringPath(frac)}
          fill="none"
          stroke="#ebebeb"
          strokeWidth={1}
        />
      ))}

      {/* Spokes */}
      {MACRO_AXES.map((_, i) => {
        const a = angle(i);
        return (
          <line
            key={`s-${i}`}
            x1={cx}
            y1={cy}
            x2={cx + R * Math.cos(a)}
            y2={cy + R * Math.sin(a)}
            stroke="#ebebeb"
            strokeWidth={1}
          />
        );
      })}

      {/* Compare polygon (drawn first so primary overlays it) */}
      {compareVals && (
        <>
          <path
            d={polygonPath(compareVals)}
            fill="rgba(189,189,189,0.15)"
            stroke="#bdbdbd"
            strokeWidth={1.25}
            strokeDasharray="4 3"
          />
          {compareVals.map((v, i) => {
            const a = angle(i);
            const r = R * Math.max(0, Math.min(1, v));
            return (
              <circle
                key={`cc-${i}`}
                cx={cx + r * Math.cos(a)}
                cy={cy + r * Math.sin(a)}
                r={3}
                fill="#bdbdbd"
              />
            );
          })}
        </>
      )}

      {/* Primary polygon (Times-italic accent: amber outline) */}
      <path
        d={polygonPath(primaryVals)}
        fill="rgba(206,141,67,0.10)"
        stroke="#ce8d43"
        strokeWidth={1.75}
      />
      {primaryVals.map((v, i) => {
        const a = angle(i);
        const r = R * Math.max(0, Math.min(1, v));
        return (
          <circle
            key={`pc-${i}`}
            cx={cx + r * Math.cos(a)}
            cy={cy + r * Math.sin(a)}
            r={4}
            fill="#ce8d43"
          />
        );
      })}

      {/* Axis labels */}
      {MACRO_AXES.map((m, i) => {
        const a = angle(i);
        const lr = R + 28;
        const x = cx + lr * Math.cos(a);
        const y = cy + lr * Math.sin(a);
        let anchor: "start" | "middle" | "end" = "middle";
        const cosA = Math.cos(a);
        if (cosA > 0.2) anchor = "start";
        else if (cosA < -0.2) anchor = "end";
        return (
          <text
            key={`l-${i}`}
            x={x}
            y={y + 4}
            textAnchor={anchor}
            fontFamily="Inter, sans-serif"
            fontSize={11}
            letterSpacing={2.2}
            fill="#8a8a8a"
          >
            {m.label}
          </text>
        );
      })}
    </svg>
  );
}

function RadialLegend({
  primary,
  compare,
}: {
  primary: H2hResult;
  compare: H2hResult | null;
}) {
  const primaryVals = macroValues(primary);
  const compareVals = compare ? macroValues(compare) : null;
  return (
    <div className="space-y-6">
      <LegendCard
        accent="#ce8d43"
        miner={primary}
        macroLabels={MACRO_AXES.map((m) => m.label)}
        values={primaryVals}
      />
      {compare && compareVals && (
        <LegendCard
          accent="#bdbdbd"
          miner={compare}
          macroLabels={MACRO_AXES.map((m) => m.label)}
          values={compareVals}
          dashed
        />
      )}
      <p className="text-[10px] text-meta leading-relaxed pt-3 border-t border-border">
        Each spoke is a <strong className="text-foreground">macro-axis</strong>{" "}
        (a mean of 2-4 v28 composite axes). Outer ring = 1.0 (best). Inner =
        0. The composite ranks miners on the worst single v28 axis (not on
        these macro means) — see the <strong>Scores</strong> sub-tab for the
        17-axis breakdown.
      </p>
    </div>
  );
}

function LegendCard({
  accent,
  miner,
  macroLabels,
  values,
  dashed,
}: {
  accent: string;
  miner: H2hResult;
  macroLabels: string[];
  values: number[];
  dashed?: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <span
          className="inline-block w-3 h-3"
          style={{
            backgroundColor: accent,
            border: dashed ? `1.5px dashed ${accent}` : "none",
            backgroundClip: dashed ? "padding-box" : undefined,
          }}
        />
        <span className="text-[14px] font-medium">
          {miner.is_king && "♛ "}#{miner.uid}
        </span>
        <span className="text-[11px] text-meta truncate" title={miner.model}>
          {miner.model}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[12px]">
        {macroLabels.map((label, i) => (
          <div key={label} className="flex items-baseline justify-between gap-2">
            <span className="text-meta text-[10px] uppercase tracking-[0.14em]">
              {label}
            </span>
            <span className="num font-medium">{values[i].toFixed(3)}</span>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-meta num mt-2">
        worst {miner.composite?.worst?.toFixed(3) ?? "—"} · weighted{" "}
        {miner.composite?.weighted?.toFixed(3) ?? "—"}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Scores sub-tab — full 17-axis breakdown
// ────────────────────────────────────────────────────────────────────

const ALL_AXES: { key: string; group: string }[] = [
  { key: "kl", group: "Distribution" },
  { key: "on_policy_rkl", group: "Distribution" },
  { key: "capability", group: "Distribution" },
  { key: "length", group: "Discipline" },
  { key: "degeneracy", group: "Discipline" },
  { key: "reasoning_density", group: "Discipline" },
  { key: "judge_probe", group: "Dialogue" },
  { key: "chat_turns_probe", group: "Dialogue" },
  { key: "math_bench", group: "Capability" },
  { key: "code_bench", group: "Capability" },
  { key: "reasoning_bench", group: "Capability" },
  { key: "ifeval_bench", group: "Capability" },
  { key: "aime_bench", group: "Capability" },
  { key: "mbpp_bench", group: "Capability" },
  { key: "tool_use_bench", group: "Capability" },
  { key: "long_context_bench", group: "Capability" },
  { key: "robustness_bench", group: "Capability" },
];

function ScoresView({
  primary,
  compare,
}: {
  primary: H2hResult | null;
  compare: H2hResult | null;
}) {
  if (!primary) {
    return (
      <p className="text-[12px] text-meta">Pick a UID to see its scores.</p>
    );
  }
  const pa = primary.composite?.axes ?? {};
  const ca = compare?.composite?.axes ?? {};
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-12 gap-y-6">
      {Array.from(new Set(ALL_AXES.map((a) => a.group))).map((grp) => (
        <div key={grp}>
          <h3 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium mb-3">
            {grp}
          </h3>
          {ALL_AXES.filter((a) => a.group === grp).map(({ key }) => {
            const pv = typeof pa[key] === "number" ? (pa[key] as number) : null;
            const cv = typeof ca[key] === "number" ? (ca[key] as number) : null;
            return (
              <ScoreRow
                key={key}
                axis={key}
                primary={pv}
                compare={cv}
                hasCompare={!!compare}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

function ScoreRow({
  axis,
  primary,
  compare,
  hasCompare,
}: {
  axis: string;
  primary: number | null;
  compare: number | null;
  hasCompare: boolean;
}) {
  const label = axis.replace(/_bench$/, "").replace(/_/g, " ");
  return (
    <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3 py-1.5 text-[12px] num">
      <div className="text-foreground capitalize">{label}</div>
      <div className="flex items-center gap-2">
        <div className="w-32 h-1.5 bg-[#f1f1f1] relative">
          {primary != null && (
            <div
              className="absolute inset-y-0 left-0 bg-[#ce8d43]"
              style={{ width: `${Math.max(0, Math.min(1, primary)) * 100}%` }}
            />
          )}
          {hasCompare && compare != null && (
            <div
              className="absolute inset-y-0 left-0 border-r-2 border-[#bdbdbd]"
              style={{ width: `${Math.max(0, Math.min(1, compare)) * 100}%` }}
            />
          )}
        </div>
        <span className="w-12 text-right">
          {primary != null ? primary.toFixed(3) : "—"}
        </span>
      </div>
      {hasCompare && (
        <span className="w-12 text-right text-meta">
          {compare != null ? compare.toFixed(3) : "—"}
        </span>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Samples sub-tab (placeholder until pod-side sample export lands)
// ────────────────────────────────────────────────────────────────────

function SamplesView({ primary }: { primary: H2hResult | null }) {
  return (
    <div className="max-w-2xl space-y-4 text-[13px] text-foreground leading-relaxed">
      <p className="text-[11px] uppercase tracking-[0.18em] text-meta font-medium">
        Coming soon
      </p>
      <p>
        Per-prompt sample grid (teacher continuation vs student continuation,
        per-axis scoring detail, link to raw eval-data file). The validator
        already records this on the pod; surfacing it through the API is
        on the roadmap.
      </p>
      {primary && (
        <p className="text-meta text-[12px]">
          Currently selected: ♛ #{primary.uid} · {primary.model}.
        </p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Benchmarks sub-tab — held-out auto-bench summary
// ────────────────────────────────────────────────────────────────────

function BenchmarksView({ primary }: { primary: H2hResult | null }) {
  return (
    <div className="max-w-3xl space-y-4 text-[13px] leading-relaxed">
      <p>
        For a fuller view of the held-out auto-bench numbers (GSM8K,
        HumanEval, IFEval, BBH, MMLU-Pro, ARC) on the king + teacher +
        reference, see the{" "}
        <a href="#bench" className="underline">
          Bench tab
        </a>
        .
      </p>
      <p className="text-meta text-[12px]">
        These benches are <strong className="text-foreground">not</strong>{" "}
        used by the validator&apos;s composite eval. The composite&apos;s
        bench axes (math_bench, code_bench, etc.) are generated
        procedurally per round from the block-seed and never use public
        datasets. See <code>paper/benchmarks_as_north_star.md</code>.
      </p>
      {primary && (
        <p className="text-meta text-[12px]">
          Currently selected: ♛ #{primary.uid} · {primary.model}.
        </p>
      )}
    </div>
  );
}
