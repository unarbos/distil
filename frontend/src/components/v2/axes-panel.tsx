"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

/**
 * Axes panel — composite breakdown for the king + top contenders.
 *
 * Data source rationale (REVISED 2026-04-27, pass #4): the previous
 * version fetched ``/api/h2h-latest``, which under single-eval mode
 * does NOT include the king (kings aren't re-scored each round). The
 * king (UID 123 today) was therefore missing from the dropdown and
 * the panel rendered empty for the operator's screenshot. Switched
 * to ``/api/leaderboard`` which returns the king + 4 contenders, all
 * with full per-axis composite. This is the canonical "five most
 * relevant miners right now" set — exactly what miners want to
 * compare themselves against.
 *
 * Tabs:
 *   RADIAL    — 6-spoke macro chart, primary + optional compare
 *   SCORES    — full 17-axis breakdown grouped by concern
 *   SAMPLES   — placeholder for per-prompt sample export (roadmap)
 *   BENCHMARKS — pointer to Bench tab
 *
 * Default primary = the current king. The Pareto chart from the
 * previous design is retired — too many dots, too little explanation.
 * The radial answers the question "where is THIS miner strong?"
 * which is what most users actually want.
 */
interface CompositePayload {
  // v30.2 ranking key (replaces ``worst`` as the dethrone gate).
  final?: number | null;
  worst_3_mean?: number | null;
  final_alpha?: number | null;
  worst?: number | null;
  weighted?: number | null;
  axes?: Record<string, number | null | undefined>;
  axes_raw?: Record<string, number | null | undefined>;
  baseline_penalty?: {
    enabled?: boolean;
    alpha?: number;
    n_docked?: number;
    applied?: Record<string, { raw: number; adjusted: number; reference: number; gap: number; dock: number }>;
  } | null;
  broken_axes?: string[];
  present_count?: number;
  version?: number;
}

interface MinerSummary {
  uid: number;
  model: string;
  h2h_kl?: number | null;
  block?: number;
  composite: CompositePayload;
}

/**
 * Public API shape: {"leaderboard": {king, contenders, phase, ...}}.
 * The router wraps the leaderboard payload one level deep so this
 * panel needs to dereference json.leaderboard.king on the response.
 * Found on 2026-04-27 pass #4 — earlier I assumed king was top-level
 * which made the panel render empty against the deployed API.
 */
interface LeaderboardInner {
  king: MinerSummary | null;
  contenders: MinerSummary[];
  phase?: string | null;
  initial_eval_complete?: boolean;
  completed_at?: number | null;
}
interface LeaderboardResponse {
  leaderboard?: LeaderboardInner | null;
  // Fallback shape: some deployments expose it flat. Try both.
  king?: MinerSummary | null;
  contenders?: MinerSummary[];
}

/**
 * v30.2/v30.3 macro axis groups for the radial chart. These reflect
 * the new composite where:
 *  - Five SKILL GROUPS replace per-bench-axis weights (sub-axes still
 *    computed for telemetry but the GROUP is the ranking driver).
 *  - super_teacher: rewards exceeding the teacher on verifiable benches.
 *  - DISTILL: teacher-similarity cluster (KL, RKL, top_k_overlap, etc.)
 *  - QUALITY: judge_probe + long_form_judge + chat_turns_probe.
 *  - DISCIPLINE: length, degeneracy, capability, reasoning_density.
 */
const MACRO_AXES: { label: string; members: string[]; description: string }[] = [
  {
    label: "DISTILL",
    members: [
      "on_policy_rkl",
      "kl",
      "top_k_overlap",
      "kl_is",
      "forking_rkl",
      "teacher_trace_plausibility",
      "entropy_aware_kl",
      "tail_decoupled_kl",
      "capability",
    ],
    description:
      "How closely the student matches the teacher's distribution. Includes the canonical KL/RKL signals + v30/v30.3 research-backed shadow signals (top_k_overlap, IS-KL, forking-RKL, teacher-trace plausibility, entropy-aware KL, tail-decoupled KL).",
  },
  {
    label: "MATH",
    members: ["math_skill_group", "math_bench", "aime_bench", "robustness_bench"],
    description:
      "v30.2 math_skill_group = mean of {math_bench, aime_bench, robustness_bench}. Sub-axes shown for transparency. The GROUP is what gates ranking.",
  },
  {
    label: "CODE",
    members: [
      "code_skill_group",
      "code_bench",
      "mbpp_bench",
      "debug_bench",
      "correction_bench",
      "refactor_bench",
    ],
    description:
      "v30.2 code_skill_group = mean of {code_bench, mbpp_bench, debug_bench, correction_bench, refactor_bench}. Covers write-from-scratch + bug fixing + behavior-preserving refactor.",
  },
  {
    label: "REASONING",
    members: [
      "reasoning_skill_group",
      "reasoning_bench",
      "multi_doc_synthesis_bench",
      "long_context_bench",
    ],
    description:
      "v30.2 reasoning_skill_group = mean of {reasoning_bench, multi_doc_synthesis_bench, long_context_bench}. Multi-step deduction + cross-document synthesis + needle-in-haystack.",
  },
  {
    label: "KNOWLEDGE",
    members: ["knowledge_skill_group", "knowledge_bench", "pragmatic_bench"],
    description:
      "v30.2 knowledge_skill_group = mean of {knowledge_bench v2 (procedural fact-like reasoning), pragmatic_bench (theory-of-mind / scalar implicature / indirect-request)}.",
  },
  {
    label: "EXCEEDS-TEACHER",
    members: ["super_teacher"],
    description:
      "v30.2 super_teacher (weight 0.10): rewards exceeding the teacher on verifiable benches via tanh(mean(max(0, student - teacher)) / 0.10). Incentivises Stage-4 GRPO + post-distillation SFT — the only paths to producing models that exceed teacher capability.",
  },
  {
    label: "QUALITY",
    members: ["judge_probe", "long_form_judge", "chat_turns_probe"],
    description:
      "Conversational quality. judge_probe: short-answer rubric (1-5 → [0,1]). long_form_judge: 300-500 word essay rubric (structure / depth / coherence / length). chat_turns_probe: 3-turn coherence.",
  },
  {
    label: "STAND-ALONE",
    members: ["tool_use_bench", "ifeval_bench", "calibration_bench"],
    description:
      "Capabilities kept separate from groups because they measure orthogonal skills: agentic Python (tool_use), instruction-following with structural constraints (ifeval), honest refusal under unsolvable items (calibration).",
  },
  {
    label: "DISCIPLINE",
    members: ["length", "degeneracy", "reasoning_density"],
    description:
      "Generation discipline. length: ramble-vs-hard-stop ratio. degeneracy: self-BLEU + termination + non-degeneracy. reasoning_density: pass_frac × length_bonus.",
  },
];

type SubTab = "radial" | "scores" | "samples" | "benchmarks";

export function AxesPanel() {
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [subtab, setSubtab] = useState<SubTab>("radial");
  const [primaryUid, setPrimaryUid] = useState<number | null>(null);
  const [compareUid, setCompareUid] = useState<number | null>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/leaderboard`, {
          cache: "no-store",
        });
        if (res.ok && !cancel) {
          const json = (await res.json()) as LeaderboardResponse;
          setData(json);
          setLoading(false);
          const king = json?.leaderboard?.king ?? json?.king ?? null;
          if (primaryUid == null && king?.uid != null) {
            setPrimaryUid(king.uid);
          }
        }
      } catch {
        if (!cancel) setLoading(false);
      }
    };
    tick();
    const id = setInterval(tick, 30_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const inner: LeaderboardInner | null =
    data?.leaderboard ?? (data?.king ? { king: data.king, contenders: data.contenders ?? [] } : null);

  const all: MinerSummary[] = (() => {
    const out: MinerSummary[] = [];
    if (inner?.king) out.push(inner.king);
    for (const c of inner?.contenders ?? []) out.push(c);
    return out.filter((r) => r.composite?.worst != null);
  })();

  const primary = all.find((r) => r.uid === primaryUid) ?? inner?.king ?? null;
  const compare = compareUid != null ? all.find((r) => r.uid === compareUid) ?? null : null;
  const isKing = primary?.uid === inner?.king?.uid && inner?.king != null;

  return (
    <div className="px-6 sm:px-9 py-8 min-h-[calc(100vh-3.5rem-3rem)] overflow-y-auto">
      {/* Sub-tab strip + UID pickers */}
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
        <div className="flex items-center gap-2 flex-wrap">
          <UidPicker
            label="Primary"
            value={primaryUid}
            onChange={setPrimaryUid}
            results={all}
            kingUid={inner?.king?.uid ?? null}
          />
          <UidPicker
            label="Compare"
            value={compareUid}
            onChange={setCompareUid}
            results={all.filter((r) => r.uid !== primaryUid)}
            kingUid={inner?.king?.uid ?? null}
            allowClear
          />
        </div>
      </div>

      {loading && !data && (
        <p className="text-[12px] text-meta">Loading composite data…</p>
      )}
      {!loading && all.length === 0 && (
        <p className="text-[12px] text-meta">
          No composite records available yet — the dashboard will populate
          once the next eval round completes.
        </p>
      )}

      {primary && (
        <>
          {subtab === "radial" && (
            <RadialView primary={primary} compare={compare} isKing={isKing} />
          )}
          {subtab === "scores" && (
            <ScoresView primary={primary} compare={compare} isKing={isKing} />
          )}
          {subtab === "samples" && <SamplesView primary={primary} />}
          {subtab === "benchmarks" && <BenchmarksView primary={primary} />}
        </>
      )}
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
  kingUid,
  allowClear,
}: {
  label: string;
  value: number | null;
  onChange: (uid: number | null) => void;
  results: MinerSummary[];
  kingUid: number | null;
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
        className="h-7 border border-border bg-[var(--surface-elevated)] px-2 text-[12px] num focus:outline-none focus:border-foreground"
      >
        {allowClear && <option value="">— none —</option>}
        {!allowClear && value == null && <option value="">— pick —</option>}
        {[...results]
          .sort((a, b) => (b.composite?.worst ?? 0) - (a.composite?.worst ?? 0))
          .map((r) => (
            <option key={r.uid} value={r.uid}>
              {r.uid === kingUid ? "♛ " : ""}#{r.uid} ·{" "}
              {r.model.split("/").pop()?.slice(0, 24)}
            </option>
          ))}
      </select>
    </label>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Radial sub-tab
// ────────────────────────────────────────────────────────────────────

function macroValues(r: MinerSummary | null): number[] {
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

function findLimitingAxis(r: MinerSummary): { axis: string; value: number } | null {
  const axes = r.composite?.axes;
  if (!axes) return null;
  const broken = new Set(r.composite?.broken_axes ?? []);
  let best: { axis: string; value: number } | null = null;
  for (const [k, v] of Object.entries(axes)) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (broken.has(k)) continue;
    if (best == null || v < best.value) best = { axis: k, value: v };
  }
  return best;
}

function prettyAxis(s: string): string {
  return s.replace(/_bench$/, "").replace(/_/g, " ");
}

function RadialView({
  primary,
  compare,
  isKing,
}: {
  primary: MinerSummary;
  compare: MinerSummary | null;
  isKing: boolean;
}) {
  const limiting = findLimitingAxis(primary);
  const finalScore = primary.composite?.final ?? null;
  const worst3 = primary.composite?.worst_3_mean ?? null;
  const worst = primary.composite?.worst ?? null;
  const weighted = primary.composite?.weighted ?? null;
  const presentCount = primary.composite?.present_count ?? null;
  const finalAlpha = primary.composite?.final_alpha ?? 0.7;

  return (
    <>
      {/* Inline explainer */}
      <div className="mb-6 px-4 py-3 border border-border bg-[var(--surface-soft)] text-[12px] leading-relaxed max-w-3xl">
        <div className="text-[10px] uppercase tracking-[0.18em] text-meta mb-1">
          How to read this — v30.2 scoring
        </div>
        <div className="text-foreground">
          Each spoke is a <strong>macro-axis</strong> rolling up the v30.2
          composite (skill groups + super_teacher + shadow axes). Outer
          ring = 1.0 (best), centre = 0 (worst). The amber polygon is{" "}
          {isKing ? <strong>the current king</strong> : "the selected miner"} (
          ♛ #{primary.uid}). The grey dashed polygon is the
          {compare ? " comparison miner" : " compare-target (pick one above to overlay)"}.
        </div>
        <div className="text-meta mt-2">
          The v30.2 ranking key is{" "}
          <strong className="text-foreground num">composite.final</strong>{" "}
          = {finalAlpha.toFixed(2)} × <code>worst_3_mean</code> +{" "}
          {(1 - finalAlpha).toFixed(2)} × <code>weighted</code>. For this
          miner: final = {finalScore != null ? <strong className="text-foreground num">{finalScore.toFixed(3)}</strong> : "—"}{" "}
          (worst_3_mean = {worst3 != null ? <span className="num">{worst3.toFixed(3)}</span> : "—"},{" "}
          weighted = {weighted != null ? <span className="num">{weighted.toFixed(3)}</span> : "—"}).
          Legacy <code>worst</code> (single-axis min) ={" "}
          {worst != null ? <span className="num">{worst.toFixed(3)}</span> : "—"}{" "}
          (kept for back-compat — no longer the gate).
        </div>
        {limiting && (
          <div className="text-meta mt-2">
            Lowest axis:{" "}
            <strong className="text-foreground num">
              {prettyAxis(limiting.axis)} = {limiting.value.toFixed(3)}
            </strong>
            . Look at where the polygon dips — that&apos;s where this
            miner needs the most work.
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-10 items-start">
        <RadialChart primary={primary} compare={compare} />
        <RadialLegend
          primary={primary}
          compare={compare}
          isKing={isKing}
          worst={worst}
          weighted={weighted}
          presentCount={presentCount}
          limitingAxis={limiting?.axis ?? null}
        />
      </div>
    </>
  );
}

function RadialChart({
  primary,
  compare,
}: {
  primary: MinerSummary;
  compare: MinerSummary | null;
}) {
  const W = 720;
  const H = 720;
  const cx = W / 2;
  const cy = H / 2;
  const R = 250;

  const N = MACRO_AXES.length;
  const angle = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / N;
  const ringPath = (frac: number): string => {
    const r = R * frac;
    const pts = MACRO_AXES.map((_, i) => {
      const a = angle(i);
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number];
    });
    return (
      pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ") +
      " Z"
    );
  };
  const polygonPath = (vals: number[]): string => {
    const pts = vals.map((v, i) => {
      const a = angle(i);
      const r = R * Math.max(0, Math.min(1, v));
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number];
    });
    return (
      pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ") +
      " Z"
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
      {/* Concentric reference rings */}
      {[0.25, 0.5, 0.75, 1].map((frac) => (
        <path
          key={frac}
          d={ringPath(frac)}
          fill="none"
          stroke="var(--border)"
          strokeWidth={1}
        />
      ))}
      {/* Ring labels (0.25 / 0.5 / 0.75 / 1.0) */}
      {[0.25, 0.5, 0.75, 1].map((frac) => (
        <text
          key={`ring-${frac}`}
          x={cx + 6}
          y={cy - R * frac + 4}
          fontFamily="Inter, sans-serif"
          fontSize={9}
          fill="var(--track-fill-soft)"
        >
          {frac.toFixed(2)}
        </text>
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
            stroke="var(--border)"
            strokeWidth={1}
          />
        );
      })}

      {/* Compare polygon (drawn first so primary overlays it) */}
      {compareVals && (
        <>
          <path
            d={polygonPath(compareVals)}
            fill="rgba(127,127,127,0.18)"
            stroke="var(--ink-meta)"
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
                r={3.25}
                fill="var(--ink-meta)"
              />
            );
          })}
        </>
      )}

      {/* Primary polygon */}
      <path
        d={polygonPath(primaryVals)}
        fill="var(--accent-amber-fill)"
        stroke="var(--accent-amber)"
        strokeWidth={1.85}
      />
      {primaryVals.map((v, i) => {
        const a = angle(i);
        const r = R * Math.max(0, Math.min(1, v));
        return (
          <circle
            key={`pc-${i}`}
            cx={cx + r * Math.cos(a)}
            cy={cy + r * Math.sin(a)}
            r={4.5}
            fill="var(--accent-amber)"
          />
        );
      })}

      {/* Macro-axis labels (with values inline on the axis tip) */}
      {MACRO_AXES.map((m, i) => {
        const a = angle(i);
        const lr = R + 28;
        const x = cx + lr * Math.cos(a);
        const y = cy + lr * Math.sin(a);
        let anchor: "start" | "middle" | "end" = "middle";
        const cosA = Math.cos(a);
        if (cosA > 0.2) anchor = "start";
        else if (cosA < -0.2) anchor = "end";
        const v = primaryVals[i];
        return (
          <g key={`l-${i}`}>
            <text
              x={x}
              y={y - 2}
              textAnchor={anchor}
              fontFamily="Inter, sans-serif"
              fontSize={11}
              letterSpacing={2.2}
              fill="var(--ink)"
            >
              <title>{m.description}</title>
              {m.label}
            </text>
            <text
              x={x}
              y={y + 13}
              textAnchor={anchor}
              fontFamily="Inter, sans-serif"
              fontSize={11}
              fill="var(--accent-amber)"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {v.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function RadialLegend({
  primary,
  compare,
  isKing,
  worst,
  weighted,
  presentCount,
  limitingAxis,
}: {
  primary: MinerSummary;
  compare: MinerSummary | null;
  isKing: boolean;
  worst: number | null;
  weighted: number | null;
  presentCount: number | null;
  limitingAxis: string | null;
}) {
  const finalScore = primary.composite?.final ?? null;
  const worst3 = primary.composite?.worst_3_mean ?? null;
  const compFinal = compare?.composite?.final ?? null;
  const compWorst3 = compare?.composite?.worst_3_mean ?? null;
  const compWorst = compare?.composite?.worst ?? null;
  const compWeighted = compare?.composite?.weighted ?? null;
  return (
    <div className="space-y-5">
      {/* Primary card */}
      <div>
        <div className="flex items-baseline gap-2 mb-2">
          <span
            className="inline-block w-3 h-3"
            style={{ backgroundColor: "#ce8d43" }}
          />
          <span className="text-[14px] font-medium">
            {isKing && "♛ "}#{primary.uid}
          </span>
          <span className="text-[11px] text-meta truncate" title={primary.model}>
            {primary.model}
          </span>
        </div>
        <dl className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-1 text-[12px] num">
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            final
          </dt>
          <dd className="font-medium">
            {finalScore != null ? finalScore.toFixed(3) : "—"}{" "}
            <span className="text-meta text-[10px]">v30.2 ranking key</span>
          </dd>
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            worst_3
          </dt>
          <dd>
            {worst3 != null ? worst3.toFixed(3) : "—"}{" "}
            <span className="text-meta text-[10px]">mean of 3 lowest</span>
          </dd>
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            weighted
          </dt>
          <dd>{weighted != null ? weighted.toFixed(3) : "—"}</dd>
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            worst
          </dt>
          <dd className="text-meta">
            {worst != null ? worst.toFixed(3) : "—"}{" "}
            <span className="text-[10px]">legacy single-min</span>
          </dd>
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            limit
          </dt>
          <dd>
            {limitingAxis ? (
              <span title={`lowest axis: ${limitingAxis}`}>
                {prettyAxis(limitingAxis)}
              </span>
            ) : (
              "—"
            )}
          </dd>
          <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
            axes
          </dt>
          <dd>{presentCount != null ? `${presentCount} present` : "—"}</dd>
        </dl>
      </div>

      {/* Compare card */}
      {compare && (
        <div className="pt-4 border-t border-border">
          <div className="flex items-baseline gap-2 mb-2">
            <span
              className="inline-block w-3 h-3 border-2 border-dashed"
              style={{ borderColor: "#888" }}
            />
            <span className="text-[14px] font-medium">#{compare.uid}</span>
            <span className="text-[11px] text-meta truncate" title={compare.model}>
              {compare.model}
            </span>
          </div>
          <dl className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-1 text-[12px] num">
            <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
              final
            </dt>
            <dd className="font-medium">
              {compFinal != null ? compFinal.toFixed(3) : "—"}
            </dd>
            <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
              worst_3
            </dt>
            <dd>{compWorst3 != null ? compWorst3.toFixed(3) : "—"}</dd>
            <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
              weighted
            </dt>
            <dd>{compWeighted != null ? compWeighted.toFixed(3) : "—"}</dd>
            <dt className="text-meta uppercase tracking-[0.14em] text-[10px]">
              worst
            </dt>
            <dd className="text-meta">
              {compWorst != null ? compWorst.toFixed(3) : "—"}
            </dd>
          </dl>
        </div>
      )}

      <p className="text-[10px] text-meta leading-relaxed pt-3 border-t border-border">
        Macro-axis values are <strong className="text-foreground">means</strong>{" "}
        of their constituent v30.2 sub-axes. The v30.2 ranker uses{" "}
        <code>composite.final</code> = α·worst_3_mean + (1−α)·weighted
        — see Scores tab for per-axis breakdown.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Scores sub-tab — full 17-axis breakdown
// ────────────────────────────────────────────────────────────────────

/**
 * v30.2/v30.3 full axis breakdown organised by ranking-relevance:
 * - SKILL GROUPS (the v30.2 ranking drivers — sub-axes feed these but
 *   don't directly drive ranking)
 * - SUPER-TEACHER (v30.2 — incentivises beyond-teacher capability)
 * - TEACHER-SIMILARITY (production + research-paper shadow signals)
 * - QUALITY (judge probes, chat coherence)
 * - DISCIPLINE (length/degeneracy/density)
 * - STAND-ALONE CAPABILITY (kept separate from groups)
 * - SUB-AXES (still computed for telemetry — no direct ranking weight)
 */
const ALL_AXES: { key: string; group: string }[] = [
  // Skill groups (v30.2)
  { key: "code_skill_group", group: "Skill Groups" },
  { key: "math_skill_group", group: "Skill Groups" },
  { key: "reasoning_skill_group", group: "Skill Groups" },
  { key: "knowledge_skill_group", group: "Skill Groups" },
  // Super-teacher
  { key: "super_teacher", group: "Beyond Teacher" },
  // Teacher-similarity (live ranking axes)
  { key: "on_policy_rkl", group: "Teacher-Similarity" },
  { key: "kl", group: "Teacher-Similarity" },
  { key: "top_k_overlap", group: "Teacher-Similarity" },
  { key: "capability", group: "Teacher-Similarity" },
  // Teacher-similarity (shadow axes — research-validated)
  { key: "kl_is", group: "Shadow Distillation Axes" },
  { key: "forking_rkl", group: "Shadow Distillation Axes" },
  { key: "teacher_trace_plausibility", group: "Shadow Distillation Axes" },
  { key: "entropy_aware_kl", group: "Shadow Distillation Axes" },
  { key: "tail_decoupled_kl", group: "Shadow Distillation Axes" },
  // Quality
  { key: "judge_probe", group: "Quality" },
  { key: "long_form_judge", group: "Quality" },
  { key: "chat_turns_probe", group: "Quality" },
  // Discipline
  { key: "length", group: "Discipline" },
  { key: "degeneracy", group: "Discipline" },
  { key: "reasoning_density", group: "Discipline" },
  // Stand-alone capability
  { key: "tool_use_bench", group: "Capability (Stand-Alone)" },
  { key: "ifeval_bench", group: "Capability (Stand-Alone)" },
  { key: "calibration_bench", group: "Capability (Stand-Alone)" },
  // Sub-axes (telemetry — fold into groups for ranking)
  { key: "math_bench", group: "Sub-Axes (Telemetry)" },
  { key: "aime_bench", group: "Sub-Axes (Telemetry)" },
  { key: "robustness_bench", group: "Sub-Axes (Telemetry)" },
  { key: "code_bench", group: "Sub-Axes (Telemetry)" },
  { key: "mbpp_bench", group: "Sub-Axes (Telemetry)" },
  { key: "debug_bench", group: "Sub-Axes (Telemetry)" },
  { key: "correction_bench", group: "Sub-Axes (Telemetry)" },
  { key: "refactor_bench", group: "Sub-Axes (Telemetry)" },
  { key: "reasoning_bench", group: "Sub-Axes (Telemetry)" },
  { key: "multi_doc_synthesis_bench", group: "Sub-Axes (Telemetry)" },
  { key: "long_context_bench", group: "Sub-Axes (Telemetry)" },
  { key: "knowledge_bench", group: "Sub-Axes (Telemetry)" },
  { key: "pragmatic_bench", group: "Sub-Axes (Telemetry)" },
];

function ScoresView({
  primary,
  compare,
  isKing,
}: {
  primary: MinerSummary;
  compare: MinerSummary | null;
  isKing: boolean;
}) {
  const pa = primary.composite?.axes ?? {};
  const ca = compare?.composite?.axes ?? {};
  const broken = new Set(primary.composite?.broken_axes ?? []);
  const limiting = findLimitingAxis(primary)?.axis;
  return (
    <>
      <div className="mb-6 max-w-3xl text-[12px] text-meta leading-relaxed">
        Full v30.2/v30.3 composite axis breakdown for{" "}
        <strong className="text-foreground">
          {isKing && "♛ "}#{primary.uid}
        </strong>
        . The amber bar = primary; the grey marker = compare. The
        ranking key is{" "}
        <code>composite.final = 0.7 × worst_3_mean + 0.3 × weighted</code>{" "}
        — see the <strong>Skill Groups</strong> section first (those
        are the ranking-relevant axes), then{" "}
        <strong>Beyond Teacher</strong> (super_teacher), then the rest.
      </div>
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
                  isLimiting={limiting === key}
                  isBroken={broken.has(key)}
                />
              );
            })}
          </div>
        ))}
      </div>
    </>
  );
}

function ScoreRow({
  axis,
  primary,
  compare,
  hasCompare,
  isLimiting,
  isBroken,
}: {
  axis: string;
  primary: number | null;
  compare: number | null;
  hasCompare: boolean;
  isLimiting: boolean;
  isBroken: boolean;
}) {
  const label = axis.replace(/_bench$/, "").replace(/_/g, " ");
  return (
    <div
      className={[
        "grid grid-cols-[140px_1fr_50px_50px] items-center gap-3 py-1.5 text-[12px] num",
        isBroken ? "opacity-50" : "",
        isLimiting ? "border-l-2 border-foreground pl-2 -ml-2" : "",
      ].join(" ")}
      title={
        isBroken
          ? "axis dropped this round (eval-broken)"
          : isLimiting
            ? "single lowest axis — counts in worst_3_mean (and legacy composite.worst)"
            : undefined
      }
    >
      <div className="text-foreground capitalize truncate" title={axis}>
        {label}
        {isLimiting && <span className="text-meta ml-1.5 text-[10px]">← limit</span>}
      </div>
      <div className="w-full h-1.5 bg-[var(--track)] relative">
        {primary != null && (
          <div
            className="absolute inset-y-0 left-0 bg-[#ce8d43]"
            style={{ width: `${Math.max(0, Math.min(1, primary)) * 100}%` }}
          />
        )}
        {hasCompare && compare != null && (
          <div
            className="absolute -top-0.5 -bottom-0.5 w-px bg-[#666]"
            style={{
              left: `${Math.max(0, Math.min(1, compare)) * 100}%`,
            }}
          />
        )}
      </div>
      <span className="text-right">
        {primary != null ? primary.toFixed(3) : "—"}
      </span>
      {hasCompare && (
        <span className="text-right text-meta">
          {compare != null ? compare.toFixed(3) : "—"}
        </span>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Samples sub-tab (placeholder until pod-side sample export lands)
// ────────────────────────────────────────────────────────────────────

function SamplesView({ primary }: { primary: MinerSummary }) {
  return (
    <div className="max-w-2xl space-y-4 text-[13px] text-foreground leading-relaxed">
      <p className="text-[11px] uppercase tracking-[0.18em] text-meta font-medium">
        Coming soon
      </p>
      <p>
        Per-prompt sample grid (teacher continuation vs student continuation,
        per-axis scoring detail, link to raw eval-data file). The validator
        already records this on the pod; surfacing it through the API is on
        the roadmap.
      </p>
      <p className="text-meta text-[12px]">
        Currently selected: #{primary.uid} · {primary.model}.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
//  Benchmarks sub-tab
// ────────────────────────────────────────────────────────────────────

function BenchmarksView({ primary }: { primary: MinerSummary }) {
  return (
    <div className="max-w-3xl space-y-4 text-[13px] leading-relaxed">
      <p>
        For the held-out auto-bench numbers (GSM8K, HumanEval, IFEval, BBH,
        MMLU-Pro, ARC) on the king + teacher + reference, see the{" "}
        <a href="#bench" className="underline">
          Bench tab
        </a>
        .
      </p>
      <p className="text-meta text-[12px]">
        Those benches are <strong className="text-foreground">not</strong>{" "}
        used by the validator&apos;s composite. The composite&apos;s bench
        axes (math_bench, code_bench, etc.) are generated procedurally per
        round from the block-seed and never use public datasets. See{" "}
        <code>paper/benchmarks_as_north_star.md</code>.
      </p>
      <p className="text-meta text-[12px]">
        Currently selected: #{primary.uid} · {primary.model}.
      </p>
    </div>
  );
}
