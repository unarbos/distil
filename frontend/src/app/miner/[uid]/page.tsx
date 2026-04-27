import Link from "next/link";
import { API_BASE } from "@/lib/subnet";

export const dynamic = "force-dynamic";

type AxisMap = Record<string, number | null>;

type MinerDetail = {
  uid: number;
  model?: string | null;
  hotkey?: string | null;
  kl_score?: number | null;
  disqualified?: string | null;
  is_king?: boolean;
  eval_status?: { status?: string; reason?: string };
  composite?: {
    worst?: number | null;
    weighted?: number | null;
    present_count?: number;
    round_block?: number;
    is_latest_round?: boolean;
    axes?: AxisMap;
    broken_axes?: string[];
  } | null;
  king_health?: Record<string, unknown>;
  h2h_history?: Array<{
    block?: number;
    kl?: number | null;
    is_king?: boolean;
    king_changed?: boolean;
    type?: string;
  }>;
};

/**
 * Per-axis weights (production v28). Mirrors composite.py:AXIS_WEIGHTS
 * + BENCH_AXIS_WEIGHTS + ARENA_V3_AXIS_WEIGHTS + the probe weights.
 *
 * Used to render the per-axis bar in the right scale (a 0.5 on `kl`
 * is much more weight than a 0.5 on `tool_use_bench`). Source of
 * truth: scripts/validator/composite.py — keep in sync.
 */
const AXIS_WEIGHTS: Record<string, number> = {
  // Tier 1 relative
  on_policy_rkl: 0.35,
  capability: 0.25,
  judge_probe: 0.15,
  kl: 0.15,
  degeneracy: 0.15,
  length: 0.10,
  // Bench
  math_bench: 0.14,
  code_bench: 0.14,
  reasoning_bench: 0.10,
  ifeval_bench: 0.07,
  // Arena V3
  aime_bench: 0.10,
  mbpp_bench: 0.08,
  chat_turns_probe: 0.08,
  tool_use_bench: 0.06,
  robustness_bench: 0.07,
  long_context_bench: 0.04,
  reasoning_density: 0.05,
};

const AXIS_GROUPS: Array<{ label: string; axes: string[] }> = [
  {
    label: "Distribution match",
    axes: ["on_policy_rkl", "kl", "capability", "length", "degeneracy"],
  },
  {
    label: "Capability vs ground truth",
    axes: [
      "math_bench",
      "code_bench",
      "reasoning_bench",
      "ifeval_bench",
      "aime_bench",
      "mbpp_bench",
      "tool_use_bench",
      "long_context_bench",
      "robustness_bench",
    ],
  },
  {
    label: "Conversational + discipline",
    axes: ["judge_probe", "chat_turns_probe", "reasoning_density"],
  },
];

function fmt(v: unknown, digits = 4) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

async function fetchMiner(uid: string): Promise<MinerDetail | null> {
  const res = await fetch(`${API_BASE}/api/miner/${uid}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export default async function MinerPage({
  params,
}: {
  params: Promise<{ uid: string }>;
}) {
  const { uid } = await params;
  const miner = await fetchMiner(uid);
  if (!miner) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-8 space-y-4">
        <Link href="/" className="text-[12px] underline">
          ← Back to dashboard
        </Link>
        <h1 className="text-2xl font-medium">Miner UID {uid}</h1>
        <p className="text-sm text-meta">No miner details found.</p>
      </main>
    );
  }

  const axes: AxisMap = miner.composite?.axes || {};
  const broken = new Set(miner.composite?.broken_axes || []);
  const axisEntries = Object.entries(axes).filter(
    ([, v]) => typeof v === "number" && Number.isFinite(v as number),
  ) as Array<[string, number]>;

  // Find limiting axis: lowest non-broken axis. This is the axis that
  // determines composite.worst — the ranking key.
  const nonBroken = axisEntries.filter(([k]) => !broken.has(k));
  const limiting = nonBroken.length
    ? nonBroken.reduce((a, b) => (a[1] < b[1] ? a : b))
    : null;

  return (
    <main className="mx-auto max-w-5xl px-6 sm:px-8 py-10 space-y-8">
      {/* Plain <a>: hash routing is purely client-side, no need for
          Next.js prefetch (which can RSC-prefetch the home route on
          hover and was historically associated with a stray
          ERR_TOO_MANY_REDIRECTS in the browser console). */}
      <a
        href="/#miners"
        className="text-[11px] uppercase tracking-[0.18em] text-meta hover:text-foreground"
      >
        ← Back to leaderboard
      </a>

      {/* Identity card */}
      <section className="space-y-2">
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-[44px] font-medium tracking-[-0.04em] leading-none">
            UID {miner.uid}
          </h1>
          {miner.is_king && (
            <span className="text-[10px] uppercase tracking-[0.18em] border border-foreground px-2 py-1">
              ♛ king
            </span>
          )}
          {miner.disqualified && (
            <span className="text-[10px] uppercase tracking-[0.18em] border border-danger text-danger px-2 py-1">
              DQ
            </span>
          )}
        </div>
        <div className="font-mono text-[14px] text-foreground break-all">
          {miner.model || "model unknown"}
        </div>
        <div className="text-[11px] text-meta break-all">
          hotkey: {miner.hotkey || "—"}
        </div>
      </section>

      {/* Headline metrics — composite first */}
      <section className="grid gap-0 grid-cols-2 sm:grid-cols-4 border-y border-border">
        <Stat
          label="Composite worst"
          big={fmt(miner.composite?.worst, 3)}
          small="ranking key"
        />
        <Stat
          label="Weighted mean"
          big={fmt(miner.composite?.weighted, 3)}
          small="Σ wᵢ·axisᵢ / Σ wᵢ"
        />
        <Stat
          label="Limiting axis"
          big={limiting ? prettyAxis(limiting[0]) : "—"}
          small={limiting ? `score ${limiting[1].toFixed(3)}` : "—"}
        />
        <Stat
          label="KL axis"
          big={fmt(axes.kl ?? miner.kl_score, 4)}
          small="1 of 17"
          last
        />
      </section>

      {miner.eval_status?.reason && (
        <p className="text-sm text-meta">{miner.eval_status.reason}</p>
      )}

      {miner.disqualified && (
        <div className="text-sm text-danger border-l-2 border-danger pl-3">
          {miner.disqualified}
        </div>
      )}

      {/* Per-axis breakdown grouped by concern */}
      <section className="space-y-6">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
            Per-axis composite breakdown
          </h2>
          <div className="text-[11px] text-meta num">
            round{" "}
            {miner.composite?.round_block ? `#${miner.composite.round_block}` : "—"}
            {miner.composite?.is_latest_round === false ? " · last known" : ""}
            {miner.composite?.present_count != null && (
              <>
                {" · "}
                {miner.composite.present_count} axes present
              </>
            )}
          </div>
        </div>

        {axisEntries.length === 0 ? (
          <p className="text-sm text-meta">
            No composite record yet. This miner hasn&apos;t been evaluated under
            the v28 schema; once they&apos;re scored their per-axis breakdown
            will appear here.
          </p>
        ) : (
          <div className="space-y-6">
            {AXIS_GROUPS.map((g) => (
              <AxisGroup
                key={g.label}
                label={g.label}
                axes={g.axes}
                values={axes}
                broken={broken}
                isLimiting={limiting?.[0]}
              />
            ))}
          </div>
        )}

        {broken.size > 0 && (
          <p className="text-[11px] text-meta leading-relaxed max-w-3xl">
            <strong className="text-foreground">{broken.size}</strong>{" "}
            axis{broken.size === 1 ? "" : "es"} dropped this round (faded):
            either the teacher itself scored below the sanity floor, or the
            reference base model scored <code>pass_frac=0</code> (eval-setup
            truncation / unsolvable items, not student skill). Dropped axes
            do not contribute to the worst/weighted ranking.
          </p>
        )}
      </section>

      {/* H2H tail */}
      {(miner.h2h_history?.length ?? 0) > 0 && (
        <section className="space-y-3">
          <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
            Recent rounds
          </h2>
          <div className="border-t border-border">
            {(miner.h2h_history || []).map((row, i) => (
              <div
                key={`${row.block}-${i}`}
                className="grid grid-cols-[80px_80px_1fr_auto] gap-3 items-center py-2 border-b border-border text-[12px] num"
              >
                <span className="text-meta">block</span>
                <span>{row.block ?? "—"}</span>
                <span className="text-meta">
                  {row.is_king ? "king" : "challenger"}
                  {row.king_changed && " · king changed"}
                </span>
                <span title="KL axis only — one of 17. Composite.worst is the ranking key.">
                  KL <span className="text-meta">(1/17)</span>{" "}
                  <strong>{fmt(row.kl, 4)}</strong>
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* King health (raw, for the king's drill-down) */}
      {miner.king_health && Object.keys(miner.king_health).length > 0 && (
        <section className="space-y-3">
          <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
            King health
          </h2>
          <pre className="font-mono text-[11px] text-meta overflow-x-auto bg-[var(--surface-soft)] border border-border px-4 py-3">
            {JSON.stringify(miner.king_health, null, 2)}
          </pre>
        </section>
      )}
    </main>
  );
}

function Stat({
  label,
  big,
  small,
  last,
}: {
  label: string;
  big: string;
  small?: string;
  last?: boolean;
}) {
  return (
    <div
      className={[
        "px-4 sm:px-6 py-5",
        last ? "" : "border-r border-border",
      ].join(" ")}
    >
      <div className="text-[10px] uppercase tracking-[0.16em] text-meta">{label}</div>
      <div className="text-[24px] font-medium tracking-[-0.025em] num mt-1 leading-none">
        {big}
      </div>
      {small && (
        <div className="text-[10px] text-meta uppercase tracking-[0.14em] mt-1">
          {small}
        </div>
      )}
    </div>
  );
}

function AxisGroup({
  label,
  axes,
  values,
  broken,
  isLimiting,
}: {
  label: string;
  axes: string[];
  values: AxisMap;
  broken: Set<string>;
  isLimiting?: string;
}) {
  const present = axes.filter(
    (k) => typeof values[k] === "number" && Number.isFinite(values[k] as number),
  );
  if (present.length === 0) return null;
  return (
    <div>
      <div className="text-[11px] text-meta mb-2">{label}</div>
      <div className="space-y-1.5">
        {present.map((axis) => {
          const v = values[axis] as number;
          const isBroken = broken.has(axis);
          const isLim = isLimiting === axis;
          const w = AXIS_WEIGHTS[axis] ?? 0;
          return (
            <div
              key={axis}
              className={[
                "grid grid-cols-[180px_1fr_56px_56px] items-center gap-3 text-[12px] num py-1.5",
                isBroken ? "opacity-50" : "",
                isLim ? "border-l-2 border-foreground pl-2 -ml-2" : "",
              ].join(" ")}
              title={
                isBroken
                  ? "axis dropped this round (eval-broken)"
                  : isLim
                    ? "limiting axis — this determines composite.worst"
                    : undefined
              }
            >
              <span className={isBroken ? "line-through text-meta" : ""}>
                {prettyAxis(axis)}
                {isLim && <span className="text-meta ml-1.5">← limit</span>}
              </span>
              <div className="h-1.5 bg-[#f1f1f1] relative">
                <div
                  className={[
                    "absolute inset-y-0 left-0",
                    isLim ? "bg-foreground" : "bg-[#bdbdbd]",
                  ].join(" ")}
                  style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}
                />
              </div>
              <span className={["text-right", isBroken ? "line-through" : ""].join(" ")}>
                {v.toFixed(3)}
              </span>
              <span className="text-right text-meta text-[10px]">
                w={w.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function prettyAxis(s: string): string {
  return s.replace(/_bench$/, "").replace(/_/g, " ");
}
