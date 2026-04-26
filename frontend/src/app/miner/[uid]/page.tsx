import Link from "next/link";
import { API_BASE } from "@/lib/subnet";

export const dynamic = "force-dynamic";

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
    axes?: Record<string, number | null>;
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

function fmt(v: unknown, digits = 4) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

async function fetchMiner(uid: string): Promise<MinerDetail | null> {
  const res = await fetch(`${API_BASE}/api/miner/${uid}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export default async function MinerPage({ params }: { params: Promise<{ uid: string }> }) {
  const { uid } = await params;
  const miner = await fetchMiner(uid);
  if (!miner) {
    return (
      <main className="mx-auto max-w-4xl p-6 space-y-4">
        <Link href="/" className="text-xs text-blue-400 hover:underline">Back to dashboard</Link>
        <h1 className="text-xl font-semibold">Miner UID {uid}</h1>
        <p className="text-sm text-muted-foreground">No miner details found.</p>
      </main>
    );
  }
  const axes = miner.composite?.axes || {};
  const broken = new Set(miner.composite?.broken_axes || []);
  const axisEntries = Object.entries(axes).filter(([, v]) => v != null);
  axisEntries.sort((a, b) => (a[1] ?? 0) - (b[1] ?? 0));

  return (
    <main className="mx-auto max-w-5xl p-4 sm:p-6 space-y-4">
      <Link href="/" className="text-xs text-blue-400 hover:underline">Back to dashboard</Link>
      <section className="rounded-xl border border-border/30 bg-card/20 p-4 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold">UID {miner.uid}</h1>
          {miner.is_king && <span className="rounded border border-yellow-400/25 bg-yellow-400/10 px-2 py-0.5 text-xs text-yellow-300">KING</span>}
          {miner.disqualified && <span className="rounded border border-red-400/25 bg-red-400/10 px-2 py-0.5 text-xs text-red-300">DQ</span>}
        </div>
        <div className="font-mono text-sm text-blue-300 break-all">{miner.model || "model unknown"}</div>
        <div className="text-xs text-muted-foreground break-all">hotkey: {miner.hotkey || "—"}</div>
      </section>

      <section className="grid gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-border/25 bg-card/15 p-3">
          <div className="text-[10px] uppercase text-muted-foreground">KL score</div>
          <div className="font-mono text-lg">{fmt(miner.kl_score)}</div>
        </div>
        <div className="rounded-lg border border-border/25 bg-card/15 p-3">
          <div className="text-[10px] uppercase text-muted-foreground">Composite worst</div>
          <div className="font-mono text-lg">{fmt(miner.composite?.worst, 3)}</div>
        </div>
        <div className="rounded-lg border border-border/25 bg-card/15 p-3">
          <div className="text-[10px] uppercase text-muted-foreground">Weighted</div>
          <div className="font-mono text-lg">{fmt(miner.composite?.weighted, 3)}</div>
        </div>
        <div className="rounded-lg border border-border/25 bg-card/15 p-3">
          <div className="text-[10px] uppercase text-muted-foreground">Eval status</div>
          <div className="font-mono text-sm">{miner.eval_status?.status || "—"}</div>
        </div>
      </section>

      {miner.eval_status?.reason && (
        <section className="rounded-lg border border-border/25 bg-card/10 p-3 text-sm text-muted-foreground">
          {miner.eval_status.reason}
        </section>
      )}

      {miner.disqualified && (
        <section className="rounded-lg border border-red-400/25 bg-red-400/[0.03] p-3 text-sm text-red-300">
          {miner.disqualified}
        </section>
      )}

      <section className="rounded-xl border border-border/30 bg-card/10 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold">Composite Axes</h2>
          <div className="text-xs text-muted-foreground">
            round {miner.composite?.round_block ?? "—"}{miner.composite?.is_latest_round === false ? " · last known" : ""}
          </div>
        </div>
        {axisEntries.length ? (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {axisEntries.map(([axis, value]) => {
              const isBroken = broken.has(axis);
              return (
                <div
                  key={axis}
                  title={isBroken ? "eval-broken this round (teacher or reference base scored below the sanity floor) — dropped from worst/weighted ranking" : undefined}
                  className={`flex items-center justify-between rounded border border-border/20 px-3 py-2 text-xs ${isBroken ? "opacity-60" : ""}`}
                >
                  <span className={`font-mono text-muted-foreground ${isBroken ? "line-through" : ""}`}>{axis}</span>
                  <span className={`font-mono ${isBroken ? "line-through" : ""}`}>{fmt(value, 3)}</span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No composite axes recorded yet.</p>
        )}
        {broken.size > 0 && (
          <p className="text-[11px] text-muted-foreground/70">
            <span className="font-mono">{broken.size}</span> axis{broken.size === 1 ? "" : "es"} dropped this round (struck-through):
            either the teacher itself fell below the sanity floor, or the reference base model scored
            <span className="font-mono"> pass_frac=0</span> on a bench axis (eval-setup truncation /
            unsolvable items, not student skill). These do not contribute to ranking.
          </p>
        )}
      </section>

      {miner.king_health && (
        <section className="rounded-xl border border-amber-400/25 bg-amber-400/[0.03] p-4 space-y-2">
          <h2 className="font-semibold text-amber-300">King Health</h2>
          <pre className="overflow-x-auto text-xs text-muted-foreground">{JSON.stringify(miner.king_health, null, 2)}</pre>
        </section>
      )}

      <section className="rounded-xl border border-border/30 bg-card/10 p-4 space-y-2">
        <h2 className="font-semibold">Recent H2H</h2>
        <div className="space-y-1">
          {(miner.h2h_history || []).map((row) => (
            <div key={`${row.block}-${row.kl}`} className="grid grid-cols-4 gap-2 rounded border border-border/10 px-3 py-2 text-xs">
              <span>block {row.block ?? "—"}</span>
              <span>KL {fmt(row.kl)}</span>
              <span>{row.is_king ? "king" : "challenger"}</span>
              <span>{row.king_changed ? "king changed" : row.type || "h2h"}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
