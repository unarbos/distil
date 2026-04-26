import {
  fetchMetagraph,
  fetchCommitments,
  fetchScores,
  fetchPrice,
  fetchAllModelInfo,
  fetchHistory,
  fetchH2hLatest,
  buildMinerList,
} from "@/lib/api";
import { AutoRefresh } from "@/components/auto-refresh";
import { DashboardTabs } from "@/components/dashboard-tabs";
import { SCORE_TO_BEAT_FACTOR } from "@/lib/subnet";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [metagraph, commitments, scores, price, history, h2hLatest] = await Promise.all([
    fetchMetagraph(),
    fetchCommitments(),
    fetchScores(),
    fetchPrice(),
    fetchHistory(),
    fetchH2hLatest(),
  ]);

  const kingUid = h2hLatest?.king_uid ?? null;
  const miners = buildMinerList(metagraph, commitments, scores, kingUid, h2hLatest);
  const modelInfoMap = await fetchAllModelInfo(miners.map((m) => m.model));
  const currentBlock = metagraph?.block ?? 0;
  const alphaPrice = price?.alpha_price_tao ?? 0;
  const alphaPriceUsd = price?.alpha_price_usd ?? 0;
  const taoUsd = price?.tao_usd ?? 0;
  const change24h = price?.price_change_24h ?? 0;
  const minersTaoDay = price?.miners_tao_per_day ?? 0;

  const king = miners.find((m) => m.isWinner);
  const kingH2hKl = h2hLatest?.king_h2h_kl ?? king?.klScore ?? null;
  const scoreToBeat = kingH2hKl != null ? kingH2hKl * SCORE_TO_BEAT_FACTOR : null;

  return (
    <div className="relative min-h-[calc(100vh-3rem)]">
      <AutoRefresh intervalMs={15000} />
      <div className="mx-auto max-w-7xl px-3 py-2 sm:px-4 space-y-2 relative">
        {/* Compact header: title + stats + price */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
              💧 Distil
            </h1>
            <span className="text-[10px] font-mono text-blue-400/70 border border-blue-400/20 rounded-md px-1.5 py-0.5 bg-blue-400/5">
              SN97
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono text-muted-foreground">
            {currentBlock > 0 && <span>#{currentBlock.toLocaleString()}</span>}
            <span>{miners.length} models</span>
            {king && <span className="text-yellow-400/80">👑 UID {king.uid}</span>}
            {scoreToBeat != null && (
              <span className="rounded-full bg-orange-400/10 border border-orange-400/20 px-2 py-0.5 text-orange-400">
                Beat: &lt;{scoreToBeat.toFixed(4)} KL
              </span>
            )}
          </div>

          {alphaPrice > 0 && (
            <div className="flex items-center gap-3 ml-auto rounded-lg border border-border/50 bg-card/30 backdrop-blur-md px-3 py-1.5">
              <div className="text-right">
                <div className="text-lg font-bold font-mono tabular-nums tracking-tight leading-tight">
                  ${Number.isFinite(alphaPriceUsd) ? alphaPriceUsd.toFixed(2) : "—"}
                </div>
                <div className="text-[10px] text-muted-foreground font-mono leading-tight">
                  {Number.isFinite(alphaPrice) ? alphaPrice.toFixed(4) : "—"}τ/{price?.symbol ?? "α"}
                </div>
              </div>
              <div className="h-6 w-px bg-border/50" />
              <div className="flex flex-col items-end gap-0 text-[10px] font-mono">
                <span className={change24h >= 0 ? "text-emerald-400" : "text-red-400"}>
                  {change24h >= 0 ? "+" : ""}{Number.isFinite(change24h) ? change24h.toFixed(1) : "0.0"}% 24h
                </span>
                {minersTaoDay > 0 && (
                  <span className="text-muted-foreground">{Number.isFinite(minersTaoDay) ? minersTaoDay.toFixed(0) : "0"}τ/day</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Tabs */}
        <DashboardTabs
          miners={miners}
          scores={scores}
          modelInfoMap={modelInfoMap}
          currentBlock={currentBlock}
          taoUsd={taoUsd}
          minersTaoDay={minersTaoDay}
          history={history}
          kingUid={kingUid}
          kingH2hKl={kingH2hKl}
        />
      </div>
    </div>
  );
}
