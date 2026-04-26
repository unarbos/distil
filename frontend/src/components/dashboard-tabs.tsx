"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";
import type { MinerEntry, ModelInfo, ScoresResponse, ScoreHistoryEntry } from "@/lib/api";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { MinersTab } from "@/components/miners-tab";
import { H2hHistory } from "@/components/h2h-round-table";
import { GpuLogs } from "@/components/gpu-logs";
import { EvalProgressBar } from "@/components/eval-progress";
import { ScoreTrend } from "@/components/score-trend";
import { ValidatorStatus } from "@/components/validator-status";
import { BenchmarksTab } from "@/components/benchmarks-tab";
import { TelemetryTab } from "@/components/telemetry-tab";
import { KingHistory } from "@/components/king-history";
import { SCORE_TO_BEAT_FACTOR } from "@/lib/subnet";
import { shortRevision } from "@/lib/utils";

interface DashboardTabsProps {
  miners: MinerEntry[];
  scores: ScoresResponse | null;
  modelInfoMap: Record<string, ModelInfo>;
  currentBlock: number;
  taoUsd: number;
  minersTaoDay: number;
  history: ScoreHistoryEntry[];
  kingUid: number | null;
  kingH2hKl: number | null;
}

const TABS = [
  { id: "live", label: "Live", hint: "king · eval · GPU" },
  { id: "rounds", label: "Rounds", hint: "H2H · dethronements" },
  { id: "axes", label: "Axes", hint: "Arena v3 · composite · pareto" },
  { id: "miners", label: "Miners", hint: "leaderboard" },
  { id: "benchmarks", label: "Benchmarks", hint: "king vs baseline" },
] as const;

const CHAT_URL = "https://chat.arbos.life";

interface Incident {
  ts: number;
  type: "issue" | "action";
  issue?: string;
  action?: string;
  resolved?: boolean;
}

function IncidentsPanel() {
  const [items, setItems] = useState<Incident[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/incidents?limit=20`, { cache: "no-store" });
        if (!res.ok) return;
        const json = await res.json();
        if (!cancelled && Array.isArray(json?.incidents)) setItems(json.incidents);
      } catch {}
    };
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (items.length === 0) return null;
  const unresolved = items.filter(i => i.type === "issue" && !i.resolved).length;

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm text-xs"
    >
      <summary className="cursor-pointer select-none px-4 py-2 flex items-center gap-2 text-muted-foreground">
        <span className={unresolved > 0 ? "text-warning" : "text-ok"}>
          {unresolved > 0 ? "!" : "OK"}
        </span>
        <span>Ops incidents ({items.length})</span>
        {unresolved > 0 && <span className="text-warning">{unresolved} unresolved</span>}
      </summary>
      <ul className="px-4 pb-3 space-y-1 font-mono max-h-64 overflow-auto">
        {items.map((ev, i) => {
          const when = new Date(ev.ts * 1000).toISOString().slice(5, 16).replace("T", " ");
          const label = ev.type === "issue" ? ev.issue : ev.action;
          const cls = ev.type === "action"
            ? "text-eval"
            : ev.resolved
              ? "text-muted-foreground/60"
              : "text-warning";
          return (
            <li key={i} className={cls}>
              <span className="text-muted-foreground/40 mr-2">{when}</span>
              <span>{label}</span>
              {ev.resolved ? <span className="text-ok ml-2">resolved</span> : null}
            </li>
          );
        })}
      </ul>
    </details>
  );
}

export function DashboardTabs({
  miners,
  scores,
  modelInfoMap,
  currentBlock,
  taoUsd,
  minersTaoDay,
  history,
  kingUid,
  kingH2hKl,
}: DashboardTabsProps) {
  const [activeTab, setActiveTab] = useState<string>("live");
  const kingMiner = kingUid != null ? miners.find((m) => m.uid === kingUid) : null;

  const scoreToBeat = kingH2hKl != null ? kingH2hKl * SCORE_TO_BEAT_FACTOR : null;

  return (
    <div className="space-y-3">
      <ValidatorStatus
        kingUid={kingUid}
        kingModel={kingMiner?.model}
        kingRevision={kingMiner?.revision}
        onViewDetails={() => setActiveTab("live")}
      />

      <Tabs value={activeTab} onValueChange={(v: unknown) => setActiveTab(String(v))}>
        <div className="overflow-x-auto scroll-snap-x -mx-2 px-2">
          <TabsList
            variant="line"
            className="w-full min-w-max justify-start border-b border-border/20 pb-0"
            role="tablist"
          >
            {TABS.map((t) => (
              <TabsTrigger
                key={t.id}
                value={t.id}
                className="scroll-snap-item text-sm px-4 py-2 whitespace-nowrap"
                aria-label={`${t.label} — ${t.hint}`}
              >
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        <TabsContent value="live" className="pt-4 space-y-4">
          {kingMiner && (
            <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm px-4 py-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono text-muted-foreground">
              <span className="text-king">UID {kingMiner.uid}</span>
              <span className="text-foreground/70 truncate max-w-[320px]">{kingMiner.model}</span>
              {shortRevision(kingMiner.revision) && (
                <span
                  className="rounded-md border border-eval/30 bg-eval/10 px-2 py-0.5 text-eval"
                  title={kingMiner.revision}
                >
                  commit {shortRevision(kingMiner.revision)}
                </span>
              )}
              <a
                href={CHAT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto rounded-md border border-king/40 bg-king/10 px-2 py-0.5 text-king hover:bg-king/20 transition-colors"
              >
                Chat with the King ↗
              </a>
            </div>
          )}
          <EvalProgressBar />
          <IncidentsPanel />
          <GpuLogs />
        </TabsContent>

        <TabsContent value="rounds" className="pt-4 space-y-6">
          <H2hHistory />
          {history.length > 0 ? (
            <ScoreTrend history={history} />
          ) : (
            <div className="text-center text-sm text-muted-foreground py-8">
              No score history available yet.
            </div>
          )}
          <KingHistory />
        </TabsContent>

        <TabsContent value="axes" className="pt-4">
          <TelemetryTab />
        </TabsContent>

        <TabsContent value="miners" className="pt-4 space-y-3">
          <MinersTab
            miners={miners}
            scores={scores}
            modelInfoMap={modelInfoMap}
            currentBlock={currentBlock}
            taoUsd={taoUsd}
            minersTaoDay={minersTaoDay}
          />
        </TabsContent>

        <TabsContent value="benchmarks" className="pt-4">
          <BenchmarksTab />
        </TabsContent>

      </Tabs>

      {scoreToBeat != null && activeTab === "miners" && (
        <p className="text-[11px] text-muted-foreground/60 px-2">
          Score-to-beat: KL &lt; {scoreToBeat.toFixed(6)} (king × {SCORE_TO_BEAT_FACTOR}).
        </p>
      )}
    </div>
  );
}
