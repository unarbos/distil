"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";
import type { MinerEntry, ModelInfo } from "@/lib/api";
import { SiteHeader, type TabId } from "./site-header";
import { HomePanel } from "./home-panel";
import { LivePanel } from "./live-panel";
import { RoundsPanel } from "./rounds-panel";
import { AxesPanel } from "./axes-panel";
import { MinersPanel } from "./miners-panel";
import { BenchPanel } from "./bench-panel";
import { DocsPanel } from "./docs-panel";

export interface DashboardV2Props {
  miners: MinerEntry[];
  modelInfoMap: Record<string, ModelInfo>;
  currentBlock: number;
  taoUsd: number;
  minersTaoDay: number;
  kingUid: number | null;
}

/**
 * Dashboard v2 shell — sticky header + 7 tabs.
 *
 * Single-page application; tab switches don't re-render the header,
 * SSR data flows in once and the inner panels use SWR-style polling
 * for live updates. The previous v1 dashboard used five tabs, all
 * dark-mode + KL-headlined; this one is monochrome + composite-first.
 */
export function DashboardV2({
  miners,
  modelInfoMap,
  currentBlock,
  taoUsd,
  minersTaoDay,
  kingUid,
}: DashboardV2Props) {
  const [active, setActive] = useState<TabId>("home");
  const [live, setLive] = useState<boolean>(false);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/eval-progress`, { cache: "no-store" });
        if (res.ok && !cancel) {
          const json = await res.json();
          setLive(!!json?.active);
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 8_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const king = kingUid != null ? miners.find((m) => m.uid === kingUid) : null;

  return (
    <div className="min-h-screen flex flex-col">
      <SiteHeader
        active={active}
        onTab={setActive}
        block={currentBlock}
        modelCount={miners.length}
        kingUid={kingUid}
        live={live}
      />
      <main className="flex-1 relative">
        {active === "home" && (
          <HomePanel
            kingUid={kingUid}
            kingModel={king?.model ?? null}
            onTab={setActive}
          />
        )}
        {active === "live" && <LivePanel />}
        {active === "rounds" && <RoundsPanel />}
        {active === "axes" && <AxesPanel />}
        {active === "miners" && (
          <MinersPanel
            miners={miners}
            modelInfoMap={modelInfoMap}
            taoUsd={taoUsd}
            minersTaoDay={minersTaoDay}
          />
        )}
        {active === "bench" && <BenchPanel />}
        {active === "docs" && <DocsPanel />}
      </main>
    </div>
  );
}
