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

const VALID_TABS: TabId[] = [
  "home",
  "live",
  "rounds",
  "axes",
  "miners",
  "bench",
  "docs",
];

/**
 * Dashboard v2 shell — sticky header + 7 tabs.
 *
 * Active-tab state is persisted in the URL hash (`#live`, `#axes`,
 * etc) so it survives `router.refresh()`. Without this the
 * AutoRefresh component re-mounts <DashboardV2> on every 30s tick and
 * the user gets bounced back to the Home tab — a bug that was
 * actively confusing miners in Discord.
 *
 * The hash is the source of truth: SSR boot defaults to `home`, then
 * the client mount reads `window.location.hash` and reconciles. Tab
 * changes write to the hash via `history.replaceState` (no nav
 * stack pollution). hashchange events from the back/forward buttons
 * are honoured.
 */
export function DashboardV2({
  miners,
  modelInfoMap,
  currentBlock,
  taoUsd,
  minersTaoDay,
  kingUid: kingUidProp,
}: DashboardV2Props) {
  const [active, setActive] = useState<TabId>("home");
  const [live, setLive] = useState<boolean>(false);
  // The king can flip mid-session. SSR seeds the prop, but we keep a
  // local mirror that the header poll can update live so a king flip
  // doesn't require a manual page reload.
  const [kingUid, setKingUid] = useState<number | null>(kingUidProp);

  // Tab ↔ hash reconciliation. Run once on mount + listen for changes.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const fromHash = (h: string): TabId | null => {
      const cleaned = h.replace(/^#/, "") as TabId;
      return VALID_TABS.includes(cleaned) ? cleaned : null;
    };
    const initial = fromHash(window.location.hash);
    if (initial && initial !== active) setActive(initial);

    const onHash = () => {
      const t = fromHash(window.location.hash);
      if (t) setActive(t);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Write active-tab back to the hash. replaceState (not pushState) so
  // the back button still leaves the dashboard, not cycles tabs.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const wanted = `#${active}`;
    if (window.location.hash !== wanted) {
      window.history.replaceState(null, "", wanted);
    }
  }, [active]);

  // Keep the SSR prop and the live state in sync if the prop changes
  // (e.g. a router.refresh() succeeds with new SSR data).
  useEffect(() => {
    setKingUid(kingUidProp);
  }, [kingUidProp]);

  // Poll eval-progress for the LIVE pulse + h2h-latest for king flips.
  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const [progRes, h2hRes] = await Promise.all([
          fetch(`${CLIENT_API_BASE}/api/eval-progress`, { cache: "no-store" }),
          fetch(`${CLIENT_API_BASE}/api/h2h-latest`, { cache: "no-store" }),
        ]);
        if (!cancel) {
          if (progRes.ok) {
            const j = await progRes.json();
            setLive(!!j?.active);
          }
          if (h2hRes.ok) {
            const j = await h2hRes.json();
            const u = typeof j?.king_uid === "number" ? j.king_uid : null;
            if (u != null) setKingUid(u);
          }
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 12_000);
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
