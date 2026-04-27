"use client";

import { useState } from "react";

/**
 * Sticky top header — single 56px row.
 *
 * Layout: [logo + SN chip] [tab strip] ............ [meta strip] [LIVE]
 *
 * Visual idiom: monochrome, hairline border, no chip backgrounds. The
 * only animated element is the LIVE pulse — everything else is static
 * to avoid the "dashboard with too many spinners" look.
 *
 * The "LIVE" pill is keyed off the `live` prop (the validator's
 * `health.eval_active` from /api/health), so it goes dim when the
 * validator is between rounds and bright when an eval is running.
 */
export type TabId =
  | "home"
  | "live"
  | "rounds"
  | "axes"
  | "miners"
  | "bench"
  | "docs";

const TABS: { id: TabId; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "live", label: "Live" },
  { id: "rounds", label: "Rounds" },
  { id: "axes", label: "Axes" },
  { id: "miners", label: "Miners" },
  { id: "bench", label: "Bench" },
  { id: "docs", label: "Docs" },
];

export interface SiteHeaderProps {
  active: TabId;
  onTab: (id: TabId) => void;
  block: number | null;
  modelCount: number;
  kingUid: number | null;
  live: boolean;
}

export function SiteHeader({
  active,
  onTab,
  block,
  modelCount,
  kingUid,
  live,
}: SiteHeaderProps) {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 h-14 px-4 sm:px-6 flex items-center gap-4 sm:gap-6 border-b border-border bg-background">
      {/* Logo */}
      <button
        onClick={() => onTab("home")}
        className="flex items-baseline gap-2 shrink-0 cursor-pointer"
      >
        <span className="text-[18px] font-semibold tracking-[-0.04em]">distil</span>
        <span className="text-[10px] uppercase tracking-[0.2em] text-meta font-medium">
          SN97
        </span>
      </button>

      {/* Tab strip */}
      <nav
        role="tablist"
        className="flex h-full ml-auto sm:ml-0 overflow-x-auto sm:overflow-visible scroll-snap-x"
      >
        {TABS.map((t) => {
          const isOn = t.id === active;
          return (
            <button
              key={t.id}
              role="tab"
              aria-selected={isOn}
              onClick={() => onTab(t.id)}
              className={[
                "h-full px-3 sm:px-4 flex items-center text-[12px] font-medium",
                "tracking-[-0.01em] whitespace-nowrap scroll-snap-item",
                "transition-colors border-b-2",
                isOn
                  ? "text-foreground border-foreground"
                  : "text-meta hover:text-foreground border-transparent",
              ].join(" ")}
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      {/* Meta strip */}
      <div className="hidden md:flex items-center gap-3 ml-auto text-[11px] text-meta num shrink-0">
        {block != null && block > 0 && (
          <span title="Current Bittensor block">
            #{block.toLocaleString()}
          </span>
        )}
        <span className="w-px h-3.5 bg-border" />
        <span>
          <span className="text-foreground font-medium">{modelCount}</span> models
        </span>
        {kingUid != null && (
          <>
            <span className="w-px h-3.5 bg-border" />
            <span title="Current king">
              <span className="text-foreground font-medium">UID {kingUid}</span>{" "}
              <span className="text-meta">king</span>
            </span>
          </>
        )}
        <span className="w-px h-3.5 bg-border" />
        <LiveBadge live={live} onClick={() => setOpen((v) => !v)} />
      </div>

      {/* Mobile: just the live badge */}
      <div className="flex md:hidden ml-auto">
        <LiveBadge live={live} />
      </div>
    </header>
  );
}

function LiveBadge({
  live,
  onClick,
}: {
  live: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={live ? "Validator running an eval" : "Validator idle"}
      className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.15em] text-meta cursor-default"
    >
      <span
        className={[
          "w-1.5 h-1.5 rounded-full",
          live ? "bg-ok live-pulse" : "bg-border-strong",
        ].join(" ")}
      />
      <span>{live ? "live" : "idle"}</span>
    </button>
  );
}
