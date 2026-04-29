"use client";

import { useEffect, useState } from "react";

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
        <ThemeToggle />
        <span className="w-px h-3.5 bg-border" />
        <LiveBadge live={live} onClick={() => setOpen((v) => !v)} />
      </div>

      {/* Mobile: theme toggle + live badge */}
      <div className="flex md:hidden ml-auto items-center gap-2">
        <ThemeToggle />
        <LiveBadge live={live} />
      </div>
    </header>
  );
}

type Theme = "light" | "dark" | "system";

/**
 * Theme cycle button. Three states (system → light → dark → system)
 * persisted to localStorage. Writes data-theme on <html> so the
 * globals.css palette swap fires immediately. Reading both
 * prefers-color-scheme + the stored value, with stored winning.
 *
 * "system" removes the data-theme attribute so the @media
 * (prefers-color-scheme: dark) block in globals.css takes effect —
 * matches OS appearance.
 *
 * Initial paint: NO_THEME on the server-rendered HTML. The script
 * below runs before paint via useEffect to apply the saved theme;
 * there is a single-frame flash of light-mode on first load when
 * the user's preference is dark, which is acceptable. (Could be
 * eliminated by inlining a tiny script in <head>; not worth the
 * Next.js complexity for now.)
 */
function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  function apply(next: Theme) {
    if (typeof document === "undefined") return;
    const el = document.documentElement;
    if (next === "system") {
      el.removeAttribute("data-theme");
    } else {
      el.setAttribute("data-theme", next);
    }
  }

  useEffect(() => {
    setMounted(true);
    let saved: Theme = "system";
    try {
      const v = window.localStorage.getItem("distil:theme") as Theme | null;
      if (v === "light" || v === "dark" || v === "system") saved = v;
    } catch {}
    setTheme(saved);
    apply(saved);
    // Listen for the footer toggle (or any other toggle) so the two
    // stay in sync in the same tab. See 2026-04-28 fix in
    // components/auto-refresh.tsx for the event contract.
    const onChange = (ev: Event) => {
      const detail = (ev as CustomEvent<{ theme?: Theme }>).detail;
      if (detail?.theme === "light" || detail?.theme === "dark" || detail?.theme === "system") {
        setTheme(detail.theme);
        apply(detail.theme);
      }
    };
    window.addEventListener("distil:theme-changed", onChange);
    return () => window.removeEventListener("distil:theme-changed", onChange);
  }, []);

  function cycle() {
    const order: Theme[] = ["system", "dark", "light"];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(next);
    apply(next);
    try {
      window.localStorage.setItem("distil:theme", next);
    } catch {}
    try {
      window.dispatchEvent(
        new CustomEvent("distil:theme-changed", { detail: { theme: next } }),
      );
    } catch {}
  }

  // Avoid hydration mismatch — render placeholder until mounted.
  if (!mounted) {
    return (
      <button
        type="button"
        aria-label="Theme"
        className="text-[10px] uppercase tracking-[0.16em] text-meta hover:text-foreground"
      >
        ◐
      </button>
    );
  }

  const glyph = theme === "dark" ? "●" : theme === "light" ? "○" : "◐";
  const label =
    theme === "dark" ? "Dark" : theme === "light" ? "Light" : "Auto";
  return (
    <button
      type="button"
      onClick={cycle}
      title={`Theme: ${label} (click to cycle)`}
      aria-label={`Theme: ${label}. Click to cycle.`}
      className="inline-flex items-center gap-1 text-[10px] uppercase tracking-[0.16em] text-meta hover:text-foreground transition-colors"
    >
      <span aria-hidden className="text-[12px] leading-none">{glyph}</span>
      <span className="hidden sm:inline">{label}</span>
    </button>
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
      title={
        live
          ? "Validator is running an eval right now. Click for the Live tab."
          : "Validator is idle. Next round starts when new on-chain commitments arrive."
      }
      className={[
        "inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.15em]",
        live ? "text-foreground font-medium" : "text-meta",
      ].join(" ")}
    >
      <span
        className={[
          "w-1.5 h-1.5 rounded-full",
          live ? "bg-ok live-pulse" : "bg-[var(--ink-meta-soft)]",
        ].join(" ")}
      />
      <span>{live ? "live" : "idle"}</span>
    </button>
  );
}
