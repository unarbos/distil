"use client";

import { useEffect, useState } from "react";

/** Custom event name dispatched on each auto-refresh cycle. Kept for
 * future consumers — no v2 panel currently listens. */
export const REFRESH_EVENT = "dashboard-refresh";

/**
 * Auto-refresh — DEPRECATED in favour of per-panel polling.
 *
 * History: this used to call ``router.refresh()`` every 30s to
 * re-run server components. As of 2026-04-27 the v2 dashboard fully
 * replaced that pattern with per-panel SWR-style fetches (each tab
 * polls /api/h2h-latest, /api/eval-progress, /api/leaderboard, etc.
 * directly). The site-header polls /api/h2h-latest itself for king
 * flips, the LiveBadge polls /api/eval-progress, and so on.
 *
 * Calling ``router.refresh()`` while the URL has a ``#hash`` (which
 * the v2 dashboard always does after the first tab change) was
 * triggering a Next.js App Router prefetch loop that surfaced as
 * ``ERR_TOO_MANY_REDIRECTS`` errors in the browser console and
 * eventually rendered the dashboard unusable until a hard refresh.
 * Trace pattern: "Failed to fetch RSC payload for ...#tab" → falls
 * back to browser navigation → triggers another navigation → loops.
 *
 * The current behaviour: dispatch a CustomEvent on the visibility
 * window so any future opt-in consumer can subscribe via
 * ``useRefreshKey()``. No router work, no SSR refetch, no redirect
 * loop. The event is currently a no-op (no listeners).
 */
export function AutoRefresh({ intervalMs = 60000 }: { intervalMs?: number }) {
  useEffect(() => {
    const tick = () => {
      if (typeof document === "undefined" || document.hidden) return;
      window.dispatchEvent(new CustomEvent(REFRESH_EVENT));
    };
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return null;
}

/**
 * Hook that returns a refreshKey that increments on each auto-refresh
 * cycle. Use as a dependency in useEffect to trigger re-fetches.
 * Currently unused by the v2 panels (each panel runs its own
 * setInterval) but kept available for future opt-in consumers.
 */
export function useRefreshKey(): number {
  const [key, setKey] = useState(0);
  useEffect(() => {
    const handler = () => setKey((k: number) => k + 1);
    window.addEventListener(REFRESH_EVENT, handler);
    return () => window.removeEventListener(REFRESH_EVENT, handler);
  }, []);
  return key;
}

type Theme = "light" | "dark" | "system";

/**
 * Footer theme toggle. The v2 site-header has its own (richer)
 * toggle; this one is the legacy footer button kept so the existing
 * <Footer> markup still works. Both write to the SAME localStorage
 * key (``distil:theme``) and the SAME data-theme attribute on
 * <html>, so the two toggles stay in sync — clicking one updates
 * what the other displays on the next render.
 *
 * Three states (system → dark → light), uses the same glyphs as the
 * site-header version (○/●/◐) so users learn one icon language.
 *
 * 2026-04-28 (Discord rao_2222): the footer button used to read the
 * saved theme on mount but never call ``apply()``, so if a user landed
 * with no saved value and clicked the FOOTER button first, the
 * data-theme attribute matched but the React state did not. The
 * site-header toggle would then disagree on the next click. Two fixes:
 *   1. ``apply(saved)`` on mount so the runtime state mirrors the DOM.
 *   2. Listen for the ``distil:theme-changed`` custom event so both
 *      toggles stay in sync within the same tab when EITHER is clicked
 *      (storage events only fire cross-tab, which is why same-tab
 *      header+footer used to drift).
 */
export const THEME_CHANGED_EVENT = "distil:theme-changed";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  function apply(next: Theme) {
    if (typeof document === "undefined") return;
    const el = document.documentElement;
    if (next === "system") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", next);
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
    const onChange = (ev: Event) => {
      const detail = (ev as CustomEvent<{ theme?: Theme }>).detail;
      if (detail?.theme === "light" || detail?.theme === "dark" || detail?.theme === "system") {
        setTheme(detail.theme);
        apply(detail.theme);
      }
    };
    window.addEventListener(THEME_CHANGED_EVENT, onChange);
    return () => window.removeEventListener(THEME_CHANGED_EVENT, onChange);
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
      window.dispatchEvent(new CustomEvent(THEME_CHANGED_EVENT, { detail: { theme: next } }));
    } catch {}
  }

  if (!mounted) {
    return (
      <span
        className="text-xs px-2 py-1 rounded border border-border/60 bg-secondary/40 text-muted-foreground"
        aria-hidden
      >
        ◐
      </span>
    );
  }

  const label = theme === "system" ? "Auto" : theme === "dark" ? "Dark" : "Light";
  const glyph = theme === "dark" ? "●" : theme === "light" ? "○" : "◐";
  return (
    <button
      type="button"
      onClick={cycle}
      aria-label={`Theme: ${label}. Click to cycle.`}
      className="text-xs px-2 py-1 rounded border border-border/60 bg-secondary/40 hover:bg-secondary/60 text-muted-foreground inline-flex items-center gap-1"
    >
      <span aria-hidden>{glyph}</span>
      {label}
    </button>
  );
}
