"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

/** Custom event name dispatched on each auto-refresh cycle */
export const REFRESH_EVENT = "dashboard-refresh";

/**
 * Auto-refresh the page by triggering a Next.js router refresh every `intervalMs`.
 * This re-runs server components without a full browser reload.
 * Also dispatches a custom "dashboard-refresh" event so client components
 * (charts, tables) can re-fetch their own data.
 *
 * We only fire while the document is visible (no point refreshing
 * server data for a backgrounded tab) and we use 60s rather than the
 * historical 30s — most tabs poll their own endpoints at finer
 * granularity client-side, so the SSR refresh is just a fallback.
 * Slower SSR refresh = less perceived click latency on tab swaps
 * because there's no race between user interaction and a router
 * refresh re-rendering the tree.
 */
export function AutoRefresh({ intervalMs = 60000 }: { intervalMs?: number }) {
  const router = useRouter();

  useEffect(() => {
    const tick = () => {
      if (typeof document !== "undefined" && document.hidden) return;
      router.refresh();
      window.dispatchEvent(new CustomEvent(REFRESH_EVENT));
    };
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [router, intervalMs]);

  return null;
}

/**
 * Hook that returns a refreshKey that increments on each auto-refresh cycle.
 * Use as a dependency in useEffect to trigger re-fetches.
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

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const saved = (typeof window !== "undefined" && (localStorage.getItem("distil-theme") as Theme)) || "system";
    setTheme(saved);
    apply(saved);
  }, []);

  function apply(next: Theme) {
    const el = document.documentElement;
    if (next === "system") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", next);
  }

  function cycle() {
    const order: Theme[] = ["system", "dark", "light"];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(next);
    apply(next);
    try { localStorage.setItem("distil-theme", next); } catch {}
  }

  const label = theme === "system" ? "Auto" : theme === "dark" ? "Dark" : "Light";
  return (
    <button
      type="button"
      onClick={cycle}
      aria-label={`Theme: ${label}. Click to cycle.`}
      className="text-xs px-2 py-1 rounded border border-border/60 bg-secondary/40 hover:bg-secondary/60 text-muted-foreground"
    >
      {label}
    </button>
  );
}
