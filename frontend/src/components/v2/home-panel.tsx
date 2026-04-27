"use client";

import dynamic from "next/dynamic";
import type { TabId } from "./site-header";

// Lazy import — Three.js only needed on Home, ~600KB gzipped, no SSR.
const DitheredSurface = dynamic(
  () => import("./dithered-surface").then((m) => m.DitheredSurface),
  { ssr: false, loading: () => null }
);

export interface HomePanelProps {
  kingUid: number | null;
  kingModel: string | null;
  onTab: (id: TabId) => void;
}

/**
 * Home tab. Decorative dithered surface on the left, giant wordmark
 * + king pointer on the right. Idiom from the v2 design reference.
 *
 * Mobile: collapses to a 38vh shader on top, wordmark below. The
 * shader is purely decorative; if WebGL fails the right column still
 * renders as a self-sufficient page.
 */
export function HomePanel({ kingUid, kingModel, onTab }: HomePanelProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-[1.4fr_1fr] h-[calc(100vh-3.5rem-3rem)] min-h-[480px]">
      {/* Visualisation */}
      <div className="relative bg-[var(--surface-soft)] border-b md:border-b-0 md:border-r border-border overflow-hidden h-[40vh] md:h-auto">
        <DitheredSurface />
      </div>

      {/* Wordmark + footer */}
      <div className="flex flex-col min-h-0 overflow-hidden">
        <div className="flex-1 flex flex-col justify-center px-8 sm:px-12 py-12 min-h-0">
          <h1 className="text-[clamp(56px,7vw,108px)] leading-[0.95] tracking-[-0.05em] font-medium">
            distil
            <span className="serif block text-[0.5em] mt-1 leading-tight tracking-[-0.02em] text-meta">
              a giant.
            </span>
          </h1>
        </div>
        <div className="border-t border-border px-8 sm:px-12 py-4 flex flex-wrap items-center gap-4 text-[11px]">
          <span className="text-[10px] uppercase tracking-[0.18em] text-meta">SN97</span>
          <span className="w-px h-3.5 bg-border" />
          <span className="num text-meta truncate flex-1 min-w-0">
            {kingModel ? (
              <>
                <span className="text-foreground font-medium">{kingModel}</span>
                {kingUid != null && <> · uid {kingUid}</>}
              </>
            ) : (
              "no king yet"
            )}
          </span>
          <button
            onClick={() => onTab("live")}
            className="text-[11px] font-medium uppercase tracking-[0.04em] text-foreground hover:opacity-60 transition-opacity"
          >
            Watch →
          </button>
        </div>
      </div>
    </div>
  );
}
