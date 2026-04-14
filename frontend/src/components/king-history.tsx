"use client";

import { useEffect, useState } from "react";
import { useRefreshKey } from "@/components/auto-refresh";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.arbos.life";

interface KingChange {
  block: number;
  timestamp: number;
  old_king_uid: number | null;
  new_king_uid: number;
  old_king_model?: string | null;
  new_king_model?: string | null;
  old_king_kl?: number | null;
  new_king_kl?: number | null;
  n_prompts?: number | null;
  paired_prompts?: number | null;
  prompts_scored?: number | null;
  dethrone_eligible?: boolean;
  p_value?: number | null;
  _exploit?: boolean;
}

function formatFixed(value: number | null | undefined, digits: number, fallback = "—"): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

function formatPromptCount(scored: number | null | undefined, total: number | null | undefined): string | null {
  if (typeof scored !== "number" || !Number.isFinite(scored)) return null;
  if (typeof total === "number" && Number.isFinite(total) && total > 0) {
    return `${scored}/${total}p`;
  }
  return `${scored}p`;
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }) + " UTC";
}

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export function KingHistory() {
  const [changes, setChanges] = useState<KingChange[]>([]);
  const [loading, setLoading] = useState(true);
  const refreshKey = useRefreshKey();

  useEffect(() => {
    let cancelled = false;

    async function fetchKingHistory() {
      try {
        const res = await fetch(`${API_BASE}/api/king-history`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setChanges(Array.isArray(data) ? data : data.changes ?? []);
          }
        } else {
          // Endpoint doesn't exist yet — try to extract from h2h-history
          if (!cancelled) await fallbackFromH2h();
        }
      } catch {
        if (!cancelled) await fallbackFromH2h();
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    async function fallbackFromH2h() {
      try {
        const res = await fetch(`${API_BASE}/api/h2h-history`, { cache: "no-store" });
        if (!res.ok) return;
        const raw = await res.json();
        const rounds = Array.isArray(raw) ? raw : Array.isArray(raw?.rounds) ? raw.rounds : [];
        // Extract king changes from h2h rounds
        const kingChanges: KingChange[] = [];
        for (const round of rounds) {
          if (round.king_changed && round.new_king_uid != null) {
            const newKing = round.results?.find(
              (r: { uid?: number; is_king?: boolean; model?: string; kl?: number; paired_prompts?: number; prompts_scored?: number; dethrone_eligible?: boolean; t_test?: { p?: number } }) =>
                r.uid === round.new_king_uid
            );
            const oldKing = round.results?.find(
              (r: { uid?: number; is_king?: boolean; model?: string; kl?: number }) => r.uid === round.prev_king_uid || r.is_king
            );
            kingChanges.push({
              block: round.block,
              timestamp: round.timestamp,
              old_king_uid: round.prev_king_uid ?? null,
              new_king_uid: round.new_king_uid,
              old_king_model: oldKing?.model ?? null,
              new_king_model: newKing?.model ?? `UID ${round.new_king_uid}`,
              old_king_kl: oldKing?.kl,
              new_king_kl: newKing?.kl ?? round.king_h2h_kl,
              n_prompts: round.n_prompts,
              paired_prompts: newKing?.paired_prompts,
              prompts_scored: newKing?.prompts_scored,
              dethrone_eligible: newKing?.dethrone_eligible,
              p_value: newKing?.t_test?.p,
            });
          }
        }
        setChanges(kingChanges);
      } catch {
        // silently fail
      }
    }

    fetchKingHistory();
    return () => { cancelled = true; };
  }, [refreshKey]);

  if (loading) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6">
        <div className="animate-pulse space-y-3">
          <div className="h-4 w-32 bg-muted-foreground/10 rounded" />
          <div className="h-3 w-48 bg-muted-foreground/5 rounded" />
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-muted-foreground/5 rounded-lg" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (changes.length === 0) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-center">
        <span className="text-sm text-muted-foreground/50 font-mono">
          No king changes recorded yet
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border/20 bg-card/10 backdrop-blur-sm p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-mono text-muted-foreground/70 uppercase tracking-wider">
          👑 King History
        </h3>
        <span className="text-[10px] text-muted-foreground/40 font-mono">
          {changes.length} dethronement{changes.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div className="relative space-y-0">
        {/* Timeline line */}
        <div className="absolute left-[9px] top-2 bottom-2 w-px bg-yellow-400/20" />

        {changes.slice(0, 10).map((change, idx) => (
          <div key={`${change.block}-${change.timestamp}`} className="relative flex gap-3 py-2.5">
            {/* Timeline dot */}
            <div className={`w-[19px] h-[19px] rounded-full border-2 flex items-center justify-center shrink-0 z-10 ${
              idx === 0
                ? "border-yellow-400 bg-yellow-400/20"
                : "border-yellow-400/30 bg-card"
            }`}>
              <span className="text-[8px]">👑</span>
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-mono text-yellow-400/90 truncate">
                  UID {change.new_king_uid}
                </span>
                <span className="text-[10px] font-mono text-muted-foreground/40">
                  dethroned UID {change.old_king_uid ?? "?"}
                </span>
                {change._exploit && (
                  <span className="inline-flex items-center rounded-full bg-red-400/10 border border-red-400/20 px-1.5 py-0.5 text-[9px] text-red-400 font-mono">
                    exploit-era
                  </span>
                )}
              </div>
              <div className="text-[10px] font-mono text-blue-400/60 truncate">
                {change.new_king_model ?? `UID ${change.new_king_uid}`}
              </div>
              <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground/40">
                <span>KL {formatFixed(change.new_king_kl, 6)}</span>
                {formatPromptCount(change.paired_prompts ?? change.prompts_scored, change.n_prompts) && (
                  <>
                    <span>·</span>
                    <span>{formatPromptCount(change.paired_prompts ?? change.prompts_scored, change.n_prompts)}</span>
                  </>
                )}
                {typeof change.p_value === "number" && Number.isFinite(change.p_value) && (
                  <>
                    <span>·</span>
                    <span>p={change.p_value.toFixed(4)}</span>
                  </>
                )}
                <span>·</span>
                <span>{formatTimestamp(change.timestamp)}</span>
                <span className="text-muted-foreground/30">({timeAgo(change.timestamp)})</span>
              </div>
            </div>
          </div>
        ))}

        {changes.length > 10 && (
          <div className="text-center text-[10px] text-muted-foreground/30 font-mono pt-2">
            + {changes.length - 10} earlier changes
          </div>
        )}
      </div>
    </div>
  );
}
