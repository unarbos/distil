"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";
import { useRefreshKey } from "@/components/auto-refresh";
import { formatFixed } from "@/lib/utils";

interface DqEntry {
  key: string;
  reason: string;
  uid?: number;
  hotkey?: string;
  block?: number;
}

interface EventEntry {
  ts: number;
  level: string;
  msg: string;
}

interface CompositeAxes {
  kl?: number;
  capability?: number;
  length?: number;
  degeneracy?: number;
  adversarial?: number;
  on_policy_rkl?: number;
  judge_probe?: number;
}

interface Composite {
  version?: number;
  axes?: CompositeAxes;
  worst?: number;
  weighted?: number;
  present_count?: number;
  broken_axes?: string[];
  judge_in_composite?: boolean;
}

interface RoundResult {
  uid: number;
  model: string;
  kl: number;
  is_king: boolean;
  vs_king?: string;
  disqualified?: boolean;
  dq_reason?: string;
  dethrone_eligible?: boolean;
  early_stopped?: boolean;
  prompts_scored?: number;
  prompts_total?: number;
  paired_prompts?: number;
  composite?: Composite;
  capability_pass_frac?: number;
  capability_teacher?: number;
  length_ratio?: number;
  length_penalty?: number;
  think_pass?: boolean;
  think_reason?: string;
  adversarial_pass_frac?: number;
  adversarial_mean_tokens?: number;
  judge_mean_score?: number;
  judge_normalized?: number;
  judge_n_valid?: number;
  judge_n?: number;
}

interface RoundDetail {
  block: number;
  block_hash?: string;
  timestamp: number;
  king_uid: number;
  prev_king_uid?: number;
  king_changed?: boolean;
  new_king_uid?: number;
  king_retained_reason?: string;
  dq_blocked_dethrone?: Array<{ uid?: number; kl?: number; reason?: string }>;
  n_prompts?: number;
  n_students?: number;
  elapsed_seconds?: number;
  epsilon?: number;
  paired_test_alpha?: number;
  results: RoundResult[];
}

interface CapabilityItem {
  q: string;
  expected: string;
  pred: string;
  ok: boolean;
}

interface CapabilityBlock {
  n?: number;
  correct?: number;
  pass_frac?: number;
  teacher_pass_frac?: number;
  items: CapabilityItem[];
}

interface ThinkSample {
  prompt: string;
  gen_tokens: number;
  terminated: boolean;
  gzip_ratio: number;
  distinct_4: number;
  top_6gram_rate: number;
  tail: string;
}

interface ThinkBlock {
  pass?: boolean;
  reason?: string;
  prompts_tested?: number;
  prompts_terminated?: number;
  prompts_degenerate?: number;
  mean_gen_tokens?: number;
  self_bleu_across_prompts?: number;
  teacher_self_bleu?: number;
  samples: ThinkSample[];
}

interface KingProbe {
  uid: number;
  model: string;
  status?: string;
  kl?: number;
  capability?: CapabilityBlock | null;
  length_axis?: { ratio?: number; penalty?: number; student_mean_gen?: number; teacher_mean_gen?: number } | null;
  think_probe?: ThinkBlock | null;
  adversarial?: { pass_frac?: number; mean_gen_tokens?: number; n?: number; details?: Array<{ q?: string; kind?: string; gen?: string; ok?: boolean; gen_tokens?: number }> } | null;
  load_time?: number;
}

interface PrivatePoolCommit {
  current?: { block?: number; root?: string; n?: number; committed_at?: number };
  latest_reveal?: { block?: number; n?: number; prompt_hashes?: string[]; revealed_at?: number };
}

interface CurrentRound {
  started_at?: number;
  block?: number;
  king_uid?: number;
  private_pool?: { n?: number; commit_root?: string; fraction?: number };
}

interface TelemetryOverview {
  server_time: number;
  current_round: CurrentRound;
  private_pool: PrivatePoolCommit;
  king_probe: KingProbe | null;
  round_detail: RoundDetail | null;
  recent_dqs: DqEntry[];
  recent_events: EventEntry[];
}

interface PodHealth {
  gpu: Array<{
    index: number;
    name: string;
    util_pct: number;
    mem_used_mb: number;
    mem_total_mb: number;
    temp_c: number;
    power_w: number;
    power_limit_w: number;
  }> | null;
  pod: { eval_active: boolean; phase?: string; started_at?: number; estimated_duration_s?: number };
  validator?: { active_state?: string; sub_state?: string };
}

function levelColor(lvl: string): string {
  const l = (lvl || "").toLowerCase();
  if (l === "error") return "text-red-400";
  if (l === "warn" || l === "warning") return "text-amber-400";
  return "text-muted-foreground/70";
}

function axisColor(v?: number): string {
  if (v == null) return "text-muted-foreground/40";
  if (v >= 0.95) return "text-emerald-400";
  if (v >= 0.8) return "text-blue-300";
  if (v >= 0.5) return "text-amber-400";
  return "text-red-400";
}

function AxisBar({ label, value }: { label: string; value?: number }) {
  const v = value ?? 0;
  const pct = Math.max(2, Math.min(100, v * 100));
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono">
      <span className="w-20 text-muted-foreground/70">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted/20 overflow-hidden">
        <div
          className={`h-full rounded-full ${v >= 0.95 ? "bg-emerald-400" : v >= 0.8 ? "bg-blue-400" : v >= 0.5 ? "bg-amber-400" : "bg-red-400"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`w-12 text-right tabular-nums ${axisColor(value)}`}>
        {value == null ? "—" : value.toFixed(3)}
      </span>
    </div>
  );
}

function formatTs(ts: number): string {
  const d = new Date(ts * 1000);
  const now = Date.now();
  const diff = Math.round((now - ts * 1000) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return d.toISOString().slice(5, 16).replace("T", " ");
}

export function TelemetryTab() {
  const [data, setData] = useState<TelemetryOverview | null>(null);
  const [health, setHealth] = useState<PodHealth | null>(null);
  const [error, setError] = useState(false);
  const [expandedUid, setExpandedUid] = useState<number | null>(null);
  const [eventFilter, setEventFilter] = useState<"all" | "warn" | "error">("all");
  const refreshKey = useRefreshKey();

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [ov, ph] = await Promise.all([
          fetch(`${CLIENT_API_BASE}/api/telemetry/overview`, { cache: "no-store" }).then((r) => r.ok ? r.json() : null),
          fetch(`${CLIENT_API_BASE}/api/telemetry/pod-health`, { cache: "no-store" }).then((r) => r.ok ? r.json() : null),
        ]);
        if (!cancelled) {
          if (ov) { setData(ov); setError(false); }
          if (ph) setHealth(ph);
          if (!ov) setError(true);
        }
      } catch {
        if (!cancelled) setError(true);
      }
    };
    load();
    const id = setInterval(load, 15_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [refreshKey]);

  if (error && !data) {
    return (
      <div className="rounded-xl border border-amber-400/30 bg-amber-400/[0.03] p-6 text-center text-sm font-mono text-amber-400/70">
        Telemetry unavailable — retrying every 15s…
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-center text-sm font-mono text-muted-foreground/60">
        Loading telemetry…
      </div>
    );
  }

  const kp = data.king_probe;
  const rd = data.round_detail;
  const pool = data.private_pool?.current;
  const reveal = data.private_pool?.latest_reveal;
  const events = (data.recent_events || []).slice().reverse().filter((e) => {
    if (eventFilter === "all") return true;
    const l = (e.level || "").toLowerCase();
    if (eventFilter === "warn") return l === "warn" || l === "warning" || l === "error";
    if (eventFilter === "error") return l === "error";
    return true;
  });
  const errorEvents = (data.recent_events || []).filter((e) => {
    const l = (e.level || "").toLowerCase();
    return l === "warn" || l === "warning" || l === "error";
  }).length;

  return (
    <div className="space-y-4">
      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          title="Current round"
          value={data.current_round?.block ? `#${data.current_round.block.toLocaleString()}` : "—"}
          sub={data.current_round?.started_at ? formatTs(data.current_round.started_at) : "idle"}
        />
        <StatCard
          title="King UID"
          value={kp?.uid != null ? String(kp.uid) : rd?.king_uid != null ? String(rd.king_uid) : "—"}
          sub={kp?.model ? kp.model.split("/").pop() : undefined}
          accent="king"
        />
        <StatCard
          title="Private holdout"
          value={pool?.n != null ? `${pool.n} prompts` : "—"}
          sub={pool?.root ? `root ${pool.root.slice(0, 12)}…` : "no commit yet"}
        />
        <StatCard
          title="Recent alerts"
          value={`${errorEvents}`}
          sub="warnings + errors"
          accent={errorEvents > 0 ? "warn" : "ok"}
        />
      </div>

      {/* GPU / validator health */}
      {health && (
        <div className="rounded-xl border border-border/20 bg-card/10 p-3 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-muted-foreground">
            <span className="uppercase tracking-wider text-[10px] text-muted-foreground/50">Validator & GPU</span>
            <span>
              validator:{" "}
              <span className={health.validator?.sub_state === "running" ? "text-emerald-400" : "text-red-400"}>
                {health.validator?.sub_state ?? "unknown"}
              </span>
              {health.pod?.eval_active && (
                <span className="ml-3 text-blue-400">
                  eval: {health.pod.phase ?? "active"}
                </span>
              )}
            </span>
          </div>
          {health.gpu && health.gpu.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {health.gpu.map((g) => (
                <div key={g.index} className="text-[11px] font-mono flex items-center gap-3">
                  <span className="text-muted-foreground/70 w-16 truncate">{g.name}</span>
                  <span className="text-blue-300 tabular-nums">{g.util_pct.toFixed(0)}% util</span>
                  <span className="text-muted-foreground/60 tabular-nums">
                    {(g.mem_used_mb / 1024).toFixed(1)}/{(g.mem_total_mb / 1024).toFixed(0)} GB
                  </span>
                  <span className="text-amber-400/70 tabular-nums">{g.temp_c.toFixed(0)}°C</span>
                  <span className="text-muted-foreground/50 tabular-nums">
                    {g.power_w.toFixed(0)}/{g.power_limit_w.toFixed(0)} W
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[11px] font-mono text-muted-foreground/40">
              GPU metrics not available on this host (eval runs on remote pod)
            </div>
          )}
        </div>
      )}

      {/* King probe */}
      {kp && (
        <div className="rounded-xl border border-yellow-400/25 bg-yellow-400/[0.02] p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-yellow-400/90 uppercase tracking-wider">
              👑 King Probe — UID {kp.uid}
            </h3>
            <span className="text-[11px] font-mono text-muted-foreground/50 truncate max-w-[280px]">
              {kp.model}
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <AxisBar label="KL" value={kp.kl != null ? 1 / Math.max(1, 1 + kp.kl * 10) : undefined} />
              <AxisBar label="Capability" value={kp.capability?.pass_frac} />
              <AxisBar label="Length" value={kp.length_axis?.penalty} />
              <AxisBar label="Coherence" value={kp.think_probe?.pass ? 1 : 0.3} />
              <AxisBar label="Adversarial" value={kp.adversarial?.pass_frac} />
            </div>
            <div className="text-[11px] font-mono space-y-1 text-muted-foreground/70">
              <div>KL: <span className="text-foreground/80">{formatFixed(kp.kl, 6)}</span></div>
              {kp.capability && (
                <div>
                  capability: {kp.capability.correct}/{kp.capability.n}
                  {kp.capability.teacher_pass_frac != null && (
                    <span className="text-muted-foreground/40"> (teacher {(kp.capability.teacher_pass_frac * 100).toFixed(0)}%)</span>
                  )}
                </div>
              )}
              {kp.length_axis && (
                <div>
                  length ratio: {formatFixed(kp.length_axis.ratio, 3)} (penalty {formatFixed(kp.length_axis.penalty, 3)})
                </div>
              )}
              {kp.think_probe && (
                <div>
                  coherence: <span className={kp.think_probe.pass ? "text-emerald-400" : "text-red-400"}>
                    {kp.think_probe.pass ? "pass" : "fail"}
                  </span>
                  {kp.think_probe.reason && <span className="text-muted-foreground/40"> — {kp.think_probe.reason}</span>}
                  <span className="text-muted-foreground/40"> · mean {kp.think_probe.mean_gen_tokens} tok</span>
                </div>
              )}
              {kp.adversarial && (
                <div>
                  adversarial: <span className={axisColor(kp.adversarial.pass_frac)}>
                    {kp.adversarial.pass_frac != null ? `${(kp.adversarial.pass_frac * 100).toFixed(0)}%` : "—"}
                  </span>
                  {kp.adversarial.n != null && <span className="text-muted-foreground/40"> · {kp.adversarial.n} prompts</span>}
                  {kp.adversarial.mean_gen_tokens != null && <span className="text-muted-foreground/40"> · mean {Math.round(kp.adversarial.mean_gen_tokens)} tok</span>}
                </div>
              )}
              {kp.load_time != null && (
                <div className="text-muted-foreground/40">load_time: {formatFixed(kp.load_time, 1)}s</div>
              )}
            </div>
          </div>
          {/* Capability detail */}
          {kp.capability?.items && kp.capability.items.length > 0 && (
            <details className="text-[11px] font-mono">
              <summary className="cursor-pointer text-muted-foreground/60 hover:text-muted-foreground">
                capability probe detail ({kp.capability.items.length} items)
              </summary>
              <div className="mt-2 space-y-1 max-h-72 overflow-auto">
                {kp.capability.items.map((it, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className={it.ok ? "text-emerald-400" : "text-red-400"}>
                      {it.ok ? "✓" : "✗"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="truncate text-muted-foreground/80">{it.q}</div>
                      <div className="text-muted-foreground/40 truncate">
                        exp: {it.expected} · got: {it.pred}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {/* Last round with composite axes */}
      {rd && (
        <div className="rounded-xl border border-border/20 bg-card/10 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Last Round — Block #{rd.block?.toLocaleString()}
            </h3>
            <span className="text-[11px] font-mono text-muted-foreground/50">
              {rd.n_students} students · {rd.n_prompts} prompts · {formatFixed(rd.elapsed_seconds, 0)}s
              {rd.timestamp && <span className="ml-2">· {formatTs(rd.timestamp)}</span>}
            </span>
          </div>
          {rd.king_retained_reason && (
            <div className="rounded-lg border border-amber-400/25 bg-amber-400/5 px-3 py-2 text-[11px] font-mono text-amber-200">
              👑 King retained — {rd.king_retained_reason}
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-[11px] font-mono">
              <thead>
                <tr className="text-left text-muted-foreground/50 border-b border-border/20">
                  <th className="py-1.5 pr-2">UID</th>
                  <th className="pr-2">Model</th>
                  <th className="pr-2">KL</th>
                  <th className="pr-2">RKL</th>
                  <th className="pr-2">Cap</th>
                  <th className="pr-2">Len</th>
                  <th className="pr-2">Deg</th>
                  <th className="pr-2" title="Teacher-as-judge score (shadow). Normalized from 1-5 rubric on 16 rotated prompts per round.">Judge*</th>
                  <th className="pr-2">Worst</th>
                  <th className="pr-2">vs King</th>
                </tr>
              </thead>
              <tbody>
                {rd.results.map((r) => {
                  const ax = r.composite?.axes || {};
                  const worst = r.composite?.worst;
                  const isDq = r.disqualified === true;
                  const isExpanded = expandedUid === r.uid;
                  return (
                    <>
                      <tr
                        key={r.uid}
                        className={`border-b border-border/10 hover:bg-card/20 cursor-pointer ${isDq ? "opacity-60" : ""}`}
                        onClick={() => setExpandedUid(isExpanded ? null : r.uid)}
                      >
                        <td className="py-1.5 pr-2 tabular-nums">
                          {r.is_king && <span className="text-yellow-400 mr-1">👑</span>}
                          {r.uid}
                        </td>
                        <td className="pr-2 max-w-[180px] truncate text-muted-foreground/80">
                          {r.model.split("/").pop()}
                          {isDq && <span className="ml-1 text-[9px] rounded bg-red-400/15 text-red-400 px-1 py-0.5">DQ</span>}
                        </td>
                        <td className="pr-2 tabular-nums">{formatFixed(r.kl, 4)}</td>
                        <td className={`pr-2 tabular-nums ${axisColor(ax.on_policy_rkl)}`}>{ax.on_policy_rkl == null ? "—" : ax.on_policy_rkl.toFixed(2)}</td>
                        <td className={`pr-2 tabular-nums ${axisColor(ax.capability)}`}>{ax.capability == null ? "—" : ax.capability.toFixed(2)}</td>
                        <td className={`pr-2 tabular-nums ${axisColor(ax.length)}`}>{ax.length == null ? "—" : ax.length.toFixed(2)}</td>
                        <td className={`pr-2 tabular-nums ${axisColor(ax.degeneracy)}`}>{ax.degeneracy == null ? "—" : ax.degeneracy.toFixed(2)}</td>
                        <td
                          className={`pr-2 tabular-nums ${axisColor(ax.judge_probe)} ${r.composite?.judge_in_composite ? "" : "opacity-70"}`}
                          title={r.composite?.judge_in_composite ? "Judge axis in composite" : "Judge axis in SHADOW — displayed only, not in ranking"}
                        >
                          {ax.judge_probe == null ? "—" : ax.judge_probe.toFixed(2)}
                        </td>
                        <td className={`pr-2 tabular-nums font-semibold ${axisColor(worst)}`}>
                          {worst == null ? "—" : worst.toFixed(2)}
                        </td>
                        <td className="pr-2 text-muted-foreground/70 max-w-[140px] truncate">
                          {isDq ? <span className="text-red-400">DQ — not crowned</span> : (r.vs_king || "—")}
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr>
                          <td colSpan={10} className="pb-2 pl-4 text-muted-foreground/60 bg-card/5">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px] py-2">
                              <div>prompts: {r.prompts_scored}/{r.prompts_total}</div>
                              <div>paired: {r.paired_prompts}</div>
                              <div>cap: {r.capability_pass_frac != null ? (r.capability_pass_frac * 100).toFixed(0) + "%" : "—"}
                                {r.capability_teacher != null && <span className="text-muted-foreground/40"> (t {(r.capability_teacher * 100).toFixed(0)}%)</span>}
                              </div>
                              <div>len ratio: {formatFixed(r.length_ratio, 3)}</div>
                              <div>
                                think: <span className={r.think_pass ? "text-emerald-400" : "text-red-400"}>
                                  {r.think_pass == null ? "—" : r.think_pass ? "pass" : "fail"}
                                </span>
                                {r.think_reason && <span className="text-muted-foreground/40"> ({r.think_reason})</span>}
                              </div>
                              {r.adversarial_pass_frac != null && (
                                <div>
                                  adv: {(r.adversarial_pass_frac * 100).toFixed(0)}%
                                  {r.adversarial_mean_tokens != null && (
                                    <span className="text-muted-foreground/40"> · mean {Math.round(r.adversarial_mean_tokens)} tok</span>
                                  )}
                                </div>
                              )}
                              {r.judge_mean_score != null && (
                                <div>
                                  judge: {r.judge_mean_score.toFixed(2)}/5
                                  {r.judge_n_valid != null && r.judge_n != null && (
                                    <span className="text-muted-foreground/40"> · {r.judge_n_valid}/{r.judge_n} parsed</span>
                                  )}
                                  <span className="text-amber-400/70"> (shadow)</span>
                                </div>
                              )}
                              {r.early_stopped && <div className="text-amber-400">early-stopped</div>}
                              {r.dq_reason && (
                                <div className="col-span-2 md:col-span-4 text-red-300">
                                  DQ reason: {r.dq_reason}
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="text-[10px] text-muted-foreground/40 font-mono">
            Composite axes (0–1): worst-of-axis drives reward. Axes = KL (fidelity), RKL (on-policy KL), capability (absolute correctness + teacher-relative), length (≈ teacher length), degeneracy (coherence).
            <span className="ml-1">Judge* = teacher-as-judge rubric score (1–5 normalized). <span className="text-amber-400">SHADOW</span> — visible here but not yet in ranking (48h telemetry before promotion).</span>
          </div>
        </div>
      )}

      {/* Private pool commit */}
      {(pool?.root || reveal?.prompt_hashes) && (
        <div className="rounded-xl border border-purple-400/25 bg-purple-400/[0.02] p-4 space-y-2">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-purple-300/80">
            Private holdout (axis A7) — commit-reveal
          </h3>
          <div className="text-[11px] font-mono space-y-1 text-muted-foreground/70">
            {pool?.block && (
              <div>
                current commit: block #{pool.block.toLocaleString()} · {pool.n ?? "?"} prompts · root <span className="text-purple-300">{pool.root?.slice(0, 24)}…</span>
              </div>
            )}
            {reveal?.block && (
              <div>
                latest reveal: block #{reveal.block?.toLocaleString()} · {reveal.prompt_hashes?.length ?? 0} hashes released
              </div>
            )}
            <div className="text-muted-foreground/40">
              A fraction of each round&apos;s prompts are drawn from a validator-private holdout. The sha256 Merkle root is committed before eval; individual prompt hashes are revealed after, so you can verify the validator didn&apos;t retro-fit the set.
            </div>
          </div>
        </div>
      )}

      {/* Recent DQs */}
      {data.recent_dqs.length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card/10 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Recent disqualifications
            </h3>
            <a href="/api/dq_reasons" className="text-[11px] font-mono text-muted-foreground/40 hover:text-muted-foreground/70">
              full list →
            </a>
          </div>
          <ul className="space-y-1 max-h-80 overflow-auto text-[11px] font-mono">
            {data.recent_dqs.slice(0, 30).map((d, i) => (
              <li key={i} className="flex items-start gap-2 py-0.5">
                <span className="w-14 text-muted-foreground/40 tabular-nums shrink-0">
                  {d.uid != null ? `UID ${d.uid}` : ""}
                </span>
                <span className="w-20 text-muted-foreground/30 tabular-nums shrink-0 truncate">
                  {d.block != null ? `#${d.block}` : ""}
                </span>
                <span className="text-red-300/80 flex-1 break-words">{d.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Validator events */}
      <div className="rounded-xl border border-border/20 bg-card/10 p-4 space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Validator event stream
          </h3>
          <div className="flex gap-1 text-[10px] font-mono">
            {(["all", "warn", "error"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setEventFilter(f)}
                className={`px-2 py-0.5 rounded border transition-colors ${
                  eventFilter === f
                    ? "border-blue-400/50 bg-blue-400/10 text-blue-300"
                    : "border-border/30 text-muted-foreground/60 hover:text-muted-foreground"
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
        <ul className="space-y-0.5 max-h-96 overflow-auto text-[11px] font-mono">
          {events.map((e, i) => (
            <li key={i} className="flex items-start gap-2 py-0.5">
              <span className="w-20 text-muted-foreground/40 tabular-nums shrink-0">
                {formatTs(e.ts)}
              </span>
              <span className={`w-12 shrink-0 uppercase text-[10px] ${levelColor(e.level)}`}>
                {e.level}
              </span>
              <span className="text-muted-foreground/80 break-words">{e.msg}</span>
            </li>
          ))}
          {events.length === 0 && (
            <li className="text-muted-foreground/30 text-center py-4">No events match filter</li>
          )}
        </ul>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  sub,
  accent,
}: {
  title: string;
  value: string;
  sub?: string;
  accent?: "king" | "warn" | "ok";
}) {
  const color = accent === "king"
    ? "text-yellow-400"
    : accent === "warn"
      ? "text-amber-400"
      : accent === "ok"
        ? "text-emerald-400"
        : "text-foreground";
  return (
    <div className="rounded-xl border border-border/20 bg-card/10 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground/50 font-mono">
        {title}
      </div>
      <div className={`text-lg font-bold font-mono tabular-nums ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground/50 font-mono truncate">{sub}</div>}
    </div>
  );
}
