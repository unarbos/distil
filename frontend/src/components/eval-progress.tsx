"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";
import { formatFixed } from "@/lib/utils";

interface CompletedStudent {
  student_idx: number;
  student_name: string;
  status: "scored" | "early_stopped" | "functional_copy";
  status_detail: string;
  kl: number;
  prompts_scored: number;
  prompts_total: number;
  scoring_time_s: number;
}

interface EvalOrder {
  uid: number;
  model: string;
  role: "king" | "challenger";
}

interface EvalProgress {
  active: boolean;
  phase?: string;
  models?: Record<string, string>;
  eval_order?: EvalOrder[];
  students_total?: number;
  students_done?: number;
  prompts_total?: number;
  king_uid?: number;
  challenger_uids?: number[];
  started_at?: number;
  estimated_duration_s?: number;
  current_student?: string;
  current_prompt?: number;
  current_kl?: number;
  current_se?: number;
  current_ci?: [number, number];
  current_best?: number;
  completed?: CompletedStudent[];
  teacher_prompts_done?: number;
}

function formatTime(seconds: number): string {
  if (seconds <= 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatEta(seconds: number): string {
  if (seconds <= 0) return "finishing...";
  if (seconds < 60) return `~${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `~${m}m ${s > 0 ? `${s}s` : ""}`;
}

function statusColor(status: string): string {
  switch (status) {
    case "functional_copy": return "text-red-400";
    case "early_stopped": return "text-yellow-400";
    case "scored": return "text-emerald-400";
    default: return "text-muted-foreground";
  }
}

function statusIcon(status: string): string {
  switch (status) {
    case "functional_copy": return "⛔";
    case "early_stopped": return "⏭";
    case "scored": return "✓";
    default: return "·";
  }
}

function _findUid(model: string, models?: Record<string, string>): string | null {
  if (!models) return null;
  for (const [uid, m] of Object.entries(models)) {
    if (m === model) return uid;
  }
  return null;
}

export function EvalProgressBar() {
  const [progress, setProgress] = useState<EvalProgress | null>(null);
  const [now, setNow] = useState(Date.now() / 1000);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/eval-progress`, { cache: "no-store" });
        if (res.ok) {
          const data: EvalProgress = await res.json();
          if (!cancelled) setProgress(data);
        }
      } catch {}
    }
    poll();

    let es: EventSource | null = null;
    try {
      es = new EventSource(`${CLIENT_API_BASE}/api/eval-stream`);
      es.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          if (payload?.progress && !cancelled) setProgress(payload.progress);
        } catch {}
      };
      es.onerror = () => { es?.close(); es = null; };
    } catch {
      es = null;
    }

    const pollId = setInterval(poll, es ? 30_000 : 5_000);
    const tickId = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => {
      cancelled = true;
      clearInterval(pollId);
      clearInterval(tickId);
      es?.close();
    };
  }, []);

  // When idle — show idle state, not last round
  if (!progress?.active) {
    return (
      <div className="rounded-xl border border-border/20 bg-card/10 p-6 text-center space-y-2">
        <div className="flex items-center justify-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="text-sm font-mono text-muted-foreground/60">
            No evaluation running
          </span>
        </div>
        <p className="text-xs text-muted-foreground/40 font-mono">
          Validator is waiting for the next eval round.
          When new models are committed on-chain, they&apos;ll appear here during evaluation.
        </p>
      </div>
    );
  }

  // Active eval — full progress display
  const startedAt = progress.started_at ?? now;
  const elapsed = now - startedAt;
  const estDuration = progress.estimated_duration_s ?? 0;
  const eta = Math.max(0, estDuration - elapsed);
  const nModels = progress.students_total ?? 0;
  const nPrompts = progress.prompts_total ?? 0;
  const completed = progress.completed ?? [];
  const evalOrder = progress.eval_order ?? [];

  const studentsDone = completed.length;
  const isTeacherPhase = ["teacher_loading", "teacher_generation", "teacher_logits",
    "vllm_starting", "vllm_generating", "gpu_precompute"].includes(progress.phase ?? "");
  const teacherFrac = isTeacherPhase && nPrompts > 0
    ? (progress.teacher_prompts_done ?? 0) / nPrompts
    : isTeacherPhase ? 0 : 1;
  const currentPromptFrac = progress.current_prompt && nPrompts > 0
    ? progress.current_prompt / nPrompts
    : 0;
  const totalUnits = 1 + nModels;
  const doneUnits = teacherFrac + studentsDone + currentPromptFrac;
  const overallPct = totalUnits > 0
    ? (doneUnits / totalUnits) * 100
    : (estDuration > 0 ? Math.min(95, (elapsed / estDuration) * 100) : 0);

  return (
    <div className="animate-fade-in rounded-xl border border-blue-400/30 bg-blue-400/[0.04] backdrop-blur-sm p-4 relative overflow-hidden eval-glow">
      <style jsx>{`
        .eval-glow { animation: evalGlow 2s ease-in-out infinite; }
        @keyframes evalGlow {
          0%, 100% { box-shadow: 0 0 8px rgba(96,165,250,0.15); }
          50% { box-shadow: 0 0 20px rgba(96,165,250,0.3); }
        }
      `}</style>

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          <span className="text-xs font-mono text-blue-400 uppercase tracking-wider font-semibold">
            Evaluating
          </span>
          <span className="text-[10px] text-muted-foreground/60 font-mono">
            {nModels} models · {nPrompts} prompts
          </span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
          <span>⏱ {formatTime(elapsed)}</span>
          {estDuration > 0 && eta > 0 && (
            <span className={eta < 60 ? "text-orange-400" : "text-blue-300"}>
              ETA {formatEta(eta)}
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-blue-400/10 rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-gradient-to-r from-blue-400/60 to-blue-400 rounded-full transition-all duration-1000 ease-linear"
          style={{ width: `${Math.max(2, Math.min(98, overallPct))}%` }}
        />
      </div>

      {/* Teacher phases */}
      {["teacher_loading", "teacher_generation", "teacher_logits",
        "vllm_starting", "vllm_generating", "gpu_precompute"].includes(progress.phase ?? "") && (
        <div className="mb-3 px-2 py-2 rounded-lg bg-purple-400/[0.06] border border-purple-400/20">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
            <span className="text-xs font-mono text-purple-200">
              {progress.phase === "teacher_loading" && "Loading teacher model..."}
              {progress.phase === "vllm_starting" && "Starting vLLM server..."}
              {progress.phase === "vllm_generating" && "Generating via vLLM (fast)"}
              {progress.phase === "teacher_generation" && "Generating teacher outputs"}
              {progress.phase === "teacher_logits" && "Extracting teacher logits"}
              {progress.phase === "gpu_precompute" && "Moving logits to GPU + precomputing softmax..."}
            </span>
            {(progress.teacher_prompts_done != null && progress.teacher_prompts_done > 0) && (
              <span className="text-[10px] text-muted-foreground/50 ml-auto">
                {progress.teacher_prompts_done}/{nPrompts} prompts
              </span>
            )}
          </div>
          {nPrompts > 0 && (progress.teacher_prompts_done ?? 0) > 0 && (
            <div className="h-1 bg-purple-400/10 rounded-full overflow-hidden mt-2">
              <div
                className="h-full bg-purple-400/40 rounded-full transition-all duration-500"
                style={{ width: `${((progress.teacher_prompts_done ?? 0) / nPrompts) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Loading student */}
      {progress.phase === "loading_student" && progress.current_student && (
        <div className="mb-3 px-2 py-2 rounded-lg bg-orange-400/[0.06] border border-orange-400/20">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
            <span className="text-xs font-mono text-orange-200 truncate">
              Loading {progress.current_student}...
            </span>
            <span className="text-[10px] text-muted-foreground/50 ml-auto">
              model {(progress.students_done ?? 0) + 1}/{nModels}
            </span>
          </div>
        </div>
      )}

      {/* Scoring */}
      {progress.phase === "scoring" && progress.current_student && (
        <div className="mb-3 px-2 py-2 rounded-lg bg-blue-400/[0.06] border border-blue-400/20">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-xs font-mono text-blue-200 truncate">
              {progress.current_student}
            </span>
            <span className="text-[10px] text-muted-foreground/50 ml-auto">
              {progress.current_prompt ?? 0}/{nPrompts} prompts
            </span>
          </div>

          {progress.current_kl != null && (
            <div className="flex items-center gap-4 text-[11px] font-mono mt-1">
              <span className="text-foreground">
                KL: <span className="text-blue-300 font-semibold">{formatFixed(progress.current_kl, 4)}</span>
              </span>
              {Array.isArray(progress.current_ci) && progress.current_ci.length === 2 && (
                <span className="text-muted-foreground/60">
                  95% CI: [{formatFixed(progress.current_ci[0], 4)}, {formatFixed(progress.current_ci[1], 4)}]
                </span>
              )}
              {progress.current_se != null && (
                <span className="text-muted-foreground/50">±{formatFixed(progress.current_se, 4)}</span>
              )}
              {progress.current_best != null && (
                <span className="text-yellow-400/70 ml-auto">king: {formatFixed(progress.current_best, 4)}</span>
              )}
            </div>
          )}

          {progress.current_prompt != null && nPrompts > 0 && (
            <div className="h-1 bg-blue-400/10 rounded-full overflow-hidden mt-2">
              <div
                className="h-full bg-blue-400/40 rounded-full transition-all duration-500"
                style={{ width: `${(progress.current_prompt / nPrompts) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Model list */}
      <div className="space-y-0.5">
        {completed.map((c) => {
          const uid = _findUid(c.student_name, progress.models);
          return (
            <div key={c.student_idx} className="flex items-center gap-2 text-[11px] font-mono px-2 py-0.5">
              <span className="w-4 text-center">{statusIcon(c.status)}</span>
              <span className={`${statusColor(c.status)} truncate max-w-[200px]`}>{c.student_name}</span>
              {uid && <span className="text-muted-foreground/30">UID {uid}</span>}
              <span className={`ml-auto ${statusColor(c.status)}`}>{c.status_detail}</span>
              <span className="text-muted-foreground/30">{c.prompts_scored}/{c.prompts_total}</span>
              <span className="text-muted-foreground/20">{c.scoring_time_s}s</span>
            </div>
          );
        })}

        {evalOrder
          .filter((e) => {
            const completedNames = new Set(completed.map((c) => c.student_name));
            return !completedNames.has(e.model) && e.model !== progress.current_student;
          })
          .map((entry) => (
            <div key={entry.uid} className="flex items-center gap-2 text-[11px] font-mono px-2 py-0.5 text-muted-foreground/40">
              <span className="w-4 text-center">·</span>
              <span className={entry.role === "king" ? "text-yellow-400/40" : ""}>
                {entry.role === "king" ? "👑 " : ""}{entry.model.length > 40 ? entry.model.slice(0, 37) + "..." : entry.model}
              </span>
              <span className="ml-auto">UID {entry.uid}</span>
              <span className="text-muted-foreground/20">queued</span>
            </div>
          ))}
      </div>
    </div>
  );
}
