"use client";

import { useEffect, useState, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.arbos.life";

interface HealthData {
  king_uid: number | null;
  king_kl: number | null;
  n_scored: number;
  n_disqualified: number;
  last_eval_block: number;
  last_eval_age_min: number;
  eval_active: boolean;
  code_revision?: string;
  eval_progress: null | {
    phase?: string;
    students_total?: number;
    students_done?: number;
    current_student?: string;
    current_prompt?: number;
    prompts_total?: number;
    current_kl?: number;
    current_best?: number;
    teacher_prompts_done?: number;
  };
}

interface ValidatorStatusProps {
  kingUid: number | null;
  kingModel?: string;
  kingRevision?: string;
  onViewDetails?: () => void;
}

const PHASE_LABELS: Record<string, string> = {
  teacher_loading: "Loading teacher model",
  vllm_starting: "Starting vLLM server",
  vllm_generating: "vLLM generation (fast)",
  teacher_generation: "Teacher generation",
  teacher_logits: "Extracting teacher logits",
  gpu_precompute: "Moving logits to GPU",
  loading_student: "Loading student model",
  scoring: "Scoring student",
};

function formatFixed(value: number | null | undefined, digits: number, fallback = "—"): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

function shortRevision(revision?: string): string | null {
  if (!revision) return null;
  return revision.length > 12 ? revision.slice(0, 8) : revision;
}

export function ValidatorStatus({
  kingUid,
  kingModel,
  kingRevision,
  onViewDetails,
}: ValidatorStatusProps) {
  const [health, setHealth] = useState<HealthData | null>(null);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
      if (res.ok) setHealth(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    fetchHealth();
    const id = setInterval(fetchHealth, 10000);
    return () => clearInterval(id);
  }, [fetchHealth]);

  const isActive = health?.eval_active === true;
  const progress = health?.eval_progress;
  const phase = progress?.phase;

  const codeRev = health?.code_revision;

  // When idle, show a minimal revision badge only
  if (!isActive) {
    if (!codeRev) return null;
    return (
      <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground/40 px-1">
        <span>validator rev</span>
        <span className="text-muted-foreground/60" title={codeRev}>{codeRev}</span>
        {health?.last_eval_age_min != null && (
          <span>· last eval {formatFixed(health.last_eval_age_min, 0)}m ago</span>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-blue-400/30 bg-blue-400/[0.04] px-4 py-3">
      <div className="flex flex-col gap-2">
        {/* Row 1: Status headline */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-sm font-medium text-blue-300">
              Evaluating
              {progress?.students_total ? ` — ${progress.students_done ?? 0}/${progress.students_total} models` : ""}
            </span>
          </div>

          {/* King badge */}
          {kingUid != null && (
            <div className="flex items-center gap-1.5 text-xs font-mono text-muted-foreground/60">
              <span className="text-yellow-400">👑</span>
              <span>UID {kingUid}</span>
              {kingModel && <span className="text-muted-foreground/40 hidden sm:inline">({kingModel.split("/").pop()})</span>}
              {shortRevision(kingRevision) && (
                <span className="text-muted-foreground/35 hidden md:inline" title={kingRevision}>
                  @{shortRevision(kingRevision)}
                </span>
              )}
              {health?.king_kl != null && (
                <span className="text-muted-foreground/40">KL {formatFixed(health.king_kl, 4)}</span>
              )}
            </div>
          )}
        </div>

        {/* Row 2: Active eval detail */}
        {progress && (
          <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground/60 pl-4">
            {phase && (
              <span className="text-blue-300/70">
                {PHASE_LABELS[phase] ?? phase}
              </span>
            )}
            {progress.current_student && (
              <span className="text-foreground/60 truncate max-w-[200px]">
                {progress.current_student}
              </span>
            )}
            {phase === "scoring" && progress.current_prompt != null && progress.prompts_total != null && (
              <span className="tabular-nums">
                {progress.current_prompt}/{progress.prompts_total} prompts
              </span>
            )}
            {phase === "teacher_logits" && progress.teacher_prompts_done != null && progress.prompts_total != null && (
              <span className="tabular-nums">
                {progress.teacher_prompts_done}/{progress.prompts_total} prompts
              </span>
            )}
            {progress.current_kl != null && (
              <span className="text-foreground/70">
                KL {formatFixed(progress.current_kl, 4)}
              </span>
            )}
            {progress.current_best != null && (
              <span className="text-muted-foreground/40">
                best {formatFixed(progress.current_best, 4)}
              </span>
            )}
            {onViewDetails && (
              <button
                onClick={onViewDetails}
                className="ml-auto text-blue-400/60 hover:text-blue-400 transition-colors"
              >
                view details →
              </button>
            )}
          </div>
        )}
        {/* Revision footer */}
        {codeRev && (
          <div className="text-[10px] font-mono text-muted-foreground/30 pl-4">
            rev {codeRev}
          </div>
        )}
      </div>
    </div>
  );
}
