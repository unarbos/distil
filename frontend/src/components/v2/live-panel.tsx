"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface EvalProgress {
  active: boolean;
  phase?: string;
  current_student?: string;
  students_done?: number;
  students_total?: number;
  prompts_done?: number;
  prompts_total?: number;
  current_prompt?: number;
  current_kl?: number;
  teacher_prompts_done?: number;
  started_at?: number;
}

interface LogLine {
  ts: number;
  source: string;
  msg: string;
  level?: "info" | "warn" | "ok";
}

const PHASE_LABELS: Record<string, string> = {
  precheck: "Pre-checks (no GPU)",
  arch_check: "Architecture check",
  copy_check: "Duplicate detection",
  fingerprint: "Logit fingerprinting",
  teacher_warmup: "Teacher vLLM warmup",
  teacher_generate: "Teacher continuations",
  scoring: "Student scoring",
  bench: "Bench battery (math/code/reasoning/...)",
  composite: "Composite assembly",
  king_select: "King selection",
  weights: "Setting weights on chain",
  cleanup: "Pod cleanup",
  idle: "Idle — waiting for next round",
};

/**
 * Live tab — eval pipeline + validator log tail.
 *
 * Layout: two-column. Left = phase ladder (done/active/queued) +
 * three stat tiles. Right = scrollable tail of validator events.
 *
 * Phase data comes from /api/eval-progress (existing endpoint). The
 * log tail comes from /api/gpu-logs and /api/incidents merged together
 * — both are already production endpoints.
 */
export function LivePanel() {
  const [progress, setProgress] = useState<EvalProgress | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/eval-progress`, { cache: "no-store" });
        if (res.ok && !cancel) {
          const json = (await res.json()) as EvalProgress;
          setProgress(json);
          if (json.started_at) {
            setElapsed(Math.max(0, Date.now() / 1000 - json.started_at));
          }
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 4_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const [gpuRes, incRes] = await Promise.all([
          fetch(`${CLIENT_API_BASE}/api/gpu-logs?limit=40`, { cache: "no-store" }).catch(() => null),
          fetch(`${CLIENT_API_BASE}/api/incidents?limit=20`, { cache: "no-store" }).catch(() => null),
        ]);
        const lines: LogLine[] = [];
        if (gpuRes && gpuRes.ok) {
          const json = await gpuRes.json();
          const arr: unknown = json?.logs ?? json?.lines ?? json;
          if (Array.isArray(arr)) {
            for (const item of arr) {
              if (typeof item === "string") {
                lines.push({ ts: Date.now() / 1000, source: "gpu", msg: item });
              } else if (item && typeof item === "object") {
                const o = item as Record<string, unknown>;
                lines.push({
                  ts: Number(o.ts ?? o.timestamp ?? Date.now() / 1000),
                  source: String(o.source ?? "gpu"),
                  msg: String(o.msg ?? o.message ?? o.line ?? ""),
                  level: o.level === "warn" || o.level === "ok"
                    ? (o.level as "warn" | "ok")
                    : "info",
                });
              }
            }
          }
        }
        if (incRes && incRes.ok) {
          const json = await incRes.json();
          if (Array.isArray(json?.incidents)) {
            for (const ev of json.incidents) {
              const o = ev as Record<string, unknown>;
              const label = String(o.action ?? o.issue ?? "");
              if (!label) continue;
              const isAction = o.type === "action";
              lines.push({
                ts: Number(o.ts ?? Date.now() / 1000),
                source: "ops",
                msg: label,
                level: isAction ? "ok" : o.resolved ? "info" : "warn",
              });
            }
          }
        }
        lines.sort((a, b) => b.ts - a.ts);
        if (!cancel) setLogs(lines.slice(0, 60));
      } catch {}
    };
    tick();
    const id = setInterval(tick, 8_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const phases = derivePhases(progress);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      {/* Pipeline */}
      <div className="px-6 sm:px-9 py-8 border-b lg:border-b-0 lg:border-r border-border overflow-y-auto flex flex-col gap-6">
        <HeadRow
          title="Eval pipeline"
          meta={
            progress?.active
              ? buildMetaText(progress)
              : "validator idle"
          }
        />
        <div className="flex flex-col gap-3">
          {phases.map((p) => (
            <PhaseRow key={p.label} phase={p} />
          ))}
        </div>
        <StatTiles progress={progress} elapsed={elapsed} />
      </div>

      {/* Log tail */}
      <div className="px-6 sm:px-9 py-8 bg-[var(--surface-soft)] overflow-y-auto flex flex-col gap-6">
        <HeadRow title="Validator log" meta={`tail · ${logs.length}`} />
        <div className="text-[12px] leading-[1.85] num space-y-0.5">
          {logs.length === 0 && (
            <div className="text-meta">Waiting for events…</div>
          )}
          {logs.map((line, i) => (
            <LogRow key={`${line.ts}-${i}`} line={line} />
          ))}
        </div>
      </div>
    </div>
  );
}

interface PhaseEntry {
  state: "done" | "active" | "queued";
  label: string;
  pct: number;
}

function derivePhases(progress: EvalProgress | null): PhaseEntry[] {
  if (!progress?.active) {
    return [
      { state: "queued", label: "Validator idle — waiting for next round", pct: 0 },
    ];
  }
  const phase = progress.phase ?? "scoring";
  const studentsDone = progress.students_done ?? 0;
  const studentsTotal = progress.students_total ?? 0;
  const promptsDone = progress.prompts_done ?? progress.current_prompt ?? 0;
  const promptsTotal = progress.prompts_total ?? 1;
  const teacherDone = progress.teacher_prompts_done ?? 0;

  const phases: PhaseEntry[] = [];

  // Pre-checks
  phases.push({ state: "done", label: "Pre-checks (no GPU)", pct: 100 });

  // Teacher generation
  if (phase === "teacher_warmup" || phase === "teacher_generate") {
    phases.push({
      state: "active",
      label: PHASE_LABELS[phase] ?? phase,
      pct: Math.min(99, Math.round((teacherDone / promptsTotal) * 100)),
    });
  } else {
    phases.push({ state: "done", label: "Teacher continuations", pct: 100 });
  }

  // Student scoring
  if (phase === "scoring") {
    const studentLabel = progress.current_student
      ? `Scoring · ${progress.current_student}`
      : "Scoring students";
    const overall = studentsTotal > 0
      ? (studentsDone + Math.min(1, promptsDone / Math.max(1, promptsTotal))) / studentsTotal
      : 0;
    phases.push({
      state: "active",
      label: studentLabel,
      pct: Math.min(99, Math.round(overall * 100)),
    });
  } else if (
    phase === "bench" ||
    phase === "composite" ||
    phase === "king_select" ||
    phase === "weights" ||
    phase === "cleanup"
  ) {
    phases.push({ state: "done", label: "Scoring students", pct: 100 });
  } else {
    phases.push({ state: "queued", label: "Scoring students", pct: 0 });
  }

  // Bench battery
  if (phase === "bench") {
    phases.push({ state: "active", label: PHASE_LABELS.bench, pct: 60 });
  } else if (phase === "composite" || phase === "king_select" || phase === "weights" || phase === "cleanup") {
    phases.push({ state: "done", label: PHASE_LABELS.bench, pct: 100 });
  } else {
    phases.push({ state: "queued", label: PHASE_LABELS.bench, pct: 0 });
  }

  // Composite + king
  if (phase === "composite" || phase === "king_select") {
    phases.push({
      state: "active",
      label: phase === "composite" ? PHASE_LABELS.composite : PHASE_LABELS.king_select,
      pct: 70,
    });
  } else if (phase === "weights" || phase === "cleanup") {
    phases.push({ state: "done", label: "Composite + king selection", pct: 100 });
  } else {
    phases.push({ state: "queued", label: "Composite + king selection", pct: 0 });
  }

  return phases;
}

function buildMetaText(p: EvalProgress): string {
  const sd = p.students_done ?? 0;
  const st = p.students_total ?? 0;
  const pd = p.prompts_done ?? p.current_prompt ?? 0;
  const pt = p.prompts_total ?? 0;
  if (st > 0 && pt > 0) {
    return `${sd} of ${st} students · ${pd}/${pt} prompts`;
  }
  if (st > 0) return `${sd} of ${st} students`;
  if (pt > 0) return `${pd}/${pt} prompts`;
  return p.phase ?? "running";
}

function HeadRow({ title, meta }: { title: string; meta: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 flex-wrap">
      <h2 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium">
        {title}
      </h2>
      <span className="text-[11px] text-meta num">{meta}</span>
    </div>
  );
}

function PhaseRow({ phase }: { phase: PhaseEntry }) {
  const pctText = phase.state === "queued" ? "—" : `${phase.pct}%`;
  const dotClass =
    phase.state === "done"
      ? "bg-foreground"
      : phase.state === "active"
        ? "bg-ok"
        : "bg-[var(--border-strong)]";
  const labelClass = phase.state === "queued" ? "text-meta" : "text-foreground";
  const pctClass = phase.state === "done" ? "text-foreground" : "text-meta";
  const fillBg = phase.state === "queued" ? "bg-border" : "bg-foreground";

  return (
    <div className="grid grid-cols-[1fr_64px] gap-3 items-center text-[13px]">
      <div className={["flex items-center gap-2.5", labelClass].join(" ")}>
        <span
          className={[
            "w-1.5 h-1.5 rounded-full shrink-0",
            dotClass,
            phase.state === "active" ? "ring-4 ring-ok/15" : "",
          ].join(" ")}
        />
        <span className="truncate">{phase.label}</span>
      </div>
      <div className={["text-[11px] text-right num", pctClass].join(" ")}>{pctText}</div>
      <div className="col-span-2 h-[3px] bg-[#f1f1f1] relative overflow-hidden">
        <div
          className={["absolute inset-y-0 left-0", fillBg].join(" ")}
          style={{ width: `${phase.pct}%` }}
        />
        {phase.state === "active" && <div className="phase-scan" />}
      </div>
    </div>
  );
}

function StatTiles({
  progress,
  elapsed,
}: {
  progress: EvalProgress | null;
  elapsed: number;
}) {
  const sd = progress?.students_done ?? 0;
  const st = progress?.students_total ?? 0;
  const pd = progress?.prompts_done ?? progress?.current_prompt ?? 0;
  const pt = progress?.prompts_total ?? 0;
  const min = Math.floor(elapsed / 60);
  const sec = Math.floor(elapsed % 60);
  return (
    <div className="grid grid-cols-3 border-t border-border pt-5 gap-0">
      <Tile label="Students" big={String(sd)} small={st > 0 ? `/${st}` : ""} />
      <Tile label="Prompts" big={String(pd)} small={pt > 0 ? `/${pt}` : ""} />
      <Tile
        label="Elapsed"
        big={progress?.active ? String(min) : "—"}
        small={progress?.active ? `m ${sec.toString().padStart(2, "0")}s` : ""}
        last
      />
    </div>
  );
}

function Tile({
  label,
  big,
  small,
  last,
}: {
  label: string;
  big: string;
  small: string;
  last?: boolean;
}) {
  return (
    <div className={["pr-4", last ? "" : "border-r border-border"].join(" ")}>
      <div className="text-[10px] uppercase tracking-[0.16em] text-meta">{label}</div>
      <div className="text-[22px] font-medium tracking-[-0.02em] num mt-1">
        {big}
        {small && <span className="text-[11px] text-meta font-normal ml-1">{small}</span>}
      </div>
    </div>
  );
}

function LogRow({ line }: { line: LogLine }) {
  const t = new Date(line.ts * 1000).toISOString().slice(11, 19);
  const dot =
    line.level === "warn" ? "bg-warning" : line.level === "ok" ? "bg-ok" : "";
  return (
    <div className="grid grid-cols-[64px_56px_1fr] gap-3.5">
      <span className="text-[var(--ink-meta-soft)]">{t}</span>
      <span className="text-meta">{line.source}</span>
      <span className="text-foreground">
        {dot && (
          <span
            className={["inline-block w-1.5 h-1.5 mr-2 align-middle", dot].join(" ")}
          />
        )}
        {line.msg}
      </span>
    </div>
  );
}
