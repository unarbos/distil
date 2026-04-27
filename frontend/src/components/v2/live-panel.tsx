"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface EvalOrderItem {
  uid: number;
  model: string;
  role: "king" | "challenger";
}

interface CompletedStudent {
  student_idx?: number;
  student_name: string;
  status: string;
  status_detail?: string;
  kl?: number;
  prompts_scored?: number;
  prompts_total?: number;
  scoring_time_s?: number;
}

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
  estimated_completion?: number;
  estimated_duration_s?: number;
  eval_order?: EvalOrderItem[];
  king_uid?: number;
  completed?: CompletedStudent[];
  models?: Record<string, string>;
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
  vllm_starting: "Starting vLLM teacher pod",
  teacher_warmup: "Teacher vLLM warmup",
  teacher_generation: "Teacher continuations",
  teacher_generate: "Teacher continuations",
  teacher_logits: "Teacher logit precompute",
  loading_student: "Loading student model",
  scoring: "Student scoring",
  bench: "Bench battery (math/code/reasoning/...)",
  composite: "Composite assembly",
  king_select: "King selection (composite.worst)",
  weights: "Setting weights on chain",
  cleanup: "Pod cleanup",
  idle: "Idle — waiting for next round",
};

/**
 * Human-readable explainer per phase. The dashboard's job is to make
 * sure miners can answer the question "is the eval running?" without
 * pinging the Discord channel — that question came up three times in
 * 50 minutes on 2026-04-27. So when a miner opens the Live tab they
 * should see a sentence telling them what is happening RIGHT NOW.
 */
const PHASE_EXPLAINERS: Record<string, string> = {
  precheck:
    "Pre-checks (architecture / duplicate / integrity). No GPU yet. Models that fail here are skipped.",
  vllm_starting:
    "Spinning up the teacher vLLM pod (~30s). After this we'll generate teacher continuations on every prompt.",
  teacher_warmup:
    "Warming up the teacher (~30s). After this we'll generate teacher continuations on every prompt.",
  teacher_generation:
    "Teacher generating continuations on the round's 300 block-seeded prompts. Counts go up here, students wait their turn.",
  teacher_generate:
    "Teacher generating continuations on the round's 300 block-seeded prompts.",
  teacher_logits:
    "Caching teacher logits (~1 min) — final teacher phase before students score.",
  loading_student:
    "Loading the next student into vLLM (~1 min). Students score sequentially, one at a time.",
  scoring:
    "Scoring the current student against the cached teacher logits. ~3 min/student typical. KL is one of 17 axes — the bench battery follows.",
  bench:
    "Running the bench battery: math, code, reasoning, IFEval, AIME, MBPP, tool-use, long-context, robustness. Procedural items, block-seeded.",
  composite:
    "Computing composite.worst across 17 axes for every student. composite.worst is the ranking key.",
  king_select:
    "Selecting the king (highest composite.worst, 3% margin to dethrone the incumbent).",
  weights:
    "Setting weights on chain. King gets 1.0, everyone else 0.0.",
  cleanup:
    "Round finished. Cleaning up the pod and resuming the chat-king vLLM.",
  idle:
    "Validator idle. New on-chain commitments will be picked up at the next round boundary.",
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
  const phase = progress?.phase ?? (progress?.active ? "running" : "idle");
  const explainer =
    PHASE_EXPLAINERS[phase] ??
    (progress?.active ? "Validator running…" : PHASE_EXPLAINERS.idle);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      {/* Pipeline + queue + log */}
      <div className="px-6 sm:px-9 py-8 border-b lg:border-b-0 lg:border-r border-border overflow-y-auto flex flex-col gap-6">
        <HeadRow
          title="Eval pipeline"
          meta={progress?.active ? buildMetaText(progress) : "validator idle"}
        />

        {/* Plain-English explainer — addresses the 'is the eval running?'
            question that came up 3× in 50 minutes on Discord. */}
        <div className="-mt-2 px-3.5 py-2.5 border border-border bg-[var(--surface-soft)] text-[12px] leading-relaxed">
          <div className="text-[10px] uppercase tracking-[0.18em] text-meta mb-1">
            Right now
          </div>
          <div>
            <strong className="text-foreground">{PHASE_LABELS[phase] ?? phase}</strong>
            {progress?.current_student && (
              <span className="text-meta">
                {" · "}
                {progress.current_student}
              </span>
            )}
          </div>
          <div className="text-meta mt-1">{explainer}</div>
        </div>

        <div className="flex flex-col gap-3">
          {phases.map((p) => (
            <PhaseRow key={p.label} phase={p} />
          ))}
        </div>
        <StatTiles progress={progress} elapsed={elapsed} />
      </div>

      {/* Right column: eval queue + log tail */}
      <div className="px-6 sm:px-9 py-8 bg-[var(--surface-soft)] overflow-y-auto flex flex-col gap-6">
        <EvalQueue progress={progress} />

        <div>
          <HeadRow title="Validator log" meta={`tail · ${logs.length}`} />
          <div className="text-[12px] leading-[1.85] num space-y-0.5 mt-3">
            {logs.length === 0 && (
              <div className="text-meta">Waiting for events…</div>
            )}
            {logs.map((line, i) => (
              <LogRow key={`${line.ts}-${i}`} line={line} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Eval queue with per-student status.
 *
 * Status derives from `eval_order` + `completed`. A student is:
 *  - ✓ completed: present in `completed[]` with status set.
 *  - ● current: matches `current_student`.
 *  - ○ queued: in `eval_order` but not yet completed or current.
 */
interface EvalQueueProps {
  progress: EvalProgress | null;
}

function EvalQueue({ progress }: EvalQueueProps) {
  const queue = progress?.eval_order ?? [];
  if (!progress?.active && queue.length === 0) {
    return (
      <div>
        <HeadRow title="Eval queue" meta="—" />
        <p className="text-[12px] text-meta mt-3">
          Validator idle. New on-chain commitments will be picked up at the next
          round boundary.
        </p>
      </div>
    );
  }

  const completed: Record<string, CompletedStudent> = {};
  for (const c of progress?.completed ?? []) {
    completed[c.student_name] = c;
  }
  const currentName = progress?.current_student;
  const totalDone = Object.keys(completed).length;
  const eta = progress?.estimated_completion;
  const etaText = (() => {
    if (!eta || eta < Date.now() / 1000) return null;
    const dt = Math.max(0, eta - Date.now() / 1000);
    if (dt < 60) return `~${Math.round(dt)}s`;
    const m = Math.floor(dt / 60);
    if (m < 60) return `~${m}m`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `~${h}h ${rm}m`;
  })();

  return (
    <div>
      <HeadRow
        title="Eval queue"
        meta={
          queue.length > 0
            ? `${totalDone} of ${queue.length} done${etaText ? ` · ETA ${etaText}` : ""}`
            : `ETA ${etaText ?? "—"}`
        }
      />
      <div className="mt-3 flex flex-col">
        {queue.map((item, idx) => {
          const isCurrent = currentName === item.model;
          const isDone = !isCurrent && item.model in completed;
          const status: "done" | "current" | "queued" = isDone
            ? "done"
            : isCurrent
              ? "current"
              : "queued";
          const c = completed[item.model];
          return (
            <QueueRow
              key={`${item.uid}-${idx}`}
              status={status}
              uid={item.uid}
              model={item.model}
              role={item.role}
              completed={c}
              current={isCurrent ? progress : null}
            />
          );
        })}
      </div>
    </div>
  );
}

interface QueueRowProps {
  status: "done" | "current" | "queued";
  uid: number;
  model: string;
  role: "king" | "challenger";
  completed?: CompletedStudent;
  current?: EvalProgress | null;
}

function QueueRow({ status, uid, model, role, completed, current }: QueueRowProps) {
  const marker =
    status === "done" ? "✓" : status === "current" ? "●" : "○";
  const markerClass =
    status === "done"
      ? "text-ok"
      : status === "current"
        ? "text-foreground"
        : "text-[var(--border-strong)]";
  const labelClass = status === "queued" ? "text-meta" : "text-foreground";
  const isReference = uid === -1;
  const isKing = role === "king";

  const detail = (() => {
    if (status === "done" && completed) {
      const klPart =
        typeof completed.kl === "number" && Number.isFinite(completed.kl)
          ? `KL ${completed.kl.toFixed(4)}`
          : completed.status_detail ?? completed.status;
      return klPart;
    }
    if (status === "current" && current) {
      const pd = current.current_prompt ?? current.prompts_done ?? 0;
      const pt = current.prompts_total ?? 0;
      const klPart =
        typeof current.current_kl === "number"
          ? ` · KL ${current.current_kl.toFixed(4)}`
          : "";
      return `${pd}/${pt} prompts${klPart}`;
    }
    return "queued";
  })();

  return (
    <div
      className={[
        "grid grid-cols-[16px_60px_1fr_auto] items-center gap-2.5 py-2 border-b border-border last:border-b-0 text-[13px]",
        status === "current" ? "bg-[var(--surface-elevated)] -mx-2 px-2 ring-1 ring-foreground/10" : "",
      ].join(" ")}
    >
      <span className={["text-center font-medium", markerClass].join(" ")}>
        {marker}
      </span>
      <span className={["num", labelClass].join(" ")}>
        {isReference ? "ref" : `#${uid}`}
        {isKing && <span className="text-meta text-[10px] ml-1">♛</span>}
      </span>
      <span
        className={[
          "truncate min-w-0",
          status === "queued" ? "text-meta" : "text-foreground",
        ].join(" ")}
        title={model}
      >
        {model}
      </span>
      <span className="text-[11px] text-meta num text-right whitespace-nowrap">
        {detail}
      </span>
    </div>
  );
}

interface PhaseEntry {
  state: "done" | "active" | "queued";
  label: string;
  pct: number;
}

// Set of phase tags emitted by the validator. Comprehensive enough that
// derivePhases() can always classify a phase as before/at/after each
// pipeline step. Source of truth: ``state/eval_progress.json`` updates
// in pod_eval_vllm.py.
const TEACHER_PHASES = new Set([
  "vllm_starting",
  "teacher_warmup",
  "teacher_generation",
  "teacher_generate",
  "teacher_logits",
]);
const STUDENT_PHASES = new Set(["loading_student", "scoring"]);
const BENCH_PHASES = new Set(["bench"]);
const POST_PHASES = new Set([
  "composite",
  "king_select",
  "weights",
  "cleanup",
]);

function derivePhases(progress: EvalProgress | null): PhaseEntry[] {
  if (!progress?.active) {
    return [
      { state: "queued", label: "Validator idle — waiting for next round", pct: 0 },
    ];
  }
  const phase = progress.phase ?? "scoring";
  const studentsDone = progress.students_done ?? 0;
  const studentsTotal = progress.students_total ?? 0;
  const promptsTotal = progress.prompts_total ?? 1;
  const teacherDone = progress.teacher_prompts_done ?? 0;
  const currentPrompt = progress.current_prompt ?? 0;

  const phases: PhaseEntry[] = [];
  phases.push({ state: "done", label: "Pre-checks (no GPU)", pct: 100 });

  // ── Teacher continuations + logit cache. The right counter here is
  //    teacher_prompts_done — NOT prompts_done. The latter only ticks
  //    once a student starts scoring; reading it during the teacher
  //    phase produces the misleading "98 min · 0 prompts" that miners
  //    were seeing on Discord on 2026-04-27. ─────────────────────────
  if (TEACHER_PHASES.has(phase)) {
    const pct = promptsTotal > 0
      ? Math.min(99, Math.round((teacherDone / promptsTotal) * 100))
      : 0;
    phases.push({
      state: "active",
      label: PHASE_LABELS[phase] ?? "Teacher continuations",
      pct,
    });
  } else {
    phases.push({ state: "done", label: "Teacher continuations", pct: 100 });
  }

  // ── Student scoring. During `loading_student` we count the
  //    just-completed students (studentsDone of studentsTotal). During
  //    `scoring` we add the current_prompt fraction of the active
  //    student's slice. ────────────────────────────────────────────────
  if (STUDENT_PHASES.has(phase)) {
    const studentLabel = progress.current_student
      ? `Scoring · ${progress.current_student}`
      : phase === "loading_student"
        ? "Loading student"
        : "Scoring students";
    const overall = studentsTotal > 0
      ? (studentsDone +
          (phase === "scoring"
            ? Math.min(1, currentPrompt / Math.max(1, promptsTotal))
            : 0)) / studentsTotal
      : 0;
    phases.push({
      state: "active",
      label: studentLabel,
      pct: Math.min(99, Math.round(overall * 100)),
    });
  } else if (BENCH_PHASES.has(phase) || POST_PHASES.has(phase)) {
    phases.push({ state: "done", label: "Scoring students", pct: 100 });
  } else {
    phases.push({ state: "queued", label: "Scoring students", pct: 0 });
  }

  // ── Bench battery + composite + king select ──────────────────────
  if (BENCH_PHASES.has(phase)) {
    phases.push({ state: "active", label: PHASE_LABELS.bench, pct: 60 });
  } else if (POST_PHASES.has(phase)) {
    phases.push({ state: "done", label: PHASE_LABELS.bench, pct: 100 });
  } else {
    phases.push({ state: "queued", label: PHASE_LABELS.bench, pct: 0 });
  }

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

/**
 * Phase-aware meta text for the pipeline header. The right counter
 * depends on which phase we're in. Reading prompts_done during teacher
 * generation gives 0 (the misleading "0 prompts" surface bug).
 */
function buildMetaText(p: EvalProgress): string {
  const phase = p.phase ?? "";
  const teacherDone = p.teacher_prompts_done ?? 0;
  const promptsTotal = p.prompts_total ?? 0;
  const studentsDone = p.students_done ?? 0;
  const studentsTotal = p.students_total ?? 0;
  const currentPrompt = p.current_prompt ?? 0;

  if (TEACHER_PHASES.has(phase)) {
    if (promptsTotal > 0) {
      return `teacher ${teacherDone}/${promptsTotal} prompts`;
    }
    return PHASE_LABELS[phase] ?? phase;
  }
  if (STUDENT_PHASES.has(phase)) {
    if (studentsTotal > 0 && promptsTotal > 0) {
      return `${studentsDone}/${studentsTotal} students · ${currentPrompt}/${promptsTotal} on current`;
    }
    return PHASE_LABELS[phase] ?? phase;
  }
  if (BENCH_PHASES.has(phase)) return "bench battery running";
  if (POST_PHASES.has(phase)) return PHASE_LABELS[phase] ?? phase;
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
      <div className="col-span-2 h-[3px] bg-[var(--track)] relative overflow-hidden">
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
  const pt = progress?.prompts_total ?? 0;
  const phase = progress?.phase ?? "";
  // The "Prompts" tile shows whichever counter is currently advancing
  // — teacher_prompts_done during teacher phases, current_prompt
  // during scoring. This mirrors the validator's mental model so the
  // dashboard agrees with what miners see in Discord screenshots.
  const promptsValue = TEACHER_PHASES.has(phase)
    ? progress?.teacher_prompts_done ?? 0
    : progress?.current_prompt ?? progress?.prompts_done ?? 0;
  const promptsLabel = TEACHER_PHASES.has(phase)
    ? "Teacher prompts"
    : STUDENT_PHASES.has(phase)
      ? "Current student"
      : "Prompts";

  const min = Math.floor(elapsed / 60);
  const sec = Math.floor(elapsed % 60);
  return (
    <div className="grid grid-cols-3 border-t border-border pt-5 gap-0">
      <Tile label="Students" big={String(sd)} small={st > 0 ? `/${st}` : ""} />
      <Tile
        label={promptsLabel}
        big={String(promptsValue)}
        small={pt > 0 ? `/${pt}` : ""}
      />
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
