"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

interface GpuLogsData {
  lines: string[];
  count: number;
}

export function GpuLogs() {
  const [logs, setLogs] = useState<GpuLogsData | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/gpu-logs?lines=80`, {
          cache: "no-store",
        });
        if (!res.ok) return;
        const data: GpuLogsData = await res.json();
        if (!cancelled) setLogs(data);
      } catch {
        // ignore
      }
    }

    poll();
    const id = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const allLines = logs?.lines ?? [];

  const filteredLines = useMemo(() => {
    if (!filter.trim()) return allLines;
    const lower = filter.toLowerCase();
    return allLines.filter((line) => line.toLowerCase().includes(lower));
  }, [allLines, filter]);

  const displayLines = expanded ? filteredLines : filteredLines.slice(-30);

  return (
    <div className="rounded-xl border border-zinc-700/50 bg-zinc-900/50 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2 bg-zinc-800/50 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-emerald-400/80 uppercase tracking-wider font-semibold">
            ▸ Live Logs
          </span>
          <span className="text-[10px] text-muted-foreground/40 font-mono">
            {allLines.length} lines
          </span>
          {filter && (
            <span className="text-[10px] bg-blue-500/10 text-blue-400/70 font-mono px-1.5 py-0.5 rounded">
              showing {filteredLines.length} of {allLines.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            className={`text-[10px] font-mono px-2 py-0.5 rounded ${
              autoScroll ? "bg-emerald-400/10 text-emerald-400/60" : "text-muted-foreground/30"
            }`}
            onClick={(e) => { e.stopPropagation(); setAutoScroll(!autoScroll); }}
          >
            {autoScroll ? "auto-scroll" : "paused"}
          </button>
          <span className="text-[10px] text-muted-foreground/30">
            {expanded ? "▾" : "▸"}
          </span>
        </div>
      </div>

      {/* Filter input */}
      <div className="px-3 pt-2 pb-1" onClick={(e) => e.stopPropagation()}>
        <div className="relative">
          <input
            type="text"
            placeholder="Filter logs..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full text-[11px] font-mono bg-zinc-800/60 border border-zinc-700/40 rounded px-2 py-1 pr-6 text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500/60"
          />
          {filter && (
            <button
              onClick={() => setFilter("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 text-[12px] font-mono leading-none"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Log content */}
      {displayLines.length === 0 ? (
        <div className="px-4 py-6 text-center text-[11px] text-muted-foreground/40 font-mono">
          {filter ? "No matching lines" : "No logs yet"}
        </div>
      ) : (
        <div
          ref={scrollRef}
          className={`overflow-y-auto font-mono text-[11px] leading-relaxed p-3 ${
            expanded ? "max-h-[800px]" : "max-h-[240px]"
          }`}
          onScroll={() => {
            if (!scrollRef.current) return;
            const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
            setAutoScroll(scrollHeight - scrollTop - clientHeight < 30);
          }}
        >
          {displayLines.map((line, i) => (
            <LogLine key={i} line={line} />
          ))}
        </div>
      )}
    </div>
  );
}

// Regex to match timestamps like 2026-03-29 14:30:01 or 14:30:01.123
const TIMESTAMP_RE = /(\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2}:\d{2}([.,]\d+)?/;
// Regex to match timing values like 123.4s or 12.3s
const TIMING_RE = /(\d+\.\d+s)/g;

function LogLine({ line }: { line: string }) {
  // Color based on content
  let cls = "text-muted-foreground/60";

  if (line.startsWith("[GPU]")) {
    cls = "text-emerald-300/70";
    if (line.includes("KL=")) cls = "text-blue-300/80";
    if (line.includes("VRAM:")) cls = "text-purple-300/70";
    if (line.includes("early stop")) cls = "text-yellow-300/80";
    if (line.includes("FRAUD") || line.includes("ERROR") || line.includes("OOM")) cls = "text-red-400/80";
    if (line.includes("vLLM") || line.includes("vllm")) cls = "text-cyan-300/70";
    if (line.includes("✓") || line.includes("DONE")) cls = "text-emerald-400/80";
    if (line.includes("King")) cls = "text-yellow-300/80";
  } else {
    // Validator logs
    if (line.includes("[VALIDATOR]")) cls = "text-zinc-400/70";
    if (line.includes("DISQUALIFIED")) cls = "text-red-400/50";
    if (line.includes("King")) cls = "text-yellow-400/60";
    if (line.includes("Setting weights")) cls = "text-emerald-400/70";
    if (line.includes("Head-to-head")) cls = "text-blue-400/60";
    if (line.includes("Block")) cls = "text-zinc-500/50";
  }

  // Additional color overrides
  if (line.includes("PHASE")) cls = "text-cyan-400/80";
  if (line.includes("Cache")) cls = "text-emerald-400/40";

  // Highlight timestamps and timing values inline
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let idx = 0;

  // Highlight timing values like "123.4s"
  const timingMatches = [...line.matchAll(TIMING_RE)];
  if (timingMatches.length > 0) {
    let lastEnd = 0;
    for (const match of timingMatches) {
      const start = match.index!;
      if (start > lastEnd) {
        parts.push(<span key={`t-${idx++}`}>{remaining.slice(lastEnd, start)}</span>);
      }
      parts.push(
        <span key={`t-${idx++}`} className="text-white/90 font-semibold">
          {match[0]}
        </span>
      );
      lastEnd = start + match[0].length;
    }
    if (lastEnd < remaining.length) {
      parts.push(<span key={`t-${idx++}`}>{remaining.slice(lastEnd)}</span>);
    }
  }

  // Check if line contains a timestamp for subtle highlight
  const hasTimestamp = TIMESTAMP_RE.test(line);

  return (
    <div className={`${cls} whitespace-pre-wrap break-all hover:bg-white/[0.02] px-1 -mx-1 rounded`}>
      {parts.length > 0 ? parts : line}
      {hasTimestamp && !parts.length && null}
    </div>
  );
}
