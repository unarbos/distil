import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatParams(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return n.toString();
}

export function formatFixed(
  value: number | null | undefined,
  digits: number,
  fallback = "—",
): string {
  return value == null || !Number.isFinite(value) ? fallback : value.toFixed(digits);
}

export function shortRevision(revision?: string | null): string | null {
  if (!revision || revision === "main") return null;
  return revision.slice(0, 8);
}

export function timeAgo(ts: number): string {
  const diff = Math.max(0, Date.now() / 1000 - ts);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function formatPromptCount(
  scored: number | null | undefined,
  total: number | null | undefined,
): string | null {
  if (scored == null) return null;
  if (total == null || total === scored) return `${scored} prompts`;
  return `${scored}/${total} prompts`;
}

export function blocksAgo(block: number | null | undefined, currentBlock: number | null | undefined): string {
  if (block == null || currentBlock == null) return "—";
  const delta = Math.max(0, currentBlock - block);
  const seconds = delta * 12;
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
