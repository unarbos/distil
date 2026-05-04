"use client";

import { useEffect, useState } from "react";
import { CLIENT_API_BASE } from "@/lib/subnet";

/**
 * 2026-05-04 — Sebastian's Discord report:
 *  > "chat functionality doesn't work with the current king. also you
 *     should show somewhere what the current king that is being used
 *     for chat.arbos.life to know what model you are talking to."
 *
 * The chat pod is co-located with the eval pod on the same H200 NVL,
 * so the chat vLLM gets killed every time eval needs the GPU. Chat
 * comes back automatically when ``/api/chat/status`` next sees the
 * server down AND eval is idle (auto-restart in
 * ``api/routes/chat.py:_ensure_chat_server``). Until then the user
 * gets a 500 from Open WebUI with no context.
 *
 * This pill polls ``/api/chat/status`` (cheap — already in the API
 * cache) and surfaces three states:
 *   - ``available`` (green dot): model is loaded, click to chat
 *   - ``paused`` (amber dot): server is down because eval is using
 *     the GPU; will auto-resume when eval ends
 *   - ``offline`` (grey dot): no king or chat pod isn't configured
 *
 * The model name (``king_model`` from the API) is the live king's HF
 * repo id, so users always see exactly which model their chat is
 * talking to. Compact variant for the home / live panels — the
 * site header keeps the existing UID-only chip.
 */

type Quality = {
  long_form_judge: number | null;
  long_gen_coherence: number | null;
  judge_probe: number | null;
  composite_final: number | null;
};

type ChatStatus = {
  available: boolean;
  king_uid: number | null;
  king_model: string | null;
  eval_active: boolean;
  server_running: boolean;
  quality?: Quality;
  note?: string;
};

const POLL_INTERVAL_MS = 30_000;

/**
 * 2026-05-04 — surfaces a warning when the king is producing degraded
 * output. Without this, users hit chat.arbos.life, get back word-salad
 * loops, and assume chat is broken (the eval *is* catching it via
 * long_form_judge ≤ 0.2 and judge_probe = 0.0, the model itself is
 * what's degraded).
 *
 * Threshold logic:
 *  - long_form_judge < 0.3 OR judge_probe < 0.2 → "degraded" warning
 *  - long_form_judge < 0.55 → "fragile" caution
 *  - otherwise no warning
 *
 * 0.3 / 0.2 are the cold-start ceilings observed on the post-Kimi
 * cohort (UID 188 ~0.16 long_form_judge, UID 190 ~0.16 long_form_judge
 * + 0.0 judge_probe). 0.55 is the ~30th percentile of pre-cutover
 * Qwen 4B kings — anything below that hasn't matured yet.
 */
function _qualityBanner(q: Quality | undefined): {
  level: "ok" | "fragile" | "degraded";
  text: string | null;
} {
  if (!q) return { level: "ok", text: null };
  const lfj = q.long_form_judge;
  const jp = q.judge_probe;
  if (
    (typeof lfj === "number" && lfj < 0.3) ||
    (typeof jp === "number" && jp < 0.2)
  ) {
    return {
      level: "degraded",
      text:
        "King is producing degraded text right now (long-form judge " +
        (typeof lfj === "number" ? lfj.toFixed(2) : "—") +
        "). Chat may show looping / word-salad output. The eval " +
        "pipeline will dethrone this king as soon as a stronger " +
        "challenger appears.",
    };
  }
  if (typeof lfj === "number" && lfj < 0.55) {
    return {
      level: "fragile",
      text:
        "King is fragile (long-form judge " +
        lfj.toFixed(2) +
        "). Short answers usually fine; long generations may derail.",
    };
  }
  return { level: "ok", text: null };
}

export function ChatKingPill({
  variant = "inline",
}: {
  variant?: "inline" | "block";
}) {
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const res = await fetch(`${CLIENT_API_BASE}/api/chat/status`, {
          cache: "no-store",
        });
        if (!cancel && res.ok) {
          const j: ChatStatus = await res.json();
          setStatus(j);
          setError(false);
        } else if (!cancel) {
          setError(true);
        }
      } catch {
        if (!cancel) setError(true);
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const state: "available" | "paused" | "offline" = (() => {
    if (!status) return "offline";
    if (status.available) return "available";
    if (status.eval_active) return "paused";
    return "offline";
  })();

  const banner = _qualityBanner(status?.quality);

  // Available + degraded → amber dot (warn) instead of green.
  const dot =
    state === "available"
      ? banner.level === "degraded"
        ? "bg-king"
        : "bg-ok"
      : state === "paused"
      ? "bg-king"
      : "bg-[var(--ink-meta-soft)]";

  // Headline + tooltip text.
  const { headline, tooltip } = (() => {
    if (error || !status) {
      return {
        headline: "Chat: status unknown",
        tooltip: "Couldn't reach /api/chat/status — try refreshing.",
      };
    }
    const uidText =
      status.king_uid != null && status.king_uid >= 0
        ? `UID ${status.king_uid}`
        : "the king";
    if (status.available) {
      const baseHeadline =
        banner.level === "degraded"
          ? `Chat live · ${uidText} (degraded)`
          : `Chat live · ${uidText}`;
      const baseTooltip = status.king_model
        ? `chat.arbos.life is talking to ${status.king_model} (${uidText}). Click to open.`
        : `chat.arbos.life is live. Click to open.`;
      return {
        headline: baseHeadline,
        tooltip: banner.text ? `${baseTooltip}\n\n${banner.text}` : baseTooltip,
      };
    }
    if (status.eval_active) {
      return {
        headline: `Chat paused · ${uidText} being benchmarked`,
        tooltip:
          "The chat pod and eval pod share a single H200, so the chat vLLM is paused while the validator is running an eval round. It auto-resumes when the round finishes (typically ~30–90 min from start).",
      };
    }
    if (status.king_uid == null) {
      return {
        headline: "Chat: no king yet",
        tooltip:
          "No king crowned yet. Chat will go live as soon as the validator crowns one.",
      };
    }
    return {
      headline: `Chat: ${uidText} loading`,
      tooltip:
        status.note || "Chat server is starting; refresh in a few seconds.",
    };
  })();

  // Block variant for the home panel / docs — full pill, with a
  // secondary line showing quality warning when applicable.
  if (variant === "block") {
    return (
      <div className="flex flex-col gap-1">
        <a
          href="https://chat.arbos.life"
          target="_blank"
          rel="noopener noreferrer"
          title={tooltip}
          className="inline-flex items-center gap-2 text-[11px] num text-meta hover:text-foreground transition-colors group"
        >
          <span className="relative inline-flex items-center justify-center w-1.5 h-1.5">
            <span
              className={`absolute inset-0 rounded-full ${dot} ${
                state === "available" && banner.level !== "degraded"
                  ? "live-pulse"
                  : ""
              }`}
            />
          </span>
          <span className="text-foreground font-medium">{headline}</span>
          {status?.king_model && state !== "offline" && (
            <span className="hidden sm:inline text-meta truncate max-w-[260px]">
              · {status.king_model}
            </span>
          )}
          <span className="text-meta group-hover:text-foreground transition-colors">
            ↗
          </span>
        </a>
        {state === "available" && banner.text && (
          <span
            className={`text-[10px] num leading-tight ${
              banner.level === "degraded" ? "text-king" : "text-meta"
            }`}
            title={banner.text}
          >
            {banner.text}
          </span>
        )}
      </div>
    );
  }

  // Inline variant for tight strips — just dot + short headline.
  return (
    <a
      href="https://chat.arbos.life"
      target="_blank"
      rel="noopener noreferrer"
      title={tooltip}
      className="inline-flex items-center gap-1.5 text-[11px] num text-meta hover:text-foreground transition-colors"
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${dot} ${
          state === "available" ? "live-pulse" : ""
        }`}
      />
      <span>{headline}</span>
    </a>
  );
}
