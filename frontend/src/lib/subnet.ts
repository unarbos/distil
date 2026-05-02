import subnetConfig from "./subnet-config.json";

export const SUBNET = subnetConfig;
export const TEACHER = subnetConfig.teacher;
// ``pendingTeacher`` (if present) is a future teacher staged for a swap;
// the current config has the swap already live so this is typically null.
// Kept in the shape so callers don't have to special-case.
export const PENDING_TEACHER =
  (subnetConfig as { pendingTeacher?: typeof subnetConfig.teacher & {
    rolloutDate?: string;
    tokenizerPath?: string;
    rolloutNotes?: string;
  } }).pendingTeacher ?? null;
// ``previousTeacher`` is the teacher we retired on the most recent swap.
// Populated after the 2026-05-02 Qwen→Kimi cutover so the dashboard can
// surface "Previously: Qwen3.6-35B-A3B" telemetry and miners understand
// which era their commitment belongs to.
export const PREVIOUS_TEACHER =
  (subnetConfig as { previousTeacher?: typeof subnetConfig.teacher & {
    retiredAt?: string;
  } }).previousTeacher ?? null;
export const VALIDATOR = subnetConfig.validator;
export const API_SETTINGS = subnetConfig.api;
export const NETUID = subnetConfig.netuid;
export const SCORE_EPSILON = subnetConfig.validator.epsilon;

/**
 * Legacy KL-headline factor — kept for callers that haven't migrated to
 * the composite-first framing. The production ranking key is
 * `composite.worst` (see `scripts/validator/composite.py`); KL is one of
 * 17 axes, not the gate.
 *
 * To dethrone the king under v28, a challenger must beat
 * `king.composite.worst` by at least `SCORE_EPSILON` (3% by default).
 * Use `compositeFloorToBeat(kingWorst)` for new code.
 *
 * @deprecated Prefer composite-worst framing. Will be removed once all
 * callers have migrated.
 */
export const SCORE_TO_BEAT_FACTOR = 1 - SCORE_EPSILON;

/** Composite-worst threshold a challenger must clear to dethrone the king. */
export function compositeFloorToBeat(kingWorst: number | null | undefined): number | null {
  if (kingWorst == null || !Number.isFinite(kingWorst)) return null;
  return kingWorst * (1 + SCORE_EPSILON);
}
export const API_BASE =
  process.env.API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  subnetConfig.api.publicUrl;
export const CLIENT_API_BASE = "";
