/**
 * Backend lifecycle states → user-facing language.
 *
 * The UI reports only what the backend actually reports. There are no
 * fabricated percentages and no invented intermediate steps.
 */

import type { GenerationStatus } from "./api";

export const STATUS_LABELS: Record<GenerationStatus, string> = {
  QUEUED: "Preparing generation",
  STARTING: "Starting AI model",
  GENERATING: "Creating your music",
  POST_PROCESSING: "Processing audio",
  UPLOADING: "Saving your track",
  COMPLETED: "Track ready",
  FAILED: "Generation failed",
  CANCELLED: "Generation cancelled",
};

export function statusLabel(status: GenerationStatus): string {
  return STATUS_LABELS[status] ?? "Working";
}

/** Ordered states shown as pipeline progress (terminal states excluded). */
export const ACTIVE_STATUS_SEQUENCE: GenerationStatus[] = [
  "QUEUED",
  "STARTING",
  "GENERATING",
  "POST_PROCESSING",
  "UPLOADING",
];

export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || Number.isNaN(seconds)) return "—";
  return formatElapsed(seconds);
}
