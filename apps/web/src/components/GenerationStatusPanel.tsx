"use client";

/**
 * Live status for an in-flight generation.
 *
 * Shows only backend-derived state — no fabricated percentages. The
 * spinner conveys activity; elapsed time is reported separately and
 * factually. Updates are announced through an `aria-live` region.
 */

import { useId } from "react";

import type { GenerationStatus } from "@/lib/api";
import {
  ACTIVE_STATUS_SEQUENCE,
  formatElapsed,
  statusLabel,
} from "@/lib/generationStatus";

export interface GenerationStatusPanelProps {
  status: GenerationStatus | null;
  elapsedSeconds: number;
  /** True before the API has confirmed a generation id. */
  submitting?: boolean;
}

export function GenerationStatusPanel({
  status,
  elapsedSeconds,
  submitting = false,
}: GenerationStatusPanelProps) {
  const label = submitting || !status ? "Submitting your request" : statusLabel(status);
  const currentIndex = status ? ACTIVE_STATUS_SEQUENCE.indexOf(status) : -1;
  // Several generations can be in flight at once since Phase 12, so the
  // heading id has to be unique per instance rather than a constant.
  const headingId = useId();

  return (
    <section
      aria-labelledby={headingId}
      className="rounded-xl border border-[var(--border-default)] bg-[var(--surface-raised)] p-6"
    >
      <h2 id={headingId} className="sr-only">
        Generation status
      </h2>

      <div className="flex items-center gap-3">
        <span
          aria-hidden="true"
          className="h-5 w-5 shrink-0 animate-spin rounded-full border-2 border-[var(--border-strong)] border-t-violet-400"
        />
        {/* The visible headline is itself the live region, so screen
            readers announce each state change exactly once. */}
        <p
          role="status"
          aria-live="polite"
          aria-atomic="true"
          className="text-lg font-medium text-[var(--text-primary)]"
        >
          {label}
        </p>
      </div>

      <p className="mt-2 text-sm text-[var(--text-secondary)]">
        Elapsed <span className="font-mono tabular-nums">{formatElapsed(elapsedSeconds)}</span>
      </p>

      <ol className="mt-5 flex flex-col gap-2">
        {ACTIVE_STATUS_SEQUENCE.map((step, index) => {
          const done = currentIndex > index;
          const active = currentIndex === index;
          return (
            <li
              key={step}
              className={`flex items-center gap-2.5 text-sm ${
                active ? "text-[var(--text-primary)]" : done ? "text-[var(--text-muted)]" : "text-[var(--text-muted)]"
              }`}
            >
              <span
                aria-hidden="true"
                className={`h-1.5 w-1.5 rounded-full ${
                  active ? "bg-[var(--brand)]" : done ? "bg-[var(--border-strong)]" : "bg-[var(--surface-overlay)]"
                }`}
              />
              {statusLabel(step)}
              {done && <span className="sr-only">(completed)</span>}
              {active && <span className="sr-only">(in progress)</span>}
            </li>
          );
        })}
      </ol>

      <p className="mt-5 text-xs text-[var(--text-muted)]">
        You can leave this page open. Refreshing will reconnect to this generation.
      </p>
    </section>
  );
}
