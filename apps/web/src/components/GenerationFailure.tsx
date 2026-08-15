"use client";

/**
 * Failure state. Renders only the pre-translated user-facing message —
 * never an exception string, path, or internal identifier.
 *
 * Retry re-submits the same form values under a *new* Idempotency-Key,
 * so it creates a genuinely new generation rather than resurfacing the
 * failed one.
 */

import type { UserFacingError } from "@/lib/errors";

export interface GenerationFailureProps {
  error: UserFacingError;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export function GenerationFailure({ error, onRetry, onDismiss }: GenerationFailureProps) {
  return (
    <section
      aria-labelledby="generation-failure-heading"
      className="rounded-xl border border-[var(--danger-muted)] bg-[var(--danger-muted)] p-6"
    >
      <h2 id="generation-failure-heading" className="text-lg font-semibold text-[var(--danger)]">
        Generation failed
      </h2>
      <p role="alert" className="mt-2 text-sm text-[var(--text-primary)]">
        {error.message}
      </p>
      <div className="mt-5 flex flex-wrap gap-3">
        {error.retryable && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="rounded-lg bg-[var(--brand)] px-5 py-2.5 text-sm font-semibold text-white
              transition-colors hover:bg-[var(--brand)] focus-visible:outline-none
              focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-2
              focus-visible:ring-offset-zinc-950"
          >
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-lg bg-[var(--surface-overlay)] px-5 py-2.5 text-sm font-semibold text-[var(--text-primary)]
              transition-colors hover:bg-[var(--surface-hover)] focus-visible:outline-none
              focus-visible:ring-2 focus-visible:ring-[var(--border-strong)] focus-visible:ring-offset-2
              focus-visible:ring-offset-zinc-950"
          >
            Start over
          </button>
        )}
      </div>
    </section>
  );
}
