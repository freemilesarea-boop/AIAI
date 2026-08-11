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
      className="rounded-xl border border-red-900/60 bg-red-950/30 p-6"
    >
      <h2 id="generation-failure-heading" className="text-lg font-semibold text-red-200">
        Generation failed
      </h2>
      <p role="alert" className="mt-2 text-sm text-red-100/90">
        {error.message}
      </p>
      <div className="mt-5 flex flex-wrap gap-3">
        {error.retryable && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="rounded-lg bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white
              transition-colors hover:bg-violet-500 focus-visible:outline-none
              focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2
              focus-visible:ring-offset-zinc-950"
          >
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-lg bg-zinc-800 px-5 py-2.5 text-sm font-semibold text-zinc-100
              transition-colors hover:bg-zinc-700 focus-visible:outline-none
              focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2
              focus-visible:ring-offset-zinc-950"
          >
            Start over
          </button>
        )}
      </div>
    </section>
  );
}
