"use client";

/**
 * Pre-flight advisories, shown next to the lyrics they describe.
 *
 * These are suggestions, never gates. The panel says so in words, the
 * Generate button ignores them entirely, and nothing here offers to
 * "fix" the user's lyrics — an advisory is a note beside the text, not
 * an edit to it.
 */

import type { Advisory } from "@/lib/songcraft";

export interface AdvisoryListProps {
  advisories: Advisory[];
  checking?: boolean;
}

const LEVEL_STYLES: Record<string, { dot: string; label: string }> = {
  warning: { dot: "bg-[var(--accent)]", label: "Warning" },
  info: { dot: "bg-[var(--info)]", label: "Note" },
};

export function AdvisoryList({ advisories, checking = false }: AdvisoryListProps) {
  if (advisories.length === 0) {
    return checking ? (
      <p className="mt-2 text-xs text-[var(--text-muted)]" role="status">
        Checking lyrics…
      </p>
    ) : null;
  }

  // Warnings before notes: the ordering is the triage.
  const ordered = [...advisories].sort((a, b) => {
    if (a.level === b.level) return 0;
    return a.level === "warning" ? -1 : 1;
  });

  return (
    <section
      aria-labelledby="advisory-heading"
      className="mt-3 rounded-lg border border-[var(--border-default)] bg-[var(--surface-raised)] p-3"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 id="advisory-heading" className="text-sm font-medium text-[var(--text-primary)]">
          Suggestions ({ordered.length})
        </h3>
        <p className="text-xs text-[var(--text-muted)]">Advice only — these never block generation.</p>
      </div>

      <ul className="mt-2 flex flex-col gap-2" role="status" aria-live="polite">
        {ordered.map((advisory) => {
          const style = LEVEL_STYLES[advisory.level] ?? LEVEL_STYLES.info;
          return (
            <li key={`${advisory.code}-${advisory.message}`} className="flex gap-2 text-sm">
              <span
                aria-hidden="true"
                className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${style.dot}`}
              />
              <span className="text-[var(--text-secondary)]">
                <span className="sr-only">{style.label}: </span>
                {advisory.message}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
