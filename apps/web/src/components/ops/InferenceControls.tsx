"use client";

/**
 * The filters, and the two banners that stop a chart from lying.
 *
 * **Staleness.** A projection-backed dashboard can be silently out of
 * date: every chart renders, every rate looks plausible, and the last
 * ingest ran on Tuesday. The banner is how an operator finds that out
 * before acting on it.
 *
 * **Partial history.** Generations from before Phase 29 have no
 * candidate trace, so their retries are unknown rather than zero. Any
 * window containing them says so, because a denominator that quietly
 * shrank is a number nobody can reconcile.
 */

import type { Coverage, IngestStatus, WindowSize } from "@/lib/ops/inference-types";

export const WINDOWS: { value: WindowSize; label: string }[] = [
  { value: "1h", label: "Last hour" },
  { value: "24h", label: "Last 24 hours" },
  { value: "7d", label: "Last 7 days" },
];

export function WindowPicker({
  value,
  onChange,
}: {
  value: WindowSize;
  onChange: (next: WindowSize) => void;
}) {
  return (
    <div role="group" aria-label="Time range" className="flex gap-1">
      {WINDOWS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={
            value === option.value
              ? "rounded-[var(--radius-md)] bg-[var(--surface-overlay)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)]"
              : "rounded-[var(--radius-md)] px-3 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function StaleBanner({ status }: { status: IngestStatus | null }) {
  if (!status || !status.stale) return null;
  return (
    <div
      role="status"
      className="mb-4 rounded-[var(--radius-md)] border border-[var(--accent)]/40 bg-[var(--accent-muted)] px-3 py-2 text-[11px] text-[var(--accent)]"
    >
      <strong>Data may be behind.</strong> {status.note}
    </div>
  );
}

export function PartialDataBanner({ coverage }: { coverage: Coverage | undefined }) {
  if (!coverage?.partial) return null;
  return (
    <div
      role="status"
      className="mb-4 rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-3 py-2 text-[11px] text-[var(--text-secondary)]"
    >
      <strong>Partial data before {coverage.boundary_commit}.</strong> {coverage.note}
    </div>
  );
}
