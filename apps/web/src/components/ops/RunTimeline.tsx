"use client";

/**
 * The Phase 25 run state machine, drawn as it actually is.
 *
 * Not a progress bar. A progress bar implies a fraction, and a run does
 * not have one — it has a position in a state machine with four
 * different endings, three of which mean something went wrong and one of
 * which (LOST) means nobody knows.
 *
 * The terminal states are all shown, including the ones this run did not
 * take, so the shape of the machine stays visible. An operator looking
 * at a LOST run should be able to see, without knowing the system, that
 * LOST is not where COMPLETED would have been.
 */

import { OpsStatus } from "@/components/ops/primitives";
import { cx } from "@/components/ui";
import { timestamp } from "@/lib/ops/format";
import type { TimelineEntry } from "@/lib/ops/types";

export function RunTimeline({ entries }: { entries: TimelineEntry[] }) {
  const linear = entries.filter((entry) => !entry.terminal);
  const terminal = entries.filter((entry) => entry.terminal);

  return (
    <div className="space-y-3">
      <ol className="flex flex-wrap items-center gap-x-1 gap-y-2">
        {linear.map((entry, index) => (
          <li key={entry.state} className="flex items-center gap-1">
            {index > 0 && (
              <span
                aria-hidden="true"
                className={cx(
                  "h-px w-5",
                  entry.reached ? "bg-[var(--brand)]" : "bg-[var(--border-default)]",
                )}
              />
            )}
            <span
              className={cx(
                "inline-flex items-center gap-1.5 rounded-[var(--radius-full)] px-2 py-0.5 text-[11px]",
                entry.current
                  ? "bg-[var(--brand-muted)] text-[var(--brand-text)] ring-1 ring-[var(--brand)]"
                  : entry.reached
                    ? "bg-[var(--surface-overlay)] text-[var(--text-secondary)]"
                    : "text-[var(--text-muted)] opacity-60",
              )}
              title={entry.at ? timestamp(entry.at) : undefined}
            >
              <span aria-hidden="true" className="text-[9px]">
                {entry.reached ? "✓" : "○"}
              </span>
              {entry.state}
            </span>
          </li>
        ))}
      </ol>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] text-[var(--text-muted)]">Ended:</span>
        {terminal.map((entry) => (
          <span
            key={entry.state}
            className={entry.reached ? "" : "opacity-40"}
            title={entry.at ? timestamp(entry.at) : undefined}
          >
            <OpsStatus status={entry.state} />
          </span>
        ))}
      </div>

      <ul className="space-y-0.5 text-[11px] text-[var(--text-muted)]">
        {entries
          .filter((entry) => entry.reached && entry.at)
          .map((entry) => (
            <li key={`${entry.state}-at`}>
              {entry.state} · {timestamp(entry.at)}
            </li>
          ))}
      </ul>
    </div>
  );
}
