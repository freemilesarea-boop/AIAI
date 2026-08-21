"use client";

/**
 * The append-only history of one entity, in the order it happened.
 *
 * Not "recent activity". The Phase 25 audit log is never rewritten or
 * compacted, and this renders it as it is — including the events
 * somebody might later prefer it had not recorded. A console that
 * summarised it would be choosing which parts of a run's history are
 * worth keeping, and that choice belongs to nobody.
 */

import type { AuditEvent } from "@/lib/ops/types";
import { timestamp } from "@/lib/ops/format";

export function AuditList({ events }: { events: AuditEvent[] }) {
  if (events.length === 0) {
    return <p className="text-xs text-[var(--text-muted)]">No events recorded.</p>;
  }
  return (
    <ol className="space-y-1.5">
      {events.map((event, index) => (
        <li
          key={`${event.timestamp}-${index}`}
          className="flex flex-wrap items-baseline gap-2 border-b border-[var(--border-subtle)] pb-1.5 last:border-0"
        >
          <span className="font-mono text-[11px] text-[var(--text-muted)]">
            {timestamp(event.timestamp)}
          </span>
          <span className="text-xs font-medium text-[var(--text-primary)]">{event.event}</span>
          <span className="text-[11px] break-all text-[var(--text-muted)]">
            {Object.entries(event.metadata)
              .filter(([, value]) => value !== null && value !== undefined)
              .map(([key, value]) => `${key}=${String(value)}`)
              .join("  ")}
          </span>
        </li>
      ))}
    </ol>
  );
}
