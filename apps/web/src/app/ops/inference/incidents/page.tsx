"use client";

/**
 * Every incident, worst first.
 *
 * A list rather than a feed: one row per logical regression, updated as
 * the evidence accumulates. A detector running every few minutes would
 * otherwise produce a page of identical rows, and a page an operator
 * skims is a page where the one that mattered gets skimmed too.
 */

import Link from "next/link";
import { useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import { OpsEmpty, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference } from "@/lib/ops/inference-client";

const PAGE = 25;

const SEVERITY_TONE: Record<string, string> = {
  CRITICAL: "border-[var(--danger,#dc2626)] text-[var(--danger,#dc2626)]",
  MAJOR: "border-[var(--accent)] text-[var(--accent)]",
  MINOR: "border-[var(--border-default)] text-[var(--text-secondary)]",
  INFO: "border-[var(--border-default)] text-[var(--text-muted)]",
};

export default function IncidentsPage() {
  const [includeClosed, setIncludeClosed] = useState(false);
  const [offset, setOffset] = useState(0);

  const incidents = useOpsResource(
    () => inference.incidents({ include_closed: includeClosed, limit: PAGE, offset }),
    { deps: [includeClosed, offset], intervalMs: 60_000 },
  );

  return (
    <>
      <OpsHeader
        title="Incidents"
        description="One row per logical regression. Detection only — nothing here was acted on automatically."
        onRefresh={incidents.refresh}
        refreshing={incidents.refreshing}
      >
        <label className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={includeClosed}
            onChange={(event) => {
              setIncludeClosed(event.target.checked);
              setOffset(0);
            }}
          />
          Include resolved and dismissed
        </label>
      </OpsHeader>

      {incidents.error && <PanelError message={incidents.error} onRetry={incidents.refresh} />}

      <Panel
        title={includeClosed ? "All incidents" : "Open incidents"}
        subtitle={incidents.data ? `${incidents.data.total} total.` : undefined}
      >
        {incidents.loading ? (
          <SectionSkeleton />
        ) : !incidents.data?.items.length ? (
          <OpsEmpty
            title={includeClosed ? "No incidents recorded" : "No open incidents"}
            description="Either nothing crossed a threshold, or there were too few samples to say."
          />
        ) : (
          <ul>
            {incidents.data.items.map((incident) => (
              <li
                key={incident.incident_id}
                className="border-b border-[var(--border-subtle)] py-2.5 last:border-0"
              >
                <div className="flex flex-wrap items-baseline gap-2">
                  <span
                    className={`rounded-[var(--radius-sm)] border px-1.5 py-0.5 text-[10px] font-medium ${
                      SEVERITY_TONE[incident.severity] ?? SEVERITY_TONE.INFO
                    }`}
                  >
                    {incident.severity}
                  </span>
                  <span className="rounded-[var(--radius-sm)] border border-[var(--border-default)] px-1.5 py-0.5 text-[10px] text-[var(--text-muted)]">
                    {incident.status}
                  </span>
                  <Link
                    href={`/ops/inference/incidents/${incident.incident_id}`}
                    className="text-xs font-medium text-[var(--text-primary)] hover:text-[var(--brand-text)]"
                  >
                    {incident.finding_type}
                  </Link>
                  <span className="text-[11px] text-[var(--text-muted)]">
                    {incident.category}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] text-[var(--text-secondary)]">
                  {incident.segment_label}
                </p>
                <p className="mt-0.5 text-[11px] text-[var(--text-muted)]">
                  First seen {incident.first_seen ?? "—"} · last seen{" "}
                  {incident.last_seen ?? "—"} · {incident.occurrence_count} occurrences
                </p>
              </li>
            ))}
          </ul>
        )}

        {incidents.data && incidents.data.total > PAGE && (
          <div className="mt-3 flex items-center justify-between text-xs">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
              className="rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-[var(--text-secondary)] disabled:opacity-40"
            >
              Previous
            </button>
            <span className="text-[var(--text-muted)]">
              {offset + 1}–{Math.min(offset + PAGE, incidents.data.total)} of{" "}
              {incidents.data.total}
            </span>
            <button
              type="button"
              disabled={offset + PAGE >= incidents.data.total}
              onClick={() => setOffset(offset + PAGE)}
              className="rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-[var(--text-secondary)] disabled:opacity-40"
            >
              Next
            </button>
          </div>
        )}
      </Panel>
    </>
  );
}
