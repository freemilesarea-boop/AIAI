"use client";

/**
 * One generation's candidate phase, told the way Phase 29 tells it.
 *
 * "Attempt 1 rejected: EARLY_COLLAPSE. Attempt 2 selected. Provider
 * calls: 2. Quality retries: 1." — the same sentences the CLI prints,
 * built server-side so the two cannot drift.
 *
 * What is absent is the point: there is no prompt, no lyrics and no
 * title on this page, because neither the projection nor the Phase 29
 * trace holds one. An operator debugging a collapse does not need to
 * read what somebody asked for.
 */

import { use } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import { KeyValue, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference } from "@/lib/ops/inference-client";

function seconds(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(2)}s`;
}

export default function GenerationDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const generation = useOpsResource(() => inference.generation(id), { deps: [id] });
  const data = generation.data;

  return (
    <>
      <OpsHeader
        title="Generation"
        breadcrumb={[{ href: "/ops/inference/generations", label: "Generations" }]}
        description={id}
        onRefresh={generation.refresh}
        refreshing={generation.refreshing}
      />

      {generation.error && (
        <PanelError message={generation.error} onRetry={generation.refresh} />
      )}

      {generation.loading || !data ? (
        <SectionSkeleton />
      ) : (
        <>
          <Panel title="What was asked for, as far as analytics records it">
            <KeyValue
              columns={2}
              items={[
                { label: "When", value: new Date(data.occurred_at).toLocaleString() },
                { label: "Status", value: data.generation_status },
                { label: "Provider", value: data.provider },
                { label: "Revision", value: data.provider_revision },
                { label: "Task", value: data.task_type },
                { label: "Duration bucket", value: data.duration_bucket },
                {
                  label: "Requested duration",
                  value:
                    data.requested_duration_seconds !== null
                      ? `${data.requested_duration_seconds}s`
                      : "UNKNOWN",
                },
                { label: "Language", value: data.language, hint: "Explicit metadata only" },
                { label: "Instrumental", value: data.instrumental },
                { label: "QC policy", value: data.qc_policy },
                { label: "Failure code", value: data.generation_failure_code ?? "—" },
                { label: "Finishing", value: data.finishing_outcome ?? "—" },
              ]}
            />
          </Panel>

          <div className="mt-4" />
          <Panel title="What the candidate phase did">
            {!data.qc_data_available ? (
              <p className="text-[11px] text-[var(--text-muted)]">
                This generation predates the candidate trace, so its retry count is unknown
                rather than zero.
              </p>
            ) : (
              <>
                <ul className="mb-3 space-y-0.5">
                  {data.explanation.map((line) => (
                    <li key={line} className="text-[11px] text-[var(--text-secondary)]">
                      {line}
                    </li>
                  ))}
                </ul>
                <KeyValue
                  columns={3}
                  items={[
                    { label: "Candidates", value: String(data.candidate_count ?? "—") },
                    { label: "Provider calls", value: String(data.provider_call_count ?? "—") },
                    { label: "Quality retries", value: String(data.quality_retry_count ?? "—") },
                    {
                      label: "Selected on attempt",
                      value:
                        data.selected_on_attempt !== null
                          ? String(data.selected_on_attempt + 1)
                          : "none",
                    },
                    {
                      label: "First candidate accepted",
                      value:
                        data.first_candidate_accepted === null
                          ? "UNKNOWN"
                          : data.first_candidate_accepted
                            ? "yes"
                            : "no",
                    },
                    {
                      label: "Retry exhausted",
                      value: data.retry_exhausted ? "yes" : "no",
                    },
                  ]}
                />
              </>
            )}
          </Panel>

          <div className="mt-4" />
          <Panel title="Attempts">
            {data.attempts.length === 0 ? (
              <p className="text-[11px] text-[var(--text-muted)]">No attempt detail recorded.</p>
            ) : (
              <ul className="space-y-2">
                {data.attempts.map((attempt) => (
                  <li
                    key={attempt.candidate_id}
                    className="border-b border-[var(--border-subtle)] pb-2 last:border-0"
                  >
                    <div className="flex flex-wrap items-baseline gap-2 text-[11px]">
                      <span className="font-medium text-[var(--text-primary)]">
                        Attempt {attempt.attempt_index + 1}
                      </span>
                      <span className="text-[var(--text-secondary)]">{attempt.status}</span>
                      <span className="text-[var(--text-muted)]">{attempt.attribution}</span>
                      {attempt.seed !== null && (
                        <span className="text-[var(--text-muted)]">seed {attempt.seed}</span>
                      )}
                    </div>
                    {attempt.critical_findings.length > 0 && (
                      <p className="mt-0.5 text-[11px] text-[var(--text-secondary)]">
                        Rejected for: {attempt.critical_findings.join(", ")}
                      </p>
                    )}
                    {attempt.soft_findings.length > 0 && (
                      <p className="mt-0.5 text-[11px] text-[var(--text-muted)]">
                        Advisories (not failures): {attempt.soft_findings.join(", ")}
                      </p>
                    )}
                    {attempt.retry_reason && (
                      <p className="mt-0.5 text-[11px] text-[var(--text-muted)]">
                        {attempt.retry_reason}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <div className="mt-4" />
          <Panel title="Where the time went">
            <KeyValue
              columns={2}
              items={[
                { label: "Provider", value: seconds(data.provider_latency_seconds) },
                { label: "Quality control", value: seconds(data.qc_latency_seconds) },
                {
                  label: "Delivery",
                  value: seconds(data.delivery_latency_seconds),
                  hint: "Post-processing, finishing, encoding and upload together — Phase 22 records no timing of its own",
                },
                { label: "Total", value: seconds(data.total_latency_seconds) },
              ]}
            />
            {data.data_quality_issues.length > 0 && (
              <p className="mt-3 text-[11px] text-[var(--accent)]">
                Telemetry problems on this row: {data.data_quality_issues.join(", ")}
              </p>
            )}
          </Panel>
        </>
      )}
    </>
  );
}
