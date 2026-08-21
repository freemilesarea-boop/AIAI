"use client";

/**
 * One experiment, with everything that descends from it.
 *
 * The lineage is the point: hypothesis → runs → checkpoints →
 * evaluations → qualification. An operator asking "did this idea work"
 * should be able to answer it here without opening five pages, and
 * should be able to see that a failed run is an execution problem rather
 * than a verdict on the idea.
 */

import Link from "next/link";
import { use } from "react";

import { AuditList } from "@/components/ops/AuditList";
import { OpsHeader } from "@/components/ops/OpsShell";
import {
  DataTable,
  KeyValue,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
  type Column,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import { runDuration, shortDigest, timestamp } from "@/lib/ops/format";
import type { EvaluationSummary, RunSummary } from "@/lib/ops/types";

export default function ExperimentDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const resource = useOpsResource(() => ops.experiment(id), { deps: [id], intervalMs: 20_000 });

  const runColumns: Column<RunSummary>[] = [
    { key: "run", header: "Run", render: (row) => row.run_id },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    { key: "backend", header: "Backend", render: (row) => row.execution_backend },
    { key: "worker", header: "Worker", render: (row) => row.worker_name ?? "—" },
    {
      key: "duration",
      header: "Duration",
      numeric: true,
      render: (row) => runDuration(row.duration_seconds, row.started_at),
    },
    {
      key: "checkpoints",
      header: "Checkpoints",
      numeric: true,
      render: (row) => row.checkpoint_count,
    },
    {
      key: "failure",
      header: "Failure",
      render: (row) =>
        row.failure ? (
          <span className="text-[11px] text-[var(--danger)]">{row.failure.code}</span>
        ) : (
          "—"
        ),
    },
    { key: "created", header: "Created", render: (row) => timestamp(row.created_at) },
  ];

  const evaluationColumns: Column<EvaluationSummary>[] = [
    { key: "id", header: "Evaluation", render: (row) => row.evaluation_id },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    {
      key: "outcome",
      header: "Qualification",
      render: (row) =>
        row.qualification_outcome ? <OpsStatus status={row.qualification_outcome} /> : "—",
    },
    { key: "suite", header: "Suite", render: (row) => `${row.suite_id} v${row.suite_version}` },
    { key: "candidate", header: "Checkpoint", render: (row) => row.checkpoint_id || "—" },
    { key: "completed", header: "Completed", render: (row) => timestamp(row.completed_at) },
  ];

  return (
    <>
      <OpsHeader
        title={resource.data?.experiment.name ?? id}
        breadcrumb={[{ href: "/ops/training/experiments", label: "Experiments" }]}
        description={resource.data?.experiment.hypothesis}
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      {resource.error && !resource.data && (
        <PanelError message={resource.error} onRetry={resource.refresh} />
      )}
      {resource.loading && <SectionSkeleton rows={6} />}

      {resource.data && (
        <div className="space-y-5">
          {resource.data.experiment.status === "BLOCKED" && (
            <Panel title="Blocked" tone="danger" id="blocked">
              <p className="text-sm text-[var(--danger)]">
                {resource.data.experiment.blocked_reason ||
                  "This experiment is blocked and no reason was recorded."}
              </p>
            </Panel>
          )}

          <Panel title="Identity" id="identity">
            <KeyValue
              columns={3}
              items={[
                { label: "Experiment id", value: resource.data.experiment.experiment_id },
                {
                  label: "Status",
                  value: <OpsStatus status={resource.data.experiment.status} />,
                },
                { label: "Operator", value: resource.data.experiment.operator || "—" },
                { label: "Base model", value: resource.data.experiment.base_model_id },
                {
                  label: "Upstream commit",
                  value: shortDigest(resource.data.base_model?.upstream_commit ?? null),
                },
                { label: "Created", value: timestamp(resource.data.experiment.created_at) },
                {
                  label: "Dataset lock reference",
                  value: resource.data.experiment.dataset_lock_ref ?? "—",
                },
                {
                  label: "Curation lock reference",
                  value: resource.data.experiment.curation_lock_ref ?? "—",
                },
                {
                  label: "Tags",
                  value: resource.data.experiment.tags.join(", ") || "—",
                },
              ]}
            />
            {resource.data.experiment.description && (
              <p className="mt-4 text-sm leading-relaxed text-[var(--text-secondary)]">
                {resource.data.experiment.description}
              </p>
            )}
          </Panel>

          <Panel
            title="Training runs"
            subtitle="A retry is a new run citing its parent. Nothing here is edited."
            id="runs"
          >
            <DataTable
              rows={resource.data.runs}
              columns={runColumns}
              rowKey={(row) => row.run_id}
              href={(row) => `/ops/training/runs/${row.run_id}`}
              caption="Runs for this experiment"
              empty={
                <OpsEmpty
                  title="No runs yet"
                  description="This experiment records a hypothesis and nothing has tested it. Create a run to do that."
                  action={
                    <Link
                      href={`/ops/training/runs/new?experiment=${id}`}
                      className="text-xs text-[var(--brand-text)] underline underline-offset-2"
                    >
                      Create a run
                    </Link>
                  }
                />
              }
            />
          </Panel>

          <Panel
            title="Evaluation candidates"
            subtitle="A candidate is a request for evidence, not a claim of quality."
            id="candidates"
          >
            {resource.data.candidates.length === 0 ? (
              <OpsEmpty
                title="No candidates"
                description="No checkpoint from this experiment has been nominated for evaluation."
              />
            ) : (
              <ul className="space-y-2">
                {resource.data.candidates.map((candidate) => (
                  <li key={candidate.candidate_id} className="flex flex-wrap items-center gap-3">
                    <OpsStatus status={candidate.status} />
                    <Link
                      href={`/ops/training/checkpoints/${candidate.checkpoint_id}`}
                      className="font-mono text-xs text-[var(--brand-text)] hover:underline"
                    >
                      {candidate.checkpoint_id}
                    </Link>
                    <span className="text-[11px] text-[var(--text-muted)]">
                      {timestamp(candidate.created_at)}
                    </span>
                    {candidate.notes && (
                      <span className="text-[11px] text-[var(--text-muted)]">
                        {candidate.notes}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel
            title="Evaluations and qualification"
            subtitle="QUALIFIED means a checkpoint may advance to promotion review. Nothing here activates a model."
            id="evaluations"
          >
            <DataTable
              rows={resource.data.evaluations}
              columns={evaluationColumns}
              rowKey={(row) => row.evaluation_id}
              href={(row) => `/ops/training/evaluations/${row.evaluation_id}`}
              caption="Evaluations for this experiment"
              empty={
                <OpsEmpty
                  title="No evaluations yet"
                  description="Nothing from this experiment has been judged against a baseline."
                />
              }
            />
          </Panel>

          <Panel title="History" subtitle="Append-only, in the order it happened." id="audit">
            <AuditList events={resource.data.audit_events} />
          </Panel>
        </div>
      )}
    </>
  );
}
