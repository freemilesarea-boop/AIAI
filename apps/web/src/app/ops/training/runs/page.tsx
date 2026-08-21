"use client";

/**
 * Every training run, filtered and paginated on the server.
 *
 * The list is deliberately cheap. Filling the latest-metric column means
 * opening one metrics file per row, so it is opt-in: at a thousand runs
 * the difference is a directory listing against a thousand reads, and a
 * list page must not be the most expensive thing the console does.
 *
 * Failures are never hidden behind a status pill. A run that failed
 * shows its code in the table, because an operator scanning for what
 * went wrong should not have to open eight runs to find the one that ran
 * out of memory.
 */

import Link from "next/link";
import { useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  DataTable,
  FilterSelect,
  OpsEmpty,
  OpsStatus,
  Pagination,
  Panel,
  PanelError,
  SectionSkeleton,
  type Column,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { Button } from "@/components/ui";
import { ops } from "@/lib/ops/client";
import { decimal, runDuration, timestamp, timestampOrNotYet } from "@/lib/ops/format";
import type { RunSummary } from "@/lib/ops/types";

const PAGE_SIZE = 25;

export default function RunsPage() {
  const [status, setStatus] = useState("");
  const [backend, setBackend] = useState("");
  const [offset, setOffset] = useState(0);
  const [withMetrics, setWithMetrics] = useState(false);

  const resource = useOpsResource(
    () =>
      ops.runs({
        status: status || undefined,
        backend: backend || undefined,
        limit: PAGE_SIZE,
        offset,
        with_metrics: withMetrics,
      }),
    { deps: [status, backend, offset, withMetrics], intervalMs: 15_000 },
  );

  const columns: Column<RunSummary>[] = [
    { key: "run", header: "Run", render: (row) => row.run_id },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    {
      key: "experiment",
      header: "Experiment",
      render: (row) => (
        <Link
          href={`/ops/training/experiments/${row.experiment_id}`}
          className="hover:text-[var(--brand-text)]"
        >
          {row.experiment_name || row.experiment_id}
        </Link>
      ),
    },
    { key: "backend", header: "Backend", render: (row) => row.execution_backend },
    {
      key: "worker",
      header: "Worker",
      render: (row) =>
        row.worker_id ? (
          <Link
            href={`/ops/training/workers/${row.worker_id}`}
            className="hover:text-[var(--brand-text)]"
          >
            {row.worker_name || row.worker_id}
          </Link>
        ) : (
          "—"
        ),
    },
    { key: "created", header: "Created", render: (row) => timestamp(row.created_at) },
    {
      key: "started",
      header: "Started",
      render: (row) => timestampOrNotYet(row.started_at, "not started"),
    },
    {
      key: "duration",
      header: "Duration",
      numeric: true,
      render: (row) => runDuration(row.duration_seconds, row.started_at),
    },
    {
      key: "metric",
      header: "Latest metric",
      numeric: true,
      render: (row) =>
        row.latest_metric ? (
          <span title={`${row.latest_metric_name} at step ${row.latest_metric.step}`}>
            {decimal(row.latest_metric.value)}
          </span>
        ) : withMetrics ? (
          "—"
        ) : (
          <span className="text-[var(--text-muted)]">not read</span>
        ),
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
          <span
            className="text-[11px] text-[var(--danger)]"
            title={row.failure.headline}
          >
            {row.failure.code}
          </span>
        ) : row.cancel_requested_at ? (
          <span className="text-[11px] text-[var(--accent)]">cancel requested</span>
        ) : (
          "—"
        ),
    },
  ];

  return (
    <>
      <OpsHeader
        title="Training runs"
        description="One concrete execution attempt each. A retry is a new run, so nothing here is ever rewritten."
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      >
        <Link href="/ops/training/runs/new">
          <Button size="sm" variant="primary">
            New run
          </Button>
        </Link>
      </OpsHeader>

      <Panel
        title="All runs"
        subtitle={resource.data ? `${resource.data.page.total.toLocaleString()} recorded` : undefined}
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <FilterSelect
              label="Status"
              value={status}
              options={resource.data?.available_statuses ?? []}
              onChange={(value) => {
                setStatus(value);
                setOffset(0);
              }}
            />
            <FilterSelect
              label="Backend"
              value={backend}
              options={resource.data?.available_backends ?? []}
              onChange={(value) => {
                setBackend(value);
                setOffset(0);
              }}
            />
            <label className="flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
              <input
                type="checkbox"
                checked={withMetrics}
                onChange={(event) => setWithMetrics(event.target.checked)}
              />
              Read latest metric
            </label>
          </div>
        }
      >
        {resource.error && !resource.data && (
          <PanelError message={resource.error} onRetry={resource.refresh} />
        )}
        {resource.loading && <SectionSkeleton rows={6} />}
        {resource.data && (
          <>
            {resource.error && (
              <div className="mb-3">
                <PanelError message={`${resource.error} Showing the last successful read.`} />
              </div>
            )}
            <DataTable
              rows={resource.data.items}
              columns={columns}
              rowKey={(row) => row.run_id}
              href={(row) => `/ops/training/runs/${row.run_id}`}
              caption="Training runs"
              empty={
                <OpsEmpty
                  title="No runs"
                  description={
                    status || backend
                      ? "No run matches these filters."
                      : "Nothing has been executed yet. Create a run against an experiment, validate it, then dispatch."
                  }
                  action={
                    <Link
                      href="/ops/training/runs/new"
                      className="text-xs text-[var(--brand-text)] underline underline-offset-2"
                    >
                      Create a run
                    </Link>
                  }
                />
              }
            />
            <Pagination page={resource.data.page} onOffset={setOffset} />
          </>
        )}
      </Panel>
    </>
  );
}
