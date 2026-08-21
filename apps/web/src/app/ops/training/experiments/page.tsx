"use client";

/**
 * Experiments: the hypotheses, not the executions.
 *
 * The distinction is the reason this page exists separately from Runs. A
 * failed run does not disprove a hypothesis, and a list that mixed them
 * would make a crashed process look like a finding.
 *
 * Search covers the name and the id. It deliberately does not reach into
 * the dataset: the console is not a way to browse training material, and
 * a search box that matched track titles would quietly become one.
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
import { timestamp } from "@/lib/ops/format";
import type { ExperimentSummary } from "@/lib/ops/types";

const PAGE_SIZE = 25;

export default function ExperimentsPage() {
  const [status, setStatus] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [tag, setTag] = useState("");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);

  const resource = useOpsResource(
    () =>
      ops.experiments({
        status: status || undefined,
        base_model_id: baseModel || undefined,
        tag: tag || undefined,
        q: query || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    { deps: [status, baseModel, tag, query, offset] },
  );

  const columns: Column<ExperimentSummary>[] = [
    { key: "name", header: "Name", render: (row) => row.name || row.experiment_id },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    {
      key: "hypothesis",
      header: "Hypothesis",
      render: (row) => (
        <span className="line-clamp-2 max-w-md text-xs">{row.hypothesis}</span>
      ),
    },
    {
      key: "base_model",
      header: "Base model",
      render: (row) => <span className="font-mono text-[11px]">{row.base_model_id}</span>,
    },
    {
      key: "dataset",
      header: "Dataset lock",
      render: (row) => (
        <span className="font-mono text-[11px]">{row.dataset_lock_ref ?? "—"}</span>
      ),
    },
    { key: "runs", header: "Runs", numeric: true, render: (row) => row.run_count },
    {
      key: "latest",
      header: "Latest run",
      render: (row) =>
        row.latest_run_status ? <OpsStatus status={row.latest_run_status} /> : "—",
    },
    {
      key: "tags",
      header: "Tags",
      render: (row) => (
        <span className="text-[11px] text-[var(--text-muted)]">{row.tags.join(", ") || "—"}</span>
      ),
    },
    { key: "created", header: "Created", render: (row) => timestamp(row.created_at) },
  ];

  return (
    <>
      <OpsHeader
        title="Experiments"
        description="A hypothesis outlives the runs that test it. Creating one starts nothing."
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      >
        <Link href="/ops/training/experiments/new">
          <Button size="sm" variant="primary">
            New experiment
          </Button>
        </Link>
      </OpsHeader>

      <Panel
        title="All experiments"
        subtitle={
          resource.data ? `${resource.data.page.total.toLocaleString()} recorded` : undefined
        }
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <span className="sr-only">Search experiments by name or id</span>
              <input
                type="search"
                value={query}
                placeholder="name or id"
                onChange={(event) => {
                  setQuery(event.target.value);
                  setOffset(0);
                }}
                className="w-40 rounded-[var(--radius-sm)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-2 py-1 text-xs text-[var(--text-primary)]"
              />
            </label>
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
              label="Base model"
              value={baseModel}
              options={resource.data?.available_base_models ?? []}
              onChange={(value) => {
                setBaseModel(value);
                setOffset(0);
              }}
            />
            <FilterSelect
              label="Tag"
              value={tag}
              options={resource.data?.available_tags ?? []}
              onChange={(value) => {
                setTag(value);
                setOffset(0);
              }}
            />
          </div>
        }
      >
        {resource.error && !resource.data && (
          <PanelError message={resource.error} onRetry={resource.refresh} />
        )}
        {resource.loading && <SectionSkeleton rows={5} />}
        {resource.data && (
          <>
            <DataTable
              rows={resource.data.items}
              columns={columns}
              rowKey={(row) => row.experiment_id}
              href={(row) => `/ops/training/experiments/${row.experiment_id}`}
              caption="Experiments"
              empty={
                <OpsEmpty
                  title="No experiments yet"
                  description={
                    status || query || tag || baseModel
                      ? "No experiment matches these filters. Clear them to see everything recorded."
                      : "An experiment records a hypothesis and the runs that test it. Register a model baseline first, then create one."
                  }
                  action={
                    <Link
                      href="/ops/training/experiments/new"
                      className="text-xs text-[var(--brand-text)] underline underline-offset-2"
                    >
                      New experiment
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
