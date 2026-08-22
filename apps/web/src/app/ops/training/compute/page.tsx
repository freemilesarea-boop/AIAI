/**
 * Compute targets: where work can run, and where it cannot.
 *
 * Deliberately small. This is not a hardware dashboard — no utilisation
 * graphs, no temperature, no fan speed. It answers one question an
 * operator arrives with, which is "can I start this run and where will
 * it go", and every row is derived from a probe rather than from
 * configuration.
 *
 * Three honesty rules shape what is rendered.
 *
 * **NOT_CONNECTED is shown, not hidden.** A missing remote row would
 * read as "we didn't check". A row saying there is no GPU worker reads
 * as "there isn't one yet", which is true and more useful.
 *
 * **An empty precision list is blank, not "none".** Nobody measured is
 * a different statement from nothing works.
 *
 * **The workload list comes from the scheduler's own policy.** A row
 * cannot promise something the scheduler will decline.
 */

"use client";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  DataTable,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import type { ComputeTarget } from "@/lib/ops/types";

/** Hardware changes when somebody rents a machine, not second to second. */
const REFRESH_MS = 60_000;

function memory(target: ComputeTarget): string {
  if (target.memory_mb === null) return "";
  return `${(target.memory_mb / 1024).toFixed(1)} GiB`;
}

const COLUMNS = [
  {
    key: "target",
    header: "Target",
    render: (row: ComputeTarget) => (
      <span className="inline-flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-[var(--text-primary)]">{row.name}</span>
        {row.planned && <OpsStatus status="PENDING" label="planned" />}
      </span>
    ),
  },
  { key: "location", header: "Location", render: (row: ComputeTarget) => row.location },
  { key: "device", header: "Device", render: (row: ComputeTarget) => row.device },
  {
    key: "status",
    header: "Status",
    render: (row: ComputeTarget) => (
      <OpsStatus status={row.status} title={row.detail || undefined} />
    ),
  },
  {
    key: "memory",
    header: "Memory",
    render: (row: ComputeTarget) => <Maybe value={memory(row)} />,
  },
  {
    key: "precision",
    header: "Precision",
    // Blank rather than "none": an empty list means nobody measured,
    // and a dash reading as "no precisions work" would be a different
    // and false claim.
    render: (row: ComputeTarget) => <Maybe value={row.precisions.join(", ")} />,
  },
  {
    key: "workloads",
    header: "Can take",
    render: (row: ComputeTarget) => <Maybe value={row.workloads.join(", ")} />,
  },
  {
    key: "detail",
    header: "Notes",
    render: (row: ComputeTarget) => (
      <span className="text-[var(--text-muted)]">
        <Maybe value={row.detail} />
      </span>
    ),
  },
];

export default function ComputeTargetsPage() {
  const targets = useOpsResource(() => ops.computeTargets(), { intervalMs: REFRESH_MS });

  return (
    <>
      <OpsHeader
        title="Compute targets"
        description="Where a workload can run, derived from probes rather than configuration."
        onRefresh={targets.refresh}
        refreshing={targets.refreshing}
      />

      {targets.error && <PanelError message={targets.error} onRetry={targets.refresh} />}

      <Panel
        title="Targets"
        subtitle="Location and device are separate: a local target need not be a CPU, and a remote one need not be a GPU."
      >
        {targets.loading ? (
          <SectionSkeleton />
        ) : (
          <>
            <p className="mb-3 text-sm text-[var(--text-secondary)]">
              {targets.data?.summary}
            </p>
            <DataTable
              rows={targets.data?.targets ?? []}
              columns={COLUMNS}
              rowKey={(row) => `${row.name}:${row.location}:${row.device}`}
              caption="Compute targets"
              empty={
                <OpsEmpty
                  title="No compute targets"
                  description="Not even this machine answered, which should be impossible."
                />
              }
            />
          </>
        )}
      </Panel>

      <Panel
        title="Local training policy"
        subtitle="The machine that runs the control plane stays a control plane."
      >
        <ul className="space-y-1.5 text-[11px] text-[var(--text-muted)]">
          <li>
            Concurrent local training jobs:{" "}
            <span className="text-[var(--text-secondary)]">
              {targets.data?.local_training_concurrency ?? 1}
            </span>
            . Two runs on shared unified memory is how the API stops answering.
          </li>
          <li>
            Memory is never planned to 100%. A fraction and a floor are reserved for the
            operating system and the services this machine is also running.
          </li>
          <li>
            Heavy training prefers a remote CUDA worker and is <em>refused</em> when none is
            connected, rather than being moved here quietly. A run that trained somewhere
            nobody chose produces a checkpoint indistinguishable from one that did not.
          </li>
          <li>
            Nothing has measured what LUBER training needs in memory, so feasibility reads
            UNKNOWN rather than a number.
          </li>
        </ul>
      </Panel>
    </>
  );
}
