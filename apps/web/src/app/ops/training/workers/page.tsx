"use client";

/**
 * The fleet, with liveness and registry status kept apart.
 *
 * A worker record saying ONLINE proves only that something wrote ONLINE.
 * Liveness is derived from the last heartbeat against the Phase 27
 * policy, and the two are different columns because a worker that stopped
 * reporting an hour ago while its record still says ONLINE is exactly the
 * case an operator needs to notice.
 *
 * ONLINE is never treated as GPU-ready either. A machine is
 * GPU_TRAINING_READY because a probe demonstrated CUDA through torch on
 * it, and nothing else grants that.
 */

import { useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  DataTable,
  FilterSelect,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Pagination,
  Panel,
  PanelError,
  SectionSkeleton,
  type Column,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import { age, megabytes, timestamp } from "@/lib/ops/format";
import type { WorkerSummary } from "@/lib/ops/types";

const PAGE_SIZE = 25;

export default function WorkersPage() {
  const [workerClass, setWorkerClass] = useState("");
  const [liveness, setLiveness] = useState("");
  const [offset, setOffset] = useState(0);

  const resource = useOpsResource(
    () =>
      ops.workers({
        worker_class: workerClass || undefined,
        liveness: liveness || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    { deps: [workerClass, liveness, offset], intervalMs: 20_000 },
  );

  const columns: Column<WorkerSummary>[] = [
    { key: "name", header: "Worker", render: (row) => row.name || row.worker_id },
    { key: "class", header: "Class", render: (row) => <OpsStatus status={row.worker_class} /> },
    { key: "liveness", header: "Liveness", render: (row) => <OpsStatus status={row.liveness} /> },
    {
      key: "status",
      header: "Registry status",
      render: (row) => <span className="text-[11px]">{row.status}</span>,
    },
    { key: "backend", header: "Backend", render: (row) => row.backend_type },
    {
      key: "heartbeat",
      header: "Last heartbeat",
      render: (row) => (
        <span title={timestamp(row.last_heartbeat)}>{age(row.heartbeat_age_seconds)}</span>
      ),
    },
    {
      key: "gpu",
      header: "GPU",
      render: (row) => <Maybe value={row.capabilities.gpu_model} />,
    },
    {
      key: "gpu_count",
      header: "Count",
      numeric: true,
      render: (row) => <Maybe value={row.capabilities.gpu_count} />,
    },
    {
      key: "vram",
      header: "VRAM",
      numeric: true,
      render: (row) => megabytes(row.capabilities.vram_total_mb),
    },
    {
      key: "cuda",
      header: "CUDA",
      render: (row) => <Maybe value={row.capabilities.cuda_version} />,
    },
    {
      key: "ram",
      header: "RAM",
      numeric: true,
      render: (row) => megabytes(row.capabilities.system_ram_mb),
    },
    {
      key: "disk",
      header: "Free disk",
      numeric: true,
      render: (row) => megabytes(row.capabilities.free_disk_mb),
    },
    {
      key: "active",
      header: "Active run",
      render: (row) =>
        row.active_run_ids.length > 0 ? (
          <span className="font-mono text-[11px]">{row.active_run_ids[0]}</span>
        ) : (
          "—"
        ),
    },
  ];

  return (
    <>
      <OpsHeader
        title="Workers"
        description="Hardware facts are reported by a probe on the machine itself. UNKNOWN means nobody has measured it — never zero."
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      <Panel
        title="Registered workers"
        subtitle={resource.data ? `${resource.data.page.total.toLocaleString()} registered` : undefined}
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <FilterSelect
              label="Class"
              value={workerClass}
              options={resource.data?.available_classes ?? []}
              onChange={(value) => {
                setWorkerClass(value);
                setOffset(0);
              }}
            />
            <FilterSelect
              label="Liveness"
              value={liveness}
              options={resource.data?.available_liveness ?? []}
              onChange={(value) => {
                setLiveness(value);
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
              rowKey={(row) => row.worker_id}
              href={(row) => `/ops/training/workers/${row.worker_id}`}
              caption="Registered training workers"
              empty={
                <OpsEmpty
                  title="No GPU workers registered"
                  description="Register one from the machine itself so its capabilities come from a probe rather than an assertion: `python -m luber_training.remote init` on the worker, then `luber-training remote worker register` from here."
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
