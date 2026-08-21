"use client";

/**
 * One worker's capability report, exactly as it was measured.
 *
 * The unknowns are listed by name rather than left for an operator to
 * find field by field. A machine with eleven UNKNOWN values is not a
 * machine with a few gaps; it is a machine nobody has probed, and the
 * page should say so in one place.
 *
 * No credential appears here. That a key reference is configured is a
 * boolean; what the deployment calls it is not something a browser
 * needs, and the API does not send it.
 */

import Link from "next/link";
import { use } from "react";

import { AuditList } from "@/components/ops/AuditList";
import { OpsHeader } from "@/components/ops/OpsShell";
import {
  CopyValue,
  DataTable,
  KeyValue,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
  Unavailable,
  type Column,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import { age, bool, duration, megabytes, runDuration, timestamp } from "@/lib/ops/format";
import type { RunSummary } from "@/lib/ops/types";

export default function WorkerDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const resource = useOpsResource(() => ops.worker(id), { deps: [id], intervalMs: 15_000 });
  const detail = resource.data;

  const runColumns: Column<RunSummary>[] = [
    { key: "run", header: "Run", render: (row) => row.run_id },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    { key: "backend", header: "Backend", render: (row) => row.execution_backend },
    {
      key: "duration",
      header: "Duration",
      numeric: true,
      render: (row) => runDuration(row.duration_seconds, row.started_at),
    },
    { key: "created", header: "Created", render: (row) => timestamp(row.created_at) },
  ];

  return (
    <>
      <OpsHeader
        title={detail?.worker.name ?? id}
        breadcrumb={[{ href: "/ops/training/workers", label: "Workers" }]}
        description={
          detail && (
            <span className="flex flex-wrap items-center gap-2">
              <OpsStatus status={detail.worker.worker_class} />
              <OpsStatus status={detail.worker.liveness} />
              <span>{detail.worker.backend_type}</span>
            </span>
          )
        }
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      {resource.error && !detail && <PanelError message={resource.error} onRetry={resource.refresh} />}
      {resource.loading && <SectionSkeleton rows={6} />}

      {detail && (
        <div className="space-y-5">
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Identity" id="identity">
              <KeyValue
                columns={1}
                items={[
                  { label: "Worker id", value: detail.worker.worker_id },
                  {
                    label: "Host fingerprint",
                    value: (
                      <CopyValue
                        value={detail.worker.host_identity}
                        display={detail.worker.host_identity.slice(0, 20)}
                        label="host fingerprint"
                      />
                    ),
                    hint: "A digest over stable machine facts. A hostname is not an identity.",
                  },
                  {
                    label: "Capability signature",
                    value: (
                      <CopyValue
                        value={detail.worker.capability_signature}
                        display={detail.worker.capability_signature?.slice(0, 20)}
                        label="capability signature"
                      />
                    ),
                  },
                  { label: "Protocol", value: <Maybe value={detail.worker.protocol_version} /> },
                  {
                    label: "Remote classification",
                    value: detail.worker.remote_classification ? (
                      <OpsStatus status={detail.worker.remote_classification} />
                    ) : (
                      <Maybe value={null} />
                    ),
                  },
                  {
                    label: "Credential configured",
                    value: detail.worker.has_credentials ? "Yes" : "No",
                    hint: "The console never receives the reference name or the key itself.",
                  },
                  {
                    label: "Concurrency",
                    value: `${detail.worker.active_run_ids.length} of ${detail.worker.max_concurrent_runs} in use`,
                  },
                  { label: "Registered", value: timestamp(detail.worker.created_at) },
                ]}
              />
            </Panel>

            <Panel
              title="Heartbeat"
              subtitle="The friendly age, and the exact instant beside it."
              id="heartbeat"
            >
              {detail.heartbeat.available ? (
                <>
                  <KeyValue
                    columns={2}
                    items={[
                      {
                        label: "Liveness",
                        value: <OpsStatus status={detail.heartbeat.liveness} />,
                      },
                      {
                        label: "Last heard",
                        value: age(detail.heartbeat.age_seconds),
                        hint: timestamp(detail.heartbeat.timestamp),
                      },
                      { label: "Worker state", value: <Maybe value={detail.heartbeat.worker_state} /> },
                      { label: "Health", value: <Maybe value={detail.heartbeat.health} /> },
                      { label: "Active run", value: <Maybe value={detail.heartbeat.active_run_id} /> },
                      { label: "Uptime", value: duration(detail.heartbeat.uptime_seconds) },
                      { label: "Free disk", value: megabytes(detail.heartbeat.free_disk_mb) },
                    ]}
                  />
                  {detail.heartbeat.detail && (
                    <p className="mt-3 text-xs text-[var(--accent)]">{detail.heartbeat.detail}</p>
                  )}
                  {detail.heartbeat.gpu.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {detail.heartbeat.gpu.map((gpu) => (
                        <p key={gpu.index} className="text-[11px] text-[var(--text-muted)]">
                          GPU {gpu.index}: utilisation <Maybe value={gpu.utilization_pct} />% ·
                          memory {megabytes(gpu.memory_used_mb)} / {megabytes(gpu.memory_total_mb)}{" "}
                          · <Maybe value={gpu.temperature_c} />°C · <Maybe value={gpu.power_w} />W
                        </p>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <Unavailable reason={detail.heartbeat.unavailable_reason} />
              )}
            </Panel>
          </div>

          <Panel
            title="Capability report"
            subtitle="Every field is a measurement or UNKNOWN. Nothing is defaulted."
            id="capabilities"
          >
            <KeyValue
              columns={3}
              items={[
                { label: "GPU vendor", value: <Maybe value={detail.worker.capabilities.gpu_vendor} /> },
                { label: "GPU model", value: <Maybe value={detail.worker.capabilities.gpu_model} /> },
                { label: "GPU count", value: <Maybe value={detail.worker.capabilities.gpu_count} /> },
                { label: "VRAM", value: megabytes(detail.worker.capabilities.vram_total_mb) },
                {
                  label: "CUDA available",
                  value: bool(detail.worker.capabilities.cuda_available),
                  hint: "torch is the authority here, not the presence of a driver.",
                },
                { label: "CUDA version", value: <Maybe value={detail.worker.capabilities.cuda_version} /> },
                { label: "Driver", value: <Maybe value={detail.worker.capabilities.driver_version} /> },
                { label: "bf16", value: bool(detail.worker.capabilities.bf16_supported) },
                { label: "torch", value: <Maybe value={detail.worker.capabilities.torch_version} /> },
                { label: "Python", value: <Maybe value={detail.worker.capabilities.python_version} /> },
                { label: "CPU count", value: <Maybe value={detail.worker.capabilities.cpu_count} /> },
                { label: "System RAM", value: megabytes(detail.worker.capabilities.system_ram_mb) },
                { label: "Free disk", value: megabytes(detail.worker.capabilities.free_disk_mb) },
                { label: "Reported by", value: detail.worker.capabilities.reported_by },
                { label: "Reported at", value: timestamp(detail.worker.capabilities.reported_at) },
              ]}
            />

            {detail.unknown_capabilities.length > 0 && (
              <div className="mt-4 rounded-[var(--radius-md)] border border-dashed border-[var(--border-strong)] px-3 py-2">
                <p className="text-[11px] font-medium text-[var(--text-secondary)]">
                  {detail.unknown_capabilities.length} capability value(s) have never been measured
                </p>
                <p className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">
                  {detail.unknown_capabilities.join(", ")}
                </p>
                <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                  Run the probe on the machine to establish them. Nothing here fills them in.
                </p>
              </div>
            )}
          </Panel>

          {Object.keys(detail.software_environment).length > 0 && (
            <Panel title="Software environment" id="software">
              <KeyValue
                columns={3}
                items={Object.entries(detail.software_environment).map(([key, value]) => ({
                  label: key,
                  value,
                }))}
              />
            </Panel>
          )}

          <Panel title="Recent runs" id="runs">
            <DataTable
              rows={detail.recent_runs}
              columns={runColumns}
              rowKey={(row) => row.run_id}
              href={(row) => `/ops/training/runs/${row.run_id}`}
              caption="Runs assigned to this worker"
              empty={
                <OpsEmpty
                  title="No runs"
                  description="Nothing has been assigned to this worker yet."
                />
              }
            />
            {detail.worker.active_run_ids.length > 0 && (
              <p className="mt-3 text-xs text-[var(--text-secondary)]">
                Currently running:{" "}
                {detail.worker.active_run_ids.map((runId) => (
                  <Link
                    key={runId}
                    href={`/ops/training/runs/${runId}`}
                    className="mr-2 font-mono text-[var(--brand-text)] hover:underline"
                  >
                    {runId}
                  </Link>
                ))}
              </p>
            )}
          </Panel>

          <Panel title="History" id="audit">
            <AuditList events={detail.audit_events} />
          </Panel>
        </div>
      )}
    </>
  );
}
