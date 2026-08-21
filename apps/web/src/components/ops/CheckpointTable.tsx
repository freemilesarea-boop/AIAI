"use client";

/**
 * Checkpoints, with the placeholder ones impossible to mistake.
 *
 * `MOCK` is a distinct kind rather than a flag precisely so no query for
 * a real checkpoint returns one by accident, and the same reasoning
 * applies to the table: it carries a TEST ONLY badge, not a subtle
 * shade. A dry run's artifact rendered like an adapter is the one thing
 * on this screen that could put a placeholder in front of a listener.
 *
 * Whether a checkpoint can be evaluated is decided by the server and
 * shown here with its reason. A UI that worked it out itself would
 * eventually disagree with the API that enforces it.
 */

import { CopyValue, DataTable, Maybe, OpsEmpty, OpsStatus, type Column } from "@/components/ops/primitives";
import { cx } from "@/components/ui";
import { bytes, timestamp } from "@/lib/ops/format";
import type { CheckpointSummary } from "@/lib/ops/types";

export function CheckpointTable({ rows }: { rows: CheckpointSummary[] }) {
  const columns: Column<CheckpointSummary>[] = [
    { key: "id", header: "Checkpoint", render: (row) => row.checkpoint_id },
    {
      key: "kind",
      header: "Kind",
      render: (row) => (
        <span className="inline-flex items-center gap-1.5">
          <OpsStatus status={row.kind} />
          {!row.is_real_model && (
            <span
              className={cx(
                "rounded-[var(--radius-sm)] bg-[var(--accent-muted)] px-1.5 py-0.5",
                "text-[10px] font-semibold text-[var(--accent)]",
              )}
            >
              TEST ONLY
            </span>
          )}
        </span>
      ),
    },
    { key: "status", header: "Status", render: (row) => <OpsStatus status={row.status} /> },
    { key: "step", header: "Step", numeric: true, render: (row) => <Maybe value={row.step} /> },
    { key: "epoch", header: "Epoch", numeric: true, render: (row) => <Maybe value={row.epoch} /> },
    { key: "size", header: "Size", numeric: true, render: (row) => bytes(row.size_bytes) },
    {
      key: "sha",
      header: "Digest",
      render: (row) => (
        <CopyValue value={row.sha256} display={row.sha256?.slice(0, 12)} label="checkpoint digest" />
      ),
    },
    {
      key: "evaluate",
      header: "Evaluable",
      render: (row) =>
        row.can_evaluate ? (
          <OpsStatus status="PASS" label="Yes" />
        ) : (
          <span title={row.evaluate_blocked_reason ?? undefined}>
            <OpsStatus status="NOT_EVALUATED" label="No" />
          </span>
        ),
    },
    { key: "created", header: "Created", render: (row) => timestamp(row.created_at) },
  ];

  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => row.checkpoint_id}
      href={(row) => `/ops/training/checkpoints/${row.checkpoint_id}`}
      caption="Checkpoints"
      empty={
        <OpsEmpty
          title="No checkpoints"
          description="A dry run produces none by design, and a run that has not started has written nothing."
        />
      }
    />
  );
}
