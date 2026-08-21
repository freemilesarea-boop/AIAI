"use client";

/**
 * Every checkpoint, with the placeholders unmistakable.
 *
 * `MOCK` is what a dry run registers so a test has something to point
 * at. It contains no trained weights and can never become an evaluation
 * candidate, and it is marked TEST ONLY wherever it appears — the one
 * thing on this screen that could otherwise put a placeholder in front
 * of a listener.
 */

import { useState } from "react";

import { CheckpointTable } from "@/components/ops/CheckpointTable";
import { OpsHeader } from "@/components/ops/OpsShell";
import {
  FilterSelect,
  Pagination,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";

const PAGE_SIZE = 25;

export default function CheckpointsPage() {
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const [offset, setOffset] = useState(0);

  const resource = useOpsResource(
    () =>
      ops.checkpoints({
        status: status || undefined,
        kind: kind || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    { deps: [status, kind, offset] },
  );

  return (
    <>
      <OpsHeader
        title="Checkpoints"
        description="A READY checkpoint is trained weights on disk. It is not an accepted model, and nothing here promotes one."
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      <Panel
        title="All checkpoints"
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
              label="Kind"
              value={kind}
              options={resource.data?.available_kinds ?? []}
              onChange={(value) => {
                setKind(value);
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
            <CheckpointTable rows={resource.data.items} />
            <Pagination page={resource.data.page} onOffset={setOffset} />
          </>
        )}
      </Panel>
    </>
  );
}
