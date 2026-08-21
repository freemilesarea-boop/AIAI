"use client";

/**
 * Individual generations, for when a rate is not enough.
 *
 * A rate tells an operator that something is wrong; opening three of the
 * failures tells them what. What this list deliberately does not carry
 * is what the user asked for: no prompt, no lyrics, no title. Those are
 * absent from the projection this reads and absent from the response
 * models, so there is nothing to redact here — the redaction happened by
 * never collecting it.
 */

import Link from "next/link";
import { useState } from "react";

import { WindowPicker } from "@/components/ops/InferenceControls";
import { OpsHeader } from "@/components/ops/OpsShell";
import { OpsEmpty, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference } from "@/lib/ops/inference-client";
import type { WindowSize } from "@/lib/ops/inference-types";

const PAGE = 25;

export default function GenerationsPage() {
  const [window, setWindow] = useState<WindowSize>("24h");
  const [onlyFailures, setOnlyFailures] = useState(false);
  const [offset, setOffset] = useState(0);

  const generations = useOpsResource(
    () =>
      inference.generations({
        window,
        limit: PAGE,
        offset,
        only_failures: onlyFailures,
      }),
    { deps: [window, onlyFailures, offset] },
  );

  return (
    <>
      <OpsHeader
        title="Generations"
        description="Individual candidate phases. No prompt, lyrics or title is stored or shown."
        onRefresh={generations.refresh}
        refreshing={generations.refreshing}
      >
        <label className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={onlyFailures}
            onChange={(event) => {
              setOnlyFailures(event.target.checked);
              setOffset(0);
            }}
          />
          Failures only
        </label>
        <WindowPicker
          value={window}
          onChange={(next) => {
            setWindow(next);
            setOffset(0);
          }}
        />
      </OpsHeader>

      {generations.error && (
        <PanelError message={generations.error} onRetry={generations.refresh} />
      )}

      <Panel
        title="Observed generations"
        subtitle={generations.data ? `${generations.data.total} in this window.` : undefined}
      >
        {generations.loading ? (
          <SectionSkeleton />
        ) : !generations.data?.items.length ? (
          <OpsEmpty
            title="Nothing to show"
            description="No generations in this window, or none have been ingested."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="border-b border-[var(--border-subtle)] text-left text-[var(--text-muted)]">
                  <th className="py-1.5 pr-3 font-medium">When</th>
                  <th className="py-1.5 pr-3 font-medium">Status</th>
                  <th className="py-1.5 pr-3 font-medium">Revision</th>
                  <th className="py-1.5 pr-3 font-medium">Task</th>
                  <th className="py-1.5 pr-3 font-medium">Duration</th>
                  <th className="py-1.5 pr-3 font-medium">Retries</th>
                  <th className="py-1.5 pr-3 font-medium">Findings</th>
                  <th className="py-1.5 font-medium">Total</th>
                </tr>
              </thead>
              <tbody>
                {generations.data.items.map((item) => (
                  <tr
                    key={item.generation_id}
                    className="border-b border-[var(--border-subtle)] last:border-0"
                  >
                    <td className="py-1.5 pr-3">
                      <Link
                        href={`/ops/inference/generations/${item.generation_id}`}
                        className="text-[var(--text-secondary)] hover:text-[var(--brand-text)]"
                      >
                        {new Date(item.occurred_at).toLocaleString()}
                      </Link>
                    </td>
                    <td className="py-1.5 pr-3 text-[var(--text-secondary)]">
                      {item.generation_status}
                    </td>
                    <td className="py-1.5 pr-3 text-[var(--text-muted)]">
                      {item.provider_revision}
                    </td>
                    <td className="py-1.5 pr-3 text-[var(--text-muted)]">{item.task_type}</td>
                    <td className="py-1.5 pr-3 text-[var(--text-muted)]">
                      {item.duration_bucket}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                      {item.quality_retry_count ?? "—"}
                    </td>
                    <td className="py-1.5 pr-3 text-[var(--text-muted)]">
                      {item.critical_findings.join(", ") || "—"}
                    </td>
                    <td className="py-1.5 tabular-nums text-[var(--text-muted)]">
                      {item.total_latency_seconds !== null
                        ? `${item.total_latency_seconds.toFixed(1)}s`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {generations.data && generations.data.total > PAGE && (
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
              {offset + 1}–{Math.min(offset + PAGE, generations.data.total)} of{" "}
              {generations.data.total}
            </span>
            <button
              type="button"
              disabled={offset + PAGE >= generations.data.total}
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
