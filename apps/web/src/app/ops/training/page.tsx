"use client";

/**
 * Training overview: what exists, and what the infrastructure can do.
 *
 * Every number here is counted from the registry. There is no target, no
 * percentage and no health score — a single figure would have to average
 * "three runs failed on rights" with "one worker is stale", and those
 * need different actions from different people.
 *
 * The system panel is where the honesty costs something. It would be
 * easy to write GPU READY next to a worker that has a GPU-shaped field
 * filled in. It says what a probe established instead, and where nothing
 * was probed it says so.
 */

import Link from "next/link";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  KeyValue,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import { timestamp } from "@/lib/ops/format";
import type { CountBreakdown } from "@/lib/ops/types";

function Counts({ title, counts, href }: { title: string; counts: CountBreakdown; href: string }) {
  const states = Object.entries(counts.by_state).filter(([name]) => name);
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-overlay)] p-3">
      <Link
        href={href}
        className="flex items-baseline justify-between hover:text-[var(--brand-text)]"
      >
        <span className="text-xs font-medium text-[var(--text-secondary)]">{title}</span>
        <span className="text-lg font-semibold tabular-nums text-[var(--text-primary)]">
          {counts.total.toLocaleString()}
        </span>
      </Link>
      {states.length > 0 ? (
        <ul className="mt-2 space-y-1">
          {states.map(([state, count]) => (
            <li key={state} className="flex items-center justify-between gap-2">
              <OpsStatus status={state} />
              <span className="text-xs tabular-nums text-[var(--text-secondary)]">{count}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-[11px] text-[var(--text-muted)]">None yet.</p>
      )}
    </div>
  );
}

export default function OverviewPage() {
  const overview = useOpsResource(() => ops.overview(), { intervalMs: 15_000 });
  const baseline = useOpsResource(() => ops.baseline());

  return (
    <>
      <OpsHeader
        title="Training overview"
        description={
          overview.data
            ? `Read from the registry at ${timestamp(overview.data.generated_at)}.`
            : undefined
        }
        onRefresh={() => {
          overview.refresh();
          baseline.refresh();
        }}
        refreshing={overview.refreshing}
      />

      {overview.error && !overview.data && <PanelError message={overview.error} onRetry={overview.refresh} />}

      {overview.loading && <SectionSkeleton rows={4} />}

      {overview.data && (
        <div className="space-y-5">
          {overview.error && (
            <PanelError
              message={`${overview.error} The figures below are the last that were read.`}
            />
          )}

          {overview.data.empty_reason ? (
            <Panel title="Nothing has been recorded yet">
              <OpsEmpty
                title="This registry is empty"
                description={overview.data.empty_reason}
                action={
                  <Link
                    href="/ops/training/experiments/new"
                    className="text-xs text-[var(--brand-text)] underline underline-offset-2"
                  >
                    Create the first experiment
                  </Link>
                }
              />
            </Panel>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Counts
                title="Experiments"
                counts={overview.data.experiments}
                href="/ops/training/experiments"
              />
              <Counts title="Runs" counts={overview.data.runs} href="/ops/training/runs" />
              <Counts title="Workers" counts={overview.data.workers} href="/ops/training/workers" />
              <Counts
                title="Checkpoints"
                counts={overview.data.checkpoints}
                href="/ops/training/checkpoints"
              />
            </div>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel
              title="System status"
              subtitle="What this deployment can actually do, on the evidence."
              id="system"
            >
              <ul className="space-y-2.5">
                {overview.data.system.map((check) => (
                  <li key={check.name} className="flex flex-wrap items-start gap-2">
                    <OpsStatus status={check.status} />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-medium text-[var(--text-primary)]">{check.name}</p>
                      <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">
                        {check.detail}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            </Panel>

            <Panel
              title="Evaluation and qualification"
              subtitle="Qualified means a checkpoint may advance to promotion review. It is not production."
              id="qualification"
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <Counts
                  title="Evaluation runs"
                  counts={overview.data.evaluations}
                  href="/ops/training/evaluations"
                />
                <Counts
                  title="Qualification verdicts"
                  counts={overview.data.qualifications}
                  href="/ops/training/evaluations"
                />
              </div>
            </Panel>
          </div>

          <Panel
            title="Worker classes"
            subtitle="A class is granted by a probe, never by an operator asserting it."
            id="classes"
          >
            <div className="flex flex-wrap gap-2">
              {Object.entries(overview.data.worker_classes.by_state).map(([name, count]) => (
                <span key={name} className="inline-flex items-center gap-1.5">
                  <OpsStatus status={name} />
                  <span className="text-xs tabular-nums text-[var(--text-secondary)]">{count}</span>
                </span>
              ))}
              {overview.data.worker_classes.total === 0 && (
                <p className="text-xs text-[var(--text-muted)]">
                  No workers are registered. Register one with{" "}
                  <code className="font-mono">luber-training remote worker register</code>.
                </p>
              )}
            </div>
          </Panel>

          <Panel
            title="Current production baseline"
            subtitle={baseline.data?.note}
            id="baseline"
          >
            {baseline.error && <PanelError message={baseline.error} onRetry={baseline.refresh} />}
            {baseline.loading && <SectionSkeleton rows={2} />}
            {baseline.data &&
              (baseline.data.production.length === 0 ? (
                <OpsEmpty
                  title="No model is marked PRODUCTION"
                  description="The product serves the unmodified ACE-Step baseline. Nothing in this console changes that."
                />
              ) : (
                <div className="space-y-3">
                  {baseline.data.production.map((model) => (
                    <KeyValue
                      key={model.model_id}
                      columns={3}
                      items={[
                        { label: "Model", value: model.model_name },
                        { label: "Version", value: model.model_version },
                        { label: "Stage", value: <OpsStatus status={model.stage} /> },
                        { label: "Upstream commit", value: model.upstream_commit.slice(0, 12) },
                        { label: "Architecture", value: model.architecture },
                        { label: "Identity basis", value: model.identity_basis },
                      ]}
                    />
                  ))}
                </div>
              ))}
          </Panel>
        </div>
      )}
    </>
  );
}
