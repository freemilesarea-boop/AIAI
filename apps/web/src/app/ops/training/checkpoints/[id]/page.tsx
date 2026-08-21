"use client";

/**
 * One checkpoint, its lineage, and whether anything has judged it.
 *
 * The panel that matters most is the one that says whether this can be
 * evaluated. A MOCK artifact says no, with the reason, because the
 * distinction between a placeholder and trained weights is the one
 * mistake here that could reach a listener.
 *
 * Physical location is a scheme and a presence flag. The filesystem
 * layout of the machine the console runs on is not something an operator
 * needs in a browser, and a path pasted from a console into an issue is
 * how a home directory ends up in a tracker.
 */

import Link from "next/link";
import { use } from "react";

import { AuditList } from "@/components/ops/AuditList";
import { OpsHeader } from "@/components/ops/OpsShell";
import {
  CopyValue,
  KeyValue,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { ops } from "@/lib/ops/client";
import { bytes, decimal, timestamp } from "@/lib/ops/format";

export default function CheckpointDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const resource = useOpsResource(() => ops.checkpoint(id), { deps: [id] });
  const detail = resource.data;

  return (
    <>
      <OpsHeader
        title={id}
        breadcrumb={[{ href: "/ops/training/checkpoints", label: "Checkpoints" }]}
        description={
          detail && (
            <span className="flex flex-wrap items-center gap-2">
              <OpsStatus status={detail.checkpoint.kind} />
              <OpsStatus status={detail.checkpoint.status} />
              {!detail.checkpoint.is_real_model && (
                <span className="rounded-[var(--radius-sm)] bg-[var(--accent-muted)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--accent)]">
                  TEST ONLY — contains no trained weights
                </span>
              )}
            </span>
          )
        }
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      {resource.error && !detail && <PanelError message={resource.error} onRetry={resource.refresh} />}
      {resource.loading && <SectionSkeleton rows={5} />}

      {detail && (
        <div className="space-y-5">
          <Panel
            title="Evaluation eligibility"
            tone={detail.checkpoint.can_evaluate ? "default" : "warning"}
            id="eligibility"
          >
            {detail.checkpoint.can_evaluate ? (
              <p className="text-sm text-[var(--text-primary)]">
                This checkpoint is READY and contains trained weights, so it may be nominated as
                an evaluation candidate.
                {detail.checkpoint.candidate_id && (
                  <>
                    {" "}
                    It already is:{" "}
                    <span className="font-mono text-xs">{detail.checkpoint.candidate_id}</span>.
                  </>
                )}
              </p>
            ) : (
              <p className="text-sm text-[var(--accent)]">
                {detail.checkpoint.evaluate_blocked_reason}
              </p>
            )}
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Identity" id="identity">
              <KeyValue
                columns={1}
                items={[
                  { label: "Checkpoint id", value: detail.checkpoint.checkpoint_id },
                  { label: "Kind", value: <OpsStatus status={detail.checkpoint.kind} /> },
                  { label: "Status", value: <OpsStatus status={detail.checkpoint.status} /> },
                  { label: "Format", value: detail.checkpoint.checkpoint_format },
                  { label: "Step", value: <Maybe value={detail.checkpoint.step} /> },
                  { label: "Epoch", value: <Maybe value={detail.checkpoint.epoch} /> },
                  { label: "Size", value: bytes(detail.checkpoint.size_bytes) },
                  {
                    label: "Digest",
                    value: (
                      <CopyValue
                        value={detail.checkpoint.sha256}
                        display={detail.checkpoint.sha256?.slice(0, 20)}
                        label="checkpoint digest"
                      />
                    ),
                  },
                  {
                    label: "Location",
                    value: (
                      <>
                        <Maybe value={detail.checkpoint.location_scheme} />
                        {detail.checkpoint.location_present === false && (
                          <span className="ml-2 text-[var(--danger)]">bytes not found</span>
                        )}
                      </>
                    ),
                    hint: "Scheme only. The console does not publish filesystem paths.",
                  },
                  { label: "Created", value: timestamp(detail.checkpoint.created_at) },
                  { label: "Finalized", value: timestamp(detail.checkpoint.finalized_at) },
                ]}
              />
            </Panel>

            <Panel
              title="Lineage"
              subtitle="Baseline → experiment → run → checkpoint → evaluation → qualification."
              id="lineage"
            >
              <ol className="space-y-2 text-sm">
                <li>
                  <span className="text-[11px] tracking-wide text-[var(--text-muted)] uppercase">
                    Experiment
                  </span>
                  <br />
                  {detail.experiment ? (
                    <Link
                      href={`/ops/training/experiments/${detail.experiment.experiment_id}`}
                      className="text-[var(--brand-text)] hover:underline"
                    >
                      {detail.experiment.name}
                    </Link>
                  ) : (
                    <Maybe value={null} />
                  )}
                </li>
                <li>
                  <span className="text-[11px] tracking-wide text-[var(--text-muted)] uppercase">
                    Run
                  </span>
                  <br />
                  <Link
                    href={`/ops/training/runs/${detail.checkpoint.run_id}`}
                    className="font-mono text-xs text-[var(--brand-text)] hover:underline"
                  >
                    {detail.checkpoint.run_id}
                  </Link>
                  {detail.run && (
                    <span className="ml-2">
                      <OpsStatus status={detail.run.status} />
                    </span>
                  )}
                </li>
                <li>
                  <span className="text-[11px] tracking-wide text-[var(--text-muted)] uppercase">
                    Evaluations
                  </span>
                  <br />
                  {detail.evaluations.length === 0 ? (
                    <span className="text-xs text-[var(--text-muted)]">none</span>
                  ) : (
                    detail.evaluations.map((evaluation) => (
                      <Link
                        key={evaluation.evaluation_id}
                        href={`/ops/training/evaluations/${evaluation.evaluation_id}`}
                        className="mr-3 font-mono text-xs text-[var(--brand-text)] hover:underline"
                      >
                        {evaluation.evaluation_id}
                      </Link>
                    ))
                  )}
                </li>
                <li>
                  <span className="text-[11px] tracking-wide text-[var(--text-muted)] uppercase">
                    Qualification
                  </span>
                  <br />
                  {detail.qualifications.length === 0 ? (
                    <span className="text-xs text-[var(--text-muted)]">no verdict recorded</span>
                  ) : (
                    detail.qualifications.map((qualification) => (
                      <span key={qualification.evaluation_id} className="mr-2">
                        <OpsStatus status={qualification.outcome} />
                      </span>
                    ))
                  )}
                </li>
              </ol>
            </Panel>
          </div>

          <Panel
            title="Metrics snapshot"
            subtitle="Recorded when the checkpoint was written. Training loss is context, not evidence of quality."
            id="metrics"
          >
            {Object.keys(detail.checkpoint.metrics_snapshot).length === 0 ? (
              <OpsEmpty
                title="No snapshot"
                description="No metric values were recorded alongside this checkpoint."
              />
            ) : (
              <KeyValue
                columns={3}
                items={Object.entries(detail.checkpoint.metrics_snapshot).map(([name, value]) => ({
                  label: name,
                  value: decimal(value),
                }))}
              />
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
