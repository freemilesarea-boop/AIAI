"use client";

/**
 * One run, in full. The most important screen in the console.
 *
 * The ordering is a claim about what matters. A failure panel sits at
 * the top, above identity, above configuration, because an operator
 * opening a failed run came to find out what happened — Step 46 is that
 * they should not have to hunt through logs first.
 *
 * Three separations are load-bearing and are not merged for tidiness:
 *
 * **Run status and worker state.** The control plane's record and the
 * worker's own view are different facts that can legitimately disagree.
 * A worker reporting RUNNING while the registry says LOST is exactly the
 * case reconciliation exists for, and one merged badge would delete it.
 *
 * **Control-plane preflight and remote preflight.** They check different
 * things on different machines. Both are shown, both can be absent, and
 * absent is never drawn as a pass.
 *
 * **Rights and leakage have no override.** There is no control on this
 * page that runs a blocked run anyway, and the panel says so rather than
 * leaving somebody hunting for one.
 */

import Link from "next/link";
import { use, useState } from "react";

import { AuditList } from "@/components/ops/AuditList";
import { CheckpointTable } from "@/components/ops/CheckpointTable";
import { LogViewer } from "@/components/ops/LogViewer";
import { MetricChart } from "@/components/ops/MetricChart";
import { OpsHeader } from "@/components/ops/OpsShell";
import { RunActions } from "@/components/ops/RunActions";
import { RunTimeline } from "@/components/ops/RunTimeline";
import {
  CopyValue,
  KeyValue,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
  Unavailable,
} from "@/components/ops/primitives";
import { runPollInterval, useOpsResource } from "@/components/ops/useOpsResource";
import { ops, opsDownload } from "@/lib/ops/client";
import {
  age,
  bytes,
  currency,
  decimal,
  duration,
  megabytes,
  num,
  runDuration,
  timestamp,
} from "@/lib/ops/format";
import type {
  CanaryRun,
  GateView,
  Preflight,
  RunDetail,
  TrainingPreflight,
  TrainingPreflightStatus,
} from "@/lib/ops/types";

/** How long ago an instant was, in seconds, or null if unparseable. */
function secondsSince(value: string | null): number | null {
  if (!value) return null;
  const parsed = new Date(value).getTime();
  if (Number.isNaN(parsed)) return null;
  return Math.max(0, (Date.now() - parsed) / 1000);
}

export default function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [pollStatus, setPollStatus] = useState<string | undefined>(undefined);

  const resource = useOpsResource<RunDetail>(
    async () => {
      const detail = await ops.run(id);
      setPollStatus(detail.run.status);
      return detail;
    },
    { deps: [id], intervalMs: runPollInterval(pollStatus) },
  );

  const detail = resource.data;
  const live = detail ? !["COMPLETED", "FAILED", "CANCELLED"].includes(detail.run.status) : false;

  return (
    <>
      <OpsHeader
        title={detail ? `Run ${detail.run.run_id}` : id}
        breadcrumb={[{ href: "/ops/training/runs", label: "Runs" }]}
        description={
          detail && (
            <span className="flex flex-wrap items-center gap-2">
              <OpsStatus status={detail.run.status} />
              <span>
                {detail.run.execution_backend} · {detail.run.experiment_name || detail.run.experiment_id}
              </span>
              {detail.run.cancel_requested_at && (
                <span
                  className="text-[var(--accent)]"
                  title={timestamp(detail.run.cancel_requested_at)}
                >
                  cancellation requested{" "}
                  {age(secondsSince(detail.run.cancel_requested_at))} — awaiting confirmation
                </span>
              )}
            </span>
          )
        }
        onRefresh={resource.refresh}
        refreshing={resource.refreshing}
      />

      {resource.error && !detail && <PanelError message={resource.error} onRetry={resource.refresh} />}
      {resource.loading && <SectionSkeleton rows={8} />}

      {detail && (
        <div className="space-y-5">
          {resource.error && (
            <PanelError message={`${resource.error} Showing the last successful read.`} />
          )}

          {detail.run.failure && <FailurePanel runId={id} detail={detail} />}

          <Panel
            title="Actions"
            subtitle="Every action is re-checked on the server. A disabled button is a courtesy, not the control."
            id="actions"
          >
            <RunActions
              runId={id}
              actions={detail.actions}
              onCompleted={() => resource.refresh()}
            />
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Lifecycle" subtitle="The Phase 25 run state machine." id="timeline">
              <RunTimeline entries={detail.timeline} />
            </Panel>

            <Panel
              title="Remote state"
              subtitle="What the worker believes, kept separate from the control plane's record."
              id="remote"
            >
              {detail.remote.available ? (
                <KeyValue
                  columns={2}
                  items={[
                    {
                      label: "Worker state",
                      value: <OpsStatus status={detail.remote.worker_state ?? "UNKNOWN"} />,
                    },
                    {
                      label: "Implies run status",
                      value: <Maybe value={detail.remote.implied_run_status} />,
                      hint:
                        detail.remote.implied_run_status &&
                        detail.remote.implied_run_status !== detail.run.status
                          ? "This differs from the control plane's record. Reconcile to establish which is current."
                          : undefined,
                    },
                    { label: "Process alive", value: <Maybe value={String(detail.remote.process_alive ?? "")} /> },
                    { label: "Exit code", value: <Maybe value={detail.remote.exit_code} /> },
                    { label: "Lease", value: <Maybe value={detail.remote.lease_id} /> },
                    { label: "Protocol", value: <Maybe value={detail.remote.protocol_version} /> },
                    {
                      label: "Plan hash on worker",
                      value: (
                        <CopyValue
                          value={detail.remote.plan_sha256}
                          display={detail.remote.plan_sha256?.slice(0, 12)}
                          label="remote plan hash"
                        />
                      ),
                    },
                    { label: "Updated", value: timestamp(detail.remote.updated_at) },
                  ]}
                />
              ) : (
                <Unavailable reason={detail.remote.unavailable_reason} />
              )}
            </Panel>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel
              title="Progress"
              subtitle="No ETA unless one can be derived from what has been measured."
              id="progress"
            >
              <KeyValue
                columns={2}
                items={[
                  { label: "Latest step", value: <Maybe value={detail.progress.latest_step} /> },
                  {
                    label: "Epoch",
                    value: (
                      <>
                        <Maybe value={detail.progress.latest_epoch} /> of{" "}
                        <Maybe value={detail.progress.total_epochs} />
                      </>
                    ),
                  },
                  {
                    label: "Elapsed",
                    value: runDuration(
                      detail.progress.elapsed_seconds,
                      detail.run.started_at,
                    ),
                  },
                  {
                    label: "Latest train loss",
                    value: decimal(detail.progress.latest_train_loss),
                  },
                  {
                    label: "Learning rate",
                    value: decimal(detail.progress.latest_learning_rate),
                  },
                  {
                    label: "Latest checkpoint",
                    value: detail.progress.latest_checkpoint_id ? (
                      <Link
                        href={`/ops/training/checkpoints/${detail.progress.latest_checkpoint_id}`}
                        className="font-mono text-xs text-[var(--brand-text)] hover:underline"
                      >
                        {detail.progress.latest_checkpoint_id}
                      </Link>
                    ) : (
                      <Maybe value={null} />
                    ),
                  },
                  {
                    label: "ETA",
                    value: <Maybe value={detail.progress.eta_seconds} />,
                    hint: detail.progress.eta_reason,
                  },
                ]}
              />
            </Panel>

            <Panel
              title="Worker and heartbeat"
              subtitle="Liveness is derived from the last heartbeat, not from the registry's status field."
              id="worker"
            >
              {detail.worker ? (
                <>
                  <KeyValue
                    columns={2}
                    items={[
                      {
                        label: "Worker",
                        value: (
                          <Link
                            href={`/ops/training/workers/${detail.worker.worker_id}`}
                            className="text-[var(--brand-text)] hover:underline"
                          >
                            {detail.worker.name}
                          </Link>
                        ),
                      },
                      { label: "Class", value: <OpsStatus status={detail.worker.worker_class} /> },
                      {
                        label: "Liveness",
                        value: <OpsStatus status={detail.heartbeat.liveness} />,
                      },
                      {
                        label: "Last heartbeat",
                        value: age(detail.heartbeat.age_seconds),
                        hint: timestamp(detail.heartbeat.timestamp),
                      },
                      { label: "Registry status", value: detail.worker.status },
                      {
                        label: "Free disk",
                        value: megabytes(detail.heartbeat.free_disk_mb),
                      },
                    ]}
                  />
                  <Telemetry detail={detail} />
                </>
              ) : (
                <Unavailable reason="No worker is assigned to this run." />
              )}
            </Panel>
          </div>

          <Panel
            title="Gates"
            subtitle="Rights and evaluation leakage have no override anywhere in this console."
            id="gates"
            tone={detail.gates.some((gate) => gate.status === "FAIL") ? "danger" : "default"}
          >
            {detail.gates_available ? (
              <ul className="space-y-2.5">
                {detail.gates.map((gate) => (
                  <GateRow key={gate.name} gate={gate} />
                ))}
              </ul>
            ) : (
              <Unavailable reason={detail.gates_unavailable_reason} />
            )}
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel
              title="Control-plane preflight"
              subtitle="Checked on this machine, before anything is transferred."
              id="control-preflight"
            >
              <PreflightPanel preflight={detail.control_preflight} />
            </Panel>
            <Panel
              title="Remote preflight"
              subtitle="Checked by the worker, on the artifacts it actually received."
              id="remote-preflight"
            >
              <PreflightPanel preflight={detail.remote_preflight} />
            </Panel>
          </div>

          <Panel
            title="Training preflight"
            subtitle={
              "Whether the selected machine can execute this plan. READY means proven; " +
              "UNVERIFIED means nobody could establish something, and is not a pass."
            }
            id="training-preflight"
            tone={detail.training_preflight.status === "BLOCKED" ? "danger" : "default"}
          >
            <TrainingPreflightPanel preflight={detail.training_preflight} />
          </Panel>

          <Panel
            title="Bounded canary"
            subtitle={
              "A training run small enough to be safe. A canary proves the mechanism and " +
              "nothing about the model; its checkpoint must never be promoted."
            }
            id="canary"
            tone={detail.canary.status === "FAILED" ? "danger" : "default"}
          >
            <CanaryPanel canary={detail.canary} />
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel
              title="Dataset and curation"
              subtitle="References and digests only. The console does not browse training audio."
              id="dataset"
            >
              <KeyValue
                columns={1}
                items={[
                  { label: "Dataset id", value: detail.dataset.dataset_id || <Maybe value={null} /> },
                  {
                    label: "Dataset lock",
                    value: (
                      <CopyValue
                        value={detail.dataset.dataset_lock_sha256}
                        display={detail.dataset.dataset_lock_sha256.slice(0, 16)}
                        label="dataset lock digest"
                      />
                    ),
                  },
                  { label: "Curation id", value: detail.dataset.curation_id || <Maybe value={null} /> },
                  {
                    label: "Curated manifest",
                    value: (
                      <CopyValue
                        value={detail.dataset.curated_manifest_sha256}
                        display={detail.dataset.curated_manifest_sha256.slice(0, 16)}
                        label="curated manifest digest"
                      />
                    ),
                  },
                  { label: "Manifest reference", value: detail.dataset.manifest_artifact_ref },
                  {
                    label: "Selected",
                    value: `${num(detail.dataset.selected_track_count)} track(s) · ${decimal(
                      detail.dataset.selected_hours,
                      2,
                    )} hours`,
                  },
                ]}
              />
            </Panel>

            <Panel
              title="Reproducibility"
              subtitle="Everything needed to rebuild this run."
              id="reproducibility"
            >
              <KeyValue
                columns={1}
                items={[
                  {
                    label: "LUBER commit",
                    value: (
                      <CopyValue
                        value={detail.reproducibility.luber_commit}
                        display={detail.reproducibility.luber_commit?.slice(0, 12)}
                        label="LUBER commit"
                      />
                    ),
                    hint:
                      detail.reproducibility.luber_dirty === true
                        ? "The working tree was dirty; this revision does not fully describe what ran."
                        : undefined,
                  },
                  {
                    label: "ACE-Step commit",
                    value: (
                      <CopyValue
                        value={detail.reproducibility.ace_step_commit}
                        display={detail.reproducibility.ace_step_commit?.slice(0, 12)}
                        label="ACE-Step commit"
                      />
                    ),
                  },
                  {
                    label: "Base model",
                    value: `${detail.reproducibility.base_model_id} @ ${
                      detail.reproducibility.base_model_upstream_commit?.slice(0, 12) ?? "UNKNOWN"
                    }`,
                  },
                  {
                    label: "Training config hash",
                    value: (
                      <CopyValue
                        value={detail.config_sha256}
                        display={detail.config_sha256.slice(0, 16)}
                        label="training config hash"
                      />
                    ),
                  },
                  {
                    label: "Training plan hash",
                    value: (
                      <CopyValue
                        value={detail.training_plan_sha256}
                        display={detail.training_plan_sha256?.slice(0, 16)}
                        label="training plan hash"
                      />
                    ),
                    hint: detail.training_plan_sha256
                      ? undefined
                      : "No plan has been compiled: the run has not been validated.",
                  },
                  {
                    label: "Environment lock",
                    value: (
                      <CopyValue
                        value={detail.reproducibility.environment_lock_digest}
                        display={detail.reproducibility.environment_lock_digest?.slice(0, 16)}
                        label="environment lock digest"
                      />
                    ),
                  },
                  {
                    label: "Worker capability signature",
                    value: (
                      <CopyValue
                        value={detail.reproducibility.worker_capability_signature}
                        display={detail.reproducibility.worker_capability_signature?.slice(0, 16)}
                        label="capability signature"
                      />
                    ),
                  },
                  {
                    label: "Python / torch",
                    value: (
                      <>
                        <Maybe value={detail.reproducibility.python_version} /> /{" "}
                        <Maybe value={detail.reproducibility.torch_version} />
                      </>
                    ),
                  },
                ]}
              />
              <p className="mt-4">
                <a
                  href={opsDownload.runBundle(id)}
                  className="text-xs text-[var(--brand-text)] underline underline-offset-2"
                >
                  Download the run bundle
                </a>
              </p>
            </Panel>
          </div>

          <Panel
            title="Training configuration"
            subtitle="Exactly the flags the installed trainer accepts. Nothing here is aspirational."
            id="config"
          >
            <KeyValue
              columns={3}
              items={[
                { label: "Strategy", value: detail.config.strategy },
                { label: "Epochs", value: num(detail.config.epochs) },
                { label: "Learning rate", value: decimal(detail.config.learning_rate) },
                { label: "Batch size", value: num(detail.config.batch_size) },
                {
                  label: "Gradient accumulation",
                  value: num(detail.config.gradient_accumulation),
                },
                { label: "Warmup steps", value: num(detail.config.warmup_steps) },
                { label: "LoRA rank", value: num(detail.config.rank) },
                { label: "Alpha", value: num(detail.config.alpha) },
                { label: "Dropout", value: decimal(detail.config.dropout, 2) },
                { label: "Precision", value: detail.config.precision },
                { label: "Optimizer", value: detail.config.optimizer_type },
                { label: "Scheduler", value: detail.config.scheduler_type },
                { label: "Seed", value: num(detail.config.seed) },
                { label: "Max grad norm", value: decimal(detail.config.max_grad_norm, 2) },
                { label: "Weight decay", value: decimal(detail.config.weight_decay, 3) },
                {
                  label: "Checkpoint every",
                  value: `${num(detail.config.checkpoint_every_epochs)} epoch(s)`,
                },
                { label: "Target modules", value: detail.config.target_modules.join(", ") },
                { label: "Attention", value: detail.config.attention_type },
              ]}
            />
          </Panel>

          <Panel
            title="Metrics"
            subtitle="Only metrics that exist are charted. Nothing is drawn empty."
            id="metrics"
          >
            {detail.metrics.length === 0 ? (
              <OpsEmpty
                title="No metrics recorded"
                description="Nothing has written a metric for this run. A run that has not started produces none, and the installed trainer computes no validation loss at all."
              />
            ) : (
              <div className="grid gap-6 xl:grid-cols-2">
                {detail.metrics.map((series) => (
                  <MetricChart key={series.metric_name} series={series} />
                ))}
              </div>
            )}
          </Panel>

          <Panel
            title="Logs"
            subtitle="Read incrementally from the server, with credentials removed before they leave it."
            id="logs"
          >
            <LogViewer runId={id} live={live} />
          </Panel>

          <Panel
            title="Artifact staging"
            subtitle="What was transferred to the worker."
            id="staging"
          >
            {detail.staging.available ? (
              <KeyValue
                columns={3}
                items={[
                  { label: "Entries", value: num(detail.staging.total_entries) },
                  { label: "Unique contents", value: num(detail.staging.unique_contents) },
                  { label: "Total size", value: bytes(detail.staging.total_bytes) },
                  {
                    label: "Present on worker",
                    value: detail.staging.presence_checked
                      ? num(detail.staging.present_entries)
                      : "not checked",
                    hint: "Presence, not integrity — the digests were verified on arrival.",
                  },
                  { label: "Missing", value: num(detail.staging.missing_entries) },
                  { label: "Built", value: timestamp(detail.staging.built_at) },
                ]}
              />
            ) : (
              <Unavailable reason={detail.staging.unavailable_reason} />
            )}
            {detail.staging.available && Object.keys(detail.staging.roles).length > 0 && (
              <details className="mt-4">
                <summary className="cursor-pointer text-xs text-[var(--text-secondary)]">
                  Breakdown by role
                </summary>
                <ul className="mt-2 space-y-1 text-xs text-[var(--text-muted)]">
                  {Object.entries(detail.staging.roles).map(([role, count]) => (
                    <li key={role}>
                      {role}: {count}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </Panel>

          <Panel title="Checkpoints" id="checkpoints">
            <CheckpointTable rows={detail.checkpoints} />
          </Panel>

          <Panel
            title="Evaluation"
            subtitle="A completed run produces checkpoints. It does not produce evidence of quality."
            id="evaluation"
          >
            {detail.evaluations.length === 0 ? (
              <OpsEmpty
                title="No evaluations"
                description="No checkpoint from this run has been judged against a baseline."
              />
            ) : (
              <ul className="space-y-2">
                {detail.evaluations.map((evaluation) => (
                  <li key={evaluation.evaluation_id} className="flex flex-wrap items-center gap-3">
                    <OpsStatus status={evaluation.status} />
                    {evaluation.qualification_outcome && (
                      <OpsStatus status={evaluation.qualification_outcome} />
                    )}
                    <Link
                      href={`/ops/training/evaluations/${evaluation.evaluation_id}`}
                      className="font-mono text-xs text-[var(--brand-text)] hover:underline"
                    >
                      {evaluation.evaluation_id}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel
            title="Cost"
            subtitle="Recorded figures only. No pricing is fetched and none is inferred."
            id="cost"
          >
            <KeyValue
              columns={3}
              items={[
                { label: "Provider", value: <Maybe value={detail.cost.provider} /> },
                { label: "Instance", value: <Maybe value={detail.cost.instance_type} /> },
                {
                  label: "Hourly rate",
                  value: currency(detail.cost.hourly_rate, detail.cost.currency),
                },
                { label: "Wall time", value: duration(detail.cost.wall_seconds) },
                { label: "GPU seconds", value: <Maybe value={detail.cost.gpu_seconds} /> },
                {
                  label: "Estimated cost",
                  value: currency(detail.cost.estimated_cost, detail.cost.currency),
                },
                {
                  label: "Actual cost",
                  value: currency(detail.cost.actual_cost, detail.cost.currency),
                },
              ]}
            />
            {detail.cost.unknown.length > 0 && (
              <ul className="mt-3 space-y-1 text-[11px] text-[var(--text-muted)]">
                {detail.cost.unknown.map((reason) => (
                  <li key={reason}>· {reason}</li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel title="Audit trail" subtitle="Append-only, in the order it happened." id="audit">
            <AuditList events={detail.audit_events} />
          </Panel>
        </div>
      )}
    </>
  );
}

/* ── failure ────────────────────────────────────────────────────────── */

function FailurePanel({ runId, detail }: { runId: string; detail: RunDetail }) {
  const failure = detail.run.failure!;
  const lost = detail.run.status === "LOST";
  const diagnostics = useOpsResource(() => ops.diagnostics(runId), { deps: [runId] });

  return (
    <Panel
      title={failure.headline}
      tone={lost ? "warning" : "danger"}
      id="failure"
      subtitle={
        <span className="font-mono text-[11px]">
          {failure.code}
          {!failure.confident && " · classification is not definitive"}
        </span>
      }
    >
      <p className="text-sm leading-relaxed text-[var(--text-primary)]">{failure.guidance}</p>

      {lost && (
        <p className="mt-3 rounded-[var(--radius-md)] border border-[var(--accent)]/40 bg-[var(--accent-muted)]/40 px-3 py-2 text-xs leading-relaxed text-[var(--accent)]">
          The remote trainer may still be running. Reconcile before retrying — this run is not
          known to have failed, only to have stopped reporting.
        </p>
      )}

      <div className="mt-4">
        <KeyValue
          columns={3}
          items={[
            { label: "Raw code", value: <span className="font-mono">{failure.code}</span> },
            {
              label: "Last heartbeat",
              value: age(detail.heartbeat.age_seconds),
              hint: timestamp(detail.heartbeat.timestamp),
            },
            {
              label: "Last metric",
              value: detail.run.latest_metric
                ? `${detail.run.latest_metric_name} ${decimal(detail.run.latest_metric.value)}`
                : "none recorded",
            },
            {
              label: "Checkpoints written",
              value: `${detail.checkpoints.length}`,
            },
            {
              label: "Worker state",
              value: <Maybe value={detail.remote.worker_state} />,
            },
            {
              label: "Exit code",
              value: <Maybe value={detail.remote.exit_code} />,
            },
          ]}
        />
      </div>

      {failure.raw_message && (
        <p className="mt-3 rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2 font-mono text-[11px] break-words text-[var(--text-secondary)]">
          {failure.raw_message}
        </p>
      )}

      <div className="mt-4">
        <p className="mb-1 text-[11px] font-medium tracking-wide text-[var(--text-muted)] uppercase">
          Last lines of stderr
        </p>
        {diagnostics.loading && <SectionSkeleton rows={2} />}
        {diagnostics.error && <PanelError message={diagnostics.error} onRetry={diagnostics.refresh} />}
        {diagnostics.data &&
          (diagnostics.data.length === 0 ? (
            <p className="text-xs text-[var(--text-muted)]">
              No stderr was captured for this run on this machine.
            </p>
          ) : (
            <pre className="max-h-48 overflow-auto rounded-[var(--radius-md)] bg-[var(--surface-sunken)] p-3 font-mono text-[11px] whitespace-pre-wrap text-[var(--text-secondary)]">
              {diagnostics.data.join("\n")}
            </pre>
          ))}
      </div>
    </Panel>
  );
}

/* ── gates and preflight ────────────────────────────────────────────── */

function GateRow({ gate }: { gate: GateView }) {
  return (
    <li className="flex flex-wrap items-start gap-2">
      <OpsStatus status={gate.status} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-[var(--text-primary)]">{gate.name}</p>
        <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">{gate.detail}</p>
        {gate.offending_count > 0 && (
          <p className="mt-1 text-[11px] text-[var(--danger)]">
            {gate.offending_count} track(s) caused this. Ids:{" "}
            <span className="font-mono">{gate.offending_ids.join(", ")}</span>
            {gate.offending_ids.length < gate.offending_count && " …"}
          </p>
        )}
      </div>
    </li>
  );
}

function PreflightPanel({ preflight }: { preflight: Preflight }) {
  if (!preflight.available) return <Unavailable reason={preflight.unavailable_reason} />;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <OpsStatus status={preflight.status} />
        <span className="text-[11px] text-[var(--text-muted)]">
          {timestamp(preflight.generated_at)}
        </span>
      </div>

      {preflight.problems.length > 0 && (
        <ul className="space-y-1 text-xs text-[var(--danger)]">
          {preflight.problems.map((problem) => (
            <li key={problem}>· {problem}</li>
          ))}
        </ul>
      )}

      {preflight.unknown.length > 0 && (
        <div className="rounded-[var(--radius-md)] border border-dashed border-[var(--border-strong)] px-3 py-2">
          <p className="text-[11px] font-medium text-[var(--text-secondary)]">
            Could not be established — not a pass
          </p>
          <ul className="mt-1 space-y-0.5 text-[11px] text-[var(--text-muted)]">
            {preflight.unknown.map((item) => (
              <li key={item}>· {item}</li>
            ))}
          </ul>
        </div>
      )}

      <ul className="space-y-1.5">
        {preflight.checks.map((check) => (
          <li key={check.name} className="flex flex-wrap items-center gap-2">
            <OpsStatus status={check.status} />
            <span className="text-xs text-[var(--text-secondary)]">{check.name}</span>
            {check.severity !== "REQUIRED" && (
              <span className="text-[10px] text-[var(--text-muted)]">{check.severity}</span>
            )}
            {check.detail && (
              <span className="text-[11px] text-[var(--text-muted)]">{check.detail}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ── phase 33: training preflight and canary ────────────────────────── */

/**
 * READY is the only status that gets a passing tone, and it gets it by
 * an explicit override. BLOCKED and UNVERIFIED fall through to the
 * shared map, where UNVERIFIED is a dashed outline rather than a
 * softer green — the whole point of the status is that it does not
 * resemble success.
 */
function TrainingPreflightStatusBadge({ status }: { status: TrainingPreflightStatus }) {
  return <OpsStatus status={status} tone={status === "READY" ? "good" : undefined} />;
}

function TrainingPreflightPanel({ preflight }: { preflight: TrainingPreflight }) {
  if (!preflight.available) return <Unavailable reason={preflight.unavailable_reason} />;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <TrainingPreflightStatusBadge status={preflight.status} />
        <span className="text-[11px] text-[var(--text-muted)]">
          {preflight.intent} · {timestamp(preflight.measured_at)}
        </span>
      </div>

      <KeyValue
        columns={2}
        items={[
          { label: "Location", value: <Maybe value={preflight.execution_location} /> },
          { label: "Device", value: <Maybe value={preflight.execution_device} /> },
          { label: "Precision", value: <Maybe value={preflight.resolved_precision} /> },
          { label: "Optimizer", value: <Maybe value={preflight.optimizer} /> },
          { label: "Target", value: <Maybe value={preflight.target_label} /> },
          {
            label: "Plan digest",
            value: preflight.plan_digest ? (
              <CopyValue
                value={preflight.plan_digest}
                display={preflight.plan_digest.slice(0, 16)}
                label="plan digest"
              />
            ) : (
              <Maybe value={null} />
            ),
          },
        ]}
      />

      <div className="flex flex-wrap gap-2">
        {[
          ["Dataset", preflight.dataset_status],
          ["Dependencies", preflight.dependency_status],
          ["Storage", preflight.storage_status],
          ["Checkpoint", preflight.checkpoint_status],
          ["Canary", preflight.canary_status],
          ["Capacity", preflight.capacity_status],
        ].map(([label, value]) => (
          <span key={label} className="inline-flex items-center gap-1.5">
            <span className="text-[11px] text-[var(--text-muted)]">{label}</span>
            <OpsStatus status={value} />
          </span>
        ))}
      </div>

      {preflight.blocking_reasons.length > 0 && (
        <ul className="space-y-1 text-xs text-[var(--danger)]">
          {preflight.blocking_reasons.map((reason) => (
            <li key={reason}>· {reason}</li>
          ))}
        </ul>
      )}

      {preflight.unverified.length > 0 && (
        <div className="rounded-[var(--radius-md)] border border-dashed border-[var(--border-strong)] px-3 py-2">
          <p className="text-[11px] font-medium text-[var(--text-secondary)]">
            Could not be established — not a pass
          </p>
          <ul className="mt-1 space-y-0.5 text-[11px] text-[var(--text-muted)]">
            {preflight.unverified.map((item) => (
              <li key={item}>· {item}</li>
            ))}
          </ul>
        </div>
      )}

      {preflight.capacity.length > 0 && (
        <div>
          <p className="text-[11px] font-medium text-[var(--text-secondary)]">Capacity evidence</p>
          <ul className="mt-1 space-y-1">
            {preflight.capacity.map((item) => (
              <li key={item.name} className="flex flex-wrap items-center gap-2">
                <OpsStatus status={item.source} />
                <span className="text-xs text-[var(--text-secondary)]">{item.name}</span>
                <span className="text-[11px] text-[var(--text-muted)]">
                  {item.value_mb === null ? "—" : `${item.value_mb} MB`}
                  {item.unified_memory && " · unified memory, shared with the OS — not VRAM"}
                  {item.derivation && ` · ${item.derivation}`}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <ul className="space-y-1.5">
        {preflight.checks.map((check) => (
          <li key={check.name} className="flex flex-wrap items-center gap-2">
            <OpsStatus status={check.status} />
            <span className="text-xs text-[var(--text-secondary)]">{check.name}</span>
            {check.reason && (
              <span className="font-mono text-[10px] text-[var(--text-muted)]">{check.reason}</span>
            )}
            {check.detail && (
              <span className="text-[11px] text-[var(--text-muted)]">{check.detail}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function CanaryPanel({ canary }: { canary: CanaryRun }) {
  if (!canary.available) return <Unavailable reason={canary.unavailable_reason} />;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <OpsStatus status={canary.status} />
        <span className="text-[11px] text-[var(--text-muted)]">
          {canary.mode}
          {canary.dataset_kind && ` · ${canary.dataset_kind} data`}
        </span>
      </div>
      <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">{canary.detail}</p>
      <KeyValue
        columns={2}
        items={[
          {
            label: "Steps",
            value: (
              <span>
                <Maybe value={canary.steps} /> of at most{" "}
                <Maybe value={canary.max_optimizer_steps} />
              </span>
            ),
          },
          {
            label: "Bounds",
            value: (
              <span>
                <Maybe value={canary.max_samples} /> sample(s) ·{" "}
                <Maybe value={canary.max_epochs} /> epoch(s)
              </span>
            ),
          },
          { label: "Exit code", value: <Maybe value={canary.exit_code} /> },
          { label: "Seconds", value: <Maybe value={canary.seconds} /> },
          {
            label: "Checkpoint",
            value:
              canary.checkpoint_ok === null ? (
                <Maybe value={null} />
              ) : (
                <OpsStatus status={canary.checkpoint_ok ? "PASS" : "FAIL"} />
              ),
          },
          {
            label: "Resume",
            value:
              canary.resume_ok === null ? (
                <Maybe value={null} />
              ) : (
                <OpsStatus status={canary.resume_ok ? "PASS" : "FAIL"} />
              ),
          },
        ]}
      />
      {canary.resume_detail && (
        <p className="text-[11px] text-[var(--text-muted)]">{canary.resume_detail}</p>
      )}
      {canary.checkpoint_problems.length > 0 && (
        <ul className="space-y-1 text-xs text-[var(--danger)]">
          {canary.checkpoint_problems.map((problem) => (
            <li key={problem}>· {problem}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ── telemetry ──────────────────────────────────────────────────────── */

function Telemetry({ detail }: { detail: RunDetail }) {
  const gpus = detail.heartbeat.gpu;
  if (gpus.length === 0 && detail.telemetry.length === 0) {
    return (
      <p className="mt-4 text-[11px] leading-relaxed text-[var(--text-muted)]">
        No hardware telemetry has been reported for this run. Nothing on this deployment has
        measured a GPU.
      </p>
    );
  }
  return (
    <div className="mt-4 space-y-3">
      {gpus.map((gpu) => (
        <div key={gpu.index} className="text-xs text-[var(--text-secondary)]">
          <p className="font-medium text-[var(--text-primary)]">GPU {gpu.index}</p>
          <p className="text-[11px] text-[var(--text-muted)]">
            utilisation <Maybe value={gpu.utilization_pct} />% · memory{" "}
            {megabytes(gpu.memory_used_mb)} / {megabytes(gpu.memory_total_mb)} · temperature{" "}
            <Maybe value={gpu.temperature_c} />°C · power <Maybe value={gpu.power_w} />W
          </p>
        </div>
      ))}
      {detail.telemetry.map((series) => (
        <MetricChart key={series.metric_name} series={series} />
      ))}
    </div>
  );
}
