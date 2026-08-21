"use client";

/**
 * One incident, and everything behind the verdict.
 *
 * The two things an operator does here are acknowledge and dismiss, and
 * neither suppresses anything. Acknowledging records that a human has
 * seen it while measurement continues; dismissing requires a reason and
 * deletes nothing, because why something was ignored is exactly what the
 * next person needs when it comes back.
 *
 * There is no "fix" button, and there will not be one in this phase.
 * Disabling a provider or moving a threshold are decisions with costs
 * this system cannot weigh.
 */

import { use, useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import { KeyValue, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference, OpsError } from "@/lib/ops/inference-client";

export default function IncidentDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [operator, setOperator] = useState("");
  const [reason, setReason] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const incident = useOpsResource(() => inference.incident(id), {
    deps: [id],
    intervalMs: 60_000,
  });

  const act = async (kind: "acknowledge" | "dismiss") => {
    setActionError(null);
    setBusy(true);
    try {
      const result =
        kind === "acknowledge"
          ? await inference.acknowledge(id, operator)
          : await inference.dismiss(id, operator, reason);
      incident.setData(result.incident);
    } catch (error) {
      setActionError(
        error instanceof OpsError ? error.message : "The action could not be completed.",
      );
    } finally {
      setBusy(false);
    }
  };

  const data = incident.data;

  return (
    <>
      <OpsHeader
        title={data?.finding_type ?? "Incident"}
        breadcrumb={[{ href: "/ops/inference/incidents", label: "Incidents" }]}
        description={data?.segment_label}
        onRefresh={incident.refresh}
        refreshing={incident.refreshing}
      />

      {incident.error && <PanelError message={incident.error} onRetry={incident.refresh} />}

      {incident.loading || !data ? (
        <SectionSkeleton />
      ) : (
        <>
          <Panel title="What was measured">
            <KeyValue
              columns={2}
              items={[
                { label: "Severity", value: data.severity },
                { label: "Worst it has been", value: data.peak_severity },
                { label: "Status", value: data.status },
                { label: "Category", value: data.category },
                { label: "Metric", value: data.metric },
                { label: "Provider", value: data.provider ?? "UNKNOWN" },
                { label: "Provider revision", value: data.provider_version ?? "UNKNOWN" },
                { label: "Occurrences", value: String(data.occurrence_count) },
                { label: "First seen", value: data.first_seen ?? "—" },
                { label: "Last seen", value: data.last_seen ?? "—" },
                {
                  label: "Baseline window",
                  value: String(data.baseline_window.start ?? "—"),
                  hint: "What normal was measured over",
                },
                {
                  label: "Current window",
                  value: String(data.current_window.start ?? "—"),
                  hint: "What it was compared against",
                },
              ]}
            />
            <p className="mt-3 text-[11px] text-[var(--text-secondary)]">{data.summary}</p>
          </Panel>

          <div className="mt-4" />
          <Panel title="Suggested next steps">
            {/* Advisory. Nothing on this page executes any of them. */}
            {data.recommendations.length === 0 ? (
              <p className="text-[11px] text-[var(--text-muted)]">None recorded.</p>
            ) : (
              <ul className="space-y-1">
                {data.recommendations.map((item) => (
                  <li key={item} className="text-[11px] text-[var(--text-secondary)]">
                    {item}
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-2 text-[11px] text-[var(--text-muted)]">
              These are things to look at. This console performs none of them, and no provider,
              policy or threshold was changed automatically.
            </p>
          </Panel>

          <div className="mt-4" />
          <Panel title="Evidence" subtitle={`${data.evidence_total} evaluations recorded.`}>
            <ul className="space-y-1.5">
              {data.evidence.map((item) => (
                <li key={item.observed_at} className="text-[11px]">
                  <span className="text-[var(--text-muted)]">{item.observed_at}</span>{" "}
                  <span className="text-[var(--text-secondary)]">{item.explanation}</span>
                </li>
              ))}
            </ul>
          </Panel>

          <div className="mt-4" />
          <Panel title="Operator actions">
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-[11px] text-[var(--text-secondary)]">
                Your name
                <input
                  value={operator}
                  onChange={(event) => setOperator(event.target.value)}
                  className="mt-1 block w-48 rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-base)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-[11px] text-[var(--text-secondary)]">
                Reason (required to dismiss)
                <input
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  className="mt-1 block w-72 rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-base)] px-2 py-1.5 text-xs"
                />
              </label>
              <button
                type="button"
                disabled={!operator || busy}
                onClick={() => act("acknowledge")}
                className="rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-xs text-[var(--text-secondary)] disabled:opacity-40"
              >
                Acknowledge
              </button>
              <button
                type="button"
                disabled={!operator || !reason || busy}
                onClick={() => act("dismiss")}
                className="rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-xs text-[var(--text-secondary)] disabled:opacity-40"
              >
                Dismiss
              </button>
            </div>
            {actionError && (
              <p className="mt-2 text-[11px] text-[var(--danger,#dc2626)]">{actionError}</p>
            )}
            <p className="mt-2 text-[11px] text-[var(--text-muted)]">
              Acknowledging does not stop measurement: evidence keeps accumulating and an
              acknowledged incident that gets worse escalates. Dismissal records the reason and
              deletes nothing.
            </p>
            {data.dismissal_reason && (
              <p className="mt-2 text-[11px] text-[var(--text-secondary)]">
                Dismissed by {data.dismissed_by}: {data.dismissal_reason}
              </p>
            )}
          </Panel>
        </>
      )}
    </>
  );
}
