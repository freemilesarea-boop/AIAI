"use client";

/**
 * Circuits: which providers the system is currently refusing to call.
 *
 * Read-only, and the page says so where an operator will look for a
 * button. The override is `python -m luber_provider_resilience`, which
 * runs wherever the database is reachable — including production, where
 * this console does not exist. A button here would be an incident tool
 * that is absent during incidents.
 *
 * Three things are shown together on purpose. **State** without the
 * **policy** is a number with no scale: "3 consecutive failures" means
 * nothing until you know the threshold is 5. And both without the
 * **transition log** cannot answer the question an operator actually
 * arrives with, which is not "what is broken" but "what changed".
 */

import { useMemo } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  DataTable,
  KeyValue,
  Maybe,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { resilience } from "@/lib/ops/resilience-client";
import type { CapabilityReadiness, Circuit, Transition } from "@/lib/ops/resilience-types";

/** Circuits move on their own, so this page refreshes without being asked. */
const REFRESH_MS = 15_000;

function when(value: string | null): string {
  if (!value) return "";
  const at = new Date(value);
  return Number.isNaN(at.getTime()) ? value : at.toLocaleString();
}

/** A rate, or an honest blank. Never 0% standing in for no samples. */
function rate(circuit: Circuit): string {
  if (circuit.failure_rate === null) return "";
  return `${Math.round(circuit.failure_rate * 100)}% of ${circuit.sample_count}`;
}

export default function CircuitsPage() {
  const circuits = useOpsResource(() => resilience.circuits(), { intervalMs: REFRESH_MS });
  const readiness = useOpsResource(() => resilience.readiness(), { intervalMs: REFRESH_MS });
  const policy = useOpsResource(() => resilience.policy());
  const transitions = useOpsResource(() => resilience.transitions(25), {
    intervalMs: REFRESH_MS,
  });

  const columns = useMemo(
    () => [
      {
        key: "provider",
        header: "Provider",
        render: (row: Circuit) => row.provider,
      },
      { key: "task", header: "Task", render: (row: Circuit) => row.task_type },
      {
        key: "state",
        header: "Circuit",
        render: (row: Circuit) => (
          <span className="inline-flex flex-wrap items-center gap-1.5">
            <OpsStatus status={row.state} title={row.open_reason ?? undefined} />
            {row.control === "MANUAL" && (
              <OpsStatus
                status="MANUAL"
                label="operator held"
                title={row.manual_reason ?? undefined}
              />
            )}
          </span>
        ),
      },
      {
        key: "failures",
        header: "Consecutive failures",
        numeric: true,
        render: (row: Circuit) => row.consecutive_failures,
      },
      {
        key: "rate",
        header: "Failure rate (window)",
        render: (row: Circuit) => <Maybe value={rate(row)} />,
      },
      {
        key: "probes",
        header: "Probes",
        numeric: true,
        render: (row: Circuit) =>
          row.state === "HALF_OPEN" ? `${row.active_probes} active` : "—",
      },
      {
        key: "reason",
        header: "Why",
        render: (row: Circuit) => (
          <span className="text-[var(--text-muted)]">
            <Maybe value={row.open_reason ?? row.last_failure_category} />
          </span>
        ),
      },
      {
        key: "changed",
        header: "Last change",
        render: (row: Circuit) => <Maybe value={when(row.last_transition_at)} />,
      },
    ],
    [],
  );

  const transitionColumns = useMemo(
    () => [
      {
        key: "at",
        header: "When",
        render: (row: Transition) => when(row.occurred_at),
      },
      {
        key: "circuit",
        header: "Circuit",
        render: (row: Transition) => `${row.provider} · ${row.task_type}`,
      },
      {
        key: "change",
        header: "Change",
        render: (row: Transition) => (
          <span className="inline-flex items-center gap-1.5">
            <OpsStatus status={row.previous_state} />
            <span aria-hidden="true" className="text-[var(--text-muted)]">
              →
            </span>
            <OpsStatus status={row.current_state} />
          </span>
        ),
      },
      {
        key: "who",
        header: "By",
        render: (row: Transition) =>
          row.automatic ? (
            <span className="text-[var(--text-muted)]">automatic</span>
          ) : (
            <Maybe value={row.operator ?? "operator"} />
          ),
      },
      { key: "reason", header: "Reason", render: (row: Transition) => row.reason },
    ],
    [],
  );

  const capabilities: CapabilityReadiness[] = readiness.data?.capabilities ?? [];

  return (
    <>
      <OpsHeader
        title="Circuits"
        description="Which providers are being called, which are being refused, and what changed."
        onRefresh={() => {
          circuits.refresh();
          readiness.refresh();
          transitions.refresh();
        }}
        refreshing={circuits.refreshing || readiness.refreshing}
      />

      {circuits.error && <PanelError message={circuits.error} onRetry={circuits.refresh} />}

      <Panel
        title="What can be generated right now"
        subtitle="Derived from providers and circuit state, not stored — so it cannot report a capability as healthy after its circuit has opened."
      >
        {readiness.loading ? (
          <SectionSkeleton />
        ) : !readiness.data ? (
          <OpsEmpty title="No readiness data" description="The API returned nothing." />
        ) : (
          <>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <OpsStatus
                status={
                  !readiness.data.generation_available
                    ? "UNAVAILABLE"
                    : readiness.data.degraded
                      ? "DEGRADED"
                      : "OK"
                }
              />
              <span className="text-sm text-[var(--text-secondary)]">
                {readiness.data.summary}
              </span>
            </div>
            <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {capabilities.map((capability) => (
                <li
                  key={capability.capability}
                  className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-[var(--text-primary)]">
                      {capability.capability}
                    </span>
                    <OpsStatus status={capability.status} />
                  </div>
                  <p className="mt-1 text-[11px] text-[var(--text-muted)]">{capability.detail}</p>
                </li>
              ))}
            </ul>
          </>
        )}
      </Panel>

      <Panel
        title="Circuits"
        subtitle="One per provider and task type, because a broken cover path is not a broken service."
      >
        {circuits.loading ? (
          <SectionSkeleton />
        ) : (
          <>
            <DataTable
              rows={circuits.data?.circuits ?? []}
              columns={columns}
              rowKey={(row) => row.circuit_key}
              caption="Provider circuits"
              empty={
                <OpsEmpty
                  title="No circuits yet"
                  description="Nothing has been recorded against a provider on this deployment. A circuit is written the first time an attempt counts."
                />
              }
            />
            {(circuits.data?.unconfigured_providers.length ?? 0) > 0 && (
              <p className="mt-3 text-[11px] text-[var(--text-muted)]">
                Circuits exist for providers this deployment no longer configures:{" "}
                {circuits.data?.unconfigured_providers.join(", ")}. They are shown rather than
                hidden — an open circuit against a removed provider explains nothing on its own,
                and hiding it would make it impossible to explain at all.
              </p>
            )}
          </>
        )}
      </Panel>

      <Panel
        title="Policy in force"
        subtitle="The thresholds these numbers are measured against. Changing them is a deployment change, not a console action."
      >
        {policy.loading ? (
          <SectionSkeleton rows={2} />
        ) : !policy.data ? (
          <OpsEmpty title="No policy" description="The API returned nothing." />
        ) : (
          <>
            <KeyValue
              columns={3}
              items={[
                {
                  label: "Resilience",
                  value: policy.data.resilience_enabled ? "enabled" : "disabled",
                  hint: policy.data.resilience_enabled
                    ? undefined
                    : "Providers are called directly; no circuit is engaged.",
                },
                {
                  label: "Failover",
                  value: policy.data.failover_mode,
                  hint: policy.data.failover_possible
                    ? undefined
                    : "No request can be moved: this deployment has one routable provider. The circuit breaker still applies.",
                },
                {
                  label: "Routable providers",
                  value: policy.data.routable_providers.join(", ") || "none",
                },
                {
                  label: "Opens after",
                  value: `${policy.data.circuit_policy.consecutive_failure_threshold} consecutive failures`,
                },
                {
                  label: "Or a failure rate above",
                  value: `${Number(policy.data.circuit_policy.failure_rate_threshold) * 100}% over ${policy.data.circuit_policy.minimum_samples}+ samples`,
                },
                {
                  label: "Cooldown",
                  value: `${policy.data.circuit_policy.open_duration_seconds}s, doubling to ${policy.data.circuit_policy.maximum_open_duration_seconds}s`,
                },
              ]}
            />
            <p className="mt-4 text-[11px] text-[var(--text-muted)]">
              This view changes nothing. To hold a circuit open or force one closed, use{" "}
              <code className="text-[var(--text-secondary)]">
                python -m luber_provider_resilience open|close|reset
              </code>
              , which works where this console does not — see docs/PROVIDER_INCIDENT_RUNBOOK.md.
            </p>
          </>
        )}
      </Panel>

      <Panel
        title="What changed"
        subtitle="Every transition, automatic and manual, newest first."
      >
        {transitions.loading ? (
          <SectionSkeleton />
        ) : (
          <DataTable
            rows={transitions.data?.transitions ?? []}
            columns={transitionColumns}
            rowKey={(row) => row.id}
            caption="Circuit transitions"
            empty={
              <OpsEmpty
                title="No transitions"
                description="No circuit has changed state on this deployment."
              />
            }
          />
        )}
      </Panel>
    </>
  );
}
