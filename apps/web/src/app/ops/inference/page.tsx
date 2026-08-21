"use client";

/**
 * Inference health: is anything wrong, and where.
 *
 * Every card carries three things — a value, the counts behind it, and
 * the window it covers. A card showing "4%" alone would be unreadable:
 * it could be 12 of 300 or 2 of 50, and those call for different
 * responses. A window with nothing in it says NO_DATA rather than 0%,
 * because zero failures out of zero requests is a green light for a
 * system that was switched off.
 *
 * The charts refuse to interpolate across a gap and draw thin buckets
 * hollow, for the same reason: the shape of a line should not be able to
 * imply something the samples do not support.
 */

import Link from "next/link";
import { useCallback, useState } from "react";

import {
  PartialDataBanner,
  StaleBanner,
  WindowPicker,
} from "@/components/ops/InferenceControls";
import { OpsHeader } from "@/components/ops/OpsShell";
import { OpsEmpty, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { TimeSeriesChart } from "@/components/ops/TimeSeriesChart";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference } from "@/lib/ops/inference-client";
import type { Rate, Regression, WindowSize } from "@/lib/ops/inference-types";

/** Inference analytics does not need sub-second updates. */
const POLL_MS = 60_000;

const CARDS: { key: string; label: string; goodIsHigh: boolean }[] = [
  { key: "generation_success_rate", label: "Success", goodIsHigh: true },
  { key: "first_candidate_accept_rate", label: "First-candidate accept", goodIsHigh: true },
  { key: "quality_retry_rate", label: "Retry", goodIsHigh: false },
  { key: "retry_exhaustion_rate", label: "Retry exhaustion", goodIsHigh: false },
  { key: "provider_failure_rate", label: "Provider failure", goodIsHigh: false },
  { key: "early_collapse_rate", label: "Early collapse", goodIsHigh: false },
];

function RateCard({ label, rate }: { label: string; rate: Rate | undefined }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-overlay)] p-3">
      <p className="text-[11px] font-medium text-[var(--text-secondary)]">{label}</p>
      {!rate || rate.status === "NO_DATA" ? (
        <>
          <p className="mt-1 text-lg font-semibold text-[var(--text-muted)]">NO_DATA</p>
          <p className="text-[11px] text-[var(--text-muted)]">0 samples</p>
        </>
      ) : (
        <>
          <p className="mt-1 text-lg font-semibold tabular-nums text-[var(--text-primary)]">
            {rate.percent?.toFixed(2)}%
          </p>
          {/* The counts are not a tooltip. A percentage whose sample size
              is hidden is a percentage that will be over-read. */}
          <p className="text-[11px] tabular-nums text-[var(--text-muted)]">
            {rate.numerator}/{rate.denominator}
            {rate.excluded > 0 ? ` · ${rate.excluded} excluded` : ""}
          </p>
        </>
      )}
    </div>
  );
}

function SeverityTag({ severity }: { severity: string }) {
  const tone =
    severity === "CRITICAL"
      ? "border-[var(--danger,#dc2626)] text-[var(--danger,#dc2626)]"
      : severity === "MAJOR"
        ? "border-[var(--accent)] text-[var(--accent)]"
        : "border-[var(--border-default)] text-[var(--text-secondary)]";
  return (
    <span className={`rounded-[var(--radius-sm)] border px-1.5 py-0.5 text-[10px] font-medium ${tone}`}>
      {severity}
    </span>
  );
}

function RegressionRow({ finding }: { finding: Regression }) {
  return (
    <li className="border-b border-[var(--border-subtle)] py-2 last:border-0">
      <div className="flex flex-wrap items-baseline gap-2">
        <SeverityTag severity={finding.severity} />
        <span className="text-xs font-medium text-[var(--text-primary)]">
          {finding.finding_type}
        </span>
        <span className="text-[11px] text-[var(--text-muted)]">{finding.category}</span>
      </div>
      <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{finding.explanation}</p>
      <p className="mt-0.5 text-[11px] text-[var(--text-muted)]">
        Threshold: {finding.threshold_crossed}. Suggested:{" "}
        {finding.recommendations.join(", ") || "none"}.
      </p>
    </li>
  );
}

export default function InferenceHealthPage() {
  const [window, setWindow] = useState<WindowSize>("24h");
  const [revision, setRevision] = useState<string>("");

  const filters = { window, revision: revision || undefined };

  const overview = useOpsResource(() => inference.overview(filters), {
    deps: [window, revision],
    intervalMs: POLL_MS,
  });
  const retry = useOpsResource(() => inference.trend("retry", filters), {
    deps: [window, revision],
    intervalMs: POLL_MS,
  });
  const failure = useOpsResource(() => inference.trend("failure", filters), {
    deps: [window, revision],
    intervalMs: POLL_MS,
  });
  const latency = useOpsResource(() => inference.trend("latency", filters), {
    deps: [window, revision],
    intervalMs: POLL_MS,
  });
  const regressions = useOpsResource(() => inference.regressions({ window }), {
    deps: [window],
    intervalMs: POLL_MS,
  });
  const providers = useOpsResource(() => inference.providers(window), { deps: [window] });
  const segments = useOpsResource(
    () => inference.segments({ window, group_by: "provider,duration_bucket" }),
    { deps: [window] },
  );
  const ingest = useOpsResource(() => inference.ingestStatus(), { intervalMs: POLL_MS });

  const refreshAll = useCallback(() => {
    overview.refresh();
    retry.refresh();
    failure.refresh();
    latency.refresh();
    regressions.refresh();
    providers.refresh();
    segments.refresh();
    ingest.refresh();
  }, [overview, retry, failure, latency, regressions, providers, segments, ingest]);

  const summary = overview.data?.summary;

  return (
    <>
      <OpsHeader
        title="Inference health"
        description="Trends over Phase 29 candidate traces. Nothing here changes anything."
        onRefresh={refreshAll}
        refreshing={overview.refreshing}
      >
        <WindowPicker value={window} onChange={setWindow} />
      </OpsHeader>

      <StaleBanner status={ingest.data} />
      <PartialDataBanner coverage={summary?.coverage} />

      {overview.error && <PanelError message={overview.error} onRetry={overview.refresh} />}

      <Panel
        title="Health"
        subtitle={
          summary
            ? `${summary.sample_count.toLocaleString()} generations in this window.`
            : undefined
        }
      >
        {overview.loading ? (
          <SectionSkeleton />
        ) : summary && summary.sample_count === 0 ? (
          <OpsEmpty
            title="No generations in this window"
            description="Nothing ran, or nothing has been ingested yet. This is not a health score of zero."
          />
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            {CARDS.map((card) => (
              <RateCard key={card.key} label={card.label} rate={summary?.overview[card.key]} />
            ))}
          </div>
        )}
        {summary && (
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] sm:grid-cols-4">
            <div className="flex justify-between gap-2">
              <dt className="text-[var(--text-muted)]">P95 total latency</dt>
              <dd className="tabular-nums text-[var(--text-secondary)]">
                {summary.latency.total_latency_seconds?.p95 !== null &&
                summary.latency.total_latency_seconds?.p95 !== undefined
                  ? `${summary.latency.total_latency_seconds.p95.toFixed(1)}s`
                  : "NO_DATA"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[var(--text-muted)]">Provider calls / request</dt>
              <dd className="tabular-nums text-[var(--text-secondary)]">
                {summary.averages.average_provider_calls_per_generation?.value?.toFixed(2) ??
                  "NO_DATA"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[var(--text-muted)]">Open incidents</dt>
              <dd className="tabular-nums text-[var(--text-secondary)]">
                <Link href="/ops/inference/incidents" className="hover:text-[var(--brand-text)]">
                  {overview.data?.open_incidents ?? 0}
                </Link>
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[var(--text-muted)]">Cancelled</dt>
              <dd className="tabular-nums text-[var(--text-secondary)]">
                {summary.counters.cancelled_generations}
              </dd>
            </div>
          </dl>
        )}
      </Panel>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {retry.data && (
          <TimeSeriesChart
            title="Retry behaviour"
            points={retry.data.points}
            unit="rate"
            series={[
              { key: "first_candidate_accept_rate", label: "Accepted first", unit: "rate" },
              { key: "quality_retry_rate", label: "Retried", unit: "rate" },
              { key: "retry_exhaustion_rate", label: "Exhausted", unit: "rate" },
            ]}
          />
        )}
        {failure.data && (
          <TimeSeriesChart
            title="Failures"
            points={failure.data.points}
            unit="rate"
            series={[
              { key: "invalid_audio_rate", label: "Invalid audio", unit: "rate" },
              { key: "early_collapse_rate", label: "Early collapse", unit: "rate" },
              { key: "duration_failure_rate", label: "Duration", unit: "rate" },
              { key: "provider_failure_rate", label: "Provider", unit: "rate" },
            ]}
          />
        )}
        {latency.data && (
          <TimeSeriesChart
            title="Latency (P95)"
            points={latency.data.points}
            unit="seconds"
            caption="Each point is the 95th percentile within its bucket."
            series={[
              { key: "total_latency_seconds", label: "Total", unit: "seconds" },
              { key: "provider_latency_seconds", label: "Provider", unit: "seconds" },
              { key: "qc_latency_seconds", label: "QC", unit: "seconds" },
            ]}
          />
        )}

        <Panel title="Regressions in this window">
          {regressions.loading ? (
            <SectionSkeleton rows={2} />
          ) : !regressions.data?.length ? (
            <OpsEmpty
              title="Nothing crossed a threshold"
              description="Every watched metric is within its normal range, or there were too few samples to say."
            />
          ) : (
            <ul>
              {regressions.data.map((finding) => (
                <RegressionRow
                  key={`${finding.finding_type}-${finding.metric}-${finding.segment_label}`}
                  finding={finding}
                />
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <Panel title="QC findings">
          {!summary ? (
            <SectionSkeleton rows={2} />
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <h3 className="text-[11px] font-medium text-[var(--text-secondary)]">
                  Rejections
                </h3>
                {Object.keys(summary.findings.critical).length === 0 ? (
                  <p className="mt-1 text-[11px] text-[var(--text-muted)]">None.</p>
                ) : (
                  <ul className="mt-1 space-y-0.5">
                    {Object.entries(summary.findings.critical).map(([code, count]) => (
                      <li key={code} className="flex justify-between gap-2 text-[11px]">
                        <span className="text-[var(--text-secondary)]">{code}</span>
                        <span className="tabular-nums text-[var(--text-muted)]">{count}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div>
                {/* Under its own heading, never in the same list: a
                    harshness advisory and invalid audio are not the same
                    news, and the finishing engine exists partly to fix
                    the first. */}
                <h3 className="text-[11px] font-medium text-[var(--text-secondary)]">
                  Advisories on delivered audio
                </h3>
                <p className="text-[10px] text-[var(--text-muted)]">Not failures.</p>
                {Object.keys(summary.findings.soft).length === 0 ? (
                  <p className="mt-1 text-[11px] text-[var(--text-muted)]">None.</p>
                ) : (
                  <ul className="mt-1 space-y-0.5">
                    {Object.entries(summary.findings.soft).map(([code, count]) => (
                      <li key={code} className="flex justify-between gap-2 text-[11px]">
                        <span className="text-[var(--text-secondary)]">{code}</span>
                        <span className="tabular-nums text-[var(--text-muted)]">{count}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </Panel>

        <Panel
          title="Most affected segments"
          subtitle={
            segments.data
              ? `Ranked by failure rate. ${segments.data.segments_below_minimum} segment(s) had too few samples to rank.`
              : undefined
          }
        >
          {segments.loading ? (
            <SectionSkeleton rows={2} />
          ) : !segments.data?.segments.length ? (
            <OpsEmpty
              title="No segment has enough samples"
              description="Ranking without a minimum would put a one-request segment at the top of every list."
            />
          ) : (
            <ul className="space-y-1">
              {segments.data.segments.map((segment) => (
                <li
                  key={segment.segment_label}
                  className="flex flex-wrap items-baseline justify-between gap-2 text-[11px]"
                >
                  <span className="text-[var(--text-secondary)]">{segment.segment_label}</span>
                  <span className="tabular-nums text-[var(--text-muted)]">{segment.render}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="mt-4" />
      <Panel title="Providers" subtitle="Filter the whole page to one revision.">
        {providers.loading ? (
          <SectionSkeleton rows={2} />
        ) : !providers.data?.providers.length ? (
          <OpsEmpty title="No providers observed" description="Nothing has been ingested." />
        ) : (
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              aria-pressed={revision === ""}
              onClick={() => setRevision("")}
              className={
                revision === ""
                  ? "rounded-[var(--radius-md)] bg-[var(--surface-overlay)] px-3 py-1.5 text-xs text-[var(--text-primary)]"
                  : "rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-xs text-[var(--text-secondary)]"
              }
            >
              All revisions
            </button>
            {providers.data.providers.map((provider) => (
              <button
                key={provider.provider_revision}
                type="button"
                aria-pressed={revision === provider.provider_revision}
                onClick={() => setRevision(provider.provider_revision)}
                className={
                  revision === provider.provider_revision
                    ? "rounded-[var(--radius-md)] bg-[var(--surface-overlay)] px-3 py-1.5 text-xs text-[var(--text-primary)]"
                    : "rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-xs text-[var(--text-secondary)]"
                }
              >
                {provider.provider_revision}
                <span className="ml-1.5 text-[10px] text-[var(--text-muted)]">
                  {provider.sample_count}
                  {provider.baseline_status === "BASELINE_BUILDING" ? " · new" : ""}
                </span>
              </button>
            ))}
          </div>
        )}
      </Panel>

      <p className="mt-4 text-[11px] text-[var(--text-muted)]">
        {overview.data?.automatic_remediation ??
          "This console detects and explains; every action is an operator's."}
      </p>
    </>
  );
}
