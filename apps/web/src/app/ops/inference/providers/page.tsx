"use client";

/**
 * Provider revisions, side by side.
 *
 * The comparison is over the *same window* for both revisions, because
 * comparing one revision's Tuesday with another's Saturday compares the
 * days as much as the models. Even then it is not an experiment:
 * requests are not randomised between revisions, so a revision serving a
 * different traffic mix can look worse for reasons that have nothing to
 * do with its weights. The caveat is on the page, not in a footnote.
 */

import { useState } from "react";

import { WindowPicker } from "@/components/ops/InferenceControls";
import { OpsHeader } from "@/components/ops/OpsShell";
import { OpsEmpty, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { inference } from "@/lib/ops/inference-client";
import type { WindowSize } from "@/lib/ops/inference-types";

interface Delta {
  status: string;
  before: number | null;
  after: number | null;
  absolute_delta?: number | null;
}

export default function ProvidersPage() {
  const [window, setWindow] = useState<WindowSize>("7d");
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");

  const providers = useOpsResource(() => inference.providers(window), { deps: [window] });
  const comparison = useOpsResource(
    () =>
      left && right
        ? inference.compareRevisions({ left, right, window })
        : Promise.resolve(null),
    { deps: [left, right, window] },
  );

  const deltas = (comparison.data?.deltas ?? {}) as Record<string, Delta>;

  return (
    <>
      <OpsHeader
        title="Providers"
        description="Every revision observed, and an honest comparison between two of them."
        onRefresh={providers.refresh}
        refreshing={providers.refreshing}
      >
        <WindowPicker value={window} onChange={setWindow} />
      </OpsHeader>

      {providers.error && <PanelError message={providers.error} onRetry={providers.refresh} />}

      <Panel title="Revisions observed in this window">
        {providers.loading ? (
          <SectionSkeleton />
        ) : !providers.data?.providers.length ? (
          <OpsEmpty
            title="No providers observed"
            description="Nothing has been ingested for this window."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="border-b border-[var(--border-subtle)] text-left text-[var(--text-muted)]">
                  <th className="py-1.5 pr-3 font-medium">Revision</th>
                  <th className="py-1.5 pr-3 font-medium">Samples</th>
                  <th className="py-1.5 pr-3 font-medium">Baseline</th>
                  <th className="py-1.5 pr-3 font-medium">Success</th>
                  <th className="py-1.5 pr-3 font-medium">First-candidate accept</th>
                  <th className="py-1.5 pr-3 font-medium">Retry</th>
                  <th className="py-1.5 font-medium">Compare</th>
                </tr>
              </thead>
              <tbody>
                {providers.data.providers.map((provider) => (
                  <tr
                    key={provider.provider_revision}
                    className="border-b border-[var(--border-subtle)] last:border-0"
                  >
                    <td className="py-1.5 pr-3 text-[var(--text-secondary)]">
                      {provider.provider_revision}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                      {provider.sample_count}
                    </td>
                    <td className="py-1.5 pr-3 text-[var(--text-muted)]">
                      {/* A revision that shipped this morning has no
                          history. Saying so beats giving it rates that
                          look comparable to a revision with a week. */}
                      {provider.baseline_status === "READY" ? "ready" : "building"}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                      {provider.rates.generation_success_rate?.render ?? "NO_DATA"}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                      {provider.rates.first_candidate_accept_rate?.render ?? "NO_DATA"}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                      {provider.rates.quality_retry_rate?.render ?? "NO_DATA"}
                    </td>
                    <td className="py-1.5">
                      <div className="flex gap-1">
                        <button
                          type="button"
                          onClick={() => setLeft(provider.provider_revision)}
                          aria-pressed={left === provider.provider_revision}
                          className="rounded-[var(--radius-sm)] border border-[var(--border-default)] px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)]"
                        >
                          A
                        </button>
                        <button
                          type="button"
                          onClick={() => setRight(provider.provider_revision)}
                          aria-pressed={right === provider.provider_revision}
                          className="rounded-[var(--radius-sm)] border border-[var(--border-default)] px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)]"
                        >
                          B
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="mt-4" />
      <Panel
        title="Comparison"
        subtitle={left && right ? `A: ${left} · B: ${right}` : "Pick an A and a B above."}
      >
        {!left || !right ? (
          <OpsEmpty
            title="Nothing selected"
            description="Choose two revisions to compare over the same window."
          />
        ) : comparison.loading ? (
          <SectionSkeleton rows={2} />
        ) : comparison.data?.status !== "OK" ? (
          <OpsEmpty
            title="Not enough data to compare"
            description="Each side needs enough observations before a difference between them means anything."
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-[var(--border-subtle)] text-left text-[var(--text-muted)]">
                    <th className="py-1.5 pr-3 font-medium">Metric</th>
                    <th className="py-1.5 pr-3 font-medium">A</th>
                    <th className="py-1.5 pr-3 font-medium">B</th>
                    <th className="py-1.5 font-medium">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(deltas).map(([metric, delta]) => (
                    <tr
                      key={metric}
                      className="border-b border-[var(--border-subtle)] last:border-0"
                    >
                      <td className="py-1.5 pr-3 text-[var(--text-secondary)]">{metric}</td>
                      <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                        {delta.status === "OK" ? delta.before?.toFixed(4) : "NO_DATA"}
                      </td>
                      <td className="py-1.5 pr-3 tabular-nums text-[var(--text-muted)]">
                        {delta.status === "OK" ? delta.after?.toFixed(4) : "NO_DATA"}
                      </td>
                      <td className="py-1.5 tabular-nums text-[var(--text-muted)]">
                        {delta.status === "OK" && delta.absolute_delta !== null
                          ? (delta.absolute_delta ?? 0).toFixed(4)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] text-[var(--text-muted)]">
              {String(comparison.data?.caveat ?? "")}
            </p>
          </>
        )}
      </Panel>
    </>
  );
}
