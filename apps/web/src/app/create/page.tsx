"use client";

/**
 * The generation workspace.
 *
 * Desktop: form on the left, status/result on the right. Mobile: single
 * column, form first. Every backend interaction goes through
 * `@/lib/api` — the browser never contacts the model runtime.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { GenerationFailure } from "@/components/GenerationFailure";
import { GenerationForm } from "@/components/GenerationForm";
import { GenerationResult } from "@/components/GenerationResult";
import { GenerationStatusPanel } from "@/components/GenerationStatusPanel";
import { RecentGenerations } from "@/components/RecentGenerations";
import { useGenerationJob } from "@/hooks/useGenerationJob";
import { getGeneration, type CreateGenerationInput, type Generation } from "@/lib/api";
import { describeGenerationFailure } from "@/lib/errors";
import { loadRecentGenerations } from "@/lib/generationStorage";

export default function CreatePage() {
  const job = useGenerationJob();
  const [recent, setRecent] = useState<Generation[]>([]);
  const lastInputRef = useRef<CreateGenerationInput | null>(null);

  const handleSubmit = useCallback(
    (input: CreateGenerationInput) => {
      lastInputRef.current = input;
      void job.submit(input);
    },
    [job],
  );

  const handleRetry = useCallback(() => {
    // A new Idempotency-Key is minted inside submit(), so this is a new job.
    const previous = lastInputRef.current;
    if (previous) void job.submit(previous);
    else job.reset();
  }, [job]);

  // Session list. The actively-tracked generation is taken from the
  // poller rather than re-fetched, so history never duplicates a
  // request that polling is already making.
  const [historyIds, setHistoryIds] = useState<string[]>([]);
  useEffect(() => {
    setHistoryIds(loadRecentGenerations().map((e) => e.id));
  }, [job.phase, job.generationId]);

  useEffect(() => {
    let cancelled = false;
    const others = historyIds.filter((id) => id !== job.generationId);
    if (others.length === 0) {
      setRecent([]);
      return;
    }
    void (async () => {
      const results = await Promise.all(
        others.map((id) => getGeneration(id).catch(() => null)),
      );
      if (!cancelled) setRecent(results.filter((g): g is Generation => g !== null));
    })();
    return () => {
      cancelled = true;
    };
  }, [historyIds, job.generationId]);

  // Merge the live generation into the list, preserving recency order.
  const historyItems = useMemo(() => {
    const byId = new Map<string, Generation>();
    if (job.generation) byId.set(job.generation.id, job.generation);
    for (const g of recent) byId.set(g.id, g);
    return historyIds
      .map((id) => byId.get(id))
      .filter((g): g is Generation => g !== undefined);
  }, [historyIds, recent, job.generation]);

  const status = job.generation?.status ?? null;
  const isBusy = job.phase === "submitting" || job.phase === "tracking";
  const completed = job.phase === "done" && job.generation?.status === "COMPLETED";
  const failedGeneration =
    job.phase === "done" &&
    (job.generation?.status === "FAILED" || job.generation?.status === "CANCELLED")
      ? job.generation
      : null;

  const showDeveloperDetails = useMemo(
    () => process.env.NEXT_PUBLIC_SHOW_GENERATION_DETAILS === "true",
    [],
  );

  return (
    <div className="flex flex-col gap-2">
      <header>
        <h1 className="text-3xl font-bold tracking-tight text-zinc-50">Create</h1>
        <p className="mt-1.5 text-zinc-400">
          Describe your track, add lyrics, and generate a finished master.
        </p>
      </header>

      <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
        <div className="lg:sticky lg:top-8">
          <GenerationForm onSubmit={handleSubmit} busy={isBusy} disabled={isBusy} />
        </div>

        <div className="flex flex-col gap-6">
          {job.phase === "idle" && (
            <section className="rounded-xl border border-dashed border-zinc-800 p-6">
              <h2 className="text-lg font-medium text-zinc-300">Your track appears here</h2>
              <p className="mt-1.5 text-sm text-zinc-500">
                Fill in the form and press Generate. Creating a 30-second track usually takes
                under a minute.
              </p>
            </section>
          )}

          {(job.phase === "submitting" || job.phase === "tracking") && (
            <GenerationStatusPanel
              status={status}
              elapsedSeconds={job.elapsedSeconds}
              submitting={job.phase === "submitting"}
            />
          )}

          {job.phase === "stalled" && (
            <section className="rounded-xl border border-amber-900/60 bg-amber-950/20 p-6">
              <h2 className="text-lg font-semibold text-amber-200">Still working</h2>
              <p className="mt-2 text-sm text-amber-100/90">
                This is taking longer than expected. Your track may still be generating —
                refresh the page to reconnect.
              </p>
            </section>
          )}

          {job.phase === "error" && job.error && (
            <GenerationFailure
              error={job.error}
              onRetry={handleRetry}
              onDismiss={job.reset}
            />
          )}

          {failedGeneration && (
            <GenerationFailure
              error={describeGenerationFailure(failedGeneration.error_code)}
              onRetry={handleRetry}
              onDismiss={job.reset}
            />
          )}

          {completed && job.generation && (
            <GenerationResult
              generation={job.generation}
              onCreateAnother={job.reset}
              showTechnicalDetails={showDeveloperDetails}
            />
          )}

          <RecentGenerations
            items={historyItems}
            activeId={job.generationId}
            onSelect={job.reconnect}
          />
        </div>
      </div>
    </div>
  );
}
