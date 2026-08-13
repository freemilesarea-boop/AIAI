"use client";

/**
 * The generation workspace.
 *
 * The rule this page is built around: **pressing Create must never take
 * the workspace away from you.** The form stays live, results appear
 * beside it, and starting a second song while the first renders is
 * ordinary rather than blocked. The Create button is disabled only while
 * the POST is in the air — a few hundred milliseconds — not for the
 * minutes the engine spends working.
 *
 * Two ways in from elsewhere in the app, and they are not the same thing:
 *
 * - `?from=<id>` — **Generate again.** Prefills the settings *and*
 *   records that track as the new generation's parent.
 * - `?duplicate=<id>` — **Duplicate settings.** Prefills the same fields
 *   and records no lineage. Nothing is created until Create is pressed.
 *
 * Every backend interaction goes through `@/lib/api`; the browser never
 * contacts the model runtime.
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";

import { GenerationFailure } from "@/components/GenerationFailure";
import {
  GenerationForm,
  type GenerationFormInitialValues,
} from "@/components/GenerationForm";
import { GenerationJobCard } from "@/components/GenerationJobCard";
import { SongCard } from "@/components/SongCard";
import { Button, EmptyState } from "@/components/ui";
import { useGenerationQueue } from "@/hooks/useGenerationQueue";
import {
  getGeneration,
  type CreateGenerationInput,
  type Generation,
  type VocalGender,
} from "@/lib/api";
import { loadRecentGenerations } from "@/lib/generationStorage";

/** A draft started from an existing track. */
interface Draft {
  /** Present only for Generate again — Duplicate settings records none. */
  parent: { id: string; title: string } | null;
  key: string;
  values: Partial<GenerationFormInitialValues>;
}

/**
 * Carry a finished generation's settings into a new draft.
 *
 * No audio is reused in either mode — the pinned engine exposes no
 * audio-conditioned variation on this path, so none is implied. The only
 * difference between the two is whether a parent is recorded.
 */
function draftFrom(generation: Generation, mode: "again" | "duplicate"): Draft {
  return {
    parent: mode === "again" ? { id: generation.id, title: generation.title } : null,
    key: `${mode}:${generation.id}`,
    values: {
      title: generation.title,
      prompt: generation.prompt,
      lyrics: generation.lyrics,
      vocalGender: generation.vocal_gender as VocalGender,
      language: generation.language ?? "ko",
      duration: generation.duration_requested,
      bpm: generation.bpm === null ? "" : String(generation.bpm),
      keyScale: generation.key_scale ?? "",
      timeSignature: generation.time_signature ?? "",
      // The seed is offered, not imposed: duplicating settings starts
      // from Random unless the original pinned one, because reusing a
      // seed is a deliberate act.
      seed: generation.seed === null ? "" : String(generation.seed),
      // A draft carrying advanced settings opens in Custom, so the
      // values the user is inheriting are visible rather than hidden.
      mode:
        generation.bpm !== null ||
        generation.key_scale !== null ||
        generation.time_signature !== null ||
        generation.seed !== null
          ? "custom"
          : "simple",
    },
  };
}

function CreateWorkspace() {
  const queue = useGenerationQueue();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [recent, setRecent] = useState<Generation[]>([]);

  const handleSubmit = useCallback(
    (input: CreateGenerationInput) => {
      void queue.submit(input);
    },
    [queue],
  );

  const startDraft = useCallback(
    (generation: Generation, mode: "again" | "duplicate") => {
      setDraft(draftFrom(generation, mode));
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [],
  );

  // Arriving from a song page seeds the draft. The parameter is cleared
  // afterwards so a refresh does not silently re-apply a prefill the
  // user may have since edited away.
  const fromId = searchParams?.get("from") ?? null;
  const duplicateId = searchParams?.get("duplicate") ?? null;
  useEffect(() => {
    const sourceId = fromId ?? duplicateId;
    if (!sourceId) return;
    let cancelled = false;
    void (async () => {
      const source = await getGeneration(sourceId).catch(() => null);
      if (!source || cancelled) return;
      setDraft(draftFrom(source, fromId ? "again" : "duplicate"));
      router.replace("/create");
    })();
    return () => {
      cancelled = true;
    };
  }, [fromId, duplicateId, router]);

  // Session history: ids this browser created, minus anything already on
  // screen as a live job, so nothing is listed twice.
  const liveIds = useMemo(
    () => new Set(queue.entries.map((entry) => entry.id)),
    [queue.entries],
  );
  const finishedCount = queue.entries.filter((entry) => entry.done).length;

  useEffect(() => {
    let cancelled = false;
    const ids = loadRecentGenerations()
      .map((entry) => entry.id)
      .filter((id) => !liveIds.has(id))
      .slice(0, 6);
    if (ids.length === 0) {
      setRecent([]);
      return;
    }
    void (async () => {
      const results = await Promise.all(ids.map((id) => getGeneration(id).catch(() => null)));
      if (!cancelled) setRecent(results.filter((g): g is Generation => g !== null));
    })();
    return () => {
      cancelled = true;
    };
    // Re-read when a job finishes: that is when history gains an entry.
  }, [liveIds, finishedCount]);

  // Siblings from one CREATE are labelled so a pair reads as a pair.
  const groupPositions = useMemo(() => {
    const counts = new Map<string, number>();
    const positions = new Map<string, string>();
    for (const entry of queue.entries) {
      if (!entry.groupId) continue;
      const seen = (counts.get(entry.groupId) ?? 0) + 1;
      counts.set(entry.groupId, seen);
      positions.set(entry.id, `Result ${seen}`);
    }
    // A group of one is not a group; drop the label.
    for (const entry of queue.entries) {
      if (entry.groupId && counts.get(entry.groupId) === 1) positions.delete(entry.id);
    }
    return positions;
  }, [queue.entries]);

  /**
   * Resubmit a failed run with the settings it recorded.
   *
   * A fresh Idempotency-Key is minted inside `submit`, so this is a new
   * generation rather than a resurrection of the failed one. Retried as
   * a single result: the user asked for this take specifically.
   */
  const handleRetry = useCallback(
    (generation: Generation) => {
      void queue.submit({
        title: generation.title,
        prompt: generation.prompt,
        lyrics: generation.lyrics,
        vocal_gender: generation.vocal_gender as VocalGender,
        language: generation.language ?? "ko",
        duration: generation.duration_requested,
        bpm: generation.bpm,
        key_scale: generation.key_scale,
        time_signature: generation.time_signature,
        parent_generation_id: generation.parent_generation_id,
        seed: null,
        result_count: 1,
      });
    },
    [queue],
  );

  const refreshRecent = useCallback((updated: Generation) => {
    setRecent((current) =>
      current.map((item) => (item.id === updated.id ? updated : item)),
    );
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Create</h1>
        <p className="mt-1.5 text-sm text-[var(--text-secondary)]">
          Describe your track, add lyrics, and generate a finished master.
        </p>
      </header>

      <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
        <div className="lg:sticky lg:top-8">
          {/* Remounting on a new draft is what applies the prefill: the
              form owns its field state, so a fresh instance is the
              cleanest way to seed it without fighting the user's edits. */}
          <GenerationForm
            key={draft?.key ?? "blank"}
            onSubmit={handleSubmit}
            busy={queue.submitting}
            initialValues={draft?.values}
            parent={draft?.parent ?? null}
            onClearParent={() => setDraft(null)}
          />
        </div>

        <div className="flex flex-col gap-6">
          {queue.submitError && (
            <GenerationFailure
              error={queue.submitError}
              onDismiss={queue.clearSubmitError}
            />
          )}

          {queue.entries.length === 0 ? (
            <EmptyState
              title="Your tracks appear here"
              description="Fill in the form and press Create. You can keep working while a song generates, and start another at any time."
            />
          ) : (
            <section aria-labelledby="queue-heading" className="flex flex-col gap-3">
              <div className="flex items-center justify-between gap-3">
                <h2 id="queue-heading" className="text-sm font-semibold">
                  This session
                </h2>
                {finishedCount > 0 && (
                  <Button size="sm" variant="ghost" onClick={queue.clearFinished}>
                    Clear finished
                  </Button>
                )}
              </div>
              {queue.entries.map((entry) => (
                <GenerationJobCard
                  key={entry.id}
                  entry={entry}
                  resultLabel={groupPositions.get(entry.id)}
                  onDismiss={queue.dismiss}
                  onRetry={handleRetry}
                  onGenerateAgain={(generation) => startDraft(generation, "again")}
                />
              ))}
            </section>
          )}

          {recent.length > 0 && (
            <section aria-labelledby="recent-heading" className="flex flex-col gap-3">
              <h2
                id="recent-heading"
                className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]"
              >
                Recent generations
              </h2>
              {recent.map((generation) => (
                <SongCard
                  key={generation.id}
                  generation={generation}
                  onChanged={refreshRecent}
                  onGenerateAgain={(item) => startDraft(item, "again")}
                />
              ))}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * `useSearchParams` requires a Suspense boundary under the App Router;
 * without one the whole route opts out of static rendering.
 */
export default function CreatePage() {
  return (
    <Suspense fallback={null}>
      <CreateWorkspace />
    </Suspense>
  );
}
