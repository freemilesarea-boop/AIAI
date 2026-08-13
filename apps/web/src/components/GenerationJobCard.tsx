"use client";

/**
 * One generation in the Create workspace, from queued to ready.
 *
 * A card, not a page state: several of these are on screen at once, each
 * reporting its own status, because Phase 12 lets a user start a second
 * song while the first is still rendering. Nothing here knows about the
 * others.
 *
 * Playback goes through the global player. A completed card deliberately
 * does *not* mount its own `<audio>` element — two cards with two
 * elements would play over each other, and neither would survive
 * navigating to the Library.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { GenerationFailure } from "@/components/GenerationFailure";
import { GenerationStatusPanel } from "@/components/GenerationStatusPanel";
import { SongActions } from "@/components/SongActions";
import { trackFromGeneration, usePlayer } from "@/components/player/PlayerProvider";
import { Button, Card, StatusPill } from "@/components/ui";
import { findMasterAsset, findPreviewAsset, type Generation, type Project } from "@/lib/api";
import { describeGenerationFailure } from "@/lib/errors";
import { formatDuration } from "@/lib/generationStatus";
import type { QueueEntry } from "@/hooks/useGenerationQueue";

export interface GenerationJobCardProps {
  entry: QueueEntry;
  projects?: Project[];
  onDismiss: (id: string) => void;
  onChanged?: (generation: Generation) => void;
  onGenerateAgain?: (generation: Generation) => void;
  /** Resubmit the same settings after a failure. */
  onRetry?: (generation: Generation) => void;
  /** Position in a multi-result group, when there is one. */
  resultLabel?: string;
}

/** Ticks only while a job is running; stops the moment it is not. */
function useElapsed(startedAt: number, running: boolean): number {
  const [seconds, setSeconds] = useState(() => Math.floor((Date.now() - startedAt) / 1000));
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(
      () => setSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(id);
  }, [startedAt, running]);
  return seconds;
}

export function GenerationJobCard({
  entry,
  projects,
  onDismiss,
  onChanged,
  onGenerateAgain,
  onRetry,
  resultLabel,
}: GenerationJobCardProps) {
  const player = usePlayer();
  const generation = entry.generation;
  const status = generation?.status ?? null;
  const running = !entry.done && !entry.stalled;
  const elapsed = useElapsed(entry.startedAt, running);

  const failed = status === "FAILED" || status === "CANCELLED";
  const ready = status === "COMPLETED";
  const track = generation && ready ? trackFromGeneration(generation) : null;
  const isCurrent = player.track?.id === entry.id;
  const master = generation ? findMasterAsset(generation) : null;
  const preview = generation ? findPreviewAsset(generation) : null;

  return (
    // A labelled group rather than a landmark region: several of these
    // are on screen at once, and a page of landmarks is noise.
    <Card role="group" aria-label={generation?.title ?? entry.title} className="p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {resultLabel && (
            <p className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              {resultLabel}
            </p>
          )}
          <h3 className="truncate text-sm font-semibold text-[var(--text-primary)]">
            {ready && generation ? (
              <Link href={`/song/${generation.id}`} className="hover:underline">
                {generation.title}
              </Link>
            ) : (
              entry.title
            )}
          </h3>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {status && <StatusPill status={status} />}
          <button
            type="button"
            onClick={() => onDismiss(entry.id)}
            aria-label={`Dismiss ${entry.title}`}
            className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="m12 10.6 5-5 1.4 1.4-5 5 5 5-1.4 1.4-5-5-5 5L5.6 17l5-5-5-5L7 5.6l5 5Z" />
            </svg>
          </button>
        </div>
      </div>

      {running && (
        <div className="mt-3">
          <GenerationStatusPanel
            status={status}
            elapsedSeconds={elapsed}
            submitting={status === null}
          />
        </div>
      )}

      {entry.stalled && (
        <p className="mt-3 rounded-[var(--radius-md)] border border-[var(--accent)]/40 bg-[var(--accent-muted)] px-3 py-2 text-sm text-[var(--text-secondary)]">
          This is taking longer than expected. The track may still be generating — refresh to
          reconnect.
        </p>
      )}

      {failed && generation && (
        <div className="mt-3">
          <GenerationFailure
            error={describeGenerationFailure(generation.error_code)}
            // A real resubmission under a fresh Idempotency-Key, built
            // from the settings this run recorded — not a prefill the
            // user has to press Create on again.
            onRetry={onRetry ? () => onRetry(generation) : undefined}
          />
        </div>
      )}

      {ready && generation && (
        <div className="mt-3 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
            <span className="font-mono tabular-nums">
              {formatDuration(generation.duration_actual ?? master?.duration ?? null)}
            </span>
            {master && (
              <span>
                Master WAV · {master.sample_rate / 1000} kHz
                {master.bit_depth ? ` · ${master.bit_depth}-bit` : ""}
              </span>
            )}
            {preview && (
              <span>
                Preview MP3
                {preview.bitrate ? ` · ${Math.round(preview.bitrate / 1000)} kbps` : ""}
              </span>
            )}
            {generation.seed !== null && (
              <span className="font-mono tabular-nums">Seed {generation.seed}</span>
            )}
          </div>

          {/* The controls this run actually used, when any were set.
              Absent means the engine chose — never a substituted default. */}
          {(generation.bpm !== null ||
            generation.key_scale !== null ||
            generation.time_signature !== null) && (
            <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
              {generation.bpm !== null && (
                <div className="flex gap-1.5">
                  <dt>BPM</dt>
                  <dd className="font-mono tabular-nums text-[var(--text-secondary)]">
                    {generation.bpm}
                  </dd>
                </div>
              )}
              {generation.key_scale !== null && (
                <div className="flex gap-1.5">
                  <dt>Key</dt>
                  <dd className="text-[var(--text-secondary)]">{generation.key_scale}</dd>
                </div>
              )}
              {generation.time_signature !== null && (
                <div className="flex gap-1.5">
                  <dt>Time</dt>
                  <dd className="font-mono tabular-nums text-[var(--text-secondary)]">
                    {generation.time_signature}
                  </dd>
                </div>
              )}
            </dl>
          )}

          <div className="flex flex-wrap items-center gap-2">
            {track && (
              <Button
                variant="primary"
                size="sm"
                onClick={() => (isCurrent ? player.toggle() : player.play(track))}
              >
                {isCurrent && player.playing ? "Pause" : "Play"}
              </Button>
            )}
            {onGenerateAgain && (
              <Button size="sm" onClick={() => onGenerateAgain(generation)}>
                Generate again
              </Button>
            )}
          </div>

          <SongActions
            generation={generation}
            projects={projects}
            onChanged={onChanged}
            onDeleted={() => onDismiss(entry.id)}
          />
        </div>
      )}
    </Card>
  );
}
