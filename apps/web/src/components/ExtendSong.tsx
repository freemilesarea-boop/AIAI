"use client";

/**
 * Extend: append newly generated music to the end of a finished song.
 *
 * The smallest possible surface for the first real editing feature — a
 * button, three lengths, and the existing queue takes over. No timeline,
 * no waveform, no engine vocabulary: the user picks how much longer, and
 * everything about how the continuation is conditioned stays on the
 * server.
 *
 * Lengths that would push the song past the maximum are not offered.
 * Showing a choice the backend will reject teaches people to distrust
 * the controls.
 */

import { useState } from "react";

import { Button, Card, cx } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import {
  EXTENSION_CHOICES,
  MAX_SONG_SECONDS,
  extendGeneration,
  findMasterAsset,
  type Generation,
} from "@/lib/api";

export interface ExtendSongProps {
  generation: Generation;
  /** Called with the queued child's id once the request is accepted. */
  onExtended?: (generationId: string) => void;
}

/** Seconds of source audio, from the master when one is recorded. */
export function sourceSeconds(generation: Generation): number {
  const master = findMasterAsset(generation);
  return master?.duration ?? generation.duration_actual ?? generation.duration_requested;
}

/** Which lengths still fit inside the maximum song length. */
export function availableExtensions(generation: Generation): number[] {
  const current = sourceSeconds(generation);
  return EXTENSION_CHOICES.filter((seconds) => current + seconds <= MAX_SONG_SECONDS);
}

export function ExtendSong({ generation, onExtended }: ExtendSongProps) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const choices = availableExtensions(generation);
  const ready = generation.status === "COMPLETED" && findMasterAsset(generation) !== null;

  // Nothing to offer: either the song is not finished, or it is already
  // long enough that no supported extension fits.
  if (!ready || choices.length === 0) return null;

  const submit = async (seconds: number) => {
    setBusy(true);
    try {
      const created = await extendGeneration(generation.id, seconds);
      toast.notify(`Extending by ${seconds}s — the new part is generating`);
      setOpen(false);
      onExtended?.(created.generation_id);
    } catch {
      toast.notifyError("Could not extend this song.");
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <Button onClick={() => setOpen(true)} aria-expanded={false}>
        Extend
      </Button>
    );
  }

  return (
    <Card className="w-full p-4">
      <h3 className="text-sm font-semibold">Extend this song</h3>
      <p className="mt-1 text-xs text-[var(--text-secondary)]">
        The existing recording is kept and new music is generated onto the end of it, using
        this song&rsquo;s own brief and lyrics. You get a new song; this one is unchanged.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {choices.map((seconds) => (
          <Button
            key={seconds}
            variant="primary"
            size="sm"
            busy={busy}
            onClick={() => void submit(seconds)}
          >
            {`+${seconds}s`}
          </Button>
        ))}
        <Button size="sm" disabled={busy} onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
      <p className={cx("mt-2 text-[11px]", "text-[var(--text-muted)]")}>
        {`Currently ${Math.round(sourceSeconds(generation))}s · maximum ${MAX_SONG_SECONDS}s`}
      </p>
    </Card>
  );
}
