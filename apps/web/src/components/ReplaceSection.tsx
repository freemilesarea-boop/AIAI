"use client";

/**
 * Replace a section: regenerate one span of a song and keep the rest.
 *
 * Time inputs, not a waveform. Phase 13C exists to prove the engine can
 * genuinely inpaint; a timeline editor is a much larger piece of work and
 * would not make the proof any stronger. No new dependency is pulled in
 * for this.
 *
 * The preset buttons are honest about what they know: "Last 10 seconds"
 * is a time range and is labelled as one. BOORDA has no idea where a verse
 * or a chorus begins, so it does not offer to replace either.
 */

import { useState } from "react";

import { Button, Card, cx, inputClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import {
  MIN_PRESERVED_SECONDS,
  MIN_REPLACE_SECONDS,
  findMasterAsset,
  replaceGenerationRange,
  type Generation,
} from "@/lib/api";

export interface ReplaceSectionProps {
  generation: Generation;
  onReplaced?: (generationId: string) => void;
}

/** Seconds of source audio, from the master when one is recorded. */
export function songSeconds(generation: Generation): number {
  const master = findMasterAsset(generation);
  return master?.duration ?? generation.duration_actual ?? generation.duration_requested;
}

/** `m:ss`, the way a player shows a position. */
export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/**
 * Accepts `m:ss` or plain seconds, because people type both.
 * Returns `null` for anything unreadable.
 */
export function parseTimeInput(raw: string): number | null {
  const text = raw.trim();
  if (!text) return null;
  const clock = /^(\d+):([0-5]?\d(?:\.\d+)?)$/.exec(text);
  if (clock) return Number(clock[1]) * 60 + Number(clock[2]);
  if (!/^\d+(\.\d+)?$/.test(text)) return null;
  return Number(text);
}

/** Why a range cannot be submitted, or `null` when it can. */
export function validateRange(
  start: number | null,
  end: number | null,
  songLength: number,
): string | null {
  if (start === null || end === null) return "Enter a start and end time.";
  if (start < 0) return "Start cannot be negative.";
  if (end <= start) return "End must come after start.";
  if (end > songLength + 0.05) {
    return `End cannot be past the song (${formatClock(songLength)}).`;
  }
  if (end - start < MIN_REPLACE_SECONDS) {
    return `Replace at least ${MIN_REPLACE_SECONDS} second.`;
  }
  if (songLength - (end - start) < MIN_PRESERVED_SECONDS) {
    return `Leave at least ${MIN_PRESERVED_SECONDS} second of the original.`;
  }
  return null;
}

export function ReplaceSection({ generation, onReplaced }: ReplaceSectionProps) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [startText, setStartText] = useState("");
  const [endText, setEndText] = useState("");
  const [prompt, setPrompt] = useState("");
  const [touched, setTouched] = useState(false);

  const songLength = songSeconds(generation);
  const ready = generation.status === "COMPLETED" && findMasterAsset(generation) !== null;
  // A song has to be long enough to hold a replaced second and keep one.
  const longEnough = songLength >= MIN_REPLACE_SECONDS + MIN_PRESERVED_SECONDS;

  if (!ready || !longEnough) return null;

  const start = parseTimeInput(startText);
  const end = parseTimeInput(endText);
  const error = validateRange(start, end, songLength);

  const applyPreset = (seconds: number) => {
    setTouched(false);
    setStartText(formatClock(Math.max(0, songLength - seconds)));
    setEndText(formatClock(songLength));
  };

  const submit = async () => {
    setTouched(true);
    if (error || start === null || end === null) return;
    setBusy(true);
    try {
      const created = await replaceGenerationRange(generation.id, {
        startSeconds: start,
        endSeconds: end,
        prompt: prompt.trim() || undefined,
      });
      toast.notify(`Replacing ${formatClock(start)}–${formatClock(end)}`);
      setOpen(false);
      onReplaced?.(created.generation_id);
    } catch {
      toast.notifyError("Could not replace that section.");
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <Button onClick={() => setOpen(true)} aria-expanded={false}>
        Replace section
      </Button>
    );
  }

  const presets = [10, 15].filter(
    (seconds) => songLength - seconds >= MIN_PRESERVED_SECONDS && seconds >= MIN_REPLACE_SECONDS,
  );

  return (
    <Card className="w-full p-4">
      <h3 className="text-sm font-semibold">Replace a section</h3>
      <p className="mt-1 text-xs text-[var(--text-secondary)]">
        Everything outside the times you choose stays exactly as it is — only that span is
        generated again. You get a new song; this one is unchanged. This song is{" "}
        {formatClock(songLength)} long.
      </p>

      {presets.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
            Quick range
          </span>
          {presets.map((seconds) => (
            <Button key={seconds} size="sm" onClick={() => applyPreset(seconds)}>
              {`Last ${seconds} seconds`}
            </Button>
          ))}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor="replace-start" className="block text-xs font-medium">
            Start
          </label>
          <input
            id="replace-start"
            value={startText}
            onChange={(e) => {
              setStartText(e.target.value);
              setTouched(true);
            }}
            placeholder="0:10"
            inputMode="numeric"
            className={cx(inputClass, "mt-1 !w-28 py-1.5 text-sm")}
          />
        </div>
        <div>
          <label htmlFor="replace-end" className="block text-xs font-medium">
            End
          </label>
          <input
            id="replace-end"
            value={endText}
            onChange={(e) => {
              setEndText(e.target.value);
              setTouched(true);
            }}
            placeholder="0:20"
            inputMode="numeric"
            className={cx(inputClass, "mt-1 !w-28 py-1.5 text-sm")}
          />
        </div>
      </div>

      <div className="mt-3">
        <label htmlFor="replace-prompt" className="block text-xs font-medium">
          New description <span className="font-normal text-[var(--text-muted)]">— optional</span>
        </label>
        <input
          id="replace-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={generation.prompt}
          className={cx(inputClass, "mt-1 py-1.5 text-sm")}
        />
        <p className="mt-1 text-[11px] text-[var(--text-muted)]">
          {/* Honest about the limit: the engine conditions the request as
              a whole, and BOORDA has no lyric-to-time alignment. */}
          Steers the regenerated part. Lyrics stay as they are — BOORDA does not yet know which
          words fall at which time.
        </p>
      </div>

      {touched && error && (
        <p role="alert" className="mt-2 text-xs text-[var(--danger)]">
          {error}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" busy={busy} onClick={() => void submit()}>
          Replace section
        </Button>
        <Button size="sm" disabled={busy} onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </Card>
  );
}
