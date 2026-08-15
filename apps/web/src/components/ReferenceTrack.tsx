"use client";

/**
 * Reference Track: upload a song to steer the sound of a new one.
 *
 * The whole component is built around not overstating what the engine
 * does. A reference shifts production character — brightness, density,
 * texture — within the territory the prompt sets. It does not clone a
 * voice, copy a song, or transfer a style, and the copy here says the
 * one thing Phase 13E's measurements actually support.
 *
 * Five states, each of which is a fact rather than an impression:
 *
 *   EMPTY      nothing chosen
 *   SELECTED   a file is chosen, upload not finished
 *   UPLOADING  bytes in flight
 *   READY      the backend accepted it and issued an id
 *   ERROR      it was refused, and the reason is shown
 *
 * Only READY yields an id, and only an id can reach a generation
 * request. A file sitting in the browser has no effect on anything, so
 * the form is told so and refuses to submit as a referenced generation.
 *
 * Limits come from the server. They are not mirrored as constants here:
 * two copies drift, and the copy shown to the user would be the wrong
 * one. If they cannot be loaded, uploading is disabled and says why
 * rather than falling back to numbers this file made up.
 */

import { useCallback, useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui";
import {
  ApiError,
  fetchReferenceAudioLimits,
  uploadReferenceAudio,
  type ReferenceAudioAsset,
  type ReferenceAudioLimits,
} from "@/lib/api";

export type ReferenceStatus = "EMPTY" | "SELECTED" | "UPLOADING" | "READY" | "ERROR";

export interface ReferenceTrackProps {
  /** Fires whenever the attached reference changes; `null` means none. */
  onChange: (referenceId: string | null) => void;
  /** Lets the form block submission while a reference is mid-flight. */
  onStatusChange?: (status: ReferenceStatus) => void;
  disabled?: boolean;
}

function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${Math.round(mb)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function formatMinutes(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  if (minutes >= 1) return `${minutes} min`;
  return `${Math.round(seconds)} sec`;
}

export function ReferenceTrack({ onChange, onStatusChange, disabled = false }: ReferenceTrackProps) {
  const inputId = useId();
  const statusId = useId();
  const describedById = useId();

  const [limits, setLimits] = useState<ReferenceAudioLimits | null>(null);
  const [limitsFailed, setLimitsFailed] = useState(false);
  const [status, setStatus] = useState<ReferenceStatus>("EMPTY");
  const [fileName, setFileName] = useState<string | null>(null);
  const [attached, setAttached] = useState<ReferenceAudioAsset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement | null>(null);
  // Guards against a slow first upload resolving after a second one and
  // re-attaching the reference the user already replaced.
  const uploadSeq = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchReferenceAudioLimits(controller.signal)
      .then(setLimits)
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setLimitsFailed(true);
      });
    return () => controller.abort();
  }, []);

  const move = useCallback(
    (next: ReferenceStatus) => {
      setStatus(next);
      onStatusChange?.(next);
    },
    [onStatusChange],
  );

  const upload = useCallback(
    async (file: File) => {
      const ticket = ++uploadSeq.current;
      setFileName(file.name);
      setError(null);
      setAttached(null);
      // Any previously attached reference stops applying the moment a
      // new file is chosen, so the form cannot submit the old id while
      // the new one is still uploading.
      onChange(null);
      move("UPLOADING");
      try {
        const asset = await uploadReferenceAudio(file);
        if (ticket !== uploadSeq.current) return;
        setAttached(asset);
        onChange(asset.reference_id);
        move("READY");
      } catch (cause: unknown) {
        if (ticket !== uploadSeq.current) return;
        setAttached(null);
        onChange(null);
        setError(
          cause instanceof ApiError
            ? cause.message
            : "That file could not be uploaded. Check your connection and try again.",
        );
        move("ERROR");
      }
    },
    [move, onChange],
  );

  const pick = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      void upload(file);
    },
    [upload],
  );

  const remove = useCallback(() => {
    // Invalidate any in-flight upload so it cannot attach afterwards.
    uploadSeq.current += 1;
    setAttached(null);
    setFileName(null);
    setError(null);
    onChange(null);
    move("EMPTY");
    if (inputRef.current) inputRef.current.value = "";
  }, [move, onChange]);

  const uploadingOrDone = status === "UPLOADING" || status === "READY";
  const canUpload = !disabled && limits !== null;
  const accept = limits ? limits.supported_formats.map((f) => `.${f}`).join(",") : undefined;

  const statusText =
    status === "UPLOADING"
      ? `Uploading ${fileName ?? "your track"}…`
      : status === "READY" && attached
        ? `${attached.display_name ?? "Reference track"} attached · ${Math.round(
            attached.duration_seconds,
          )}s`
        : status === "ERROR"
          ? (error ?? "That file could not be used.")
          : "No reference track attached.";

  return (
    <section
      aria-labelledby={`${inputId}-heading`}
      className="rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-4"
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3
          id={`${inputId}-heading`}
          className="text-sm font-medium text-[var(--text-primary)]"
        >
          Reference Track
        </h3>
        <span className="text-xs text-[var(--text-muted)]">Optional</span>
      </div>
      <p id={describedById} className="mt-1 text-xs text-[var(--text-secondary)]">
        Upload a track to guide the new song&rsquo;s sound and production character.
      </p>

      {limitsFailed ? (
        <p
          className="mt-3 text-xs text-[var(--danger)]"
          role="status"
          aria-label="Reference track availability"
        >
          Reference track requirements could not be loaded, so uploads are unavailable right
          now. You can still create a song without one.
        </p>
      ) : limits ? (
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          {limits.supported_formats.map((f) => f.toUpperCase()).join(", ")} · up to{" "}
          {formatMinutes(limits.max_duration_seconds)} · max {formatBytes(limits.max_file_bytes)}
        </p>
      ) : (
        <p className="mt-2 text-xs text-[var(--text-muted)]">Loading requirements…</p>
      )}

      {/* The drop zone is a convenience layered over a real file input.
          Drag and drop is never the only way in: the input is a labelled,
          keyboard-reachable control in its own right. */}
      <div
        onDragOver={(event) => {
          if (!canUpload) return;
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          if (!canUpload) return;
          event.preventDefault();
          setDragging(false);
          pick(event.dataTransfer.files);
        }}
        className={`mt-3 rounded-[var(--radius-md)] border border-dashed p-4 transition-colors ${
          dragging
            ? "border-[var(--brand)] bg-[var(--brand-muted)]"
            : "border-[var(--border-strong)]"
        }`}
      >
        <label htmlFor={inputId} className="block text-sm text-[var(--text-secondary)]">
          Choose an audio file, or drag one here
        </label>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={accept}
          disabled={!canUpload || status === "UPLOADING"}
          aria-describedby={`${describedById} ${statusId}`}
          aria-invalid={status === "ERROR" || undefined}
          onChange={(event) => pick(event.target.files)}
          className="mt-2 block w-full max-w-full text-sm text-[var(--text-secondary)] file:mr-3 file:rounded-[var(--radius-sm)] file:border-0 file:bg-[var(--surface-overlay)] file:px-3 file:py-1.5 file:text-sm file:text-[var(--text-primary)] hover:file:bg-[var(--surface-hover)] disabled:opacity-60"
        />
      </div>

      {/* Named: the Create page now has more than one live region, and
          unnamed ones give a screen-reader user two unattributed
          announcements with no way to tell which is which. */}
      <p
        id={statusId}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        aria-label="Reference track status"
        className={`mt-3 break-words text-xs ${
          status === "ERROR"
            ? "text-[var(--danger)]"
            : status === "READY"
              ? "text-[var(--success)]"
              : "text-[var(--text-muted)]"
        }`}
      >
        {/* Indeterminate on purpose: fetch() exposes no upload progress,
            so a percentage here would be an animation, not a measurement. */}
        {statusText}
      </p>

      {(uploadingOrDone || status === "ERROR") && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            onClick={() => inputRef.current?.click()}
            disabled={!canUpload || status === "UPLOADING"}
          >
            Replace
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={remove}
            disabled={disabled}
            aria-label="Remove reference track"
          >
            Remove
          </Button>
        </div>
      )}
    </section>
  );
}
