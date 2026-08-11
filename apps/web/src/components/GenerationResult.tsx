"use client";

/**
 * Completed-track result card: listen, then download.
 *
 * The player and the download both point at the LUBER API's audio
 * endpoint, addressed by generation id. The browser never receives a
 * storage key or a filesystem path, and never talks to the model
 * runtime. A native <audio controls> element supplies play/pause and
 * scrubbing — no custom waveform in this phase.
 */

import {
  findMasterAsset,
  getMasterAudioUrl,
  type Generation,
} from "@/lib/api";
import { formatDuration } from "@/lib/generationStatus";

export interface GenerationResultProps {
  generation: Generation;
  onCreateAnother?: () => void;
  /** Show provider/model details — useful locally, off by default. */
  showTechnicalDetails?: boolean;
}

export function GenerationResult({
  generation,
  onCreateAnother,
  showTechnicalDetails = false,
}: GenerationResultProps) {
  const master = findMasterAsset(generation);
  const audioUrl = getMasterAudioUrl(generation.id);
  const downloadUrl = getMasterAudioUrl(generation.id, true);

  return (
    <section
      aria-labelledby="generation-result-heading"
      className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-6"
    >
      <p className="text-xs font-medium uppercase tracking-wide text-violet-400">Track ready</p>
      <h2
        id="generation-result-heading"
        className="mt-1.5 text-2xl font-bold tracking-tight text-zinc-50"
      >
        {generation.title}
      </h2>

      <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-zinc-400">
        <div className="flex gap-1.5">
          <dt>Duration</dt>
          <dd className="font-mono tabular-nums text-zinc-300">
            {formatDuration(generation.duration_actual ?? master?.duration ?? null)}
          </dd>
        </div>
        {master && (
          <div className="flex gap-1.5">
            <dt>Audio</dt>
            <dd className="text-zinc-300">
              {master.sample_rate / 1000} kHz · {master.channels === 2 ? "Stereo" : "Mono"}
            </dd>
          </div>
        )}
      </dl>

      {master ? (
        <>
          {/* Generated music has no captions track; the player is labelled instead. */}
          <audio
            controls
            preload="metadata"
            src={audioUrl}
            aria-label={`Audio player for ${generation.title}`}
            className="mt-5 w-full"
          >
            Your browser does not support audio playback.
          </audio>

          <div className="mt-5 flex flex-wrap gap-3">
            <a
              href={downloadUrl}
              download
              className="rounded-lg bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white
                transition-colors hover:bg-violet-500 focus-visible:outline-none
                focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2
                focus-visible:ring-offset-zinc-950"
            >
              Download WAV
            </a>
            {onCreateAnother && (
              <button
                type="button"
                onClick={onCreateAnother}
                className="rounded-lg bg-zinc-800 px-5 py-2.5 text-sm font-semibold text-zinc-100
                  transition-colors hover:bg-zinc-700 focus-visible:outline-none
                  focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2
                  focus-visible:ring-offset-zinc-950"
              >
                Create another
              </button>
            )}
          </div>
        </>
      ) : (
        <p className="mt-5 text-sm text-zinc-400">
          This track finished but its audio is not available.
        </p>
      )}

      {showTechnicalDetails && (
        <dl className="mt-6 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 border-t border-zinc-800 pt-4 text-xs text-zinc-500">
          <dt>Provider</dt>
          <dd className="font-mono text-zinc-400">{generation.provider ?? "—"}</dd>
          <dt>Model</dt>
          <dd className="font-mono text-zinc-400">
            {generation.model_name ?? "—"}
            {generation.model_version ? ` (${generation.model_version})` : ""}
          </dd>
          {master && (
            <>
              <dt>SHA-256</dt>
              <dd className="break-all font-mono text-zinc-400">{master.sha256}</dd>
            </>
          )}
        </dl>
      )}
    </section>
  );
}
