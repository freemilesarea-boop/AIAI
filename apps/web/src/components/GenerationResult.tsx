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
  findPreviewAsset,
  getAudioAssetUrl,
  type Generation,
} from "@/lib/api";
import { formatDuration } from "@/lib/generationStatus";

export interface GenerationResultProps {
  generation: Generation;
  onCreateAnother?: () => void;
  /**
   * Start a new draft from this track, keeping its settings and
   * recording it as the new generation's parent.
   */
  onGenerateAgain?: (generation: Generation) => void;
  /** Show provider/model details — useful locally, off by default. */
  showTechnicalDetails?: boolean;
}

export function GenerationResult({
  generation,
  onCreateAnother,
  onGenerateAgain,
  showTechnicalDetails = false,
}: GenerationResultProps) {
  const master = findMasterAsset(generation);
  const preview = findPreviewAsset(generation);
  // Play the compressed preview: far smaller than a 24-bit master, so
  // playback starts quickly. The master stays the download deliverable.
  const playbackAsset = preview ? "preview" : "master";
  const audioUrl = getAudioAssetUrl(generation.id, playbackAsset);
  const wavDownloadUrl = getAudioAssetUrl(generation.id, "master", true);
  const mp3DownloadUrl = getAudioAssetUrl(generation.id, "preview", true);

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
            <dt>Master</dt>
            <dd className="text-zinc-300">
              WAV · {master.sample_rate / 1000} kHz ·{" "}
              {master.bit_depth ? `${master.bit_depth}-bit · ` : ""}
              {master.channels === 2 ? "Stereo" : "Mono"}
            </dd>
          </div>
        )}
        {preview && (
          <div className="flex gap-1.5">
            <dt>Preview</dt>
            <dd className="text-zinc-300">
              MP3{preview.bitrate ? ` · ${Math.round(preview.bitrate / 1000)} kbps` : ""}
            </dd>
          </div>
        )}
      </dl>

      {(generation.bpm !== null ||
        generation.key_scale !== null ||
        generation.time_signature !== null ||
        generation.parent_generation_id !== null) && (
        <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-zinc-400">
          {generation.bpm !== null && (
            <div className="flex gap-1.5">
              <dt>BPM</dt>
              <dd className="font-mono tabular-nums text-zinc-300">{generation.bpm}</dd>
            </div>
          )}
          {generation.key_scale !== null && (
            <div className="flex gap-1.5">
              <dt>Key</dt>
              <dd className="text-zinc-300">{generation.key_scale}</dd>
            </div>
          )}
          {generation.time_signature !== null && (
            <div className="flex gap-1.5">
              <dt>Time</dt>
              <dd className="font-mono tabular-nums text-zinc-300">
                {generation.time_signature}
              </dd>
            </div>
          )}
          {generation.parent_generation_id !== null && (
            <div className="flex gap-1.5">
              <dt>Lineage</dt>
              <dd className="text-zinc-300">
                Generated again from an earlier take
                {generation.variation_label ? ` · ${generation.variation_label}` : ""}
              </dd>
            </div>
          )}
        </dl>
      )}

      {master ? (
        <>
          {/* Generated music has no captions track; the player is labelled instead. */}
          <audio
            controls
            preload="metadata"
            src={audioUrl}
            aria-label={`${preview ? "Preview" : "Audio"} player for ${generation.title}`}
            className="mt-5 w-full"
          >
            Your browser does not support audio playback.
          </audio>

          <div className="mt-5 flex flex-wrap gap-3">
            <a
              href={wavDownloadUrl}
              download
              className="rounded-lg bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white
                transition-colors hover:bg-violet-500 focus-visible:outline-none
                focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2
                focus-visible:ring-offset-zinc-950"
            >
              Download WAV
            </a>
            {preview && (
              <a
                href={mp3DownloadUrl}
                download
                className="rounded-lg bg-zinc-800 px-5 py-2.5 text-sm font-semibold text-zinc-100
                  transition-colors hover:bg-zinc-700 focus-visible:outline-none
                  focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2
                  focus-visible:ring-offset-zinc-950"
              >
                Download MP3
              </a>
            )}
            {onGenerateAgain && (
              <button
                type="button"
                onClick={() => onGenerateAgain(generation)}
                className="rounded-lg bg-zinc-800 px-5 py-2.5 text-sm font-semibold text-zinc-100
                  transition-colors hover:bg-zinc-700 focus-visible:outline-none
                  focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2
                  focus-visible:ring-offset-zinc-950"
              >
                Generate again
              </button>
            )}
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

      {showTechnicalDetails && generation.request_trace && (
        <details className="mt-4 border-t border-zinc-800 pt-4">
          <summary className="cursor-pointer select-none text-xs font-medium text-zinc-400">
            Request trace
          </summary>
          <p className="mt-1.5 text-xs text-zinc-500">
            What LUBER actually sent to the generation provider. Credentials, hostnames and
            local paths are never recorded.
          </p>
          <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-400">
            {JSON.stringify(generation.request_trace, null, 2)}
          </pre>
        </details>
      )}
    </section>
  );
}
