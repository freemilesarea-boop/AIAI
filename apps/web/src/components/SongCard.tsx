"use client";

/**
 * A generated track, as the product presents it.
 *
 * Deliberately free of diagnostics: no seed, no provider, no model
 * version, no request trace. Those exist and matter, but they belong
 * behind Advanced on the detail page, not on the object a listener
 * scans through.
 *
 * The artwork is a deterministic gradient derived from the generation
 * id, so a track looks the same every time you see it without the
 * product pretending to have generated cover art.
 */

import Link from "next/link";

import { usePlayer, trackFromGeneration } from "@/components/player/PlayerProvider";
import { Card, StatusPill, cx } from "@/components/ui";
import { getAudioAssetUrl, type Generation } from "@/lib/api";

/** Stable hue pair from the id — same track, same colours, always. */
function artwork(id: string): { from: string; to: string } {
  let hash = 0;
  for (let i = 0; i < id.length; i += 1) hash = (hash * 31 + id.charCodeAt(i)) % 360;
  return { from: `hsl(${hash} 62% 42%)`, to: `hsl(${(hash + 48) % 360} 58% 22%)` };
}

function formatClock(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatWhen(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const minutes = Math.floor((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(iso).toLocaleDateString();
}

export interface SongCardProps {
  generation: Generation;
  onGenerateAgain?: (generation: Generation) => void;
  /** Rendered in the overflow area — used for project assignment. */
  extraActions?: React.ReactNode;
}

export function SongCard({ generation, onGenerateAgain, extraActions }: SongCardProps) {
  const player = usePlayer();
  const colours = artwork(generation.id);
  const track = trackFromGeneration(generation);
  const isCurrent = player.track?.id === generation.id;
  const ready = generation.status === "COMPLETED" && track !== null;
  const duration = generation.duration_actual ?? generation.duration_requested;

  return (
    <Card className="group flex gap-4 p-3 transition-colors hover:border-[var(--border-default)]">
      <div className="relative shrink-0">
        <div
          aria-hidden="true"
          className="h-[72px] w-[72px] rounded-[var(--radius-md)]"
          style={{ background: `linear-gradient(135deg, ${colours.from}, ${colours.to})` }}
        />
        {ready && (
          <button
            type="button"
            onClick={() => {
              if (!track) return;
              // One handler, one meaning: pressing the artwork of the
              // track that is already loaded toggles it; anything else
              // starts it. Two handlers here previously fought and
              // produced a play-then-pause on the same click.
              if (isCurrent) player.toggle();
              else player.play(track);
            }}
            aria-label={
              isCurrent && player.playing ? `Pause ${generation.title}` : `Play ${generation.title}`
            }
            className={cx(
              "absolute inset-0 flex items-center justify-center rounded-[var(--radius-md)]",
              "bg-black/45 text-white transition-opacity",
              isCurrent && player.playing
                ? "opacity-100"
                : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
            )}
          >
            <svg viewBox="0 0 24 24" className="h-7 w-7" fill="currentColor" aria-hidden="true">
              {isCurrent && player.playing ? (
                <>
                  <rect x="6" y="5" width="4" height="14" rx="1" />
                  <rect x="14" y="5" width="4" height="14" rx="1" />
                </>
              ) : (
                <path d="M8 5.14v13.72a1 1 0 0 0 1.54.84l10.1-6.86a1 1 0 0 0 0-1.68L9.54 4.3A1 1 0 0 0 8 5.14Z" />
              )}
            </svg>
          </button>
        )}
      </div>

      <div className="flex min-w-0 flex-1 flex-col justify-between py-0.5">
        <div className="min-w-0">
          <div className="flex items-start justify-between gap-3">
            <Link
              href={`/song/${generation.id}`}
              className="-my-1 inline-flex min-h-8 min-w-0 items-center truncate py-1 text-sm
                font-semibold text-[var(--text-primary)] hover:underline"
            >
              {generation.title}
            </Link>
            {generation.status !== "COMPLETED" && <StatusPill status={generation.status} />}
          </div>
          <p className="mt-0.5 truncate text-xs text-[var(--text-secondary)]">
            {generation.prompt}
          </p>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-[var(--text-muted)]">
          <span className="font-mono tabular-nums">{formatClock(duration)}</span>
          <span>{formatWhen(generation.created_at)}</span>
          {generation.parent_generation_id && <span>· from an earlier take</span>}
          <span className="ml-auto flex items-center gap-1">
            {ready && (
              <a
                href={getAudioAssetUrl(generation.id, "master", true)}
                download
                className="inline-flex h-8 items-center rounded-[var(--radius-sm)] px-2 transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
              >
                WAV
              </a>
            )}
            {onGenerateAgain && generation.status === "COMPLETED" && (
              <button
                type="button"
                onClick={() => onGenerateAgain(generation)}
                className="inline-flex h-8 items-center rounded-[var(--radius-sm)] px-2 transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
              >
                Generate again
              </button>
            )}
            {extraActions}
          </span>
        </div>
      </div>
    </Card>
  );
}
