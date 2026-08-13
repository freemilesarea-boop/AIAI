"use client";

/**
 * The persistent bottom player.
 *
 * Purely a control surface: the `<audio>` element lives in
 * `PlayerProvider` so that nothing here unmounting can stop playback.
 * When no track is loaded the bar collapses entirely rather than
 * sitting there as an empty strip.
 *
 * Two layouts, because a phone cannot fit a usable scrub bar on one
 * line at 390px:
 *
 * - **Mobile**: transport, title and actions on the first row; the seek
 *   bar on the second. A player you cannot scrub is not a player, so
 *   seek stays visible even at the narrowest width.
 * - **Desktop**: a single row with volume added.
 *
 * Volume is intentionally desktop-only — phones have hardware volume
 * keys, and the space is better spent on the scrub bar.
 *
 * Every control is a real labelled button and both sliders are native
 * range inputs, so the player is fully operable from the keyboard.
 */

import { formatTime, usePlayer } from "@/components/player/PlayerProvider";

function PlayIcon({ playing }: { playing: boolean }) {
  return playing ? (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <rect x="6" y="5" width="4" height="14" rx="1" />
      <rect x="14" y="5" width="4" height="14" rx="1" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <path d="M8 5.14v13.72a1 1 0 0 0 1.54.84l10.1-6.86a1 1 0 0 0 0-1.68L9.54 4.3A1 1 0 0 0 8 5.14Z" />
    </svg>
  );
}

export function PlayerBar() {
  const player = usePlayer();
  const { track } = player;

  if (!track) return null;

  const duration = player.duration || track.durationHint || 0;

  /** Download + dismiss. Rendered once per breakpoint, never twice at once. */
  const actions = (
    <>
      <a
        href={track.downloadUrl}
        download
        className="flex h-9 items-center rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
      >
        WAV
      </a>
      <button
        type="button"
        onClick={player.stop}
        aria-label="Close player"
        className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
          <path d="m12 10.6 5-5 1.4 1.4-5 5 5 5-1.4 1.4-5-5-5 5L5.6 17l5-5-5-5L7 5.6l5 5Z" />
        </svg>
      </button>
    </>
  );

  return (
    <div
      role="region"
      aria-label="Player"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-[var(--border-default)] bg-[var(--surface-overlay)]/95 backdrop-blur"
      style={{ height: "var(--player-height)" }}
    >
      <div className="mx-auto flex h-full max-w-[1400px] flex-col justify-center gap-1 px-4 sm:flex-row sm:items-center sm:gap-4 sm:px-6">
        {/* `sm:contents` dissolves this wrapper on desktop so its children
            become direct flex items of the single-row layout. */}
        <div className="flex min-w-0 items-center gap-3 sm:contents">
          <button
            type="button"
            onClick={player.toggle}
            aria-label={player.playing ? `Pause ${track.title}` : `Play ${track.title}`}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--brand)] text-white transition-colors hover:bg-[var(--brand-hover)]"
          >
            <PlayIcon playing={player.playing} />
          </button>

          <div className="min-w-0 flex-1 sm:w-48 sm:flex-none">
            <p className="truncate text-sm font-medium text-[var(--text-primary)]">
              {track.title}
            </p>
            <p className="truncate text-xs text-[var(--text-muted)]">
              {player.error ? (
                <span className="text-[var(--danger)]">{player.error}</span>
              ) : (
                "LUBER"
              )}
            </p>
          </div>

          {/* Mobile keeps the actions on the transport row; a third row
              would not fit inside the bar's height. */}
          <div className="flex shrink-0 items-center gap-1 sm:hidden">{actions}</div>
        </div>

        <div className="flex flex-1 items-center gap-2 sm:gap-3">
          <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-[var(--text-muted)] sm:w-10 sm:text-xs">
            {formatTime(player.currentTime)}
          </span>
          <input
            type="range"
            className="luber-range h-4 flex-1"
            min={0}
            max={Math.max(duration, 0.1)}
            step={0.1}
            value={Math.min(player.currentTime, duration || 0)}
            onChange={(e) => player.seek(Number(e.target.value))}
            aria-label="Seek"
            aria-valuetext={`${formatTime(player.currentTime)} of ${formatTime(duration)}`}
          />
          <span className="w-9 shrink-0 font-mono text-[11px] tabular-nums text-[var(--text-muted)] sm:w-10 sm:text-xs">
            {formatTime(duration)}
          </span>
        </div>

        <div className="hidden shrink-0 items-center gap-2 sm:flex">
          <div className="hidden items-center gap-2 md:flex">
            <button
              type="button"
              onClick={player.toggleMute}
              aria-label={player.muted ? "Unmute" : "Mute"}
              className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-secondary)] transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                {player.muted || player.volume === 0 ? (
                  <path d="M4 9v6h4l5 4V5L8 9H4Zm12.5 3 2.5 2.5-1 1L15.5 13 13 15.5l-1-1L14.5 12 12 9.5l1-1L15.5 11 18 8.5l1 1L16.5 12Z" />
                ) : (
                  <path d="M4 9v6h4l5 4V5L8 9H4Zm12 3a4 4 0 0 0-2-3.46v6.92A4 4 0 0 0 16 12Z" />
                )}
              </svg>
            </button>
            <input
              type="range"
              className="luber-range h-4 w-20"
              min={0}
              max={1}
              step={0.01}
              value={player.muted ? 0 : player.volume}
              onChange={(e) => player.setVolume(Number(e.target.value))}
              aria-label="Volume"
            />
          </div>
          {actions}
        </div>
      </div>
    </div>
  );
}
