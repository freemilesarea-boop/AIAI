"use client";

/**
 * The global audio player.
 *
 * One `<audio>` element lives here, mounted by the root layout, so it
 * survives navigation between Create, Library and Projects — moving
 * pages must never interrupt playback, and must never re-fetch audio
 * that is already streaming.
 *
 * Only one track can play at a time, which falls out of there being
 * exactly one element: loading a new track replaces the source rather
 * than adding a second player.
 *
 * The element is deliberately *not* rendered by the visible player bar.
 * If it were, any layout change that unmounted the bar would kill
 * playback.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { getAudioAssetUrl, type Generation } from "@/lib/api";

export interface PlayerTrack {
  id: string;
  title: string;
  /** Preview when available: far smaller than a 24-bit master. */
  src: string;
  downloadUrl: string;
  durationHint: number | null;
}

export interface PlayerState {
  track: PlayerTrack | null;
  playing: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  muted: boolean;
  /** Set when the browser could not play the track. */
  error: string | null;
  play: (track: PlayerTrack) => void;
  toggle: () => void;
  seek: (seconds: number) => void;
  setVolume: (value: number) => void;
  toggleMute: () => void;
  stop: () => void;
}

const PlayerContext = createContext<PlayerState | null>(null);

/** Build a player track from a generation, or `null` if it has no audio. */
export function trackFromGeneration(generation: Generation): PlayerTrack | null {
  if (!generation.audio_assets?.length) return null;
  const hasPreview = generation.audio_assets.some((a) => a.asset_type === "PREVIEW");
  return {
    id: generation.id,
    title: generation.title,
    src: getAudioAssetUrl(generation.id, hasPreview ? "preview" : "master"),
    downloadUrl: getAudioAssetUrl(generation.id, "master", true),
    durationHint: generation.duration_actual,
  };
}

export function PlayerProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [track, setTrack] = useState<PlayerTrack | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(1);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const play = useCallback(
    (next: PlayerTrack) => {
      const element = audioRef.current;
      if (!element) return;
      setError(null);
      // Same track: resume rather than reload, so seeking position and
      // buffered audio survive a second press.
      if (track?.id !== next.id || element.src !== next.src) {
        setTrack(next);
        setCurrentTime(0);
        setDuration(next.durationHint ?? 0);
        element.src = next.src;
      }
      void element.play().catch(() => {
        setPlaying(false);
        setError("This track could not be played.");
      });
    },
    [track],
  );

  const toggle = useCallback(() => {
    const element = audioRef.current;
    if (!element || !track) return;
    if (element.paused) {
      void element.play().catch(() => setError("This track could not be played."));
    } else {
      element.pause();
    }
  }, [track]);

  const seek = useCallback((seconds: number) => {
    const element = audioRef.current;
    if (!element || !Number.isFinite(seconds)) return;
    element.currentTime = seconds;
    setCurrentTime(seconds);
  }, []);

  const setVolume = useCallback((value: number) => {
    const element = audioRef.current;
    const clamped = Math.min(1, Math.max(0, value));
    setVolumeState(clamped);
    if (element) {
      element.volume = clamped;
      // Nudging the slider up is an unmute in every player people use.
      if (clamped > 0 && element.muted) {
        element.muted = false;
        setMuted(false);
      }
    }
  }, []);

  const toggleMute = useCallback(() => {
    const element = audioRef.current;
    if (!element) return;
    element.muted = !element.muted;
    setMuted(element.muted);
  }, []);

  const stop = useCallback(() => {
    const element = audioRef.current;
    if (element) {
      element.pause();
      element.removeAttribute("src");
      element.load();
    }
    setTrack(null);
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, []);

  // Wire element events once. The element itself never unmounts.
  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setCurrentTime(element.currentTime);
    const onMeta = () => {
      if (Number.isFinite(element.duration)) setDuration(element.duration);
    };
    const onEnded = () => {
      setPlaying(false);
      setCurrentTime(0);
    };
    const onError = () => {
      setPlaying(false);
      if (element.currentSrc) setError("This track could not be played.");
    };
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onPause);
    element.addEventListener("timeupdate", onTime);
    element.addEventListener("loadedmetadata", onMeta);
    element.addEventListener("durationchange", onMeta);
    element.addEventListener("ended", onEnded);
    element.addEventListener("error", onError);
    return () => {
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onPause);
      element.removeEventListener("timeupdate", onTime);
      element.removeEventListener("loadedmetadata", onMeta);
      element.removeEventListener("durationchange", onMeta);
      element.removeEventListener("ended", onEnded);
      element.removeEventListener("error", onError);
    };
  }, []);

  const value = useMemo<PlayerState>(
    () => ({
      track, playing, currentTime, duration, volume, muted, error,
      play, toggle, seek, setVolume, toggleMute, stop,
    }),
    [track, playing, currentTime, duration, volume, muted, error,
     play, toggle, seek, setVolume, toggleMute, stop],
  );

  return (
    <PlayerContext.Provider value={value}>
      {/* Generated music has no caption track; the element is labelled
          instead, and the visible controls live in the player bar. */}
      <audio ref={audioRef} preload="metadata" aria-label="LUBER audio player" hidden />
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerState {
  const context = useContext(PlayerContext);
  if (!context) throw new Error("usePlayer must be used inside PlayerProvider");
  return context;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
