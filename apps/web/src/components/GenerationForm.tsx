"use client";

/**
 * The generation form: title, prompt, lyrics, vocal, language, duration.
 *
 * Semantic form controls with real labels throughout — validation
 * errors are announced via `aria-describedby` and the invalid field
 * receives focus, so keyboard and screen-reader users get the same
 * feedback as sighted mouse users.
 */

import { useId, useRef, useState } from "react";

import type { CreateGenerationInput, VocalGender } from "@/lib/api";

const VOCAL_OPTIONS: { value: VocalGender; label: string }[] = [
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
  { value: "instrumental", label: "Instrumental" },
];

const LANGUAGE_OPTIONS = [
  { value: "ko", label: "Korean" },
  { value: "en", label: "English" },
];

/** Phase 3 exposes conservative presets, not the upstream 600s ceiling. */
const DURATION_OPTIONS = [
  { value: 30, label: "30 seconds" },
  { value: 60, label: "60 seconds" },
];

export const TITLE_MAX = 200;
export const PROMPT_MAX = 4000;
export const LYRICS_MAX = 20000;

export interface GenerationFormProps {
  onSubmit: (input: CreateGenerationInput) => void;
  disabled?: boolean;
  busy?: boolean;
}

interface FieldErrors {
  title?: string;
  prompt?: string;
  lyrics?: string;
}

export function GenerationForm({ onSubmit, disabled = false, busy = false }: GenerationFormProps) {
  const ids = {
    title: useId(),
    prompt: useId(),
    lyrics: useId(),
    vocal: useId(),
    language: useId(),
    duration: useId(),
  };

  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [vocalGender, setVocalGender] = useState<VocalGender>("female");
  const [language, setLanguage] = useState("ko");
  const [duration, setDuration] = useState(30);
  const [errors, setErrors] = useState<FieldErrors>({});

  const titleRef = useRef<HTMLInputElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const lyricsRef = useRef<HTMLTextAreaElement>(null);

  const instrumental = vocalGender === "instrumental";

  const validate = (): FieldErrors => {
    const next: FieldErrors = {};
    if (!title.trim()) next.title = "Add a title for your track.";
    else if (title.length > TITLE_MAX) next.title = `Title must be ${TITLE_MAX} characters or fewer.`;

    if (!prompt.trim()) next.prompt = "Describe the music you want.";
    else if (prompt.length > PROMPT_MAX)
      next.prompt = `Description must be ${PROMPT_MAX} characters or fewer.`;

    if (lyrics.length > LYRICS_MAX)
      next.lyrics = `Lyrics must be ${LYRICS_MAX} characters or fewer.`;
    if (!instrumental && !lyrics.trim())
      next.lyrics = "Add lyrics, or switch the vocal to Instrumental.";

    return next;
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (disabled || busy) return;

    const found = validate();
    setErrors(found);
    if (Object.keys(found).length > 0) {
      if (found.title) titleRef.current?.focus();
      else if (found.prompt) promptRef.current?.focus();
      else if (found.lyrics) lyricsRef.current?.focus();
      return;
    }

    onSubmit({
      title: title.trim(),
      prompt: prompt.trim(),
      // Lyrics keep their line breaks and section tags verbatim.
      lyrics: instrumental ? "" : lyrics,
      vocal_gender: vocalGender,
      language,
      duration,
    });
  };

  const fieldClass =
    "w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-zinc-100 " +
    "placeholder:text-zinc-500 focus:border-violet-500 focus:outline-none " +
    "focus-visible:ring-2 focus-visible:ring-violet-500 disabled:opacity-60";

  const labelClass = "block text-sm font-medium text-zinc-200";
  const errorClass = "mt-1 text-sm text-red-400";

  return (
    <form onSubmit={handleSubmit} noValidate className="flex flex-col gap-5">
      <div>
        <label htmlFor={ids.title} className={labelClass}>
          Title
        </label>
        <input
          id={ids.title}
          ref={titleRef}
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Midnight Window"
          maxLength={TITLE_MAX}
          disabled={disabled}
          aria-invalid={Boolean(errors.title)}
          aria-describedby={errors.title ? `${ids.title}-error` : undefined}
          className={`mt-1.5 ${fieldClass}`}
        />
        {errors.title && (
          <p id={`${ids.title}-error`} className={errorClass}>
            {errors.title}
          </p>
        )}
      </div>

      <div>
        <label htmlFor={ids.prompt} className={labelClass}>
          Music description
        </label>
        <textarea
          id={ids.prompt}
          ref={promptRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Dreamy Korean indie pop with warm electric piano, soft drums and emotional female lead vocal"
          rows={4}
          maxLength={PROMPT_MAX}
          disabled={disabled}
          aria-invalid={Boolean(errors.prompt)}
          aria-describedby={errors.prompt ? `${ids.prompt}-error` : undefined}
          className={`mt-1.5 resize-y ${fieldClass}`}
        />
        {errors.prompt && (
          <p id={`${ids.prompt}-error`} className={errorClass}>
            {errors.prompt}
          </p>
        )}
      </div>

      <div>
        <label htmlFor={ids.lyrics} className={labelClass}>
          Lyrics
        </label>
        <p className="mt-1 text-xs text-zinc-500">
          Section tags like [Verse] and [Chorus] are passed through as written.
        </p>
        <textarea
          id={ids.lyrics}
          ref={lyricsRef}
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder={"[Verse]\n오늘 밤 너를 생각해\n조용한 창가에 앉아"}
          rows={8}
          maxLength={LYRICS_MAX}
          disabled={disabled || instrumental}
          aria-invalid={Boolean(errors.lyrics)}
          aria-describedby={errors.lyrics ? `${ids.lyrics}-error` : undefined}
          className={`mt-1.5 resize-y font-mono text-sm leading-relaxed ${fieldClass}`}
        />
        {instrumental && (
          <p className="mt-1 text-sm text-zinc-500">
            Instrumental selected — lyrics are not used.
          </p>
        )}
        {errors.lyrics && (
          <p id={`${ids.lyrics}-error`} className={errorClass}>
            {errors.lyrics}
          </p>
        )}
      </div>

      <div className="grid gap-5 sm:grid-cols-3">
        <div>
          <label htmlFor={ids.vocal} className={labelClass}>
            Vocal
          </label>
          <select
            id={ids.vocal}
            value={vocalGender}
            onChange={(e) => setVocalGender(e.target.value as VocalGender)}
            disabled={disabled}
            className={`mt-1.5 ${fieldClass}`}
          >
            {VOCAL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor={ids.language} className={labelClass}>
            Language
          </label>
          <select
            id={ids.language}
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={disabled}
            className={`mt-1.5 ${fieldClass}`}
          >
            {LANGUAGE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor={ids.duration} className={labelClass}>
            Duration
          </label>
          <select
            id={ids.duration}
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            disabled={disabled}
            className={`mt-1.5 ${fieldClass}`}
          >
            {DURATION_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled || busy}
        className="mt-1 rounded-lg bg-violet-600 px-6 py-3 text-base font-semibold text-white
          transition-colors hover:bg-violet-500 focus-visible:outline-none
          focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2
          focus-visible:ring-offset-zinc-950 disabled:cursor-not-allowed
          disabled:bg-violet-900 disabled:text-zinc-400"
      >
        {busy ? "Generating…" : "Generate"}
      </button>
    </form>
  );
}
