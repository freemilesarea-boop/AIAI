"use client";

/**
 * The generation form: title, prompt, lyrics, vocal, language, duration,
 * plus the optional Phase 8 advanced controls and structure editor.
 *
 * Semantic form controls with real labels throughout — validation
 * errors are announced via `aria-describedby` and the invalid field
 * receives focus, so keyboard and screen-reader users get the same
 * feedback as sighted mouse users.
 *
 * Two Phase 8 rules are enforced here rather than trusted:
 *
 * - **Advanced controls are optional and default to empty.** A form the
 *   user never touches submits exactly the fields Phase 7 submitted.
 * - **Advisories never gate submission.** They are rendered beside the
 *   lyrics and ignored by `handleSubmit`; only objectively invalid input
 *   (an out-of-range BPM, a missing title) stops a request.
 */

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

import { AdvancedControls } from "@/components/AdvancedControls";
import { AdvisoryList } from "@/components/AdvisoryList";
import { PromptChips } from "@/components/PromptChips";
import { SongPresets } from "@/components/SongPresets";
import { StructureOutline } from "@/components/StructureOutline";
import { Tabs } from "@/components/ui";
import { usePreflight } from "@/hooks/usePreflight";
import type { CreateGenerationInput, VocalGender } from "@/lib/api";
import {
  PRODUCT_DURATIONS,
  SECTION_TAG_PALETTE,
  formatDurationLabel,
  parseBpmInput,
} from "@/lib/songcraft";

const VOCAL_OPTIONS: { value: VocalGender; label: string }[] = [
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
  { value: "instrumental", label: "Instrumental" },
];

const LANGUAGE_OPTIONS = [
  { value: "ko", label: "Korean" },
  { value: "en", label: "English" },
];

/**
 * The durations validated end to end on this deployment.
 *
 * 30/60 came from Phase 3; 120/180/240 were each generated for real
 * against the pinned engine in the Phase 9 long-form gates. The engine
 * accepts up to 600s and the API schema up to 360s — neither is offered,
 * because neither has been validated. See
 * docs/PHASE9_LONG_FORM_ENGINE_AUDIT.md.
 */
const DURATION_OPTIONS = PRODUCT_DURATIONS.map((value) => ({
  value,
  label: formatDurationLabel(value),
}));

export const TITLE_MAX = 200;
export const PROMPT_MAX = 4000;
export const LYRICS_MAX = 20000;

/** Everything "Generate again" carries over from a previous track. */
export interface GenerationFormInitialValues {
  title: string;
  prompt: string;
  lyrics: string;
  vocalGender: VocalGender;
  language: string;
  duration: number;
  bpm: string;
  keyScale: string;
  timeSignature: string;
  /** Opens straight into Custom when a draft carries advanced settings. */
  mode: "simple" | "custom";
}

export interface GenerationFormProps {
  onSubmit: (input: CreateGenerationInput) => void;
  disabled?: boolean;
  busy?: boolean;
  initialValues?: Partial<GenerationFormInitialValues>;
  /** Lineage banner: the track this draft was started from. */
  parent?: { id: string; title: string } | null;
  onClearParent?: () => void;
}

interface FieldErrors {
  title?: string;
  prompt?: string;
  lyrics?: string;
  bpm?: string;
}

export function GenerationForm({
  onSubmit,
  disabled = false,
  busy = false,
  initialValues,
  parent = null,
  onClearParent,
}: GenerationFormProps) {
  const ids = {
    title: useId(),
    prompt: useId(),
    lyrics: useId(),
    vocal: useId(),
    language: useId(),
    duration: useId(),
  };

  const [title, setTitle] = useState(initialValues?.title ?? "");
  const [prompt, setPrompt] = useState(initialValues?.prompt ?? "");
  const [lyrics, setLyrics] = useState(initialValues?.lyrics ?? "");
  const [vocalGender, setVocalGender] = useState<VocalGender>(
    initialValues?.vocalGender ?? "female",
  );
  const [language, setLanguage] = useState(initialValues?.language ?? "ko");
  const [duration, setDuration] = useState(initialValues?.duration ?? 30);
  const [bpm, setBpm] = useState(initialValues?.bpm ?? "");
  const [keyScale, setKeyScale] = useState(initialValues?.keyScale ?? "");
  const [timeSignature, setTimeSignature] = useState(initialValues?.timeSignature ?? "");
  const [errors, setErrors] = useState<FieldErrors>({});
  // Simple is the default: a first-time user must be able to
  // generate without meeting a single advanced control.
  const [mode, setMode] = useState<"simple" | "custom">(
    initialValues?.mode ?? "simple",
  );
  const custom = mode === "custom";

  const titleRef = useRef<HTMLInputElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const lyricsRef = useRef<HTMLTextAreaElement>(null);
  const pendingCaretRef = useRef<number | null>(null);

  const instrumental = vocalGender === "instrumental";

  // Advisories come from the backend so the editor and the stored
  // record can never disagree. Instrumental tracks still get checked:
  // "these lyrics will not be sung" is a useful thing to hear.
  const preflight = usePreflight({
    lyrics,
    duration,
    language,
    instrumental,
  });

  // Restore the caret after a section tag is inserted, so the user can
  // keep typing where the tag left them.
  useLayoutEffect(() => {
    const caret = pendingCaretRef.current;
    if (caret === null) return;
    pendingCaretRef.current = null;
    const element = lyricsRef.current;
    if (!element) return;
    element.focus();
    element.setSelectionRange(caret, caret);
  }, [lyrics]);

  useEffect(() => {
    // Clearing a BPM error as soon as the value becomes valid keeps the
    // message from lingering after the user has fixed it.
    if (!errors.bpm) return;
    if (!parseBpmInput(bpm).error) setErrors((prev) => ({ ...prev, bpm: undefined }));
  }, [bpm, errors.bpm]);

  /** Insert a section tag on its own line at the caret. */
  const insertSectionTag = (tag: string) => {
    if (disabled || instrumental) return;
    const element = lyricsRef.current;
    const start = element?.selectionStart ?? lyrics.length;
    const end = element?.selectionEnd ?? start;

    const before = lyrics.slice(0, start);
    const after = lyrics.slice(end);
    const prefix = before === "" || before.endsWith("\n") ? "" : "\n";
    const suffix = after.startsWith("\n") ? "" : "\n";
    const insertion = `${prefix}${tag}${suffix}`;

    pendingCaretRef.current = (before + insertion).length;
    setLyrics(before + insertion + after);
  };

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

    // The only advanced-control rejection: a value the engine cannot
    // accept. Nothing here rejects a draft for being *unwise*.
    const bpmError = parseBpmInput(bpm).error;
    if (bpmError) next.bpm = bpmError;

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
      // Unset controls are sent as null, never as a substituted default.
      bpm: parseBpmInput(bpm).bpm,
      key_scale: keyScale || null,
      time_signature: timeSignature || null,
      parent_generation_id: parent?.id ?? null,
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
      <div className="flex items-center justify-between gap-3">
        <Tabs
          label="Generation mode"
          value={mode}
          onChange={setMode}
          options={[
            { value: "simple", label: "Simple" },
            { value: "custom", label: "Custom" },
          ]}
        />
        {!custom && (
          <p className="hidden text-xs text-[var(--text-muted)] sm:block">
            Everything else is chosen for you.
          </p>
        )}
      </div>

      {parent && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-violet-900/60 bg-violet-950/20 px-3 py-2">
          <p className="text-sm text-violet-200">
            Based on <span className="font-medium">{parent.title}</span> — adjust anything
            before generating.
          </p>
          {onClearParent && (
            <button
              type="button"
              onClick={onClearParent}
              className="rounded border border-violet-800 px-2 py-1 text-xs font-medium
                text-violet-200 transition-colors hover:bg-violet-900/40
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
            >
              Start fresh
            </button>
          )}
        </div>
      )}

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
        <PromptChips value={prompt} onChange={setPrompt} disabled={disabled} />
      </div>

      {custom && (
      <SongPresets
        lyrics={lyrics}
        disabled={disabled}
        onApplyPreset={(preset, nextLyrics) => {
          // A preset sets the frame only. Lyrics change solely when the
          // preset carries a structure and the user accepted applying it.
          setDuration(preset.duration);
          setVocalGender(preset.instrumental ? "instrumental" : vocalGender);
          if (nextLyrics !== null) setLyrics(nextLyrics);
        }}
        onApplyTemplate={setLyrics}
      />
      )}

      <div>
        <label htmlFor={ids.lyrics} className={labelClass}>
          Lyrics
        </label>
        <p className="mt-1 text-xs text-zinc-500">
          Section tags like [Verse] and [Chorus] are passed through as written. Plain lyrics
          with no tags work too.
        </p>

        {!instrumental && (
          <div className="mt-2 flex flex-wrap gap-1.5" role="group" aria-label="Insert section tag">
            {SECTION_TAG_PALETTE.map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => insertSectionTag(tag)}
                disabled={disabled}
                className="inline-flex min-h-9 items-center rounded-[var(--radius-sm)]
                  border border-[var(--border-default)] px-2.5 font-mono text-xs
                  text-[var(--text-secondary)] transition-colors
                  hover:border-[var(--brand)] hover:text-[var(--brand-text)]
                  disabled:opacity-50"
              >
                {tag}
              </button>
            ))}
          </div>
        )}

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

        <StructureOutline
          sections={preflight.sections}
          preambleLineCount={preflight.preambleLineCount}
          estimatedSyllables={preflight.estimatedSyllables}
        />
        <AdvisoryList advisories={preflight.advisories} checking={preflight.checking} />
      </div>

      <div className={custom ? "grid gap-5 sm:grid-cols-3" : "grid gap-5 sm:grid-cols-1"}>
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

        {custom && (
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
        )}

        {custom && (
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
        )}
      </div>

      {custom && (
      <AdvancedControls
        bpm={bpm}
        keyScale={keyScale}
        timeSignature={timeSignature}
        onBpmChange={setBpm}
        onKeyScaleChange={setKeyScale}
        onTimeSignatureChange={setTimeSignature}
        onClear={() => {
          setBpm("");
          setKeyScale("");
          setTimeSignature("");
        }}
        bpmError={errors.bpm}
        disabled={disabled}
      />
      )}

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
