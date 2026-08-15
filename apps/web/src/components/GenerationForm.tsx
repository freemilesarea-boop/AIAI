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
import { ReferenceTrack, type ReferenceStatus } from "@/components/ReferenceTrack";
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
  /** Empty means Random. A value pins the seed. */
  seed: string;
  /** How many songs one press of Generate should produce. */
  resultCount: 1 | 2;
}

/**
 * Two by default.
 *
 * Comparing alternatives is how people actually pick a take, and a
 * single result makes every generation feel like a verdict. Each result
 * is an independent job — this is not a provider batch size.
 */
export const DEFAULT_RESULT_COUNT = 2;

/** Largest seed the API accepts. Mirrors ``SEED_MAX`` in the backend. */
const SEED_MAX = 2 ** 53 - 1;

/** Parse the seed field. Empty is valid and means "engine chooses". */
export function parseSeedInput(raw: string): { seed: number | null; error?: string } {
  const trimmed = raw.trim();
  if (!trimmed) return { seed: null };
  if (!/^\d+$/.test(trimmed)) return { seed: null, error: "A seed must be a whole number." };
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value > SEED_MAX) {
    return { seed: null, error: "That seed is too large." };
  }
  return { seed: value };
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
  seed?: string;
  /** Not a field error in the usual sense: the reference is valid, it is
      simply not finished uploading yet. */
  reference?: string;
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
    seed: useId(),
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
  const [seed, setSeed] = useState(initialValues?.seed ?? "");
  const [seedMode, setSeedMode] = useState<"random" | "fixed">(
    initialValues?.seed ? "fixed" : "random",
  );
  // A reference only exists once the backend has accepted it. The id is
  // the whole state the form needs; the status is kept so a submit can
  // be refused while a file is chosen but not yet uploaded.
  const [referenceId, setReferenceId] = useState<string | null>(null);
  const [referenceStatus, setReferenceStatus] = useState<ReferenceStatus>("EMPTY");

  const [resultCount, setResultCount] = useState<1 | 2>(
    initialValues?.resultCount ?? DEFAULT_RESULT_COUNT,
  );
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

    // The only advanced-control rejections: values the engine cannot
    // accept. Nothing here rejects a draft for being *unwise*.
    const bpmError = parseBpmInput(bpm).error;
    if (bpmError) next.bpm = bpmError;

    if (seedMode === "fixed") {
      const seedError = parseSeedInput(seed).error;
      if (seedError) next.seed = seedError;
    }

    return next;
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (disabled || busy) return;

    // A file chosen but not uploaded has no id and therefore no effect.
    // Submitting here would produce an unreferenced song while the form
    // still showed a reference attached, so it is refused instead.
    if (referenceStatus === "SELECTED" || referenceStatus === "UPLOADING") {
      setErrors({ reference: "Wait for the reference track to finish uploading." });
      return;
    }

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
      // Random is the absence of a seed, not a seed we invent.
      seed: seedMode === "fixed" ? parseSeedInput(seed).seed : null,
      result_count: resultCount,
      // Omitted entirely when nothing is attached, so an ordinary
      // generation sends exactly the request it always did.
      ...(referenceId ? { reference_audio_id: referenceId } : {}),
    });
  };

  const fieldClass =
    "w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2 text-[var(--text-primary)] " +
    "placeholder:text-[var(--text-muted)] focus:border-[var(--brand)] focus:outline-none " +
    "focus-visible:ring-2 focus-visible:ring-[var(--brand)] disabled:opacity-60";

  const labelClass = "block text-sm font-medium text-[var(--text-primary)]";
  const errorClass = "mt-1 text-sm text-[var(--danger)]";

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
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[var(--brand-muted)] bg-[var(--brand-muted)] px-3 py-2">
          <p className="text-sm text-[var(--brand-text)]">
            Based on <span className="font-medium">{parent.title}</span> — adjust anything
            before generating.
          </p>
          {onClearParent && (
            <button
              type="button"
              onClick={onClearParent}
              className="rounded border border-[var(--brand-muted)] px-2 py-1 text-xs font-medium
                text-[var(--brand-text)] transition-colors hover:bg-[var(--brand-muted)]
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)]"
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
        <p className="mt-1 text-xs text-[var(--text-muted)]">
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
          <p className="mt-1 text-sm text-[var(--text-muted)]">
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
        <fieldset className="rounded-[var(--radius-lg)] border border-[var(--border-subtle)] p-4">
          <legend className="px-1 text-sm font-medium text-[var(--text-primary)]">Seed</legend>
          <p className="text-xs text-[var(--text-muted)]">
            The seed is the engine&rsquo;s starting point. Reusing one keeps a generation close
            to a previous take — it does not promise identical audio, and this engine makes no
            such guarantee.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Tabs
              label="Seed mode"
              value={seedMode}
              onChange={setSeedMode}
              options={[
                { value: "random", label: "Random" },
                { value: "fixed", label: "Fixed" },
              ]}
            />
            {seedMode === "fixed" && (
              <div>
                <label htmlFor={ids.seed} className="sr-only">
                  Seed value
                </label>
                <input
                  id={ids.seed}
                  type="text"
                  inputMode="numeric"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="e.g. 12345"
                  disabled={disabled}
                  aria-invalid={Boolean(errors.seed)}
                  aria-describedby={errors.seed ? `${ids.seed}-error` : undefined}
                  className={`w-40 ${fieldClass}`}
                />
              </div>
            )}
          </div>
          {errors.seed && (
            <p id={`${ids.seed}-error`} className={errorClass}>
              {errors.seed}
            </p>
          )}
          {seedMode === "fixed" && resultCount === 2 && (
            <p className="mt-2 text-xs text-[var(--text-muted)]">
              With two songs, the seed applies to the first. The second gets its own — two
              identical seeds would give you the same song twice.
            </p>
          )}
        </fieldset>
      )}

      <ReferenceTrack
        onChange={setReferenceId}
        onStatusChange={setReferenceStatus}
        disabled={disabled || busy}
      />
      {errors.reference && <p className={errorClass}>{errors.reference}</p>}

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

      {/* Result count sits beside Create, because it is a property of
          pressing Create rather than a property of the song. */}
      <div className="mt-1 flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={disabled || busy}
          className="rounded-lg bg-[var(--brand)] px-6 py-3 text-base font-semibold text-white
            transition-colors hover:bg-[var(--brand)] focus-visible:outline-none
            focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-2
            focus-visible:ring-offset-zinc-950 disabled:cursor-not-allowed
            disabled:bg-[var(--brand-muted)] disabled:text-[var(--text-secondary)]"
        >
          {busy ? "Sending…" : "Create"}
        </button>
        <Tabs
          label="Number of songs"
          value={String(resultCount) as "1" | "2"}
          onChange={(value) => setResultCount(value === "2" ? 2 : 1)}
          options={[
            { value: "1", label: "1 Song" },
            { value: "2", label: "2 Songs" },
          ]}
        />
      </div>
      <p className="text-xs text-[var(--text-muted)]">
        {resultCount === 2
          ? "Two independent songs so you can compare. Each takes its own turn on the engine."
          : "One song."}
      </p>
    </form>
  );
}
