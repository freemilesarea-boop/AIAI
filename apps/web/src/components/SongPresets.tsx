"use client";

/**
 * Song presets and structure templates.
 *
 * A preset sets the *frame* — a duration and a section skeleton — and
 * leaves the writing to the user. It never supplies a prompt or lyrics.
 *
 * The rule this component exists to enforce: **applying a template
 * never silently destroys writing.** If the lyric sheet already has
 * words in it, the template is appended and the user is told; replacing
 * requires a second, explicit confirmation. A bare skeleton (tags only)
 * is swapped without ceremony, because nothing is lost.
 *
 * Templates are conditioning aids, not controls. ACE-Step reads section
 * tags as part of the lyric text; a template makes a recognisable
 * arrangement more likely and enforces nothing. The copy says so.
 */

import { useState } from "react";

import {
  SONG_PRESETS,
  STRUCTURE_TEMPLATES,
  findTemplate,
  formatDurationLabel,
  lyricsHaveContent,
  templateText,
  type SongPreset,
  type StructureTemplate,
} from "@/lib/songcraft";

export interface SongPresetsProps {
  lyrics: string;
  disabled?: boolean;
  /** Apply a preset's frame: duration, instrumental flag, structure. */
  onApplyPreset: (preset: SongPreset, lyrics: string | null) => void;
  /** Apply just a structure template to the lyric sheet. */
  onApplyTemplate: (lyrics: string) => void;
}

type Pending =
  | { kind: "preset"; preset: SongPreset; template: StructureTemplate }
  | { kind: "template"; template: StructureTemplate };

export function SongPresets({
  lyrics,
  disabled = false,
  onApplyPreset,
  onApplyTemplate,
}: SongPresetsProps) {
  const [pending, setPending] = useState<Pending | null>(null);

  const hasWriting = lyricsHaveContent(lyrics);

  const applyTemplateText = (template: StructureTemplate, replace: boolean): string => {
    if (!lyrics.trim() || !hasWriting || replace) return templateText(template);
    const separator = lyrics.endsWith("\n") ? "" : "\n";
    return `${lyrics}${separator}\n${templateText(template)}`;
  };

  const choosePreset = (preset: SongPreset) => {
    const template = findTemplate(preset.templateId);
    if (!template) {
      onApplyPreset(preset, null); // Instrumental: frame only, no structure.
      return;
    }
    if (hasWriting) {
      setPending({ kind: "preset", preset, template });
      return;
    }
    onApplyPreset(preset, applyTemplateText(template, false));
  };

  const chooseTemplate = (template: StructureTemplate) => {
    if (hasWriting) {
      setPending({ kind: "template", template });
      return;
    }
    onApplyTemplate(applyTemplateText(template, false));
  };

  const resolve = (replace: boolean) => {
    if (!pending) return;
    const text = applyTemplateText(pending.template, replace);
    if (pending.kind === "preset") onApplyPreset(pending.preset, text);
    else onApplyTemplate(text);
    setPending(null);
  };

  const chipClass =
    "inline-flex min-h-9 items-center rounded-lg border border-zinc-700 px-3 " +
    "text-xs font-medium text-zinc-300 " +
    "transition-colors hover:border-violet-600 hover:text-violet-200 " +
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 " +
    "disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <details className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3">
      <summary className="cursor-pointer select-none text-sm font-medium text-zinc-200 marker:text-zinc-600">
        Song presets <span className="font-normal text-zinc-500">— optional</span>
      </summary>

      <p className="mt-2 text-xs text-zinc-500">
        A preset sets the length and drops in a section skeleton. It never writes lyrics for
        you, and section tags guide the model rather than forcing it.
      </p>

      <div className="mt-3">
        <h4 className="text-xs font-medium uppercase tracking-wide text-zinc-500">Presets</h4>
        <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Song presets">
          {SONG_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              disabled={disabled}
              onClick={() => choosePreset(preset)}
              title={preset.description}
              className={chipClass}
            >
              {preset.name}
              <span className="ml-1.5 text-zinc-500">
                {formatDurationLabel(preset.duration)}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4">
        <h4 className="text-xs font-medium uppercase tracking-wide text-zinc-500">
          Structure only
        </h4>
        <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Structure templates">
          {STRUCTURE_TEMPLATES.map((template) => (
            <button
              key={template.id}
              type="button"
              disabled={disabled}
              onClick={() => chooseTemplate(template)}
              title={template.description}
              className={chipClass}
            >
              {template.name}
              <span className="ml-1.5 text-zinc-500">{template.sections.length}</span>
            </button>
          ))}
        </div>
      </div>

      {pending && (
        <div
          role="alertdialog"
          aria-label="Apply structure to existing lyrics"
          className="mt-4 rounded-lg border border-amber-900/60 bg-amber-950/20 p-3"
        >
          <p className="text-sm text-amber-100">
            You already have lyrics. Add the {pending.template.name} structure after them, or
            replace what you have written?
          </p>
          <div className="mt-2.5 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => resolve(false)}
              className="inline-flex min-h-9 items-center rounded-lg bg-violet-600 px-3 text-xs font-semibold text-white
                transition-colors hover:bg-violet-500 focus-visible:outline-none
                focus-visible:ring-2 focus-visible:ring-violet-400"
            >
              Add after my lyrics
            </button>
            <button
              type="button"
              onClick={() => resolve(true)}
              className="inline-flex min-h-9 items-center rounded-lg border border-amber-700 px-3 text-xs font-semibold
                text-amber-200 transition-colors hover:bg-amber-900/40
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
            >
              Replace my lyrics
            </button>
            <button
              type="button"
              onClick={() => setPending(null)}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium
                text-zinc-300 transition-colors hover:bg-zinc-800
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </details>
  );
}
