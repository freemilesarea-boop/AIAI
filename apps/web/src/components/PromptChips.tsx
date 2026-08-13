"use client";

/**
 * Genre and mood suggestions for the song brief.
 *
 * These are **text**, not hidden parameters. Tapping a chip appends a
 * word to the description the user can then edit, delete, or ignore —
 * there is no secret provider field behind any of them, and what the
 * user sees in the box is exactly what gets sent.
 *
 * Tapping a chip that is already present removes it again, so the chips
 * read as toggles rather than a one-way append that quietly duplicates.
 */

import { Chip } from "@/components/ui";

const GENRES = [
  "Pop", "R&B", "Ballad", "Rock", "Hip-Hop", "Electronic", "Lo-fi", "Jazz",
];

const MOODS = [
  "Warm", "Dreamy", "Energetic", "Dark", "Romantic", "Melancholic", "Bright",
];

export interface PromptChipsProps {
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
}

/** Whether *term* already appears in the brief as a whole word. */
export function hasTerm(prompt: string, term: string): boolean {
  return new RegExp(`(^|[\\s,])${term}([\\s,]|$)`, "i").test(prompt.trim());
}

export function toggleTerm(prompt: string, term: string): string {
  const trimmed = prompt.trim();
  if (!hasTerm(trimmed, term)) {
    return trimmed ? `${trimmed.replace(/,\s*$/, "")}, ${term}` : term;
  }
  const stripped = trimmed
    .replace(new RegExp(`(^|,\\s*)${term}(?=([\\s,]|$))`, "i"), "")
    .replace(/^\s*,\s*/, "")
    .replace(/,\s*,/g, ",")
    .replace(/,\s*$/, "")
    .trim();
  return stripped;
}

export function PromptChips({ value, onChange, disabled = false }: PromptChipsProps) {
  const row = (label: string, terms: string[]) => (
    <div className="mt-2">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
        {label}
      </span>
      <div className="mt-1.5 flex flex-wrap gap-1.5" role="group" aria-label={label}>
        {terms.map((term) => (
          <Chip
            key={term}
            selected={hasTerm(value, term)}
            disabled={disabled}
            onClick={() => onChange(toggleTerm(value, term))}
          >
            {term}
          </Chip>
        ))}
      </div>
    </div>
  );

  return (
    <div className="mt-3">
      {row("Genre", GENRES)}
      {row("Mood", MOODS)}
    </div>
  );
}
