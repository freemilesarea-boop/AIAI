"use client";

/**
 * The song's shape, as the backend parser reads it.
 *
 * Read-only on purpose. It reflects the lyrics back so the user can see
 * how their tags were understood — including tags that were *not*
 * understood, which are shown as-is rather than corrected. Untagged
 * lyrics are a legitimate way to write a song, so an empty outline is
 * reported plainly and never nagged about.
 */

import type { SectionSummary } from "@/lib/songcraft";

export interface StructureOutlineProps {
  sections: SectionSummary[];
  preambleLineCount: number;
  estimatedSyllables: number;
}

export function StructureOutline({
  sections,
  preambleLineCount,
  estimatedSyllables,
}: StructureOutlineProps) {
  if (sections.length === 0 && preambleLineCount === 0) return null;

  return (
    <section
      aria-labelledby="structure-outline-heading"
      className="mt-3 rounded-lg border border-[var(--border-default)] bg-[var(--surface-raised)] p-3"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 id="structure-outline-heading" className="text-sm font-medium text-[var(--text-primary)]">
          Structure
        </h3>
        <p className="text-xs text-[var(--text-muted)]">≈{estimatedSyllables} syllables</p>
      </div>

      {sections.length === 0 ? (
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          No section tags — the lyrics are sent as one block. That is fine; tags are optional.
        </p>
      ) : (
        <ol className="mt-2 flex flex-wrap gap-1.5">
          {preambleLineCount > 0 && (
            <li className="rounded border border-dashed border-[var(--border-strong)] px-2 py-1 text-xs text-[var(--text-muted)]">
              {preambleLineCount} line{preambleLineCount === 1 ? "" : "s"} before the first tag
            </li>
          )}
          {sections.map((section) => (
            <li
              key={`${section.line_number}-${section.label}`}
              className={`rounded border px-2 py-1 text-xs ${
                section.recognised
                  ? "border-[var(--border-strong)] text-[var(--text-secondary)]"
                  : "border-[var(--accent-muted)] text-[var(--accent)]"
              }`}
              title={section.recognised ? undefined : "Not a tag LUBER recognises — sent as written"}
            >
              [{section.label}]
              {!section.has_content && <span className="ml-1 text-[var(--text-muted)]">(empty)</span>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
