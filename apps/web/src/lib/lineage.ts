/**
 * How one generation relates to the one it came from.
 *
 * Three relationships now share a parent link and they are not the same
 * thing. A re-generation reuses only settings; an extension and a
 * replacement are conditioned on the parent's actual recording. Labelling
 * them all "variation" — or any of them "remix" — would describe work the
 * engine did not do.
 */

import type { Generation } from "@/lib/api";

export type RelationKind = "generated-again" | "extended" | "replaced";

export interface Relation {
  kind: RelationKind;
  /** Short label for a lineage list. */
  label: string;
  /** One sentence on what actually happened to the audio. */
  detail: string;
}

/** `m:ss`, matching how the player shows a position. */
export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/**
 * Describe a child's relationship to its parent.
 *
 * Driven by `edit_kind`, which the server sets and clients cannot forge —
 * not by guessing from durations.
 */
export function describeRelation(generation: Generation): Relation | null {
  if (!generation.parent_generation_id) return null;

  const start = generation.edit_start_seconds;
  const end = generation.edit_end_seconds;

  if (generation.edit_kind === "EXTEND") {
    const added = start !== null && end !== null ? Math.round(end - start) : null;
    return {
      kind: "extended",
      label: added === null ? "Extended" : `Extended +${added}s`,
      detail: "The original recording is kept and new music continues from its end.",
    };
  }

  if (generation.edit_kind === "REPLACE_RANGE") {
    const range =
      start !== null && end !== null ? `${formatClock(start)}–${formatClock(end)}` : null;
    return {
      kind: "replaced",
      label: range === null ? "Replaced a section" : `Replaced ${range}`,
      detail: "Only that span was generated again; the rest is the original recording.",
    };
  }

  return {
    kind: "generated-again",
    label: "Generated again",
    detail: "A fresh generation from the same settings. No audio was reused.",
  };
}
