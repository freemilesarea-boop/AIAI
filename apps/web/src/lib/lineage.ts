/**
 * How one generation relates to the one it came from.
 *
 * Three relationships now share a parent link and they are not the same
 * thing. A re-generation reuses only settings; an extension and a
 * replacement are conditioned on the parent's actual recording. Labelling
 * them all "variation" — or any of them "remix" — would describe work the
 * engine did not do.
 */

import type { Generation, LineageNode, LineageOperation } from "@/lib/api";

export type RelationKind = "generated-again" | "extended" | "replaced" | "cover";

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

  if (generation.edit_kind === "COVER") {
    return {
      kind: "cover",
      label: "Cover",
      // No claim that the recording or the voice survives — calibration
      // showed neither does.
      detail:
        "A new performance in a different style, guided by this song's musical structure.",
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


/**
 * The one place an operation becomes words.
 *
 * Every surface reads from here so two screens cannot disagree about the
 * same row, and so the stored ``REPLACE_RANGE`` never escapes: the
 * classifier already renamed it, and this only ever sees
 * ``REPLACE_SECTION``.
 */
export function operationLabel(node: {
  operation: LineageOperation;
  edit_start_seconds: number | null;
  edit_end_seconds: number | null;
}): string {
  switch (node.operation) {
    case "ORIGINAL":
      return "Original";
    case "GENERATE_AGAIN":
      return "Generated again";
    case "EXTEND": {
      const { edit_start_seconds: from, edit_end_seconds: to } = node;
      // How much was added is the only thing that distinguishes two
      // extensions of the same song, so it stays in the label.
      const added = from !== null && to !== null ? Math.round(to - from) : null;
      return added !== null && added > 0 ? `Extended +${added}s` : "Extended";
    }
    case "COVER":
      return "Cover";
    case "REPLACE_SECTION": {
      const { edit_start_seconds: start, edit_end_seconds: end } = node;
      // The span is what makes this label useful; without it the user
      // cannot tell two replacements of the same song apart.
      return start !== null && end !== null
        ? `Replaced ${formatClock(start)}–${formatClock(end)}`
        : "Replaced a section";
    }
    default:
      // An operation this build does not know about. Saying nothing
      // specific beats guessing, and the node still renders.
      return "Derived version";
  }
}

/**
 * How the current version relates to the one it came from.
 *
 * Phrased as an origin statement rather than a similarity claim: the
 * engine's own calibration never established that a cover sounds like
 * its source, so the copy says where it came from and stops.
 */
export function derivedContext(
  operation: LineageOperation,
  parentTitle: string,
): string | null {
  switch (operation) {
    case "GENERATE_AGAIN":
      return `Generated again from “${parentTitle}”`;
    case "EXTEND":
      return `Extended from “${parentTitle}”`;
    case "REPLACE_SECTION":
      return `Replaced section from “${parentTitle}”`;
    case "COVER":
      return `Cover of “${parentTitle}”`;
    default:
      return null;
  }
}

/** Depth of each node from the root, for indentation. Bounded by construction. */
export function lineageDepths(nodes: LineageNode[]): Map<string, number> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const depths = new Map<string, number>();
  for (const node of nodes) {
    let depth = 0;
    let cursor: LineageNode | undefined = node;
    const seen = new Set<string>();
    // Guarded: lineage data can be imperfect, and an indentation helper
    // must not be the thing that hangs the page.
    while (cursor?.parent_generation_id && !seen.has(cursor.id) && depth < 24) {
      seen.add(cursor.id);
      const parent: LineageNode | undefined = byId.get(cursor.parent_generation_id);
      if (!parent) break;
      depth += 1;
      cursor = parent;
    }
    depths.set(node.id, depth);
  }
  return depths;
}
