/**
 * Advanced musical controls and song-structure vocabulary.
 *
 * Every value here mirrors `packages/schemas` (`luber_schemas.songcraft`),
 * which in turn was read out of the pinned ACE-Step build
 * (`acestep/constants.py` @ 6d467e4b) rather than chosen. The UI must
 * never offer a value the engine does not accept, so
 * `songcraft.parity.test.ts` asserts this file against the Python source
 * of truth — a drifting constant fails the web test suite.
 *
 * Only parameters LUBER has verified are exposed. Several other engine
 * parameters exist and are deliberately absent; the reasons live next to
 * `UNEXPOSED_ENGINE_PARAMETERS` in the Python module.
 */

/** Upstream `BPM_MIN` / `BPM_MAX`. */
export const BPM_MIN = 30;
export const BPM_MAX = 300;

/** Upstream allows 10–600; LUBER caps at 360 (verified path only). */
export const DURATION_MIN = 10;
export const DURATION_MAX = 360;

/**
 * Upstream `VALID_TIME_SIGNATURES = [2, 3, 4, 6]`.
 *
 * The value the engine conditions on is the bare numerator ("4"), not
 * "4/4" — the LM's metadata vocabulary is constrained to those integers.
 * The label is for humans only; the value is what gets sent.
 */
export const TIME_SIGNATURE_OPTIONS: { value: string; label: string }[] = [
  { value: "2", label: "2 (2/4)" },
  { value: "3", label: "3 (3/4)" },
  { value: "4", label: "4 (4/4)" },
  { value: "6", label: "6 (6/8)" },
];

export const KEYSCALE_NOTES = ["A", "B", "C", "D", "E", "F", "G"] as const;
/** Upstream also accepts ♯/♭; LUBER offers one spelling per key. */
export const KEYSCALE_ACCIDENTALS = ["", "#", "b"] as const;
export const KEYSCALE_MODES = ["major", "minor"] as const;

/** The 42 key/scale values the pinned engine accepts, in the same order. */
export const VALID_KEY_SCALES: string[] = KEYSCALE_NOTES.flatMap((note) =>
  KEYSCALE_ACCIDENTALS.flatMap((accidental) =>
    KEYSCALE_MODES.map((mode) => `${note}${accidental} ${mode}`),
  ),
);

/** Canonical section tags the editor offers, in song order. */
export const SECTION_TAG_PALETTE = [
  "[Intro]",
  "[Verse]",
  "[Verse 1]",
  "[Verse 2]",
  "[Pre-Chorus]",
  "[Chorus]",
  "[Post-Chorus]",
  "[Bridge]",
  "[Break]",
  "[Instrumental]",
  "[Outro]",
] as const;

/** A non-blocking pre-flight finding. Never prevents generation. */
export interface Advisory {
  code: string;
  level: "info" | "warning";
  message: string;
  detail: Record<string, unknown>;
}

/** One parsed lyric section, as reported by the backend parser. */
export interface SectionSummary {
  kind: string | null;
  label: string;
  index: number | null;
  line_number: number;
  line_count: number;
  has_content: boolean;
  recognised: boolean;
}

export interface PreflightResponse {
  advisories: Advisory[];
  sections: SectionSummary[];
  preamble_line_count: number;
  estimated_syllables: number;
}

export function isWarning(advisory: Advisory): boolean {
  return advisory.level === "warning";
}

/**
 * Parse a BPM text field.
 *
 * Empty means "not specified" — the engine decides — which is a
 * different answer from any particular number and must stay
 * distinguishable from one.
 */
export function parseBpmInput(raw: string): { bpm: number | null; error: string | null } {
  const trimmed = raw.trim();
  if (!trimmed) return { bpm: null, error: null };
  if (!/^\d+$/.test(trimmed)) return { bpm: null, error: "BPM must be a whole number." };
  const value = Number(trimmed);
  if (value < BPM_MIN || value > BPM_MAX) {
    return { bpm: null, error: `BPM must be between ${BPM_MIN} and ${BPM_MAX}.` };
  }
  return { bpm: value, error: null };
}

/**
 * Durations the product offers, shortest first.
 *
 * Mirrors `luber_schemas.songform.PRODUCT_DURATIONS`. Each is a
 * validated point: 30/60 from Phase 3, 120/180/240 from the Phase 9
 * long-form gates. The engine accepts up to 600s and the API schema up
 * to 360s — neither is offered, because neither has been validated end
 * to end on this deployment.
 */
export const PRODUCT_DURATIONS = [30, 60, 120, 180, 240] as const;
export const PRODUCT_MAX_DURATION = 240;

/** At or above this, a request is a full song rather than a demo. */
export const FULL_SONG_THRESHOLD_SECONDS = 120;

export interface StructureTemplate {
  id: string;
  name: string;
  description: string;
  sections: string[];
  suggestedDuration: number;
}

/**
 * Section-tag skeletons the editor can insert.
 *
 * These are conditioning aids, not controls: ACE-Step reads section
 * tags as part of the lyric text and nothing more. A template makes a
 * recognisable arrangement more likely; it does not enforce one.
 */
export const STRUCTURE_TEMPLATES: StructureTemplate[] = [
  {
    id: "pop",
    name: "Pop",
    description: "Two verses into a repeated chorus, with a bridge before the last one.",
    sections: ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]",
      "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Final Chorus]", "[Outro]"],
    suggestedDuration: 180,
  },
  {
    id: "ballad",
    name: "Ballad",
    description: "Verse-led and slower to arrive; the chorus lands after more story.",
    sections: ["[Intro]", "[Verse 1]", "[Verse 2]", "[Chorus]", "[Verse 3]",
      "[Chorus]", "[Bridge]", "[Final Chorus]", "[Outro]"],
    suggestedDuration: 240,
  },
  {
    id: "rnb",
    name: "R&B",
    description: "Tighter frame with room for phrasing rather than more sections.",
    sections: ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]",
      "[Chorus]", "[Bridge]", "[Outro]"],
    suggestedDuration: 180,
  },
  {
    id: "band",
    name: "Band",
    description: "Leaves an instrumental slot where a solo would sit.",
    sections: ["[Intro]", "[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]",
      "[Instrumental]", "[Final Chorus]", "[Outro]"],
    suggestedDuration: 180,
  },
  {
    id: "minimal",
    name: "Verse / Chorus",
    description: "The smallest shape that still reads as a song.",
    sections: ["[Verse]", "[Chorus]"],
    suggestedDuration: 60,
  },
];

export interface SongPreset {
  id: string;
  name: string;
  description: string;
  duration: number;
  templateId: string | null;
  instrumental: boolean;
}

/** A starting frame. Never carries a prompt or lyrics — that stays the user's. */
export const SONG_PRESETS: SongPreset[] = [
  { id: "short_demo", name: "Short Demo", description: "One verse and a chorus, for trying an idea quickly.", duration: 60, templateId: "minimal", instrumental: false },
  { id: "full_pop_song", name: "Full Pop Song", description: "A complete pop arrangement with a bridge and a final chorus.", duration: 180, templateId: "pop", instrumental: false },
  { id: "ballad", name: "Ballad", description: "Longer and verse-led, for a slower emotional build.", duration: 240, templateId: "ballad", instrumental: false },
  { id: "rnb", name: "R&B", description: "A tighter frame that leaves space for vocal phrasing.", duration: 180, templateId: "rnb", instrumental: false },
  { id: "band_song", name: "Band Song", description: "Verse/chorus with an instrumental section for a solo.", duration: 180, templateId: "band", instrumental: false },
  { id: "instrumental", name: "Instrumental", description: "No vocals. Structure tags are left out entirely.", duration: 120, templateId: null, instrumental: true },
];

export function templateText(template: StructureTemplate): string {
  return template.sections.join("\n\n") + "\n";
}

export function findTemplate(id: string | null): StructureTemplate | null {
  return id ? (STRUCTURE_TEMPLATES.find((t) => t.id === id) ?? null) : null;
}

/**
 * Whether the sheet holds writing the user would mind losing.
 *
 * Section tags alone do not count: swapping one bare skeleton for
 * another loses nothing. Any other non-blank line does count.
 */
export function lyricsHaveContent(lyrics: string): boolean {
  return lyrics
    .split("\n")
    .some((line) => line.trim() !== "" && !/^\s*\[[^\]]{1,40}\]\s*$/.test(line));
}

export function formatDurationLabel(seconds: number): string {
  if (seconds < 60) return `${seconds} seconds`;
  const minutes = seconds / 60;
  if (!Number.isInteger(minutes)) return `${seconds} seconds`;
  return minutes === 1 ? "1 minute" : `${minutes} minutes`;
}
