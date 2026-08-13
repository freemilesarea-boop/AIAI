/**
 * Drift guard: the browser must never offer a value the engine rejects.
 *
 * `songcraft.ts` mirrors `luber_schemas.songcraft`, which mirrors the
 * pinned ACE-Step build. Mirrors rot silently, so this test reads the
 * Python module and asserts the two agree. If an ACE-Step upgrade
 * changes the accepted parameter surface, the Python constants change,
 * and this test fails until the UI is updated to match — which is the
 * point: the failure is the notification.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  BPM_MAX,
  BPM_MIN,
  DURATION_MAX,
  DURATION_MIN,
  KEYSCALE_ACCIDENTALS,
  KEYSCALE_MODES,
  KEYSCALE_NOTES,
  SECTION_TAG_PALETTE,
  TIME_SIGNATURE_OPTIONS,
  VALID_KEY_SCALES,
} from "./songcraft";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const SONGCRAFT_PY = resolve(
  REPO_ROOT,
  "packages/schemas/src/luber_schemas/songcraft.py",
);

const source = readFileSync(SONGCRAFT_PY, "utf8");

/** Read `NAME = 123` out of the Python module. */
function pyInt(name: string): number {
  const match = source.match(new RegExp(`^${name}\\s*=\\s*(\\d+)`, "m"));
  if (!match) throw new Error(`could not find int ${name} in songcraft.py`);
  return Number(match[1]);
}

/** Read a single-line or multi-line tuple of quoted strings. */
function pyStringTuple(name: string): string[] {
  const match = source.match(new RegExp(`^${name}[^=]*=\\s*\\(([\\s\\S]*?)\\)`, "m"));
  if (!match) throw new Error(`could not find tuple ${name} in songcraft.py`);
  return [...match[1].matchAll(/"([^"]*)"/g)].map((m) => m[1]);
}

describe("songcraft.ts mirrors luber_schemas.songcraft", () => {
  it("reads the real Python module", () => {
    // Guard against a silently-empty source making everything pass.
    expect(source).toContain("Engine-verified parameter surface");
  });

  it("uses the engine's BPM bounds", () => {
    expect(BPM_MIN).toBe(pyInt("BPM_MIN"));
    expect(BPM_MAX).toBe(pyInt("BPM_MAX"));
  });

  it("uses the same duration bounds", () => {
    expect(DURATION_MIN).toBe(pyInt("DURATION_MIN"));
    expect(DURATION_MAX).toBe(pyInt("DURATION_MAX"));
  });

  it("offers exactly the engine's time signatures", () => {
    expect(TIME_SIGNATURE_OPTIONS.map((o) => o.value)).toEqual(
      pyStringTuple("VALID_TIME_SIGNATURE_VALUES"),
    );
  });

  it("builds key/scale values from the same notes, accidentals and modes", () => {
    expect([...KEYSCALE_NOTES]).toEqual(pyStringTuple("KEYSCALE_NOTES"));
    expect([...KEYSCALE_ACCIDENTALS]).toEqual(pyStringTuple("KEYSCALE_ACCIDENTALS"));
    expect([...KEYSCALE_MODES]).toEqual(pyStringTuple("KEYSCALE_MODES"));
  });

  it("produces the same 42 key/scale values in the same order", () => {
    const notes = pyStringTuple("KEYSCALE_NOTES");
    const accidentals = pyStringTuple("KEYSCALE_ACCIDENTALS");
    const modes = pyStringTuple("KEYSCALE_MODES");
    const expected = notes.flatMap((note) =>
      accidentals.flatMap((accidental) => modes.map((mode) => `${note}${accidental} ${mode}`)),
    );
    expect(VALID_KEY_SCALES).toEqual(expected);
    expect(VALID_KEY_SCALES).toHaveLength(42);
  });

  it("offers the same section tag palette", () => {
    expect([...SECTION_TAG_PALETTE]).toEqual(pyStringTuple("SECTION_TAG_PALETTE"));
  });

  it("never offers a unicode accidental the backend does not store", () => {
    expect(VALID_KEY_SCALES.some((k) => k.includes("♯") || k.includes("♭"))).toBe(false);
  });

  it("never offers a fraction-style time signature", () => {
    expect(TIME_SIGNATURE_OPTIONS.every((o) => !o.value.includes("/"))).toBe(true);
  });
});
