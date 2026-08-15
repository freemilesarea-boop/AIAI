/**
 * Phase 15 — productization guards.
 *
 * Two classes of regression are covered here, both of which previously
 * shipped unnoticed because nothing asserted them:
 *
 * *Playback reliability.* The player assumed `HTMLMediaElement.play()`
 * returns a Promise. Where it does not, the assumption was itself the
 * crash, and the failure surfaced as a dead play button rather than as
 * the error state the UI already had.
 *
 * *Design-system coherence.* The token layer exists, but seven Create
 * surfaces bypassed it with raw Tailwind palette classes, so the same
 * "muted text" was two different greys depending on which phase built
 * the screen. A grep-style assertion is the only thing that keeps that
 * from drifting back.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PlayerProvider, usePlayer, type PlayerTrack } from "@/components/player/PlayerProvider";

const TRACK: PlayerTrack = {
  id: "11111111-1111-4111-8111-111111111111",
  title: "Smoke",
  src: "http://api.test/v1/generations/x/audio?asset=preview",
  downloadUrl: "http://api.test/v1/generations/x/audio?asset=master&download=true",
  durationHint: 30,
};

function Harness() {
  const player = usePlayer();
  return (
    <div>
      <button onClick={() => player.play(TRACK)}>Play</button>
      <span data-testid="error">{player.error ?? "none"}</span>
      <span data-testid="playing">{String(player.playing)}</span>
    </div>
  );
}

describe("playback reliability", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("reports an error instead of crashing when play() rejects", async () => {
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() =>
      Promise.reject(new DOMException("NotAllowedError")),
    );
    const user = userEvent.setup();
    render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(await screen.findByText("This track could not be played.")).toBeTruthy();
    expect(screen.getByTestId("playing").textContent).toBe("false");
  });

  it("survives a play() that throws synchronously", async () => {
    // jsdom's own behaviour, and that of several embedded webviews.
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => {
      throw new Error("Not implemented: HTMLMediaElement.prototype.play");
    });
    const user = userEvent.setup();
    render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByTestId("error").textContent).toBe("This track could not be played.");
  });

  it("survives a play() that returns undefined instead of a Promise", async () => {
    // Older WebKit. `play().catch(...)` is a TypeError here, which is
    // what used to take out the click handler.
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(
      () => undefined as unknown as Promise<void>,
    );
    const user = userEvent.setup();
    render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Play" }));
    // No throw, and no false error either: nothing reported a failure.
    expect(screen.getByTestId("error").textContent).toBe("none");
  });

  it("clears a previous error when a new track starts successfully", async () => {
    const play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockImplementationOnce(() => Promise.reject(new DOMException("NotAllowedError")))
      .mockImplementation(() => Promise.resolve());
    const user = userEvent.setup();
    render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(await screen.findByText("This track could not be played.")).toBeTruthy();
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "Play" }));
    });
    expect(screen.getByTestId("error").textContent).toBe("none");
    expect(play).toHaveBeenCalledTimes(2);
  });
});

describe("design system coherence", () => {
  const ROOT = join(process.cwd(), "src");
  /**
   * Raw Tailwind palette classes. Tokens are the product's single source
   * of colour; a literal palette step here means one screen quietly
   * disagreeing with every other about what "muted" means.
   */
  const RAW_PALETTE =
    /\b(?:bg|text|border|ring|from|to|via|divide|placeholder|accent|outline)-(?:zinc|slate|gray|neutral|stone|violet|purple|indigo|amber|yellow|red|rose|green|emerald|teal|blue|sky)-\d{2,3}(?:\/\d{1,3})?\b/g;

  function sourceFiles(dir: string): string[] {
    return readdirSync(dir).flatMap((entry) => {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) return sourceFiles(full);
      if (!/\.tsx$/.test(entry) || /\.test\.tsx$/.test(entry)) return [];
      return [full];
    });
  }

  it("no component paints with a raw Tailwind palette class", () => {
    const offenders = sourceFiles(ROOT)
      .map((file) => ({ file, hits: readFileSync(file, "utf8").match(RAW_PALETTE) ?? [] }))
      .filter(({ hits }) => hits.length > 0)
      .map(({ file, hits }) => `${file.replace(ROOT, "src")}: ${[...new Set(hits)].join(", ")}`);
    expect(offenders).toEqual([]);
  });

  it("the token contract the components rely on is actually defined", () => {
    const css = readFileSync(join(ROOT, "app", "globals.css"), "utf8");
    const required = [
      "--surface-base", "--surface-raised", "--surface-overlay", "--surface-hover",
      "--text-primary", "--text-secondary", "--text-muted",
      "--border-subtle", "--border-default", "--border-strong",
      "--brand", "--brand-hover", "--brand-muted", "--brand-text",
      "--accent", "--accent-muted", "--danger", "--danger-muted", "--success",
    ];
    for (const token of required) {
      expect(css.includes(`${token}:`)).toBe(true);
    }
  });

  it("every token a component references exists in globals.css", () => {
    const css = readFileSync(join(ROOT, "app", "globals.css"), "utf8");
    const declared = new Set(Array.from(css.matchAll(/(--[a-z-]+):/g), (m) => m[1]));
    const used = new Set<string>();
    for (const file of sourceFiles(ROOT)) {
      for (const match of readFileSync(file, "utf8").matchAll(/var\((--[a-z-]+)\)/g)) {
        used.add(match[1]);
      }
    }
    expect([...used].filter((token) => !declared.has(token))).toEqual([]);
  });
});
