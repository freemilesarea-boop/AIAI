/**
 * Full-song durations, presets, and structure templates in the workspace.
 *
 * The rule these tests defend hardest: **a preset or template must never
 * silently destroy lyrics the user has written.** Applying one to a
 * sheet with words in it asks first, and "add after" is the default
 * answer. Everything else here — the duration set, the preset frames —
 * exists to make full songs reachable without changing what a 30-second
 * request does.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import CreatePage from "./page";

const GEN_ID = "d1d76e27-119a-41e8-a358-a492141efaba";

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function isPreflight(url: unknown): boolean {
  return String(url).includes("/preflight");
}

function isCreatePost(call: readonly unknown[]): boolean {
  const [url, init] = call as [string, RequestInit | undefined];
  return init?.method === "POST" && !isPreflight(url);
}

function jsonBodyOf(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

/** Preset names collide with template names and tag buttons; always scope. */
function presetGroup() {
  return within(screen.getByRole("group", { name: "Song presets" }));
}

function stubServer() {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "POST" && isPreflight(url)) {
      return jsonResponse({
        advisories: [],
        sections: [],
        preamble_line_count: 0,
        estimated_syllables: 0,
      });
    }
    if (init?.method === "POST") {
      return jsonResponse({ generation_id: GEN_ID, status: "QUEUED", advisories: [] });
    }
    return jsonResponse({ id: GEN_ID, status: "QUEUED", audio_assets: [] });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock };
}

async function submittedBody(fetchMock: ReturnType<typeof vi.fn>) {
  await waitFor(() => expect(fetchMock.mock.calls.some(isCreatePost)).toBe(true));
  return jsonBodyOf(fetchMock.mock.calls.find(isCreatePost)![1] as RequestInit);
}

async function fillRequired(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Title"), "Full Song");
  await user.type(screen.getByLabelText("Music description"), "Korean pop ballad");
}

/**
 * Advanced controls live behind the Custom tab from Phase 11 onward.
 * Simple mode is the default landing experience, so every test that
 * touches BPM, key, duration, language or presets must switch first —
 * the same click a real user makes.
 */
async function switchToCustom() {
  const user = userEvent.setup();
  const tab = screen.queryByRole("tab", { name: "Custom" });
  if (tab && tab.getAttribute("aria-selected") !== "true") await user.click(tab);
}

beforeEach(() => {
  window.localStorage.clear();
  window.scrollTo = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ── Full-song durations ───────────────────────────────────────────────

describe("full-song durations", () => {
  it("offers exactly the validated set", async () => {
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    const values = within(screen.getByLabelText("Duration"))
      .getAllByRole("option")
      .map((o) => (o as HTMLOptionElement).value);
    expect(values).toEqual(["30", "60", "120", "180", "240"]);
  });

  it("does not offer durations the engine accepts but we have not validated", async () => {
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    const values = within(screen.getByLabelText("Duration"))
      .getAllByRole("option")
      .map((o) => (o as HTMLOptionElement).value);
    expect(values).not.toContain("360");
    expect(values).not.toContain("600");
  });

  it("still defaults to 30 seconds, so nothing changes for existing users", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await fillRequired(user);
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect((await submittedBody(fetchMock)).duration).toBe(30);
  });

  it("submits a full-song duration when chosen", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await fillRequired(user);
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.selectOptions(screen.getByLabelText("Duration"), "240");
    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect((await submittedBody(fetchMock)).duration).toBe(240);
  });

  it("labels minute-length durations readably", async () => {
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    const labels = within(screen.getByLabelText("Duration"))
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(labels).toEqual([
      "30 seconds",
      "1 minute",
      "2 minutes",
      "3 minutes",
      "4 minutes",
    ]);
  });
});

// ── Presets ───────────────────────────────────────────────────────────

describe("song presets", () => {
  it("are offered and marked optional", async () => {
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    expect(screen.getByText(/Song presets/)).toBeInTheDocument();
    expect(presetGroup().getByRole("button", { name: /Full Pop Song/ })).toBeInTheDocument();
    expect(presetGroup().getByRole("button", { name: /^Ballad/ })).toBeInTheDocument();
    expect(presetGroup().getByRole("button", { name: /^Instrumental/ })).toBeInTheDocument();
  });

  it("sets the duration and inserts a structure into an empty sheet", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));

    expect(screen.getByLabelText("Duration")).toHaveValue("180");
    const lyrics = screen.getByLabelText("Lyrics") as HTMLTextAreaElement;
    expect(lyrics.value).toContain("[Verse 1]");
    expect(lyrics.value).toContain("[Final Chorus]");
  });

  it("never writes lyrics, only structure", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));

    const lyrics = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    const wordLines = lyrics
      .split("\n")
      .filter((line) => line.trim() && !/^\[[^\]]+\]$/.test(line.trim()));
    expect(wordLines).toEqual([]);
  });

  it("the instrumental preset switches the vocal off and adds no tags", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();

    await user.click(presetGroup().getByRole("button", { name: /^Instrumental/ }));

    expect(screen.getByLabelText("Vocal")).toHaveValue("instrumental");
    expect(screen.getByLabelText("Duration")).toHaveValue("120");
  });

  it("a preset frame reaches the API", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    render(<CreatePage />);
    await switchToCustom();

    await user.click(presetGroup().getByRole("button", { name: /^Ballad/ }));
    await fillRequired(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const body = await submittedBody(fetchMock);
    expect(body.duration).toBe(240);
    expect(String(body.lyrics)).toContain("[Bridge]");
  });
});

// ── Structure templates ───────────────────────────────────────────────

describe("structure templates", () => {
  it("are offered separately from presets", async () => {
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    const group = screen.getByRole("group", { name: "Structure templates" });
    expect(within(group).getByRole("button", { name: /Pop/ })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: /R&B/ })).toBeInTheDocument();
  });

  it("insert their sections into an empty sheet", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();

    const group = screen.getByRole("group", { name: "Structure templates" });
    await user.click(within(group).getByRole("button", { name: /R&B/ }));

    const lyrics = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    expect(lyrics).toContain("[Pre-Chorus]");
    expect(lyrics).toContain("[Outro]");
  });

  it("use only tags the backend recognises", async () => {
    // A template that warned about its own tags would be a bug.
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();

    const group = screen.getByRole("group", { name: "Structure templates" });
    await user.click(within(group).getByRole("button", { name: /^Pop/ }));

    const lyrics = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    const known = new Set([
      "[Intro]", "[Verse 1]", "[Verse 2]", "[Verse 3]", "[Pre-Chorus]", "[Chorus]",
      "[Final Chorus]", "[Post-Chorus]", "[Bridge]", "[Break]", "[Instrumental]", "[Outro]",
    ]);
    for (const line of lyrics.split("\n").filter((l) => l.trim())) {
      expect(known.has(line.trim())).toBe(true);
    }
  });
});

// ── The rule that matters: never destroy writing ──────────────────────

describe("applying structure to existing lyrics", () => {
  const WRITTEN = "[Verse]\n창밖에 비가 내려와";

  async function writeLyrics(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste(WRITTEN);
  }

  it("asks before touching lyrics that have words in them", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await writeLyrics(user);

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));

    expect(await screen.findByRole("alertdialog")).toBeInTheDocument();
    // Nothing has changed yet.
    expect(screen.getByLabelText("Lyrics")).toHaveValue(WRITTEN);
  });

  it("appends after the existing lyrics when asked to", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await writeLyrics(user);

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));
    await user.click(await screen.findByRole("button", { name: "Add after my lyrics" }));

    const lyrics = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    expect(lyrics).toContain("창밖에 비가 내려와");
    expect(lyrics).toContain("[Bridge]");
  });

  it("replaces only on a second, explicit confirmation", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await writeLyrics(user);

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));
    await user.click(await screen.findByRole("button", { name: "Replace my lyrics" }));

    const lyrics = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    expect(lyrics).not.toContain("창밖에 비가 내려와");
    expect(lyrics).toContain("[Verse 1]");
  });

  it("cancelling leaves the lyrics untouched", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await writeLyrics(user);

    await user.click(presetGroup().getByRole("button", { name: /Full Pop Song/ }));
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(screen.getByLabelText("Lyrics")).toHaveValue(WRITTEN);
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("swaps a bare skeleton without asking, because nothing is lost", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("[Verse]\n[Chorus]");

    const group = screen.getByRole("group", { name: "Structure templates" });
    await user.click(within(group).getByRole("button", { name: /^Pop/ }));

    expect(screen.queryByRole("alertdialog")).toBeNull();
    const swapped = (screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value;
    expect(swapped).toContain("[Pre-Chorus]");
    expect(swapped).toContain("[Final Chorus]");
  });

  it("a preset applied over written lyrics still sets the duration", async () => {
    const user = userEvent.setup();
    stubServer();
    render(<CreatePage />);
    await switchToCustom();
    await writeLyrics(user);

    await user.click(presetGroup().getByRole("button", { name: /^Ballad/ }));
    await user.click(await screen.findByRole("button", { name: "Add after my lyrics" }));

    expect(screen.getByLabelText("Duration")).toHaveValue("240");
  });
});
