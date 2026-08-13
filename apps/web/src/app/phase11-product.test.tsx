/**
 * Phase 11 product surfaces: shell, player, library, projects, chips.
 *
 * The properties worth defending here are the ones a screenshot cannot
 * show: that the player survives navigation, that Simple mode really is
 * simple, that prompt chips edit visible text rather than hidden
 * parameters, and that an empty library says so instead of inventing
 * demo songs.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AppShell } from "@/components/shell/AppShell";
import { PlayerProvider, formatTime } from "@/components/player/PlayerProvider";
import { hasTerm, toggleTerm } from "@/components/PromptChips";
import LibraryPage from "@/app/library/page";
import { matchesFilter } from "@/lib/library";
import ProjectsPage from "@/app/projects/page";
import CreatePage from "@/app/create/page";
import type { Generation } from "@/lib/api";

let pathname = "/create";
vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ id: "gen-1" }),
}));

function generation(overrides: Partial<Generation> = {}): Generation {
  return {
    id: "gen-1", title: "Midnight Window", prompt: "Dreamy Korean indie pop",
    lyrics: "[Verse]\n가사", vocal_gender: "female", duration_requested: 30,
    duration_actual: 30, seed: 1, language: "ko", instrumental: false,
    bpm: null, key_scale: null, time_signature: null,
    parent_generation_id: null, variation_label: null, advisories: [],
    request_trace: null, status: "COMPLETED", provider: "ace_step",
    model_name: "acestep-v15-turbo", model_version: "1.5.0",
    created_at: new Date().toISOString(), started_at: null, completed_at: null,
    error_code: null, error_message: null,
    audio_assets: [
      { id: "a1", asset_type: "MASTER", format: "wav", mime_type: "audio/wav",
        file_extension: "wav", sample_rate: 48000, bit_depth: 24, bitrate: null,
        channels: 2, duration: 30, storage_key: "k", sha256: "s", file_size: 1,
        created_at: new Date().toISOString() },
      { id: "a2", asset_type: "PREVIEW", format: "mp3", mime_type: "audio/mpeg",
        file_extension: "mp3", sample_rate: 48000, bit_depth: null, bitrate: 320000,
        channels: 2, duration: 30, storage_key: "k2", sha256: "s2", file_size: 1,
        created_at: new Date().toISOString() },
    ],
    ...overrides,
  };
}

/** SongCard consumes the player context, so any page that renders
 *  cards must be mounted inside the provider — same as in the app. */
function renderWithPlayer(ui: React.ReactNode) {
  return render(<PlayerProvider>{ui}</PlayerProvider>);
}

function json(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

beforeEach(() => {
  pathname = "/create";
  window.localStorage.clear();
  window.scrollTo = vi.fn();
  window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  window.HTMLMediaElement.prototype.pause = vi.fn();
  window.HTMLMediaElement.prototype.load = vi.fn();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ── App shell ─────────────────────────────────────────────────────────

describe("app shell", () => {
  it("offers the three primary destinations", () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    render(
      <PlayerProvider>
        <AppShell>content</AppShell>
      </PlayerProvider>,
    );
    const nav = screen.getAllByRole("navigation", { name: "Main" })[0];
    expect(within(nav).getByRole("link", { name: "Create" })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: "Library" })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: "Projects" })).toBeInTheDocument();
  });

  it("marks the current destination for assistive tech", () => {
    pathname = "/library";
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    render(
      <PlayerProvider>
        <AppShell>content</AppShell>
      </PlayerProvider>,
    );
    const nav = screen.getAllByRole("navigation", { name: "Main" })[0];
    expect(within(nav).getByRole("link", { name: "Library" })).toHaveAttribute(
      "aria-current", "page",
    );
  });

  it("hides the player bar until something is loaded", () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    render(
      <PlayerProvider>
        <AppShell>content</AppShell>
      </PlayerProvider>,
    );
    // An empty strip would eat 84px of a phone screen for nothing.
    expect(screen.queryByRole("region", { name: "Player" })).toBeNull();
  });

  it("mounts exactly one audio element for the whole app", () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    const { container } = render(
      <PlayerProvider>
        <AppShell>content</AppShell>
      </PlayerProvider>,
    );
    // One element is what guarantees only one track can ever play.
    expect(container.querySelectorAll("audio")).toHaveLength(1);
  });
});

// ── Player ────────────────────────────────────────────────────────────

describe("global player", () => {
  it("appears and plays when a track is started from the library", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [generation()], total: 1, limit: 20, offset: 0 })));
    render(
      <PlayerProvider>
        <AppShell>
          <LibraryPage />
        </AppShell>
      </PlayerProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "Play Midnight Window" }));

    const player = await screen.findByRole("region", { name: "Player" });
    expect(within(player).getByText("Midnight Window")).toBeInTheDocument();
    expect(window.HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it("exposes seek and volume as labelled controls", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [generation()], total: 1, limit: 20, offset: 0 })));
    render(
      <PlayerProvider>
        <AppShell>
          <LibraryPage />
        </AppShell>
      </PlayerProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "Play Midnight Window" }));
    const player = await screen.findByRole("region", { name: "Player" });
    expect(within(player).getByLabelText("Seek")).toBeInTheDocument();
    expect(within(player).getByLabelText("Volume")).toBeInTheDocument();
  });

  it("keeps the same audio element across a page change", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [generation()], total: 1, limit: 20, offset: 0 })));
    const { container, rerender } = render(
      <PlayerProvider>
        <AppShell>
          <LibraryPage />
        </AppShell>
      </PlayerProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "Play Midnight Window" }));
    const before = container.querySelector("audio");

    // Simulate navigating to another page inside the same shell.
    pathname = "/projects";
    rerender(
      <PlayerProvider>
        <AppShell>
          <div>another page</div>
        </AppShell>
      </PlayerProvider>,
    );
    // Same node identity => playback was never torn down.
    expect(container.querySelector("audio")).toBe(before);
    expect(window.HTMLMediaElement.prototype.pause).not.toHaveBeenCalled();
  });

  it("formats clock times", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(65)).toBe("1:05");
    expect(formatTime(Number.NaN)).toBe("0:00");
  });
});

// ── Create: simple vs custom ──────────────────────────────────────────

describe("create modes", () => {
  function stub() {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "POST" && String(url).includes("preflight")) {
        return json({ advisories: [], sections: [], preamble_line_count: 0, estimated_syllables: 0 });
      }
      if (init?.method === "POST") return json({ generation_id: "gen-1", status: "QUEUED", advisories: [] });
      return json(generation({ status: "QUEUED" }));
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("lands in Simple mode", () => {
    stub();
    render(<CreatePage />);
    expect(screen.getByRole("tab", { name: "Simple" })).toHaveAttribute("aria-selected", "true");
  });

  it("Simple mode hides the advanced surface entirely", () => {
    stub();
    render(<CreatePage />);
    // A first-time user must reach Create without meeting any of this.
    expect(screen.queryByLabelText("BPM")).toBeNull();
    expect(screen.queryByLabelText("Key / Scale")).toBeNull();
    expect(screen.queryByLabelText("Duration")).toBeNull();
    expect(screen.queryByText(/Song presets/)).toBeNull();
    // But the essentials are all there.
    expect(screen.getByLabelText("Title")).toBeInTheDocument();
    expect(screen.getByLabelText("Music description")).toBeInTheDocument();
    expect(screen.getByLabelText("Lyrics")).toBeInTheDocument();
    expect(screen.getByLabelText("Vocal")).toBeInTheDocument();
  });

  it("Custom mode reveals the advanced surface", async () => {
    const user = userEvent.setup();
    stub();
    render(<CreatePage />);
    await user.click(screen.getByRole("tab", { name: "Custom" }));
    expect(screen.getByLabelText("BPM")).toBeInTheDocument();
    expect(screen.getByLabelText("Duration")).toBeInTheDocument();
    expect(screen.getByLabelText("Language")).toBeInTheDocument();
    expect(screen.getByText(/Song presets/)).toBeInTheDocument();
  });

  it("a Simple generation submits without touching anything advanced", async () => {
    const user = userEvent.setup();
    const fetchMock = stub();
    render(<CreatePage />);
    await user.type(screen.getByLabelText("Title"), "Quick Song");
    await user.type(screen.getByLabelText("Music description"), "Warm lo-fi");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      const post = fetchMock.mock.calls.find(
        ([u, i]) => (i as RequestInit)?.method === "POST" && !String(u).includes("preflight"),
      );
      expect(post).toBeTruthy();
      const body = JSON.parse(String((post![1] as RequestInit).body));
      expect(body.title).toBe("Quick Song");
      expect(body.bpm).toBeNull();
      expect(body.duration).toBe(30);
    });
  });
});

// ── Prompt chips are text, not hidden parameters ──────────────────────

describe("prompt chips", () => {
  it("append a term to the visible brief", () => {
    expect(toggleTerm("", "Pop")).toBe("Pop");
    expect(toggleTerm("Warm piano", "Pop")).toBe("Warm piano, Pop");
  });

  it("toggle off when applied twice", () => {
    expect(toggleTerm("Warm piano, Pop", "Pop")).toBe("Warm piano");
    expect(hasTerm("Warm piano, Pop", "Pop")).toBe(true);
    expect(hasTerm("Warm piano", "Pop")).toBe(false);
  });

  it("do not match a term inside another word", () => {
    expect(hasTerm("Popcorn energy", "Pop")).toBe(false);
  });

  it("edit the description the user can see", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () => json({ advisories: [], sections: [], preamble_line_count: 0, estimated_syllables: 0 })));
    render(<CreatePage />);
    await user.click(screen.getByRole("button", { name: "Pop" }));
    // The chip's effect is visible in the textarea, not hidden in a payload.
    expect(screen.getByLabelText("Music description")).toHaveValue("Pop");
  });
});

// ── Library ───────────────────────────────────────────────────────────

describe("library", () => {
  it("shows an honest empty state rather than demo songs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [], total: 0, limit: 20, offset: 0 })));
    render(<LibraryPage />);
    expect(await screen.findByText("No songs yet")).toBeInTheDocument();
  });

  it("lists real generations", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [generation()], total: 1, limit: 20, offset: 0 })));
    renderWithPlayer(<LibraryPage />);
    expect(await screen.findByText("Midnight Window")).toBeInTheDocument();
  });

  it("filters by title", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () =>
      json({ items: [generation(), generation({ id: "g2", title: "Other Song" })], total: 2, limit: 20, offset: 0 })));
    renderWithPlayer(<LibraryPage />);
    await screen.findByText("Midnight Window");
    await user.type(screen.getByLabelText("Search by title"), "Other");
    expect(screen.queryByText("Midnight Window")).toBeNull();
    expect(screen.getByText("Other Song")).toBeInTheDocument();
  });

  it("explains an empty search result differently from an empty library", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [generation()], total: 1, limit: 20, offset: 0 })));
    renderWithPlayer(<LibraryPage />);
    await screen.findByText("Midnight Window");
    await user.type(screen.getByLabelText("Search by title"), "zzzz");
    expect(await screen.findByText("No matches")).toBeInTheDocument();
  });

  it("classifies status for the filter tabs", () => {
    expect(matchesFilter(generation(), "completed")).toBe(true);
    expect(matchesFilter(generation({ status: "GENERATING" }), "generating")).toBe(true);
    expect(matchesFilter(generation({ status: "FAILED" }), "failed")).toBe(true);
    expect(matchesFilter(generation({ status: "FAILED" }), "completed")).toBe(false);
    expect(matchesFilter(generation({ status: "FAILED" }), "all")).toBe(true);
  });

  it("surfaces a connection failure without a stack trace", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("ECONNREFUSED 127.0.0.1")));
    render(<LibraryPage />);
    expect(await screen.findByText("Could not load your library")).toBeInTheDocument();
    expect(screen.queryByText(/ECONNREFUSED/)).toBeNull();
  });
});

// ── Projects ──────────────────────────────────────────────────────────

describe("projects", () => {
  it("shows an empty state before any project exists", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    render(<ProjectsPage />);
    expect(await screen.findByText(/No projects yet. Name one above/)).toBeInTheDocument();
  });

  it("creates a project and opens it", async () => {
    const user = userEvent.setup();
    const created = { id: "p1", name: "Summer EP", generation_count: 0,
                      created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
    let listed: unknown[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") { listed = [created]; return json(created); }
      if (String(url).includes("/generations")) return json({ items: [], total: 0, limit: 0, offset: 0 });
      if (String(url).includes("/v1/projects")) return json({ items: listed });
      return json({ items: [], total: 0, limit: 20, offset: 0 });
    }));
    render(<ProjectsPage />);

    await user.type(await screen.findByLabelText("New project name"), "Summer EP");
    await user.click(screen.getByRole("button", { name: "Add" }));

    expect(await screen.findByRole("heading", { name: "Summer EP" })).toBeInTheDocument();
    expect(await screen.findByText("Nothing filed here yet")).toBeInTheDocument();
  });

  it("says plainly that deleting a project keeps the music", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [] })));
    render(<ProjectsPage />);
    expect(
      await screen.findByText(/Deleting a project never deletes its music/),
    ).toBeInTheDocument();
  });
});
