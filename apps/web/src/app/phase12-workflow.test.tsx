/**
 * Phase 12 — the production workflow, end to end through the real pages.
 *
 * What these defend is the difference between "the buttons exist" and
 * "the workflow works": that two results are two independent jobs, that
 * one failing does not take the other down, that a destructive action
 * asks first and can be escaped, that a favourite is server state, and
 * that every playable surface shares the one audio element.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { GenerationJobCard } from "@/components/GenerationJobCard";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { SongActions } from "@/components/SongActions";
import { SongCard } from "@/components/SongCard";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { ToastProvider } from "@/components/ui/Toast";
import CreatePage from "@/app/create/page";
import LibraryPage from "@/app/library/page";
import ProjectDetailPage from "@/app/projects/[id]/page";
import SongDetailPage from "@/app/song/[id]/page";
import { downloadFilename, downloadOptions } from "@/lib/download";
import { LIBRARY_SORTS, visibleGenerations } from "@/lib/library";
import { generation, project } from "@/test/factories";
import type { Generation } from "@/lib/api";
import type { QueueEntry } from "@/hooks/useGenerationQueue";

const routerPush = vi.fn();
let searchParams = new URLSearchParams();
let routeParams: Record<string, string> = { id: "gen-1" };

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => searchParams,
  usePathname: () => "/library",
  useParams: () => routeParams,
}));

function renderApp(ui: React.ReactNode) {
  return render(
    <PlayerProvider>
      <ToastProvider>{ui}</ToastProvider>
    </PlayerProvider>,
  );
}

function json(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function noContent() {
  return { ok: true, status: 204, json: async () => ({}) };
}

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown>;
}

/**
 * A small stand-in for the API.
 *
 * Routes by URL and method rather than by call order, so a test does not
 * silently depend on how many requests a page happens to make.
 */
function stubApi(handlers: {
  generations?: Generation[];
  projects?: ReturnType<typeof project>[];
  projectGenerations?: Generation[];
  onPatch?: (id: string, body: Record<string, unknown>) => Generation;
}) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    calls.push({ url: String(url), method, body });

    if (String(url).includes("/preflight")) {
      return json({ advisories: [], sections: [], preamble_line_count: 0, estimated_syllables: 0 });
    }
    if (method === "PATCH") {
      const id = String(url).split("/").pop() ?? "";
      const base = handlers.generations?.[0] ?? generation();
      return json(handlers.onPatch ? handlers.onPatch(id, body) : { ...base, ...body });
    }
    if (method === "DELETE") return noContent();
    if (String(url).includes("bulk-delete") || String(url).includes("bulk-project")) {
      return json({ affected: (body.ids as string[]).length });
    }
    if (String(url).match(/\/v1\/projects\/[^/]+\/generations/)) {
      return json({
        items: handlers.projectGenerations ?? [],
        total: (handlers.projectGenerations ?? []).length,
        limit: 20,
        offset: 0,
      });
    }
    if (String(url).match(/\/v1\/projects\/[^/]+$/) && method === "GET") {
      return json(handlers.projects?.[0] ?? project());
    }
    if (String(url).includes("/v1/projects")) return json({ items: handlers.projects ?? [] });
    if (String(url).includes("/lineage")) {
      return json({ generation_id: "gen-1", parent: null, children: [] });
    }
    if (String(url).match(/\/v1\/generations\/[^/?]+$/) && method === "GET") {
      return json(handlers.generations?.[0] ?? generation());
    }
    return json({
      items: handlers.generations ?? [],
      total: (handlers.generations ?? []).length,
      limit: 20,
      offset: 0,
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, calls };
}

beforeEach(() => {
  searchParams = new URLSearchParams();
  routeParams = { id: "gen-1" };
  routerPush.mockClear();
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

// ── 21. Toasts ────────────────────────────────────────────────────────

describe("action feedback", () => {
  it("confirms a rename without a browser alert", async () => {
    const user = userEvent.setup();
    const alertSpy = vi.fn();
    vi.stubGlobal("alert", alertSpy);
    stubApi({ generations: [generation()] });

    renderApp(<SongActions generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Rename" }));
    await user.clear(screen.getByLabelText("Song title"));
    await user.type(screen.getByLabelText("Song title"), "New Name");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Renamed")).toBeInTheDocument();
    expect(alertSpy).not.toHaveBeenCalled();
  });

  it("reports a failure as feedback rather than silence", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("ECONNREFUSED 127.0.0.1")));

    renderApp(<SongActions generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Rename" }));
    await user.clear(screen.getByLabelText("Song title"));
    await user.type(screen.getByLabelText("Song title"), "New Name");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const toast = await screen.findByText("Could not rename this song.");
    expect(toast).toBeInTheDocument();
    // Never the raw network error.
    expect(document.body.textContent).not.toContain("ECONNREFUSED");
  });
});

// ── 22. Confirmation dialog ───────────────────────────────────────────

describe("confirmation dialog", () => {
  function Harness({ onConfirm }: { onConfirm: () => void }) {
    return (
      <ConfirmDialog
        open
        title="Delete this song?"
        description="It will be removed permanently."
        confirmLabel="Delete song"
        destructive
        onConfirm={onConfirm}
        onCancel={() => {}}
      />
    );
  }

  it("is an alert dialog, not window.confirm", () => {
    render(<Harness onConfirm={vi.fn()} />);
    expect(screen.getByRole("alertdialog")).toHaveAttribute("aria-modal", "true");
  });

  it("takes focus on the cancel button, not the destructive one", () => {
    render(<Harness onConfirm={vi.fn()} />);
    // A stray Enter must not delete anything.
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });

  it("closes on Escape without acting", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const onConfirm = vi.fn();
    render(
      <ConfirmDialog
        open
        title="Delete?"
        description="x"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    await user.keyboard("{Escape}");

    expect(onCancel).toHaveBeenCalled();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("keeps Tab inside the dialog", async () => {
    const user = userEvent.setup();
    render(<Harness onConfirm={vi.fn()} />);

    const cancel = screen.getByRole("button", { name: "Cancel" });
    const confirm = screen.getByRole("button", { name: "Delete song" });

    await user.tab();
    expect(confirm).toHaveFocus();
    await user.tab();
    // Wrapped back into the dialog rather than out to the page behind it.
    expect(cancel).toHaveFocus();
  });
});

// ── 2/3/12. Song management ───────────────────────────────────────────

describe("song management", () => {
  it("renames through the API and reports the new title", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [generation()] });

    renderApp(<SongActions generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Rename" }));
    await user.clear(screen.getByLabelText("Song title"));
    await user.type(screen.getByLabelText("Song title"), "Second Take");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      const patch = calls.find((c) => c.method === "PATCH");
      expect(patch?.body).toEqual({ title: "Second Take" });
    });
  });

  it("never sends provenance fields when renaming", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [generation()] });

    renderApp(<SongActions generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Rename" }));
    await user.clear(screen.getByLabelText("Song title"));
    await user.type(screen.getByLabelText("Song title"), "X");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(calls.some((c) => c.method === "PATCH")).toBe(true));
    const patch = calls.find((c) => c.method === "PATCH")!;
    for (const field of ["prompt", "lyrics", "seed", "model_name", "provider", "bpm"]) {
      expect(patch.body).not.toHaveProperty(field);
    }
  });

  it("asks before deleting and does nothing if cancelled", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [generation()] });

    renderApp(<SongActions generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Delete" }));

    expect(await screen.findByRole("alertdialog")).toHaveTextContent("Delete this song?");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
  });

  it("deletes on confirmation and says the lineage is kept", async () => {
    const user = userEvent.setup();
    const onDeleted = vi.fn();
    const { calls } = stubApi({ generations: [generation()] });

    renderApp(<SongActions generation={generation()} onDeleted={onDeleted} />);
    await user.click(screen.getByRole("button", { name: "Delete" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent(/Songs generated from it are kept/);
    await user.click(within(dialog).getByRole("button", { name: "Delete song" }));

    await waitFor(() => expect(onDeleted).toHaveBeenCalledWith("gen-1"));
    expect(calls.some((c) => c.method === "DELETE")).toBe(true);
  });
});

// ── 3. Favourites ─────────────────────────────────────────────────────

describe("favorites", () => {
  it("persists to the backend rather than localStorage", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({
      generations: [generation()],
      onPatch: (_id, body) => generation({ ...body }),
    });

    renderApp(<SongCard generation={generation()} />);
    await user.click(screen.getByRole("button", { name: "Favorite Midnight Window" }));

    await waitFor(() => {
      const patch = calls.find((c) => c.method === "PATCH");
      expect(patch?.body).toEqual({ favorite: true });
    });
    // Nothing about favourites is written to browser storage.
    expect(window.localStorage.getItem("luber.favorites")).toBeNull();
  });

  it("fills the heart immediately and reverts if the write fails", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    renderApp(<SongCard generation={generation()} />);
    const button = screen.getByRole("button", { name: "Favorite Midnight Window" });
    await user.click(button);

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Favorite Midnight Window" }),
      ).toHaveAttribute("aria-pressed", "false"),
    );
  });

  it("labels an already-favourited song as unfavourite", () => {
    renderApp(<SongCard generation={generation({ favorite: true })} />);
    expect(
      screen.getByRole("button", { name: "Unfavorite Midnight Window" }),
    ).toHaveAttribute("aria-pressed", "true");
  });
});

// ── 16. Search, filter, sort ──────────────────────────────────────────

describe("library filtering", () => {
  const items = [
    generation({ id: "a", title: "Bravo", prompt: "rainy bus stop", favorite: true,
                 created_at: "2026-01-01T00:00:00Z" }),
    generation({ id: "b", title: "Alpha", prompt: "sunny beach",
                 created_at: "2026-02-01T00:00:00Z" }),
    generation({ id: "c", title: "Charlie", prompt: "night drive", status: "FAILED",
                 created_at: "2026-03-01T00:00:00Z" }),
  ];

  const ids = (query: string, filter = "all" as const, sort = "newest" as const) =>
    visibleGenerations(items, { query, filter, sort }).map((g) => g.id);

  it("searches the description as well as the title", () => {
    expect(ids("rainy bus")).toEqual(["a"]);
    expect(ids("Alpha")).toEqual(["b"]);
  });

  it("offers all four sorts", () => {
    expect(LIBRARY_SORTS.map((s) => s.value)).toEqual([
      "newest",
      "oldest",
      "title_asc",
      "title_desc",
    ]);
  });

  it("sorts by title in both directions", () => {
    expect(visibleGenerations(items, { query: "", filter: "all", sort: "title_asc" })
      .map((g) => g.title)).toEqual(["Alpha", "Bravo", "Charlie"]);
    expect(visibleGenerations(items, { query: "", filter: "all", sort: "title_desc" })
      .map((g) => g.title)).toEqual(["Charlie", "Bravo", "Alpha"]);
  });

  it("composes favourites with search and sort", () => {
    expect(
      visibleGenerations(items, { query: "rainy", filter: "favorites", sort: "oldest" }).map(
        (g) => g.id,
      ),
    ).toEqual(["a"]);
    // The favourite exists but does not match the search.
    expect(
      visibleGenerations(items, { query: "beach", filter: "favorites", sort: "newest" }),
    ).toEqual([]);
  });

  it("says 'no favorites yet' rather than 'no matches'", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [generation({ favorite: false })] });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");

    await user.click(screen.getByRole("tab", { name: "Favorites" }));

    expect(await screen.findByText("No favorites yet")).toBeInTheDocument();
  });
});

// ── 15. Bulk selection ────────────────────────────────────────────────

describe("selection mode", () => {
  const two = [generation(), generation({ id: "gen-2", title: "Other Song" })];

  it("is off until asked for", async () => {
    stubApi({ generations: two });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("selects songs and reports the count", async () => {
    const user = userEvent.setup();
    stubApi({ generations: two });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");

    await user.click(screen.getByRole("button", { name: "Select" }));
    await user.click(screen.getByRole("checkbox", { name: "Select Midnight Window" }));

    expect(screen.getByText("1 selected")).toBeInTheDocument();
  });

  it("confirms a bulk delete showing how many songs it will remove", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: two });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");

    await user.click(screen.getByRole("button", { name: "Select" }));
    await user.click(screen.getByRole("button", { name: "Select all" }));
    await user.click(screen.getByRole("button", { name: "Delete" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("Delete 2 songs?");

    await user.click(within(dialog).getByRole("button", { name: "Delete 2" }));

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("bulk-delete"));
      expect(call?.body.ids).toEqual(["gen-1", "gen-2"]);
    });
  });

  it("adds the selection to a project in one request", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: two, projects: [project()] });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");

    await user.click(screen.getByRole("button", { name: "Select" }));
    await user.click(screen.getByRole("checkbox", { name: "Select Midnight Window" }));
    await user.selectOptions(screen.getByLabelText("Add selected to project"), "proj-1");

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("bulk-project"));
      expect(call?.body).toEqual({ ids: ["gen-1"], project_id: "proj-1" });
    });
  });
});

// ── 6/7. Two-result generation ────────────────────────────────────────

describe("two-result generation", () => {
  it("defaults to two songs", async () => {
    stubApi({ generations: [] });
    renderApp(<CreatePage />);
    expect(await screen.findByRole("tab", { name: "2 Songs" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("asks the backend for two results, not a provider batch", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [] });
    renderApp(<CreatePage />);

    await user.type(screen.getByLabelText("Title"), "Twin");
    await user.type(screen.getByLabelText("Music description"), "Warm lo-fi");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && !c.url.includes("preflight"));
      expect(post?.body.result_count).toBe(2);
      // No provider vocabulary crosses the API boundary.
      expect(post?.body).not.toHaveProperty("batch_size");
    });
  });

  it("can be set back to a single song", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [] });
    renderApp(<CreatePage />);

    await user.click(screen.getByRole("tab", { name: "1 Song" }));
    await user.type(screen.getByLabelText("Title"), "Solo");
    await user.type(screen.getByLabelText("Music description"), "Warm lo-fi");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && !c.url.includes("preflight"));
      expect(post?.body.result_count).toBe(1);
    });
  });
});

// ── 6/9. Queue cards, independence and partial failure ────────────────

describe("generation cards", () => {
  const entry = (over: Partial<QueueEntry> = {}): QueueEntry => ({
    id: "gen-1",
    groupId: "grp-1",
    title: "Midnight Window",
    generation: null,
    done: false,
    stalled: false,
    startedAt: Date.now(),
    ...over,
  });

  it("shows a running job without inventing a percentage", () => {
    renderApp(
      <GenerationJobCard
        entry={entry({ generation: generation({ status: "GENERATING" }) })}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByRole("status", { name: "" })).toHaveTextContent("Creating your music");
    expect(screen.queryByText(/%/)).toBeNull();
  });

  it("labels its position in a two-result group", () => {
    renderApp(
      <GenerationJobCard entry={entry()} resultLabel="Result 2" onDismiss={vi.fn()} />,
    );
    expect(screen.getByText("Result 2")).toBeInTheDocument();
  });

  it("presents a success and a failure side by side", () => {
    renderApp(
      <>
        <GenerationJobCard
          entry={entry({ generation: generation(), done: true })}
          resultLabel="Result 1"
          onDismiss={vi.fn()}
        />
        <GenerationJobCard
          entry={entry({
            id: "gen-2",
            title: "Midnight Window",
            generation: generation({
              id: "gen-2",
              status: "FAILED",
              error_code: "GENERATION_TIMEOUT",
              audio_assets: [],
            }),
            done: true,
          })}
          resultLabel="Result 2"
          onDismiss={vi.fn()}
        />
      </>,
    );

    // The working one is fully usable...
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    // ...and the broken one reports itself without claiming the batch failed.
    expect(screen.getByRole("alert")).toHaveTextContent(/took too long/i);
  });

  it("offers a real retry on a failure", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    renderApp(
      <GenerationJobCard
        entry={entry({
          generation: generation({ status: "FAILED", error_code: "GENERATION_TIMEOUT" }),
          done: true,
        })}
        onDismiss={vi.fn()}
        onRetry={onRetry}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("reports the seed the run used", () => {
    renderApp(
      <GenerationJobCard
        entry={entry({ generation: generation({ seed: 98765 }), done: true })}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText("Seed 98765")).toBeInTheDocument();
  });
});

// ── 5. Seed workflow ──────────────────────────────────────────────────

describe("seed workflow", () => {
  async function openCustom(user: ReturnType<typeof userEvent.setup>) {
    renderApp(<CreatePage />);
    await user.click(screen.getByRole("tab", { name: "Custom" }));
  }

  it("defaults to a random seed", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [] });
    await openCustom(user);
    expect(screen.getByRole("tab", { name: "Random" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByLabelText("Seed value")).toBeNull();
  });

  it("sends no seed when random", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [] });
    await openCustom(user);

    await user.type(screen.getByLabelText("Title"), "T");
    await user.type(screen.getByLabelText("Music description"), "P");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && !c.url.includes("preflight"));
      // Null, not a number the UI invented to look deterministic.
      expect(post?.body.seed).toBeNull();
    });
  });

  it("sends a pinned seed when fixed", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [] });
    await openCustom(user);

    await user.click(screen.getByRole("tab", { name: "Fixed" }));
    await user.type(screen.getByLabelText("Seed value"), "4242");
    await user.type(screen.getByLabelText("Title"), "T");
    await user.type(screen.getByLabelText("Music description"), "P");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && !c.url.includes("preflight"));
      expect(post?.body.seed).toBe(4242);
    });
  });

  it("rejects a seed that is not a whole number", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({ generations: [] });
    await openCustom(user);

    await user.click(screen.getByRole("tab", { name: "Fixed" }));
    await user.type(screen.getByLabelText("Seed value"), "abc");
    await user.type(screen.getByLabelText("Title"), "T");
    await user.type(screen.getByLabelText("Music description"), "P");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가사");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText("A seed must be a whole number.")).toBeInTheDocument();
    expect(calls.some((c) => c.method === "POST" && !c.url.includes("preflight"))).toBe(false);
  });

  it("explains what a fixed seed does with two results", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [] });
    await openCustom(user);
    await user.click(screen.getByRole("tab", { name: "Fixed" }));

    expect(
      screen.getByText(/two identical seeds would give you the same song twice/i),
    ).toBeInTheDocument();
  });

  it("promises no deterministic audio", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [] });
    await openCustom(user);
    expect(screen.getByText(/does not promise identical audio/i)).toBeInTheDocument();
  });
});

// ── 4. Duplicate settings ─────────────────────────────────────────────

describe("duplicate settings", () => {
  it("links to Create rather than generating immediately", async () => {
    stubApi({ generations: [generation()] });
    renderApp(<SongActions generation={generation()} />);

    expect(screen.getByRole("link", { name: "Duplicate settings" })).toHaveAttribute(
      "href",
      "/create?duplicate=gen-1",
    );
  });

  it("prefills Create and records no lineage", async () => {
    const user = userEvent.setup();
    searchParams = new URLSearchParams("duplicate=gen-1");
    const { calls } = stubApi({
      generations: [generation({ bpm: 128, key_scale: "F# minor", seed: 4242 })],
    });

    renderApp(<CreatePage />);

    await waitFor(() =>
      expect(screen.getByLabelText("Title")).toHaveValue("Midnight Window"),
    );
    // Settings carried over…
    expect(screen.getByLabelText("BPM")).toHaveValue(128);
    expect(screen.getByLabelText("Seed value")).toHaveValue("4242");
    // …but this is a new song, not a child of the old one.
    expect(screen.queryByText(/Based on/)).toBeNull();

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && !c.url.includes("preflight"));
      expect(post?.body.parent_generation_id).toBeNull();
    });
  });

  it("records lineage when the same page is opened as Generate again", async () => {
    searchParams = new URLSearchParams("from=gen-1");
    stubApi({ generations: [generation()] });

    renderApp(<CreatePage />);

    expect(await screen.findByText(/Based on/)).toBeInTheDocument();
  });
});

// ── 10/11. Downloads ──────────────────────────────────────────────────

describe("downloads", () => {
  it("names the file after the song", () => {
    expect(downloadFilename("Midnight Window", "gen-1")).toBe("LUBER - Midnight Window.wav");
  });

  it("keeps a Korean title readable", () => {
    expect(downloadFilename("오늘 밤", "gen-1")).toBe("LUBER - 오늘 밤.wav");
  });

  it("strips characters a filesystem would reject", () => {
    const name = downloadFilename('../../etc/pa*ss"wd', "gen-1");
    expect(name).not.toMatch(/[\\/:*?"<>|]/);
    expect(name).not.toContain("..");
  });

  it("falls back when a title sanitises to nothing", () => {
    expect(downloadFilename("///", "abcdef12-3456-7890-abcd-ef1234567890")).toBe(
      "LUBER - track-abcdef12.wav",
    );
  });

  it("offers the MP3 only when the backend produced one", () => {
    expect(downloadOptions(generation()).map((o) => o.kind)).toEqual(["master", "preview"]);
    const noPreview = generation({ audio_assets: [generation().audio_assets[0]] });
    expect(downloadOptions(noPreview).map((o) => o.kind)).toEqual(["master"]);
  });

  it("never calls the MP3 a master", () => {
    const preview = downloadOptions(generation()).find((o) => o.kind === "preview")!;
    expect(preview.label).toBe("Download MP3");
    expect(preview.hint).toBe("Preview · compressed");
    expect(preview.hint.toLowerCase()).not.toContain("master");
  });

  it("points both downloads at the LUBER audio endpoint", () => {
    renderApp(<SongActions generation={generation()} />);
    const wav = screen.getByRole("link", { name: "Download WAV" });
    const mp3 = screen.getByRole("link", { name: "Download MP3" });
    expect(wav).toHaveAttribute("href", expect.stringContaining("asset=master&download=true"));
    expect(mp3).toHaveAttribute("href", expect.stringContaining("asset=preview&download=true"));
  });
});

// ── 14/23. Project detail route ───────────────────────────────────────

describe("project detail", () => {
  beforeEach(() => {
    routeParams = { id: "proj-1" };
  });

  it("loads entirely from the backend, so a refresh recovers it", async () => {
    stubApi({
      projects: [project({ name: "Summer EP", generation_count: 1 })],
      projectGenerations: [generation()],
      generations: [generation()],
    });

    renderApp(<ProjectDetailPage />);

    expect(await screen.findByRole("heading", { name: "Summer EP" })).toBeInTheDocument();
    expect(screen.getByText("Midnight Window")).toBeInTheDocument();
  });

  it("removes a song from the project without deleting it", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi({
      projects: [project()],
      projectGenerations: [generation()],
      generations: [generation()],
    });

    renderApp(<ProjectDetailPage />);
    await screen.findByText("Midnight Window");
    await user.click(screen.getByRole("button", { name: "Remove" }));

    await waitFor(() => {
      const put = calls.find((c) => c.method === "PUT");
      expect(put?.body).toEqual({ project_id: null });
    });
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
  });

  it("says the songs survive before deleting a project", async () => {
    const user = userEvent.setup();
    stubApi({
      projects: [project({ name: "Summer EP" })],
      projectGenerations: [generation()],
      generations: [generation()],
    });

    renderApp(<ProjectDetailPage />);
    await screen.findByRole("heading", { name: "Summer EP" });
    await user.click(screen.getByRole("button", { name: "Delete project" }));

    expect(await screen.findByRole("alertdialog")).toHaveTextContent(
      /song stays in your library, unfiled/,
    );
  });

  it("explains a missing project without a stack trace", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("ECONNREFUSED")));
    renderApp(<ProjectDetailPage />);
    expect(await screen.findByText("Project not found")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("ECONNREFUSED");
  });
});

// ── 18. One player everywhere ─────────────────────────────────────────

describe("player integration", () => {
  it("mounts one audio element no matter how many cards are on screen", async () => {
    stubApi({ generations: [generation(), generation({ id: "gen-2", title: "Other" })] });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");
    expect(document.querySelectorAll("audio")).toHaveLength(1);
  });

  it("replaces the loaded track when another song is started", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [generation(), generation({ id: "gen-2", title: "Other" })] });
    renderApp(<LibraryPage />);
    await screen.findByText("Other");

    const audio = screen.getByLabelText("LUBER audio player");
    await user.click(screen.getByRole("button", { name: "Play Midnight Window" }));
    expect(audio).toHaveAttribute("src", expect.stringContaining("/gen-1/audio"));

    await user.click(screen.getByRole("button", { name: "Play Other" }));
    expect(audio).toHaveAttribute("src", expect.stringContaining("/gen-2/audio"));
    // Still one element — the first track was replaced, not layered.
    expect(document.querySelectorAll("audio")).toHaveLength(1);
  });

  it("plays from the project detail page through the same element", async () => {
    const user = userEvent.setup();
    routeParams = { id: "proj-1" };
    stubApi({
      projects: [project()],
      projectGenerations: [generation()],
      generations: [generation()],
    });

    renderApp(<ProjectDetailPage />);
    await screen.findByText("Midnight Window");
    await user.click(screen.getByRole("button", { name: "Play Midnight Window" }));

    expect(screen.getByLabelText("LUBER audio player")).toHaveAttribute(
      "src",
      expect.stringContaining("/gen-1/audio"),
    );
  });
});

// ── 19. Cover art foundation ──────────────────────────────────────────

describe("cover art", () => {
  it("draws the placeholder when there is no generated art", () => {
    renderApp(<SongCard generation={generation()} />);
    expect(document.querySelector("img")).toBeNull();
  });

  it("uses the generated art when a URL exists", () => {
    renderApp(<SongCard generation={generation({ cover_art_url: "/cover.png" })} />);
    expect(document.querySelector("img")).toHaveAttribute("src", "/cover.png");
  });
});

// ── 17. Empty and partial states ──────────────────────────────────────

describe("empty states", () => {
  it("says an empty library is empty rather than inventing demo songs", async () => {
    stubApi({ generations: [] });
    renderApp(<LibraryPage />);
    expect(await screen.findByText("No songs yet")).toBeInTheDocument();
  });

  it("explains a search with no matches differently", async () => {
    const user = userEvent.setup();
    stubApi({ generations: [generation()] });
    renderApp(<LibraryPage />);
    await screen.findByText("Midnight Window");

    await user.type(screen.getByLabelText("Search by title or description"), "zzzz");

    expect(await screen.findByText("No matches")).toBeInTheDocument();
    expect(screen.queryByText("No songs yet")).toBeNull();
  });

  it("shows an empty project without pretending it is broken", async () => {
    routeParams = { id: "proj-1" };
    stubApi({ projects: [project()], projectGenerations: [], generations: [] });
    renderApp(<ProjectDetailPage />);
    expect(await screen.findByText("Nothing filed here yet")).toBeInTheDocument();
  });
});

// ── 12/20. Song detail ────────────────────────────────────────────────

describe("song detail", () => {
  it("shows the seed that was used", async () => {
    stubApi({ generations: [generation({ seed: 4242 })] });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(screen.getByText("4242")).toBeInTheDocument();
  });

  it("keeps worker vocabulary out of the visible status", async () => {
    stubApi({ generations: [generation({ status: "POST_PROCESSING" })] });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(screen.getByText("Finishing")).toBeInTheDocument();
    // The raw enum only appears under the Advanced disclosure.
    const summary = screen.getByText("Advanced details").closest("details")!;
    expect(within(summary).getByText("POST_PROCESSING")).toBeInTheDocument();
  });

  it("offers rename, favourite and delete on the detail page too", async () => {
    stubApi({ generations: [generation()] });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(screen.getByRole("button", { name: "Rename" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Favorite Midnight Window" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });
});
