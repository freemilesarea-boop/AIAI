/**
 * Phase 17 — version history on Song Detail.
 *
 * The tree is rendered from one backend response and never reconstructed
 * in the browser, so most of what is defended here is what the component
 * does with a response it did not expect: a server that predates the
 * `nodes` field, a single-node lineage, a chain deeper than the layout
 * can indent, an operation this build has never heard of. None of those
 * may blank the page.
 *
 * The rest is vocabulary. `REPLACE_RANGE` is what the database stores and
 * must never be what a user reads.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { PlayerProvider } from "@/components/player/PlayerProvider";
import { SongActions } from "@/components/SongActions";
import { ToastProvider } from "@/components/ui/Toast";
import { VersionHistory } from "@/components/VersionHistory";
import SongDetailPage from "@/app/song/[id]/page";
import type { LineageNode } from "@/lib/api";
import { generation, masterAsset, previewAsset } from "@/test/factories";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/song/gen-1",
  useParams: () => ({ id: "gen-1" }),
}));

function renderApp(ui: React.ReactNode) {
  return render(
    <PlayerProvider>
      <ToastProvider>{ui}</ToastProvider>
    </PlayerProvider>,
  );
}

function node(overrides: Partial<LineageNode> & { id: string }): LineageNode {
  return {
    parent_generation_id: null,
    title: overrides.id,
    status: "COMPLETED",
    operation: "ORIGINAL",
    created_at: "2026-08-15T12:00:00Z",
    duration_actual: 30,
    cover_art_url: null,
    edit_start_seconds: null,
    edit_end_seconds: null,
    ...overrides,
  };
}

/** A → B (extend) → C (replace), the shape Step 19 creates for real. */
const CHAIN: LineageNode[] = [
  node({ id: "a", title: "Midnight Window" }),
  node({
    id: "b",
    title: "Midnight Window (longer)",
    parent_generation_id: "a",
    operation: "EXTEND",
    edit_start_seconds: 30,
    edit_end_seconds: 45,
  }),
  node({
    id: "c",
    title: "Midnight Window (patched)",
    parent_generation_id: "b",
    operation: "REPLACE_SECTION",
    edit_start_seconds: 10,
    edit_end_seconds: 20,
  }),
];

describe("version history", () => {
  it("shows every version in the family, not just the current one", () => {
    render(<VersionHistory nodes={CHAIN} currentId="c" rootId="a" />);

    expect(screen.getByRole("heading", { name: "Version history" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
    for (const title of CHAIN.map((n) => n.title)) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }
  });

  it("marks the current version in text, not only in colour", () => {
    render(<VersionHistory nodes={CHAIN} currentId="b" rootId="a" />);

    const marked = screen.getByText("Current version").closest("[aria-current]");
    expect(marked).not.toBeNull();
    expect(marked).toHaveTextContent("Midnight Window (longer)");
    // Exactly one, or the user cannot tell where they are.
    expect(screen.getAllByText("Current version")).toHaveLength(1);
  });

  it("links to the other versions and not to itself", () => {
    render(<VersionHistory nodes={CHAIN} currentId="c" rootId="a" />);

    const links = screen.getAllByRole("link");
    expect(links.map((a) => a.getAttribute("href")).sort()).toEqual([
      "/song/a",
      "/song/b",
    ]);
  });

  it("names each operation in product vocabulary", () => {
    render(<VersionHistory nodes={CHAIN} currentId="a" rootId="a" />);

    expect(screen.getByText("Original")).toBeInTheDocument();
    expect(screen.getByText("Extended +15s")).toBeInTheDocument();
    expect(screen.getByText("Replaced 0:10–0:20")).toBeInTheDocument();
  });

  it("never shows the stored enum or an engine word", () => {
    const { container } = render(<VersionHistory nodes={CHAIN} currentId="c" rootId="a" />);
    const text = container.textContent ?? "";

    for (const banned of ["REPLACE_RANGE", "RANGE", "edit_kind", "Remix", "remix"]) {
      expect(text).not.toContain(banned);
    }
  });

  it("labels a cover and a plain re-generation distinctly", () => {
    const nodes = [
      node({ id: "a", title: "Original take" }),
      node({
        id: "b",
        title: "Cover take",
        parent_generation_id: "a",
        operation: "COVER",
      }),
      node({
        id: "c",
        title: "Another take",
        parent_generation_id: "a",
        operation: "GENERATE_AGAIN",
      }),
    ];
    render(<VersionHistory nodes={nodes} currentId="a" rootId="a" />);

    expect(screen.getByText("Cover")).toBeInTheDocument();
    expect(screen.getByText("Generated again")).toBeInTheDocument();
  });

  it("renders an unfinished version with its status rather than hiding it", () => {
    const nodes = [
      node({ id: "a", title: "Original take" }),
      node({
        id: "b",
        title: "Still working",
        parent_generation_id: "a",
        operation: "EXTEND",
        status: "GENERATING",
        duration_actual: null,
      }),
    ];
    render(<VersionHistory nodes={nodes} currentId="a" rootId="a" />);

    expect(screen.getByText("Still working")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  describe("shapes it did not ask for", () => {
    it("renders nothing when the song is the only version", () => {
      const { container } = render(
        <VersionHistory nodes={[node({ id: "a" })]} currentId="a" rootId="a" />,
      );
      expect(container).toBeEmptyDOMElement();
    });

    it("renders nothing rather than throwing when nodes are absent", () => {
      const { container } = render(
        // A server that predates the field. The page must survive it.
        <VersionHistory
          nodes={undefined as unknown as LineageNode[]}
          currentId="a"
          rootId={null}
        />,
      );
      expect(container).toBeEmptyDOMElement();
    });

    it("caps indentation so a deep chain cannot leave the screen", () => {
      const deep: LineageNode[] = Array.from({ length: 10 }, (_, index) =>
        node({
          id: `n${index}`,
          title: `Take ${index}`,
          parent_generation_id: index === 0 ? null : `n${index - 1}`,
          operation: index === 0 ? "ORIGINAL" : "EXTEND",
        }),
      );
      render(<VersionHistory nodes={deep} currentId="n9" rootId="n0" />);

      const pads = screen
        .getAllByRole("listitem")
        .map((li) => parseFloat((li as HTMLElement).style.paddingLeft) || 0);
      expect(pads).toHaveLength(10);
      // Four steps of 0.9rem, then flat — never growing without bound.
      expect(Math.max(...pads)).toBeCloseTo(3.6, 5);
    });

    it("still renders a node whose operation this build does not know", () => {
      const nodes = [
        node({ id: "a", title: "Original take" }),
        node({
          id: "b",
          title: "From a newer server",
          parent_generation_id: "a",
          operation: "SOMETHING_NEW" as LineageNode["operation"],
        }),
      ];
      render(<VersionHistory nodes={nodes} currentId="a" rootId="a" />);

      expect(screen.getByText("From a newer server")).toBeInTheDocument();
      expect(screen.getByText("Derived version")).toBeInTheDocument();
    });

    it("survives a node whose parent is not in the response", () => {
      const nodes = [
        node({ id: "a", title: "Original take" }),
        node({
          id: "b",
          title: "Orphan",
          parent_generation_id: "gone",
          operation: "EXTEND",
        }),
      ];
      render(<VersionHistory nodes={nodes} currentId="a" rootId="a" />);
      expect(screen.getByText("Orphan")).toBeInTheDocument();
    });
  });
});

// ── on the song page ──────────────────────────────────────────────────

function stubApi(lineage: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = String(url).includes("/lineage")
        ? lineage
        : String(url).includes("/v1/projects")
          ? { items: [] }
          : generation({
              id: "gen-1",
              audio_assets: [masterAsset({ duration: 30 }), previewAsset()],
            });
      return { ok: true, status: 200, json: async () => body };
    }),
  );
}

describe("song detail", () => {
  it("says where a derived version came from", async () => {
    stubApi({
      generation_id: "gen-1",
      parent: null,
      children: [],
      root_generation_id: "root-1",
      current_generation_id: "gen-1",
      nodes: [
        node({ id: "root-1", title: "First take" }),
        node({
          id: "gen-1",
          title: "Midnight Window",
          parent_generation_id: "root-1",
          operation: "EXTEND",
          edit_start_seconds: 30,
          edit_end_seconds: 45,
        }),
      ],
    });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    const history = await screen.findByRole("heading", { name: "Version history" });
    expect(history).toBeInTheDocument();
    expect(screen.getByText("First take")).toBeInTheDocument();
  });

  it("shows no version history for a song with no other versions", async () => {
    stubApi({
      generation_id: "gen-1",
      parent: null,
      children: [],
      root_generation_id: "gen-1",
      current_generation_id: "gen-1",
      nodes: [node({ id: "gen-1", title: "Midnight Window" })],
    });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(screen.queryByRole("heading", { name: "Version history" })).toBeNull();
  });

  it("does not crash on a lineage response with no nodes field", async () => {
    stubApi({ generation_id: "gen-1", parent: null, children: [] });
    renderApp(<SongDetailPage />);

    // The page still renders; the history is simply absent.
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(screen.queryByRole("heading", { name: "Version history" })).toBeNull();
  });

  it("offers the edit operations that have a real backend path", async () => {
    stubApi({ generation_id: "gen-1", parent: null, children: [] });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(screen.getByRole("heading", { name: "Edit this song" })).toBeInTheDocument();
    for (const name of ["Extend", "Replace section", "Create cover"]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });
});

// ── refusing to delete a song other versions came from ────────────────

describe("deleting a song with derived versions", () => {
  function stubDelete(detail: unknown) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if ((init?.method ?? "GET") === "DELETE") {
          return { ok: false, status: 409, json: async () => ({ detail }) };
        }
        return { ok: true, status: 200, json: async () => ({ items: [] }) };
      }),
    );
  }

  async function attemptDelete() {
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Delete song" }));
  }

  function renderActions(onDeleted: () => void) {
    renderApp(
      <SongActions
        generation={generation({ id: "gen-1", title: "Midnight Window" })}
        projects={[]}
        variant="detail"
        onChanged={() => {}}
        onDeleted={onDeleted}
      />,
    );
  }

  it("says why, and does not pretend the song is gone", async () => {
    stubDelete({
      code: "GENERATION_HAS_DERIVED_VERSIONS",
      message: "This version has derived versions. Delete those first.",
      derived_count: 2,
    });
    const onDeleted = vi.fn();
    renderActions(onDeleted);
    await attemptDelete();

    expect(
      await screen.findByText("Other versions were made from this song. Delete those first."),
    ).toBeInTheDocument();
    expect(onDeleted).not.toHaveBeenCalled();
    expect(screen.queryByText("Song deleted")).toBeNull();
  });

  it("keeps the generic message for a failure it cannot explain", async () => {
    stubDelete("SOMETHING_ELSE");
    renderActions(vi.fn());
    await attemptDelete();

    expect(await screen.findByText("Could not delete this song.")).toBeInTheDocument();
  });

  it("does not promise that derived versions are orphaned", async () => {
    stubDelete({ code: "GENERATION_HAS_DERIVED_VERSIONS", derived_count: 1 });
    renderActions(vi.fn());
    await userEvent.setup().click(screen.getByRole("button", { name: "Delete" }));

    const dialog = screen.getByRole("alertdialog");
    // The old copy said "Songs generated from it are kept", which
    // described the orphaning this phase removed.
    expect(dialog).not.toHaveTextContent("Songs generated from it are kept");
    expect(dialog).toHaveTextContent(/cannot be deleted until they are/);
  });
});
