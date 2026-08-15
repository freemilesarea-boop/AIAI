/**
 * Phase 13C — the Replace-section control and truthful lineage labels.
 *
 * Two things are being defended. The control must refuse a range the
 * backend would reject, rather than sending it and surfacing a 422. And
 * the lineage must say which of three different things happened to a
 * song's audio, because "variation" for all of them — or "remix" for any
 * of them — would describe work the engine did not do.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  ReplaceSection,
  formatClock,
  parseTimeInput,
  validateRange,
} from "@/components/ReplaceSection";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { ToastProvider } from "@/components/ui/Toast";
import SongDetailPage from "@/app/song/[id]/page";
import { describeRelation } from "@/lib/lineage";
import { generation, masterAsset, previewAsset } from "@/test/factories";

const routerPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), refresh: vi.fn() }),
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

function json(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown>;
}

function stubApi(options: { fail?: boolean; children?: unknown[] } = {}) {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      calls.push({
        url: String(url),
        method,
        body: init?.body ? JSON.parse(String(init.body)) : {},
      });
      if (String(url).includes("/replace-range")) {
        if (options.fail) return { ok: false, status: 422, json: async () => ({}) };
        return json({
          generation_id: "child-9",
          status: "QUEUED",
          advisories: [],
          generation_group_id: "grp-9",
          generations: [{ generation_id: "child-9", status: "QUEUED", seed: null }],
        });
      }
      if (String(url).includes("/lineage")) {
        // Mirrors the real response since Phase 17: the tree is what the
        // UI reads, and `parent`/`children` are kept for older callers.
        const kids = (options.children ?? []) as Record<string, unknown>[];
        const toNode = (g: Record<string, unknown>) => ({
          id: g.id,
          parent_generation_id: g.parent_generation_id ?? null,
          title: g.title,
          status: g.status ?? "COMPLETED",
          operation:
            g.parent_generation_id == null
              ? "ORIGINAL"
              : g.edit_kind === "EXTEND"
                ? "EXTEND"
                : g.edit_kind === "REPLACE_RANGE"
                  ? "REPLACE_SECTION"
                  : g.edit_kind === "COVER"
                    ? "COVER"
                    : "GENERATE_AGAIN",
          created_at: g.created_at ?? "2026-08-15T12:00:00Z",
          duration_actual: g.duration_actual ?? null,
          cover_art_url: null,
          edit_start_seconds: g.edit_start_seconds ?? null,
          edit_end_seconds: g.edit_end_seconds ?? null,
        });
        return json({
          generation_id: "gen-1",
          parent: null,
          children: kids,
          root_generation_id: "gen-1",
          current_generation_id: "gen-1",
          nodes: [
            toNode({ id: "gen-1", title: "Midnight Window", parent_generation_id: null }),
            ...kids.map(toNode),
          ],
        });
      }
      if (String(url).includes("/v1/projects")) return json({ items: [] });
      return json(generation({ audio_assets: [masterAsset({ duration: 30 }), previewAsset()] }));
    }),
  );
  return { calls };
}

/** A 30-second song — long enough for a comfortable interior range. */
function song(overrides = {}) {
  return generation({
    duration_actual: 30,
    audio_assets: [masterAsset({ duration: 30 }), previewAsset({ duration: 30 })],
    ...overrides,
  });
}

beforeEach(() => {
  routerPush.mockClear();
  window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  window.HTMLMediaElement.prototype.pause = vi.fn();
  window.HTMLMediaElement.prototype.load = vi.fn();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ── time parsing ──────────────────────────────────────────────────────

describe("time input", () => {
  it("reads m:ss", () => {
    expect(parseTimeInput("1:05")).toBe(65);
    expect(parseTimeInput("0:10")).toBe(10);
  });

  it("reads plain seconds, because people type those too", () => {
    expect(parseTimeInput("45")).toBe(45);
    expect(parseTimeInput("12.5")).toBe(12.5);
  });

  it("rejects nonsense rather than guessing", () => {
    for (const bad of ["", "abc", "1:75", "-5", "1:2:3"]) {
      expect(parseTimeInput(bad)).toBeNull();
    }
  });

  it("formats as a player would", () => {
    expect(formatClock(65)).toBe("1:05");
    expect(formatClock(9)).toBe("0:09");
  });
});

// ── range validation, mirroring the server ────────────────────────────

describe("range validation", () => {
  it("accepts a comfortable interior range", () => {
    expect(validateRange(10, 20, 30)).toBeNull();
  });

  it("requires both times", () => {
    expect(validateRange(null, 20, 30)).toMatch(/Enter a start and end/);
  });

  it("rejects an end at or before the start", () => {
    expect(validateRange(20, 20, 30)).toMatch(/End must come after start/);
    expect(validateRange(20, 10, 30)).toMatch(/End must come after start/);
  });

  it("rejects a range past the end of the song", () => {
    expect(validateRange(20, 40, 30)).toMatch(/past the song/);
  });

  it("rejects a span shorter than the engine's crossfade", () => {
    expect(validateRange(10, 10.5, 30)).toMatch(/at least 1 second/);
  });

  it("rejects replacing so much that nothing is preserved", () => {
    expect(validateRange(0, 30, 30)).toMatch(/Leave at least/);
  });
});

// ── availability ──────────────────────────────────────────────────────

describe("availability", () => {
  it("is offered on a finished song with a master", () => {
    stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    expect(screen.getByRole("button", { name: "Replace section" })).toBeInTheDocument();
  });

  it("is absent while the song is still generating", () => {
    stubApi();
    renderApp(<ReplaceSection generation={song({ status: "GENERATING" })} />);
    expect(screen.queryByRole("button", { name: "Replace section" })).toBeNull();
  });

  it("is absent without a master", () => {
    stubApi();
    renderApp(<ReplaceSection generation={song({ audio_assets: [previewAsset()] })} />);
    expect(screen.queryByRole("button", { name: "Replace section" })).toBeNull();
  });

  it("is absent when the song is too short to hold a replaced second", () => {
    stubApi();
    const tiny = song({ audio_assets: [masterAsset({ duration: 1.5 })] });
    renderApp(<ReplaceSection generation={tiny} />);
    expect(screen.queryByRole("button", { name: "Replace section" })).toBeNull();
  });
});

// ── submitting ────────────────────────────────────────────────────────

describe("submission", () => {
  async function open(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole("button", { name: "Replace section" }));
  }

  it("shows the song length so the times mean something", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);
    expect(screen.getByText(/0:30 long/)).toBeInTheDocument();
  });

  it("posts the chosen range", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Start"), "0:10");
    await user.type(screen.getByLabelText("End"), "0:20");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/replace-range"));
      expect(call?.method).toBe("POST");
      expect(call?.body).toEqual({ start_seconds: 10, end_seconds: 20 });
    });
  });

  it("sends an optional description when given", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Start"), "10");
    await user.type(screen.getByLabelText("End"), "20");
    await user.type(screen.getByLabelText(/New description/), "sparse piano");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/replace-range"));
      expect(call?.body.prompt).toBe("sparse piano");
    });
  });

  it("sends no engine vocabulary", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);
    await user.type(screen.getByLabelText("Start"), "10");
    await user.type(screen.getByLabelText("End"), "20");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    await waitFor(() =>
      expect(calls.some((c) => c.url.includes("/replace-range"))).toBe(true),
    );
    const body = calls.find((c) => c.url.includes("/replace-range"))!.body;
    for (const field of ["task_type", "repainting_start", "repaint_mode", "repaint_strength"]) {
      expect(body).not.toHaveProperty(field);
    }
  });

  it("refuses an invalid range instead of letting the server reject it", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Start"), "20");
    await user.type(screen.getByLabelText("End"), "10");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    expect(await screen.findByRole("alert")).toHaveTextContent("End must come after start.");
    expect(calls.some((c) => c.url.includes("/replace-range"))).toBe(false);
  });

  it("fills a quick range without inventing section names", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);

    // Time ranges, labelled as time ranges.
    expect(screen.queryByRole("button", { name: /Chorus|Verse|Bridge/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Last 10 seconds" }));

    expect(screen.getByLabelText("Start")).toHaveValue("0:20");
    expect(screen.getByLabelText("End")).toHaveValue("0:30");
  });

  it("hands the queued child to the caller", async () => {
    const user = userEvent.setup();
    stubApi();
    const onReplaced = vi.fn();
    renderApp(<ReplaceSection generation={song()} onReplaced={onReplaced} />);
    await open(user);
    await user.type(screen.getByLabelText("Start"), "10");
    await user.type(screen.getByLabelText("End"), "20");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    await waitFor(() => expect(onReplaced).toHaveBeenCalledWith("child-9"));
  });

  it("reports a rejection through the existing toast, without internals", async () => {
    const user = userEvent.setup();
    stubApi({ fail: true });
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);
    await user.type(screen.getByLabelText("Start"), "10");
    await user.type(screen.getByLabelText("End"), "20");
    await user.click(screen.getAllByRole("button", { name: "Replace section" })[0]);

    expect(await screen.findByText("Could not replace that section.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/422|repaint/);
  });

  it("says lyrics are not time-aligned rather than implying they are", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<ReplaceSection generation={song()} />);
    await open(user);
    expect(screen.getByText(/does not yet know which words fall at which time/)).toBeInTheDocument();
  });
});

// ── lineage vocabulary ────────────────────────────────────────────────

describe("lineage labels", () => {
  it("names an extension by how much was added", () => {
    const relation = describeRelation(
      generation({ parent_generation_id: "p", edit_kind: "EXTEND",
                   edit_start_seconds: 30, edit_end_seconds: 45 }),
    );
    expect(relation?.kind).toBe("extended");
    expect(relation?.label).toBe("Extended +15s");
  });

  it("names a replacement by the span it replaced", () => {
    const relation = describeRelation(
      generation({ parent_generation_id: "p", edit_kind: "REPLACE_RANGE",
                   edit_start_seconds: 45, edit_end_seconds: 60 }),
    );
    expect(relation?.kind).toBe("replaced");
    expect(relation?.label).toBe("Replaced 0:45–1:00");
  });

  it("distinguishes a plain re-generation, which reuses no audio", () => {
    const relation = describeRelation(
      generation({ parent_generation_id: "p", edit_kind: null }),
    );
    expect(relation?.kind).toBe("generated-again");
    expect(relation?.detail).toMatch(/No audio was reused/);
  });

  it("says nothing about a song with no parent", () => {
    expect(describeRelation(generation())).toBeNull();
  });

  it("never calls any of them a remix or a variation", () => {
    const kinds = ["EXTEND", "REPLACE_RANGE", null];
    for (const edit_kind of kinds) {
      const relation = describeRelation(
        generation({ parent_generation_id: "p", edit_kind,
                     edit_start_seconds: 10, edit_end_seconds: 20 }),
      );
      expect(relation!.label.toLowerCase()).not.toMatch(/remix|variation/);
      expect(relation!.detail.toLowerCase()).not.toMatch(/remix/);
    }
  });

  it("labels each child in the song's history", async () => {
    stubApi({
      children: [
        generation({ id: "c1", title: "Extended take", parent_generation_id: "gen-1",
                     edit_kind: "EXTEND", edit_start_seconds: 30, edit_end_seconds: 45 }),
        generation({ id: "c2", title: "Patched take", parent_generation_id: "gen-1",
                     edit_kind: "REPLACE_RANGE", edit_start_seconds: 10, edit_end_seconds: 20 }),
      ],
    });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(await screen.findByText("Extended +15s")).toBeInTheDocument();
    expect(screen.getByText("Replaced 0:10–0:20")).toBeInTheDocument();
  });
});

// ── on the song page ──────────────────────────────────────────────────

describe("song detail integration", () => {
  it("offers Replace section beside Extend", async () => {
    stubApi();
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(screen.getByRole("button", { name: "Replace section" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Extend" })).toBeInTheDocument();
  });
});
