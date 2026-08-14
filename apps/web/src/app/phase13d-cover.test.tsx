/**
 * Phase 13D-2 — Create Cover in the product UI.
 *
 * Two things are being defended. The control must send only product
 * vocabulary, since every engine setting it could leak is one calibration
 * showed to be either uncalibrated or inert. And the word "Remix" must not
 * appear anywhere in the normal feature UI: the engine does not preserve
 * the recording, so that name would be a claim the audio cannot support.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { CreateCover, STYLE_EXAMPLES } from "@/components/CreateCover";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { ToastProvider } from "@/components/ui/Toast";
import SongDetailPage from "@/app/song/[id]/page";
import { COVER_STRENGTHS } from "@/lib/api";
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
      if (String(url).includes("/cover")) {
        if (options.fail) return { ok: false, status: 409, json: async () => ({}) };
        return json({
          generation_id: "cover-1",
          status: "QUEUED",
          advisories: [],
          generation_group_id: "grp-c",
          generations: [{ generation_id: "cover-1", status: "QUEUED", seed: null }],
        });
      }
      if (String(url).includes("/lineage")) {
        return json({ generation_id: "gen-1", parent: null, children: options.children ?? [] });
      }
      if (String(url).includes("/v1/projects")) return json({ items: [] });
      return json(song());
    }),
  );
  return { calls };
}

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

// ── availability ──────────────────────────────────────────────────────

describe("availability", () => {
  it("is offered on a finished song with a master", () => {
    stubApi();
    renderApp(<CreateCover generation={song()} />);
    expect(screen.getByRole("button", { name: "Create cover" })).toBeInTheDocument();
  });

  it("is absent while the song is still generating", () => {
    stubApi();
    renderApp(<CreateCover generation={song({ status: "GENERATING" })} />);
    expect(screen.queryByRole("button", { name: "Create cover" })).toBeNull();
  });

  it("is absent on a failed song", () => {
    stubApi();
    renderApp(<CreateCover generation={song({ status: "FAILED" })} />);
    expect(screen.queryByRole("button", { name: "Create cover" })).toBeNull();
  });

  it("is absent without a master to build on", () => {
    stubApi();
    renderApp(<CreateCover generation={song({ audio_assets: [previewAsset()] })} />);
    expect(screen.queryByRole("button", { name: "Create cover" })).toBeNull();
  });
});

// ── the panel ─────────────────────────────────────────────────────────

describe("the cover panel", () => {
  async function open(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole("button", { name: "Create cover" }));
  }

  it("offers a style field and the two calibrated strengths", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    expect(screen.getByLabelText("Target style")).toBeInTheDocument();
    for (const option of COVER_STRENGTHS) {
      expect(screen.getByRole("radio", { name: option.label })).toBeInTheDocument();
    }
    // Exactly two: a third would be a setting nobody measured.
    expect(screen.getAllByRole("radio")).toHaveLength(2);
  });

  it("suggests styles without naming any artist", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    for (const example of STYLE_EXAMPLES) {
      expect(screen.getByRole("button", { name: example })).toBeInTheDocument();
    }
    expect(document.body.textContent).not.toMatch(/\bin the style of\b|\bsounds like\b/i);
  });

  it("fills the field from an example", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.click(screen.getByRole("button", { name: STYLE_EXAMPLES[1] }));
    expect(screen.getByLabelText("Target style")).toHaveValue(STYLE_EXAMPLES[1]);
  });

  it("does not promise the vocal or recording is preserved", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    const copy = document.body.textContent ?? "";
    expect(copy).toMatch(/original recording and vocal are not kept/i);
    for (const claim of [
      /preserves the (exact )?vocal/i,
      /keeps every melody note/i,
      /voice clon/i,
      /remix(es)? the original recording/i,
    ]) {
      expect(copy).not.toMatch(claim);
    }
  });
});

// ── submission ────────────────────────────────────────────────────────

describe("submission", () => {
  async function open(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole("button", { name: "Create cover" }));
  }

  it("posts the style and strength to the cover endpoint", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Target style"), "warm contemporary R&B");
    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/cover"));
      expect(call?.method).toBe("POST");
      expect(call?.url).toContain("/v1/generations/gen-1/cover");
      expect(call?.body).toEqual({ prompt: "warm contemporary R&B", strength: "subtle" });
    });
  });

  it("sends the chosen strength", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Target style"), "glossy synth pop");
    await user.click(screen.getByRole("radio", { name: "More transformed" }));
    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/cover"));
      expect(call?.body.strength).toBe("strong");
    });
  });

  it("never sends an engine control", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Target style"), "glossy synth pop");
    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    await waitFor(() => expect(calls.some((c) => c.url.includes("/cover"))).toBe(true));
    const body = calls.find((c) => c.url.includes("/cover"))!.body;
    for (const field of [
      "task_type",
      "audio_cover_strength",
      "cover_noise_strength",
      "src_audio",
      "thinking",
    ]) {
      expect(body).not.toHaveProperty(field);
    }
  });

  it("refuses an empty style instead of letting the server reject it", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    expect(await screen.findByRole("alert")).toHaveTextContent("Describe the style you want.");
    expect(calls.some((c) => c.url.includes("/cover"))).toBe(false);
  });

  it("hands the queued child to the caller", async () => {
    const user = userEvent.setup();
    stubApi();
    const onCovered = vi.fn();
    renderApp(<CreateCover generation={song()} onCovered={onCovered} />);
    await open(user);

    await user.type(screen.getByLabelText("Target style"), "glossy synth pop");
    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    await waitFor(() => expect(onCovered).toHaveBeenCalledWith("cover-1"));
  });

  it("reports a rejection through the existing feedback surface", async () => {
    const user = userEvent.setup();
    stubApi({ fail: true });
    renderApp(<CreateCover generation={song()} />);
    await open(user);

    await user.type(screen.getByLabelText("Target style"), "glossy synth pop");
    await user.click(screen.getAllByRole("button", { name: "Create cover" })[0]);

    expect(
      await screen.findByText("Could not create a cover of this song."),
    ).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/409|cover_strength/);
  });
});

// ── lineage ───────────────────────────────────────────────────────────

describe("lineage", () => {
  it("labels a cover truthfully", () => {
    const relation = describeRelation(
      generation({ parent_generation_id: "p", edit_kind: "COVER" }),
    );
    expect(relation?.kind).toBe("cover");
    expect(relation?.label).toBe("Cover");
    expect(relation?.detail).not.toMatch(/original recording is kept/i);
  });

  it("keeps the four relationships distinct", () => {
    const kinds = ["EXTEND", "REPLACE_RANGE", "COVER", null];
    const labels = kinds.map(
      (edit_kind) =>
        describeRelation(
          generation({
            parent_generation_id: "p",
            edit_kind,
            edit_start_seconds: 10,
            edit_end_seconds: 20,
          }),
        )!.label,
    );
    expect(new Set(labels).size).toBe(4);
  });

  it("never says remix or variation for any relationship", () => {
    for (const edit_kind of ["EXTEND", "REPLACE_RANGE", "COVER", null]) {
      const relation = describeRelation(
        generation({
          parent_generation_id: "p",
          edit_kind,
          edit_start_seconds: 10,
          edit_end_seconds: 20,
        }),
      )!;
      expect(`${relation.label} ${relation.detail}`.toLowerCase()).not.toMatch(
        /remix|variation/,
      );
    }
  });

  it("shows the cover label in a song's history", async () => {
    stubApi({
      children: [
        generation({
          id: "c1",
          title: "Synth pop take",
          parent_generation_id: "gen-1",
          edit_kind: "COVER",
          source_adherence: 1.0,
        }),
      ],
    });
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(await screen.findByText("Cover")).toBeInTheDocument();
  });
});

// ── the product never says remix ──────────────────────────────────────

describe("product vocabulary", () => {
  it("offers Create cover beside the other actions", async () => {
    stubApi();
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(screen.getByRole("button", { name: "Create cover" })).toBeInTheDocument();
  });

  it("never uses the word Remix on the song page", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    await user.click(screen.getByRole("button", { name: "Create cover" }));

    expect(document.body.textContent).not.toMatch(/remix/i);
  });
});
