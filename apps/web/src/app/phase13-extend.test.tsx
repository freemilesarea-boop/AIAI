/**
 * Phase 13B — the Extend control.
 *
 * The UI's job here is small and mostly about restraint: offer Extend
 * only when it can actually work, never offer a length the backend would
 * reject, and hand the result to the workflow that already exists. None
 * of the engine's editing vocabulary belongs on this side of the wire,
 * so a test checks that too.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ExtendSong, availableExtensions } from "@/components/ExtendSong";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { ToastProvider } from "@/components/ui/Toast";
import SongDetailPage from "@/app/song/[id]/page";
import { MAX_SONG_SECONDS } from "@/lib/api";
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

function stubApi(overrides: { failExtend?: boolean } = {}) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    calls.push({
      url: String(url),
      method,
      body: init?.body ? JSON.parse(String(init.body)) : {},
    });
    if (String(url).includes("/extend")) {
      if (overrides.failExtend) return { ok: false, status: 409, json: async () => ({}) };
      return json({
        generation_id: "child-1",
        status: "QUEUED",
        advisories: [],
        generation_group_id: "grp-2",
        generations: [{ generation_id: "child-1", status: "QUEUED", seed: null }],
      });
    }
    if (String(url).includes("/lineage")) {
      return json({ generation_id: "gen-1", parent: null, children: [] });
    }
    if (String(url).includes("/v1/projects")) return json({ items: [] });
    return json(generation());
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls };
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

// ── when Extend is offered at all ─────────────────────────────────────

describe("availability", () => {
  it("offers Extend on a finished song with a master", () => {
    stubApi();
    renderApp(<ExtendSong generation={generation()} />);
    expect(screen.getByRole("button", { name: "Extend" })).toBeInTheDocument();
  });

  it("is absent while a song is still generating", () => {
    stubApi();
    renderApp(<ExtendSong generation={generation({ status: "GENERATING" })} />);
    expect(screen.queryByRole("button", { name: "Extend" })).toBeNull();
  });

  it("is absent on a failed song", () => {
    stubApi();
    renderApp(<ExtendSong generation={generation({ status: "FAILED" })} />);
    expect(screen.queryByRole("button", { name: "Extend" })).toBeNull();
  });

  it("is absent when there is no master to build on", () => {
    stubApi();
    renderApp(<ExtendSong generation={generation({ audio_assets: [previewAsset()] })} />);
    expect(screen.queryByRole("button", { name: "Extend" })).toBeNull();
  });

  it("is absent when no supported length still fits", () => {
    stubApi();
    const nearlyMax = generation({
      audio_assets: [masterAsset({ duration: MAX_SONG_SECONDS - 2 })],
    });
    renderApp(<ExtendSong generation={nearlyMax} />);
    expect(screen.queryByRole("button", { name: "Extend" })).toBeNull();
  });
});

// ── which lengths are offered ─────────────────────────────────────────

describe("length choices", () => {
  it("offers all three for a short song", () => {
    expect(availableExtensions(generation())).toEqual([15, 30, 60]);
  });

  it("drops lengths that would exceed the maximum", () => {
    const long = generation({
      audio_assets: [masterAsset({ duration: MAX_SONG_SECONDS - 20 })],
    });
    // +30 and +60 would overshoot; only +15 survives.
    expect(availableExtensions(long)).toEqual([15]);
  });

  it("measures from the master, not the requested duration", () => {
    const drifted = generation({
      duration_requested: 30,
      audio_assets: [masterAsset({ duration: MAX_SONG_SECONDS - 10 })],
    });
    expect(availableExtensions(drifted)).toEqual([]);
  });

  it("renders only the surviving choices", async () => {
    const user = userEvent.setup();
    stubApi();
    const long = generation({
      audio_assets: [masterAsset({ duration: MAX_SONG_SECONDS - 20 })],
    });
    renderApp(<ExtendSong generation={long} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));

    expect(screen.getByRole("button", { name: "+15s" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "+30s" })).toBeNull();
    expect(screen.queryByRole("button", { name: "+60s" })).toBeNull();
  });
});

// ── submitting ────────────────────────────────────────────────────────

describe("submission", () => {
  it("opens a control with the available lengths", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));

    expect(screen.getByText("Extend this song")).toBeInTheDocument();
    for (const label of ["+15s", "+30s", "+60s"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("posts the chosen length to the extend endpoint", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "+30s" }));

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/extend"));
      expect(call?.method).toBe("POST");
      expect(call?.url).toContain("/v1/generations/gen-1/extend");
      expect(call?.body).toEqual({ seconds: 30 });
    });
  });

  it("sends no engine vocabulary", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "+15s" }));

    await waitFor(() => expect(calls.some((c) => c.url.includes("/extend"))).toBe(true));
    const body = calls.find((c) => c.url.includes("/extend"))!.body;
    for (const field of ["task_type", "repainting_start", "repainting_end", "src_audio"]) {
      expect(body).not.toHaveProperty(field);
    }
  });

  it("hands the queued child to the caller", async () => {
    const user = userEvent.setup();
    stubApi();
    const onExtended = vi.fn();
    renderApp(<ExtendSong generation={generation()} onExtended={onExtended} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "+15s" }));

    await waitFor(() => expect(onExtended).toHaveBeenCalledWith("child-1"));
  });

  it("confirms through the existing toast surface", async () => {
    const user = userEvent.setup();
    stubApi();
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "+15s" }));

    expect(await screen.findByText(/Extending by 15s/)).toBeInTheDocument();
  });

  it("reports a rejection without a stack trace", async () => {
    const user = userEvent.setup();
    stubApi({ failExtend: true });
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "+15s" }));

    expect(await screen.findByText("Could not extend this song.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/409|Error:|fetch/);
  });

  it("can be cancelled without submitting", async () => {
    const user = userEvent.setup();
    const { calls } = stubApi();
    renderApp(<ExtendSong generation={generation()} />);

    await user.click(screen.getByRole("button", { name: "Extend" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.getByRole("button", { name: "Extend" })).toBeInTheDocument();
    expect(calls.some((c) => c.url.includes("/extend"))).toBe(false);
  });
});

// ── on the song page ──────────────────────────────────────────────────

describe("song detail integration", () => {
  it("shows Extend beside the other actions on a ready song", async () => {
    stubApi();
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    expect(screen.getByRole("button", { name: "Extend" })).toBeInTheDocument();
  });

  it("describes lineage without claiming audio is never reused", async () => {
    stubApi();
    renderApp(<SongDetailPage />);
    await screen.findByRole("heading", { name: "Midnight Window" });
    // Phase 12's copy said "No audio was reused", which extension makes
    // false. The page must not assert it any more.
    expect(screen.queryByText(/No audio was reused/)).toBeNull();
  });
});
