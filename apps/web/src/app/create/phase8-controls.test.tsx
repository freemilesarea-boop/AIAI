/**
 * Phase 8 control surface, driven through the real page component.
 *
 * The load-bearing test here is
 * `"a form with no advanced controls touched submits Phase 7 fields"`:
 * the new UI must be additive. If a user ignores everything Phase 8
 * added, the request that leaves the browser has to describe the same
 * generation it described before.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { PlayerProvider } from "@/components/player/PlayerProvider";
import { ToastProvider } from "@/components/ui/Toast";
import CreatePage from "./page";

/** The page needs the providers the real layout mounts. */
function renderCreate() {
  return render(
    <PlayerProvider>
      <ToastProvider>
        <CreatePage />
      </ToastProvider>
    </PlayerProvider>,
  );
}

// The Create page navigates (it clears the ?from / ?duplicate parameter
// after applying a prefill), so the router has to exist in tests.
const routerReplace = vi.fn();
const searchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: routerReplace, refresh: vi.fn() }),
  useSearchParams: () => searchParams,
  usePathname: () => "/create",
}));


const GEN_ID = "d1d76e27-119a-41e8-a358-a492141efaba";

interface GenerationOverrides {
  status?: string;
  withMaster?: boolean;
  bpm?: number | null;
  key_scale?: string | null;
  time_signature?: string | null;
  parent_generation_id?: string | null;
  request_trace?: Record<string, unknown> | null;
}

function generationBody(overrides: GenerationOverrides = {}) {
  const { status = "COMPLETED", withMaster = true, ...rest } = overrides;
  return {
    id: GEN_ID,
    title: "Midnight Window",
    prompt: "Dreamy Korean indie pop",
    lyrics: "[Verse]\n오늘 밤 너를 생각해",
    vocal_gender: "female",
    duration_requested: 60,
    duration_actual: withMaster ? 60 : null,
    seed: 1234,
    language: "en",
    instrumental: false,
    bpm: null,
    key_scale: null,
    time_signature: null,
    parent_generation_id: null,
    variation_label: null,
    advisories: [],
    request_trace: null,
    status,
    provider: "ace_step",
    model_name: "acestep-v15-turbo",
    model_version: "1.5.0",
    created_at: "2026-08-11T12:00:00Z",
    started_at: null,
    completed_at: "2026-08-11T12:00:40Z",
    error_code: null,
    audio_assets: withMaster
      ? [
          {
            id: "a3203cdb-a4dd-40b3-829b-ceaf3b6e8fe4",
            asset_type: "MASTER",
            format: "wav",
            mime_type: "audio/wav",
            file_extension: "wav",
            sample_rate: 48000,
            bit_depth: 24,
            bitrate: null,
            channels: 2,
            duration: 60,
            storage_key: `audio/${GEN_ID}/master.wav`,
            sha256: "504aa20655af7f4756c604c071b5e6bdafb087d61c78b21d6b12a939ca653a31",
            file_size: 8640102,
            created_at: "2026-08-11T12:00:40Z",
          },
        ]
      : [],
    ...rest,
  };
}

interface Advisory {
  code: string;
  level: "info" | "warning";
  message: string;
  detail: Record<string, unknown>;
}

interface PreflightStub {
  advisories?: Advisory[];
  sections?: {
    kind: string | null;
    label: string;
    index: number | null;
    line_number: number;
    line_count: number;
    has_content: boolean;
    recognised: boolean;
  }[];
  estimated_syllables?: number;
}

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

function stubServer(options: { generation?: GenerationOverrides; preflight?: PreflightStub } = {}) {
  const preflightCalls: Record<string, unknown>[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "POST" && isPreflight(url)) {
      preflightCalls.push(jsonBodyOf(init));
      return jsonResponse({
        advisories: options.preflight?.advisories ?? [],
        sections: options.preflight?.sections ?? [],
        preamble_line_count: 0,
        estimated_syllables: options.preflight?.estimated_syllables ?? 0,
      });
    }
    if (init?.method === "POST") {
      return jsonResponse({
        generation_id: GEN_ID,
        status: "QUEUED",
        advisories: [],
        generation_group_id: "8b2f4a3e-5c6d-4e7f-8a9b-0c1d2e3f4a5b",
        generations: [{ generation_id: GEN_ID, status: "QUEUED", seed: null }],
      });
    }
    return jsonResponse(generationBody(options.generation));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, preflightCalls };
}

const LYRICS_TEXT = "[Verse]\n오늘 밤 너를 생각해";

async function fillValidForm(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Title"), "Midnight Window");
  await user.type(screen.getByLabelText("Music description"), "Dreamy Korean indie pop");
  await user.click(screen.getByLabelText("Lyrics"));
  await user.paste(LYRICS_TEXT);
}

async function submittedBody(fetchMock: ReturnType<typeof vi.fn>) {
  await waitFor(() => expect(fetchMock.mock.calls.some(isCreatePost)).toBe(true));
  const call = fetchMock.mock.calls.find(isCreatePost)!;
  return jsonBodyOf(call[1] as RequestInit);
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
  // jsdom has no layout engine; the page scrolls after "Generate again".
  window.scrollTo = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ── Advanced controls are visible, optional, and additive ─────────────

describe("advanced controls", () => {
  it("are present and clearly optional", async () => {
    stubServer();
    renderCreate();
    await switchToCustom();

    expect(screen.getByText(/Advanced controls/)).toBeInTheDocument();
    expect(screen.getByLabelText("BPM")).toBeInTheDocument();
    expect(screen.getByLabelText("Key / Scale")).toBeInTheDocument();
    expect(screen.getByLabelText("Time Signature")).toBeInTheDocument();
    expect(
      screen.getByText(/Leave any of these empty and the model chooses for you/),
    ).toBeInTheDocument();
  });

  it("default to unset", async () => {
    stubServer();
    renderCreate();
    await switchToCustom();

    expect(screen.getByLabelText("BPM")).toHaveValue(null);
    expect(screen.getByLabelText("Key / Scale")).toHaveValue("");
    expect(screen.getByLabelText("Time Signature")).toHaveValue("");
  });

  it("a form with no advanced controls touched submits Phase 7 fields", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    const body = await submittedBody(fetchMock);
    // The Phase 7 contract, unchanged.
    expect(body.title).toBe("Midnight Window");
    expect(body.prompt).toBe("Dreamy Korean indie pop");
    expect(body.lyrics).toBe(LYRICS_TEXT);
    expect(body.vocal_gender).toBe("female");
    expect(body.language).toBe("ko");
    expect(body.duration).toBe(30);
    // Untouched controls are explicitly "not specified" — never a
    // substituted default that would change what the engine does.
    expect(body.bpm).toBeNull();
    expect(body.key_scale).toBeNull();
    expect(body.time_signature).toBeNull();
    expect(body.parent_generation_id).toBeNull();
    expect(body).not.toHaveProperty("variation_label");
  });

  it("sends a chosen BPM", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.type(screen.getByLabelText("BPM"), "128");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect((await submittedBody(fetchMock)).bpm).toBe(128);
  });

  it("sends a chosen key/scale", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.selectOptions(screen.getByLabelText("Key / Scale"), "F# minor");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect((await submittedBody(fetchMock)).key_scale).toBe("F# minor");
  });

  it("sends a chosen time signature as the bare numerator", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.selectOptions(screen.getByLabelText("Time Signature"), "3");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect((await submittedBody(fetchMock)).time_signature).toBe("3");
  });

  it("sends all three together", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.type(screen.getByLabelText("BPM"), "92");
    await user.selectOptions(screen.getByLabelText("Key / Scale"), "Bb major");
    await user.selectOptions(screen.getByLabelText("Time Signature"), "6");
    await user.click(screen.getByRole("button", { name: "Create" }));

    const body = await submittedBody(fetchMock);
    expect(body.bpm).toBe(92);
    expect(body.key_scale).toBe("Bb major");
    expect(body.time_signature).toBe("6");
  });

  it("only offers key/scale values the engine accepts", async () => {
    stubServer();
    renderCreate();
    await switchToCustom();

    const select = screen.getByLabelText("Key / Scale");
    const values = within(select)
      .getAllByRole("option")
      .map((o) => (o as HTMLOptionElement).value);
    expect(values[0]).toBe(""); // "let the model decide"
    expect(values).toHaveLength(43); // 42 engine values + auto
    expect(values).toContain("C major");
    expect(values).not.toContain("C dorian");
    expect(values.some((v) => v.includes("♯"))).toBe(false);
  });

  it("only offers time signatures the engine accepts", async () => {
    stubServer();
    renderCreate();
    await switchToCustom();

    const values = within(screen.getByLabelText("Time Signature"))
      .getAllByRole("option")
      .map((o) => (o as HTMLOptionElement).value);
    expect(values).toEqual(["", "2", "3", "4", "6"]);
  });

  it("rejects an out-of-range BPM without submitting", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.type(screen.getByLabelText("BPM"), "900");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText("BPM must be between 30 and 300.")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(0);
  });

  it("clears the advanced controls on request", async () => {
    const user = userEvent.setup();
    stubServer();
    renderCreate();
    await switchToCustom();

    await user.type(screen.getByLabelText("BPM"), "128");
    await user.click(screen.getByRole("button", { name: "Clear advanced controls" }));

    expect(screen.getByLabelText("BPM")).toHaveValue(null);
  });
});

// ── Structure editor ──────────────────────────────────────────────────

describe("song structure editor", () => {
  it("offers section tags without requiring them", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer();
    renderCreate();
    await switchToCustom();

    expect(screen.getByRole("button", { name: "[Chorus]" })).toBeInTheDocument();

    // Plain, untagged lyrics remain perfectly valid.
    await user.type(screen.getByLabelText("Title"), "Plain");
    await user.type(screen.getByLabelText("Music description"), "Soft piano");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("just a line\nand another");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect((await submittedBody(fetchMock)).lyrics).toBe("just a line\nand another");
  });

  it("inserts a section tag at the cursor on an explicit click", async () => {
    const user = userEvent.setup();
    stubServer();
    renderCreate();
    await switchToCustom();

    await user.click(screen.getByRole("button", { name: "[Verse 1]" }));

    expect(screen.getByLabelText("Lyrics")).toHaveValue("[Verse 1]\n");
  });

  it("never rewrites lyrics on its own", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer({
      preflight: {
        advisories: [
          {
            code: "UNKNOWN_SECTION_TAG",
            level: "warning",
            message: "1 section tag(s) are not recognised: Drop.",
            detail: {},
          },
        ],
      },
    });
    renderCreate();
    await switchToCustom();

    const messy = "[Drop]\n  spaced out  \n\n\n[chorus]\nhook";
    await user.type(screen.getByLabelText("Title"), "Messy");
    await user.type(screen.getByLabelText("Music description"), "Bass music");
    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste(messy);

    // Advisory shown…
    expect(await screen.findByText(/not recognised/)).toBeInTheDocument();
    // …and the text is untouched, on screen and on the wire.
    expect(screen.getByLabelText("Lyrics")).toHaveValue(messy);
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect((await submittedBody(fetchMock)).lyrics).toBe(messy);
  });

  it("shows the parsed structure outline", async () => {
    const user = userEvent.setup();
    stubServer({
      preflight: {
        estimated_syllables: 12,
        sections: [
          {
            kind: "verse",
            label: "Verse 1",
            index: 1,
            line_number: 1,
            line_count: 2,
            has_content: true,
            recognised: true,
          },
          {
            kind: null,
            label: "Drop",
            index: null,
            line_number: 4,
            line_count: 1,
            has_content: true,
            recognised: false,
          },
        ],
      },
    });
    renderCreate();
    await switchToCustom();

    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("[Verse 1]\na\nb\n[Drop]\nc");

    const outline = await screen.findByRole("region", { name: "Structure" });
    expect(within(outline).getByText("[Verse 1]")).toBeInTheDocument();
    expect(within(outline).getByText("[Drop]")).toBeInTheDocument();
    expect(within(outline).getByText("≈12 syllables")).toBeInTheDocument();
  });
});

// ── Advisories ────────────────────────────────────────────────────────

describe("pre-flight advisories", () => {
  it("are fetched from the backend rather than recomputed in the browser", async () => {
    const user = userEvent.setup();
    const { preflightCalls } = stubServer();
    renderCreate();
    await switchToCustom();

    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste(LYRICS_TEXT);

    await waitFor(() => expect(preflightCalls.length).toBeGreaterThan(0));
    expect(preflightCalls[0]).toMatchObject({
      lyrics: LYRICS_TEXT,
      duration: 30,
      language: "ko",
      instrumental: false,
    });
  });

  it("are displayed with a clear non-blocking framing", async () => {
    const user = userEvent.setup();
    stubServer({
      preflight: {
        advisories: [
          {
            code: "LYRICS_DENSE_FOR_DURATION",
            level: "warning",
            message: "About 300 syllables in 30s is dense (13.3/s).",
            detail: {},
          },
        ],
      },
    });
    renderCreate();
    await switchToCustom();

    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste("가".repeat(300));

    expect(await screen.findByText(/is dense/)).toBeInTheDocument();
    expect(screen.getByText("Advice only — these never block generation.")).toBeInTheDocument();
  });

  it("never block submission", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer({
      preflight: {
        advisories: [
          {
            code: "LYRICS_DENSE_FOR_DURATION",
            level: "warning",
            message: "About 300 syllables in 30s is dense (13.3/s).",
            detail: {},
          },
          {
            code: "KOREAN_LYRICS_LANGUAGE_MISMATCH",
            level: "warning",
            message: "The lyrics are 100% Korean but the vocal language is 'en'.",
            detail: {},
          },
        ],
      },
    });
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    expect(await screen.findByText(/is dense/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(1));
  });

  it("stay silent when a pre-flight request fails", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (init?.method === "POST" && isPreflight(url)) {
          return { ok: false, status: 500, json: async () => ({}) };
        }
        if (init?.method === "POST") {
          return jsonResponse({
        generation_id: GEN_ID,
        status: "QUEUED",
        advisories: [],
        generation_group_id: "8b2f4a3e-5c6d-4e7f-8a9b-0c1d2e3f4a5b",
        generations: [{ generation_id: GEN_ID, status: "QUEUED", seed: null }],
      });
        }
        return jsonResponse(generationBody());
      }),
    );
    renderCreate();
    await switchToCustom();

    await user.click(screen.getByLabelText("Lyrics"));
    await user.paste(LYRICS_TEXT);

    // A failed diagnostic must not become a user-facing error.
    await waitFor(() => {
      expect(screen.queryByText("Advice only — these never block generation.")).toBeNull();
    });
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

// ── Generate again ────────────────────────────────────────────────────

describe("generate again", () => {
  async function completeAGeneration(
    user: ReturnType<typeof userEvent.setup>,
    generation: GenerationOverrides = {},
  ) {
    const stub = stubServer({ generation });
    renderCreate();
    await switchToCustom();
    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Create" }));
    await screen.findByRole("heading", { name: "Midnight Window" });
    return stub;
  }

  it("offers the action on a completed track", async () => {
    const user = userEvent.setup();
    await completeAGeneration(user);
    expect(screen.getByRole("button", { name: "Generate again" })).toBeInTheDocument();
  });

  it("carries the previous generation's settings into the new draft", async () => {
    const user = userEvent.setup();
    await completeAGeneration(user, {
      bpm: 128,
      key_scale: "F# minor",
      time_signature: "3",
    });

    await user.click(screen.getByRole("button", { name: "Generate again" }));

    expect(screen.getByLabelText("Title")).toHaveValue("Midnight Window");
    expect(screen.getByLabelText("Music description")).toHaveValue("Dreamy Korean indie pop");
    expect(screen.getByLabelText("Lyrics")).toHaveValue("[Verse]\n오늘 밤 너를 생각해");
    expect(screen.getByLabelText("Duration")).toHaveValue("60");
    expect(screen.getByLabelText("Language")).toHaveValue("en");
    expect(screen.getByLabelText("BPM")).toHaveValue(128);
    expect(screen.getByLabelText("Key / Scale")).toHaveValue("F# minor");
    expect(screen.getByLabelText("Time Signature")).toHaveValue("3");
  });

  it("shows which track the draft is based on", async () => {
    const user = userEvent.setup();
    await completeAGeneration(user);
    await user.click(screen.getByRole("button", { name: "Generate again" }));

    expect(screen.getByText(/Based on/)).toBeInTheDocument();
    expect(screen.getByText(/adjust anything before generating/)).toBeInTheDocument();
  });

  it("submits the previous generation as the parent", async () => {
    const user = userEvent.setup();
    const { fetchMock } = await completeAGeneration(user);

    await user.click(screen.getByRole("button", { name: "Generate again" }));
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(2));
    const second = fetchMock.mock.calls.filter(isCreatePost)[1];
    expect(jsonBodyOf(second[1] as RequestInit).parent_generation_id).toBe(GEN_ID);
  });

  it("lets the user change the settings before submitting", async () => {
    const user = userEvent.setup();
    const { fetchMock } = await completeAGeneration(user, { bpm: 128 });

    await user.click(screen.getByRole("button", { name: "Generate again" }));
    await user.clear(screen.getByLabelText("BPM"));
    await user.type(screen.getByLabelText("BPM"), "90");
    await user.clear(screen.getByLabelText("Title"));
    await user.type(screen.getByLabelText("Title"), "Second Take");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(2));
    const body = jsonBodyOf(fetchMock.mock.calls.filter(isCreatePost)[1][1] as RequestInit);
    expect(body.bpm).toBe(90);
    expect(body.title).toBe("Second Take");
    expect(body.parent_generation_id).toBe(GEN_ID);
  });

  it("can be abandoned, dropping the lineage", async () => {
    const user = userEvent.setup();
    const { fetchMock } = await completeAGeneration(user);

    await user.click(screen.getByRole("button", { name: "Generate again" }));
    await user.click(screen.getByRole("button", { name: "Start fresh" }));

    expect(screen.queryByText(/Based on/)).toBeNull();
    expect(screen.getByLabelText("Title")).toHaveValue("");

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(2));
    const body = jsonBodyOf(fetchMock.mock.calls.filter(isCreatePost)[1][1] as RequestInit);
    expect(body.parent_generation_id).toBeNull();
  });
});

// ── Result card ───────────────────────────────────────────────────────

describe("completed track details", () => {
  it("reports the controls that were used", async () => {
    const user = userEvent.setup();
    stubServer({ generation: { bpm: 128, key_scale: "F# minor", time_signature: "3" } });
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Create" }));
    await screen.findByRole("heading", { name: "Midnight Window" });

    // Scoped to the result card: "F# minor" is also a select option.
    const card = screen.getByRole("group", { name: "Midnight Window" });
    expect(within(card).getByText("128")).toBeInTheDocument();
    expect(within(card).getByText("F# minor")).toBeInTheDocument();
    expect(within(card).getByText("3")).toBeInTheDocument();
  });

  it("still offers playback and download", async () => {
    const user = userEvent.setup();
    stubServer();
    renderCreate();
    await switchToCustom();

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Create" }));
    await screen.findByRole("heading", { name: "Midnight Window" });

    expect(screen.getByRole("link", { name: "Download WAV" })).toHaveAttribute(
      "href",
      expect.stringContaining(`/v1/generations/${GEN_ID}/audio?asset=master&download=true`),
    );
  });
});
