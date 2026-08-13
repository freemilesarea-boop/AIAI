/**
 * Generation workspace behaviour, driven through the real page component.
 *
 * `fetch` is stubbed at the network boundary so the API client, polling
 * hook, and UI all run for real — only the server is simulated.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import CreatePage from "./page";

const GEN_ID = "d1d76e27-119a-41e8-a358-a492141efaba";

interface StubGeneration {
  status: string;
  error_code?: string | null;
  withMaster?: boolean;
}

function generationBody({ status, error_code = null, withMaster = false }: StubGeneration) {
  return {
    id: GEN_ID,
    title: "Midnight Window",
    prompt: "Dreamy Korean indie pop",
    lyrics: "[Verse]\n오늘 밤 너를 생각해",
    vocal_gender: "female",
    duration_requested: 30,
    duration_actual: withMaster ? 30 : null,
    seed: 1234,
    language: "ko",
    instrumental: false,
    bpm: null,
    key_scale: null,
    time_signature: null,
    parent_generation_id: null,
    variation_label: null,
    advisories: [],
    request_trace: null,
    status,
    provider: withMaster ? "ace_step" : null,
    model_name: withMaster ? "acestep-v15-turbo" : null,
    model_version: withMaster ? "1.5.0" : null,
    created_at: "2026-08-11T12:00:00Z",
    started_at: null,
    completed_at: withMaster ? "2026-08-11T12:00:40Z" : null,
    error_code,
    error_message: null,
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
            duration: 30,
            storage_key: `audio/${GEN_ID}/master.wav`,
            sha256: "504aa20655af7f4756c604c071b5e6bdafb087d61c78b21d6b12a939ca653a31",
            file_size: 8640102,
            created_at: "2026-08-11T12:00:40Z",
          },
          {
            id: "b1119a02-6f2a-4f0f-9a4c-2b8b0b7a1d55",
            asset_type: "PREVIEW",
            format: "mp3",
            mime_type: "audio/mpeg",
            file_extension: "mp3",
            sample_rate: 48000,
            bit_depth: null,
            bitrate: 320000,
            channels: 2,
            duration: 30,
            storage_key: `audio/${GEN_ID}/preview.mp3`,
            sha256: "2856229138ab61096e6cd1c6b6befcd67fbb020a83702373b2bbc0517789ebae",
            file_size: 1200480,
            created_at: "2026-08-11T12:00:41Z",
          },
        ]
      : [],
  };
}

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

/** Empty advisory result — the editor's pre-flight probe. */
function preflightBody() {
  return { advisories: [], sections: [], preamble_line_count: 0, estimated_syllables: 0 };
}

function isPreflight(url: unknown): boolean {
  return String(url).includes("/preflight");
}

/**
 * A POST that creates a generation.
 *
 * The lyrics editor also POSTs to `/v1/generations/preflight` for
 * advisories, so "did we submit?" assertions must exclude it — a
 * pre-flight probe is not a submission.
 */
function isCreatePost(call: readonly unknown[]): boolean {
  const [url, init] = call as [string, RequestInit | undefined];
  return init?.method === "POST" && !isPreflight(url);
}

/** `RequestInit.headers` is a union; the client always sends a record. */
function headerOf(init: RequestInit | undefined, name: string): string | undefined {
  return (init?.headers as Record<string, string> | undefined)?.[name];
}

function jsonBodyOf(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

/** Sequences GET /v1/generations/{id} responses; POST always succeeds. */
function stubServer(statuses: StubGeneration[]) {
  const getCalls: string[] = [];
  let index = 0;
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "POST" && isPreflight(url)) {
      return jsonResponse(preflightBody());
    }
    if (init?.method === "POST") {
      return jsonResponse({ generation_id: GEN_ID, status: "QUEUED", advisories: [] });
    }
    getCalls.push(url);
    const spec = statuses[Math.min(index, statuses.length - 1)];
    index += 1;
    return jsonResponse(generationBody(spec));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, getCalls };
}

const LYRICS_TEXT = "[Verse]\n오늘 밤 너를 생각해";

async function fillValidForm(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Title"), "Midnight Window");
  await user.type(screen.getByLabelText("Music description"), "Dreamy Korean indie pop");
  // Pasted rather than typed: userEvent reads "[" as key-descriptor
  // syntax, and pasting is how lyrics realistically arrive anyway.
  await user.click(screen.getByLabelText("Lyrics"));
  await user.paste(LYRICS_TEXT);
}

beforeEach(() => {
  window.localStorage.clear();
  vi.useRealTimers();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("form validation", () => {
  it("blocks submission and reports missing fields", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("Add a title for your track.")).toBeInTheDocument();
    expect(screen.getByText("Describe the music you want.")).toBeInTheDocument();
    // Nothing was sent to the backend.
    expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(0);
  });

  it("requires lyrics unless the track is instrumental", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await user.type(screen.getByLabelText("Title"), "T");
    await user.type(screen.getByLabelText("Music description"), "P");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(
      await screen.findByText("Add lyrics, or switch the vocal to Instrumental."),
    ).toBeInTheDocument();
  });

  it("disables the lyrics field for instrumental tracks", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await user.selectOptions(screen.getByLabelText("Vocal"), "instrumental");

    expect(screen.getByLabelText("Lyrics")).toBeDisabled();
    expect(screen.getByText("Instrumental selected — lyrics are not used.")).toBeInTheDocument();
  });

  it("marks invalid fields with aria-invalid", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByLabelText("Title")).toHaveAttribute("aria-invalid", "true");
  });
});

describe("submission", () => {
  it("posts to the LUBER API with a fresh Idempotency-Key and Korean lyrics intact", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(fetchMock.mock.calls.some(isCreatePost)).toBe(true);
    });
    const [url, init] = fetchMock.mock.calls.find(isCreatePost)!;
    expect(url).toContain("/v1/generations");
    expect(headerOf(init, "Idempotency-Key")).toMatch(/.+/);
    const payload = jsonBodyOf(init);
    // Korean, the section tag, and the line break all survive verbatim.
    expect(payload.lyrics).toBe(LYRICS_TEXT);
    expect(payload.vocal_gender).toBe("female");
    expect(payload.language).toBe("ko");
    expect(payload.duration).toBe(30);
    // Never a direct model-runtime call.
    expect(url).not.toContain("8001");
  });

  it("uses a different Idempotency-Key for each generation", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByRole("heading", { name: "Midnight Window" });
    await user.click(screen.getByRole("button", { name: "Create another" }));

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      const posts = fetchMock.mock.calls.filter(isCreatePost);
      expect(posts.length).toBe(2);
      expect(headerOf(posts[0][1], "Idempotency-Key")).not.toBe(
        headerOf(posts[1][1], "Idempotency-Key"),
      );
    });
  });

  it("prevents duplicate submission while a generation is in flight", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer([{ status: "GENERATING" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    const button = screen.getByRole("button", { name: "Generate" });
    await user.click(button);

    await waitFor(() => expect(screen.getByRole("button", { name: "Generating…" })).toBeDisabled());
    await user.click(screen.getByRole("button", { name: "Generating…" }));

    expect(fetchMock.mock.calls.filter(isCreatePost)).toHaveLength(1);
  });
});

describe("status rendering", () => {
  it("shows QUEUED as user-facing language", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "QUEUED" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Preparing generation");
  });

  it("shows GENERATING and an elapsed timer, with no fake percentage", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "GENERATING" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Creating your music"),
    );
    expect(screen.getByText(/Elapsed/)).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
});

describe("completed result", () => {
  it("renders the player and download pointing at the LUBER audio endpoint", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("Track ready")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Midnight Window" })).toBeInTheDocument();

    // Playback uses the compressed preview; the master stays downloadable.
    const player = screen.getByLabelText("Preview player for Midnight Window");
    expect(player).toHaveAttribute("src", expect.stringContaining(`/v1/generations/${GEN_ID}/audio`));
    expect(player).toHaveAttribute("src", expect.stringContaining("asset=preview"));
    expect(player).toHaveAttribute("controls");

    const wav = screen.getByRole("link", { name: "Download WAV" });
    expect(wav).toHaveAttribute("href", expect.stringContaining("asset=master"));
    expect(wav).toHaveAttribute("href", expect.stringContaining("download=true"));
    expect(wav).toHaveAttribute("download");

    const mp3 = screen.getByRole("link", { name: "Download MP3" });
    expect(mp3).toHaveAttribute("href", expect.stringContaining("asset=preview"));
    expect(mp3).toHaveAttribute("href", expect.stringContaining("download=true"));
    expect(mp3).toHaveAttribute("download");
  });

  it("shows the production master and preview formats", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Track ready");

    expect(screen.getByText(/WAV · 48 kHz · 24-bit · Stereo/)).toBeInTheDocument();
    expect(screen.getByText(/MP3 · 320 kbps/)).toBeInTheDocument();
  });

  it("never exposes storage keys, absolute paths, or the model runtime", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "COMPLETED", withMaster: true }]);
    const { container } = render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Track ready");

    const html = container.innerHTML;
    expect(html).not.toContain("/Users/");
    expect(html).not.toContain("master.wav");
    expect(html).not.toContain("preview.mp3");
    expect(html).not.toContain("bucket");
    expect(html).not.toContain("8001");
    expect(html).not.toContain("acestep-api");
    expect(html).not.toContain("postgresql");
    expect(html).not.toContain("redis");
  });

  it("stops polling once the generation is COMPLETED", async () => {
    const user = userEvent.setup();
    const { getCalls } = stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Track ready");

    const afterTerminal = getCalls.length;
    await act(async () => {
      await new Promise((r) => setTimeout(r, 2500));
    });
    // Only the session-history refresh may run; no new status polling.
    const statusPolls = getCalls.filter((u) => u.endsWith(GEN_ID)).length;
    expect(getCalls.length).toBeLessThanOrEqual(afterTerminal + 1);
    expect(statusPolls).toBeLessThanOrEqual(afterTerminal + 1);
  });
});

describe("failure UX", () => {
  it("renders a safe message and a Retry action when the generation FAILS", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "FAILED", error_code: "MODEL_LOAD_FAILED" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const panel = await screen.findByRole("alert");
    expect(panel).toHaveTextContent("The music model is not available right now.");
    expect(panel.textContent).not.toMatch(/Traceback|\/Users\/|acestep|redis/i);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("Retry submits again under a new Idempotency-Key", async () => {
    const user = userEvent.setup();
    const { fetchMock } = stubServer([{ status: "FAILED", error_code: "GENERATION_TIMEOUT" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByRole("button", { name: "Retry" });
    await user.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => {
      const posts = fetchMock.mock.calls.filter(isCreatePost);
      expect(posts.length).toBe(2);
      expect(headerOf(posts[0][1], "Idempotency-Key")).not.toBe(
        headerOf(posts[1][1], "Idempotency-Key"),
      );
    });
  });

  it("reports an unreachable API without leaking the raw network error", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("fetch failed ECONNREFUSED 127.0.0.1:8000")),
    );
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not reach the LUBER service.");
    expect(alert.textContent).not.toContain("ECONNREFUSED");
  });

  it("stops polling after a terminal FAILED state", async () => {
    const user = userEvent.setup();
    const { getCalls } = stubServer([{ status: "FAILED", error_code: "OUT_OF_MEMORY" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByRole("alert");

    const before = getCalls.filter((u) => u.endsWith(GEN_ID)).length;
    await act(async () => {
      await new Promise((r) => setTimeout(r, 2500));
    });
    expect(getCalls.filter((u) => u.endsWith(GEN_ID)).length).toBe(before);
  });
});

describe("refresh recovery", () => {
  it("resumes an in-flight generation stored before the reload", async () => {
    window.localStorage.setItem("luber.activeGenerationId", GEN_ID);
    stubServer([{ status: "GENERATING" }]);

    render(<CreatePage />);

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Creating your music"),
    );
  });

  it("shows the finished result when the stored generation already COMPLETED", async () => {
    window.localStorage.setItem("luber.activeGenerationId", GEN_ID);
    stubServer([{ status: "COMPLETED", withMaster: true }]);

    render(<CreatePage />);

    expect(await screen.findByText("Track ready")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Preview player for Midnight Window"),
    ).toBeInTheDocument();
  });

  it("shows failure when the stored generation already FAILED", async () => {
    window.localStorage.setItem("luber.activeGenerationId", GEN_ID);
    stubServer([{ status: "FAILED", error_code: "UNKNOWN_GENERATION_ERROR" }]);

    render(<CreatePage />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong");
  });

  it("ignores a corrupt stored id instead of calling the API with it", async () => {
    window.localStorage.setItem("luber.activeGenerationId", "../../etc/passwd");
    const { getCalls } = stubServer([{ status: "QUEUED" }]);

    render(<CreatePage />);

    await waitFor(() => {
      expect(screen.getByText("Your track appears here")).toBeInTheDocument();
    });
    expect(getCalls.some((u) => u.includes("etc/passwd"))).toBe(false);
  });

  it("persists the active id so a refresh can recover it", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "GENERATING" }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(window.localStorage.getItem("luber.activeGenerationId")).toBe(GEN_ID);
    });
  });

  it("clears the active id once the generation finishes", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Track ready");

    await waitFor(() => {
      expect(window.localStorage.getItem("luber.activeGenerationId")).toBeNull();
    });
  });
});

describe("session history", () => {
  it("lists a completed generation and restores it on click", async () => {
    const user = userEvent.setup();
    stubServer([{ status: "COMPLETED", withMaster: true }]);
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Track ready");

    const history = await screen.findByRole("region", { name: /this session/i });
    const entry = within(history).getByRole("button", { name: /Midnight Window/ });
    expect(entry).toBeInTheDocument();

    await user.click(entry);
    expect(await screen.findByText("Track ready")).toBeInTheDocument();
  });
});

describe("polling discipline", () => {
  it("keeps at most one status request in flight", async () => {
    const user = userEvent.setup();
    let inFlight = 0;
    let maxConcurrent = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (init?.method === "POST" && isPreflight(url)) {
          return jsonResponse(preflightBody());
        }
        if (init?.method === "POST") {
          return jsonResponse({ generation_id: GEN_ID, status: "QUEUED", advisories: [] });
        }
        inFlight += 1;
        maxConcurrent = Math.max(maxConcurrent, inFlight);
        await new Promise((r) => setTimeout(r, 60));
        inFlight -= 1;
        return jsonResponse(generationBody({ status: "GENERATING" }));
      }),
    );
    render(<CreatePage />);

    await fillValidForm(user);
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 300));
    });

    expect(maxConcurrent).toBeLessThanOrEqual(1);
  });
});
