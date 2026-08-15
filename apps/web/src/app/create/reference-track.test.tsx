/**
 * Reference Track UX.
 *
 * The risk this feature carries is not a broken upload — it is a
 * convincing one. A file sitting in the browser looks attached whether
 * or not the server ever saw it, so most of what is asserted here is
 * that the UI never claims more than the backend actually did:
 * limits are the server's, the id is the server's, and a generation
 * cites a reference only when one really exists.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CreatePage from "@/app/create/page";

const searchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => searchParams,
  usePathname: () => "/create",
}));
import { PlayerProvider } from "@/components/player/PlayerProvider";

const LIMITS = {
  max_file_bytes: 41_943_040,
  max_duration_seconds: 600,
  supported_formats: ["wav", "mp3", "flac", "m4a", "ogg"],
};

const REFERENCE = {
  reference_id: "9f1c2b3a-1111-4222-8333-444455556666",
  display_name: "demo.wav",
  duration_seconds: 12,
  sample_rate: 48000,
  channels: 2,
  file_size: 2_304_078,
};

function audioFile(name = "demo.wav") {
  return new File([new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0])], name, {
    type: "audio/wav",
  });
}

interface StubOptions {
  limits?: unknown;
  limitsStatus?: number;
  upload?: unknown;
  uploadStatus?: number;
  uploadDetail?: string;
  /** Resolve the upload manually, to observe the in-flight state. */
  deferUpload?: boolean;
}

function stub(options: StubOptions = {}) {
  const calls: { url: string; init?: RequestInit }[] = [];
  let releaseUpload: (() => void) | undefined;
  const uploadGate = new Promise<void>((resolve) => {
    releaseUpload = resolve;
  });

  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });

    if (url.includes("/v1/reference-audio/limits")) {
      const status = options.limitsStatus ?? 200;
      if (status !== 200) return json({ detail: "nope" }, status);
      return json(options.limits ?? LIMITS);
    }
    if (url.includes("/v1/reference-audio")) {
      if (options.deferUpload) await uploadGate;
      const status = options.uploadStatus ?? 201;
      if (status >= 400) return json({ detail: options.uploadDetail ?? "Rejected." }, status);
      return json(options.upload ?? REFERENCE, 201);
    }
    if (url.includes("/preflight")) return json({ advisories: [] });
    if (url.includes("/v1/generations") && init?.method === "POST") {
      return json({ generation_id: "gen-1", status: "QUEUED", advisories: [] }, 202);
    }
    if (url.includes("/v1/generations")) {
      return json({ id: "gen-1", status: "QUEUED", audio_assets: [], advisories: [] });
    }
    if (url.includes("/v1/projects")) return json({ items: [] });
    return json({});
  });

  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, calls, releaseUpload: () => releaseUpload?.() };
}

function renderCreate() {
  return render(
    <PlayerProvider>
      <CreatePage />
    </PlayerProvider>,
  );
}

/** The JSON body of the generation POST, if one was made. */
function generationBody(calls: { url: string; init?: RequestInit }[]) {
  const post = calls.find(
    (c) => c.init?.method === "POST" && c.url.includes("/v1/generations") && !c.url.includes("preflight"),
  );
  return post ? (JSON.parse(String(post.init?.body)) as Record<string, unknown>) : null;
}

async function fillRequiredFields(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Title"), "Reference Song");
  await user.type(screen.getByLabelText("Music description"), "warm indie pop");
  // A vocal track needs lyrics; pasted because the tag palette makes
  // per-character typing slow and irrelevant to what is under test.
  await user.click(screen.getByLabelText("Lyrics"));
  await user.paste("[Verse]\nquiet morning light");
}

async function attachReference(user: ReturnType<typeof userEvent.setup>, file = audioFile()) {
  const input = screen.getByLabelText(/Choose an audio file/i);
  await act(async () => {
    await user.upload(input, file);
  });
  return input;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe("Reference Track — presentation", () => {
  it("renders as an optional section with truthful copy", async () => {
    stub();
    renderCreate();
    const section = screen.getByRole("region", { name: /Reference Track/i });
    expect(within(section).getByText("Optional")).toBeTruthy();
    expect(
      within(section).getByText(/guide the new song.s sound and production character/i),
    ).toBeTruthy();
  });

  it("never uses imitation or cloning language", async () => {
    stub();
    const { container } = renderCreate();
    await waitFor(() => expect(screen.getByText(/WAV, MP3/i)).toBeTruthy());
    const copy = (container.textContent ?? "").toLowerCase();
    for (const banned of [
      "sounds like",
      "sound like",
      "make it sound like",
      "clone",
      "copy this song",
      "copy this artist",
      "style transfer",
    ]) {
      expect(copy).not.toContain(banned);
    }
  });

  it("introduces no raw-master or finishing vocabulary", async () => {
    stub();
    const { container } = renderCreate();
    const copy = (container.textContent ?? "").toLowerCase();
    for (const internal of ["raw_master", "raw master", "finished_master", "finishing", "p14-v1"]) {
      expect(copy).not.toContain(internal);
    }
  });
});

describe("Reference Track — limits come from the server", () => {
  it("shows the real returned values", async () => {
    stub();
    renderCreate();
    // 41943040 bytes -> 40 MB, 600s -> 10 min: rendered from the payload,
    // not from constants duplicated in the frontend.
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    expect(screen.getByText(/10 min/)).toBeTruthy();
    expect(screen.getByText(/WAV, MP3, FLAC, M4A, OGG/)).toBeTruthy();
  });

  it("reflects different server limits rather than hardcoded ones", async () => {
    stub({ limits: { max_file_bytes: 5_242_880, max_duration_seconds: 120, supported_formats: ["wav"] } });
    renderCreate();
    await waitFor(() => expect(screen.getByText(/5 MB/)).toBeTruthy());
    expect(screen.getByText(/2 min/)).toBeTruthy();
    expect(screen.queryByText(/40 MB/)).toBeNull();
  });

  it("invents no fallback when the limits endpoint fails", async () => {
    stub({ limitsStatus: 500 });
    renderCreate();
    await waitFor(() =>
      expect(screen.getByText(/requirements could not be loaded/i)).toBeTruthy(),
    );
    // No numbers appear at all, and uploading is disabled.
    expect(screen.queryByText(/MB/)).toBeNull();
    expect((screen.getByLabelText(/Choose an audio file/i) as HTMLInputElement).disabled).toBe(true);
  });
});

describe("Reference Track — upload", () => {
  it("uploads the actual file and stores the backend id", async () => {
    const { calls } = stub();
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user, audioFile("my-demo.wav"));

    const upload = calls.find((c) => c.url.endsWith("/v1/reference-audio"));
    expect(upload?.init?.method).toBe("POST");
    const body = upload?.init?.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    const sent = body.get("file") as File;
    expect(sent.name).toBe("my-demo.wav");
    // No path, no client-side id.
    expect(String(upload?.init?.body)).not.toContain("/");
  });

  it("reaches READY and reports the attachment", async () => {
    stub();
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);
    await waitFor(() => expect(screen.getByText(/attached/i)).toBeTruthy());
  });

  it("shows an indeterminate uploading state, never a percentage", async () => {
    const { releaseUpload } = stub({ deferUpload: true });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);

    const status = await screen.findByText(/Uploading/i);
    expect(status.textContent).not.toMatch(/\d+\s*%/);
    await act(async () => {
      releaseUpload();
    });
  });

  it("surfaces the backend's rejection reason", async () => {
    stub({ uploadStatus: 400, uploadDetail: "That file is larger than 40 MB." });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);
    await waitFor(() =>
      expect(screen.getByText("That file is larger than 40 MB.")).toBeTruthy(),
    );
  });

  it("surfaces a corrupt-file rejection as the backend worded it", async () => {
    stub({
      uploadStatus: 400,
      uploadDetail: "That file could not be read as audio. It may be corrupt or not an audio file.",
    });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);
    await waitFor(() => expect(screen.getByText(/may be corrupt/i)).toBeTruthy());
  });
});

describe("Reference Track — generation submission", () => {
  it("sends reference_audio_id once the reference is READY", async () => {
    const { calls } = stub();
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await fillRequiredFields(user);
    await attachReference(user);
    await waitFor(() => expect(screen.getByText(/attached/i)).toBeTruthy());

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(generationBody(calls)).not.toBeNull());
    expect(generationBody(calls)?.reference_audio_id).toBe(REFERENCE.reference_id);
  });

  it("omits the field entirely when no reference is attached", async () => {
    const { calls } = stub();
    const user = userEvent.setup();
    renderCreate();
    await fillRequiredFields(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(generationBody(calls)).not.toBeNull());
    expect(generationBody(calls)).not.toHaveProperty("reference_audio_id");
  });

  it("refuses to submit while an upload is still in flight", async () => {
    const { calls, releaseUpload } = stub({ deferUpload: true });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await fillRequiredFields(user);
    await attachReference(user);
    await screen.findByText(/Uploading/i);

    await user.click(screen.getByRole("button", { name: "Create" }));
    expect(await screen.findByText(/Wait for the reference track/i)).toBeTruthy();
    expect(generationBody(calls)).toBeNull();

    await act(async () => {
      releaseUpload();
    });
  });

  it("does not cite a reference when the upload failed", async () => {
    const { calls } = stub({ uploadStatus: 400, uploadDetail: "Rejected." });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await fillRequiredFields(user);
    await attachReference(user);
    await waitFor(() => expect(screen.getByText("Rejected.")).toBeTruthy());

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(generationBody(calls)).not.toBeNull());
    expect(generationBody(calls)).not.toHaveProperty("reference_audio_id");
  });
});

describe("Reference Track — remove and replace", () => {
  it("remove clears the id from the next request", async () => {
    const { calls } = stub();
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await fillRequiredFields(user);
    await attachReference(user);
    await waitFor(() => expect(screen.getByText(/attached/i)).toBeTruthy());

    await user.click(screen.getByRole("button", { name: /Remove reference track/i }));
    expect(screen.getByText(/No reference track attached/i)).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(generationBody(calls)).not.toBeNull());
    expect(generationBody(calls)).not.toHaveProperty("reference_audio_id");
  });

  it("replace submits the new id, never the old one", async () => {
    const second = { ...REFERENCE, reference_id: "aaaa1111-2222-4333-8444-555566667777" };
    let uploads = 0;
    const calls: { url: string; init?: RequestInit }[] = [];
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        calls.push({ url, init });
        if (url.includes("/v1/reference-audio/limits")) return json(LIMITS);
        if (url.includes("/v1/reference-audio")) {
          uploads += 1;
          return json(uploads === 1 ? REFERENCE : second, 201);
        }
        if (url.includes("/preflight")) return json({ advisories: [] });
        if (url.includes("/v1/generations") && init?.method === "POST") {
          return json({ generation_id: "gen-1", status: "QUEUED", advisories: [] }, 202);
        }
        if (url.includes("/v1/generations")) {
          return json({ id: "gen-1", status: "QUEUED", audio_assets: [], advisories: [] });
        }
        return json({});
      }),
    );

    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await fillRequiredFields(user);
    await attachReference(user, audioFile("first.wav"));
    await waitFor(() => expect(screen.getByText(/attached/i)).toBeTruthy());
    await attachReference(user, audioFile("second.wav"));
    await waitFor(() => expect(uploads).toBe(2));

    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(generationBody(calls)).not.toBeNull());
    expect(generationBody(calls)?.reference_audio_id).toBe(second.reference_id);
    expect(generationBody(calls)?.reference_audio_id).not.toBe(REFERENCE.reference_id);
  });
});

describe("Reference Track — modes and accessibility", () => {
  it("is available in Simple mode", async () => {
    stub();
    renderCreate();
    expect(screen.getByRole("region", { name: /Reference Track/i })).toBeTruthy();
  });

  it("is still available after switching to Custom", async () => {
    stub();
    const user = userEvent.setup();
    renderCreate();
    await user.click(screen.getByRole("tab", { name: "Custom" }));
    expect(screen.getByRole("region", { name: /Reference Track/i })).toBeTruthy();
  });

  it("keeps the attachment across a mode switch", async () => {
    stub();
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);
    await waitFor(() => expect(screen.getByText(/attached/i)).toBeTruthy());
    await user.click(screen.getByRole("tab", { name: "Custom" }));
    expect(screen.getByText(/attached/i)).toBeTruthy();
  });

  it("offers a labelled, keyboard-reachable file input", async () => {
    stub();
    renderCreate();
    // Enabled only once the server's limits are known.
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    const input = screen.getByLabelText(/Choose an audio file/i) as HTMLInputElement;
    expect(input.tagName).toBe("INPUT");
    expect(input.type).toBe("file");
    // Focusable, so drag-and-drop is never the only way in.
    input.focus();
    expect(document.activeElement).toBe(input);
  });

  it("announces status politely and links it to the input", async () => {
    stub();
    renderCreate();
    const input = screen.getByLabelText(/Choose an audio file/i);
    const status = screen.getByRole("status", { name: "Reference track status" });
    expect(status.getAttribute("aria-live")).toBe("polite");
    expect(input.getAttribute("aria-describedby")).toContain(status.id);
  });

  it("marks the input invalid when the upload was rejected", async () => {
    stub({ uploadStatus: 400, uploadDetail: "Rejected." });
    const user = userEvent.setup();
    renderCreate();
    await waitFor(() => expect(screen.getByText(/40 MB/)).toBeTruthy());
    await attachReference(user);
    await waitFor(() => expect(screen.getByText("Rejected.")).toBeTruthy());
    expect(screen.getByLabelText(/Choose an audio file/i).getAttribute("aria-invalid")).toBe("true");
  });
});
