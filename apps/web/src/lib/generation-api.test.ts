import {
  ApiError,
  createGeneration,
  findMasterAsset,
  getGeneration,
  getMasterAudioUrl,
  isTerminalStatus,
  newIdempotencyKey,
  type Generation,
} from "./api";
import { describeApiError, describeGenerationFailure } from "./errors";
import { statusLabel } from "./generationStatus";

const INPUT = {
  title: "Midnight Window",
  prompt: "Dreamy Korean indie pop",
  lyrics: "[Verse]\n오늘 밤 너를 생각해",
  vocal_gender: "female" as const,
  language: "ko",
  duration: 30,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createGeneration", () => {
  it("posts the payload with the supplied Idempotency-Key", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ generation_id: "abc", status: "QUEUED" }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const res = await createGeneration(INPUT, "key-123");

    expect(res).toEqual({ generation_id: "abc", status: "QUEUED" });
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/v1/generations");
    expect(init.method).toBe("POST");
    expect(init.headers["Idempotency-Key"]).toBe("key-123");
    // Korean and line breaks survive serialization intact.
    expect(JSON.parse(init.body).lyrics).toBe("[Verse]\n오늘 밤 너를 생각해");
    expect(JSON.parse(init.body).vocal_gender).toBe("female");
  });

  it("raises ApiError carrying the backend error code", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ detail: "QUEUE_FAILED" }),
      }),
    );

    await expect(createGeneration(INPUT, "k")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("newIdempotencyKey", () => {
  it("returns a distinct key on every call", () => {
    const keys = new Set(Array.from({ length: 50 }, () => newIdempotencyKey()));
    expect(keys.size).toBe(50);
  });

  it("still produces a key when crypto.randomUUID is unavailable", () => {
    vi.stubGlobal("crypto", {});
    expect(newIdempotencyKey().length).toBeGreaterThan(0);
  });
});

describe("getGeneration", () => {
  it("requests the generation without caching", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "x" }) });
    vi.stubGlobal("fetch", mockFetch);

    await getGeneration("x");

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/v1/generations/x");
    expect(init.cache).toBe("no-store");
  });

  it("throws a 404 ApiError for an unknown id", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) }),
    );
    await expect(getGeneration("missing")).rejects.toMatchObject({ status: 404 });
  });
});

describe("getMasterAudioUrl", () => {
  it("addresses audio by generation id only", () => {
    const url = getMasterAudioUrl("gen-1");
    expect(url).toContain("/v1/generations/gen-1/audio");
    // No storage key, no filesystem path ever reaches the browser.
    expect(url).not.toContain("storage");
    expect(url).not.toContain("/Users/");
    expect(url).not.toContain("..");
  });

  it("adds the download flag when requested", () => {
    expect(getMasterAudioUrl("gen-1", true)).toContain("download=true");
  });

  it("never points at the ACE-Step runtime", () => {
    const url = getMasterAudioUrl("gen-1");
    expect(url).not.toContain("8001");
    expect(url).not.toContain("acestep");
    expect(url).not.toContain("release_task");
  });

  it("encodes ids so they cannot alter the path", () => {
    expect(getMasterAudioUrl("../../etc/passwd")).not.toContain("../");
  });
});

describe("status helpers", () => {
  it("treats only terminal states as terminal", () => {
    expect(isTerminalStatus("COMPLETED")).toBe(true);
    expect(isTerminalStatus("FAILED")).toBe(true);
    expect(isTerminalStatus("CANCELLED")).toBe(true);
    expect(isTerminalStatus("QUEUED")).toBe(false);
    expect(isTerminalStatus("GENERATING")).toBe(false);
    expect(isTerminalStatus("UPLOADING")).toBe(false);
  });

  it("maps every backend state to user-facing language", () => {
    expect(statusLabel("QUEUED")).toBe("Preparing generation");
    expect(statusLabel("STARTING")).toBe("Starting AI model");
    expect(statusLabel("GENERATING")).toBe("Creating your music");
    expect(statusLabel("POST_PROCESSING")).toBe("Processing audio");
    expect(statusLabel("UPLOADING")).toBe("Saving your track");
    expect(statusLabel("COMPLETED")).toBe("Track ready");
    expect(statusLabel("FAILED")).toBe("Generation failed");
  });
});

describe("findMasterAsset", () => {
  it("selects the MASTER asset and ignores other types", () => {
    const generation = {
      audio_assets: [
        { asset_type: "PREVIEW", id: "p" },
        { asset_type: "MASTER", id: "m" },
      ],
    } as unknown as Generation;
    expect(findMasterAsset(generation)?.id).toBe("m");
  });

  it("returns null when no MASTER exists", () => {
    const generation = { audio_assets: [] } as unknown as Generation;
    expect(findMasterAsset(generation)).toBeNull();
  });
});

describe("error translation", () => {
  it("never leaks internals for known failure codes", () => {
    const msg = describeGenerationFailure("MODEL_LOAD_FAILED").message;
    expect(msg).not.toMatch(/Traceback|Exception|\/Users\/|postgres|redis|acestep/i);
    expect(msg.length).toBeGreaterThan(0);
  });

  it("falls back to a safe message for unknown codes", () => {
    expect(describeGenerationFailure(null).message).toContain("Something went wrong");
    expect(describeGenerationFailure("WEIRD_NEW_CODE").message).toContain("Something went wrong");
  });

  it("describes network failure without exposing raw errors", () => {
    const described = describeApiError(new TypeError("fetch failed: ECONNREFUSED 127.0.0.1:8000"));
    expect(described.message).not.toContain("ECONNREFUSED");
    expect(described.message).not.toContain("127.0.0.1");
    expect(described.retryable).toBe(true);
  });

  it("marks 404 as not retryable and 503 as retryable", () => {
    expect(describeApiError(new ApiError("x", 404)).retryable).toBe(false);
    expect(describeApiError(new ApiError("x", 503)).retryable).toBe(true);
  });
});
