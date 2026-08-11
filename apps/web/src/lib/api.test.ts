import { fetchHealth } from "./api";

describe("fetchHealth", () => {
  it("returns parsed health payload", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok", service: "luber-api" }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const health = await fetchHealth();
    expect(health).toEqual({ status: "ok", service: "luber-api" });
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/health"),
      expect.objectContaining({ cache: "no-store" }),
    );

    vi.unstubAllGlobals();
  });

  it("throws on non-2xx responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 503 }),
    );
    await expect(fetchHealth()).rejects.toThrow("503");
    vi.unstubAllGlobals();
  });
});
