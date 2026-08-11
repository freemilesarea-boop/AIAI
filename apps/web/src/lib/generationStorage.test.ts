import {
  clearActiveGenerationId,
  isValidGenerationId,
  loadActiveGenerationId,
  loadRecentGenerations,
  rememberGeneration,
  saveActiveGenerationId,
} from "./generationStorage";

const VALID_ID = "d1d76e27-119a-41e8-a358-a492141efaba";

beforeEach(() => {
  window.localStorage.clear();
});

describe("isValidGenerationId", () => {
  it("accepts UUIDs and rejects anything else", () => {
    expect(isValidGenerationId(VALID_ID)).toBe(true);
    expect(isValidGenerationId("not-a-uuid")).toBe(false);
    expect(isValidGenerationId("../../etc/passwd")).toBe(false);
    expect(isValidGenerationId("")).toBe(false);
    expect(isValidGenerationId(null)).toBe(false);
    expect(isValidGenerationId(42)).toBe(false);
  });
});

describe("active generation id", () => {
  it("round-trips a valid id for refresh recovery", () => {
    saveActiveGenerationId(VALID_ID);
    expect(loadActiveGenerationId()).toBe(VALID_ID);
  });

  it("clears the id", () => {
    saveActiveGenerationId(VALID_ID);
    clearActiveGenerationId();
    expect(loadActiveGenerationId()).toBeNull();
  });

  it("discards a stale or tampered stored value", () => {
    window.localStorage.setItem("luber.activeGenerationId", "../../etc/passwd");
    expect(loadActiveGenerationId()).toBeNull();
    // The bad value is removed rather than left to fail again.
    expect(window.localStorage.getItem("luber.activeGenerationId")).toBeNull();
  });

  it("refuses to persist an invalid id", () => {
    saveActiveGenerationId("nope");
    expect(loadActiveGenerationId()).toBeNull();
  });
});

describe("recent generations", () => {
  it("stores newest first without duplicates", () => {
    const second = "20bbd4a9-010f-4abe-afb0-d29b99120e20";
    rememberGeneration({ id: VALID_ID, title: "One", createdAt: "2026-01-01T00:00:00Z" });
    rememberGeneration({ id: second, title: "Two", createdAt: "2026-01-01T00:01:00Z" });
    rememberGeneration({ id: VALID_ID, title: "One again", createdAt: "2026-01-01T00:02:00Z" });

    const items = loadRecentGenerations();
    expect(items.map((i) => i.id)).toEqual([VALID_ID, second]);
    expect(items[0].title).toBe("One again");
  });

  it("survives corrupt JSON without throwing", () => {
    window.localStorage.setItem("luber.recentGenerations", "{not json");
    expect(loadRecentGenerations()).toEqual([]);
  });

  it("filters entries with invalid ids", () => {
    window.localStorage.setItem(
      "luber.recentGenerations",
      JSON.stringify([{ id: "bad", title: "x" }, { id: VALID_ID, title: "ok" }]),
    );
    expect(loadRecentGenerations().map((i) => i.id)).toEqual([VALID_ID]);
  });

  it("caps the stored list", () => {
    for (let i = 0; i < 20; i++) {
      rememberGeneration({
        id: `d1d76e27-119a-41e8-a358-a49214${i.toString().padStart(6, "0")}`,
        title: `T${i}`,
        createdAt: "2026-01-01T00:00:00Z",
      });
    }
    expect(loadRecentGenerations().length).toBeLessThanOrEqual(8);
  });
});
