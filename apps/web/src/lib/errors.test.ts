/**
 * Failure copy for the reliability codes Phase 18 introduced.
 *
 * The value of a separate code is entirely in what it lets the product
 * say. PROVIDER_BUSY exists so a saturated engine does not read as a
 * failed song, and that distinction only survives if the copy keeps it.
 */

import { describeGenerationFailure } from "./errors";

describe("reliability failure codes", () => {
  it("tells the user a busy engine is not their fault", () => {
    const { message } = describeGenerationFailure("PROVIDER_BUSY");
    expect(message).toMatch(/바쁩니다/);
    // The distinction the code exists to preserve: nothing failed, and
    // the user's settings are not at fault. Asserted in the language the
    // message is actually written in — a negative match against English
    // words would pass vacuously on Korean copy.
    expect(message).not.toMatch(/실패|오류|잘못/);
  });

  it("asks for a retry after an interruption", () => {
    expect(describeGenerationFailure("GENERATION_INTERRUPTED").message).toMatch(
      /중단되었습니다/,
    );
  });

  it("still falls back for a code this build has never seen", () => {
    const { message } = describeGenerationFailure("SOME_FUTURE_CODE");
    expect(message).toBe(describeGenerationFailure("UNKNOWN_GENERATION_ERROR").message);
  });

  it("never renders an internal detail for any known code", () => {
    for (const code of [
      "PROVIDER_BUSY",
      "GENERATION_INTERRUPTED",
      "UPLOAD_FAILED",
      "OUT_OF_MEMORY",
      "UNKNOWN_GENERATION_ERROR",
    ]) {
      const { message } = describeGenerationFailure(code);
      expect(message).not.toMatch(/\/Users\/|Traceback|Exception|ACE-Step|arq|redis/i);
    }
  });
});
