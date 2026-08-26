/**
 * The Create page must always reach a state a person can act on.
 *
 * An operator found /create stuck on "Loading…" forever. The cause was a
 * stale dev server serving 404s for Next's own runtime chunks, so React
 * never booted and the page froze on its server-rendered snapshot — not
 * an application bug. But the incident is worth a guard anyway, because
 * "Loading…" with no exit is the worst failure a page can have: it looks
 * like progress and never ends.
 *
 * So these pin the property rather than the incident. Whatever the API
 * does — 401, 500, a network error, a hang — the shell must leave
 * `loading` and land on something: the product, a sign-in redirect, or a
 * stated error. And with the generation engine offline, the Create form
 * must still render; only generating may fail.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthProvider } from "@/components/auth/AuthProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { ToastProvider } from "@/components/ui/Toast";
import { GenerationForm } from "@/components/GenerationForm";
import { describeGenerationFailure } from "@/lib/errors";

let pathname = "/create";
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useRouter: () => ({ push: vi.fn(), replace, refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
}));

const USER = {
  id: "u1",
  email: "a@boorda.kr",
  display_name: null,
  created_at: "2026-01-01T00:00:00Z",
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function shell(children: React.ReactNode) {
  return render(
    <AuthProvider>
      <PlayerProvider>
        <ToastProvider>
          <RequireAuth pathname={pathname}>{children}</RequireAuth>
        </ToastProvider>
      </PlayerProvider>
    </AuthProvider>,
  );
}

/** Whatever else is on screen, it must not still be the loading state. */
async function expectNotStuckLoading() {
  await waitFor(() => {
    expect(screen.queryByText("Loading…")).not.toBeInTheDocument();
  });
}

beforeEach(() => {
  pathname = "/create";
  replace.mockClear();
  vi.unstubAllGlobals();
});

describe("the shell always leaves the loading state", () => {
  it("when the session check says guest", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "unauthorized" }, 401)));
    shell(<p>create</p>);
    await expectNotStuckLoading();
  });

  it("when the session check fails outright", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    shell(<p>create</p>);
    await expectNotStuckLoading();
  });

  it("when the API answers with a server error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "boom" }, 500)));
    shell(<p>create</p>);
    await expectNotStuckLoading();
  });

  it("when the API is reachable and the user is signed in", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json(USER)));
    shell(<p>create</p>);
    expect(await screen.findByText("create")).toBeInTheDocument();
    await expectNotStuckLoading();
  });
});

describe("the create form does not depend on the generation engine", () => {
  it("renders its controls while every non-auth request fails", async () => {
    // Auth succeeds; anything the page asks for afterwards does not —
    // which is what an offline engine and a cold cache look like.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes("/auth/me")) return json(USER);
        return Promise.reject(new TypeError("Failed to fetch"));
      }),
    );
    render(
      <AuthProvider>
        <PlayerProvider>
          <ToastProvider>
            <GenerationForm onSubmit={vi.fn()} busy={false} />
          </ToastProvider>
        </PlayerProvider>
      </AuthProvider>,
    );

    // The controls an operator needs to inspect the product.
    expect(await screen.findByLabelText("제목")).toBeInTheDocument();
    expect(screen.getByLabelText("어떤 음악을 만들까요?")).toBeInTheDocument();
    expect(screen.getByLabelText("가사")).toBeInTheDocument();
    expect(screen.getByLabelText("보컬")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "음악 만들기" })).toBeInTheDocument();
  });
});

describe("an offline engine is reported, not hidden", () => {
  it("names the engine rather than blaming the user's settings", () => {
    const { message } = describeGenerationFailure("MODEL_LOAD_FAILED");
    expect(message).toMatch(/음악 모델/);
    expect(message).toMatch(/다시 시도/);
    // No stack traces, paths, worker ids or engine internals.
    for (const leak of ["Traceback", "/Users/", "127.0.0.1", "ace_step", "worker"]) {
      expect(message).not.toContain(leak);
    }
  });

  it("offers a retry rather than a dead end", () => {
    expect(describeGenerationFailure("MODEL_LOAD_FAILED").retryable).toBe(true);
  });

  it("never reports a failed generation as finished", () => {
    // The status the library filters on must stay terminal-failed, so a
    // failed attempt can never be mistaken for a playable song.
    const { message } = describeGenerationFailure("MODEL_LOAD_FAILED");
    expect(message).not.toMatch(/완료|성공/);
  });
});
