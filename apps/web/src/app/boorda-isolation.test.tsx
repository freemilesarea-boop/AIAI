/**
 * What must not survive a change of account on a shared browser.
 *
 * The server already refuses to serve one user another's songs — that is
 * covered exhaustively in `apps/api/tests/test_ownership_enforcement.py`.
 * These tests cover the half the server cannot reach: state the previous
 * user left inside this browser tab.
 *
 * Two things in particular. The player holds an audio element with a src
 * pointing at a private endpoint, and it lives above the router so it
 * survives navigation by design — which means it also survives a logout
 * unless something stops it. And the home dashboard holds a list of
 * recent tracks in component state.
 *
 * Both are wired to the session-ended event. If that wiring is ever
 * removed, these fail.
 */

import { act, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PlayerProvider, usePlayer } from "@/components/player/PlayerProvider";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { ToastProvider } from "@/components/ui/Toast";
import HomePage from "@/app/page";
import { emitSessionEnded } from "@/lib/session-events";

let pathname = "/";
vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
}));

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function song(id: string, title: string) {
  return {
    id,
    title,
    prompt: "p",
    lyrics: "",
    vocal_gender: "female",
    duration_requested: 60,
    duration_actual: 60,
    seed: null,
    language: "ko",
    instrumental: false,
    bpm: null,
    key_scale: null,
    time_signature: null,
    parent_generation_id: null,
    variation_label: null,
    favorite: false,
    status: "SUCCEEDED",
    created_at: "2026-02-01T00:00:00Z",
  };
}

/** Serves whichever user's library the caller sets up. */
function stubApi(user: unknown, generations: unknown[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/auth/me")) return json(user);
      if (url.includes("/generations")) return json({ items: generations });
      return json({ items: [] });
    }),
  );
}

const USER_A = { id: "a", email: "a@boorda.kr", display_name: "A", created_at: "2026-01-01T00:00:00Z" };

beforeEach(() => {
  pathname = "/";
  vi.unstubAllGlobals();
  window.localStorage.clear();
  window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  window.HTMLMediaElement.prototype.pause = vi.fn();
  window.HTMLMediaElement.prototype.load = vi.fn();
});

describe("player state does not outlive a session", () => {
  function Harness() {
    const { track, play } = usePlayer();
    return (
      <div>
        <button
          type="button"
          onClick={() =>
            play({
              id: "a1",
              title: "A의 노래",
              src: "http://api.test/v1/generations/a1/audio?asset=preview",
              downloadUrl: "http://api.test/v1/generations/a1/audio?download=true",
              durationHint: 60,
            })
          }
        >
          play
        </button>
        <p data-testid="track">{track?.title ?? "none"}</p>
      </div>
    );
  }

  it("drops the track and the audio source when the session ends", async () => {
    const { container } = render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );

    await act(async () => {
      screen.getByRole("button", { name: "play" }).click();
    });
    expect(screen.getByTestId("track")).toHaveTextContent("A의 노래");
    expect(container.querySelector("audio")?.getAttribute("src")).toBeTruthy();

    // User A signs out. B must not inherit a playing private track.
    await act(async () => {
      emitSessionEnded();
    });

    expect(screen.getByTestId("track")).toHaveTextContent("none");
    // The src is removed, not merely paused: a paused element still
    // holds a URL to a private endpoint.
    expect(container.querySelector("audio")?.getAttribute("src")).toBeNull();
  });

  it("pauses the element rather than leaving it running", async () => {
    render(
      <PlayerProvider>
        <Harness />
      </PlayerProvider>,
    );
    await act(async () => {
      screen.getByRole("button", { name: "play" }).click();
    });
    await act(async () => {
      emitSessionEnded();
    });
    expect(window.HTMLMediaElement.prototype.pause).toHaveBeenCalled();
  });
});

describe("home shows only what the scoped API returned", () => {
  it("renders the caller's tracks and nothing it was not given", async () => {
    stubApi(USER_A, [song("a1", "A의 첫 곡"), song("a2", "A의 둘째 곡")]);
    render(
      <AuthProvider>
        <PlayerProvider>
          <ToastProvider>
            <HomePage />
          </ToastProvider>
        </PlayerProvider>
      </AuthProvider>,
    );
    expect(await screen.findByText("A의 첫 곡")).toBeInTheDocument();
    expect(screen.getByText("A의 둘째 곡")).toBeInTheDocument();
    expect(screen.queryByText("B의 곡")).not.toBeInTheDocument();
  });

  it("asks the server for the list instead of filtering a shared one", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER_A);
      return json({ items: [song("a1", "A의 첫 곡")] });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(
      <AuthProvider>
        <PlayerProvider>
          <ToastProvider>
            <HomePage />
          </ToastProvider>
        </PlayerProvider>
      </AuthProvider>,
    );
    await screen.findByText("A의 첫 곡");

    const calls = fetchMock.mock.calls.map((call) => String(call[0]));
    const listing = calls.find((url) => url.includes("/v1/generations"));
    expect(listing).toBeTruthy();
    // No owner is named in the request: the session decides, and a
    // client-supplied id must never be the authority.
    expect(listing).not.toMatch(/user_id|owner_id|account_id/);
    // Credentials ride along, so the server can identify the caller.
    const init = fetchMock.mock.calls.find((call) =>
      String(call[0]).includes("/v1/generations"),
    )?.[1] as RequestInit | undefined;
    expect(init?.credentials).toBe("include");
  });

  it("does not cache the listing for whoever signs in next", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER_A);
      return json({ items: [song("a1", "A의 첫 곡")] });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(
      <AuthProvider>
        <PlayerProvider>
          <ToastProvider>
            <HomePage />
          </ToastProvider>
        </PlayerProvider>
      </AuthProvider>,
    );
    await screen.findByText("A의 첫 곡");
    const init = fetchMock.mock.calls.find((call) =>
      String(call[0]).includes("/v1/generations"),
    )?.[1] as RequestInit | undefined;
    expect(init?.cache).toBe("no-store");
  });
});

describe("no global feed exists in the private library", () => {
  it("home never asks for anyone else's songs", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER_A);
      return json({ items: [] });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(
      <AuthProvider>
        <PlayerProvider>
          <ToastProvider>
            <HomePage />
          </ToastProvider>
        </PlayerProvider>
      </AuthProvider>,
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    for (const call of fetchMock.mock.calls) {
      const url = String(call[0]);
      expect(url).not.toMatch(/public|explore|feed|all=true|everyone/i);
    }
  });
});
