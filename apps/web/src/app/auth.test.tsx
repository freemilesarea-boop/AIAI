/**
 * The browser side of authentication.
 *
 * What is defended here is mostly about *ordering and absence*: that a
 * protected page does not render before the session is known, that a
 * signed-out user never sees the previous user's data, that no token is
 * written anywhere JavaScript can read it, and that a `?next=` value
 * cannot send someone to another origin.
 *
 * The auth calls themselves go through the real provider against a
 * stubbed transport. Mocking the provider would leave nothing worth
 * testing.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AuthProvider, useAuth } from "@/components/auth/AuthProvider";
import { RequireAuth } from "@/components/auth/RequireAuth";
import LoginPage from "@/app/login/page";
import SignupPage from "@/app/signup/page";
import { DEFAULT_DESTINATION, loginUrlFor, safeDestination } from "@/lib/redirect";

const replace = vi.fn();
let searchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => searchParams,
  usePathname: () => "/library",
}));

const USER = {
  id: "user-1",
  email: "person@example.com",
  display_name: null,
  created_at: "2026-08-18T00:00:00Z",
};

interface Stub {
  me?: { status: number; body?: unknown };
  login?: { status: number; body?: unknown };
  signup?: { status: number; body?: unknown };
}

function stubApi(stub: Stub = {}) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const path = String(url);
      calls.push(`${init?.method ?? "GET"} ${path}`);
      const reply = (spec?: { status: number; body?: unknown }) => ({
        ok: (spec?.status ?? 200) < 400,
        status: spec?.status ?? 200,
        json: async () => spec?.body ?? {},
      });
      if (path.includes("/v1/auth/me")) return reply(stub.me ?? { status: 401 });
      if (path.includes("/v1/auth/login")) return reply(stub.login ?? { status: 200, body: USER });
      if (path.includes("/v1/auth/signup"))
        return reply(stub.signup ?? { status: 201, body: USER });
      if (path.includes("/v1/auth/logout")) return reply({ status: 204 });
      return reply({ status: 200, body: { items: [], total: 0 } });
    }),
  );
  return calls;
}

beforeEach(() => {
  replace.mockClear();
  searchParams = new URLSearchParams();
  localStorage.clear();
  sessionStorage.clear();
});

// ── safe redirects ────────────────────────────────────────────────────

describe("return-to destinations", () => {
  it("keeps an ordinary in-app path", () => {
    expect(safeDestination("/song/abc")).toBe("/song/abc");
  });

  it.each([
    ["https://evil.example", "an absolute URL"],
    ["//evil.example", "a scheme-relative URL the browser treats as another origin"],
    ["javascript:alert(1)", "a script URL"],
    ["/\\evil.example", "a backslash some parsers normalise to a slash"],
    ["", "an empty value"],
    [null, "a missing value"],
  ])("refuses %s (%s)", (candidate, _reason) => {
    expect(safeDestination(candidate as string | null)).toBe(DEFAULT_DESTINATION);
  });

  it("does not send a user back to an auth page after signing in", () => {
    expect(loginUrlFor("/login")).toBe("/login");
    expect(loginUrlFor("/signup")).toBe("/login");
  });

  it("carries a private destination through the login URL", () => {
    expect(loginUrlFor("/song/abc")).toBe("/login?next=%2Fsong%2Fabc");
  });

  it("omits ?next when the destination is already the default", () => {
    expect(loginUrlFor("/library")).toBe("/login");
  });
});

// ── bootstrap ─────────────────────────────────────────────────────────

function Probe() {
  const { status, user } = useAuth();
  return <span data-testid="state">{`${status}:${user?.email ?? "none"}`}</span>;
}

describe("session bootstrap", () => {
  it("asks the server who is signed in", async () => {
    const calls = stubApi({ me: { status: 200, body: USER } });
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("state")).toHaveTextContent("authenticated:person@example.com"),
    );
    expect(calls.some((c) => c.includes("/v1/auth/me"))).toBe(true);
  });

  it("treats a 401 as a guest rather than an error", async () => {
    stubApi({ me: { status: 401 } });
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("state")).toHaveTextContent("unauthenticated:none"),
    );
  });

  it("assumes guest when the check fails outright", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("network down");
    }));
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("state")).toHaveTextContent("unauthenticated"),
    );
  });
});

// ── the gate ──────────────────────────────────────────────────────────

describe("protected routes", () => {
  it("does not render the page while the session is unknown", async () => {
    // Never resolves: the state stays "loading".
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(
      <AuthProvider>
        <RequireAuth pathname="/library">
          <p>PRIVATE LIBRARY</p>
        </RequireAuth>
      </AuthProvider>,
    );
    expect(screen.queryByText("PRIVATE LIBRARY")).toBeNull();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("never renders the page for a guest, even briefly", async () => {
    stubApi({ me: { status: 401 } });
    render(
      <AuthProvider>
        <RequireAuth pathname="/library">
          <p>PRIVATE LIBRARY</p>
        </RequireAuth>
      </AuthProvider>,
    );
    // No ?next: /library is already where a signed-in user lands, so
    // carrying it would be noise in the URL for no behavioural gain.
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/login"));
    expect(screen.queryByText("PRIVATE LIBRARY")).toBeNull();
  });

  it("renders the page once the session is confirmed", async () => {
    stubApi({ me: { status: 200, body: USER } });
    render(
      <AuthProvider>
        <RequireAuth pathname="/library">
          <p>PRIVATE LIBRARY</p>
        </RequireAuth>
      </AuthProvider>,
    );
    expect(await screen.findByText("PRIVATE LIBRARY")).toBeInTheDocument();
    expect(replace).not.toHaveBeenCalled();
  });
});

// ── login ─────────────────────────────────────────────────────────────

describe("login", () => {
  it("signs in and goes to the library", async () => {
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.type(screen.getByLabelText("Email"), "person@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/library"));
  });

  it("returns the user to where they were headed", async () => {
    searchParams = new URLSearchParams("next=%2Fsong%2Fabc");
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.type(screen.getByLabelText("Email"), "person@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/song/abc"));
  });

  it("refuses to follow a foreign destination", async () => {
    searchParams = new URLSearchParams("next=https%3A%2F%2Fevil.example");
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.type(screen.getByLabelText("Email"), "person@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/library"));
    expect(replace).not.toHaveBeenCalledWith("https://evil.example");
  });

  it("shows the server's message when credentials are wrong", async () => {
    stubApi({
      me: { status: 401 },
      login: { status: 401, body: { detail: "Email or password is incorrect." } },
    });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.type(screen.getByLabelText("Email"), "person@example.com");
    await user.type(screen.getByLabelText("Password"), "wrong password here");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Email or password is incorrect.",
    );
    expect(replace).not.toHaveBeenCalled();
  });

  it("asks for both fields before calling the server", async () => {
    const calls = stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(calls.filter((c) => c.includes("/login"))).toHaveLength(0);
  });

  it("sends an already-signed-in visitor into the product", async () => {
    stubApi({ me: { status: 200, body: USER } });
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/library"));
  });

  it("offers no password reset, because none exists", async () => {
    stubApi({ me: { status: 401 } });
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    expect(screen.queryByText(/forgot/i)).toBeNull();
  });
});

// ── signup ────────────────────────────────────────────────────────────

describe("signup", () => {
  async function fill(user: ReturnType<typeof userEvent.setup>, password: string, confirm = password) {
    await user.type(screen.getByLabelText("Email"), "new@example.com");
    await user.type(screen.getByLabelText("Password"), password);
    await user.type(screen.getByLabelText("Confirm password"), confirm);
  }

  it("creates the account and enters the product", async () => {
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <SignupPage />
      </AuthProvider>,
    );
    await fill(user, "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/library"));
  });

  it("enforces the same minimum length the backend does", async () => {
    const calls = stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <SignupPage />
      </AuthProvider>,
    );
    await fill(user, "short");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("at least 10 characters");
    expect(calls.filter((c) => c.includes("/signup"))).toHaveLength(0);
  });

  it("catches a mistyped confirmation before the server does", async () => {
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <SignupPage />
      </AuthProvider>,
    );
    await fill(user, "correct horse battery", "correct horse batteries");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("do not match");
  });

  it("reports a duplicate email in the server's words", async () => {
    stubApi({
      me: { status: 401 },
      signup: { status: 409, body: { detail: "An account with that email already exists." } },
    });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <SignupPage />
      </AuthProvider>,
    );
    await fill(user, "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already exists");
  });
});

// ── secrets ───────────────────────────────────────────────────────────

describe("nothing secret is stored client-side", () => {
  it("writes no token to localStorage or sessionStorage", async () => {
    stubApi({ me: { status: 401 } });
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );
    await user.type(screen.getByLabelText("Email"), "person@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(replace).toHaveBeenCalled());

    // The session is an HttpOnly cookie; there is nothing to keep.
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
    const stored = JSON.stringify({ ...localStorage, ...sessionStorage });
    expect(stored).not.toMatch(/password|token|session/i);
  });

  it("sends credentials on the auth calls so the cookie travels", async () => {
    stubApi({ me: { status: 200, body: USER } });
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("authenticated"));
    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls as unknown[][];
    const meCall = calls.find((args) => String(args[0]).includes("/v1/auth/me"));
    expect((meCall?.[1] as RequestInit | undefined)?.credentials).toBe("include");
  });
});

// ── cache isolation ───────────────────────────────────────────────────

describe("private cache does not survive the session", () => {
  it("is cleared on sign-out", async () => {
    const { clearPrivateGenerationCache } = await import("@/lib/generationStorage");
    localStorage.setItem("luber.recentGenerations", JSON.stringify([{ id: "a", title: "A SONG" }]));
    localStorage.setItem("luber.activeGenerationId", "a");

    clearPrivateGenerationCache();

    expect(localStorage.getItem("luber.recentGenerations")).toBeNull();
    expect(localStorage.getItem("luber.activeGenerationId")).toBeNull();
  });

  it("leaves no trace of the previous user's song titles", async () => {
    const { clearPrivateGenerationCache } = await import("@/lib/generationStorage");
    localStorage.setItem(
      "luber.recentGenerations",
      JSON.stringify([{ id: "x", title: "USER A PRIVATE SONG" }]),
    );

    clearPrivateGenerationCache();

    expect(JSON.stringify({ ...localStorage })).not.toContain("USER A PRIVATE SONG");
  });

  it("signing out empties the cache the next user would read", async () => {
    stubApi({ me: { status: 200, body: USER } });
    localStorage.setItem(
      "luber.recentGenerations",
      JSON.stringify([{ id: "x", title: "USER A PRIVATE SONG" }]),
    );

    function SignOutButton() {
      const { signOut } = useAuth();
      return <button onClick={() => void signOut()}>Sign out</button>;
    }
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <SignOutButton />
      </AuthProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Sign out" }));

    await waitFor(() => expect(replace).toHaveBeenCalledWith("/login"));
    expect(JSON.stringify({ ...localStorage })).not.toContain("USER A PRIVATE SONG");
  });
});
