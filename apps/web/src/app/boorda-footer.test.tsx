/**
 * The footer, tested where it actually has to appear.
 *
 * The previous round shipped a `Footer` component with passing tests and
 * a site with no visible footer, because every test rendered the
 * component directly. A component that renders correctly in isolation
 * and is mounted on three pages out of thirty is not a global footer.
 *
 * So these tests drive `AppShell` at real pathnames — the same component
 * the root layout mounts — and ask what a user would actually see.
 */

import { render, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/shell/AppShell";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { BUSINESS } from "@/lib/legal";

let pathname = "/";
vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  redirect: vi.fn(),
}));

const USER = {
  id: "u1",
  email: "singer@boorda.kr",
  display_name: null,
  created_at: "2026-01-15T00:00:00Z",
  role: "USER",
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** The real tree: root layout mounts AuthProvider → PlayerProvider → AppShell. */
function renderAt(path: string) {
  pathname = path;
  return render(
    <AuthProvider>
      <PlayerProvider>
        <AppShell>
          <p>페이지 본문</p>
        </AppShell>
      </PlayerProvider>
    </AuthProvider>,
  );
}

function footer() {
  return document.querySelector("footer");
}

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER);
      return json({ items: [], plans: [] });
    }),
  );
  window.scrollTo = vi.fn();
});

// ── where it appears ─────────────────────────────────────────────────

describe("footer mounting", () => {
  /**
   * The regression this file exists for. Every one of these is a route
   * a real user lands on, and the previous implementation had a footer
   * on none of them.
   */
  const PRODUCT_ROUTES = [
    "/",
    "/create",
    "/library",
    "/lab",
    "/plans",
    "/settings",
    "/support",
    "/support/contact",
    "/song/abc",
    "/projects",
  ];

  it.each(PRODUCT_ROUTES)("renders on %s", (path) => {
    renderAt(path);

    expect(footer()).not.toBeNull();
    expect(within(footer()!).getByRole("link", { name: "개인정보처리방침" })).toBeInTheDocument();
  });

  const LEGAL_ROUTES = ["/privacy", "/terms", "/refund-policy"];

  it.each(LEGAL_ROUTES)("still renders on %s", (path) => {
    renderAt(path);

    expect(footer()).not.toBeNull();
  });

  it("gives signed-out visitors a way to the terms before they agree", () => {
    /**
     * Sign-in is deliberately free of product chrome, but "no chrome"
     * must not mean "no way to read the terms" at the moment somebody
     * is deciding whether to accept them.
     */
    renderAt("/login");

    const legal = within(footer()!).getByRole("navigation", { name: "법적 고지" });
    expect(within(legal).getByRole("link", { name: "이용약관" })).toBeInTheDocument();
    expect(within(legal).getByRole("link", { name: "개인정보처리방침" })).toBeInTheDocument();
  });
});

// ── where it must not appear ─────────────────────────────────────────

describe("admin isolation", () => {
  const ADMIN_ROUTES = ["/admin", "/admin/users", "/admin/acquisition", "/admin/revenue"];

  it.each(ADMIN_ROUTES)("does not put a consumer footer on %s", (path) => {
    /**
     * The console renders inside the product shell — it needs the
     * sidebar and the session — so excluding it has to be deliberate.
     * A back office does not carry business registration details.
     */
    renderAt(path);

    expect(footer()).toBeNull();
  });

  it("does not reach the operator training console either", () => {
    renderAt("/ops/training");

    expect(footer()).toBeNull();
  });
});

// ── what it contains ─────────────────────────────────────────────────

describe("footer content", () => {
  beforeEach(() => {
    renderAt("/");
  });

  it("links every legal document to its real route", () => {
    const scope = within(footer()!);

    expect(scope.getByRole("link", { name: "이용약관" })).toHaveAttribute("href", "/terms");
    expect(scope.getByRole("link", { name: "개인정보처리방침" })).toHaveAttribute(
      "href",
      "/privacy",
    );
    expect(scope.getByRole("link", { name: "구독·결제·환불 정책" })).toHaveAttribute(
      "href",
      "/refund-policy",
    );
  });

  it("links support and the product routes that exist", () => {
    const scope = within(footer()!);

    expect(scope.getByRole("link", { name: "고객지원" })).toHaveAttribute("href", "/support");
    expect(scope.getByRole("link", { name: "문의하기" })).toHaveAttribute(
      "href",
      "/support/contact",
    );
    expect(scope.getByRole("link", { name: "음악 만들기" })).toHaveAttribute("href", "/create");
    expect(scope.getByRole("link", { name: "요금제" })).toHaveAttribute("href", "/plans");
  });

  it("shows the approved business identity from the shared config", () => {
    const scope = within(footer()!);

    expect(scope.getByText(BUSINESS.name)).toBeInTheDocument();
    expect(scope.getByText(BUSINESS.representative)).toBeInTheDocument();
    expect(scope.getByText(BUSINESS.registrationNumber)).toBeInTheDocument();
    expect(scope.getByRole("link", { name: BUSINESS.contactEmail })).toHaveAttribute(
      "href",
      `mailto:${BUSINESS.contactEmail}`,
    );
  });

  it("shows no field nobody has verified", () => {
    const text = footer()!.textContent ?? "";

    expect(text).not.toMatch(/주소/);
    expect(text).not.toMatch(/통신판매업/);
    expect(text).not.toMatch(/전화/);
    expect(text).not.toMatch(/보호책임자/);
  });

  it("claims copyright for BOORDA", () => {
    expect(within(footer()!).getByText(/© 2026 BOORDA/)).toBeInTheDocument();
    expect(footer()!.textContent).not.toMatch(/SRR/);
  });

  it("hardcodes none of the business values in the component", async () => {
    /**
     * One source of truth. A second copy in the component is a second
     * thing to correct when the registration number changes, and the
     * one nobody remembers.
     */
    const { readFileSync } = await import("node:fs");
    const { resolve } = await import("node:path");
    // Resolved from the vitest working directory (apps/web) rather than
    // import.meta.url, which is not a file URL under the test transform.
    const source = readFileSync(resolve("src/components/Footer.tsx"), "utf8");

    for (const value of [
      BUSINESS.name,
      BUSINESS.representative,
      BUSINESS.registrationNumber,
      BUSINESS.contactEmail,
    ]) {
      expect(source).not.toContain(value);
    }
    expect(source).toContain("BUSINESS");
  });
});

// ── layout behaviour ─────────────────────────────────────────────────

describe("layout", () => {
  it("sits after the content, not pinned over it", () => {
    /**
     * A fixed footer would cover the player bar and the last row of a
     * long list. It follows the content instead, pushed down by a
     * flex-1 spacer on short pages.
     */
    renderAt("/");
    const element = footer()!;

    expect(element.className).not.toMatch(/\bfixed\b/);
    expect(element.className).not.toMatch(/\babsolute\b/);
    expect(element.className).not.toMatch(/\bsticky\b/);
  });

  it("is reachable while the session is still resolving", () => {
    /**
     * Outside RequireAuth on purpose: someone being redirected to sign
     * in still deserves a route to the privacy policy.
     */
    renderAt("/library");

    expect(footer()).not.toBeNull();
  });

  it("wraps rather than overflowing on a narrow screen", () => {
    renderAt("/");
    const element = footer()!;

    // The business line and the link columns both wrap. `max-w-` is a
    // ceiling and cannot overflow; a fixed `w-[…px]` could, so only that
    // is disallowed.
    expect(element.querySelector("dl")?.className).toMatch(/flex-wrap/);
    const fixedWidth = /(?<![a-z-])w-\[\d{3,}px\]/;
    expect(element.innerHTML).not.toMatch(fixedWidth);
  });
});
