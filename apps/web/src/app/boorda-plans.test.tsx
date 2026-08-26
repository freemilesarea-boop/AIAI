/**
 * What the product shows about plans, allowance and downloads.
 *
 * The server is what enforces every rule here — these tests do not and
 * cannot prove enforcement, and none of them claims to. What they prove
 * is the other half: that the interface tells the truth about the rules,
 * before the user runs into them.
 *
 * Three things in particular are worth guarding:
 *
 *  - No figure on any screen is invented. Every price, limit and count
 *    comes from the server's response, so a stub that changes the number
 *    must change the screen.
 *  - An exhausted account cannot press 만들기. The server would refuse
 *    it anyway; a form that submits into a guaranteed 402 is a form that
 *    wastes the user's time to prove a point.
 *  - A Free account's download control goes to /plans instead of to a
 *    refusal. Locking it is presentation; the request is still refused
 *    server-side if anyone gets past it.
 */

import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CreatePage from "@/app/create/page";
import PlansPage from "@/app/plans/page";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { EntitlementProvider } from "@/components/EntitlementProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { SongCard } from "@/components/SongCard";
import { ToastProvider } from "@/components/ui/Toast";
import type { Entitlement } from "@/lib/plans";
import {
  entitlementFixture,
  exhaustedFixture,
  planCatalogueFixture,
} from "@/test/entitlement-factories";
import { generation as makeGeneration } from "@/test/factories";

vi.mock("next/navigation", () => ({
  usePathname: () => "/create",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
}));

const USER = {
  id: "u1",
  email: "singer@boorda.kr",
  display_name: "부르다",
  created_at: "2026-01-15T00:00:00Z",
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stub(entitlement: Entitlement) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER);
      if (url.includes("/account/entitlement")) return json(entitlement);
      if (url.includes("/v1/plans")) return json(planCatalogueFixture());
      if (url.includes("/generations")) return json({ items: [] });
      return json({ items: [] });
    }),
  );
}

function renderPage(node: React.ReactNode) {
  return render(
    <AuthProvider>
      <EntitlementProvider>
        <PlayerProvider>
          <ToastProvider>{node}</ToastProvider>
        </PlayerProvider>
      </EntitlementProvider>
    </AuthProvider>,
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

// ── the pricing page ─────────────────────────────────────────────────

describe("plans page", () => {
  it("renders the server's figures rather than its own", async () => {
    stub(entitlementFixture("free"));
    renderPage(<PlansPage />);

    const creator = await screen.findByTestId("plan-creator");
    expect(within(creator).getByText("₩49,900")).toBeInTheDocument();
    expect(within(creator).getByText("1,000곡")).toBeInTheDocument();

    const free = screen.getByTestId("plan-free");
    expect(within(free).getByText("무료")).toBeInTheDocument();
    expect(within(free).getByText("20곡")).toBeInTheDocument();
  });

  it("marks the account's own tier", async () => {
    stub(entitlementFixture("pro"));
    renderPage(<PlansPage />);

    const pro = await screen.findByTestId("plan-pro");
    expect(within(pro).getByText("현재 플랜")).toBeInTheDocument();
    expect(within(screen.getByTestId("plan-free")).queryByText("현재 플랜")).toBeNull();
  });

  it("shows what Free does and does not include, without softening it", async () => {
    stub(entitlementFixture("free"));
    renderPage(<PlansPage />);

    const free = await screen.findByTestId("plan-free");
    // Two downloads and commercial use, all excluded. A tier that showed
    // a dash here would be hiding a "no" behind an "undecided".
    expect(within(free).getAllByLabelText("미포함")).toHaveLength(3);
    expect(within(free).queryAllByLabelText("포함")).toHaveLength(0);
  });

  it("shows the account's current usage beside the tiers", async () => {
    stub(entitlementFixture("basic", 40));
    renderPage(<PlansPage />);

    expect(await screen.findByText("40곡 / 200곡")).toBeInTheDocument();
  });
});

// ── the allowance meter ──────────────────────────────────────────────

describe("usage on create", () => {
  it("shows how much of the month is spent", async () => {
    stub(entitlementFixture("basic", 25));
    renderPage(<CreatePage />);

    expect(await screen.findByText("25곡 / 200곡")).toBeInTheDocument();
    const bar = screen.getByRole("progressbar", { name: "이번 달 생성 사용량" });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(bar).toHaveAttribute("aria-valuemax", "200");
  });

  it("stays quiet while there is plenty left", async () => {
    stub(entitlementFixture("basic", 25));
    renderPage(<CreatePage />);

    await screen.findByText("25곡 / 200곡");
    // A banner shown at 25 out of 200 is a banner nobody reads at 199.
    expect(screen.queryByText(/남았습니다/)).toBeNull();
  });

  it("warns once the allowance is nearly gone", async () => {
    stub(entitlementFixture("free", 18));
    renderPage(<CreatePage />);

    expect(await screen.findByText("2곡 남았습니다")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "플랜 보기" })).toHaveAttribute("href", "/plans");
  });

  it("explains an exhausted allowance and where it resets", async () => {
    stub(exhaustedFixture("free"));
    renderPage(<CreatePage />);

    expect(await screen.findByText("이번 달 생성 한도를 모두 사용했습니다")).toBeInTheDocument();
    expect(screen.getByText(/다음 기간에 다시 초기화되며/)).toBeInTheDocument();
  });

  it("does not let an exhausted account submit into a guaranteed refusal", async () => {
    stub(exhaustedFixture("free"));
    renderPage(<CreatePage />);

    await screen.findByText("이번 달 생성 한도를 모두 사용했습니다");
    expect(await screen.findByRole("button", { name: "음악 만들기" })).toBeDisabled();
  });

  it("leaves the form usable when the allowance is merely low", async () => {
    stub(entitlementFixture("free", 19));
    renderPage(<CreatePage />);

    await screen.findByText("1곡 남았습니다");
    // One song left is one song, not zero. The last slot is usable.
    expect(await screen.findByRole("button", { name: "음악 만들기" })).toBeEnabled();
  });
});

// ── downloads ────────────────────────────────────────────────────────

describe("download entitlement", () => {
  const song = makeGeneration({ id: "g1", title: "Midnight Window", status: "COMPLETED" });

  it("offers the download on a plan that includes it", async () => {
    stub(entitlementFixture("basic"));
    renderPage(<SongCard generation={song} />);

    const link = await screen.findByRole("link", { name: "WAV" });
    expect(link).toHaveAttribute("download");
    expect(link.getAttribute("href")).toContain("/generations/g1/audio");
  });

  it("sends a Free account to the plans page instead of to a refusal", async () => {
    stub(entitlementFixture("free"));
    renderPage(<SongCard generation={song} />);

    // Waited for by its title rather than its name: both states are
    // named "WAV", and the unlocked one renders first while the
    // entitlement is still in flight.
    const link = await screen.findByTitle("다운로드는 유료 플랜에 포함됩니다");
    expect(link).toHaveAttribute("href", "/plans");
    expect(link).not.toHaveAttribute("download");
  });

  it("still lets a Free account play the song it made", async () => {
    stub(entitlementFixture("free"));
    renderPage(<SongCard generation={song} />);

    // The plan gates the save, not the song. Nothing about playback is
    // conditional on the tier.
    expect(await screen.findByRole("button", { name: /재생|Play/i })).toBeEnabled();
  });
});
