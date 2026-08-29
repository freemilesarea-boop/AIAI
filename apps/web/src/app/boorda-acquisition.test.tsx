/**
 * The acquisition console and the beacon behind it.
 *
 * Two things carry weight here. The beacon must never be the reason a
 * page misbehaves — it reports once per tab, skips the console, and
 * forwards only campaign parameters. And the dashboard must not invent
 * a channel for accounts that predate attribution: "we do not know" is
 * a real answer and the page says it in words.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AcquisitionPage from "@/app/admin/acquisition/page";
import { AdminShell } from "@/components/admin/AdminShell";
import {
  CampaignLinkBuilder,
  buildCampaignUrl,
  normaliseDestination,
  normaliseValue,
} from "@/components/admin/CampaignLinkBuilder";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { ToastProvider } from "@/components/ui/Toast";
import { formatRate } from "@/lib/admin";
import { TRACKED_PARAMS, shouldTrack, trackedParams } from "@/lib/acquisition";

vi.mock("next/navigation", () => ({
  usePathname: () => "/admin/acquisition",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams("from=2026-08-01&to=2026-08-28"),
  useParams: () => ({}),
  redirect: vi.fn(),
}));

const ADMIN = {
  id: "u1",
  email: "operator@boorda.kr",
  display_name: null,
  created_at: "2026-01-15T00:00:00Z",
  role: "ADMIN",
};

const SUMMARY = {
  range: { start: "2026-08-01", end: "2026-08-28", days: 28, bucketing: "day" },
  mode: "first_touch",
  visitors: 120,
  signups: 12,
  conversions: 3,
  revenue_krw: 59700,
  signup_rate: 0.1,
  conversion_rate: 0.025,
  unattributed_users: 3,
};

const CHANNELS = [
  {
    key: "instagram_ads",
    label: "Instagram 광고",
    source: "instagram",
    medium: "paid_social",
    visitors: 80,
    signups: 9,
    conversions: 2,
    revenue_krw: 39800,
    signup_rate: 0.1125,
    conversion_rate: 0.025,
  },
  {
    key: "google_organic",
    label: "Google 검색",
    source: "google",
    medium: "organic",
    visitors: 40,
    signups: 3,
    conversions: 1,
    revenue_krw: 19900,
    signup_rate: 0.075,
    conversion_rate: 0.025,
  },
];

const CAMPAIGNS = [
  {
    source: "instagram",
    medium: "paid_social",
    campaign: "summer_launch",
    visitors: 50,
    signups: 6,
    conversions: 2,
    revenue_krw: 39800,
  },
  {
    source: "instagram",
    medium: "paid_social",
    campaign: "creator_campaign_01",
    visitors: 30,
    signups: 3,
    conversions: 0,
    revenue_krw: 0,
  },
];

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stub(handlers: Record<string, () => Response> = {}) {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      for (const [fragment, make] of Object.entries(handlers)) {
        if (url.includes(fragment)) return make();
      }
      if (url.includes("/auth/me")) return json(ADMIN);
      if (url.includes("/acquisition/summary")) return json(SUMMARY);
      if (url.includes("/acquisition/channels")) return json(CHANNELS);
      if (url.includes("/acquisition/campaigns")) return json(CAMPAIGNS);
      return json({});
    }),
  );
  return { calls };
}

function renderPage(node: React.ReactNode) {
  return render(
    <AuthProvider>
      <ToastProvider>{node}</ToastProvider>
    </AuthProvider>,
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  try {
    window.sessionStorage.clear();
  } catch {
    // Some environments refuse storage; the beacon copes and so does this.
  }
});

// ── the beacon ───────────────────────────────────────────────────────

describe("visit beacon", () => {
  it("forwards only campaign parameters", () => {
    /** A landing URL can carry a reset token; none of it belongs here. */
    const kept = trackedParams(
      "?utm_source=instagram&utm_campaign=summer&gclid=abc&access_token=secret&email=a@b.c",
    );

    expect(kept).toEqual({ utm_source: "instagram", utm_campaign: "summer", gclid: "abc" });
    expect(JSON.stringify(kept)).not.toContain("secret");
  });

  it("knows every parameter the server accepts", () => {
    expect([...TRACKED_PARAMS]).toEqual([
      "utm_source",
      "utm_medium",
      "utm_campaign",
      "utm_content",
      "utm_term",
      "gclid",
      "gbraid",
      "wbraid",
      "fbclid",
    ]);
  });

  it("does not count operator navigation as acquisition", () => {
    /** Our own traffic must not end up in the marketing report. */
    expect(shouldTrack("/")).toBe(true);
    expect(shouldTrack("/plans")).toBe(true);
    expect(shouldTrack("/admin")).toBe(false);
    expect(shouldTrack("/admin/acquisition")).toBe(false);
    expect(shouldTrack("/ops/training")).toBe(false);
  });

  it("reports once per tab, not once per navigation", async () => {
    const { reportVisit } = await import("@/lib/acquisition");
    const { calls } = stub();

    await reportVisit({ pathname: "/", search: "?utm_source=x" }, "https://instagram.com/");
    await reportVisit({ pathname: "/plans", search: "" }, "");

    expect(calls.filter((c) => c.url.includes("/acquisition/visit"))).toHaveLength(1);
  });

  it("sends the cookie, because the cookie is the whole point", async () => {
    const { reportVisit } = await import("@/lib/acquisition");
    const { calls } = stub();

    await reportVisit({ pathname: "/", search: "" }, "");

    const call = calls.find((c) => c.url.includes("/acquisition/visit"));
    expect(call?.init?.credentials).toBe("include");
  });

  it("stays silent when the request fails", async () => {
    /** Analytics that breaks the page it measures is not worth having. */
    const { reportVisit } = await import("@/lib/acquisition");
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));

    await expect(
      reportVisit({ pathname: "/", search: "" }, ""),
    ).resolves.toBeUndefined();
  });
});

// ── navigation ───────────────────────────────────────────────────────

describe("admin navigation", () => {
  it("offers 유입 분석 in the operator menu", async () => {
    stub();

    renderPage(
      <AdminShell>
        <p>본문</p>
      </AdminShell>,
    );

    await screen.findByText("본문");
    const nav = screen.getByRole("navigation", { name: "운영 관리" });
    expect(within(nav).getByRole("link", { name: "유입 분석" })).toBeInTheDocument();
  });
});

// ── the dashboard ────────────────────────────────────────────────────

describe("acquisition dashboard", () => {
  it("shows the funnel and both conversion rates", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    // Scoped to the KPI row: the same percentages legitimately appear
    // again in the channel table below, and an unscoped query would
    // pass on whichever rendered.
    const visitors = await screen.findByText("120명");
    const kpis = visitors.closest("div")!.parentElement!;
    expect(within(kpis).getByText("12명")).toBeInTheDocument();
    expect(within(kpis).getByText("3건")).toBeInTheDocument();
    expect(within(kpis).getByText("₩59,700")).toBeInTheDocument();
    expect(within(kpis).getByText("10.0%")).toBeInTheDocument();
    expect(within(kpis).getByText("2.5%")).toBeInTheDocument();
  });

  it("states what each rate divides by", async () => {
    /** A rate whose denominator is unstated is a rate nobody can check. */
    stub();

    renderPage(<AcquisitionPage />);

    expect(await screen.findByText("가입 ÷ 방문자")).toBeInTheDocument();
    expect(screen.getByText("첫 결제 ÷ 방문자")).toBeInTheDocument();
  });

  it("says the figures are event-period, not cohort", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    expect(
      await screen.findByText(/방문 · 가입 · 첫 결제가 해당 기간에 발생한 건/),
    ).toBeInTheDocument();
  });

  it("lists channels with their funnel", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    // Scoped to the table: "Instagram 광고" is also a preset button in
    // the link builder further down the page.
    const table = await screen.findByRole("table", { name: "유입 채널별 성과" });
    expect(within(table).getByText("Instagram 광고")).toBeInTheDocument();
    expect(within(table).getByText("Google 검색")).toBeInTheDocument();
    expect(within(table).getByText("₩39,800")).toBeInTheDocument();
  });

  it("breaks campaigns down under source and medium", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    expect(await screen.findByText("summer_launch")).toBeInTheDocument();
    expect(screen.getByText("creator_campaign_01")).toBeInTheDocument();
  });

  it("offers first-touch and last-touch, and explains the difference", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    expect(await screen.findByRole("tab", { name: "최초 유입" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "최종 유입" })).toBeInTheDocument();
    expect(screen.getByText(/전환을 처음 데려온 경로에 귀속/)).toBeInTheDocument();
  });

  it("reuses the shared date presets", async () => {
    stub();

    renderPage(<AcquisitionPage />);

    for (const label of ["오늘", "7일", "30일", "이번 달", "올해", "직접 선택"]) {
      expect(await screen.findByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("says existing members are excluded rather than calling them direct", async () => {
    /**
     * There is no evidence about where they came from, and inventing
     * one gives a channel budget it did not earn.
     */
    stub();

    renderPage(<AcquisitionPage />);

    expect(await screen.findByText(/유입 경로가 기록되지 않은 회원 3명/)).toBeInTheDocument();
    expect(screen.getByText(/직접 유입으로 분류하지 않습니다/)).toBeInTheDocument();
  });

  it("renders an intentional empty state before any data exists", async () => {
    stub({
      "/acquisition/summary": () =>
        json({ ...SUMMARY, visitors: 0, signups: 0, conversions: 0, revenue_krw: 0,
               signup_rate: null, conversion_rate: null, unattributed_users: 0 }),
      "/acquisition/channels": () => json([]),
      "/acquisition/campaigns": () => json([]),
    });

    renderPage(<AcquisitionPage />);

    expect(await screen.findByText("아직 수집된 유입 데이터가 없습니다")).toBeInTheDocument();
    expect(screen.getByText(/UTM 링크를 사용하거나/)).toBeInTheDocument();
  });

  it("shows a dash rather than a rate when nobody visited", () => {
    expect(formatRate(null)).toBe("—");
    expect(formatRate(0.1234)).toBe("12.3%");
  });

  it("reports a failure instead of an empty dashboard", async () => {
    stub({ "/acquisition/summary": () => json({ detail: "boom" }, 500) });

    renderPage(<AcquisitionPage />);

    expect(await screen.findByText(/불러오지 못했습니다/)).toBeInTheDocument();
  });
});

// ── the link builder ─────────────────────────────────────────────────

describe("campaign link builder", () => {
  it("builds a standard UTM link", () => {
    expect(
      buildCampaignUrl({
        destination: "/",
        source: "instagram",
        medium: "paid_social",
        campaign: "august_launch",
        content: "reel_01",
      }),
    ).toBe(
      "https://boorda.kr/?utm_source=instagram&utm_medium=paid_social&utm_campaign=august_launch&utm_content=reel_01",
    );
  });

  it("normalises values the way the server will", () => {
    /** Otherwise `Instagram` and `instagram ` become two rows. */
    expect(normaliseValue("  Instagram  ")).toBe("instagram");
    expect(normaliseValue("Summer Launch")).toBe("summer_launch");
  });

  it("cannot be pointed off BOORDA", () => {
    expect(normaliseDestination("https://evil.example/phish")).toBe("/phish");
    expect(normaliseDestination("plans")).toBe("/plans");
    expect(buildCampaignUrl({
      destination: "https://evil.example/x",
      source: "s", medium: "m", campaign: "c",
    })).toContain("https://boorda.kr/x?");
  });

  it("drops a query the caller tried to smuggle into the destination", () => {
    expect(normaliseDestination("/plans?token=secret")).toBe("/plans");
  });

  it("will not copy an incomplete link", async () => {
    render(
      <ToastProvider>
        <CampaignLinkBuilder />
      </ToastProvider>,
    );

    expect(screen.getByRole("button", { name: "복사" })).toBeDisabled();
    expect(screen.getByText(/소스 · 매체 · 캠페인을 모두 입력하면/)).toBeInTheDocument();
  });

  it("enables copying once the required fields are filled", async () => {
    render(
      <ToastProvider>
        <CampaignLinkBuilder />
      </ToastProvider>,
    );

    await userEvent.type(screen.getByLabelText("캠페인 (utm_campaign)"), "august_launch");

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "복사" })).toBeEnabled(),
    );
    const result = screen.getByLabelText("생성된 링크") as HTMLInputElement;
    expect(result.value).toContain("utm_campaign=august_launch");
    expect(result.value.startsWith("https://boorda.kr/")).toBe(true);
  });

  it("offers presets that still produce standard parameters", async () => {
    render(
      <ToastProvider>
        <CampaignLinkBuilder />
      </ToastProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: "YouTube 광고" }));

    expect(screen.getByLabelText("소스 (utm_source)")).toHaveValue("youtube");
    expect(screen.getByLabelText("매체 (utm_medium)")).toHaveValue("paid_video");
  });
});
