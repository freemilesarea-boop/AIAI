/**
 * BOORDA's six product areas, and the honesty of the ones with no
 * backend behind them.
 *
 * Home replaces a redirect, so the first test here is that `/` is a page
 * at all. The rest guard the thing most likely to rot: LAB, Plans and
 * most of Settings render placeholders for capabilities that do not
 * exist, and the moment one of those turns into a plausible-looking
 * number or a button that does nothing, the product is lying to the
 * user. These tests fail if that happens.
 *
 * The LAB assertions are the sharpest of them: every "체험하기" must be
 * genuinely disabled, not merely styled grey.
 */

import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthProvider } from "@/components/auth/AuthProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { ToastProvider } from "@/components/ui/Toast";
import HomePage from "@/app/page";
import PlansPage from "@/app/plans/page";
import LabPage from "@/app/lab/page";
import SettingsPage from "@/app/settings/page";
import { PLANS, creditBalance, currentPlan, formatCredits, formatPriceKrw } from "@/lib/plans";
import { PREVIEW_ENTRIES, isUsable, labCatalog } from "@/lib/lab";

let pathname = "/";

vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
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

/** Answers `/me` with a signed-in user and every list with nothing. */
function stubApi(generations: unknown[] = []) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/auth/me")) return json(USER);
      if (url.includes("/generations")) return json({ items: generations });
      return json({ items: [] });
    }),
  );
}

function renderPage(node: React.ReactNode) {
  return render(
    <AuthProvider>
      <PlayerProvider>
        <ToastProvider>{node}</ToastProvider>
      </PlayerProvider>
    </AuthProvider>,
  );
}

beforeEach(() => {
  pathname = "/";
  vi.unstubAllGlobals();
});

describe("home", () => {
  it("is a real page rather than a redirect to create", async () => {
    stubApi();
    renderPage(<HomePage />);
    expect(
      await screen.findByRole("heading", { name: /오늘은 어떤 음악을 만들까요/ }),
    ).toBeInTheDocument();
  });

  it("puts creating music in front of everything else", async () => {
    stubApi();
    renderPage(<HomePage />);
    const cta = await screen.findAllByRole("link", { name: "음악 만들기" });
    expect(cta[0]).toHaveAttribute("href", "/create");
  });

  it("shows plan and credits as undecided rather than inventing figures", async () => {
    stubApi();
    renderPage(<HomePage />);
    const account = await screen.findByRole("region", { name: "내 계정" });
    // Two stats, both undecided. A number here would be fabricated.
    expect(within(account).getAllByText("미정")).toHaveLength(2);
  });

  it("offers a way into the library", async () => {
    stubApi();
    renderPage(<HomePage />);
    expect(
      await screen.findByRole("link", { name: "라이브러리 전체 보기" }),
    ).toHaveAttribute("href", "/library");
  });

  it("shows recent tracks when there are any", async () => {
    stubApi([
      {
        id: "g1",
        title: "밤 산책",
        prompt: "night walk",
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
      },
    ]);
    renderPage(<HomePage />);
    expect(await screen.findByText("밤 산책")).toBeInTheDocument();
  });
});

describe("home lab entry", () => {
  it("points at LAB without displacing Create as the primary action", async () => {
    stubApi();
    renderPage(<HomePage />);
    const lab = await screen.findByRole("link", { name: /BOORDA LAB/ });
    expect(lab).toHaveAttribute("href", "/lab");
    // Create is still the CTA the page is built around.
    const create = screen.getAllByRole("link", { name: "음악 만들기" });
    expect(create[0]).toHaveAttribute("href", "/create");
  });
});

describe("lab", () => {
  it("exists and announces itself as a preview area", () => {
    renderPage(<LabPage />);
    expect(screen.getByRole("heading", { name: "BOORDA LAB" })).toBeInTheDocument();
    expect(
      screen.getByText("새로운 모델과 실험 기능을 가장 먼저 만나보세요."),
    ).toBeInTheDocument();
  });

  it("lists every catalogue entry with a status badge and a version", () => {
    renderPage(<LabPage />);
    for (const entry of PREVIEW_ENTRIES) {
      expect(screen.getByRole("heading", { name: entry.name })).toBeInTheDocument();
      expect(screen.getAllByText(entry.version).length).toBeGreaterThan(0);
    }
  });

  it("never offers a working action for something that is not wired", () => {
    renderPage(<LabPage />);
    const tryButtons = screen.getAllByRole("button", { name: "체험하기" });
    expect(tryButtons).toHaveLength(PREVIEW_ENTRIES.length);
    // Disabled in fact, and stated for assistive tech — not merely greyed.
    for (const button of tryButtons) {
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute("aria-disabled", "true");
    }
  });

  it("says plainly that nothing is available yet", () => {
    renderPage(<LabPage />);
    expect(screen.getAllByText("아직 불가")).toHaveLength(PREVIEW_ENTRIES.length);
  });

  it("warns on the experimental entries and not on the stable one", () => {
    renderPage(<LabPage />);
    const warned = PREVIEW_ENTRIES.filter((e) => e.experimentalWarning).length;
    expect(screen.getAllByRole("note")).toHaveLength(warned);
    expect(warned).toBeGreaterThan(0);
  });

  it("shows the promotion path a model follows", () => {
    renderPage(<LabPage />);
    expect(screen.getByRole("heading", { name: "모델이 공개되는 단계" })).toBeInTheDocument();
    expect(screen.getByText("DEFAULT")).toBeInTheDocument();
  });
});

describe("lab catalogue configuration", () => {
  it("marks every entry unavailable while no model API exists", () => {
    const catalog = labCatalog();
    for (const entry of catalog.entries) {
      expect(entry.available).toBe(false);
      expect(isUsable(entry, catalog)).toBe(false);
    }
  });

  it("claims no default model and grants access to none", () => {
    const catalog = labCatalog();
    expect(catalog.defaultModelId).toBeNull();
    expect(catalog.accessibleModelIds).toHaveLength(0);
  });

  it("treats availability and per-account access as separate gates", () => {
    const catalog = labCatalog();
    const entry = { ...catalog.entries[0], available: true };
    // Available but not granted to this account: still not usable.
    expect(isUsable(entry, catalog)).toBe(false);
    expect(isUsable(entry, { ...catalog, accessibleModelIds: [entry.id] })).toBe(true);
  });
});

describe("plans", () => {
  it("names the three tiers", () => {
    renderPage(<PlansPage />);
    for (const name of ["Free", "Basic", "Pro"]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
  });

  it("quotes no price, because none has been decided", () => {
    renderPage(<PlansPage />);
    // Every tier's price reads "미정"; a ₩ figure would be invented.
    expect(screen.getAllByText("미정").length).toBeGreaterThanOrEqual(PLANS.length);
    expect(screen.queryByText(/₩/)).not.toBeInTheDocument();
  });

  it("offers no subscribe control while there is nothing to subscribe to", () => {
    renderPage(<PlansPage />);
    expect(screen.queryByRole("button", { name: /구독|결제|시작하기/ })).not.toBeInTheDocument();
    expect(screen.getAllByText("결제는 아직 준비 중입니다.")).toHaveLength(PLANS.length);
  });
});

describe("settings", () => {
  it("shows the account the session actually reports", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    expect(await screen.findByText("singer@boorda.kr")).toBeInTheDocument();
    expect(screen.getByText("부르다")).toBeInTheDocument();
  });

  it("has all six account-management sections", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    for (const name of ["계정", "구독", "크레딧", "결제", "데이터", "보안"]) {
      expect(await screen.findByRole("heading", { name })).toBeInTheDocument();
    }
  });

  it("marks every unbacked capability as unavailable", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    // One per section that has no backend: account edit, subscription,
    // credits, payments, data export/deletion, security extras.
    expect(await screen.findAllByText("준비 중")).toHaveLength(6);
  });

  it("reports subscription, credits and payments as undecided rather than zero", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    // plan, status, renewal, balance, monthly grant, method, last payment
    expect(await screen.findAllByText("미정")).toHaveLength(7);
  });

  it("offers no cancel, upgrade, or delete control while none of them works", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    await screen.findByRole("heading", { name: "계정" });
    for (const label of [/해지/, /업그레이드/, /삭제/, /저장/, /변경하기/]) {
      expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    }
  });

  it("sends plan comparison to /plans instead of duplicating it", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    expect(await screen.findByRole("link", { name: "플랜 비교하기" })).toHaveAttribute(
      "href", "/plans",
    );
    // The tier cards themselves belong to /plans only.
    expect(screen.queryByRole("heading", { name: "Free" })).not.toBeInTheDocument();
  });

  it("signs out through the auth the product already has", async () => {
    stubApi();
    renderPage(<SettingsPage />);
    expect(await screen.findByRole("button", { name: "로그아웃" })).toBeInTheDocument();
  });
});

describe("plan configuration", () => {
  it("leaves every price and limit undecided", () => {
    for (const plan of PLANS) {
      expect(plan.monthlyPriceKrw).toBeNull();
      expect(plan.monthlyCredits).toBeNull();
      expect(plan.maxDurationSeconds).toBeNull();
    }
  });

  it("reports no current plan and no balance until a backend exists", () => {
    expect(currentPlan()).toBeNull();
    expect(creditBalance()).toBeNull();
  });

  it("renders an undecided figure as undecided, not as zero or free", () => {
    expect(formatPriceKrw(null)).toBe("미정");
    expect(formatCredits(null)).toBe("미정");
    // And once decided, it formats properly rather than staying vague.
    expect(formatPriceKrw(0)).toBe("무료");
    expect(formatPriceKrw(9900)).toBe("₩9,900");
    expect(formatCredits(120)).toBe("120");
  });
});
