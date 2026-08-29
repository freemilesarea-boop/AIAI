/**
 * The legal layer.
 *
 * These tests exist to stop the two failure modes that matter for legal
 * text. The first is drift: a page saying something the product does
 * not do. The second is invention: a plausible-looking value nobody
 * confirmed, which is worse than a visible gap because only one of the
 * two ever gets fixed.
 */

import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PrivacyPage from "@/app/privacy/page";
import RefundPolicyPage from "@/app/refund-policy/page";
import TermsPage from "@/app/terms/page";
import { Footer } from "@/components/Footer";
import {
  ACQUISITION_RETENTION_MONTHS,
  BUSINESS,
  COOKIES,
  EFFECTIVE_DATE,
  HISTORY,
  PROCESSORS,
} from "@/lib/legal";

vi.mock("next/navigation", () => ({
  usePathname: () => "/privacy",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  redirect: vi.fn(),
}));

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PLANS = {
  plans: [
    {
      plan_id: "free",
      display_name: "Free",
      monthly_price_krw: 0,
      monthly_generation_limit: 20,
      download_mp3: false,
      download_wav: false,
      commercial_use: false,
    },
    {
      plan_id: "basic",
      display_name: "Basic",
      monthly_price_krw: 19900,
      monthly_generation_limit: 200,
      download_mp3: true,
      download_wav: true,
      commercial_use: true,
    },
  ],
  checkout_available: true,
};

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.stubGlobal("fetch", vi.fn(async () => json(PLANS)));
});

// ── footer ───────────────────────────────────────────────────────────

describe("footer", () => {
  it("links to every legal document and to support", () => {
    render(<Footer />);

    for (const name of ["이용약관", "개인정보처리방침", "구독·결제·환불 정책"]) {
      expect(screen.getByRole("link", { name })).toBeInTheDocument();
    }
    expect(screen.getByRole("link", { name: "고객지원" })).toHaveAttribute("href", "/support");
    expect(screen.getByRole("link", { name: "문의하기" })).toHaveAttribute(
      "href",
      "/support/contact",
    );
  });

  it("shows the business identity from the single configuration source", () => {
    render(<Footer />);

    expect(screen.getByText(BUSINESS.name)).toBeInTheDocument();
    expect(screen.getByText(BUSINESS.representative)).toBeInTheDocument();
    expect(screen.getByText(BUSINESS.registrationNumber)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: BUSINESS.contactEmail })).toHaveAttribute(
      "href",
      `mailto:${BUSINESS.contactEmail}`,
    );
  });

  it("omits fields nobody has confirmed rather than inventing them", () => {
    /**
     * A plausible placeholder on a legal notice is worse than a gap:
     * the gap gets fixed.
     */
    render(<Footer />);

    expect(screen.queryByText(/주소/)).not.toBeInTheDocument();
    expect(screen.queryByText(/통신판매업신고/)).not.toBeInTheDocument();
    expect(screen.queryByText(/전화/)).not.toBeInTheDocument();
  });

  it("claims copyright for BOORDA and nobody else", () => {
    render(<Footer />);

    expect(screen.getByText(/© 2026 BOORDA/)).toBeInTheDocument();
    expect(screen.queryByText(/SRR/)).not.toBeInTheDocument();
  });
});

// ── privacy ──────────────────────────────────────────────────────────

describe("privacy policy", () => {
  it("renders with its effective date", () => {
    render(<PrivacyPage />);

    expect(
      screen.getByRole("heading", { level: 1, name: "개인정보처리방침" }),
    ).toBeInTheDocument();
    expect(screen.getByText(`시행일: ${EFFECTIVE_DATE}`)).toBeInTheDocument();
  });

  it("discloses the acquisition analytics it actually runs", () => {
    render(<PrivacyPage />);

    // Appears in several sections by design — the disclosure is
    // repeated where it is relevant rather than buried once.
    expect(screen.getAllByText(/유입 경로/).length).toBeGreaterThan(0);
    expect(screen.getByText(/gclid/)).toBeInTheDocument();
    expect(screen.getByText(/익명 방문자 식별자/)).toBeInTheDocument();
  });

  it("names both cookies with their real attributes", () => {
    render(<PrivacyPage />);

    for (const cookie of COOKIES) {
      expect(screen.getByText(cookie.name)).toBeInTheDocument();
    }
    expect(screen.getAllByText(/HttpOnly/).length).toBeGreaterThan(0);
    expect(screen.getByText(/400일/)).toBeInTheDocument();
  });

  it("states the twelve-month raw analytics retention", () => {
    render(<PrivacyPage />);

    expect(ACQUISITION_RETENTION_MONTHS).toBe(12);
    expect(
      screen.getByText(new RegExp(`${ACQUISITION_RETENTION_MONTHS}개월간 보관`)),
    ).toBeInTheDocument();
  });

  it("separates statutory transaction retention from the analytics period", () => {
    /** Billing records are kept for years and must not be swept up. */
    render(<PrivacyPage />);

    expect(screen.getByText(/삭제 대상에서 제외/)).toBeInTheDocument();
    expect(screen.getByText(/대금결제 및 재화 등의 공급에 관한 기록/)).toBeInTheDocument();
  });

  it("describes account closure as anonymisation, not deletion", () => {
    /**
     * The product anonymises. Claiming immediate deletion would be a
     * false statement about our own code.
     */
    render(<PrivacyPage />);

    expect(screen.getAllByText(/익명 처리/).length).toBeGreaterThan(0);
    expect(screen.getByText(/복구할 수 없는 값으로 대체/)).toBeInTheDocument();
    expect(screen.queryByText(/모든 정보를 즉시 삭제/)).not.toBeInTheDocument();
  });

  it("does not claim to store card numbers", () => {
    render(<PrivacyPage />);

    expect(screen.getByText(/카드 정보를 저장하지 않습니다/)).toBeInTheDocument();
  });

  it("lists only processors that actually handle data", () => {
    render(<PrivacyPage />);

    for (const processor of PROCESSORS) {
      expect(screen.getByText(processor.name)).toBeInTheDocument();
    }
    // No email provider is configured, so none may be listed.
    expect(screen.queryByText(/Resend/)).not.toBeInTheDocument();
  });

  it("discloses that some processing happens outside Korea", () => {
    render(<PrivacyPage />);

    expect(screen.getByText(/싱가포르/)).toBeInTheDocument();
    expect(screen.getByText(/암스테르담/)).toBeInTheDocument();
  });

  it("is honest that IP is processed for rate limiting", () => {
    /** The audit found it; the policy says it. */
    render(<PrivacyPage />);

    expect(screen.getAllByText(/접속 IP 주소/).length).toBeGreaterThan(0);
  });
});

// ── terms ────────────────────────────────────────────────────────────

describe("terms", () => {
  it("renders", () => {
    render(<TermsPage />);

    expect(
      screen.getByRole("heading", { level: 1, name: "BOORDA 이용약관" }),
    ).toBeInTheDocument();
  });

  it("does not claim copyright automatically belongs to the user", () => {
    /**
     * The clause that matters. Whether an AI-assisted work attracts
     * copyright, and to whom, is a statutory question that depends on
     * human contribution and jurisdiction — not something a service can
     * decide in its own terms.
     */
    render(<TermsPage />);

    const body = document.body.textContent ?? "";
    expect(body).not.toMatch(/저작권은 항상 사용자에게/);
    expect(body).not.toMatch(/저작권은 이용자에게 있습니다/);
    expect(screen.getByText(/계약상 이용 권한/)).toBeInTheDocument();
    expect(screen.getByText(/저작권의 성립 또는 귀속을 확정하는 것이 아닙니다/)).toBeInTheDocument();
  });

  it("points at the separate billing policy rather than restating prices", () => {
    render(<TermsPage />);

    expect(screen.getByRole("link", { name: "구독·결제·환불 정책" })).toHaveAttribute(
      "href",
      "/refund-policy",
    );
  });

  it("describes closure the way the product performs it", () => {
    render(<TermsPage />);

    expect(screen.getByText(/구독을 먼저 해지해야 탈퇴할 수 있습니다/)).toBeInTheDocument();
  });
});

// ── refund policy ────────────────────────────────────────────────────

describe("refund policy", () => {
  it("renders", async () => {
    render(<RefundPolicyPage />);

    expect(
      await screen.findByRole("heading", { level: 1, name: "구독·결제·환불 정책" }),
    ).toBeInTheDocument();
  });

  it("takes plan figures from the live catalogue, not a second copy", async () => {
    render(<RefundPolicyPage />);

    const table = await screen.findByRole("table", { name: "요금제" });
    expect(within(table).getByText("Basic")).toBeInTheDocument();
    expect(within(table).getByText("₩19,900")).toBeInTheDocument();
  });

  it("discloses recurring billing and where the card is held", async () => {
    render(<RefundPolicyPage />);

    expect(await screen.findByText(/월 단위 정기결제/)).toBeInTheDocument();
    expect(screen.getByText(/카드 정보를 저장하지 않습니다/)).toBeInTheDocument();
  });

  it("says cancellation keeps access until the paid period ends", async () => {
    /** Which is what the billing code does. */
    render(<RefundPolicyPage />);

    expect(
      await screen.findByText(/이미 결제한 이용 기간이 끝날 때까지는 기능을 계속 이용/),
    ).toBeInTheDocument();
  });

  it("makes no blanket claim that refunds are impossible", async () => {
    render(<RefundPolicyPage />);
    await screen.findByRole("heading", { level: 1 });

    const body = document.body.textContent ?? "";
    expect(body).not.toMatch(/환불이 불가능합니다/);
    expect(body).not.toMatch(/환불 불가/);
    expect(screen.getByText(/부당하게 제한하지 않습니다/)).toBeInTheDocument();
  });

  it("survives a failed plan fetch rather than blanking the page", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));

    render(<RefundPolicyPage />);

    expect(await screen.findByText(/월 단위 정기결제/)).toBeInTheDocument();
  });
});

// ── versioning ───────────────────────────────────────────────────────

describe("versioning", () => {
  it("claims no revision history it does not have", () => {
    /** This is the first publication. Inventing a past would be
     * inventing a record of decisions nobody made. */
    expect(HISTORY).toEqual([]);
  });
});
