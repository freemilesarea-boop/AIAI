/**
 * What the interface says about money.
 *
 * The server decides everything; these tests prove the pages describe it
 * honestly. Three things in particular:
 *
 *  - The return page never reads the URL. It is reached by paying, by
 *    closing the PayApp window, and by typing it, and a page that
 *    treated arrival as proof would make a subscription cost one
 *    bookmark.
 *  - No page sends a price. The checkout request carries a plan name and
 *    a phone number, and a test asserts the request body to keep it that
 *    way.
 *  - The cancel dialog describes PayApp's actual semantics: the next
 *    charge stops, the last one is not refunded, and access runs to the
 *    end of the period already paid for.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import BillingReturnPage from "@/app/billing/return/page";
import PlansPage from "@/app/plans/page";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { CheckoutDialog } from "@/components/CheckoutDialog";
import { EntitlementProvider } from "@/components/EntitlementProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { SubscriptionPanel } from "@/components/SubscriptionPanel";
import { ToastProvider } from "@/components/ui/Toast";
import type { BillingStatus } from "@/lib/billing";
import { entitlementFixture, planCatalogueFixture, planFixture } from "@/test/entitlement-factories";

const push = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/plans",
  useRouter: () => ({ push, replace: vi.fn(), refresh: vi.fn() }),
  // Deliberately throws. Nothing on the return page may read the query
  // string, and a test that only asserted the rendered output could miss
  // a later change that started to.
  useSearchParams: () => {
    throw new Error("the return page must not read query parameters");
  },
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

function billingStatus(overrides: Partial<BillingStatus> = {}): BillingStatus {
  return {
    plan_id: "pro",
    display_name: "Pro",
    status: "PENDING_INITIAL_PAYMENT",
    auto_renew: true,
    period_start: null,
    period_end: null,
    next_renewal_at: null,
    last_payment_at: null,
    awaiting_payment: true,
    checkout_available: true,
    ...overrides,
  };
}

/** Records every request so tests can assert what was actually sent. */
interface Stub {
  calls: { url: string; init?: RequestInit }[];
}

function stub(options: {
  status?: BillingStatus;
  statusSequence?: BillingStatus[];
  checkout?: unknown;
  checkoutStatus?: number;
  payments?: unknown[];
  catalogue?: ReturnType<typeof planCatalogueFixture>;
}): Stub {
  const calls: { url: string; init?: RequestInit }[] = [];
  const sequence = options.statusSequence ? [...options.statusSequence] : null;

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      if (url.includes("/auth/me")) return json(USER);
      if (url.includes("/billing/checkout")) {
        return json(options.checkout ?? {}, options.checkoutStatus ?? 201);
      }
      if (url.includes("/billing/cancel")) {
        return json(billingStatus({ status: "CANCEL_PENDING", auto_renew: false }));
      }
      if (url.includes("/billing/payments")) return json({ items: options.payments ?? [] });
      if (url.includes("/billing/status")) {
        const next = sequence && sequence.length > 1 ? sequence.shift()! : (sequence?.[0] ?? options.status);
        return json(next ?? billingStatus());
      }
      if (url.includes("/account/entitlement")) return json(entitlementFixture("free"));
      if (url.includes("/v1/plans")) return json(options.catalogue ?? planCatalogueFixture());
      return json({ items: [] });
    }),
  );

  return { calls };
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
  push.mockReset();
});

// ── the return page ──────────────────────────────────────────────────

describe("the return page", () => {
  it("asks the server rather than reading the URL", async () => {
    // `useSearchParams` throws in this file's mock, so a render that
    // succeeds is proof the page never called it.
    const stubbed = stub({ status: billingStatus({ status: "ACTIVE", awaiting_payment: false }) });
    renderPage(<BillingReturnPage />);

    await screen.findByTestId("billing-active");
    expect(stubbed.calls.some((c) => c.url.includes("/billing/status"))).toBe(true);
  });

  it("says it is still checking while the notification has not arrived", async () => {
    stub({ status: billingStatus() });
    renderPage(<BillingReturnPage />);

    expect(await screen.findByTestId("billing-checking")).toBeInTheDocument();
    expect(screen.getByText("결제 확인 중…")).toBeInTheDocument();
  });

  it("confirms only once the server says the subscription is active", async () => {
    stub({
      status: billingStatus({
        status: "ACTIVE",
        awaiting_payment: false,
        period_end: "2026-09-27T00:00:00Z",
      }),
    });
    renderPage(<BillingReturnPage />);

    const panel = await screen.findByTestId("billing-active");
    expect(within(panel).getByText("구독이 활성화되었습니다")).toBeInTheDocument();
    expect(within(panel).getByText("Pro")).toBeInTheDocument();
  });

  it("reports a declined card as a failure", async () => {
    stub({ status: billingStatus({ status: "PAST_DUE", awaiting_payment: false }) });
    renderPage(<BillingReturnPage />);

    expect(await screen.findByTestId("billing-failed")).toBeInTheDocument();
    expect(screen.getByText("결제를 완료하지 못했습니다")).toBeInTheDocument();
  });

  it("does not claim a payment failed when it merely has not been confirmed", async () => {
    /**
     * The distinction that matters. A pending payment is unknown, not
     * failed, and telling someone their payment failed when it did not
     * invites a second charge.
     */
    stub({ status: billingStatus() });
    renderPage(<BillingReturnPage />);

    await screen.findByTestId("billing-checking");
    expect(screen.queryByText("결제를 완료하지 못했습니다")).toBeNull();
  });
});

// ── checkout ─────────────────────────────────────────────────────────

describe("checkout", () => {
  it("sends a plan and a phone number, and no price", async () => {
    const stubbed = stub({ checkout: { payurl: "https://payapp.kr/pay/900001" } });
    // jsdom refuses a real navigation; the assertion is about the request.
    vi.stubGlobal("location", { assign: vi.fn() } as unknown as Location);
    const user = userEvent.setup();
    render(<CheckoutDialog plan={planFixture("pro")} onClose={vi.fn()} />);

    await user.type(screen.getByLabelText("휴대폰 번호"), "010-1234-5678");
    await user.click(screen.getByRole("button", { name: "결제 진행" }));

    await waitFor(() => {
      const call = stubbed.calls.find((c) => c.url.includes("/billing/checkout"));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call!.init!.body));
      expect(body).toEqual({ plan_id: "pro", phone: "010-1234-5678" });
      // The three fields that must never come from a browser.
      expect(body).not.toHaveProperty("amount_krw");
      expect(body).not.toHaveProperty("price");
      expect(body).not.toHaveProperty("rebill_no");
    });
  });

  it("refuses to submit a number PayApp could not use", async () => {
    stub({ checkout: {} });
    const user = userEvent.setup();
    render(<CheckoutDialog plan={planFixture("pro")} onClose={vi.fn()} />);

    await user.type(screen.getByLabelText("휴대폰 번호"), "02-123-4567");

    expect(screen.getByRole("button", { name: "결제 진행" })).toBeDisabled();
  });

  it("states the price it is about to start charging", () => {
    stub({ checkout: {} });
    render(<CheckoutDialog plan={planFixture("creator")} onClose={vi.fn()} />);

    expect(screen.getByText(/₩49,900/)).toBeInTheDocument();
  });

  it("says where card details go", () => {
    stub({ checkout: {} });
    render(<CheckoutDialog plan={planFixture("pro")} onClose={vi.fn()} />);

    expect(screen.getByText(/카드 정보는 부르다에 저장되지 않습니다/)).toBeInTheDocument();
  });

  it("explains a refused checkout without activating anything", async () => {
    stub({ checkout: { detail: "SUBSCRIPTION_ALREADY_ACTIVE" }, checkoutStatus: 409 });
    const user = userEvent.setup();
    render(<CheckoutDialog plan={planFixture("pro")} onClose={vi.fn()} />);

    await user.type(screen.getByLabelText("휴대폰 번호"), "010-1234-5678");
    await user.click(screen.getByRole("button", { name: "결제 진행" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/이미 구독 중입니다/);
  });

  it("sends an unauthenticated visitor to log in, keeping the plan", async () => {
    stub({ catalogue: { ...planCatalogueFixture(), checkout_available: true } });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        // Signed out.
        if (url.includes("/auth/me")) return json({ detail: "unauthenticated" }, 401);
        if (url.includes("/v1/plans")) {
          return json({ ...planCatalogueFixture(), checkout_available: true });
        }
        return json({ detail: "unauthenticated" }, 401);
      }),
    );
    const user = userEvent.setup();
    renderPage(<PlansPage />);

    await user.click(await screen.findByRole("button", { name: "Pro 시작하기" }));

    expect(push).toHaveBeenCalledWith(expect.stringContaining("/login?next="));
    expect(push.mock.calls[0][0]).toContain(encodeURIComponent("plan=pro"));
  });
});

// ── settings ─────────────────────────────────────────────────────────

describe("the subscription panel", () => {
  it("shows the server's dates rather than computing its own", async () => {
    stub({
      status: billingStatus({
        status: "ACTIVE",
        awaiting_payment: false,
        period_end: "2026-09-27T00:00:00Z",
        next_renewal_at: "2026-09-27T00:00:00Z",
        last_payment_at: "2026-08-27T00:00:00Z",
      }),
    });
    renderPage(<SubscriptionPanel />);

    expect(await screen.findByText("이용 중")).toBeInTheDocument();
    expect(screen.getByText("2026년 9월 27일")).toBeInTheDocument();
  });

  it("offers cancellation only while auto-renew is on", async () => {
    stub({
      status: billingStatus({ status: "CANCEL_PENDING", auto_renew: false, awaiting_payment: false }),
    });
    renderPage(<SubscriptionPanel />);

    await screen.findByText("해지 예정");
    expect(screen.queryByRole("button", { name: "구독 해지" })).toBeNull();
  });

  it("describes what cancelling actually does", async () => {
    /** PayApp stops the next charge and does not refund the last one. */
    stub({
      status: billingStatus({
        status: "ACTIVE",
        auto_renew: true,
        awaiting_payment: false,
        period_end: "2026-09-27T00:00:00Z",
      }),
    });
    const user = userEvent.setup();
    renderPage(<SubscriptionPanel />);

    await user.click(await screen.findByRole("button", { name: "구독 해지" }));

    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText(/현재 결제 기간이 끝날 때까지 이용할 수 있으며/),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/라이브러리에 그대로 남습니다/)).toBeInTheDocument();
  });

  it("shows no next billing date once auto-renew is off", async () => {
    stub({
      status: billingStatus({
        status: "CANCEL_PENDING",
        auto_renew: false,
        awaiting_payment: false,
        next_renewal_at: null,
        period_end: "2026-09-27T00:00:00Z",
      }),
    });
    renderPage(<SubscriptionPanel />);

    await screen.findByText("해지 예정");
    expect(screen.getByText("없음")).toBeInTheDocument();
  });

  it("explains a failed renewal instead of just showing a status", async () => {
    stub({ status: billingStatus({ status: "PAST_DUE", awaiting_payment: false }) });
    renderPage(<SubscriptionPanel />);

    expect(await screen.findByText("결제가 실패했습니다")).toBeInTheDocument();
    expect(screen.getByText(/결제 수단을 확인한 뒤 다시 구독해 주세요/)).toBeInTheDocument();
  });
});
