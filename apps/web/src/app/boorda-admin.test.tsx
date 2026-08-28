/**
 * The operator console screens.
 *
 * Two things carry the weight.
 *
 * The console must not present itself as protection. Hiding the nav from
 * a customer is a courtesy; the API is the boundary. So the tests here
 * check that the shell refuses to render for a `USER` *and* that nothing
 * in this code path treats that refusal as the security control — the
 * console fires no admin requests on behalf of an account without a
 * role.
 *
 * And the console must not claim to have done what it did not. Email
 * saves a draft and says so, because BOORDA has no mail provider
 * configured, and the cost of a reassuring lie lands on the operator who
 * believes an announcement went out.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AdminAdminsPage from "@/app/admin/admins/page";
import AdminAuditPage from "@/app/admin/audit/page";
import AdminDashboardPage from "@/app/admin/page";
import AdminEmailPage from "@/app/admin/email/page";
import AdminSupportPage from "@/app/admin/support/page";
import AdminUsersPage from "@/app/admin/users/page";
import { AdminShell } from "@/components/admin/AdminShell";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { ToastProvider } from "@/components/ui/Toast";
import { isAdmin, isSuperAdmin } from "@/lib/admin";

vi.mock("next/navigation", () => ({
  usePathname: () => "/admin",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  redirect: vi.fn(),
}));

const BASE_USER = {
  id: "u1",
  email: "operator@boorda.kr",
  display_name: null,
  created_at: "2026-01-15T00:00:00Z",
};

const ADMIN = { ...BASE_USER, role: "ADMIN" };
const SUPER_ADMIN = { ...BASE_USER, role: "SUPER_ADMIN" };
const CUSTOMER = { ...BASE_USER, email: "singer@boorda.kr", role: "USER" };

const DASHBOARD = {
  range: { start: "2026-08-01", end: "2026-08-28" },
  generated_at: "2026-08-28T00:00:00Z",
  revenue_krw: 19900,
  revenue_today_krw: 0,
  payment_count: 1,
  users: { total: 12, paid: 1, free: 11, new_in_range: 3 },
  generations: { requested: 0, completed: 0, failed: 0, creators: 0, average_per_creator: 0 },
  downloads: 0,
  support: { OPEN: 2, IN_PROGRESS: 0, RESOLVED: 1, CLOSED: 0 },
  plans: [
    { plan_id: "free", count: 11, share: 0.9167 },
    { plan_id: "basic", count: 1, share: 0.0833 },
    { plan_id: "pro", count: 0, share: 0 },
    { plan_id: "creator", count: 0, share: 0 },
  ],
  revenue_series: [{ day: "2026-08-28", value: 19900, secondary: 1 }],
  generation_series: [],
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface Stub {
  calls: { url: string; init?: RequestInit }[];
}

/**
 * Handlers are keyed by URL fragment, optionally prefixed with a method
 * (`"POST /admin/email/campaigns"`). The prefix matters: the campaigns
 * path is a listing on GET and a create on POST, and one handler
 * answering both is how a test ends up asserting against a shape the
 * product never returns.
 */
function stub(user: unknown, handlers: Record<string, () => Response> = {}): Stub {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      calls.push({ url, init });
      for (const [key, make] of Object.entries(handlers)) {
        const [wanted, fragment] = key.includes(" ") ? key.split(" ") : [null, key];
        if (wanted !== null && wanted !== method) continue;
        if (url.includes(fragment)) return make();
      }
      if (url.includes("/auth/me")) return json(user);
      if (url.includes("/admin/dashboard")) return json(DASHBOARD);
      if (url.includes("/admin/users")) return json({ items: [], total: 0 });
      if (url.includes("/admin/support")) return json({ items: [], total: 0 });
      if (url.includes("/admin/email/campaigns")) return json({ items: [] });
      if (url.includes("/admin/email/audience")) return json({ recipient_count: 12 });
      if (url.includes("/admin/admins")) return json([]);
      if (url.includes("/admin/audit")) return json({ items: [], total: 0 });
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
});

// ── who sees the console ─────────────────────────────────────────────

describe("access", () => {
  it("refuses to render the console to an ordinary account", async () => {
    stub(CUSTOMER);

    renderPage(
      <AdminShell>
        <p>매출 상세</p>
      </AdminShell>,
    );

    expect(await screen.findByText(/접근 권한이 없습니다/)).toBeInTheDocument();
    expect(screen.queryByText("매출 상세")).not.toBeInTheDocument();
  });

  it("fires no admin request on behalf of an account without a role", async () => {
    /**
     * The point is request hygiene, not protection: the API would refuse
     * these anyway. Rendering the children first would produce a burst
     * of 403s and, briefly, a page shaped like the console.
     */
    const { calls } = stub(CUSTOMER);

    renderPage(
      <AdminShell>
        <AdminDashboardPage />
      </AdminShell>,
    );

    await screen.findByText(/접근 권한이 없습니다/);
    expect(calls.filter((c) => c.url.includes("/v1/admin/"))).toHaveLength(0);
  });

  it("renders the console to an administrator", async () => {
    stub(ADMIN);

    renderPage(
      <AdminShell>
        <p>매출 상세</p>
      </AdminShell>,
    );

    expect(await screen.findByText("매출 상세")).toBeInTheDocument();
  });

  it("offers the administrators section only to a super administrator", async () => {
    stub(ADMIN);
    const { unmount } = renderPage(
      <AdminShell>
        <p>본문</p>
      </AdminShell>,
    );

    await screen.findByText("본문");
    const nav = screen.getByRole("navigation", { name: "운영 관리" });
    expect(within(nav).queryByRole("link", { name: "관리자" })).not.toBeInTheDocument();
    unmount();

    stub(SUPER_ADMIN);
    renderPage(
      <AdminShell>
        <p>본문</p>
      </AdminShell>,
    );

    await screen.findByText("본문");
    const superNav = screen.getByRole("navigation", { name: "운영 관리" });
    expect(within(superNav).getByRole("link", { name: "관리자" })).toBeInTheDocument();
  });

  it("treats an unrecognised role as no role at all", () => {
    // The column is a string; a typo must not be a promotion.
    expect(isAdmin("SUPERADMIN")).toBe(false);
    expect(isAdmin(undefined)).toBe(false);
    expect(isSuperAdmin("ADMIN")).toBe(false);
    expect(isSuperAdmin("SUPER_ADMIN")).toBe(true);
  });
});

// ── the dashboard ────────────────────────────────────────────────────

describe("dashboard", () => {
  it("shows revenue, members and support in one view", async () => {
    stub(ADMIN);

    renderPage(<AdminDashboardPage />);

    // Twice on purpose: once as the headline figure, once inside the
    // chart's screen-reader table, which carries the same numbers.
    expect(await screen.findAllByText("₩19,900")).toHaveLength(2);
    expect(screen.getByText("12명")).toBeInTheDocument();
  });

  it("renders zero as zero rather than hiding the panel", async () => {
    /**
     * Generation is switched off in production, so an empty chart is the
     * truth. A panel that disappeared when a figure was zero would teach
     * its reader to distrust the ones that remain.
     */
    stub(ADMIN);

    renderPage(<AdminDashboardPage />);

    expect(await screen.findByText("일별 생성")).toBeInTheDocument();
    expect(screen.getByText(/이 기간에는 생성 요청이 없습니다/)).toBeInTheDocument();
  });

  it("says the numbers are Korean-time", async () => {
    stub(ADMIN);

    renderPage(<AdminDashboardPage />);

    expect(await screen.findByText(/한국 시간\(KST\) 기준/)).toBeInTheDocument();
  });

  it("reports a failure instead of an empty dashboard", async () => {
    stub(ADMIN, { "/admin/dashboard": () => json({ detail: "boom" }, 500) });

    renderPage(<AdminDashboardPage />);

    expect(await screen.findByText(/불러오지 못했습니다/)).toBeInTheDocument();
  });
});

// ── members ──────────────────────────────────────────────────────────

describe("members", () => {
  it("offers no control that deletes an account", async () => {
    stub(ADMIN, {
      "/admin/users": () =>
        json({
          items: [
            {
              id: "u2",
              email: "singer@boorda.kr",
              display_name: null,
              role: "USER",
              created_at: "2026-08-01T00:00:00Z",
              deleted_at: null,
              plan_id: "basic",
              subscription_status: "ACTIVE",
            },
          ],
          total: 1,
        }),
    });

    renderPage(<AdminUsersPage />);

    await screen.findByText("singer@boorda.kr");
    expect(screen.queryByRole("button", { name: /삭제/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /탈퇴/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /환불/ })).not.toBeInTheDocument();
  });
});

// ── support ──────────────────────────────────────────────────────────

describe("support queue", () => {
  const TICKET = {
    reference: "SUP-ABCD1234",
    user_email: "singer@boorda.kr",
    category: "BILLING",
    subject: "결제 확인 요청",
    message: "8월 결제가 정상 처리되었는지 확인 부탁드립니다.",
    context_url: null,
    status: "OPEN",
    admin_note: null,
    created_at: "2026-08-28T00:00:00Z",
    updated_at: "2026-08-28T00:00:00Z",
    resolved_at: null,
  };

  it("opens a ticket and offers a status change", async () => {
    stub(ADMIN, {
      "/admin/support/SUP-ABCD1234": () => json(TICKET),
      "/admin/support": () => json({ items: [TICKET], total: 1 }),
    });

    renderPage(<AdminSupportPage />);

    await userEvent.click(await screen.findByText("결제 확인 요청"));

    expect(await screen.findByText(/8월 결제가 정상 처리되었는지/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /답변완료\(으\)로 변경/ })).toBeInTheDocument();
  });

  it("offers no way to edit what the customer wrote", async () => {
    stub(ADMIN, {
      "/admin/support/SUP-ABCD1234": () => json(TICKET),
      "/admin/support": () => json({ items: [TICKET], total: 1 }),
    });

    renderPage(<AdminSupportPage />);
    await userEvent.click(await screen.findByText("결제 확인 요청"));
    await screen.findByText(/8월 결제가 정상 처리되었는지/);

    // The only editable field is the internal note.
    const boxes = screen.getAllByRole("textbox");
    expect(boxes).toHaveLength(1);
    expect(boxes[0]).toHaveAttribute("id", "admin-note");
  });

  it("says the internal note is not shown to the customer", async () => {
    stub(ADMIN, {
      "/admin/support/SUP-ABCD1234": () => json(TICKET),
      "/admin/support": () => json({ items: [TICKET], total: 1 }),
    });

    renderPage(<AdminSupportPage />);
    await userEvent.click(await screen.findByText("결제 확인 요청"));

    expect(await screen.findByText(/고객에게는 표시되지 않습니다/)).toBeInTheDocument();
  });
});

// ── email ────────────────────────────────────────────────────────────

describe("email", () => {
  it("says plainly that sending is not available", async () => {
    stub(ADMIN);

    renderPage(<AdminEmailPage />);

    expect(await screen.findByText(/현재 이메일 발송은 제공되지 않습니다/)).toBeInTheDocument();
  });

  it("offers no send button", async () => {
    stub(ADMIN);

    renderPage(<AdminEmailPage />);

    await screen.findByRole("button", { name: "초안 저장" });
    expect(screen.queryByRole("button", { name: /발송/ })).not.toBeInTheDocument();
  });

  it("takes the recipient count from the server", async () => {
    /** Not counted in the browser from a page of results. */
    const { calls } = stub(ADMIN);

    renderPage(<AdminEmailPage />);

    expect(await screen.findByText("12명")).toBeInTheDocument();
    expect(calls.some((c) => c.url.includes("/admin/email/audience"))).toBe(true);
  });

  it("confirms that a saved draft was not sent", async () => {
    stub(ADMIN, {
      "POST /admin/email/campaigns": () =>
        json({
          id: "c1",
          subject: "공지",
          body: "내용",
          audience_type: "ALL",
          audience_plan_id: null,
          recipient_count: 12,
          status: "DRAFT",
          created_by_email: "operator@boorda.kr",
          created_at: "2026-08-28T00:00:00Z",
          sent_at: null,
          delivery_note: "No email provider is configured.",
        }),
    });

    renderPage(<AdminEmailPage />);

    await userEvent.type(await screen.findByLabelText("제목"), "8월 공지");
    await userEvent.type(screen.getByLabelText("내용"), "새로운 기능을 소개합니다.");
    await userEvent.click(screen.getByRole("button", { name: "초안 저장" }));

    expect(await screen.findByText(/발송은 되지 않았습니다/)).toBeInTheDocument();
  });
});

// ── administrators ───────────────────────────────────────────────────

describe("administrators", () => {
  const ROW = {
    id: "u1",
    email: "operator@boorda.kr",
    display_name: null,
    role: "SUPER_ADMIN",
    created_at: "2026-01-15T00:00:00Z",
    deleted_at: null,
    plan_id: "free",
    subscription_status: null,
  };

  it("explains the last-super-admin refusal in its own words", async () => {
    /**
     * A generic "failed" would leave the operator retrying a refusal
     * that will never succeed.
     */
    let first = true;
    stub(SUPER_ADMIN, {
      "/admin/admins": () => {
        if (first) {
          first = false;
          return json([ROW]);
        }
        return json({ detail: "At least one super administrator must remain." }, 409);
      },
    });

    renderPage(<AdminAdminsPage />);

    await userEvent.click(await screen.findByRole("button", { name: "권한 해제" }));

    expect(
      await screen.findByText(/최고 관리자는 최소 한 명이 남아 있어야 합니다/),
    ).toBeInTheDocument();
  });

  it("says a revoked role does not delete the account", async () => {
    stub(SUPER_ADMIN, { "/admin/admins": () => json([ROW]) });

    renderPage(<AdminAdminsPage />);

    expect(await screen.findByText(/계정을 삭제하지 않습니다/)).toBeInTheDocument();
  });

  it("says promotion only applies to accounts that already exist", async () => {
    stub(SUPER_ADMIN, { "/admin/admins": () => json([]) });

    renderPage(<AdminAdminsPage />);

    expect(await screen.findByText(/이미 가입한 회원만 관리자로 지정할 수 있습니다/)).toBeInTheDocument();
  });
});

// ── audit ────────────────────────────────────────────────────────────

describe("audit", () => {
  it("lists what an operator did, with no control that edits it", async () => {
    stub(ADMIN, {
      "/admin/audit": () =>
        json({
          items: [
            {
              id: "a1",
              action: "ADMIN_GRANTED",
              actor_email: "operator@boorda.kr",
              target_email: "singer@boorda.kr",
              metadata: { from: "USER", to: "ADMIN" },
              created_at: "2026-08-28T00:00:00Z",
            },
          ],
          total: 1,
        }),
    });

    renderPage(<AdminAuditPage />);

    expect(await screen.findByText("관리자 권한 부여")).toBeInTheDocument();
    expect(screen.getByText("singer@boorda.kr")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /삭제/ })).not.toBeInTheDocument();
  });

  it("shows an empty log as empty rather than as a failure", async () => {
    stub(ADMIN);

    renderPage(<AdminAuditPage />);

    expect(await screen.findByText("기록이 없습니다")).toBeInTheDocument();
  });
});

// ── formatting ───────────────────────────────────────────────────────

describe("presentation", () => {
  it("formats won the way an operator reconciles it", async () => {
    const { formatWon, formatCount } = await import("@/lib/admin");

    expect(formatWon(19900)).toBe("₩19,900");
    expect(formatCount(1234)).toBe("1,234");
  });

  it("renders a day in Korean time regardless of the browser's zone", async () => {
    const { formatDay } = await import("@/lib/admin");

    // The bucket label is a Korean calendar day and must not shift when
    // the operator's laptop is set to another timezone.
    expect(formatDay("2026-08-28")).toContain("28");
  });
});

// ── waiting for the shell ────────────────────────────────────────────

describe("loading", () => {
  it("shows a placeholder rather than a flash of refusal", async () => {
    stub(ADMIN);

    renderPage(
      <AdminShell>
        <p>본문</p>
      </AdminShell>,
    );

    expect(screen.getByText("불러오는 중…")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("본문")).toBeInTheDocument());
  });
});
