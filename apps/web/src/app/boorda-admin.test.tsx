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
import { BarChart, GenerationChart, RevenueChart } from "@/components/admin/Charts";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { ToastProvider } from "@/components/ui/Toast";
import { DateRangePicker } from "@/components/admin/DateRangePicker";
import {
  MAX_RANGE_DAYS,
  type DateRange,
  addDays,
  formatDelta,
  formatRange,
  isAdmin,
  isSuperAdmin,
  isValidRange,
  kstToday,
  matchingPreset,
  presetRange,
  rangeFromParams,
  rangeLength,
} from "@/lib/admin";

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
  range: { start: "2026-08-01", end: "2026-08-28", days: 28, bucketing: "day" },
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
  comparison: {
    start: "2026-07-04",
    end: "2026-07-31",
    revenue_krw: 10000,
    payment_count: 1,
    new_users: 2,
    generations: 0,
    revenue_delta_pct: 99.0,
    payment_delta_pct: 0.0,
    user_delta_pct: 50.0,
    generation_delta_pct: null,
  },
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
    // The members card leads with the figure the range controls — new
    // signups — and carries the standing totals underneath it.
    expect(screen.getByText("3명")).toBeInTheDocument();
    expect(screen.getByText(/전체 12명/)).toBeInTheDocument();
  });

  it("renders zero as zero rather than hiding the panel", async () => {
    /**
     * Generation is switched off in production, so an empty chart is the
     * truth. A panel that disappeared when a figure was zero would teach
     * its reader to distrust the ones that remain.
     */
    stub(ADMIN);

    renderPage(<AdminDashboardPage />);

    expect(await screen.findByText("생성 추이")).toBeInTheDocument();
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

// ── the date range ───────────────────────────────────────────────────

describe("date range", () => {
  /** A fixed instant, so "today" is not whatever day the suite runs. */
  const NOW = new Date("2026-08-28T03:00:00Z"); // 12:00 KST

  it("resolves each preset against the Korean day, not the browser's", () => {
    expect(presetRange("today", NOW)).toEqual({ from: "2026-08-28", to: "2026-08-28" });
    // Seven days *including* today, which is what "7일" reads as.
    expect(presetRange("7d", NOW)).toEqual({ from: "2026-08-22", to: "2026-08-28" });
    expect(presetRange("30d", NOW)).toEqual({ from: "2026-07-30", to: "2026-08-28" });
    expect(presetRange("month", NOW)).toEqual({ from: "2026-08-01", to: "2026-08-28" });
    expect(presetRange("year", NOW)).toEqual({ from: "2026-01-01", to: "2026-08-28" });
  });

  it("reads a Korean day from a UTC evening", () => {
    /**
     * 23:00 UTC on the 27th is already the 28th in Seoul. An operator
     * abroad must still be shown Korean days, because that is what the
     * figures are bucketed by.
     */
    expect(kstToday(new Date("2026-08-27T23:00:00Z"))).toBe("2026-08-28");
    expect(kstToday(new Date("2026-08-28T14:59:00Z"))).toBe("2026-08-28");
    expect(kstToday(new Date("2026-08-28T15:00:00Z"))).toBe("2026-08-29");
  });

  it("counts an inclusive range, so one day is one", () => {
    expect(rangeLength({ from: "2026-08-28", to: "2026-08-28" })).toBe(1);
    expect(rangeLength({ from: "2026-08-01", to: "2026-08-28" })).toBe(28);
  });

  it("adds days without tripping over the browser's timezone", () => {
    expect(addDays("2026-08-28", 1)).toBe("2026-08-29");
    expect(addDays("2026-03-01", -1)).toBe("2026-02-28");
    expect(addDays("2026-01-01", -1)).toBe("2025-12-31");
  });

  it("refuses ranges the API would refuse", () => {
    expect(isValidRange({ from: "2026-08-01", to: "2026-08-28" })).toBe(true);
    expect(isValidRange({ from: "2026-08-28", to: "2026-08-01" })).toBe(false);
    expect(isValidRange({ from: "not-a-date", to: "2026-08-28" })).toBe(false);
    // Date-shaped but not a date.
    expect(isValidRange({ from: "2026-02-31", to: "2026-03-01" })).toBe(false);
    expect(
      isValidRange({ from: "2026-01-01", to: addDays("2026-01-01", MAX_RANGE_DAYS) }),
    ).toBe(false);
  });

  it("falls back to a valid default rather than propagating nonsense", () => {
    /** A bookmarked link with a typo shows this month, not an error. */
    const fallback = presetRange("month", NOW);

    for (const query of [
      "",
      "from=2026-08-28&to=2026-08-01",
      "from=garbage&to=2026-08-28",
      "from=2026-08-01",
      "to=2026-08-28",
    ]) {
      expect(rangeFromParams(new URLSearchParams(query), NOW)).toEqual(fallback);
    }
  });

  it("reads a valid range straight out of the URL", () => {
    expect(rangeFromParams(new URLSearchParams("from=2026-03-01&to=2026-03-31"), NOW)).toEqual({
      from: "2026-03-01",
      to: "2026-03-31",
    });
  });

  it("recognises a range that happens to equal a preset", () => {
    expect(matchingPreset({ from: "2026-08-01", to: "2026-08-28" }, NOW)).toBe("month");
    expect(matchingPreset({ from: "2026-03-01", to: "2026-03-31" }, NOW)).toBeNull();
  });

  it("shows the selected range the way an operator reads it", () => {
    expect(formatRange({ from: "2026-08-01", to: "2026-08-28" })).toBe("2026.08.01 ~ 2026.08.28");
    expect(formatRange({ from: "2026-08-28", to: "2026-08-28" })).toBe("2026.08.28");
  });
});

describe("date range picker", () => {
  const MONTH: DateRange = presetRange("month");

  it("offers the presets and a custom option", async () => {
    render(<DateRangePicker range={MONTH} onChange={vi.fn()} />);

    for (const label of ["오늘", "7일", "30일", "이번 달", "올해", "직접 선택"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("reports the selected range in Korean time", () => {
    render(<DateRangePicker range={{ from: "2026-08-01", to: "2026-08-28" }} onChange={vi.fn()} />);

    expect(screen.getByText(/2026\.08\.01 ~ 2026\.08\.28/)).toBeInTheDocument();
    expect(screen.getByText(/KST/)).toBeInTheDocument();
  });

  it("hands back the preset's range when one is chosen", async () => {
    const onChange = vi.fn();
    render(<DateRangePicker range={MONTH} onChange={onChange} />);

    await userEvent.click(screen.getByRole("button", { name: "7일" }));

    expect(onChange).toHaveBeenCalledWith(presetRange("7d"));
  });

  it("opens a real calendar for a custom range", async () => {
    render(<DateRangePicker range={MONTH} onChange={vi.fn()} />);

    await userEvent.click(screen.getByRole("button", { name: "직접 선택" }));

    // Native date inputs: a real picker on every platform, keyboard
    // navigable and labelled, at no bundle cost.
    expect(screen.getByLabelText("시작 날짜")).toHaveAttribute("type", "date");
    expect(screen.getByLabelText("종료 날짜")).toHaveAttribute("type", "date");
  });

  it("applies a custom range only when it is valid", async () => {
    const onChange = vi.fn();
    render(<DateRangePicker range={MONTH} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "직접 선택" }));

    const from = screen.getByLabelText("시작 날짜");
    const to = screen.getByLabelText("종료 날짜");
    await userEvent.clear(from);
    await userEvent.type(from, "2026-03-01");
    await userEvent.clear(to);
    await userEvent.type(to, "2026-03-31");
    await userEvent.click(screen.getByRole("button", { name: "적용" }));

    expect(onChange).toHaveBeenCalledWith({ from: "2026-03-01", to: "2026-03-31" });
  });

  it("says why a backwards range will not apply, before the request", async () => {
    const onChange = vi.fn();
    render(
      <DateRangePicker range={{ from: "2026-08-28", to: "2026-08-01" }} onChange={onChange} />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(/앞설 수 없습니다/);
    expect(screen.getByRole("button", { name: "적용" })).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows the custom controls when the range matches no preset", () => {
    render(<DateRangePicker range={{ from: "2026-03-01", to: "2026-03-31" }} onChange={vi.fn()} />);

    expect(screen.getByLabelText("시작 날짜")).toHaveValue("2026-03-01");
    expect(screen.getByRole("button", { name: "직접 선택" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
});

// ── comparison ───────────────────────────────────────────────────────

describe("previous-period comparison", () => {
  it("formats a delta with its sign", () => {
    expect(formatDelta(12.4)).toBe("+12.4%");
    expect(formatDelta(-3.1)).toBe("-3.1%");
    expect(formatDelta(0)).toBe("0.0%");
  });

  it("has no percentage at all for a zero base", () => {
    /** Not Infinity, not 100% — undefined, and shown as "신규". */
    expect(formatDelta(null)).toBeNull();
  });

  it("shows the delta beside the figure it describes", async () => {
    stub(ADMIN);

    renderPage(<AdminDashboardPage />);

    expect(await screen.findByText("+99.0%")).toBeInTheDocument();
    expect(screen.getByText(/직전 동일 기간/)).toBeInTheDocument();
  });

  it("renders a zero-base comparison as 신규 rather than a number", async () => {
    stub(ADMIN, {
      "/admin/dashboard": () =>
        json({
          ...DASHBOARD,
          comparison: { ...DASHBOARD.comparison, revenue_krw: 0, revenue_delta_pct: null },
        }),
    });

    renderPage(<AdminDashboardPage />);

    // Scoped to the revenue card: the generation card also has a null
    // base in this fixture, and asserting on a bare "신규" would pass
    // whichever one rendered.
    const card = (await screen.findByText("기간 매출")).parentElement!;
    expect(within(card).getByText("신규")).toBeInTheDocument();
    expect(within(card).queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Infinity/)).not.toBeInTheDocument();
  });
});

// ── chart geometry ───────────────────────────────────────────────────

describe("chart bars", () => {
  /**
   * jsdom does no layout, so none of these can assert a rendered pixel
   * height — which is exactly how the original bug shipped: the charts
   * held correct numbers and drew nothing, and every test passed because
   * they all read the screen-reader table.
   *
   * What is checkable, and what actually broke, is the CSS contract. A
   * percentage height resolves against the parent's height. So the
   * element carrying `height: N%` must be a direct child of the element
   * with the definite height; a wrapper sized by its content resolves
   * every bar to zero.
   */
  const SERIES = [
    { day: "2026-08-27", value: 1_000, secondary: 1 },
    { day: "2026-08-28", value: 19_900, secondary: 1 },
  ];

  function bars(container: HTMLElement): HTMLElement[] {
    return [...container.querySelectorAll<HTMLElement>("[style*='height']")];
  }

  it("anchors every bar's percentage height to an element with a definite height", () => {
    const { container } = render(<BarChart title="일별 매출" data={SERIES} />);

    const drawn = bars(container);
    expect(drawn).toHaveLength(SERIES.length);

    for (const bar of drawn) {
      expect(bar.style.height).toMatch(/^[\d.]+%$/);
      // The parent must be the track that carries the height, not a
      // content-sized wrapper.
      expect(bar.parentElement?.className).toMatch(/\bh-32\b/);
    }
  });

  it("scales bars against the peak so the tallest fills the track", () => {
    const { container } = render(<BarChart title="일별 매출" data={SERIES} />);

    const heights = bars(container).map((b) => parseFloat(b.style.height));

    expect(Math.max(...heights)).toBe(100);
    // 1_000 of 19_900 is ~5%, comfortably above the 2% floor that keeps
    // a tiny non-zero value visible at all.
    expect(Math.min(...heights)).toBeGreaterThan(2);
    expect(Math.min(...heights)).toBeLessThan(10);
  });

  it("gives a lone bar the full track rather than nothing", () => {
    /** Production's real case: one payment, one day. */
    const { container } = render(
      <RevenueChart data={[{ day: "2026-08-28", value: 19_900, secondary: 1 }]} />,
    );

    expect(bars(container).map((b) => b.style.height)).toEqual(["100%"]);
  });

  it("keeps a zero-valued day visible instead of collapsing it", () => {
    const { container } = render(
      <BarChart
        title="일별 생성"
        data={[
          { day: "2026-08-27", value: 0, secondary: 0 },
          { day: "2026-08-28", value: 5, secondary: 0 },
        ]}
      />,
    );

    // A floor, not zero: a day with nothing still occupies its slot, so
    // the axis reads as a continuous run of days.
    expect(bars(container).map((b) => b.style.height)).toEqual(["2%", "100%"]);
  });

  it("draws no bars at all when there is no data", () => {
    const { container } = render(<BarChart title="일별 생성" data={[]} />);

    expect(bars(container)).toHaveLength(0);
    expect(screen.getByText(/이 기간에는 데이터가 없습니다/)).toBeInTheDocument();
  });
});

// ── chart presentation ───────────────────────────────────────────────

describe("chart presentation", () => {
  const ONE_DAY = [{ day: "2026-08-28", value: 19_900, secondary: 1 }];

  it("caps a lone bar's width instead of filling the plot", () => {
    /**
     * Production's real case. Without a cap one data point stretches to
     * the whole plot width and reads as a filled rectangle rather than a
     * measurement.
     */
    const { container } = render(<RevenueChart data={ONE_DAY} />);

    const bar = container.querySelector<HTMLElement>("[style*='height']")!;
    expect(bar.style.maxWidth).toBe("3.5rem");
    expect(bar.style.height).toBe("100%");
  });

  it("centres the bars so a short series is not stranded on the left", () => {
    const { container } = render(<RevenueChart data={ONE_DAY} />);

    const track = [...container.querySelectorAll("div")].find((d) =>
      /\bh-32\b/.test(d.className),
    )!;
    expect(track.className).toMatch(/\bjustify-center\b/);
  });

  it("labels a weekly bucket as a week and a monthly one as a month", () => {
    const weekly = render(
      <BarChart title="매출 추이" data={[{ day: "2026-08-24", value: 5, secondary: 0 }]} bucketing="week" />,
    );
    expect(weekly.container.textContent).toMatch(/주/);
    weekly.unmount();

    const monthly = render(
      <BarChart title="매출 추이" data={[{ day: "2026-08-01", value: 5, secondary: 0 }]} bucketing="month" />,
    );
    // A month bucket reads as a month, not as its first day.
    expect(monthly.container.textContent).toMatch(/2026년 8월/);
  });

  it("says which bucket size it is drawn at", () => {
    const { container, rerender } = render(<RevenueChart data={ONE_DAY} bucketing="day" />);
    expect(container.textContent).toMatch(/하루 단위/);

    rerender(<RevenueChart data={ONE_DAY} bucketing="week" />);
    expect(container.textContent).toMatch(/주 단위/);
    expect(container.textContent).toMatch(/월요일에 시작/);

    rerender(<RevenueChart data={ONE_DAY} bucketing="month" />);
    expect(container.textContent).toMatch(/월 단위/);
  });

  it("gives every bar an accessible name carrying its own figure", () => {
    /** So the chart is not a wall of unnamed buttons to a screen reader. */
    render(<RevenueChart data={ONE_DAY} />);

    expect(
      screen.getByRole("button", { name: /8월 28일 ₩19,900/ }),
    ).toBeInTheDocument();
  });

  it("shows a tooltip on hover and on keyboard focus alike", async () => {
    render(<RevenueChart data={ONE_DAY} />);
    const bar = screen.getByRole("button", { name: /₩19,900/ });

    await userEvent.hover(bar);
    expect(await screen.findByRole("status")).toHaveTextContent("₩19,900");

    await userEvent.unhover(bar);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    // The same tooltip must be reachable without a pointer.
    bar.focus();
    expect(await screen.findByRole("status")).toHaveTextContent("₩19,900");
  });

  it("carries the second figure in the tooltip too", async () => {
    render(<RevenueChart data={ONE_DAY} />);

    await userEvent.hover(screen.getByRole("button", { name: /₩19,900/ }));

    expect(await screen.findByRole("status")).toHaveTextContent("결제 1건");
  });

  it("keeps every figure in the table, not only in the tooltip", () => {
    /**
     * The rule that matters for accessibility: nothing critical is
     * reachable only by hovering.
     */
    const { container } = render(
      <GenerationChart
        data={[{ day: "2026-08-28", value: 14, secondary: 13 }]}
        bucketing="day"
      />,
    );

    const table = container.querySelector("table.sr-only")!;
    expect(table.textContent).toMatch(/8월 28일/);
    expect(table.textContent).toMatch(/14건/);
    expect(table.textContent).toMatch(/13건/);
  });

  it("renders an empty generation chart as an empty state", () => {
    render(<GenerationChart data={[]} />);

    expect(screen.getByText(/이 기간에는 생성 요청이 없습니다/)).toBeInTheDocument();
  });

  it("shows one axis label when there is a single bucket", () => {
    /** Two identical labels at both ends read as a broken axis. */
    const { container } = render(<RevenueChart data={ONE_DAY} />);

    const axis = [...container.querySelectorAll("div")].find((d) =>
      /justify-between/.test(d.className),
    )!;
    expect(axis.children).toHaveLength(1);
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
