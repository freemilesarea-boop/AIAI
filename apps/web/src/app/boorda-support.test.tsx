/**
 * The support screens.
 *
 * Two things carry the weight here. The contact form must not carry an
 * email address — a ticket that could claim to be from someone else is
 * the whole problem — and the FAQ must not answer a question nobody has
 * decided, because a plausible answer becomes a promise.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ContactPage from "@/app/support/contact/page";
import InquiriesPage from "@/app/support/inquiries/page";
import SupportPage from "@/app/support/page";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { ToastProvider } from "@/components/ui/Toast";
import { FAQ, OPEN_POLICY_QUESTIONS } from "@/lib/faq";

const push = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/support",
  useRouter: () => ({ push, replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ reference: "SUP-ABCD1234" }),
  redirect: vi.fn(),
}));

const USER = {
  id: "u1",
  email: "singer@boorda.kr",
  display_name: null,
  created_at: "2026-01-15T00:00:00Z",
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

function stub(handlers: Record<string, () => Response> = {}): Stub {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      for (const [fragment, make] of Object.entries(handlers)) {
        if (url.includes(fragment)) return make();
      }
      if (url.includes("/auth/me")) return json(USER);
      if (url.includes("/support/inquiries")) return json({ items: [], total: 0 });
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
  push.mockReset();
});

// ── support home ─────────────────────────────────────────────────────

describe("support home", () => {
  it("offers the three actions", () => {
    stub();
    renderPage(<SupportPage />);

    const quick = screen.getByRole("navigation", { name: "고객지원 바로가기" });
    expect(within(quick).getByRole("link", { name: "문의하기" })).toHaveAttribute(
      "href",
      "/support/contact",
    );
    expect(within(quick).getByRole("link", { name: "내 문의내역" })).toHaveAttribute(
      "href",
      "/support/inquiries",
    );
  });

  it("shows FAQ answers for the selected category", async () => {
    stub();
    const user = userEvent.setup();
    renderPage(<SupportPage />);

    await user.click(screen.getByRole("tab", { name: "구독 및 결제" }));

    expect(screen.getByText(/월 19,900원에 200곡/)).toBeInTheDocument();
  });

  it("names the undecided questions instead of answering them", () => {
    /**
     * A FAQ is where a product quietly invents policy. These are listed
     * as open and routed to the contact form, where a person can give
     * an answer the company stands behind.
     */
    stub();
    renderPage(<SupportPage />);

    expect(screen.getByText("아직 안내가 준비되지 않은 항목")).toBeInTheDocument();
    for (const question of OPEN_POLICY_QUESTIONS) {
      expect(screen.getByText(question)).toBeInTheDocument();
    }
  });
});

describe("faq content", () => {
  it("invents no refund, copyright or retention policy", () => {
    const answers = FAQ.map((entry) => entry.answer).join(" ");

    // Words that would only appear if someone had made up a rule.
    expect(answers).not.toMatch(/환불(은|이|을)?\s*(가능|불가|해\s*드립니다)/);
    expect(answers).not.toMatch(/\d+\s*일\s*(이내|안에)/);
    expect(answers).not.toMatch(/저작권.*(귀속|양도|소유)/);
    expect(answers).not.toMatch(/보존\s*기간/);
  });

  it("quotes the plan figures the server actually enforces", () => {
    const plans = FAQ.find((entry) => entry.id === "billing-plans");

    expect(plans?.answer).toContain("20곡");
    expect(plans?.answer).toContain("19,900");
    expect(plans?.answer).toContain("29,900");
    expect(plans?.answer).toContain("49,900");
  });
});

// ── contact ──────────────────────────────────────────────────────────

describe("contact form", () => {
  it("sends no email address", async () => {
    /**
     * The account's own address is used server-side. A form field would
     * mean a ticket could claim to come from someone else.
     */
    const stubbed = stub({
      "/support/inquiries": () =>
        json({ reference: "SUP-ABCD1234", status: "OPEN" }, 201),
    });
    const user = userEvent.setup();
    renderPage(<ContactPage />);

    await user.type(screen.getByLabelText("제목"), "결제 문의");
    await user.type(screen.getByLabelText("내용"), "두 번 청구된 것 같습니다.");
    await user.click(screen.getByRole("button", { name: "문의 접수" }));

    await waitFor(() => {
      const call = stubbed.calls.find(
        (c) => c.url.includes("/support/inquiries") && c.init?.method === "POST",
      );
      expect(call).toBeDefined();
      const body = JSON.parse(String(call!.init!.body));
      expect(body).not.toHaveProperty("email");
      expect(body).not.toHaveProperty("user_id");
      expect(body).not.toHaveProperty("status");
      expect(body.subject).toBe("결제 문의");
    });
  });

  it("shows which address the reply will go to", async () => {
    stub();
    renderPage(<ContactPage />);

    expect(await screen.findByText("singer@boorda.kr")).toBeInTheDocument();
    // Displayed, not editable.
    expect(screen.queryByLabelText("답변받을 이메일")).toBeNull();
  });

  it("will not submit an empty subject or message", async () => {
    stub();
    const user = userEvent.setup();
    renderPage(<ContactPage />);

    expect(screen.getByRole("button", { name: "문의 접수" })).toBeDisabled();

    await user.type(screen.getByLabelText("제목"), "제목만 있음");
    expect(screen.getByRole("button", { name: "문의 접수" })).toBeDisabled();
  });

  it("goes to the new inquiry on success", async () => {
    stub({
      "/support/inquiries": () => json({ reference: "SUP-ABCD1234", status: "OPEN" }, 201),
    });
    const user = userEvent.setup();
    renderPage(<ContactPage />);

    await user.type(screen.getByLabelText("제목"), "결제 문의");
    await user.type(screen.getByLabelText("내용"), "확인 부탁드립니다.");
    await user.click(screen.getByRole("button", { name: "문의 접수" }));

    await waitFor(() =>
      expect(push).toHaveBeenCalledWith("/support/inquiries/SUP-ABCD1234?filed=1"),
    );
  });

  it("explains a rate limit rather than failing silently", async () => {
    stub({ "/support/inquiries": () => json({ detail: "Too many" }, 429) });
    const user = userEvent.setup();
    renderPage(<ContactPage />);

    await user.type(screen.getByLabelText("제목"), "결제 문의");
    await user.type(screen.getByLabelText("내용"), "확인 부탁드립니다.");
    await user.click(screen.getByRole("button", { name: "문의 접수" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/너무 많이 접수/);
  });
});

// ── my inquiries ─────────────────────────────────────────────────────

describe("my inquiries", () => {
  it("offers a way in when there are none", async () => {
    stub();
    renderPage(<InquiriesPage />);

    expect(await screen.findByText("아직 접수한 문의가 없습니다")).toBeInTheDocument();
  });

  it("lists tickets newest-first as the server returned them", async () => {
    stub({
      "/support/inquiries": () =>
        json({
          items: [
            {
              reference: "SUP-NEWER01",
              category: "BILLING",
              subject: "최근 문의",
              status: "OPEN",
              created_at: "2026-08-28T10:00:00Z",
            },
            {
              reference: "SUP-OLDER01",
              category: "BUG",
              subject: "이전 문의",
              status: "RESOLVED",
              created_at: "2026-08-01T10:00:00Z",
            },
          ],
          total: 2,
        }),
    });
    renderPage(<InquiriesPage />);

    const items = await screen.findAllByRole("listitem");
    expect(within(items[0]).getByText("최근 문의")).toBeInTheDocument();
    expect(within(items[0]).getByText("접수됨")).toBeInTheDocument();
    expect(within(items[1]).getByText("답변완료")).toBeInTheDocument();
  });

  it("links each ticket by its reference, not a database id", async () => {
    stub({
      "/support/inquiries": () =>
        json({
          items: [
            {
              reference: "SUP-ABCD1234",
              category: "ACCOUNT",
              subject: "계정 문의",
              status: "OPEN",
              created_at: "2026-08-28T10:00:00Z",
            },
          ],
          total: 1,
        }),
    });
    renderPage(<InquiriesPage />);

    const link = await screen.findByRole("link", { name: /계정 문의/ });
    expect(link).toHaveAttribute("href", "/support/inquiries/SUP-ABCD1234");
  });
});
