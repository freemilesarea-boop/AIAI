/**
 * What the account controls say before they do anything.
 *
 * The server owns every rule here — these tests are about the interface
 * telling the truth about consequences, and about the destructive one
 * being hard to trigger by accident.
 *
 * Three things in particular:
 *
 *  - Closing an account takes two deliberate steps and a password. One
 *    click cannot do it.
 *  - The password form says it will sign other devices out, because it
 *    will.
 *  - A live subscription is explained rather than reported as a generic
 *    failure — "your account was not closed" is useless without "because
 *    you would keep being charged".
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthProvider } from "@/components/auth/AuthProvider";
import { DangerZone, DisplayNameForm, PasswordForm } from "@/components/AccountPanel";
import { ToastProvider } from "@/components/ui/Toast";

vi.mock("next/navigation", () => ({
  usePathname: () => "/settings",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
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
      return json({});
    }),
  );
  return { calls };
}

function renderPanel(node: React.ReactNode) {
  return render(
    <AuthProvider>
      <ToastProvider>{node}</ToastProvider>
    </AuthProvider>,
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

// ── password ─────────────────────────────────────────────────────────

describe("password change", () => {
  it("sends only the three password fields", async () => {
    const stubbed = stub({ "/auth/password": () => new Response(null, { status: 204 }) });
    const user = userEvent.setup();
    renderPanel(<PasswordForm />);

    await user.type(screen.getByLabelText("현재 비밀번호"), "old-password");
    await user.type(screen.getByLabelText("새 비밀번호"), "a new long passphrase");
    await user.type(screen.getByLabelText("새 비밀번호 확인"), "a new long passphrase");
    await user.click(screen.getByRole("button", { name: "비밀번호 변경" }));

    await waitFor(() => {
      const call = stubbed.calls.find((c) => c.url.includes("/auth/password"));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call!.init!.body));
      expect(Object.keys(body).sort()).toEqual([
        "current_password",
        "new_password",
        "new_password_confirm",
      ]);
      // No user id anywhere: the account is the session's.
      expect(body).not.toHaveProperty("user_id");
      expect(body).not.toHaveProperty("email");
    });
  });

  it("will not submit while the confirmation does not match", async () => {
    stub();
    const user = userEvent.setup();
    renderPanel(<PasswordForm />);

    await user.type(screen.getByLabelText("현재 비밀번호"), "old-password");
    await user.type(screen.getByLabelText("새 비밀번호"), "a new long passphrase");
    await user.type(screen.getByLabelText("새 비밀번호 확인"), "something else");

    expect(screen.getByText("새 비밀번호가 일치하지 않습니다.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "비밀번호 변경" })).toBeDisabled();
  });

  it("says up front that other devices will be signed out", () => {
    stub();
    renderPanel(<PasswordForm />);

    expect(
      screen.getByText(/이 브라우저를 제외한 모든 기기에서 로그아웃됩니다/),
    ).toBeInTheDocument();
  });

  it("shows the server's reason when it refuses", async () => {
    stub({
      "/auth/password": () => json({ detail: "Your current password is incorrect." }, 400),
    });
    const user = userEvent.setup();
    renderPanel(<PasswordForm />);

    await user.type(screen.getByLabelText("현재 비밀번호"), "wrong");
    await user.type(screen.getByLabelText("새 비밀번호"), "a new long passphrase");
    await user.type(screen.getByLabelText("새 비밀번호 확인"), "a new long passphrase");
    await user.click(screen.getByRole("button", { name: "비밀번호 변경" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/current password is incorrect/i);
  });

  it("clears the fields after a successful change", async () => {
    stub({ "/auth/password": () => new Response(null, { status: 204 }) });
    const user = userEvent.setup();
    renderPanel(<PasswordForm />);

    await user.type(screen.getByLabelText("현재 비밀번호"), "old-password");
    await user.type(screen.getByLabelText("새 비밀번호"), "a new long passphrase");
    await user.type(screen.getByLabelText("새 비밀번호 확인"), "a new long passphrase");
    await user.click(screen.getByRole("button", { name: "비밀번호 변경" }));

    await screen.findByRole("status");
    expect(screen.getByLabelText("현재 비밀번호")).toHaveValue("");
    expect(screen.getByLabelText("새 비밀번호")).toHaveValue("");
  });
});

// ── display name ─────────────────────────────────────────────────────

describe("display name", () => {
  it("saves the trimmed value", async () => {
    const stubbed = stub({ "/auth/profile": () => json({ ...USER, display_name: "부르다" }) });
    const user = userEvent.setup();
    renderPanel(<DisplayNameForm />);

    await user.type(screen.getByLabelText("표시 이름"), "  부르다  ");
    await user.click(screen.getByRole("button", { name: "저장" }));

    await waitFor(() => {
      const call = stubbed.calls.find((c) => c.url.includes("/auth/profile"));
      expect(JSON.parse(String(call!.init!.body))).toEqual({ display_name: "부르다" });
    });
  });

  it("sends null when the field is emptied", async () => {
    const stubbed = stub({ "/auth/profile": () => json({ ...USER, display_name: null }) });
    const user = userEvent.setup();
    renderPanel(<DisplayNameForm />);

    await user.click(screen.getByRole("button", { name: "저장" }));

    await waitFor(() => {
      const call = stubbed.calls.find((c) => c.url.includes("/auth/profile"));
      expect(JSON.parse(String(call!.init!.body))).toEqual({ display_name: null });
    });
  });
});

// ── closing the account ──────────────────────────────────────────────

describe("account closure", () => {
  it("cannot be done in one click", async () => {
    const stubbed = stub();
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));

    // The dialog opened; nothing was sent.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(stubbed.calls.some((c) => c.url.includes("/account/delete"))).toBe(false);
  });

  it("explains the consequences before asking for anything", async () => {
    stub();
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/되돌릴 수 없습니다/)).toBeInTheDocument();
    expect(within(dialog).getByText(/결제 내역은 회계·분쟁 대응을 위해 보관됩니다/)).toBeInTheDocument();
    // No password field yet — step one is information only.
    expect(within(dialog).queryByLabelText("현재 비밀번호")).toBeNull();
  });

  it("asks for the password on the second step", async () => {
    const stubbed = stub();
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "계속" }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByLabelText("현재 비밀번호")).toBeInTheDocument();
    // Still nothing sent, and the confirm button is inert while empty.
    expect(within(dialog).getByRole("button", { name: "회원 탈퇴" })).toBeDisabled();
    expect(stubbed.calls.some((c) => c.url.includes("/account/delete"))).toBe(false);
  });

  it("sends only the password", async () => {
    const stubbed = stub({
      "/account/delete": () => new Response(null, { status: 204 }),
    });
    vi.stubGlobal("location", { assign: vi.fn() } as unknown as Location);
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "계속" }));
    await user.type(screen.getByLabelText("현재 비밀번호"), "old-password");
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "회원 탈퇴" }));

    await waitFor(() => {
      const call = stubbed.calls.find((c) => c.url.includes("/account/delete"));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call!.init!.body));
      expect(Object.keys(body)).toEqual(["current_password"]);
      expect(body).not.toHaveProperty("user_id");
    });
  });

  it("explains a live subscription instead of reporting a bare failure", async () => {
    /**
     * "Your account was not closed" is useless without "because you
     * would keep being charged for something you cannot reach".
     */
    stub({ "/account/delete": () => json({ detail: "SUBSCRIPTION_ACTIVE" }, 409) });
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "계속" }));
    await user.type(screen.getByLabelText("현재 비밀번호"), "old-password");
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "회원 탈퇴" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/먼저 구독을 해지한 뒤/);
    expect(alert).toHaveTextContent(/결제가 계속될 수 있습니다/);
  });

  it("can be abandoned with Escape", async () => {
    stub();
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("marks the dialog for assistive technology", async () => {
    stub();
    const user = userEvent.setup();
    renderPanel(<DangerZone />);

    await user.click(screen.getByRole("button", { name: "회원 탈퇴" }));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby");
  });
});
