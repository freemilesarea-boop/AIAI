"use client";

/**
 * The frame around every operator page.
 *
 * It renders a sidebar and refuses to render the console to an account
 * without a role — and that refusal is a courtesy, not a control. The
 * real boundary is the API: every `/v1/admin/*` request is checked
 * against the session's own row, so a customer who reaches this route
 * by typing it sees a page whose every request answers 403.
 *
 * Saying that plainly matters, because the failure mode of an admin
 * console is someone reading a client-side check as the protection and
 * later relaxing the server one to "avoid the duplication".
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { EmptyState } from "@/components/ui";
import { isAdmin, isSuperAdmin } from "@/lib/admin";

interface NavItem {
  href: string;
  label: string;
  /** Only a super administrator is offered this. */
  superOnly?: boolean;
}

const NAV: NavItem[] = [
  { href: "/admin", label: "대시보드" },
  { href: "/admin/revenue", label: "매출" },
  { href: "/admin/users", label: "회원" },
  { href: "/admin/support", label: "고객문의" },
  { href: "/admin/email", label: "이메일" },
  { href: "/admin/audit", label: "활동 기록" },
  { href: "/admin/admins", label: "관리자", superOnly: true },
];

export function AdminShell({ children }: { children: ReactNode }) {
  const { status, user } = useAuth();
  const pathname = usePathname() ?? "/admin";

  if (status === "loading") {
    return (
      <div role="status" aria-live="polite" className="py-16 text-center">
        <span className="text-sm text-[var(--text-muted)]">불러오는 중…</span>
      </div>
    );
  }

  if (!isAdmin(user?.role)) {
    return (
      <EmptyState
        title="접근 권한이 없습니다"
        description="운영 관리자만 이용할 수 있는 영역입니다."
        action={
          <Link
            href="/"
            className="text-sm font-medium text-[var(--accent)] underline underline-offset-4"
          >
            홈으로 돌아가기
          </Link>
        }
      />
    );
  }

  const items = NAV.filter((item) => !item.superOnly || isSuperAdmin(user?.role));

  return (
    <div className="flex flex-col gap-6 lg:flex-row lg:gap-10">
      <nav aria-label="운영 관리" className="lg:w-48 lg:shrink-0">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          운영 관리
        </p>
        <ul className="flex gap-1 overflow-x-auto lg:flex-col lg:overflow-visible">
          {items.map((item) => {
            const active =
              item.href === "/admin" ? pathname === "/admin" : pathname.startsWith(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={`block whitespace-nowrap rounded-[var(--radius-md)] px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-[var(--accent-muted)] font-medium text-[var(--accent)]"
                      : "text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]"
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
