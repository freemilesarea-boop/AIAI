/**
 * The footer, and the only place BOORDA states who operates it.
 *
 * Business identity comes from `lib/legal`, not from strings typed here,
 * so a corrected registration number is one edit rather than four and
 * the terms page can never disagree with this one about who we are.
 *
 * Fields BOORDA has not confirmed are absent rather than guessed. Korean
 * e-commerce law requires a 통신판매업자 to show an address, a phone
 * number and a 통신판매업 신고번호; none is configured, so none is
 * rendered. A plausible-looking placeholder on a legal notice is worse
 * than a visible gap, because only one of the two gets fixed.
 */

import Link from "next/link";

import { BUSINESS, SERVICE_NAME } from "@/lib/legal";

const SERVICE_LINKS = [
  { href: "/create", label: "음악 만들기" },
  { href: "/plans", label: "요금제" },
];

const SUPPORT_LINKS = [
  { href: "/support", label: "고객지원" },
  { href: "/support/contact", label: "문의하기" },
];

const LEGAL_LINKS = [
  { href: "/terms", label: "이용약관" },
  { href: "/privacy", label: "개인정보처리방침" },
  { href: "/refund-policy", label: "구독·결제·환불 정책" },
];

function Column({
  title,
  links,
}: {
  title: string;
  links: { href: string; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-xs font-semibold text-[var(--text-primary)]">{title}</h2>
      <ul className="flex flex-col gap-1.5">
        {links.map((link) => (
          <li key={link.href}>
            <Link
              href={link.href}
              className="text-xs text-[var(--text-secondary)] transition-colors hover:text-[var(--text-primary)]"
            >
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Footer() {
  // Rendered only when configured — see the module docstring.
  const optional = [
    BUSINESS.address ? { label: "주소", value: BUSINESS.address } : null,
    BUSINESS.phone ? { label: "전화", value: BUSINESS.phone } : null,
    BUSINESS.mailOrderNumber
      ? { label: "통신판매업신고", value: BUSINESS.mailOrderNumber }
      : null,
  ].filter((item): item is { label: string; value: string } => item !== null);

  return (
    <footer className="mt-16 border-t border-[var(--border-subtle)] px-5 py-8 sm:px-8">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-8">
        <div className="flex flex-col gap-8 sm:flex-row sm:justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-bold tracking-tight text-[var(--text-primary)]">
              {SERVICE_NAME}
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              설명을 입력하면 완성된 음악이 나옵니다.
            </span>
          </div>

          <div className="grid grid-cols-2 gap-8 sm:grid-cols-3">
            <Column title="서비스" links={SERVICE_LINKS} />
            <Column title="고객지원" links={SUPPORT_LINKS} />
            <Column title="법적 고지" links={LEGAL_LINKS} />
          </div>
        </div>

        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[var(--text-muted)]">
          <div className="flex gap-1.5">
            <dt>상호명</dt>
            <dd className="text-[var(--text-secondary)]">{BUSINESS.name}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt>대표자</dt>
            <dd className="text-[var(--text-secondary)]">{BUSINESS.representative}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt>사업자등록번호</dt>
            <dd className="text-[var(--text-secondary)]">{BUSINESS.registrationNumber}</dd>
          </div>
          {optional.map((item) => (
            <div key={item.label} className="flex gap-1.5">
              <dt>{item.label}</dt>
              <dd className="text-[var(--text-secondary)]">{item.value}</dd>
            </div>
          ))}
          <div className="flex gap-1.5">
            <dt>고객문의</dt>
            <dd>
              <a
                href={`mailto:${BUSINESS.contactEmail}`}
                className="text-[var(--text-secondary)] underline underline-offset-2"
              >
                {BUSINESS.contactEmail}
              </a>
            </dd>
          </div>
        </dl>

        <p className="text-[11px] text-[var(--text-muted)]">
          © 2026 {SERVICE_NAME}. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
