"use client";

/**
 * Settings — the account hub, six sections, one of which works.
 *
 * ACCOUNT and the logout control in SECURITY read and use the auth API
 * that exists (`/auth/me`, `/auth/logout`). SUBSCRIPTION, CREDITS,
 * PAYMENTS, DATA and the rest of SECURITY have no backend, so they are
 * laid out in full and marked unavailable.
 *
 * The discipline that matters here: a section with no backend renders no
 * button. Not a disabled button that hints at a save, and certainly not
 * an enabled one — a control the user can press that silently does
 * nothing is the worst version of this page. Where a real destination
 * exists (`/plans`, `/library`) the section links to it instead.
 *
 * Plan *comparison* lives at `/plans` and is not duplicated here; this
 * page manages the account, that page sells the tiers.
 */

import { useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { Button, ButtonLink, Card, Skeleton } from "@/components/ui";
import { useEntitlement } from "@/components/EntitlementProvider";
import { UsageNotice } from "@/components/UsageMeter";
import { formatPeriod, formatPriceKrw, formatSongs } from "@/lib/plans";

//: The sections, in the order they appear. Used for the jump list so
//: the page stays navigable once every section has content.
const SECTIONS = [
  { id: "account", label: "계정" },
  { id: "subscription", label: "구독" },
  { id: "usage", label: "사용량" },
  { id: "payments", label: "결제" },
  { id: "data", label: "데이터" },
  { id: "security", label: "보안" },
] as const;

function Section({
  id,
  title,
  description,
  children,
}: {
  id: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-heading`} className="flex scroll-mt-6 flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <h2 id={`${id}-heading`} className="text-sm font-semibold text-[var(--text-primary)]">
          {title}
        </h2>
        <p className="text-xs text-[var(--text-muted)]">{description}</p>
      </div>
      {children}
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 border-b border-[var(--border-subtle)] px-5 py-4 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <dt className="text-sm text-[var(--text-secondary)]">{label}</dt>
      <dd className="text-sm font-medium text-[var(--text-primary)]">{value}</dd>
    </div>
  );
}

/** A value the product cannot report yet. */
function Unknown() {
  return <span className="text-[var(--text-muted)]">미정</span>;
}

/**
 * A capability that exists in the layout and not in the API.
 *
 * Renders no control. See the module docstring.
 */
function NotYet({ children }: { children: React.ReactNode }) {
  return (
    <Card className="px-5 py-4">
      <p className="text-sm text-[var(--text-secondary)]">{children}</p>
      <p className="mt-2 inline-flex rounded-[var(--radius-full)] bg-[var(--surface-sunken)] px-2.5 py-1 text-[11px] font-medium text-[var(--text-muted)]">
        준비 중
      </p>
    </Card>
  );
}

function formatJoined(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "long" }).format(date);
}

export default function SettingsPage() {
  const { status, user, signOut } = useAuth();
  const { entitlement } = useEntitlement();
  const [leaving, setLeaving] = useState(false);

  return (
    <div className="flex max-w-3xl flex-col gap-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          설정
        </h1>
        <p className="text-sm text-[var(--text-secondary)]">
          계정, 구독, 결제, 데이터를 관리합니다.
        </p>
      </header>

      <nav aria-label="설정 항목" className="flex flex-wrap gap-2">
        {SECTIONS.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            className="rounded-[var(--radius-full)] border border-[var(--border-default)] px-3 py-1 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
          >
            {section.label}
          </a>
        ))}
      </nav>

      <Section id="account" title="계정" description="현재 로그인한 계정 정보입니다.">
        <Card className="p-0">
          {status === "loading" || !user ? (
            <div className="flex flex-col gap-3 p-5">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-4 w-56" />
            </div>
          ) : (
            <dl className="flex flex-col">
              <Row label="이메일" value={user.email} />
              <Row
                label="표시 이름"
                value={
                  user.display_name?.trim() || (
                    <span className="text-[var(--text-muted)]">설정하지 않음</span>
                  )
                }
              />
              <Row label="가입일" value={formatJoined(user.created_at)} />
            </dl>
          )}
        </Card>
        {/*
          Read-only. The auth API has signup, login, logout and /me —
          there is no profile update endpoint, so there is no save.
        */}
        <NotYet>표시 이름과 프로필 수정은 아직 제공하지 않습니다.</NotYet>
      </Section>

      <Section id="subscription" title="구독" description="현재 요금제와 이용 범위입니다.">
        <Card className="p-0">
          <dl className="flex flex-col">
            <Row label="현재 플랜" value={entitlement ? entitlement.plan.display_name : <Unknown />} />
            <Row
              label="월 요금"
              value={entitlement ? formatPriceKrw(entitlement.plan.monthly_price_krw) : <Unknown />}
            />
            <Row
              label="다운로드"
              value={
                entitlement ? (entitlement.download_mp3 ? "MP3 · WAV" : "미포함") : <Unknown />
              }
            />
            <Row
              label="상업적 이용"
              value={entitlement ? (entitlement.commercial_use ? "가능" : "불가") : <Unknown />}
            />
          </dl>
        </Card>
        <div>
          <ButtonLink href="/plans" variant="secondary">
            플랜 비교하기
          </ButtonLink>
        </div>
        {/*
          The plan and its entitlements are real — they come from the
          server and the server enforces them. Changing plans is not:
          there is no payment provider, so there is nothing to charge and
          no subscription to cancel.
        */}
        <NotYet>플랜 변경과 구독 해지는 아직 제공하지 않습니다.</NotYet>
      </Section>

      <Section
        id="usage"
        title="사용량"
        description="이번 기간에 만든 곡 수입니다. 실패한 생성은 차감되지 않습니다."
      >
        <Card className="p-0">
          <dl className="flex flex-col">
            <Row
              label="이번 기간"
              value={entitlement ? formatPeriod(entitlement) : <Unknown />}
            />
            <Row
              label="생성한 곡"
              value={
                entitlement
                  ? `${formatSongs(entitlement.generation_used)} / ${formatSongs(entitlement.generation_limit)}`
                  : <Unknown />
              }
            />
            <Row
              label="남은 생성"
              value={entitlement ? formatSongs(entitlement.generation_remaining) : <Unknown />}
            />
          </dl>
        </Card>
        {entitlement ? <UsageNotice entitlement={entitlement} /> : null}
        {/* No ledger yet: the reservations exist, but nothing renders
            a per-song history and inventing one would be a claim. */}
        <NotYet>곡별 사용 내역은 아직 제공하지 않습니다.</NotYet>
      </Section>

      <Section id="payments" title="결제" description="결제 수단과 결제 내역입니다.">
        <Card className="p-0">
          <dl className="flex flex-col">
            <Row label="결제 수단" value={<Unknown />} />
            <Row label="최근 결제" value={<Unknown />} />
          </dl>
        </Card>
        <NotYet>결제 수단 등록, 결제 내역, 영수증, 환불은 아직 제공하지 않습니다.</NotYet>
      </Section>

      <Section id="data" title="데이터" description="내 음악과 계정 데이터입니다.">
        <Card className="px-5 py-4">
          <p className="text-sm text-[var(--text-secondary)]">
            만든 음악은 라이브러리에서 개별로 내려받거나 삭제할 수 있습니다.
          </p>
          <div className="mt-3">
            <ButtonLink href="/library" variant="secondary" size="sm">
              라이브러리 열기
            </ButtonLink>
          </div>
        </Card>
        {/*
          Account deletion is destructive and has no endpoint. A button
          here would either do nothing or, worse, look like it worked.
        */}
        <NotYet>전체 데이터 내보내기와 계정 삭제는 아직 제공하지 않습니다.</NotYet>
      </Section>

      <Section id="security" title="보안" description="로그인과 계정 보호 설정입니다.">
        <Card className="px-5 py-4">
          <p className="text-sm text-[var(--text-secondary)]">
            이 브라우저의 세션에서 로그아웃합니다.
          </p>
          <div className="mt-3">
            <Button
              variant="secondary"
              busy={leaving}
              onClick={() => {
                setLeaving(true);
                // signOut navigates; no need to clear the flag on success.
                void signOut().catch(() => setLeaving(false));
              }}
            >
              로그아웃
            </Button>
          </div>
        </Card>
        {/*
          Password change, session listing and social linking each need a
          backend that does not exist. Listed together so the section
          reads as one honest gap rather than three empty cards.
        */}
        <NotYet>
          비밀번호 변경, 로그인된 기기 관리, 구글·카카오 계정 연결은 아직 제공하지 않습니다.
        </NotYet>
      </Section>
    </div>
  );
}
