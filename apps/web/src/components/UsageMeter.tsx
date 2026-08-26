"use client";

/**
 * How many songs are left this month, said once.
 *
 * Home, Create and Settings all show this. Writing it three times would
 * be three chances for the wording, the warning threshold or the
 * arithmetic to drift apart — and the number is the same number, so it
 * should be the same component.
 *
 * Every figure comes from the server's entitlement. Nothing here counts
 * anything itself.
 */

import Link from "next/link";

import { useEntitlement } from "@/components/EntitlementProvider";
import { Card } from "@/components/ui";
import {
  formatPeriod,
  formatSongs,
  isExhausted,
  isNearlyExhausted,
  usageRatio,
  type Entitlement,
} from "@/lib/plans";

function barColour(entitlement: Entitlement): string {
  if (isExhausted(entitlement)) return "bg-[var(--danger)]";
  if (isNearlyExhausted(entitlement)) return "bg-[var(--accent)]";
  return "bg-[var(--brand)]";
}

/** The bar and its labels. Used inside cards and inline alike. */
export function UsageBar({ entitlement }: { entitlement: Entitlement }) {
  const used = entitlement.generation_used;
  const limit = entitlement.generation_limit;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-[var(--text-secondary)]">이번 달 생성</span>
        <span className="font-medium text-[var(--text-primary)]">
          {formatSongs(used)} / {formatSongs(limit)}
        </span>
      </div>
      <div
        role="progressbar"
        aria-label="이번 달 생성 사용량"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={limit}
        className="h-1.5 w-full overflow-hidden rounded-[var(--radius-full)] bg-[var(--surface-sunken)]"
      >
        <div
          className={`h-full rounded-[var(--radius-full)] transition-[width] ${barColour(entitlement)}`}
          style={{ width: `${usageRatio(entitlement) * 100}%` }}
        />
      </div>
    </div>
  );
}

/**
 * The warning, the exhausted state, or nothing.
 *
 * Silent while there is plenty left: a banner shown at 3 songs out of
 * 200 is a banner nobody reads at 199.
 */
export function UsageNotice({ entitlement }: { entitlement: Entitlement }) {
  if (isExhausted(entitlement)) {
    return (
      <div
        role="status"
        className="flex flex-col gap-2 rounded-[var(--radius-md)] border border-[var(--danger)] bg-[var(--danger-muted)] px-4 py-3 text-sm"
      >
        <p className="font-medium text-[var(--text-primary)]">
          이번 달 생성 한도를 모두 사용했습니다
        </p>
        <p className="text-[var(--text-secondary)]">
          {formatPeriod(entitlement)} 기간의 {formatSongs(entitlement.generation_limit)}을 모두
          사용했습니다. 다음 기간에 다시 초기화되며, 지금 더 만들려면 플랜을 올려야 합니다.
        </p>
        <Link
          href="/plans"
          className="w-fit text-sm font-medium text-[var(--brand-text)] underline underline-offset-4"
        >
          플랜 보기
        </Link>
      </div>
    );
  }

  if (isNearlyExhausted(entitlement)) {
    return (
      <div
        role="status"
        className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-[var(--radius-md)] border border-[var(--accent)] bg-[var(--accent-muted)] px-4 py-3 text-sm text-[var(--text-secondary)]"
      >
        <span className="font-medium text-[var(--text-primary)]">
          {formatSongs(entitlement.generation_remaining)} 남았습니다
        </span>
        <Link href="/plans" className="text-[var(--brand-text)] underline underline-offset-4">
          플랜 보기
        </Link>
      </div>
    );
  }

  return null;
}

/** The full card: plan name, bar, period, and the notice when relevant. */
export function UsageCard({ className }: { className?: string }) {
  const { entitlement, loading, error } = useEntitlement();

  if (loading) {
    return (
      <Card className={`p-5 ${className ?? ""}`}>
        <p className="text-sm text-[var(--text-muted)]">사용량 불러오는 중…</p>
      </Card>
    );
  }

  // Nothing rather than a guess: an invented allowance is worse than an
  // absent one.
  if (error || !entitlement) return null;

  return (
    <Card className={`flex flex-col gap-4 p-5 ${className ?? ""}`}>
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">사용량</h2>
        <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
          {entitlement.plan.display_name}
        </span>
      </div>
      <UsageBar entitlement={entitlement} />
      <p className="text-xs text-[var(--text-muted)]">{formatPeriod(entitlement)}</p>
      <UsageNotice entitlement={entitlement} />
    </Card>
  );
}
