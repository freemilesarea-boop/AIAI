"use client";

/**
 * BOORDA home — a dashboard for someone who is already signed in.
 *
 * Not a landing page. Everyone who reaches this route has an account
 * (the shell's RequireAuth sees to that), so the job is to get them to
 * the next track rather than to explain the product.
 *
 * Plan and credits are rendered as real components with real layout and
 * no data, because the billing backend does not exist. They read "미정"
 * rather than a plausible-looking number: a placeholder that lies is
 * worse than one that admits what it is.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { SongCard } from "@/components/SongCard";
import { Button, ButtonLink, Card, EmptyState, SkeletonCard } from "@/components/ui";
import { useAuth } from "@/components/auth/AuthProvider";
import { listGenerations, type Generation } from "@/lib/api";
import { creditBalance, currentPlan, formatCredits } from "@/lib/plans";

//: How many recent tracks the dashboard shows. Enough to recognise
//: what you were last working on, few enough that the CTA stays above
//: the fold on a laptop.
const RECENT_LIMIT = 4;

function greeting(name: string | null, email: string): string {
  const who = name?.trim() || email.split("@")[0];
  return `${who}님, 환영합니다`;
}

/**
 * A figure the product cannot produce yet.
 *
 * Kept as its own component so that when the billing service lands
 * there is exactly one place that changes, and so the "미정" state is
 * impossible to confuse with a loaded value.
 */
function PendingStat({
  label,
  value,
  hint,
  action,
}: {
  label: string;
  value: string;
  hint: string;
  action?: React.ReactNode;
}) {
  return (
    <Card className="flex flex-col gap-1 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
        {label}
      </p>
      <p className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">{value}</p>
      <p className="text-xs text-[var(--text-muted)]">{hint}</p>
      {action ? <div className="mt-2">{action}</div> : null}
    </Card>
  );
}

export default function HomePage() {
  const { user } = useAuth();
  const [recent, setRecent] = useState<Generation[] | null>(null);
  const [failed, setFailed] = useState(false);

  //: Neither has a backend yet; both are typed as the real thing so
  //: the components are already correct when one arrives.
  const plan = currentPlan();
  const credits = creditBalance();

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const page = await listGenerations(RECENT_LIMIT, 0, signal);
      setRecent(page.items);
      setFailed(false);
    } catch {
      if (!signal?.aborted) setFailed(true);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-1">
        <p className="text-sm text-[var(--text-secondary)]">
          {user ? greeting(user.display_name, user.email) : "환영합니다"}
        </p>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          오늘은 어떤 음악을 만들까요?
        </h1>
      </header>

      {/* The front door. Everything else on this page is secondary. */}
      <Card className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-1">
          <h2 className="text-lg font-semibold text-[var(--text-primary)]">음악 만들기</h2>
          <p className="max-w-prose text-sm text-[var(--text-secondary)]">
            원하는 분위기를 설명하고 가사를 더하면 완성된 트랙이 나옵니다.
          </p>
        </div>
        <ButtonLink href="/create" variant="primary" size="lg">
          음악 만들기
        </ButtonLink>
      </Card>

      {/*
        LAB is secondary by construction: a quiet row under the CTA, not
        a second hero. Create stays the front door.
      */}
      <Link
        href="/lab"
        className="flex items-center justify-between gap-4 rounded-[var(--radius-lg)] border border-[var(--border-subtle)] bg-[var(--surface-raised)] px-5 py-4 transition-colors hover:border-[var(--border-strong)]"
      >
        <span className="flex flex-col gap-0.5">
          <span className="flex items-center gap-2">
            <span className="text-sm font-semibold text-[var(--text-primary)]">BOORDA LAB</span>
            <span className="rounded-[var(--radius-full)] bg-[var(--surface-overlay)] px-1.5 py-0.5 text-[9px] font-semibold tracking-wide text-[var(--text-muted)]">
              BETA
            </span>
          </span>
          <span className="text-xs text-[var(--text-secondary)]">
            새로운 모델과 기능을 미리 확인하세요.
          </span>
        </span>
        <span aria-hidden="true" className="text-[var(--text-muted)]">
          →
        </span>
      </Link>

      <section aria-labelledby="account-heading" className="flex flex-col gap-3">
        <h2 id="account-heading" className="text-sm font-semibold text-[var(--text-primary)]">
          내 계정
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <PendingStat
            label="현재 플랜"
            value={plan?.name ?? "미정"}
            hint="요금제가 아직 정해지지 않았습니다."
            action={
              <Link
                href="/plans"
                className="text-xs font-medium text-[var(--brand)] hover:underline"
              >
                플랜 살펴보기
              </Link>
            }
          />
          <PendingStat
            label="남은 크레딧"
            value={formatCredits(credits)}
            hint="크레딧 정책이 아직 정해지지 않았습니다."
          />
        </div>
      </section>

      <section aria-labelledby="recent-heading" className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <h2 id="recent-heading" className="text-sm font-semibold text-[var(--text-primary)]">
            최근 만든 음악
          </h2>
          <Link
            href="/library"
            className="text-xs font-medium text-[var(--brand)] hover:underline"
          >
            라이브러리 전체 보기
          </Link>
        </div>

        {failed ? (
          <EmptyState
            title="최근 음악을 불러오지 못했습니다"
            description="연결 문제입니다. 저장된 음악은 그대로 있습니다."
            action={
              <Button variant="secondary" onClick={() => void load()}>
                다시 시도
              </Button>
            }
          />
        ) : recent === null ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : recent.length === 0 ? (
          <EmptyState
            title="아직 만든 음악이 없습니다"
            description="첫 트랙을 만들면 여기에 표시됩니다."
            action={
              <ButtonLink href="/create" variant="primary">
                음악 만들기
              </ButtonLink>
            }
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {recent.map((generation) => (
              <SongCard
                key={generation.id}
                generation={generation}
                onChanged={(updated) =>
                  setRecent((items) =>
                    items?.map((item) => (item.id === updated.id ? updated : item)) ?? items,
                  )
                }
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
