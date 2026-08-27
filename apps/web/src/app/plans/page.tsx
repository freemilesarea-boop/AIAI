"use client";

/**
 * Plans — four tiers with real prices, and no checkout.
 *
 * The prices and allowances here are the ones the server enforces: they
 * are fetched from `/v1/plans`, which serves the same definition the
 * generation and download routes read. Nothing on this page is a
 * number typed into a component.
 *
 * There is still no payment provider, and the page says so plainly
 * rather than showing a subscribe button that opens nothing. A control
 * that cannot do what it says is worse than no control — the user
 * presses it, nothing happens, and they conclude the product is broken
 * rather than unfinished.
 */

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { CheckoutDialog } from "@/components/CheckoutDialog";
import { useEntitlement } from "@/components/EntitlementProvider";
import { Button, Card } from "@/components/ui";
import { UsageBar } from "@/components/UsageMeter";
import { fetchBillingStatus, hasPaidAccess, type BillingStatus } from "@/lib/billing";
import { fetchPlans, formatPriceKrw, formatSongs, type Plan } from "@/lib/plans";

function Mark({ included }: { included: boolean }) {
  return included ? (
    <span aria-label="포함" className="text-[var(--brand)]">
      ✓
    </span>
  ) : (
    <span aria-label="미포함" className="text-[var(--text-muted)]">
      ✕
    </span>
  );
}

function planAction({
  plan,
  current,
  checkoutAvailable,
  subscribed,
  onSubscribe,
}: {
  plan: Plan;
  current: boolean;
  checkoutAvailable: boolean;
  subscribed: boolean;
  onSubscribe: (plan: Plan) => void;
}) {
  const note = (text: string) => (
    <p className="rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2 text-xs text-[var(--text-muted)]">
      {text}
    </p>
  );

  if (plan.monthly_price_krw === 0) return note("가입하면 바로 사용할 수 있습니다.");
  if (current) return note("현재 사용 중인 플랜입니다.");
  if (!checkoutAvailable) return note("결제는 아직 준비 중입니다.");
  if (subscribed) return note("플랜을 바꾸려면 현재 구독을 먼저 해지해 주세요.");

  return (
    <Button
      type="button"
      variant={plan.recommended ? "primary" : "secondary"}
      className="w-full"
      onClick={() => onSubscribe(plan)}
    >
      {plan.display_name} 시작하기
    </Button>
  );
}

const TAGLINES: Record<string, string> = {
  free: "부르다를 처음 써보는 분께",
  basic: "꾸준히 만드는 분께",
  pro: "작업량이 많은 분께",
  creator: "매일 작업하는 분께",
};

function PlanCard({
  plan,
  current,
  checkoutAvailable,
  subscribed,
  onSubscribe,
}: {
  plan: Plan;
  current: boolean;
  checkoutAvailable: boolean;
  /** True when this account already pays for something. */
  subscribed: boolean;
  onSubscribe: (plan: Plan) => void;
}) {
  return (
    <Card
      className={
        "flex flex-col gap-4 p-6" +
        (plan.recommended ? " border-[var(--brand)] ring-1 ring-[var(--brand)]" : "")
      }
      data-testid={`plan-${plan.plan_id}`}
    >
      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--text-primary)]">{plan.display_name}</h2>
          {plan.recommended ? (
            <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
              추천
            </span>
          ) : null}
          {current ? (
            <span className="rounded-[var(--radius-full)] border border-[var(--border-default)] px-2 py-0.5 text-[11px] font-medium text-[var(--text-secondary)]">
              현재 플랜
            </span>
          ) : null}
        </div>
        <p className="text-sm text-[var(--text-secondary)]">{TAGLINES[plan.plan_id] ?? ""}</p>
      </div>

      <div className="flex flex-col gap-0.5">
        <p className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
          {formatPriceKrw(plan.monthly_price_krw)}
        </p>
        <p className="text-xs text-[var(--text-muted)]">
          {plan.monthly_price_krw === 0 ? "가입 시 바로 사용" : "월 결제"}
        </p>
      </div>

      <dl className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">월 생성</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {formatSongs(plan.monthly_generation_limit)}
          </dd>
        </div>
      </dl>

      <ul className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">MP3 다운로드</span>
          <Mark included={plan.download_mp3} />
        </li>
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">WAV 다운로드</span>
          <Mark included={plan.download_wav} />
        </li>
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">상업적 이용</span>
          <Mark included={plan.commercial_use} />
        </li>
      </ul>

      {/*
        The CTA appears only when the *server* says checkout is available.
        The flag comes from `/v1/plans`, so a deployment without PayApp
        credentials renders an honest unavailable state rather than a
        button that opens nothing.

        Changing plan is not offered. Doing it safely means either
        proration or two live recurring contracts, and the second is how
        people get billed twice — so V1 asks the user to cancel first and
        says exactly that. Payment correctness over convenience.
      */}
      <div className="mt-auto">{planAction({ plan, current, checkoutAvailable, subscribed, onSubscribe })}</div>
    </Card>
  );
}

export default function PlansPage() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [checkoutAvailable, setCheckoutAvailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [billing, setBilling] = useState<BillingStatus | null>(null);
  const [checkoutPlan, setCheckoutPlan] = useState<Plan | null>(null);
  const { entitlement } = useEntitlement();
  const { status: authStatus } = useAuth();
  const router = useRouter();

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const catalogue = await fetchPlans(controller.signal);
        setPlans(catalogue.plans);
        setCheckoutAvailable(catalogue.checkout_available);
      } catch {
        // An abort is not a failure. React's development double-effect
        // tears the first attempt down immediately, and treating that
        // rejection as an error would leave the page showing "불러오지
        // 못했습니다" over a catalogue that loaded fine on the retry.
        if (controller.signal.aborted) return;
        setFailed(true);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, []);

  // Read separately from the catalogue: pricing is public and this is
  // not. A signed-out visitor sees the tiers and no subscription state.
  useEffect(() => {
    if (authStatus !== "authenticated") return;
    const controller = new AbortController();
    void (async () => {
      try {
        setBilling(await fetchBillingStatus(controller.signal));
      } catch {
        // Leaves `billing` null, which renders as "not subscribed" —
        // the CTA then leads to a checkout the server will refuse if it
        // is wrong. Failing towards the server's judgement, not ours.
      }
    })();
    return () => controller.abort();
  }, [authStatus]);

  function startCheckout(plan: Plan) {
    if (authStatus !== "authenticated") {
      // Preserve the intent through login. `next` is a path on this
      // origin only — `redirect.ts` refuses anything else — so it cannot
      // become an open redirect.
      router.push(`/login?next=${encodeURIComponent(`/plans?plan=${plan.plan_id}`)}`);
      return;
    }
    setCheckoutPlan(plan);
  }

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          플랜
        </h1>
        <p className="max-w-prose text-sm text-[var(--text-secondary)]">
          부르다의 요금제입니다. 생성 한도는 매달 초기화되고, 실패한 생성은 한도에서 차감되지
          않습니다.
        </p>
      </header>

      {entitlement ? (
        <Card className="flex flex-col gap-3 p-5">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">현재 사용량</h2>
            <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
              {entitlement.plan.display_name}
            </span>
          </div>
          <UsageBar entitlement={entitlement} />
        </Card>
      ) : null}

      {loading ? (
        <p className="text-sm text-[var(--text-muted)]">플랜 불러오는 중…</p>
      ) : failed ? (
        <Card className="p-5">
          <p className="text-sm text-[var(--text-secondary)]">
            요금제를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.
          </p>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {plans.map((plan) => (
            <PlanCard
              key={plan.plan_id}
              plan={plan}
              current={entitlement?.plan.plan_id === plan.plan_id}
              checkoutAvailable={checkoutAvailable}
              subscribed={hasPaidAccess(billing)}
              onSubscribe={startCheckout}
            />
          ))}
        </div>
      )}

      {checkoutPlan ? (
        <CheckoutDialog plan={checkoutPlan} onClose={() => setCheckoutPlan(null)} />
      ) : null}

      <Card className="p-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">알아두실 점</h2>
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm text-[var(--text-secondary)]">
          <li>생성 한도는 완성된 곡 기준입니다. 실패한 생성은 차감되지 않습니다.</li>
          <li>플랜을 바꿔도 이번 기간의 사용량은 초기화되지 않습니다.</li>
          <li>Free 플랜은 만든 곡을 들을 수 있지만 내려받을 수는 없습니다.</li>
          <li>구독은 매달 자동으로 갱신되며, 언제든지 설정에서 해지할 수 있습니다.</li>
          <li>해지해도 이미 결제한 기간이 끝날 때까지는 그대로 이용할 수 있습니다.</li>
          <li>결제는 PayApp에서 처리되며, 카드 정보는 부르다에 저장되지 않습니다.</li>
        </ul>
      </Card>
    </div>
  );
}
