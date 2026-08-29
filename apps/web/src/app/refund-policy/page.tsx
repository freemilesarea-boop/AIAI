/**
 * 구독·결제·환불 정책.
 *
 * Describes what the billing code actually does, verified against
 * `billing_repository` and `luber_billing.states`: cancellation keeps
 * access until the paid period ends, a failed renewal moves the
 * subscription to PAST_DUE rather than cutting access instantly, and
 * BOORDA never sees a card number.
 *
 * What it deliberately does not say is "환불 불가". BOORDA has no refund
 * mechanism in the product, and a blanket denial of a statutory right
 * would be both unenforceable and untrue. The page states the process
 * that exists — contact support — and the operator decision that is
 * still outstanding is reported rather than invented.
 *
 * Plan figures are fetched from `/v1/plans`, the same source the pricing
 * page uses. A second hardcoded table here would drift the first time a
 * price changed, and the page that drifted would be the legal one.
 */

"use client";

import { useEffect, useState } from "react";

import { Bullets, DataTable, LegalPage, Section } from "@/components/legal/LegalPage";
import { BUSINESS, EFFECTIVE_DATE, SERVICE_NAME } from "@/lib/legal";
import { fetchPlans, formatPriceKrw, formatSongs, type Plan } from "@/lib/plans";

export default function RefundPolicyPage() {
  const [plans, setPlans] = useState<Plan[] | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setPlans((await fetchPlans(controller.signal)).plans);
      } catch {
        // The policy text stands without the table; a failed fetch must
        // not blank a legal page.
        if (!controller.signal.aborted) setPlans([]);
      }
    })();
    return () => controller.abort();
  }, []);

  return (
    <LegalPage
      title="구독·결제·환불 정책"
      effective={EFFECTIVE_DATE}
      intro={
        <>
          {SERVICE_NAME}의 유료 구독, 정기결제, 해지 및 환불 처리에 관한 안내입니다. 이
          정책은 실제 서비스 동작을 기준으로 작성되었습니다.
        </>
      }
    >
      <Section id="plans" title="1. 요금제">
        <p>
          {SERVICE_NAME}은 무료 요금제와 유료 구독 요금제를 제공합니다. 각 요금제의 가격과
          제공량은 아래와 같으며, 서비스 내 요금제 화면과 동일한 정보입니다.
        </p>
        {plans && plans.length > 0 ? (
          <DataTable
            caption="요금제"
            headers={["요금제", "월 이용료", "월 생성 가능 곡", "다운로드"]}
            rows={plans.map((plan) => [
              plan.display_name,
              formatPriceKrw(plan.monthly_price_krw),
              formatSongs(plan.monthly_generation_limit),
              plan.download_mp3 || plan.download_wav ? "제공" : "미제공",
            ])}
          />
        ) : (
          <p className="text-xs text-[var(--text-muted)]">
            요금제 정보는 요금제 화면에서 확인하실 수 있습니다.
          </p>
        )}
        <p className="text-xs text-[var(--text-muted)]">
          표시 금액은 부가세 포함 여부를 포함하여 결제 화면에 안내된 금액을 기준으로 합니다.
        </p>
      </Section>

      <Section id="billing" title="2. 정기결제">
        <Bullets
          items={[
            "유료 구독은 월 단위 정기결제입니다. 최초 결제일을 기준으로 매월 같은 주기로 자동 결제됩니다.",
            "결제는 결제대행사(PayApp)를 통해 처리되며, 회사는 카드번호·유효기간·CVC 등 카드 정보를 저장하지 않습니다.",
            "정기결제 등록 시 결제대행사 요구에 따라 휴대전화번호를 입력받습니다.",
            "결제가 정상적으로 완료되면 해당 결제 주기 동안 요금제의 기능을 이용할 수 있습니다.",
          ]}
        />
      </Section>

      <Section id="cancellation" title="3. 해지">
        <Bullets
          items={[
            "구독은 설정 화면에서 언제든지 해지할 수 있습니다.",
            "해지하더라도 이미 결제한 이용 기간이 끝날 때까지는 기능을 계속 이용할 수 있습니다. 해지 즉시 이용이 중단되지 않습니다.",
            "해지 이후에는 다음 결제가 이루어지지 않습니다.",
            "이용 기간이 끝나면 무료 요금제의 이용 범위로 전환됩니다.",
          ]}
        />
      </Section>

      <Section id="failure" title="4. 결제 실패">
        <p>
          정기결제가 실패한 경우 구독은 즉시 종료되지 않고 결제 확인이 필요한 상태로
          전환됩니다. 결제 수단을 확인하여 결제가 완료되면 이용이 계속되며, 일정 기간 내에
          결제가 확인되지 않으면 유료 기능 이용이 중단될 수 있습니다.
        </p>
      </Section>

      <Section id="refund" title="5. 환불">
        <p>
          환불은 「전자상거래 등에서의 소비자보호에 관한 법률」 등 관계 법령에 따릅니다.
          회사는 법령상 인정되는 청약철회 및 환불 요청을 부당하게 제한하지 않습니다.
        </p>
        <p>
          다만 디지털 콘텐츠의 특성상, 이용자가 이미 제공받아 사용한 부분에 대해서는 법령이
          정하는 범위에서 청약철회가 제한될 수 있습니다. 구체적인 처리 여부와 범위는 이용
          내역과 요청 사유에 따라 개별적으로 판단합니다.
        </p>
        <p>
          환불을 요청하시려면 고객지원을 통해 문의해 주시기 바랍니다. 현재 환불은 서비스
          화면에서 자동으로 처리되지 않으며, 문의 접수 후 개별적으로 안내합니다.
        </p>
        <Bullets
          items={[
            <>
              문의:{" "}
              <a className="underline underline-offset-2" href="/support/contact">
                고객지원
              </a>{" "}
              또는{" "}
              <a
                className="underline underline-offset-2"
                href={`mailto:${BUSINESS.contactEmail}`}
              >
                {BUSINESS.contactEmail}
              </a>
            </>,
            "요청 시 결제 일시와 요금제를 함께 알려주시면 확인이 빠릅니다.",
          ]}
        />
      </Section>

      <Section id="history" title="6. 결제 내역 확인">
        <p>
          결제 내역과 다음 결제 예정일은 설정 화면에서 확인할 수 있습니다. 회원 탈퇴 후에도
          결제 기록은 관계 법령이 정한 기간 동안 보존됩니다.
        </p>
      </Section>

      <Section id="contact" title="7. 문의">
        <Bullets
          items={[
            <>상호명: {BUSINESS.name}</>,
            <>대표자: {BUSINESS.representative}</>,
            <>사업자등록번호: {BUSINESS.registrationNumber}</>,
            <>
              고객문의:{" "}
              <a
                className="underline underline-offset-2"
                href={`mailto:${BUSINESS.contactEmail}`}
              >
                {BUSINESS.contactEmail}
              </a>
            </>,
          ]}
        />
      </Section>
    </LegalPage>
  );
}
