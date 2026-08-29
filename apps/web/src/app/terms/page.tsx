/**
 * BOORDA 이용약관.
 *
 * The clause that matters most is 제9조. It is tempting to write
 * "생성된 음악의 저작권은 이용자에게 있습니다" — it reads well and it is
 * what a customer wants to hear. It is also a claim about statutory
 * authorship that no service can make on a user's behalf: whether an
 * AI-assisted work attracts copyright, and to whom, depends on human
 * creative contribution and on jurisdiction.
 *
 * So the terms separate two things that are genuinely separate: the
 * permission BOORDA grants (contractual, ours to give) from copyright
 * subsistence and ownership (statutory, not ours to decide). The
 * commercial-use scope is drawn from the plan catalogue's
 * `commercial_use` flag, which is the only place the product actually
 * records it.
 */

import type { Metadata } from "next";

import { Bullets, LegalPage, Section } from "@/components/legal/LegalPage";
import { BUSINESS, EFFECTIVE_DATE, SERVICE_NAME } from "@/lib/legal";

export const metadata: Metadata = {
  title: "BOORDA 이용약관",
  description:
    "BOORDA 서비스 이용에 관한 계약 조건, 계정, 구독, 생성 결과물의 이용 범위 및 책임 범위를 안내합니다.",
};

export default function TermsPage() {
  return (
    <LegalPage
      title="BOORDA 이용약관"
      effective={EFFECTIVE_DATE}
      intro={
        <>
          이 약관은 {BUSINESS.name}(이하 &ldquo;회사&rdquo;)이 제공하는 {SERVICE_NAME}{" "}
          서비스의 이용 조건을 정합니다.
        </>
      }
    >
      <Section id="purpose" title="제1조 (목적)">
        <p>
          이 약관은 회사가 제공하는 AI 음악 생성 서비스 {SERVICE_NAME}(이하
          &ldquo;서비스&rdquo;)의 이용과 관련하여 회사와 이용자의 권리·의무 및 책임 사항을
          정하는 것을 목적으로 합니다.
        </p>
      </Section>

      <Section id="definitions" title="제2조 (정의)">
        <Bullets
          items={[
            "&ldquo;서비스&rdquo;란 회사가 제공하는 AI 음악 생성 및 관련 기능 전체를 말합니다.",
            "&ldquo;이용자&rdquo;란 이 약관에 따라 서비스를 이용하는 자를 말합니다.",
            "&ldquo;생성 결과물&rdquo;이란 이용자의 입력에 따라 서비스가 생성한 음원 및 부수 정보를 말합니다.",
            "&ldquo;구독&rdquo;이란 월 단위로 자동 결제되는 유료 이용 계약을 말합니다.",
          ]}
        />
      </Section>

      <Section id="account" title="제3조 (계정)">
        <Bullets
          items={[
            "이용자는 이메일 주소와 비밀번호로 계정을 생성합니다.",
            "이용자는 계정 정보를 정확하게 입력해야 하며, 변경 시 서비스 내에서 수정할 수 있습니다.",
            "계정과 비밀번호의 관리 책임은 이용자에게 있으며, 제3자에게 이용하게 해서는 안 됩니다.",
            "계정 도용이 의심되는 경우 즉시 회사에 알리고 안내에 따라야 합니다.",
          ]}
        />
      </Section>

      <Section id="service" title="제4조 (서비스의 제공)">
        <Bullets
          items={[
            "회사는 연중무휴 서비스를 제공하기 위해 노력합니다.",
            "설비 점검, 장애, 천재지변 등 부득이한 사유가 있는 경우 서비스 제공이 일시적으로 중단될 수 있습니다.",
            "AI 생성 결과물의 품질은 입력 내용과 모델 특성에 따라 달라질 수 있으며, 회사는 특정한 결과나 품질을 보장하지 않습니다.",
            "동일한 입력에 대해 항상 동일한 결과가 생성되지는 않습니다.",
          ]}
        />
      </Section>

      <Section id="plans" title="제5조 (요금제와 이용량)">
        <p>
          서비스는 무료 요금제와 유료 구독 요금제로 구성되며, 각 요금제의 가격과 월 생성
          가능 곡 수 등 제공 범위는 서비스 내 요금제 화면에 안내된 내용을 따릅니다.
        </p>
        <p>
          결제, 정기결제, 해지 및 환불에 관한 사항은 별도의{" "}
          <a className="underline underline-offset-2" href="/refund-policy">
            구독·결제·환불 정책
          </a>
          을 따릅니다.
        </p>
      </Section>

      <Section id="prohibited" title="제6조 (금지 행위)">
        <p>이용자는 다음 행위를 해서는 안 됩니다.</p>
        <Bullets
          items={[
            "타인의 권리를 침해하는 내용을 입력하거나 그러한 결과물을 생성·유통하는 행위",
            "법령을 위반하거나 타인의 명예를 훼손하는 내용을 생성·유통하는 행위",
            "서비스의 정상적인 운영을 방해하는 행위, 자동화된 방법으로 비정상적인 요청을 발생시키는 행위",
            "서비스에 대한 무단 접근, 역설계 또는 보안 조치를 우회하려는 행위",
            "계정을 타인에게 판매·대여·양도하는 행위",
          ]}
        />
      </Section>

      <Section id="user-content" title="제7조 (이용자가 입력한 내용)">
        <Bullets
          items={[
            "이용자는 자신이 입력한 프롬프트, 가사, 참조 음원 등에 대해 필요한 권리를 보유하고 있음을 확인합니다.",
            "타인의 권리를 침해하는 자료를 입력하여 발생하는 문제에 대한 책임은 이용자에게 있습니다.",
            "회사는 서비스 제공에 필요한 범위에서 입력 내용을 처리합니다.",
          ]}
        />
      </Section>

      <Section id="generated" title="제8조 (생성 결과물의 이용)">
        <p>
          회사는 이용자가 자신의 요금제에서 허용하는 범위 내에서 생성 결과물을 이용할 수
          있도록 <strong>계약상 이용 권한</strong>을 부여합니다. 상업적 이용 가능 여부는
          요금제에 따라 다르며, 서비스 내 요금제 화면에 표시된 범위를 따릅니다.
        </p>
        <p>
          다만 AI가 생성한 결과물에 저작권이 발생하는지, 발생한다면 누구에게 귀속되는지는
          이용자의 창작적 기여 정도와 각 국가의 법률에 따라 달라질 수 있으며, 회사는 이에
          대해 특정한 법적 결론을 보장하지 않습니다. 위 이용 권한은 회사가 계약으로 부여하는
          권한이며, 저작권의 성립 또는 귀속을 확정하는 것이 아닙니다.
        </p>
        <p className="text-xs text-[var(--text-muted)]">
          생성 결과물을 상업적으로 이용하려는 경우, 이용 목적과 지역의 법률을 확인하시기
          바랍니다.
        </p>
      </Section>

      <Section id="ip" title="제9조 (서비스에 대한 권리)">
        <p>
          서비스 자체와 이를 구성하는 소프트웨어, 디자인, 상표 등에 대한 권리는 회사에
          있습니다. 이 약관은 이용자에게 서비스 자체에 대한 권리를 이전하지 않습니다.
        </p>
      </Section>

      <Section id="changes" title="제10조 (서비스의 변경 및 중단)">
        <p>
          회사는 서비스의 내용을 변경하거나 일부 기능의 제공을 중단할 수 있습니다. 이용자에게
          불리한 중대한 변경은 사전에 서비스 내에 공지합니다.
        </p>
      </Section>

      <Section id="termination" title="제11조 (이용 제한 및 계약 해지)">
        <Bullets
          items={[
            "이용자는 언제든지 설정 화면에서 회원 탈퇴를 할 수 있습니다.",
            "진행 중인 유료 구독이 있는 경우, 결제가 계속되는 것을 막기 위해 구독을 먼저 해지해야 탈퇴할 수 있습니다.",
            "회원 탈퇴 시 계정은 복구할 수 없도록 익명 처리되며, 법령상 보존이 필요한 기록은 해당 기간 동안 보존됩니다.",
            "이용자가 제6조를 위반하는 경우 회사는 서비스 이용을 제한하거나 계약을 해지할 수 있습니다.",
          ]}
        />
        <p className="text-xs text-[var(--text-muted)]">
          탈퇴 시 처리 방식의 자세한 내용은{" "}
          <a className="underline underline-offset-2" href="/privacy">
            개인정보처리방침
          </a>
          을 참고하시기 바랍니다.
        </p>
      </Section>

      <Section id="liability" title="제12조 (책임의 범위)">
        <p>
          회사는 관계 법령이 허용하는 범위에서 책임을 부담합니다. 회사의 고의 또는 중대한
          과실로 인한 손해에 대한 책임은 제한되지 않습니다.
        </p>
        <p>
          회사는 천재지변, 이용자의 귀책사유, 제3자의 행위 등 회사의 통제를 벗어난 사유로
          발생한 손해에 대해서는 책임을 지지 않습니다.
        </p>
      </Section>

      <Section id="notice" title="제13조 (통지)">
        <p>
          회사는 이용자에게 서비스 내 공지 또는 이용자가 등록한 이메일 주소로 통지할 수
          있습니다.
        </p>
      </Section>

      <Section id="governing" title="제14조 (준거법 및 분쟁 해결)">
        <p>
          이 약관은 대한민국 법률에 따라 해석되며, 서비스 이용과 관련하여 분쟁이 발생한
          경우 회사와 이용자는 원만한 해결을 위해 성실히 협의합니다. 협의가 이루어지지 않는
          경우 관계 법령이 정한 절차에 따릅니다.
        </p>
      </Section>

      <Section id="contact" title="제15조 (문의)">
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
