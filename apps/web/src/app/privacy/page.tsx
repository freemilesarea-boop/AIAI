/**
 * 개인정보처리방침.
 *
 * Written from the audited schema, not from a template. Every item
 * listed is a column that exists; nothing is claimed that the code does
 * not do. The two statements most worth getting right, because a
 * generic policy gets both wrong:
 *
 * - BOORDA never receives a card number. PayApp holds it; the billing
 *   layer's `SENSITIVE_KEYS` drops card fields before anything is
 *   written, so the honest claim is "we do not store it" rather than
 *   "we store it securely".
 * - Closing an account anonymises rather than deletes, because commerce
 *   records must survive it. Saying "all data is deleted immediately"
 *   would be false, and this page says what actually happens.
 */

import type { Metadata } from "next";

import { Bullets, DataTable, LegalPage, Section } from "@/components/legal/LegalPage";
import {
  ACQUISITION_RETENTION_MONTHS,
  BUSINESS,
  COOKIES,
  EFFECTIVE_DATE,
  PROCESSORS,
  SERVICE_NAME,
  STATUTORY_RETENTION,
} from "@/lib/legal";

export const metadata: Metadata = {
  title: "개인정보처리방침 | BOORDA",
  description:
    "BOORDA가 수집하는 개인정보 항목, 이용 목적, 보유 기간, 처리 위탁 및 이용자 권리를 안내합니다.",
};

export default function PrivacyPage() {
  return (
    <LegalPage
      title="개인정보처리방침"
      effective={EFFECTIVE_DATE}
      intro={
        <>
          {BUSINESS.name}(이하 &ldquo;회사&rdquo;)은 {SERVICE_NAME} 서비스를 제공하면서 이용자의
          개인정보를 소중히 다루며, 「개인정보 보호법」 등 관계 법령을 준수합니다. 이
          방침은 회사가 실제로 처리하는 정보만을 기준으로 작성되었습니다.
        </>
      }
    >
      <Section id="purpose" title="1. 개인정보의 처리 목적">
        <Bullets
          items={[
            "회원 가입 및 계정 관리, 본인 확인, 로그인 상태 유지",
            "AI 음악 생성 서비스 제공 및 이용 이력 관리",
            "유료 구독의 결제·정기결제·해지 처리 및 결제 내역 확인",
            "고객 문의 접수 및 처리",
            "서비스 이용 현황과 유입 경로 분석, 마케팅 효과 측정 및 서비스 개선",
            "부정 이용 방지 및 요청 빈도 제한 등 서비스 안정성 확보",
          ]}
        />
      </Section>

      <Section id="items" title="2. 처리하는 개인정보의 항목">
        <p>회사는 아래 항목을 처리합니다. 이용자가 직접 입력하지 않은 항목은 표기했습니다.</p>
        <DataTable
          caption="처리하는 개인정보 항목"
          headers={["구분", "항목", "수집 방법"]}
          rows={[
            ["계정", "이메일 주소, 비밀번호(암호화 저장), 표시 이름(선택), 가입 일시", "회원 가입 시 이용자 입력"],
            ["인증", "세션 식별자(쿠키)", "로그인 시 자동 생성"],
            [
              "결제",
              "요금제, 결제 금액, 결제 상태, 결제·정기결제 식별번호, 결제 일시, 휴대전화번호",
              "구독 신청 시 이용자 입력 및 결제사 통지",
            ],
            ["문의", "문의 유형, 제목, 내용, 발생 위치(선택)", "문의 접수 시 이용자 입력"],
            [
              "서비스 이용",
              "생성 요청 내용(제목·프롬프트·가사 등), 생성 결과, 다운로드 기록",
              "서비스 이용 과정에서 생성",
            ],
            [
              "유입 분석",
              "익명 방문자 식별자, 유입 경로(source·medium·campaign·content·term), 광고 클릭 식별정보(gclid·gbraid·wbraid·fbclid), 유입 도메인, 방문 경로, 최초·최종 방문 일시",
              "서비스 접속 시 자동 수집",
            ],
            ["서비스 안정성", "접속 IP 주소", "요청 빈도 제한을 위해 일시적으로만 처리"],
          ]}
        />
        <p className="text-xs text-[var(--text-muted)]">
          회사는 신용카드 번호, 유효기간, CVC 등 카드 정보를 저장하지 않습니다. 해당 정보는
          결제대행사가 처리하며, 회사 시스템에는 기록되지 않습니다. 주민등록번호 또한 수집하지
          않습니다.
        </p>
      </Section>

      <Section id="auto" title="3. 자동으로 수집되는 정보와 쿠키">
        <p>
          회사는 아래 자사(first-party) 쿠키를 사용합니다. 광고 목적의 제3자 추적 픽셀은 현재
          설치되어 있지 않습니다.
        </p>
        <DataTable
          caption="사용하는 쿠키"
          headers={["이름", "목적", "보관 기간", "속성"]}
          rows={COOKIES.map((cookie) => [
            cookie.name,
            cookie.purpose,
            cookie.lifetime,
            cookie.attributes,
          ])}
        />
        <p>
          이용자는 웹 브라우저 설정에서 쿠키 저장을 거부하거나 저장된 쿠키를 삭제할 수 있습니다.
          다만 로그인 유지에 사용되는 쿠키를 거부하면 로그인이 필요한 기능을 이용할 수 없습니다.
        </p>
        <p>
          유입 분석 목적의 식별자는 회사가 임의로 생성한 무작위 값이며, 브라우저 지문(fingerprint)
          이나 IP 주소로부터 생성되지 않습니다. 또한 다른 사업자의 웹사이트에서는 사용되지
          않습니다.
        </p>
      </Section>

      <Section id="analytics" title="4. 유입 경로 분석">
        <p>
          회사는 서비스 개선과 마케팅 효과 측정을 위해 이용자가 어떤 경로로 서비스에
          접속했는지를 자사 분석 기능으로 수집합니다. 수집 항목은 위 2항의 &ldquo;유입
          분석&rdquo;과 같습니다.
        </p>
        <Bullets
          items={[
            "브라우저 지문(fingerprint)을 수집하거나 생성하지 않습니다.",
            "IP 주소로부터 이용자 식별자를 만들지 않습니다.",
            "현재 제3자 광고 픽셀(Meta Pixel, Google Analytics 등)을 설치하고 있지 않습니다.",
            "관리자 화면 등 서비스 운영 목적의 접속은 분석에서 제외합니다.",
          ]}
        />
        <p className="text-xs text-[var(--text-muted)]">
          위 내용은 현재 시점의 처리 방식이며, 향후 변경 시 이 방침을 개정하여 공지합니다.
        </p>
      </Section>

      <Section id="retention" title="5. 개인정보의 보유 및 이용 기간">
        <p>
          회사는 원칙적으로 회원 탈퇴 시 개인정보를 지체 없이 파기하거나 복원할 수 없도록
          익명 처리합니다. 다만 아래의 경우 예외적으로 보존합니다.
        </p>
        <p className="font-medium text-[var(--text-primary)]">유입 분석 정보</p>
        <p>
          원본 유입 분석 기록(익명 방문자 및 방문 기록)은 수집일로부터{" "}
          {ACQUISITION_RETENTION_MONTHS}개월간 보관한 뒤 삭제합니다. 개인을 식별하거나 특정
          방문자와 다시 연결할 수 없는 통계 형태의 정보는 보관할 수 있습니다.
        </p>
        <p className="font-medium text-[var(--text-primary)]">법령에 따른 보존</p>
        <p>
          「전자상거래 등에서의 소비자보호에 관한 법률」 등 관계 법령이 보존을 요구하는 거래
          기록은 해당 법령이 정한 기간 동안 보존하며, 위 {ACQUISITION_RETENTION_MONTHS}개월
          삭제 대상에서 제외됩니다.
        </p>
        <DataTable
          caption="법령에 따른 보존"
          headers={["보존 대상", "보존 기간", "근거"]}
          rows={STATUTORY_RETENTION.map((row) => [row.category, row.period, row.basis])}
        />
      </Section>

      <Section id="destruction" title="6. 개인정보의 파기 절차 및 방법">
        <p>
          보유 기간이 지나거나 처리 목적이 달성된 개인정보는 지체 없이 파기합니다. 전자적
          파일 형태의 정보는 복구할 수 없는 방법으로 삭제하며, 법령에 따라 보존해야 하는
          정보는 다른 정보와 분리하여 보존합니다.
        </p>
        <p>
          유입 분석 정보의 삭제는 보관 기간이 지난 기록을 정기적으로 삭제하는 절차를 통해
          이루어집니다.
        </p>
      </Section>

      <Section id="withdrawal" title="7. 회원 탈퇴 시 처리">
        <p>
          회원 탈퇴 시 회사는 계정을 삭제하는 대신 <strong>익명 처리</strong>합니다. 이는
          결제 기록 등 법령상 보존이 필요한 정보를 함께 삭제하지 않기 위한 조치입니다.
        </p>
        <DataTable
          caption="회원 탈퇴 시 처리"
          headers={["구분", "처리 방식"]}
          rows={[
            ["이메일 주소", "복구할 수 없는 값으로 대체되어 더 이상 이용자를 식별하지 않습니다."],
            ["표시 이름", "삭제됩니다."],
            ["비밀번호", "삭제되며, 해당 계정으로는 로그인할 수 없습니다."],
            ["로그인 세션", "모두 즉시 만료됩니다."],
            ["유입 분석 연결", "방문자 식별자와 계정의 연결이 해제됩니다."],
            [
              "결제·구독 기록",
              "법령상 보존 기간 동안 보존됩니다. 다만 위와 같이 익명 처리된 계정에 연결됩니다.",
            ],
            ["문의 기록", "법령상 보존 기간 동안 보존됩니다."],
          ]}
        />
        <p className="text-xs text-[var(--text-muted)]">
          진행 중인 유료 구독이 있는 경우, 결제가 계속되는 것을 방지하기 위해 구독을 먼저
          해지해야 탈퇴할 수 있습니다.
        </p>
      </Section>

      <Section id="third-party" title="8. 개인정보의 제3자 제공">
        <p>
          회사는 이용자의 개인정보를 제3자에게 제공하지 않습니다. 다만 법령에 따라 요구되는
          경우에는 관계 법령이 정한 절차에 따릅니다.
        </p>
      </Section>

      <Section id="processors" title="9. 개인정보 처리업무의 위탁">
        <p>회사는 서비스 제공을 위해 아래와 같이 처리 업무를 위탁하고 있습니다.</p>
        <DataTable
          caption="처리 위탁 현황"
          headers={["수탁자", "위탁 업무", "처리 정보", "처리 지역"]}
          rows={PROCESSORS.map((p) => [p.name, p.purpose, p.data, p.region])}
        />
        <p className="text-xs text-[var(--text-muted)]">
          위 수탁자 중 일부는 대한민국 외의 지역에서 정보를 처리합니다. 회사는 위탁 계약 시
          개인정보가 안전하게 관리되도록 필요한 사항을 규정하고 있습니다.
        </p>
      </Section>

      <Section id="rights" title="10. 이용자의 권리와 행사 방법">
        <p>
          이용자는 언제든지 자신의 개인정보에 대해 열람, 정정, 삭제, 처리정지를 요구할 수
          있습니다. 일부 항목은 서비스 내에서 직접 확인하고 변경할 수 있습니다.
        </p>
        <Bullets
          items={[
            "표시 이름 및 비밀번호 변경: 설정 화면",
            "결제 및 구독 내역 확인, 구독 해지: 설정 화면",
            "회원 탈퇴: 설정 화면",
            <>그 밖의 요청: 고객지원 또는 {BUSINESS.contactEmail}</>,
          ]}
        />
        <p>
          법령에 따라 보존이 요구되는 정보는 삭제 요청의 대상에서 제외될 수 있으며, 이 경우
          그 사유를 안내합니다.
        </p>
      </Section>

      <Section id="security" title="11. 개인정보의 안전성 확보 조치">
        <p>회사는 아래와 같은 조치를 실제로 적용하고 있습니다.</p>
        <Bullets
          items={[
            "비밀번호는 복호화할 수 없는 방식(Argon2id)으로 해시하여 저장합니다.",
            "로그인 세션 식별자는 원본이 아닌 해시 형태로 저장합니다.",
            "인증 및 유입 분석 쿠키는 스크립트가 읽을 수 없도록 HttpOnly로 설정하며, 운영 환경에서는 Secure 속성을 적용합니다.",
            "서비스 통신은 HTTPS로 암호화합니다.",
            "로그인·문의 등 주요 요청에 빈도 제한을 적용합니다.",
            "운영 관리 기능은 권한이 부여된 계정만 접근할 수 있으며, 권한 변경 이력을 기록합니다.",
            "결제 카드 정보는 회사가 보관하지 않습니다.",
          ]}
        />
      </Section>

      <Section id="contact" title="12. 문의처">
        <p>
          개인정보 처리에 관한 문의, 열람·정정·삭제 요청은 아래로 연락해 주시기 바랍니다.
        </p>
        <Bullets
          items={[
            <>상호명: {BUSINESS.name}</>,
            <>대표자: {BUSINESS.representative}</>,
            <>사업자등록번호: {BUSINESS.registrationNumber}</>,
            <>
              이메일:{" "}
              <a
                className="underline underline-offset-2"
                href={`mailto:${BUSINESS.contactEmail}`}
              >
                {BUSINESS.contactEmail}
              </a>
            </>,
          ]}
        />
        <p className="text-xs text-[var(--text-muted)]">
          개인정보 침해에 관한 상담이 필요한 경우 개인정보침해신고센터(privacy.kisa.or.kr,
          국번없이 118) 등에 문의하실 수 있습니다.
        </p>
      </Section>

      <Section id="changes" title="13. 방침의 변경">
        <p>
          이 방침의 내용이 변경되는 경우, 변경 사항과 시행일을 이 페이지에 게시하여
          공지합니다. 이용자에게 중대한 영향을 미치는 변경은 시행일 전에 안내합니다.
        </p>
      </Section>
    </LegalPage>
  );
}
