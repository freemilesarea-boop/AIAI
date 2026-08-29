/**
 * Business identity and legal metadata, in one place.
 *
 * Every legal page and the footer read from here, so a corrected
 * registration number is one edit rather than four, and no page can
 * quietly disagree with another about who operates BOORDA.
 *
 * **Unverified fields are not published.** Korean e-commerce law
 * requires a 통신판매업자 to display an address, a telephone number and
 * a 통신판메업 신고번호. BOORDA has none of these configured anywhere in
 * the repository or its deployment, and inventing them would put false
 * statements on a page whose entire purpose is being true. They are
 * present below as `null`, the renderers skip nulls, and the gap is
 * reported rather than filled.
 */

export interface BusinessInfo {
  /** Trading name. */
  name: string;
  /** Representative. */
  representative: string;
  /** 사업자등록번호. */
  registrationNumber: string;
  /** Where customers reach a human. */
  contactEmail: string;
  /**
   * Fields required for 통신판매 but not yet confirmed. Rendered only
   * when non-null — see the module docstring.
   */
  mailOrderNumber: string | null;
  address: string | null;
  phone: string | null;
  /**
   * 개인정보 보호책임자. Null until designated: naming someone who has
   * not accepted the role is worse than leaving it to the contact
   * address below.
   */
  privacyOfficer: string | null;
}

/**
 * Supplied by the operator for this publication.
 *
 * Not independently confirmed from the repository or deployment
 * configuration — nothing in either carried business identity before
 * this file existed. Recorded here so a later correction has one home.
 */
export const BUSINESS: BusinessInfo = {
  name: "로베르 콘텐츠 스튜디오",
  representative: "이승현",
  registrationNumber: "234-52-00922",
  contactEmail: "freemilesarea@gmail.com",
  mailOrderNumber: null,
  address: null,
  phone: null,
  privacyOfficer: null,
};

/** The service's own name, and the copyright line's subject. */
export const SERVICE_NAME = "BOORDA";

/**
 * When these documents take effect.
 *
 * One date for all three, because they were published together. A later
 * revision adds an entry to `HISTORY` rather than overwriting this.
 */
export const EFFECTIVE_DATE = "2026년 8월 29일";

/**
 * Previous versions, newest first. Empty because there are none — this
 * is the first publication, and inventing a revision history would be
 * inventing a past.
 */
export const HISTORY: { version: string; effective: string; note: string }[] = [];

/**
 * How long raw first-party acquisition records are kept.
 *
 * Enforced by `scripts/ops/purge_acquisition.py`, not merely written
 * here — a retention period a product cannot perform is a promise, not
 * a policy.
 */
export const ACQUISITION_RETENTION_MONTHS = 12;

/**
 * Transaction records that outlive the analytics period.
 *
 * These come from 전자상거래 등에서의 소비자보호에 관한 법률. The Act was
 * amended on 2026-01-20 (법률 제21312호, effective 2026-07-21) and the
 * exact periods below were supplied by the operator; they could not be
 * confirmed against the primary statute text while writing this, so
 * they are marked unverified and must be checked before anyone relies
 * on them. The *principle* — that these records are excluded from the
 * 12-month analytics deletion — is a fact about the implementation and
 * is not in doubt.
 */
export const STATUTORY_RETENTION: {
  category: string;
  period: string;
  basis: string;
  verified: boolean;
}[] = [
  {
    category: "계약 또는 청약철회 등에 관한 기록",
    period: "5년",
    basis: "전자상거래 등에서의 소비자보호에 관한 법률",
    verified: false,
  },
  {
    category: "대금결제 및 재화 등의 공급에 관한 기록",
    period: "5년",
    basis: "전자상거래 등에서의 소비자보호에 관한 법률",
    verified: false,
  },
  {
    category: "소비자의 불만 또는 분쟁처리에 관한 기록",
    period: "3년",
    basis: "전자상거래 등에서의 소비자보호에 관한 법률",
    verified: false,
  },
  {
    category: "표시·광고에 관한 기록",
    period: "6개월",
    basis: "전자상거래 등에서의 소비자보호에 관한 법률",
    verified: false,
  },
];

/**
 * The cookies BOORDA sets, exactly as the server sets them.
 *
 * Read off `session.py` and `routes/acquisition.py`; a live response was
 * checked against this before publication. Both are first-party and
 * neither is readable by scripts.
 */
export const COOKIES: {
  name: string;
  purpose: string;
  lifetime: string;
  attributes: string;
}[] = [
  {
    name: "luber_session",
    purpose: "로그인 상태 유지",
    lifetime: "로그아웃 시 또는 세션 만료 시까지",
    attributes: "자사 쿠키 · HttpOnly · SameSite=Lax · 운영 환경에서 Secure",
  },
  {
    name: "boorda_visitor",
    purpose: "유입 경로 분석을 위한 익명 방문자 구분",
    lifetime: "400일",
    attributes: "자사 쿠키 · HttpOnly · SameSite=Lax · 운영 환경에서 Secure",
  },
];

/**
 * External services that handle personal data, from actual configuration.
 *
 * Audited from the deployment rather than assumed: an email provider is
 * *not* listed because BOORDA has none configured, and object storage is
 * absent because `STORAGE_PROVIDER=local`. Listing a processor BOORDA
 * does not use would be as wrong as omitting one it does.
 */
export const PROCESSORS: {
  name: string;
  purpose: string;
  data: string;
  region: string;
}[] = [
  {
    name: "PayApp (주식회사 페이앱)",
    purpose: "결제 및 정기결제 처리",
    data: "결제 요청 정보, 휴대전화번호, 결제 결과",
    region: "대한민국",
  },
  {
    name: "Neon",
    purpose: "데이터베이스 호스팅",
    data: "계정·결제·문의·유입 분석 정보",
    region: "싱가포르 (ap-southeast-1)",
  },
  {
    name: "Railway",
    purpose: "애플리케이션 서버 운영",
    data: "서비스 처리 과정에서 경유하는 정보",
    region: "네덜란드 암스테르담",
  },
  {
    name: "Upstash",
    purpose: "요청 빈도 제한 등 임시 처리",
    data: "접속 IP 주소 (제한 시간 동안만)",
    region: "확인 필요",
  },
  {
    name: "Vercel",
    purpose: "웹 프론트엔드 호스팅",
    data: "서비스 이용 과정의 요청 정보",
    region: "확인 필요",
  },
];
