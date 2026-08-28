/**
 * Frequently asked questions, restricted to what is actually settled.
 *
 * The hard rule here is what is *absent*. A FAQ is where a product
 * quietly invents policy: someone writes "환불은 7일 이내 가능합니다"
 * because it sounds reasonable, and it becomes the thing a customer
 * relies on and a support agent has to honour. Nothing in this file is
 * written that way.
 *
 * Every answer below is derived from something the code already
 * enforces — the plan table, the download gate, the allowance ledger,
 * the cancellation semantics PayApp documents. Where a question needs a
 * decision nobody has made, it is listed in `OPEN_POLICY_QUESTIONS`
 * rather than answered, and the support page points those at the
 * contact form.
 *
 * Refunds, copyright ownership, commercial-use scope, data retention
 * and the deletion grace period are all in that second list. They are
 * business and legal decisions, not implementation details, and
 * guessing at them here would be worse than saying nothing.
 */

export interface FaqEntry {
  id: string;
  category: FaqCategory;
  question: string;
  /** Plain text. Rendered escaped, like everything else user-facing. */
  answer: string;
}

export type FaqCategory = "ACCOUNT" | "BILLING" | "GENERATION" | "DOWNLOAD" | "OTHER";

export const FAQ_CATEGORIES: { id: FaqCategory; label: string }[] = [
  { id: "ACCOUNT", label: "계정" },
  { id: "BILLING", label: "구독 및 결제" },
  { id: "GENERATION", label: "음악 생성" },
  { id: "DOWNLOAD", label: "다운로드" },
  { id: "OTHER", label: "기타" },
];

/**
 * Answers that restate enforced behaviour.
 *
 * If one of these ever stops matching the code, the code is right and
 * this file is a bug.
 */
export const FAQ: FaqEntry[] = [
  // ── 계정 ───────────────────────────────────────────────────────────
  {
    id: "account-password",
    category: "ACCOUNT",
    question: "비밀번호를 바꾸려면 어떻게 하나요?",
    answer:
      "설정 → 보안에서 현재 비밀번호를 입력한 뒤 새 비밀번호로 변경할 수 있습니다. " +
      "비밀번호를 바꾸면 지금 사용 중인 브라우저를 제외한 모든 기기에서 로그아웃됩니다.",
  },
  {
    id: "account-email-change",
    category: "ACCOUNT",
    question: "이메일 주소를 바꿀 수 있나요?",
    answer:
      "현재는 제공하지 않습니다. 로그인에 쓰이는 주소를 바꾸려면 새 주소로 인증 메일을 보내 " +
      "확인하는 절차가 필요한데, 그 절차 없이 변경을 허용하면 오타 하나로 계정에 접근할 수 " +
      "없게 되기 때문입니다.",
  },
  {
    id: "account-delete",
    category: "ACCOUNT",
    question: "회원 탈퇴하면 어떻게 되나요?",
    answer:
      "설정 → 보안 맨 아래 Danger Zone에서 탈퇴할 수 있습니다. 탈퇴하면 로그인할 수 없게 되고 " +
      "모든 기기에서 로그아웃되며, 라이브러리와 만든 음악에 접근할 수 없게 됩니다. " +
      "구독이 진행 중이면 먼저 해지해야 합니다. 계정만 삭제되고 결제가 계속되는 상황을 막기 " +
      "위해서입니다.",
  },

  // ── 구독 및 결제 ───────────────────────────────────────────────────
  {
    id: "billing-plans",
    category: "BILLING",
    question: "플랜별 가격과 생성 한도가 어떻게 되나요?",
    answer:
      "Free는 무료로 월 20곡이며 다운로드는 제공하지 않습니다. " +
      "Basic은 월 19,900원에 200곡, Pro는 월 29,900원에 500곡, " +
      "Creator는 월 49,900원에 1,000곡입니다. 유료 플랜은 MP3·WAV 다운로드를 포함합니다.",
  },
  {
    id: "billing-failed-generation",
    category: "BILLING",
    question: "생성에 실패해도 한도가 차감되나요?",
    answer:
      "차감되지 않습니다. 한도는 완성된 곡을 기준으로 계산하며, 실패한 생성은 사용량에서 " +
      "제외됩니다.",
  },
  {
    id: "billing-cancel",
    category: "BILLING",
    question: "구독을 해지하면 바로 이용이 중단되나요?",
    answer:
      "아닙니다. 해지하면 다음 결제부터 청구되지 않으며, 이미 결제한 기간이 끝날 때까지는 " +
      "그대로 이용할 수 있습니다. 설정 → 구독에서 해지할 수 있습니다.",
  },
  {
    id: "billing-plan-change",
    category: "BILLING",
    question: "플랜을 바로 변경할 수 있나요?",
    answer:
      "현재는 제공하지 않습니다. 플랜을 바꾸려면 사용 중인 구독을 먼저 해지한 뒤 새 플랜을 " +
      "선택해 주세요. 중복 청구가 발생하지 않도록 하기 위한 조치입니다.",
  },
  {
    id: "billing-period-change",
    category: "BILLING",
    question: "플랜을 바꾸면 이번 달 사용량이 초기화되나요?",
    answer: "초기화되지 않습니다. 기간 중 플랜을 바꿔도 이번 기간의 사용량은 그대로 유지됩니다.",
  },
  {
    id: "billing-card",
    category: "BILLING",
    question: "카드 정보는 어디에 저장되나요?",
    answer:
      "부르다는 카드번호를 저장하지 않습니다. 결제는 결제사(PayApp)에서 처리되며 카드 정보도 " +
      "결제사가 보관합니다.",
  },

  // ── 음악 생성 ──────────────────────────────────────────────────────
  {
    id: "generation-availability",
    category: "GENERATION",
    question: "지금 음악 생성이 되지 않습니다.",
    answer:
      "음악 생성은 현재 준비 중이며 일시적으로 사용할 수 없습니다. 이 기간에는 생성을 " +
      "시도해도 이용권이 차감되지 않습니다.",
  },
  {
    id: "generation-counting",
    category: "GENERATION",
    question: "한 번에 두 곡을 만들면 한도가 얼마나 줄어드나요?",
    answer: "완성된 곡 수만큼 차감됩니다. 두 곡을 받으면 2곡이 사용된 것으로 계산됩니다.",
  },

  // ── 다운로드 ───────────────────────────────────────────────────────
  {
    id: "download-free",
    category: "DOWNLOAD",
    question: "Free 플랜에서 만든 곡을 내려받을 수 있나요?",
    answer:
      "Free 플랜은 다운로드를 포함하지 않습니다. 만든 곡은 라이브러리에서 그대로 들을 수 " +
      "있으며, 유료 플랜으로 변경하면 이전에 만든 곡도 내려받을 수 있습니다.",
  },
  {
    id: "download-formats",
    category: "DOWNLOAD",
    question: "어떤 형식으로 내려받을 수 있나요?",
    answer: "유료 플랜에서는 MP3와 WAV를 모두 내려받을 수 있습니다.",
  },

  // ── 기타 ───────────────────────────────────────────────────────────
  {
    id: "other-contact",
    category: "OTHER",
    question: "여기에 없는 내용을 문의하려면?",
    answer:
      "고객지원 → 문의하기에서 접수해 주세요. 접수하면 문의번호가 발급되고, 내 문의내역에서 " +
      "진행 상태를 확인할 수 있습니다.",
  },
];

/**
 * Questions customers will ask that nobody has decided the answer to.
 *
 * Listed rather than answered. The support page shows them so a
 * customer is not left hunting for an answer that is not there, and
 * routes them to the contact form — where a person can give an answer
 * the company actually stands behind.
 *
 * Each of these needs a business or legal decision, not a code change.
 */
export const OPEN_POLICY_QUESTIONS: string[] = [
  "환불 정책과 환불 가능 기간",
  "생성된 음악의 저작권 귀속",
  "상업적 이용의 정확한 범위와 조건",
  "계정 삭제 후 데이터 보존 기간",
  "구독 해지 시 잔여 기간 환불 여부",
];

export function faqFor(category: FaqCategory): FaqEntry[] {
  return FAQ.filter((entry) => entry.category === category);
}
