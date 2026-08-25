/**
 * Plan shapes, with no prices and no policy in them yet.
 *
 * BOORDA has three tiers by name and nothing else decided: pricing,
 * credit allowance and per-tier limits are all open questions. So this
 * file describes the *shape* a plan will have and leaves every number
 * `null`, and the UI renders "미정" wherever a null appears rather than
 * inventing a figure that would later have to be walked back.
 *
 * When pricing is settled this becomes configuration — the same three
 * objects with values filled in, or a fetch from the billing service —
 * and no component below has to change.
 *
 * Nothing here talks to a payment provider, because there isn't one.
 */

export type PlanId = "free" | "basic" | "pro";

export interface PlanFeature {
  /** Short label shown in the tier's feature list. */
  label: string;
  /**
   * Whether this tier includes it. `null` means undecided, which is a
   * different statement from "no" and is rendered differently.
   */
  included: boolean | null;
}

export interface Plan {
  id: PlanId;
  name: string;
  /** One line under the name. Describes who the tier is for. */
  tagline: string;
  /**
   * Monthly price in KRW. `null` until pricing is decided — it is not
   * `0`, because `0` is a price and this is the absence of one.
   */
  monthlyPriceKrw: number | null;
  /** Monthly generation credits. `null` until the policy exists. */
  monthlyCredits: number | null;
  /** Longest single track, in seconds. `null` until decided. */
  maxDurationSeconds: number | null;
  features: PlanFeature[];
  /** Marks the tier the page highlights. Presentation only. */
  highlighted?: boolean;
}

/**
 * The three tiers, by name only.
 *
 * Every numeric field is deliberately `null`. Reviewers should read a
 * null here as "not decided", never as "free" or "unlimited".
 */
export const PLANS: readonly Plan[] = [
  {
    id: "free",
    name: "Free",
    tagline: "부르다를 처음 써보는 분께",
    monthlyPriceKrw: null,
    monthlyCredits: null,
    maxDurationSeconds: null,
    features: [
      { label: "음악 생성", included: true },
      { label: "라이브러리 보관", included: true },
      { label: "MP3 다운로드", included: null },
      { label: "상업적 이용", included: null },
    ],
  },
  {
    id: "basic",
    name: "Basic",
    tagline: "꾸준히 만드는 분께",
    monthlyPriceKrw: null,
    monthlyCredits: null,
    maxDurationSeconds: null,
    highlighted: true,
    features: [
      { label: "음악 생성", included: true },
      { label: "라이브러리 보관", included: true },
      { label: "MP3 다운로드", included: null },
      { label: "상업적 이용", included: null },
    ],
  },
  {
    id: "pro",
    name: "Pro",
    tagline: "작업량이 많은 분께",
    monthlyPriceKrw: null,
    monthlyCredits: null,
    maxDurationSeconds: null,
    features: [
      { label: "음악 생성", included: true },
      { label: "라이브러리 보관", included: true },
      { label: "MP3 다운로드", included: null },
      { label: "상업적 이용", included: null },
    ],
  },
];

/**
 * The plan a signed-in user is on.
 *
 * There is no subscription backend, so this cannot be answered yet and
 * the type says so. Components render an "미정" state for `null` rather
 * than defaulting to Free, because defaulting would be a claim about
 * the user's account that nothing has verified.
 */
export type CurrentPlan = Plan | null;

/**
 * Credits remaining.
 *
 * `null` until the credit ledger exists. Same reasoning as above: zero
 * is a balance, and we do not have one to report.
 */
export type CreditBalance = number | null;

export function formatPriceKrw(price: number | null): string {
  if (price === null) return "미정";
  if (price === 0) return "무료";
  return `₩${price.toLocaleString("ko-KR")}`;
}

export function formatCredits(credits: number | null): string {
  return credits === null ? "미정" : credits.toLocaleString("ko-KR");
}

/**
 * The plan the signed-in user is on, and the credits they have left.
 *
 * Both return `null`: there is no subscription record and no credit
 * ledger to read. They exist as functions rather than as inline `null`
 * so that there is exactly one place to wire the billing service in,
 * and so every caller is already written against the eventual type.
 */
export function currentPlan(): CurrentPlan {
  return null;
}

export function creditBalance(): CreditBalance {
  return null;
}
