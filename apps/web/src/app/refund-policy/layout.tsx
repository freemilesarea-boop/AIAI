/**
 * Metadata for the refund policy.
 *
 * In a layout because the page itself is a client component — it reads
 * the live plan catalogue rather than repeating prices that could drift
 * — and `metadata` may only be exported from a server component.
 */

import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "구독·결제·환불 정책 | BOORDA",
  description:
    "BOORDA 유료 구독의 정기결제, 해지 시점, 결제 실패 처리 및 환불 요청 방법을 안내합니다.",
};

export default function RefundPolicyLayout({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
