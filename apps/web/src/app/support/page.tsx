"use client";

/**
 * Support home: what you can do, and answers to what is already settled.
 *
 * The FAQ below only restates behaviour the code enforces. Questions
 * that need a policy decision nobody has made — refunds, copyright,
 * commercial-use scope — are listed as open rather than answered, and
 * point at the contact form. A FAQ that invents an answer creates a
 * promise the company then has to keep.
 */

import { useState } from "react";

import { ButtonLink, Card } from "@/components/ui";
import { FAQ_CATEGORIES, OPEN_POLICY_QUESTIONS, faqFor, type FaqCategory } from "@/lib/faq";

function Faq({ category }: { category: FaqCategory }) {
  const entries = faqFor(category);
  if (entries.length === 0) {
    return <p className="text-sm text-[var(--text-muted)]">아직 등록된 항목이 없습니다.</p>;
  }
  return (
    <div className="flex flex-col gap-2">
      {entries.map((entry) => (
        <details
          key={entry.id}
          className="group rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-raised)] px-4 py-3"
        >
          <summary className="cursor-pointer list-none text-sm font-medium text-[var(--text-primary)] marker:content-none">
            {entry.question}
          </summary>
          <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-[var(--text-secondary)]">
            {entry.answer}
          </p>
        </details>
      ))}
    </div>
  );
}

export default function SupportPage() {
  const [active, setActive] = useState<FaqCategory>("ACCOUNT");

  return (
    <div className="flex max-w-3xl flex-col gap-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          고객지원
        </h1>
        <p className="text-sm text-[var(--text-secondary)]">
          부르다 이용 중 궁금한 점이나 문제가 있으신가요?
        </p>
      </header>

      <nav aria-label="고객지원 바로가기" className="grid gap-3 sm:grid-cols-3">
        <ButtonLink href="/support/contact" variant="primary">
          문의하기
        </ButtonLink>
        <ButtonLink href="/support/inquiries" variant="secondary">
          내 문의내역
        </ButtonLink>
        <ButtonLink href="#faq" variant="secondary">
          자주 묻는 질문
        </ButtonLink>
      </nav>

      <section id="faq" aria-labelledby="faq-heading" className="flex scroll-mt-6 flex-col gap-3">
        <h2 id="faq-heading" className="text-sm font-semibold text-[var(--text-primary)]">
          자주 묻는 질문
        </h2>

        <div role="tablist" aria-label="자주 묻는 질문 분류" className="flex flex-wrap gap-2">
          {FAQ_CATEGORIES.map((category) => (
            <button
              key={category.id}
              type="button"
              role="tab"
              aria-selected={active === category.id}
              onClick={() => setActive(category.id)}
              className={
                "rounded-[var(--radius-full)] border px-3 py-1 text-xs font-medium transition-colors " +
                (active === category.id
                  ? "border-[var(--brand)] bg-[var(--brand-muted)] text-[var(--brand-text)]"
                  : "border-[var(--border-default)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]")
              }
            >
              {category.label}
            </button>
          ))}
        </div>

        <Faq category={active} />
      </section>

      {/*
        Named rather than answered. Each of these needs a business or
        legal decision, and a plausible-sounding guess here would become
        the thing a customer relies on.
      */}
      <Card className="p-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">
          아직 안내가 준비되지 않은 항목
        </h2>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          아래 내용은 정책이 확정되면 안내드리겠습니다. 지금 확인이 필요하시면 문의해 주세요.
        </p>
        <ul className="mt-3 flex list-disc flex-col gap-1 pl-5 text-sm text-[var(--text-secondary)]">
          {OPEN_POLICY_QUESTIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <div className="mt-4">
          <ButtonLink href="/support/contact" variant="secondary" size="sm">
            문의하기
          </ButtonLink>
        </div>
      </Card>
    </div>
  );
}
