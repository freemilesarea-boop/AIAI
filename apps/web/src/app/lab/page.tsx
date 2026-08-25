"use client";

/**
 * BOORDA LAB — a technical preview area, deliberately not a launch page.
 *
 * Every entry here is currently unavailable, so the page's main job is
 * to be honest about that while still showing the shape of what is
 * coming. Each card carries a disabled "체험하기" that is visibly
 * disabled — greyed, `aria-disabled`, and paired with a sentence saying
 * why — because a control that looks live and does nothing is worse
 * than no control at all.
 *
 * The catalogue comes from `lib/lab.ts`. When a model API exists, this
 * page renders whatever it returns without changing.
 */

import { Card } from "@/components/ui";
import {
  LIFECYCLE,
  STATUS_PRESENTATION,
  formatReleaseDate,
  isUsable,
  labCatalog,
  type LabEntry,
  type LabCatalog,
  type ModelStatus,
} from "@/lib/lab";

function StatusBadge({ status }: { status: ModelStatus }) {
  const { label, className } = STATUS_PRESENTATION[status];
  return (
    <span
      className={
        "inline-flex items-center rounded-[var(--radius-full)] px-2 py-0.5 " +
        "text-[10px] font-semibold tracking-wide " +
        className
      }
    >
      {label}
    </span>
  );
}

function EntryCard({ entry, catalog }: { entry: LabEntry; catalog: LabCatalog }) {
  const usable = isUsable(entry, catalog);

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={entry.status} />
          <span className="text-[11px] font-medium text-[var(--text-muted)]">
            {entry.version}
          </span>
        </div>
        <h2 className="text-base font-semibold text-[var(--text-primary)]">{entry.name}</h2>
        <p className="text-sm text-[var(--text-secondary)]">{entry.description}</p>
      </div>

      <div className="flex flex-col gap-1.5 border-t border-[var(--border-subtle)] pt-3">
        <p className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
          개선하려는 점
        </p>
        <ul className="flex list-disc flex-col gap-1 pl-4 text-sm text-[var(--text-secondary)]">
          {entry.improvements.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <dl className="flex flex-col gap-1.5 border-t border-[var(--border-subtle)] pt-3 text-sm">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">공개일</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {formatReleaseDate(entry.releaseDate)}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">이용 가능</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {usable ? "이용 가능" : "아직 불가"}
          </dd>
        </div>
      </dl>

      {entry.experimentalWarning ? (
        <p
          role="note"
          className="rounded-[var(--radius-md)] bg-[var(--danger-muted)] px-3 py-2 text-xs text-[var(--danger)]"
        >
          {entry.experimentalWarning}
        </p>
      ) : null}

      {entry.technicalNotes ? (
        <p className="text-xs leading-relaxed text-[var(--text-muted)]">{entry.technicalNotes}</p>
      ) : null}

      <div className="mt-auto flex flex-col gap-1.5 pt-1">
        {/*
          Rendered as a real disabled button rather than hidden, so the
          shape of the future action is visible. `disabled` blocks the
          click, `aria-disabled` states it, and the line below says why —
          a greyed control with no explanation is a dead end.
        */}
        <button
          type="button"
          disabled={!usable}
          aria-disabled={!usable}
          className={
            "inline-flex h-9 items-center justify-center rounded-[var(--radius-md)] " +
            "px-4 text-sm font-medium transition-colors " +
            (usable
              ? "bg-[var(--brand)] text-white hover:bg-[var(--brand-hover)]"
              : "cursor-not-allowed bg-[var(--surface-sunken)] text-[var(--text-muted)]")
          }
        >
          체험하기
        </button>
        {!usable ? (
          <p className="text-[11px] text-[var(--text-muted)]">
            아직 사용할 수 없습니다. 준비되면 여기에서 안내합니다.
          </p>
        ) : null}
      </div>
    </Card>
  );
}

export default function LabPage() {
  const catalog = labCatalog();

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
            BOORDA LAB
          </h1>
          <span className="inline-flex items-center rounded-[var(--radius-full)] bg-[var(--surface-overlay)] px-2 py-0.5 text-[10px] font-semibold tracking-wide text-[var(--text-secondary)]">
            BETA
          </span>
        </div>
        <p className="max-w-prose text-sm text-[var(--text-secondary)]">
          새로운 모델과 실험 기능을 가장 먼저 만나보세요.
        </p>
        {/*
          Said once, at the top, rather than repeated on every card:
          nothing in LAB is selectable yet.
        */}
        <p className="max-w-prose rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2 text-xs text-[var(--text-muted)]">
          아래 항목은 준비 중인 예시입니다. 현재 음악 생성에 사용할 수 있는 모델은 없으며,
          모델 선택 기능도 아직 제공하지 않습니다.
        </p>
      </header>

      <section aria-labelledby="experiments-heading" className="flex flex-col gap-3">
        <h2 id="experiments-heading" className="sr-only">
          실험 중인 모델과 기능
        </h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {catalog.entries.map((entry) => (
            <EntryCard key={entry.id} entry={entry} catalog={catalog} />
          ))}
        </div>
      </section>

      <section aria-labelledby="lifecycle-heading">
        <Card className="p-5">
          <h2
            id="lifecycle-heading"
            className="text-sm font-semibold text-[var(--text-primary)]"
          >
            모델이 공개되는 단계
          </h2>
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            실험을 거쳐 안정화되면 기본 모델이 됩니다.
          </p>
          <ol className="mt-3 flex flex-wrap items-center gap-2">
            {LIFECYCLE.filter((status) => status !== "COMING_SOON").map((status, index) => (
              <li key={status} className="flex items-center gap-2">
                {index > 0 ? (
                  <span aria-hidden="true" className="text-[var(--text-muted)]">
                    →
                  </span>
                ) : null}
                <StatusBadge status={status} />
              </li>
            ))}
            <li className="flex items-center gap-2">
              <span aria-hidden="true" className="text-[var(--text-muted)]">
                →
              </span>
              <span className="inline-flex items-center rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[10px] font-semibold tracking-wide text-[var(--brand-text)]">
                DEFAULT
              </span>
            </li>
          </ol>
        </Card>
      </section>
    </div>
  );
}
