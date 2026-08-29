/**
 * Shared furniture for the legal pages.
 *
 * Legal text is only useful if it is read, and the default failure mode
 * is a wall of small grey paragraphs nobody scrolls. So: a bounded
 * measure, real heading hierarchy, sections that can be linked to, and
 * tables that scroll inside themselves rather than pushing the page
 * sideways on a phone.
 */

import type { ReactNode } from "react";

export function LegalPage({
  title,
  effective,
  intro,
  children,
}: {
  title: string;
  effective: string;
  intro?: ReactNode;
  children: ReactNode;
}) {
  return (
    <article className="mx-auto flex w-full max-w-3xl flex-col gap-8">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          {title}
        </h1>
        <p className="text-xs text-[var(--text-muted)]">시행일: {effective}</p>
        {intro ? (
          <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">{intro}</div>
        ) : null}
      </header>
      <div className="flex flex-col gap-10">{children}</div>
    </article>
  );
}

export function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="flex scroll-mt-8 flex-col gap-3">
      <h2 className="text-base font-semibold text-[var(--text-primary)]">{title}</h2>
      <div className="flex flex-col gap-3 text-sm leading-relaxed text-[var(--text-secondary)]">
        {children}
      </div>
    </section>
  );
}

export function Bullets({ items }: { items: ReactNode[] }) {
  return (
    <ul className="flex list-disc flex-col gap-1.5 pl-5">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}

/** A table that scrolls inside its own box rather than widening the page. */
export function DataTable({
  caption,
  headers,
  rows,
}: {
  caption: string;
  headers: string[];
  rows: ReactNode[][];
}) {
  return (
    <div className="overflow-x-auto rounded-[var(--radius-md)] border border-[var(--border-default)]">
      <table className="w-full min-w-[34rem] text-left text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-[var(--border-default)] text-[var(--text-muted)]">
            {headers.map((header) => (
              <th key={header} scope="col" className="p-3 font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-[var(--border-default)] last:border-0">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="p-3 align-top text-[var(--text-secondary)]">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
