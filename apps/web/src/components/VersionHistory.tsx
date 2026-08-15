"use client";

/**
 * Where this version sits among everything made from the same original.
 *
 * The tree is drawn from one backend response, not reconstructed here.
 * That matters after a refresh: the database is the only thing that
 * knows what was derived from what, and a locally-assembled graph would
 * drift the moment a sibling finished generating in another tab.
 *
 * Rendered as a real list with indentation rather than a graph. A git-
 * style canvas would need a dependency and would be unreadable at 390px,
 * where the whole hierarchy has to fit in a single column. Depth is
 * capped so a long chain cannot indent itself off the screen.
 *
 * The current version is marked in text, not only by colour — a user who
 * cannot distinguish the highlight still has to be able to tell which
 * song they are looking at.
 */

import Link from "next/link";

import { StatusPill } from "@/components/ui";
import type { LineageNode } from "@/lib/api";
import { formatClock, lineageDepths, operationLabel } from "@/lib/lineage";

export interface VersionHistoryProps {
  nodes: LineageNode[];
  currentId: string;
  rootId: string | null;
}

/** Indentation stops here so deep chains stay readable on a phone. */
const MAX_INDENT_STEPS = 4;
const INDENT_REM = 0.9;

function formatCreated(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
}

export function VersionHistory({ nodes, currentId, rootId }: VersionHistoryProps) {
  // A lineage of one is just this song; a history with a single entry
  // tells the user nothing they cannot already see. An absent array is
  // treated the same way rather than thrown on: a server that predates
  // this field, or a partial response, must not blank the page.
  if (!Array.isArray(nodes) || nodes.length <= 1) return null;

  const depths = lineageDepths(nodes);

  return (
    <section aria-labelledby="version-history-heading">
      <h2 id="version-history-heading" className="text-sm font-semibold">
        Version history
      </h2>
      <p className="mt-1 text-xs text-[var(--text-muted)]">
        Every version made from the same original. Each is its own song and keeps its own
        title.
      </p>

      <ol className="mt-3 flex flex-col gap-1.5">
        {nodes.map((node) => {
          const isCurrent = node.id === currentId;
          const indent = Math.min(depths.get(node.id) ?? 0, MAX_INDENT_STEPS) * INDENT_REM;
          const label = operationLabel(node);
          const duration =
            node.duration_actual !== null ? formatClock(node.duration_actual) : null;

          const body = (
            <>
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="truncate font-medium text-[var(--text-primary)]">
                    {node.title}
                  </span>
                  {isCurrent && (
                    // Text, not just the ring: colour alone would leave a
                    // screen-reader user with no way to locate themselves.
                    <span className="shrink-0 rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
                      Current version
                    </span>
                  )}
                </span>
                <span className="flex flex-wrap items-center gap-x-2 text-[11px] text-[var(--text-muted)]">
                  <span>{label}</span>
                  {duration && <span>· {duration}</span>}
                  <span>· {formatCreated(node.created_at)}</span>
                </span>
              </span>
              <StatusPill status={node.status} />
            </>
          );

          return (
            <li key={node.id} style={{ paddingLeft: `${indent}rem` }}>
              {isCurrent ? (
                <div
                  aria-current="true"
                  className="flex items-center justify-between gap-3 rounded-[var(--radius-md)] border border-[var(--brand)] bg-[var(--surface-raised)] px-3 py-2"
                >
                  {body}
                </div>
              ) : (
                <Link
                  href={`/song/${node.id}`}
                  className="flex items-center justify-between gap-3 rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-raised)] px-3 py-2 hover:border-[var(--border-strong)] hover:bg-[var(--surface-hover)]"
                >
                  {body}
                </Link>
              )}
            </li>
          );
        })}
      </ol>

      {rootId && rootId !== currentId && (
        <p className="mt-2 text-[11px] text-[var(--text-muted)]">
          The first entry is the original this family of versions came from.
        </p>
      )}
    </section>
  );
}
