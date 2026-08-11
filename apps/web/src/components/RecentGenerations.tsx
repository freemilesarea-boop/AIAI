"use client";

/**
 * Lightweight session history.
 *
 * The backend already lists generations, so this reads the real API and
 * uses locally remembered ids only to scope the list to what this
 * browser created. Per-user ownership is a later phase.
 */

import type { Generation } from "@/lib/api";
import { statusLabel } from "@/lib/generationStatus";

export interface RecentGenerationsProps {
  items: Generation[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

function formatCreatedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function RecentGenerations({ items, activeId, onSelect }: RecentGenerationsProps) {
  if (items.length === 0) return null;

  return (
    <section aria-labelledby="recent-generations-heading" className="mt-8">
      <h2
        id="recent-generations-heading"
        className="text-xs font-medium uppercase tracking-wide text-zinc-500"
      >
        This session
      </h2>
      <ul className="mt-3 flex flex-col gap-1.5">
        {items.map((item) => {
          const isActive = item.id === activeId;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                aria-current={isActive ? "true" : undefined}
                className={`flex w-full items-center justify-between gap-3 rounded-lg border
                  px-3 py-2 text-left text-sm transition-colors focus-visible:outline-none
                  focus-visible:ring-2 focus-visible:ring-violet-400 ${
                    isActive
                      ? "border-violet-700 bg-violet-950/40 text-zinc-100"
                      : "border-zinc-800 bg-zinc-900/40 text-zinc-300 hover:bg-zinc-900"
                  }`}
              >
                <span className="min-w-0 flex-1 truncate font-medium">{item.title}</span>
                <span className="shrink-0 text-xs text-zinc-500">
                  {item.duration_actual ? `${Math.round(item.duration_actual)}s · ` : ""}
                  {statusLabel(item.status)}
                </span>
                <span className="shrink-0 font-mono text-xs text-zinc-600">
                  {formatCreatedAt(item.created_at)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
