/**
 * Library filtering rules.
 *
 * Lives outside the page module because Next.js page files may only
 * export a default component — and because the classification is worth
 * testing on its own, without rendering a page to do it.
 */

import type { Generation } from "@/lib/api";

export type LibraryFilter = "all" | "completed" | "generating" | "failed";
export type LibrarySort = "newest" | "oldest";

export const LIBRARY_FILTERS: { value: LibraryFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "completed", label: "Completed" },
  { value: "generating", label: "Generating" },
  { value: "failed", label: "Failed" },
];

/** Statuses that mean "the engine still has this one". */
const IN_FLIGHT = new Set([
  "QUEUED",
  "STARTING",
  "GENERATING",
  "POST_PROCESSING",
  "UPLOADING",
]);

export function matchesFilter(generation: Generation, filter: LibraryFilter): boolean {
  if (filter === "all") return true;
  if (filter === "completed") return generation.status === "COMPLETED";
  if (filter === "failed") {
    return generation.status === "FAILED" || generation.status === "CANCELLED";
  }
  return IN_FLIGHT.has(generation.status);
}

export function sortGenerations(items: Generation[], sort: LibrarySort): Generation[] {
  return [...items].sort((a, b) => {
    const delta = Date.parse(a.created_at) - Date.parse(b.created_at);
    return sort === "newest" ? -delta : delta;
  });
}
