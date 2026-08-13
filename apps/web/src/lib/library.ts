/**
 * Library filtering, searching and sorting rules.
 *
 * Lives outside the page module because Next.js page files may only
 * export a default component — and because these rules are worth testing
 * on their own, without rendering a page to do it.
 *
 * Filter, search and sort compose: each is applied independently, so
 * "favourites, matching 'night', oldest first" behaves the way it reads.
 */

import type { Generation } from "@/lib/api";

export type LibraryFilter = "all" | "favorites" | "completed" | "generating" | "failed";
export type LibrarySort = "newest" | "oldest" | "title_asc" | "title_desc";

export const LIBRARY_FILTERS: { value: LibraryFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "favorites", label: "Favorites" },
  { value: "completed", label: "Completed" },
  { value: "generating", label: "Generating" },
  { value: "failed", label: "Failed" },
];

export const LIBRARY_SORTS: { value: LibrarySort; label: string }[] = [
  { value: "newest", label: "Newest" },
  { value: "oldest", label: "Oldest" },
  { value: "title_asc", label: "Title A–Z" },
  { value: "title_desc", label: "Title Z–A" },
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
  if (filter === "favorites") return generation.favorite;
  if (filter === "completed") return generation.status === "COMPLETED";
  if (filter === "failed") {
    return generation.status === "FAILED" || generation.status === "CANCELLED";
  }
  return IN_FLIGHT.has(generation.status);
}

/**
 * Free-text match over the fields a user would actually search by.
 *
 * Title *and* prompt: people remember "the one about the rainy bus stop"
 * far more often than they remember what they called it.
 */
export function matchesQuery(generation: Generation, query: string): boolean {
  const term = query.trim().toLowerCase();
  if (!term) return true;
  return (
    generation.title.toLowerCase().includes(term) ||
    generation.prompt.toLowerCase().includes(term)
  );
}

export function sortGenerations(items: Generation[], sort: LibrarySort): Generation[] {
  const byTitle = (a: Generation, b: Generation) =>
    // Locale-aware so Korean titles order sensibly rather than by code point.
    a.title.localeCompare(b.title, undefined, { numeric: true, sensitivity: "base" });
  return [...items].sort((a, b) => {
    switch (sort) {
      case "title_asc":
        return byTitle(a, b);
      case "title_desc":
        return byTitle(b, a);
      case "oldest":
        return Date.parse(a.created_at) - Date.parse(b.created_at);
      default:
        return Date.parse(b.created_at) - Date.parse(a.created_at);
    }
  });
}

/** Filter, search and sort applied together. */
export function visibleGenerations(
  items: Generation[],
  { query, filter, sort }: { query: string; filter: LibraryFilter; sort: LibrarySort },
): Generation[] {
  return sortGenerations(
    items.filter((g) => matchesFilter(g, filter) && matchesQuery(g, query)),
    sort,
  );
}
