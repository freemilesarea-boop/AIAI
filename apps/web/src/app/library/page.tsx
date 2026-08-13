"use client";

/**
 * Library — every generation this deployment has produced.
 *
 * Backed entirely by real data from the API. There are no demo or
 * placeholder tracks mixed in: an empty library shows an empty state
 * and says so, rather than inventing songs to make the page look full.
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { SongCard } from "@/components/SongCard";
import { Button, EmptyState, SkeletonCard, Tabs, cx, inputClass } from "@/components/ui";
import { listGenerations, type Generation } from "@/lib/api";
import {
  LIBRARY_FILTERS,
  matchesFilter,
  sortGenerations,
  type LibraryFilter,
  type LibrarySort,
} from "@/lib/library";

export default function LibraryPage() {
  const [items, setItems] = useState<Generation[] | null>(null);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<LibraryFilter>("all");
  const [sort, setSort] = useState<LibrarySort>("newest");

  const load = useCallback(async () => {
    try {
      setError(false);
      const body = await listGenerations(100, 0);
      setItems(body.items);
    } catch {
      setItems([]);
      setError(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const visible = useMemo(() => {
    const term = query.trim().toLowerCase();
    const filtered = (items ?? []).filter(
      (g) => matchesFilter(g, filter) && (!term || g.title.toLowerCase().includes(term)),
    );
    return sortGenerations(filtered, sort);
  }, [items, query, filter, sort]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Library</h1>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Every track you have generated.
          </p>
        </div>
        <Button variant="primary" onClick={() => (window.location.href = "/create")}>
          New song
        </Button>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-[200px] flex-1">
          <label htmlFor="library-search" className="sr-only">
            Search by title
          </label>
          <input
            id="library-search"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by title"
            className={inputClass}
          />
        </div>
        <Tabs label="Filter by status" value={filter} onChange={setFilter} options={LIBRARY_FILTERS} />
        <div>
          <label htmlFor="library-sort" className="sr-only">
            Sort
          </label>
          <select
            id="library-sort"
            value={sort}
            onChange={(e) => setSort(e.target.value as LibrarySort)}
            className={cx(inputClass, "w-auto")}
          >
            <option value="newest">Newest</option>
            <option value="oldest">Oldest</option>
          </select>
        </div>
      </div>

      {items === null ? (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading library">
          {[0, 1, 2, 3].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          title="Could not load your library"
          description="The LUBER service did not respond. Your tracks are safe — this is a connection problem."
          action={<Button onClick={() => void load()}>Try again</Button>}
        />
      ) : visible.length === 0 ? (
        (items.length === 0 ? (
          <EmptyState
            title="No songs yet"
            description="Write a short brief, add lyrics if you want vocals, and generate your first track."
            action={
              <Link href="/create">
                <Button variant="primary">Create your first song</Button>
              </Link>
            }
          />
        ) : (
          <EmptyState
            title="No matches"
            description="Nothing here matches that search and filter. Try a different title or set the filter back to All."
            action={
              <Button
                onClick={() => {
                  setQuery("");
                  setFilter("all");
                }}
              >
                Clear filters
              </Button>
            }
          />
        ))
      ) : (
        <div className="flex flex-col gap-3">
          {visible.map((generation) => (
            <SongCard key={generation.id} generation={generation} />
          ))}
        </div>
      )}
    </div>
  );
}
