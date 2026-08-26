"use client";

/**
 * Library — every generation this deployment has produced.
 *
 * Backed entirely by real data from the API. There are no demo or
 * placeholder tracks mixed in: an empty library shows an empty state and
 * says so, rather than inventing songs to make the page look full.
 *
 * Filtering, searching and sorting happen in the browser over one fetch.
 * At this data volume that is instant and composes cleanly; the API
 * keeps its `limit`/`offset` parameters so moving the work server-side
 * later is a change of caller, not a redesign.
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { SongCard } from "@/components/SongCard";
import { Button, EmptyState, SkeletonCard, Tabs, cx, inputClass } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { useToast } from "@/components/ui/Toast";
import {
  bulkAssignProject,
  bulkDeleteGenerations,
  listGenerations,
  listProjects,
  type Generation,
  type Project,
} from "@/lib/api";
import {
  LIBRARY_FILTERS,
  LIBRARY_SORTS,
  visibleGenerations,
  type LibraryFilter,
  type LibrarySort,
} from "@/lib/library";

const PAGE_SIZE = 100;

export default function LibraryPage() {
  const toast = useToast();
  const [items, setItems] = useState<Generation[] | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<LibraryFilter>("all");
  const [sort, setSort] = useState<LibrarySort>("newest");
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const load = useCallback(async () => {
    try {
      setError(false);
      const body = await listGenerations(PAGE_SIZE, 0);
      setItems(body.items);
    } catch {
      setItems([]);
      setError(true);
    }
  }, []);

  useEffect(() => {
    void load();
    void listProjects()
      .then(setProjects)
      .catch(() => setProjects([]));
  }, [load]);

  const visible = useMemo(
    () => visibleGenerations(items ?? [], { query, filter, sort }),
    [items, query, filter, sort],
  );

  // One song changing does not warrant re-reading the whole library.
  const replaceItem = useCallback((updated: Generation) => {
    setItems((current) =>
      current ? current.map((item) => (item.id === updated.id ? updated : item)) : current,
    );
  }, []);

  const toggleSelected = useCallback((id: string, on: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);

  const exitSelection = useCallback(() => {
    setSelecting(false);
    setSelected(new Set());
  }, []);

  const bulkDelete = async () => {
    setConfirmingDelete(false);
    const ids = [...selected];
    try {
      const { affected } = await bulkDeleteGenerations(ids);
      setItems((current) => (current ? current.filter((item) => !selected.has(item.id)) : current));
      toast.notify(`${affected}곡을 삭제했습니다`);
    } catch {
      toast.notifyError("곡을 삭제하지 못했습니다.");
    }
    exitSelection();
  };

  const bulkProject = async (projectId: string | null) => {
    const ids = [...selected];
    if (ids.length === 0) return;
    try {
      const { affected } = await bulkAssignProject(ids, projectId);
      toast.notify(
        projectId === null
          ? `${affected}곡을 프로젝트에서 뺐습니다`
          : `${affected}곡을 프로젝트에 넣었습니다`,
      );
      await load();
    } catch {
      toast.notifyError("곡을 옮기지 못했습니다.");
    }
    exitSelection();
  };

  const allVisibleSelected = visible.length > 0 && visible.every((g) => selected.has(g.id));

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">라이브러리</h1>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            내가 만든 모든 음악입니다.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {items !== null && items.length > 0 && (
            <Button onClick={() => (selecting ? exitSelection() : setSelecting(true))}>
              {selecting ? "완료" : "선택"}
            </Button>
          )}
          {/*
            Projects is no longer in the primary navigation — BOORDA's
            IA is five areas — but the feature still exists and is
            reached from the library it organises.
          */}
          <Link href="/projects">
            <Button>프로젝트</Button>
          </Link>
          <Link href="/create">
            <Button variant="primary">새 음악</Button>
          </Link>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-[200px] flex-1">
          <label htmlFor="library-search" className="sr-only">
            제목이나 설명으로 검색
          </label>
          <input
            id="library-search"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="제목이나 설명으로 검색"
            className={inputClass}
          />
        </div>
        <Tabs
          label="상태로 거르기"
          value={filter}
          onChange={setFilter}
          options={LIBRARY_FILTERS}
        />
        <div>
          <label htmlFor="library-sort" className="sr-only">
            정렬
          </label>
          <select
            id="library-sort"
            value={sort}
            onChange={(e) => setSort(e.target.value as LibrarySort)}
            className={cx(inputClass, "!w-auto")}
          >
            {LIBRARY_SORTS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {selecting && (
        <div
          role="toolbar"
          aria-label="선택 항목 작업"
          className="flex flex-wrap items-center gap-3 rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-raised)] px-3 py-2"
        >
          <span className="text-sm text-[var(--text-secondary)]">
            {selected.size}개 선택됨
          </span>
          <Button
            size="sm"
            onClick={() =>
              setSelected(allVisibleSelected ? new Set() : new Set(visible.map((g) => g.id)))
            }
          >
            {allVisibleSelected ? "선택 해제" : "전체 선택"}
          </Button>
          {projects.length > 0 && (
            // Wrapped: as a direct flex child the select stretches to
            // fill the row and pushes the rest of the toolbar onto extra
            // lines. The width utility alone does not win against the
            // shared input class.
            <div>
              <label htmlFor="bulk-project" className="sr-only">
                선택한 곡을 프로젝트에 추가
              </label>
              <select
                id="bulk-project"
                value=""
                disabled={selected.size === 0}
                onChange={(e) => void bulkProject(e.target.value)}
                className={cx(inputClass, "!w-auto py-1.5 text-xs")}
              >
                <option value="" disabled>
                  프로젝트에 추가…
                </option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          <Button
            size="sm"
            disabled={selected.size === 0}
            onClick={() => void bulkProject(null)}
          >
            Remove from project
          </Button>
          <Button
            size="sm"
            variant="danger"
            disabled={selected.size === 0}
            onClick={() => setConfirmingDelete(true)}
          >
            Delete
          </Button>
        </div>
      )}

      {items === null ? (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="라이브러리 불러오는 중">
          {[0, 1, 2, 3].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          title="라이브러리를 불러오지 못했습니다"
          description="BOORDA 서버가 응답하지 않았습니다. 저장된 음악은 그대로 있습니다 — 연결 문제입니다."
          action={<Button onClick={() => void load()}>다시 시도</Button>}
        />
      ) : visible.length === 0 ? (
        items.length === 0 ? (
          <EmptyState
            title="아직 만든 음악이 없습니다"
            description="원하는 분위기를 짧게 적고, 보컬을 원하면 가사를 더해 첫 곡을 만들어 보세요."
            action={
              <Link href="/create">
                <Button variant="primary">첫 음악 만들기</Button>
              </Link>
            }
          />
        ) : filter === "favorites" ? (
          <EmptyState
            title="즐겨찾기가 없습니다"
            description="곡의 하트를 누르면 여기에 모입니다."
            action={<Button onClick={() => setFilter("all")}>전체 보기</Button>}
          />
        ) : (
          <EmptyState
            title="검색 결과가 없습니다"
            description="검색어와 필터에 맞는 곡이 없습니다. 다른 단어로 찾거나 필터를 전체로 되돌려 보세요."
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
        )
      ) : (
        <div className="flex flex-col gap-3">
          {visible.map((generation) => (
            <SongCard
              key={generation.id}
              generation={generation}
              onChanged={replaceItem}
              selected={selecting ? selected.has(generation.id) : undefined}
              onSelectedChange={
                selecting ? (on) => toggleSelected(generation.id, on) : undefined
              }
            />
          ))}
        </div>
      )}

      <ConfirmDialog
        open={confirmingDelete}
        title={`Delete ${selected.size} ${selected.size === 1 ? "song" : "songs"}?`}
        description="선택한 곡과 오디오가 영구히 삭제됩니다. 되돌릴 수 없습니다."
        confirmLabel={`Delete ${selected.size}`}
        destructive
        onConfirm={() => void bulkDelete()}
        onCancel={() => setConfirmingDelete(false)}
      />
    </div>
  );
}
