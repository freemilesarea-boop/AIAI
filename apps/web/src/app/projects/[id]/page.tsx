"use client";

/**
 * One project and the songs filed under it.
 *
 * A real route, so the opened project has a URL and survives a refresh.
 * Everything on this page is re-read from the backend on load; nothing
 * depends on state handed over by the index page.
 *
 * Deliberately *not* a second Library. The controls here are the ones
 * that make sense inside a folder — play, favourite, remove, download,
 * open — plus adding songs that are not filed yet. Filtering by status
 * and bulk deletion stay in the Library, where a user is working with
 * their whole catalogue.
 */

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { SongCard } from "@/components/SongCard";
import {
  Button,
  Card,
  EmptyState,
  Skeleton,
  SkeletonCard,
  cx,
  inputClass,
} from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { useToast } from "@/components/ui/Toast";
import {
  assignGenerationToProject,
  deleteProject,
  getProject,
  listGenerations,
  listProjectGenerations,
  renameProject,
  type Generation,
  type Project,
} from "@/lib/api";
import { LIBRARY_SORTS, sortGenerations, type LibrarySort } from "@/lib/library";

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const toast = useToast();
  const id = params?.id;

  const [project, setProject] = useState<Project | null>(null);
  const [missing, setMissing] = useState(false);
  const [tracks, setTracks] = useState<Generation[] | null>(null);
  const [unfiled, setUnfiled] = useState<Generation[]>([]);
  const [sort, setSort] = useState<LibrarySort>("newest");
  const [query, setQuery] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const [found, filed, all] = await Promise.all([
        getProject(id),
        listProjectGenerations(id),
        listGenerations(100, 0),
      ]);
      setProject(found);
      setTracks(filed);
      const inProject = new Set(filed.map((g) => g.id));
      setUnfiled(
        all.items.filter((g) => !inProject.has(g.id) && g.status === "COMPLETED"),
      );
    } catch {
      setMissing(true);
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  const visible = useMemo(() => {
    const term = query.trim().toLowerCase();
    const filtered = (tracks ?? []).filter(
      (g) => !term || g.title.toLowerCase().includes(term),
    );
    return sortGenerations(filtered, sort);
  }, [tracks, query, sort]);

  const replaceTrack = useCallback((updated: Generation) => {
    setTracks((current) =>
      current ? current.map((item) => (item.id === updated.id ? updated : item)) : current,
    );
  }, []);

  const move = async (generation: Generation, projectId: string | null) => {
    try {
      await assignGenerationToProject(generation.id, projectId);
      toast.notify(projectId ? "Added to project" : "Removed from project");
      await load();
    } catch {
      toast.notifyError("Could not move that song.");
    }
  };

  const commitRename = async () => {
    if (!project) return;
    const name = renameValue.trim();
    if (!name || name === project.name) {
      setRenaming(false);
      return;
    }
    try {
      setProject(await renameProject(project.id, name));
      toast.notify("Project renamed");
    } catch {
      toast.notifyError("Could not rename that project.");
    }
    setRenaming(false);
  };

  const confirmDelete = async () => {
    if (!project) return;
    setConfirmingDelete(false);
    try {
      await deleteProject(project.id);
      toast.notify("Project deleted. Its songs were kept.");
      router.push("/projects");
    } catch {
      toast.notifyError("Could not delete that project.");
    }
  };

  if (missing) {
    return (
      <EmptyState
        title="Project not found"
        description="This project may have been deleted. Its songs are still in your library."
        action={
          <Link href="/projects">
            <Button variant="primary">Back to projects</Button>
          </Link>
        }
      />
    );
  }

  if (!project) {
    return (
      <div className="flex flex-col gap-4" aria-busy="true" aria-label="Loading project">
        <Skeleton className="h-8 w-56" />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <Link
          href="/projects"
          className="-ml-2 inline-flex min-h-8 items-center rounded-[var(--radius-sm)] px-2
            text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        >
          ← Projects
        </Link>

        {renaming ? (
          <div className="mt-2 flex flex-wrap gap-2">
            <label htmlFor="rename-project" className="sr-only">
              Project name
            </label>
            <input
              id="rename-project"
              autoFocus
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void commitRename();
                if (e.key === "Escape") setRenaming(false);
              }}
              className={cx(inputClass, "max-w-xs py-2 text-sm")}
            />
            <Button variant="primary" onClick={() => void commitRename()}>
              Save
            </Button>
            <Button onClick={() => setRenaming(false)}>Cancel</Button>
          </div>
        ) : (
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">{project.name}</h1>
              <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                {tracks?.length ?? project.generation_count}{" "}
                {(tracks?.length ?? project.generation_count) === 1 ? "song" : "songs"} · updated{" "}
                {new Date(project.updated_at).toLocaleDateString()}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => {
                  setRenameValue(project.name);
                  setRenaming(true);
                }}
              >
                Rename
              </Button>
              <Button size="sm" variant="danger" onClick={() => setConfirmingDelete(true)}>
                Delete project
              </Button>
            </div>
          </div>
        )}
      </div>

      {tracks !== null && tracks.length > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <div className="min-w-[180px] flex-1">
            <label htmlFor="project-search" className="sr-only">
              Search songs in this project
            </label>
            <input
              id="project-search"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by title"
              className={inputClass}
            />
          </div>
          <div>
            <label htmlFor="project-sort" className="sr-only">
              Sort
            </label>
            <select
              id="project-sort"
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
      )}

      {tracks === null ? (
        <div className="flex flex-col gap-3">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : tracks.length === 0 ? (
        <EmptyState
          title="Nothing filed here yet"
          description="Add a finished track from the list below, or from any song's page."
        />
      ) : visible.length === 0 ? (
        <EmptyState
          title="No matches"
          description="No song in this project matches that search."
          action={<Button onClick={() => setQuery("")}>Clear search</Button>}
        />
      ) : (
        <div className="flex flex-col gap-3">
          {visible.map((generation) => (
            <SongCard
              key={generation.id}
              generation={generation}
              onChanged={replaceTrack}
              extraActions={
                <button
                  type="button"
                  onClick={() => void move(generation, null)}
                  className="inline-flex h-8 items-center rounded-[var(--radius-sm)] px-2 transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
                >
                  Remove
                </button>
              }
            />
          ))}
        </div>
      )}

      {unfiled.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-[var(--text-secondary)]">
            Add to this project
          </h2>
          <div className="mt-2 flex flex-col gap-2">
            {unfiled.slice(0, 8).map((generation) => (
              <Card
                key={generation.id}
                className="flex items-center justify-between gap-3 px-3 py-2"
              >
                <span className="min-w-0 truncate text-sm">{generation.title}</span>
                <Button size="sm" onClick={() => void move(generation, project.id)}>
                  Add
                </Button>
              </Card>
            ))}
          </div>
        </section>
      )}

      <ConfirmDialog
        open={confirmingDelete}
        title={`Delete “${project.name}”?`}
        description={
          tracks && tracks.length > 0
            ? `The project will be removed. Its ${tracks.length} ${
                tracks.length === 1 ? "song stays" : "songs stay"
              } in your library, unfiled.`
            : "The project will be removed. It has no songs in it."
        }
        confirmLabel="Delete project"
        destructive
        onConfirm={() => void confirmDelete()}
        onCancel={() => setConfirmingDelete(false)}
      />
    </div>
  );
}
