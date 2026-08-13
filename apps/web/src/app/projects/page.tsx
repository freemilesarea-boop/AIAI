"use client";

/**
 * Projects — a folder for grouping generations.
 *
 * Intentionally small: create, rename, open, and move tracks in and
 * out. No collaboration, no sharing, no permissions. Deleting a project
 * never deletes the music inside it, and the copy says so before you
 * confirm.
 */

import { useCallback, useEffect, useState } from "react";

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
import {
  assignGenerationToProject,
  createProject,
  deleteProject,
  listGenerations,
  listProjectGenerations,
  listProjects,
  renameProject,
  type Generation,
  type Project,
} from "@/lib/api";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [selected, setSelected] = useState<Project | null>(null);
  const [tracks, setTracks] = useState<Generation[] | null>(null);
  const [unfiled, setUnfiled] = useState<Generation[]>([]);
  const [newName, setNewName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [error, setError] = useState<string | null>(null);

  const loadProjects = useCallback(async () => {
    try {
      setProjects(await listProjects());
    } catch {
      setProjects([]);
      setError("Could not load projects.");
    }
  }, []);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  const openProject = useCallback(async (project: Project) => {
    setSelected(project);
    setTracks(null);
    setRenaming(false);
    try {
      const [inProject, all] = await Promise.all([
        listProjectGenerations(project.id),
        listGenerations(100, 0),
      ]);
      setTracks(inProject);
      const filed = new Set(inProject.map((g) => g.id));
      setUnfiled(all.items.filter((g) => !filed.has(g.id) && g.status === "COMPLETED"));
    } catch {
      setTracks([]);
      setError("Could not open that project.");
    }
  }, []);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    try {
      const project = await createProject(name);
      setNewName("");
      await loadProjects();
      void openProject(project);
    } catch {
      setError("Could not create that project.");
    }
  };

  const handleRename = async () => {
    if (!selected) return;
    const name = renameValue.trim();
    if (!name) return;
    const updated = await renameProject(selected.id, name);
    setSelected(updated);
    setRenaming(false);
    await loadProjects();
  };

  const handleDelete = async () => {
    if (!selected) return;
    await deleteProject(selected.id);
    setSelected(null);
    setTracks(null);
    await loadProjects();
  };

  const move = async (generationId: string, projectId: string | null) => {
    await assignGenerationToProject(generationId, projectId);
    if (selected) await openProject(selected);
    await loadProjects();
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Group related tracks together. Deleting a project never deletes its music.
        </p>
      </header>

      {error && (
        <div
          role="alert"
          className="rounded-[var(--radius-md)] border border-[var(--danger)]/40 bg-[var(--danger-muted)] px-4 py-3 text-sm text-[var(--danger)]"
        >
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)] lg:items-start">
        <Card className="p-3">
          <div className="flex gap-2">
            <label htmlFor="new-project" className="sr-only">
              New project name
            </label>
            <input
              id="new-project"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleCreate();
              }}
              placeholder="New project"
              className={cx(inputClass, "py-2 text-sm")}
            />
            <Button variant="primary" size="md" onClick={() => void handleCreate()}>
              Add
            </Button>
          </div>

          <div className="mt-3 flex flex-col gap-1">
            {projects === null ? (
              <>
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
              </>
            ) : projects.length === 0 ? (
              <p className="px-2 py-6 text-center text-xs text-[var(--text-muted)]">
                No projects yet. Name one above to get started.
              </p>
            ) : (
              projects.map((project) => (
                <button
                  key={project.id}
                  type="button"
                  onClick={() => void openProject(project)}
                  aria-current={selected?.id === project.id ? "true" : undefined}
                  className={cx(
                    "flex items-center justify-between rounded-[var(--radius-md)] px-3 py-2 text-left text-sm transition-colors",
                    selected?.id === project.id
                      ? "bg-[var(--surface-overlay)] text-[var(--text-primary)]"
                      : "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]",
                  )}
                >
                  <span className="truncate">{project.name}</span>
                  <span className="ml-2 shrink-0 text-[11px] text-[var(--text-muted)]">
                    {project.generation_count}
                  </span>
                </button>
              ))
            )}
          </div>
        </Card>

        <div>
          {!selected ? (
            <EmptyState
              title={projects?.length ? "Pick a project" : "No projects yet"}
              description={
                projects?.length
                  ? "Choose a project to see the tracks filed under it."
                  : "Projects group related tracks. Create one above, then move songs into it."
              }
            />
          ) : (
            <div className="flex flex-col gap-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                {renaming ? (
                  <div className="flex flex-1 gap-2">
                    <label htmlFor="rename-project" className="sr-only">
                      Rename project
                    </label>
                    <input
                      id="rename-project"
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      className={cx(inputClass, "max-w-xs py-2 text-sm")}
                    />
                    <Button variant="primary" onClick={() => void handleRename()}>
                      Save
                    </Button>
                    <Button onClick={() => setRenaming(false)}>Cancel</Button>
                  </div>
                ) : (
                  <>
                    <h2 className="text-lg font-semibold">{selected.name}</h2>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={() => {
                          setRenameValue(selected.name);
                          setRenaming(true);
                        }}
                      >
                        Rename
                      </Button>
                      <Button size="sm" variant="danger" onClick={() => void handleDelete()}>
                        Delete project
                      </Button>
                    </div>
                  </>
                )}
              </div>

              {tracks === null ? (
                <div className="flex flex-col gap-3">
                  <SkeletonCard />
                  <SkeletonCard />
                </div>
              ) : tracks.length === 0 ? (
                <EmptyState
                  title="Nothing filed here yet"
                  description="Add a finished track from the list below."
                />
              ) : (
                <div className="flex flex-col gap-3">
                  {tracks.map((generation) => (
                    <SongCard
                      key={generation.id}
                      generation={generation}
                      extraActions={
                        <button
                          type="button"
                          onClick={() => void move(generation.id, null)}
                          className="rounded-[var(--radius-sm)] px-2 py-1 transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
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
                  <h3 className="text-sm font-medium text-[var(--text-secondary)]">
                    Add to this project
                  </h3>
                  <div className="mt-2 flex flex-col gap-2">
                    {unfiled.slice(0, 8).map((generation) => (
                      <Card
                        key={generation.id}
                        className="flex items-center justify-between gap-3 px-3 py-2"
                      >
                        <span className="min-w-0 truncate text-sm">{generation.title}</span>
                        <Button
                          size="sm"
                          onClick={() => void move(generation.id, selected.id)}
                        >
                          Add
                        </Button>
                      </Card>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
