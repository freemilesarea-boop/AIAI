"use client";

/**
 * Projects — a folder for grouping generations.
 *
 * This page is the index only. Opening a project navigates to
 * `/projects/[id]`, which means an opened project has a URL: it can be
 * bookmarked, shared with yourself, and — the reason it matters most —
 * survives a refresh. Phase 11 held the opened project in React state
 * and lost it on reload.
 *
 * Intentionally small: create, open, delete. No collaboration, no
 * sharing, no permissions. Deleting a project never deletes the music
 * inside it, and the copy says so before you confirm.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Button, Card, EmptyState, Skeleton, cx, inputClass } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { useToast } from "@/components/ui/Toast";
import { createProject, deleteProject, listProjects, type Project } from "@/lib/api";

function formatWhen(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString();
}

export default function ProjectsPage() {
  const toast = useToast();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Project | null>(null);

  const load = useCallback(async () => {
    try {
      setProjects(await listProjects());
    } catch {
      setProjects([]);
      setError("Could not load projects.");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    try {
      await createProject(name);
      setNewName("");
      toast.notify("Project created");
      await load();
    } catch {
      toast.notifyError("Could not create that project.");
    }
  };

  const handleDelete = async () => {
    const target = deleting;
    setDeleting(null);
    if (!target) return;
    try {
      await deleteProject(target.id);
      toast.notify("Project deleted. Its songs were kept.");
      await load();
    } catch {
      toast.notifyError("Could not delete that project.");
    }
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
          className={cx(inputClass, "max-w-sm py-2 text-sm")}
        />
        <Button variant="primary" onClick={() => void handleCreate()}>
          Add
        </Button>
      </div>

      {projects === null ? (
        <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading projects">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : projects.length === 0 ? (
        <EmptyState
          title="No projects yet"
          description="Projects group related tracks. Name one above, then move songs into it."
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {projects.map((project) => (
            <li key={project.id}>
              <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
                <div className="min-w-0">
                  <Link
                    href={`/projects/${project.id}`}
                    className="-my-1 inline-flex min-h-8 items-center py-1 text-sm
                      font-semibold text-[var(--text-primary)] hover:underline"
                  >
                    {project.name}
                  </Link>
                  <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                    {project.generation_count}{" "}
                    {project.generation_count === 1 ? "song" : "songs"} · updated{" "}
                    {formatWhen(project.updated_at)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Link href={`/projects/${project.id}`}>
                    <Button size="sm">Open</Button>
                  </Link>
                  <Button size="sm" variant="danger" onClick={() => setDeleting(project)}>
                    Delete
                  </Button>
                </div>
              </Card>
            </li>
          ))}
        </ul>
      )}

      <ConfirmDialog
        open={deleting !== null}
        title={`Delete “${deleting?.name ?? ""}”?`}
        description={
          deleting?.generation_count
            ? `The project will be removed. Its ${deleting.generation_count} ${
                deleting.generation_count === 1 ? "song stays" : "songs stay"
              } in your library, unfiled.`
            : "The project will be removed. It has no songs in it."
        }
        confirmLabel="Delete project"
        destructive
        onConfirm={() => void handleDelete()}
        onCancel={() => setDeleting(null)}
      />
    </div>
  );
}
