"use client";

/**
 * Everything you can do *to* a song, in one component.
 *
 * Library, Projects and Song detail all need favourite, rename, delete,
 * duplicate-settings, download and project assignment. Implemented per
 * page, those six behaviours drift: one surface forgets the
 * confirmation, another forgets the toast, a third renames without
 * telling the parent to refresh. Implemented once, they cannot.
 *
 * The component owns its own dialogs and feedback and reports the result
 * upwards through `onChanged` / `onDeleted`, so a page only has to say
 * how it wants to re-read its data.
 */

import Link from "next/link";
import { useState } from "react";

import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { useToast } from "@/components/ui/Toast";
import { Button, cx, inputClass } from "@/components/ui";
import {
  assignGenerationToProject,
  deleteGeneration,
  getAudioAssetUrl,
  updateGeneration,
  type Generation,
  type Project,
} from "@/lib/api";
import { downloadOptions } from "@/lib/download";

export interface SongActionsProps {
  generation: Generation;
  /** Projects available to file into. Omit to hide the control. */
  projects?: Project[];
  onChanged?: (generation: Generation) => void;
  onDeleted?: (id: string) => void;
  /** Renders the wide layout used on the detail page. */
  variant?: "inline" | "detail";
}

function HeartIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={filled ? 0 : 1.8}
      aria-hidden="true"
    >
      <path d="M12 20.3l-1.45-1.32C5.4 14.25 2 11.17 2 7.5A4.5 4.5 0 0 1 6.5 3c1.74 0 3.41.81 4.5 2.09A5.98 5.98 0 0 1 15.5 3 4.5 4.5 0 0 1 20 7.5c0 3.67-3.4 6.75-8.55 11.49L12 20.3Z" />
    </svg>
  );
}

/**
 * The favourite control.
 *
 * Optimistic: the heart fills immediately and reverts if the write
 * fails. A favourite is a low-stakes, high-frequency action, and waiting
 * for a round trip before the icon responds makes the whole app feel
 * broken.
 */
export function FavoriteButton({
  generation,
  onChanged,
  className,
}: {
  generation: Generation;
  onChanged?: (generation: Generation) => void;
  className?: string;
}) {
  const toast = useToast();
  const [optimistic, setOptimistic] = useState<boolean | null>(null);
  const favorite = optimistic ?? generation.favorite;

  const toggle = async () => {
    const next = !favorite;
    setOptimistic(next);
    try {
      const updated = await updateGeneration(generation.id, { favorite: next });
      onChanged?.(updated);
      toast.notify(next ? "Added to favorites" : "Removed from favorites");
    } catch {
      setOptimistic(!next);
      toast.notifyError("Could not update this favorite.");
    }
  };

  return (
    <button
      type="button"
      onClick={() => void toggle()}
      aria-pressed={favorite}
      aria-label={favorite ? `Unfavorite ${generation.title}` : `Favorite ${generation.title}`}
      className={cx(
        "inline-flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)]",
        "transition-colors hover:bg-[var(--surface-hover)]",
        favorite ? "text-[var(--danger)]" : "text-[var(--text-muted)]",
        className,
      )}
    >
      <HeartIcon filled={favorite} />
    </button>
  );
}

export function SongActions({
  generation,
  projects,
  onChanged,
  onDeleted,
  variant = "inline",
}: SongActionsProps) {
  const toast = useToast();
  const [renaming, setRenaming] = useState(false);
  const [draftTitle, setDraftTitle] = useState(generation.title);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const options = downloadOptions(generation);

  const commitRename = async () => {
    const title = draftTitle.trim();
    if (!title || title === generation.title) {
      setRenaming(false);
      return;
    }
    try {
      const updated = await updateGeneration(generation.id, { title });
      onChanged?.(updated);
      toast.notify("Renamed");
    } catch {
      toast.notifyError("Could not rename this song.");
    }
    setRenaming(false);
  };

  const confirmDelete = async () => {
    setConfirmingDelete(false);
    try {
      await deleteGeneration(generation.id);
      onDeleted?.(generation.id);
      toast.notify("Song deleted");
    } catch {
      toast.notifyError("Could not delete this song.");
    }
  };

  const moveToProject = async (projectId: string) => {
    try {
      const updated = await assignGenerationToProject(
        generation.id,
        projectId === "" ? null : projectId,
      );
      onChanged?.(updated);
      toast.notify(projectId === "" ? "Removed from project" : "Added to project");
    } catch {
      toast.notifyError("Could not move this song.");
    }
  };

  if (renaming) {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={`rename-${generation.id}`} className="sr-only">
          Song title
        </label>
        <input
          id={`rename-${generation.id}`}
          autoFocus
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void commitRename();
            if (e.key === "Escape") {
              setDraftTitle(generation.title);
              setRenaming(false);
            }
          }}
          className={cx(inputClass, "max-w-xs py-1.5 text-sm")}
        />
        <Button size="sm" variant="primary" onClick={() => void commitRename()}>
          Save
        </Button>
        <Button
          size="sm"
          onClick={() => {
            setDraftTitle(generation.title);
            setRenaming(false);
          }}
        >
          Cancel
        </Button>
      </div>
    );
  }

  const buttonSize = variant === "detail" ? "md" : "sm";

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <FavoriteButton generation={generation} onChanged={onChanged} />

        <Button
          size={buttonSize}
          onClick={() => {
            setDraftTitle(generation.title);
            setRenaming(true);
          }}
        >
          Rename
        </Button>

        {/* Duplicate settings opens Create prefilled and records no
            lineage — nothing exists until the user presses Create. */}
        <Link href={`/create?duplicate=${generation.id}`}>
          <Button size={buttonSize}>Duplicate settings</Button>
        </Link>

        {options.map((option) => (
          <a
            key={option.kind}
            href={getAudioAssetUrl(generation.id, option.kind, true)}
            download={option.filename}
            onClick={() => toast.notify("Download started")}
            title={`${option.hint} · ${option.filename}`}
          >
            <Button size={buttonSize}>{option.label}</Button>
          </a>
        ))}

        {projects && projects.length > 0 && (
          <div>
            <label htmlFor={`project-${generation.id}`} className="sr-only">
              Project
            </label>
            <select
              id={`project-${generation.id}`}
              value={generation.project_id ?? ""}
              onChange={(e) => void moveToProject(e.target.value)}
              className={cx(inputClass, "!w-auto py-1.5 text-xs")}
            >
              <option value="">No project</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>
        )}

        <Button size={buttonSize} variant="danger" onClick={() => setConfirmingDelete(true)}>
          Delete
        </Button>
      </div>

      <ConfirmDialog
        open={confirmingDelete}
        title="Delete this song?"
        description={
          <>
            <span className="font-medium text-[var(--text-primary)]">{generation.title}</span>{" "}
            and its audio will be removed permanently. Songs generated from it are kept.
          </>
        }
        confirmLabel="Delete song"
        destructive
        onConfirm={() => void confirmDelete()}
        onCancel={() => setConfirmingDelete(false)}
      />
    </>
  );
}
