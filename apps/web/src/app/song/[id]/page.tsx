"use client";

/**
 * Song detail.
 *
 * The top of this page is for a listener: title, brief, lyrics, and the
 * settings that were used in plain language. Everything a developer
 * needs — provider, model, seed, request trace — is real and preserved,
 * but it sits behind an Advanced disclosure so the normal product does
 * not read as a debugging console.
 */

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { ExtendSong } from "@/components/ExtendSong";
import { ReplaceSection } from "@/components/ReplaceSection";
import { trackFromGeneration, usePlayer } from "@/components/player/PlayerProvider";
import { SongActions } from "@/components/SongActions";
import { SongCard } from "@/components/SongCard";
import { Button, Card, EmptyState, Skeleton, StatusPill } from "@/components/ui";
import {
  getGeneration,
  getLineage,
  listProjects,
  type Generation,
  type Lineage,
  type Project,
} from "@/lib/api";
import { describeGenerationFailure } from "@/lib/errors";
import { describeRelation } from "@/lib/lineage";

function Detail({ label, value }: { label: string; value: string | number | null }) {
  if (value === null || value === "") return null;
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">{label}</dt>
      <dd className="mt-0.5 text-sm text-[var(--text-primary)]">{value}</dd>
    </div>
  );
}

export default function SongDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const player = usePlayer();
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [missing, setMissing] = useState(false);

  const id = params?.id;

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const [g, l, p] = await Promise.all([
        getGeneration(id),
        getLineage(id).catch(() => null),
        listProjects().catch(() => []),
      ]);
      setGeneration(g);
      setLineage(l);
      setProjects(p);
    } catch {
      setMissing(true);
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (missing) {
    return (
      <EmptyState
        title="Song not found"
        description="This track may have been deleted."
        action={
          <Link href="/library">
            <Button variant="primary">Back to library</Button>
          </Link>
        }
      />
    );
  }

  if (!generation) {
    return (
      <div className="flex flex-col gap-4" aria-busy="true" aria-label="Loading song">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  const track = trackFromGeneration(generation);
  const ready = generation.status === "COMPLETED" && track !== null;
  const failed = generation.status === "FAILED" || generation.status === "CANCELLED";

  return (
    <div className="flex flex-col gap-6">
      <div>
        <Link
          href="/library"
          className="-ml-2 inline-flex min-h-8 items-center rounded-[var(--radius-sm)] px-2
            text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        >
          ← Library
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">{generation.title}</h1>
          <StatusPill status={generation.status} />
        </div>
      </div>

      {failed && (
        <Card className="border-[var(--danger)]/40 bg-[var(--danger-muted)]/40 p-4">
          <p className="text-sm text-[var(--text-primary)]">
            {describeGenerationFailure(generation.error_code).message}
          </p>
          <div className="mt-3">
            <Button variant="primary" onClick={() => router.push("/create")}>
              Try again
            </Button>
          </div>
        </Card>
      )}

      <div className="flex flex-col gap-3">
        {ready && (
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" onClick={() => track && player.play(track)}>
              Play
            </Button>
            <Link href={`/create?from=${generation.id}`}>
              <Button>Generate again</Button>
            </Link>
            {/* The extension is queued like any other generation, so the
                user follows it in the Library rather than on this page. */}
            <ExtendSong
              generation={generation}
              onExtended={(id) => router.push(`/song/${id}`)}
            />
            <ReplaceSection
              generation={generation}
              onReplaced={(id) => router.push(`/song/${id}`)}
            />
          </div>
        )}
        {/* Rename, favourite, duplicate, downloads, project and delete —
            the same component every other surface uses, so they cannot
            behave differently here. */}
        <SongActions
          generation={generation}
          projects={projects}
          variant="detail"
          onChanged={setGeneration}
          onDeleted={() => router.push("/library")}
        />
      </div>

      <Card className="p-5">
        <h2 className="text-sm font-semibold">Brief</h2>
        <p className="mt-1.5 text-sm text-[var(--text-secondary)]">{generation.prompt}</p>

        <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Detail
            label="Duration"
            value={`${Math.round(generation.duration_actual ?? generation.duration_requested)}s`}
          />
          <Detail label="Vocal" value={generation.vocal_gender} />
          <Detail label="Language" value={generation.language} />
          <Detail label="BPM" value={generation.bpm} />
          <Detail label="Key" value={generation.key_scale} />
          <Detail label="Time signature" value={generation.time_signature} />
          <Detail label="Seed" value={generation.seed} />
          <Detail
            label="Created"
            value={new Date(generation.created_at).toLocaleString()}
          />
        </dl>
      </Card>

      {generation.lyrics.trim() && (
        <Card className="p-5">
          <h2 className="text-sm font-semibold">Lyrics</h2>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-sm leading-relaxed text-[var(--text-secondary)]">
            {generation.lyrics}
          </pre>
        </Card>
      )}

      {lineage && (lineage.parent || lineage.children.length > 0) && (
        <section>
          <h2 className="text-sm font-semibold">Generation history</h2>
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            {/* Two kinds of relative now exist and they are not the same
                thing: a re-generation shares only settings, an extension
                is built on this song's actual audio. Saying "no audio was
                reused" would be false for the second. */}
            Tracks related to this one. Each is labelled with what actually happened to the
            audio, which is not the same for all of them.
          </p>
          <div className="mt-3 flex flex-col gap-3">
            {lineage.parent && (
              <div>
                <p className="mb-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                  Came from
                </p>
                <SongCard generation={lineage.parent} />
              </div>
            )}
            {lineage.children.map((child) => {
              const relation = describeRelation(child);
              return (
                <div key={child.id} className="sm:pl-6">
                  {relation && (
                    <p
                      className="mb-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]"
                      title={relation.detail}
                    >
                      {relation.label}
                    </p>
                  )}
                  <SongCard generation={child} />
                </div>
              );
            })}
          </div>
        </section>
      )}

      <details className="rounded-[var(--radius-lg)] border border-[var(--border-subtle)] bg-[var(--surface-raised)] px-5 py-4">
        <summary className="cursor-pointer select-none text-sm font-medium text-[var(--text-secondary)]">
          Advanced details
        </summary>
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          Diagnostics for debugging. Not part of the normal experience.
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Detail label="Generation id" value={generation.id} />
          <Detail label="Provider" value={generation.provider} />
          <Detail label="Model" value={generation.model_name} />
          <Detail label="Model version" value={generation.model_version} />
          <Detail label="Status" value={generation.status} />
          <Detail label="Error code" value={generation.error_code} />
        </dl>
        {generation.request_trace && (
          <pre className="mt-3 max-h-72 overflow-auto rounded-[var(--radius-md)] bg-[var(--surface-sunken)] p-3 text-[11px] leading-relaxed text-[var(--text-muted)]">
            {JSON.stringify(generation.request_trace, null, 2)}
          </pre>
        )}
      </details>
    </div>
  );
}
