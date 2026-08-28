/**
 * Typed client for the BOORDA backend API.
 *
 * Every network call the browser makes goes through this module. The
 * browser talks only to the BOORDA API — it never contacts ACE-Step or
 * any model runtime directly, and it never sees a storage key or a
 * filesystem path. Audio is addressed by generation id.
 *
 * Values mirror `packages/schemas` (`luber_schemas.enums`), which owns
 * the persisted contract.
 */

import type { Advisory, PreflightResponse } from "@/lib/songcraft";

/**
 * Where the browser sends API requests.
 *
 * Empty by default, which makes every call a same-origin `/api/...`
 * path handled by the Next rewrite. That is deliberate: a same-origin
 * request carries the `SameSite=Lax` session cookie, while a direct
 * call to the backend's own port would not.
 *
 * The override exists for server-side rendering and tests, which have
 * no origin of their own to be same as.
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";

export interface HealthResponse {
  status: string;
  service: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, { cache: "no-store", credentials: "include" });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return (await res.json()) as HealthResponse;
}

/** Lifecycle of a generation job — mirrors `GenerationStatus`. */
export const GENERATION_STATUSES = [
  "QUEUED",
  "STARTING",
  "GENERATING",
  "POST_PROCESSING",
  "UPLOADING",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
] as const;

export type GenerationStatus = (typeof GENERATION_STATUSES)[number];

const TERMINAL_STATUSES: ReadonlySet<GenerationStatus> = new Set([
  "COMPLETED",
  "FAILED",
  "CANCELLED",
]);

export function isTerminalStatus(status: GenerationStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

/** Mirrors `VocalGender`. */
export type VocalGender = "female" | "male" | "instrumental";

/** Mirrors `AssetType`. */
export type AssetType = "MASTER" | "FINISHED_MASTER" | "PREVIEW" | "STEM";

export interface AudioAsset {
  id: string;
  asset_type: AssetType;
  format: string;
  mime_type: string;
  file_extension: string;
  sample_rate: number;
  bit_depth: number | null;
  bitrate: number | null;
  channels: number;
  duration: number;
  sha256: string;
  file_size: number;
  created_at: string;
}

export interface Generation {
  id: string;
  title: string;
  prompt: string;
  lyrics: string;
  vocal_gender: string;
  duration_requested: number;
  duration_actual: number | null;
  seed: number | null;
  language: string | null;
  instrumental: boolean;
  /** Advanced controls. `null` means the engine chose, not a default. */
  bpm: number | null;
  key_scale: string | null;
  time_signature: string | null;
  /** Lineage: set when this came from "Generate again". */
  parent_generation_id: string | null;
  variation_label: string | null;
  /** Workspace this generation is filed under, if any. */
  project_id?: string | null;
  /** Server-side favourite state, not browser storage. */
  favorite: boolean;
  /**
   * `null` for an ordinary generation. Set when this song was produced
   * by editing its parent's audio; the range is in seconds from the
   * start of the source.
   */
  edit_kind: string | null;
  edit_start_seconds: number | null;
  edit_end_seconds: number | null;
  /** How closely a cover was asked to follow its source. `null` otherwise. */
  source_adherence: number | null;
  /** Shared by songs produced by the same CREATE. */
  generation_group_id: string | null;
  /**
   * Generated cover art. `null` means there is none — the UI draws its
   * own placeholder rather than the API inventing a URL.
   */
  cover_art_url: string | null;
  /** Pre-flight findings recorded at submission. */
  advisories: Advisory[];
  /**
   * Sanitized record of what was sent to the provider. `null` means no
   * trace was recorded — the row predates Phase 8, or the run never
   * reached the provider.
   */
  request_trace: Record<string, unknown> | null;
  status: GenerationStatus;
  provider: string | null;
  model_name: string | null;
  model_version: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_code: string | null;
  audio_assets: AudioAsset[];
}

export interface GenerationListResponse {
  items: Generation[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateGenerationInput {
  title: string;
  prompt: string;
  lyrics: string;
  vocal_gender: VocalGender;
  language: string;
  duration: number;
  /**
   * Advanced controls — all optional. Omitted (or `null`) means the
   * engine decides; the backend sends nothing at all for an unset
   * control rather than substituting a default of its own.
   */
  bpm?: number | null;
  key_scale?: string | null;
  time_signature?: string | null;
  /**
   * A reference track uploaded through `POST /v1/reference-audio`. Only
   * the backend-issued id is ever sent — never a filename, a path or a
   * storage key. Omitted entirely when no reference is attached.
   */
  reference_audio_id?: string | null;
  /** Set when this request came from "Generate again". */
  parent_generation_id?: string | null;
  variation_label?: string | null;
  /** Pinned seed. Omitted means the engine chooses. */
  seed?: number | null;
  /**
   * How many songs to produce. Each is an independent generation with
   * its own job, seed and status — never a provider batch.
   */
  result_count?: number;
}

/** One accepted generation from a CREATE. */
export interface CreatedGeneration {
  generation_id: string;
  status: GenerationStatus;
  seed: number | null;
}

export interface CreateGenerationResponse {
  /** The first result. Retained for single-result callers. */
  generation_id: string;
  status: GenerationStatus;
  /** Informational. The generation was accepted regardless. */
  advisories: Advisory[];
  generation_group_id: string | null;
  /** Every accepted result, in order. */
  generations: CreatedGeneration[];
}

/**
 * An API failure the UI can present safely.
 *
 * `code` is the backend's machine-readable `ErrorCode` when available.
 * Raw server text is deliberately not surfaced to users.
 */
/**
 * Notified when a product request finds the session gone.
 *
 * A module-level hook rather than a React import: this file is plain
 * TypeScript used from server and client alike, and importing the
 * provider here would invert the dependency and bind the transport to
 * the UI. AuthProvider registers itself once at mount.
 */
let onSessionExpired: (() => void) | null = null;

export function setSessionExpiredHandler(handler: (() => void) | null): void {
  onSessionExpired = handler;
}

/**
 * Route a 401 from a *product* request to the session handler.
 *
 * Only 401. A 403 is an origin refusal, a 404 is somebody else's
 * resource or none at all, and 422/500 are the request or the server —
 * treating any of them as a dead session would sign people out for
 * typing a bad value.
 *
 * The auth routes call this deliberately not at all: `/v1/auth/me`
 * answering 401 is the normal reply for a guest, and `login` answering
 * 401 means a wrong password, not an expired session.
 */
function noteAuthFailure(status: number): void {
  if (status === 401) onSessionExpired?.();
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(message: string, status: number, code: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** A fresh Idempotency-Key so a retry never collides with a prior submit. */
export function newIdempotencyKey(): string {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

async function readErrorCode(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    // Most routes send `detail` as a bare code string. Routes that also
    // need to say how much is in the way — the delete refusal counts its
    // derived versions — send an object instead, and the code lives in
    // it. Reading only the string form silently loses those.
    if (typeof body.detail === "string") return body.detail;
    if (body.detail && typeof body.detail === "object") {
      const code = (body.detail as { code?: unknown }).code;
      if (typeof code === "string") return code;
    }
    return null;
  } catch {
    return null;
  }
}

export async function createGeneration(
  input: CreateGenerationInput,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CreateGenerationResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/generations`, {
    credentials: "include",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify(input),
    signal,
  });
  if (!res.ok) {
    noteAuthFailure(res.status);
    throw new ApiError(
      `Create generation failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as CreateGenerationResponse;
}

/**
 * Advisories for a draft, without creating anything.
 *
 * The backend runs the same `preflight` the create call runs, so the
 * editor cannot show a different verdict from the one that gets stored.
 * Deliberately not reimplemented in the browser.
 */
export async function preflightGeneration(
  input: {
    lyrics: string;
    duration: number;
    language?: string | null;
    instrumental?: boolean;
  },
  signal?: AbortSignal,
): Promise<PreflightResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/generations/preflight`, {
    credentials: "include",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });
  if (!res.ok) {
    throw new ApiError(
      `Preflight failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as PreflightResponse;
}

export async function getGeneration(
  generationId: string,
  signal?: AbortSignal,
): Promise<Generation> {
  const res = await fetch(
    `${API_BASE_URL}/v1/generations/${encodeURIComponent(generationId)}`,
    { cache: "no-store", signal, credentials: "include" },
  );
  if (!res.ok) {
    noteAuthFailure(res.status);
    throw new ApiError(
      `Get generation failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as Generation;
}

export async function listGenerations(
  limit = 20,
  offset = 0,
  signal?: AbortSignal,
): Promise<GenerationListResponse> {
  const res = await fetch(
    `${API_BASE_URL}/v1/generations?limit=${limit}&offset=${offset}`,
    { cache: "no-store", signal, credentials: "include" },
  );
  if (!res.ok) {
    noteAuthFailure(res.status);
    throw new ApiError(
      `List generations failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as GenerationListResponse;
}

/* ── Song management (Phase 12) ────────────────────────────────────── */

/**
 * Edit presentation metadata.
 *
 * Only the title and the favourite flag are editable. Prompt, lyrics,
 * seed, model and generation parameters describe a run that already
 * happened; the backend rejects any attempt to send them.
 */
export async function updateGeneration(
  generationId: string,
  patch: { title?: string; favorite?: boolean },
): Promise<Generation> {
  return request<Generation>(`/v1/generations/${encodeURIComponent(generationId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

/**
 * The longest a song may become. Mirrors `EXTENSION_TOTAL_MAX_SECONDS`
 * in the API, so the UI can hide options the backend would reject
 * instead of offering them and failing.
 */
export const MAX_SONG_SECONDS = 360;

/** Extension lengths the product offers. */
export const EXTENSION_CHOICES = [15, 30, 60] as const;

/**
 * Shortest span worth replacing, mirroring `MIN_REPLACE_SECONDS`.
 * Below this the engine's boundary crossfade consumes the whole range.
 */
export const MIN_REPLACE_SECONDS = 1;

/** A replacement must leave this much of the original behind. */
export const MIN_PRESERVED_SECONDS = 1;

/**
 * How much a cover should depart from its source.
 *
 * Two levels, because calibration only validated two engine settings.
 * The server maps these onto the measured band; the browser never sees
 * an engine value.
 */
export type CoverStrength = "subtle" | "strong";

export const COVER_STRENGTHS: { value: CoverStrength; label: string; hint: string }[] = [
  { value: "subtle", label: "Closer to the original", hint: "Follows the source most closely" },
  { value: "strong", label: "More transformed", hint: "Leans further into the new style" },
];

/**
 * Create a new performance of a song in a different style.
 *
 * The engine regenerates the whole performance steered by the source — it
 * does not keep the original recording. That is why this is a cover.
 */
export async function coverGeneration(
  generationId: string,
  options: { prompt: string; strength: CoverStrength },
): Promise<CreateGenerationResponse> {
  return request<CreateGenerationResponse>(
    `/v1/generations/${encodeURIComponent(generationId)}/cover`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: options.prompt, strength: options.strength }),
    },
  );
}

/**
 * Regenerate one interior span of a song, keeping the rest.
 *
 * Real inpainting on the server: the audio outside the span is the
 * original recording, preserved by the model. The client sends times.
 */
export async function replaceGenerationRange(
  generationId: string,
  range: { startSeconds: number; endSeconds: number; prompt?: string },
): Promise<CreateGenerationResponse> {
  return request<CreateGenerationResponse>(
    `/v1/generations/${encodeURIComponent(generationId)}/replace-range`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_seconds: range.startSeconds,
        end_seconds: range.endSeconds,
        ...(range.prompt ? { prompt: range.prompt } : {}),
      }),
    },
  );
}

/**
 * Append newly generated music to the end of a song.
 *
 * The backend performs a real audio edit — the parent's master
 * conditions the engine — but that is not this contract's business. The
 * client asks for seconds and receives an ordinary queued generation.
 */
export async function extendGeneration(
  generationId: string,
  seconds: number,
): Promise<CreateGenerationResponse> {
  return request<CreateGenerationResponse>(
    `/v1/generations/${encodeURIComponent(generationId)}/extend`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seconds }),
    },
  );
}

export async function deleteGeneration(generationId: string): Promise<void> {
  await request<void>(`/v1/generations/${encodeURIComponent(generationId)}`, {
    method: "DELETE",
  });
}

/** Every song produced by one CREATE — how a group survives a refresh. */
export async function listGroupGenerations(groupId: string): Promise<Generation[]> {
  const body = await request<GenerationListResponse>(
    `/v1/generations/groups/${encodeURIComponent(groupId)}`,
  );
  return body.items;
}

export interface BulkResult {
  affected: number;
}

export async function bulkDeleteGenerations(ids: string[]): Promise<BulkResult> {
  return request<BulkResult>("/v1/generations/bulk-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
}

export async function bulkAssignProject(
  ids: string[],
  projectId: string | null,
): Promise<BulkResult> {
  return request<BulkResult>("/v1/generations/bulk-project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, project_id: projectId }),
  });
}

/** Which delivery asset to fetch. */
export type AudioAssetKind = "master" | "preview";

/**
 * URL the browser uses to fetch one delivery asset.
 *
 * Audio is addressed by generation id and asset role; the client never
 * handles a storage key, bucket name, or filesystem path. In production
 * the backend may answer with a redirect to a short-lived signed URL,
 * which the browser follows transparently.
 */
export function getAudioAssetUrl(
  generationId: string,
  asset: AudioAssetKind = "master",
  download = false,
): string {
  const params = new URLSearchParams({ asset });
  if (download) params.set("download", "true");
  return `${API_BASE_URL}/v1/generations/${encodeURIComponent(generationId)}/audio?${params}`;
}

/**
 * The master a listener should get: the finished one when the finishing
 * engine produced it, otherwise the raw one.
 *
 * `"MASTER"` is the raw generation master, not the delivery master —
 * matching it directly is how you silently serve unfinished audio.
 */
/* ── Reference audio ───────────────────────────────────────────────── */

/** What the server will accept. Read from the API, never hardcoded. */
export interface ReferenceAudioLimits {
  max_file_bytes: number;
  max_duration_seconds: number;
  supported_formats: string[];
}

/** A stored reference. `reference_id` is the only handle the UI keeps. */
export interface ReferenceAudioAsset {
  reference_id: string;
  display_name: string | null;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  file_size: number;
}

/**
 * The server's own limits.
 *
 * Deliberately not mirrored as frontend constants: two copies of a limit
 * drift, and the copy the user is shown would be the one that is wrong.
 * A failure here is surfaced rather than papered over with defaults.
 */
export async function fetchReferenceAudioLimits(
  signal?: AbortSignal,
): Promise<ReferenceAudioLimits> {
  const res = await fetch(`${API_BASE_URL}/v1/reference-audio/limits`, {
    signal,
    credentials: "include",
  });
  if (!res.ok) {
    throw new ApiError(
      `Reference limits unavailable: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  const body: unknown = await res.json();
  // The shape is checked rather than asserted. A 200 carrying the wrong
  // body is indistinguishable from a working endpoint to a bare cast,
  // and the first thing the UI does with it is call .map — so a
  // malformed payload would take out the whole Create page instead of
  // degrading to "requirements unavailable".
  if (!isReferenceAudioLimits(body)) {
    throw new ApiError("Reference limits response was malformed", res.status, undefined);
  }
  return body;
}

function isReferenceAudioLimits(value: unknown): value is ReferenceAudioLimits {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.max_file_bytes === "number" &&
    candidate.max_file_bytes > 0 &&
    typeof candidate.max_duration_seconds === "number" &&
    candidate.max_duration_seconds > 0 &&
    Array.isArray(candidate.supported_formats) &&
    candidate.supported_formats.length > 0 &&
    candidate.supported_formats.every((entry) => typeof entry === "string")
  );
}

/**
 * Upload the actual file the user picked.
 *
 * The browser sends the bytes; it never sends a path, and it never
 * invents an identifier. Whatever comes back is the only thing a
 * generation request may cite.
 */
export async function uploadReferenceAudio(
  file: File,
  signal?: AbortSignal,
): Promise<ReferenceAudioAsset> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE_URL}/v1/reference-audio`, {
    credentials: "include",
    method: "POST",
    body,
    signal,
  });
  if (!res.ok) {
    // The backend's rejection text is written for users ("That file is
    // larger than 40 MB"), so it is shown rather than replaced with a
    // generic message that hides which rule was broken.
    let detail = `Upload failed: ${res.status}`;
    try {
      const parsed = (await res.json()) as { detail?: unknown };
      if (typeof parsed.detail === "string" && parsed.detail) detail = parsed.detail;
    } catch {
      // Non-JSON error body; the status-derived message stands.
    }
    throw new ApiError(detail, res.status, undefined);
  }
  return (await res.json()) as ReferenceAudioAsset;
}

export function findMasterAsset(generation: Generation): AudioAsset | null {
  return (
    generation.audio_assets.find((a) => a.asset_type === "FINISHED_MASTER") ??
    generation.audio_assets.find((a) => a.asset_type === "MASTER") ??
    null
  );
}

/** The unprocessed master, for callers that specifically need the source. */
export function findRawMasterAsset(generation: Generation): AudioAsset | null {
  return generation.audio_assets.find((a) => a.asset_type === "MASTER") ?? null;
}

export function findPreviewAsset(generation: Generation): AudioAsset | null {
  return generation.audio_assets.find((a) => a.asset_type === "PREVIEW") ?? null;
}

/* ── Projects ──────────────────────────────────────────────────────── */

export interface Project {
  id: string;
  name: string;
  generation_count: number;
  created_at: string;
  updated_at: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // The session cookie rides on every request. Same-origin would send
  // it regardless; explicit keeps it true if the base URL is overridden.
  const res = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    noteAuthFailure(res.status);
    throw new ApiError(`${init?.method ?? "GET"} ${path} failed: ${res.status}`, res.status,
      await readErrorCode(res));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export async function listProjects(): Promise<Project[]> {
  return (await request<{ items: Project[] }>("/v1/projects")).items;
}

export async function getProject(id: string): Promise<Project> {
  return request<Project>(`/v1/projects/${encodeURIComponent(id)}`);
}

export async function createProject(name: string): Promise<Project> {
  return request<Project>("/v1/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function renameProject(id: string, name: string): Promise<Project> {
  return request<Project>(`/v1/projects/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteProject(id: string): Promise<void> {
  await request<void>(`/v1/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function listProjectGenerations(id: string): Promise<Generation[]> {
  const body = await request<GenerationListResponse>(
    `/v1/projects/${encodeURIComponent(id)}/generations`,
  );
  return body.items;
}

/** File a generation under a project, or pass `null` to unfile it. */
export async function assignGenerationToProject(
  generationId: string,
  projectId: string | null,
): Promise<Generation> {
  return request<Generation>(`/v1/generations/${encodeURIComponent(generationId)}/project`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId }),
  });
}

/* ── Lineage ───────────────────────────────────────────────────────── */

export interface Lineage {
  generation_id: string;
  parent: Generation | null;
  children: Generation[];
  /** Phase 17: the whole bounded tree, so version history is one request. */
  root_generation_id: string | null;
  current_generation_id: string | null;
  nodes: LineageNode[];
}

/** One generation as version history needs it. Never a storage key. */
export interface LineageNode {
  id: string;
  parent_generation_id: string | null;
  title: string;
  status: GenerationStatus;
  operation: LineageOperation;
  created_at: string;
  duration_actual: number | null;
  cover_art_url: string | null;
  edit_start_seconds: number | null;
  edit_end_seconds: number | null;
}

export type LineageOperation =
  | "ORIGINAL"
  | "GENERATE_AGAIN"
  | "EXTEND"
  | "REPLACE_SECTION"
  | "COVER";

/**
 * A generation's origin and descendants.
 *
 * Called lineage rather than "variations" deliberately: the provider
 * does not perform audio-to-audio mutation, so a child is a
 * re-generation that recorded where it came from.
 */
export async function getLineage(generationId: string): Promise<Lineage> {
  return request<Lineage>(`/v1/generations/${encodeURIComponent(generationId)}/lineage`);
}

// ── authentication ────────────────────────────────────────────────────

/** The public shape of a user. No hash, no session, nothing secret. */
export interface AuthUser {
  id: string;
  email: string;
  display_name: string | null;
  created_at: string;
  /**
   * `USER`, `ADMIN` or `SUPER_ADMIN`.
   *
   * Used to decide whether to render a link to the operator console —
   * presentation, not permission. Every `/v1/admin/*` request is checked
   * server-side against the session's own row, so a browser that lies
   * about this gets a nav item and a 403 behind everything it opens.
   */
  role: string;
}

/**
 * The signed-in user, or `null` when there is no valid session.
 *
 * A 401 here is the normal answer for a guest, not an error, so it is
 * translated rather than thrown — the bootstrap call happens on every
 * page load and a rejected promise would make guests look broken.
 */
export async function fetchCurrentUser(signal?: AbortSignal): Promise<AuthUser | null> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/me`, {
    cache: "no-store",
    credentials: "include",
    signal,
  });
  if (res.status === 401) return null;
  if (!res.ok) throw new ApiError(`Session check failed: ${res.status}`, res.status);
  return (await res.json()) as AuthUser;
}

export async function signup(
  email: string,
  password: string,
  displayName?: string,
): Promise<AuthUser> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      email,
      password,
      ...(displayName ? { display_name: displayName } : {}),
    }),
  });
  if (!res.ok) {
    throw new ApiError(await readAuthMessage(res), res.status);
  }
  return (await res.json()) as AuthUser;
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    throw new ApiError(await readAuthMessage(res), res.status);
  }
  return (await res.json()) as AuthUser;
}

export async function logout(): Promise<void> {
  await fetch(`${API_BASE_URL}/v1/auth/logout`, {
    method: "POST",
    credentials: "include",
  });
}

/**
 * A sentence to show the user, from whatever the server sent.
 *
 * The backend's `detail` is already written for humans and carries no
 * internals — that is a Part 1 property. Anything unrecognised falls
 * back to a generic line rather than rendering a raw body.
 */
async function readAuthMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      // FastAPI validation errors: surface the first readable message.
      const first = body.detail[0] as { msg?: unknown } | undefined;
      if (first && typeof first.msg === "string") return first.msg;
    }
  } catch {
    /* fall through */
  }
  return res.status >= 500
    ? "Something went wrong on our side. Please try again."
    : "That did not work. Please check your details and try again.";
}

/**
 * Change the signed-in account's own password.
 *
 * No user id: the server takes the account from the session, so there
 * is no field here through which another one could be named.
 *
 * Every other session is ended server-side, and this browser is given a
 * fresh one — so the tab that made the change stays signed in while
 * anything else holding the old credential does not.
 */
export async function changePassword(
  currentPassword: string,
  newPassword: string,
  newPasswordConfirm: string,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
      new_password_confirm: newPasswordConfirm,
    }),
  });
  if (!res.ok) {
    throw new ApiError(await readAuthMessage(res), res.status);
  }
}

/** Set or clear the display name — the only profile field the schema has. */
export async function updateDisplayName(displayName: string | null): Promise<AuthUser> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/profile`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!res.ok) {
    throw new ApiError(await readAuthMessage(res), res.status);
  }
  return (await res.json()) as AuthUser;
}

/**
 * Close the signed-in account.
 *
 * Takes only the password, as re-authentication. The account is the
 * session's, so this cannot be aimed at anyone else.
 *
 * Refused with 409 while a PayApp subscription is live: an account
 * closed out from under a recurring contract would keep being charged
 * for something nobody can reach.
 */
export async function deleteAccount(currentPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/v1/auth/account/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ current_password: currentPassword }),
  });
  if (!res.ok) {
    let code: string | undefined;
    try {
      const body = (await res.clone().json()) as { detail?: unknown };
      if (typeof body.detail === "string") code = body.detail;
    } catch {
      code = undefined;
    }
    throw new ApiError(await readAuthMessage(res), res.status, code);
  }
}
