/**
 * Shared test fixtures.
 *
 * One builder for a `Generation`, so adding a field to the wire contract
 * updates every test at once instead of leaving half of them constructing
 * an object the type no longer describes.
 */

import type { AudioAsset, Generation, Project } from "@/lib/api";

/**
 * A fixed timestamp rather than `new Date()`. Two fixtures built in the
 * same expression usually land in the same millisecond and sort as a
 * tie, which a stable sort resolves by insertion order — but not always,
 * and a library test that sorts newest-first would then flip at random.
 */
const FIXED_CREATED_AT = "2026-08-15T12:00:00.000Z";


export function masterAsset(overrides: Partial<AudioAsset> = {}): AudioAsset {
  return {
    id: "a1",
    asset_type: "MASTER",
    format: "wav",
    mime_type: "audio/wav",
    file_extension: "wav",
    sample_rate: 48000,
    bit_depth: 24,
    bitrate: null,
    channels: 2,
    duration: 30,
    storage_key: "k",
    sha256: "s",
    file_size: 1,
    created_at: FIXED_CREATED_AT,
    ...overrides,
  };
}

export function previewAsset(overrides: Partial<AudioAsset> = {}): AudioAsset {
  return masterAsset({
    id: "a2",
    asset_type: "PREVIEW",
    format: "mp3",
    mime_type: "audio/mpeg",
    file_extension: "mp3",
    bit_depth: null,
    bitrate: 320000,
    storage_key: "k2",
    sha256: "s2",
    ...overrides,
  });
}

export function generation(overrides: Partial<Generation> = {}): Generation {
  return {
    id: "gen-1",
    title: "Midnight Window",
    prompt: "Dreamy Korean indie pop",
    lyrics: "[Verse]\n가사",
    vocal_gender: "female",
    duration_requested: 30,
    duration_actual: 30,
    seed: 1,
    language: "ko",
    instrumental: false,
    bpm: null,
    key_scale: null,
    time_signature: null,
    parent_generation_id: null,
    variation_label: null,
    project_id: null,
    favorite: false,
    generation_group_id: null,
    cover_art_url: null,
    edit_kind: null,
    edit_start_seconds: null,
    edit_end_seconds: null,
    source_adherence: null,
    advisories: [],
    request_trace: null,
    status: "COMPLETED",
    provider: "ace_step",
    model_name: "acestep-v15-turbo",
    model_version: "1.5.0",
    created_at: FIXED_CREATED_AT,
    started_at: null,
    completed_at: null,
    error_code: null,
    error_message: null,
    audio_assets: [masterAsset(), previewAsset()],
    ...overrides,
  };
}

export function project(overrides: Partial<Project> = {}): Project {
  return {
    id: "proj-1",
    name: "Summer EP",
    generation_count: 0,
    created_at: FIXED_CREATED_AT,
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}
