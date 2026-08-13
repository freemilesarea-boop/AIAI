# Generation Pipeline

Target pipeline (implemented incrementally, Phases 1–5):

```
Browser → Next.js → FastAPI (POST /v1/generations)
  → PostgreSQL: generations row (status=QUEUED) [+ Idempotency-Key]
  → Redis queue (ARQ job, job_id = generation_id)
  → generation-worker: STARTING → GENERATING
       provider.generate(GenerationRequest) → raw audio
  → audio-worker: POST_PROCESSING
       decode → NaN/corruption check → channel validation →
       sample-rate conversion → peak safety → WAV encode →
       MP3 encode (ffmpeg) → SHA256
  → UPLOADING: object storage (audio/{user_id}/{generation_id}/…)
  → PostgreSQL: audio_assets rows + status=COMPLETED
  → Frontend polling / WebSocket → playback + signed download URLs
```

## Contracts

- API request/response shapes: `docs/../apps/api` (Phase 1) with
  `generation_id` + `status` returned immediately; work is always async.
- Provider contract: `packages/generation-client`
  (`GenerationRequest` → `GenerationResult`), model-agnostic.
- Status lifecycle + error codes: `packages/schemas`.
- Advanced musical controls (bpm / key_scale / time_signature),
  pre-flight advisories, request trace and lineage: `PHASE8_ADVANCED_CONTROLS.md`.

## Rules

- No model inference in the API process.
- No destructive mastering in post-processing (no aggressive limiting /
  normalization / EQ / widening) — model output quality must remain
  auditable.
- Idempotency: duplicate submits (double click, network retry) must not
  double-charge or double-generate (`Idempotency-Key`, Phase 1).
- Retry policy: transient storage/network/worker failures are
  retryable; invalid input, unsupported parameters, consistent CUDA
  OOM, corrupted models are not.
- Worker lease/heartbeat (Phase 5/7): a crashed worker's job never
  stays `GENERATING` forever.
