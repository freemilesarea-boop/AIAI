# Architecture

## Mission

Users provide **title, prompt, lyrics, vocal gender** (and optionally
duration) → the platform generates a real, complete song → playable in
the browser, downloadable as WAV and MP3.

## Strategy

- **Stage A** — Self-host an open-weight music foundation model
  (ACE-Step 1.5 candidate) for the product MVP.
- **Stage B** — Build a rights-cleared internal dataset; LoRA/fine-tune.
- **Stage C** — Research an independent foundation model when data and
  revenue justify it.

Suno API use, reverse engineering, output scraping, and credential
workarounds are forbidden platform-wide (see `DATASET_POLICY.md`).

## System overview

```
Browser
  └─ Next.js (apps/web)
       └─ FastAPI (apps/api)
            ├─ PostgreSQL  (jobs, generations, assets, users)
            └─ Redis queue (ARQ)
                 ├─ generation-worker (GPU host, model server behind
                 │    MusicGenerationProvider)
                 └─ audio-worker (validate → WAV/MP3 encode → hash →
                      upload to object storage)
Object storage (S3-compatible) → signed URLs → playback + download
```

Key boundaries:

1. **API never runs model inference.** Generation happens only in
   workers pulled from the Redis queue. A crashed model process cannot
   take down web/API.
2. **Business logic never imports a concrete engine.** Everything goes
   through `MusicGenerationProvider`
   (`packages/generation-client`) so ACE-Step, Stable Audio, or a
   custom model can be swapped per deployment.
3. **Audio is never stored in PostgreSQL.** Files live in object
   storage; the DB stores metadata + SHA256.

## Correlation & observability

Every request/job carries: `request_id`, `generation_id`, `user_id`,
`worker_id`, `model_version`. All services log single-line JSON via
`luber_shared.configure_logging`. GPU metrics (utilization, VRAM,
latency, queue wait, real-time factor, `gpu_seconds`) are collected from
workers (Phase 5/7).

## Status & error contracts

Generation status lifecycle (persisted, see `packages/schemas`):
`QUEUED → STARTING → GENERATING → POST_PROCESSING → UPLOADING →
COMPLETED | FAILED | CANCELLED`.

Clients only ever see standard error codes
(`GENERATION_TIMEOUT`, `MODEL_LOAD_FAILED`, `OUT_OF_MEMORY`,
`INVALID_AUDIO`, `UPLOAD_FAILED`, `ENCODING_FAILED`, `QUEUE_FAILED`,
`UNKNOWN_GENERATION_ERROR`) — never raw exception strings.

## Phase roadmap

| Phase | Deliverable |
|---|---|
| 0 | Repository skeleton (this) — monorepo, web, API, DB, Redis, workers, CI |
| 1 | Generation contract + MockGenerationProvider + generation API |
| 2 | ACE-Step 1.5 provider on GPU worker |
| 3 | Create page → real generation → playback UX |
| 4 | Production audio pipeline (48kHz/24-bit WAV master, MP3, storage, signed URLs) |
| 5/7 | GPU job system: lease, heartbeat, retry, OOM handling, metrics |
| 6 | Dataset foundation (provenance, licenses, dedup, splits) |
| 8 | Authentication + private generations |
| 9 | Benchmark set (100 prompts) |
| 10/11 | Dataset platform, authorized LoRA experiments |
