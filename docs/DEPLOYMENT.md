# Deployment

## Production topology (target)

```
CDN → Next.js → API Gateway → FastAPI
                                ├── PostgreSQL
                                └── Redis → GPU workers (N)
                                              └── model servers
                                                    └── audio processor
                                                          └── object storage → CDN
```

- API is stateless and horizontally scalable.
- Workers scale by adding GPU instances that pull from the Redis queue
  (1 → 2 → 4 → 8 → 16 GPUs).
- Model servers are separate processes from web/API; a model crash never
  takes down the product.
- Storage: S3-compatible (R2 preferred for CDN egress cost; S3 and
  Supabase Storage are alternatives). Buckets are private; downloads use
  signed URLs (10–30 min expiry).

## Images

- `infra/docker/api.Dockerfile` — CPU, FastAPI.
- `infra/docker/worker.Dockerfile` — CPU workers (generation stub +
  audio).
- `infra/gpu/` — CUDA generation image (Phase 2+).
- `infra/docker/web.Dockerfile` — Next.js production build.

## Queues

ARQ over Redis. Future queue classes (`fast` / `standard` / `long`)
prevent long-song starvation; the scheduler design keeps this pluggable.

## Cost accounting

Every generation records `gpu_seconds` so cost per generation, cost per
minute, and user margin are computable (Phase 5+).

## Rules

- No production deployment is performed automatically by tooling or
  agents; deploys are explicit human actions.
- Secrets come from the environment/secret manager — never the repo.
