# Security

## API (Phase 8 completes; designed now)

- Authentication required for user-scoped endpoints.
- Rate limiting per user/IP.
- Strict request validation: prompt max length, lyrics max length,
  duration limits (enforced in `GenerationRequest`).
- Standard error codes only — no raw exception strings to clients.
- Raw internal model APIs are never exposed to the Internet; only the
  FastAPI gateway is public.

## Downloads & storage

- Storage buckets are private.
- Downloads use **signed URLs** with 10–30 minute expiry; no public
  permanent audio URLs.
- Asset integrity: SHA256 recorded per file.

## Secrets

- Never commit secrets. Only `.env.example` (placeholders) is tracked;
  `.env*` is gitignored.
- Production secrets come from the deployment environment / secret
  manager.

## Idempotency & abuse

- `Idempotency-Key` on generation submission prevents duplicate
  charging/generation from double-clicks or retries (Phase 1).
- Credit accounting (`usage_records`) is designed in from the start so
  billing can attach later without rework.

## Content safety

- No real-artist voice cloning features; artist-name prompts are
  normalized to generic musical characteristics
  (see `MODEL_PROVIDER.md`, `DATASET_POLICY.md`).
