# Local Development

## Prerequisites

- Node.js ≥ 22 with pnpm (`corepack enable pnpm`)
- Python 3.11 with [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- (Phase 4+) ffmpeg for MP3 encoding

## Setup

```bash
cp .env.example .env      # never commit .env
pnpm install              # JS workspace (apps/web, packages/ui)
uv sync --all-packages    # Python workspace (creates .venv)
```

## Run infrastructure

```bash
docker compose up -d postgres redis
```

## Migrations

```bash
uv run alembic -c packages/database/alembic.ini upgrade head
```

`DATABASE_URL` (env var) overrides the URL in `alembic.ini`.
Migrations are version-controlled; never hand-edit a shared database.

## Run services

```bash
# API — http://localhost:8000 (docs at /docs)
uv run uvicorn luber_api.main:app --reload --port 8000

# Workers (Phase 0 stubs; prove the Redis/ARQ loop)
uv run arq luber_generation_worker.worker.WorkerSettings
uv run arq luber_audio_worker.worker.WorkerSettings

# Web — http://localhost:3000
pnpm dev:web
```

Or run everything in containers: `docker compose up --build`.

## Tests & checks

```bash
uv run pytest
uv run ruff check .
uv run mypy .
pnpm lint
pnpm typecheck
pnpm test
pnpm build:web
```

## Apple Silicon note

Web/API/DB/Redis run in local Docker on macOS. From Phase 2 the model
can run locally via MPS where supported; heavy generation benchmarks
run on remote NVIDIA GPUs.
