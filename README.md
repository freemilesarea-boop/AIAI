# LUBER MUSIC AI

Web-based commercial generative music service: users enter a **title,
prompt, lyrics, and vocal gender** — the platform generates a real,
complete song they can play in the browser and download as **WAV / MP3**.

> MVP engine: self-hosted open-weight model (ACE-Step 1.5 candidate)
> behind a provider-agnostic `MusicGenerationProvider` interface.
> No Suno API usage, scraping, or reverse engineering — see
> `docs/DATASET_POLICY.md`.

## Monorepo layout

```
apps/web                  Next.js (TypeScript, Tailwind) frontend
apps/api                  FastAPI backend (public API)
services/generation-worker  Queue worker that runs generation providers
services/audio-worker       Audio post-processing worker
packages/database         SQLAlchemy models + Alembic migrations
packages/schemas          Shared Pydantic schemas / domain enums
packages/shared           Logging + settings shared by services
packages/generation-client  MusicGenerationProvider boundary
packages/audio-utils      Audio format contract + helpers
packages/ui               Shared React components
infra/                    Docker, GPU, nginx, terraform
docs/                     Architecture & ops documentation
```

## Quick start

Prereqs: Node 22 + pnpm, Python 3.11 + [uv](https://docs.astral.sh/uv/), Docker.

```bash
cp .env.example .env

# Infrastructure
docker compose up -d postgres redis

# Backend
uv sync --all-packages
DATABASE_URL=postgresql+asyncpg://luber:luber_dev_password@localhost:5432/luber \
  uv run alembic -c packages/database/alembic.ini upgrade head
uv run uvicorn luber_api.main:app --reload --port 8000

# Frontend
pnpm install
pnpm dev:web
```

Full instructions: `docs/LOCAL_DEVELOPMENT.md`.

## Verification

```bash
uv run pytest                 # backend tests
uv run ruff check .           # backend lint
uv run mypy .                 # backend types
pnpm lint && pnpm typecheck   # frontend lint + types
pnpm test                     # frontend tests
pnpm build:web                # production build
docker compose config -q      # compose validation
```

## Development phases

Phase 0 (this commit) is the production-ready skeleton: monorepo, web,
API, DB, Redis, workers, Docker Compose, CI, docs. Phase roadmap lives
in `docs/ARCHITECTURE.md`.
