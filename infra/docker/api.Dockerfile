# LUBER API — CPU-only image. Model inference never runs here.
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Workspace metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY services/generation-worker/pyproject.toml services/generation-worker/pyproject.toml
COPY services/audio-worker/pyproject.toml services/audio-worker/pyproject.toml
COPY packages/database/pyproject.toml packages/database/pyproject.toml
COPY packages/schemas/pyproject.toml packages/schemas/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml
COPY packages/generation-client/pyproject.toml packages/generation-client/pyproject.toml
COPY packages/audio-utils/pyproject.toml packages/audio-utils/pyproject.toml

RUN uv sync --frozen --no-dev --no-install-workspace

# Source
COPY apps/api apps/api
COPY services services
COPY packages/database packages/database
COPY packages/schemas packages/schemas
COPY packages/shared packages/shared
COPY packages/generation-client packages/generation-client
COPY packages/audio-utils packages/audio-utils

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "luber_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
