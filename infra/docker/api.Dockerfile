# BOORDA API — CPU-only image. Model inference never runs here.
#
# The workspace member list below must stay in step with
# `[tool.uv.workspace] members` in the root pyproject.toml. It has drifted
# before: this file long copied five members while `apps/api` had grown to
# need thirteen, so `uv sync` produced an environment with no `luber_api`
# in it at all and the container failed at import time rather than at
# build time. Phase 7 made that concrete — `luber-billing` is a hard
# dependency of the API, and an image without it has no PayApp endpoints.
#
# `scripts/deployment/check_docker_workspace.py` fails if the two lists
# disagree, so the drift is caught by a gate rather than by a deploy.
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── manifests first, for dependency-layer caching ────────────────────
#
# Every workspace member's pyproject.toml must be present or `uv sync`
# cannot resolve the lockfile, even with --no-install-workspace.
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY services/generation-worker/pyproject.toml services/generation-worker/pyproject.toml
COPY services/audio-worker/pyproject.toml services/audio-worker/pyproject.toml
COPY packages/audio-finishing/pyproject.toml packages/audio-finishing/pyproject.toml
COPY packages/audio-utils/pyproject.toml packages/audio-utils/pyproject.toml
COPY packages/billing/pyproject.toml packages/billing/pyproject.toml
COPY packages/database/pyproject.toml packages/database/pyproject.toml
COPY packages/dataset/pyproject.toml packages/dataset/pyproject.toml
COPY packages/evaluation/pyproject.toml packages/evaluation/pyproject.toml
COPY packages/generation-client/pyproject.toml packages/generation-client/pyproject.toml
COPY packages/hardware/pyproject.toml packages/hardware/pyproject.toml
COPY packages/inference-observability/pyproject.toml packages/inference-observability/pyproject.toml
COPY packages/inference-qc/pyproject.toml packages/inference-qc/pyproject.toml
COPY packages/provider-resilience/pyproject.toml packages/provider-resilience/pyproject.toml
COPY packages/schemas/pyproject.toml packages/schemas/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml
COPY packages/training/pyproject.toml packages/training/pyproject.toml

RUN uv sync --frozen --no-dev --no-install-workspace

# ── source ───────────────────────────────────────────────────────────
COPY apps/api apps/api
COPY services services
COPY packages packages

RUN uv sync --frozen --no-dev

# Railway (and most managed runtimes) assign the port at run time and
# expect the process to bind it. Shell form so ${PORT} expands; the
# default keeps `docker run` and Compose working unchanged.
ENV PORT=8000
EXPOSE 8000

CMD uv run --no-sync uvicorn luber_api.main:app --host 0.0.0.0 --port ${PORT:-8000}
