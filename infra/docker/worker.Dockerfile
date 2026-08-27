# BOORDA queue workers — CPU image shared by generation-worker and
# audio-worker. The real GPU generation image lives in infra/gpu.
#
# Same maintenance hazard as api.Dockerfile: the member list must match
# `[tool.uv.workspace] members` in the root pyproject.toml. This image was
# missing `packages/inference-observability`, and since Phase 7 it also
# needs `packages/billing` — the worker imports `luber_database`, whose
# allowance repository resolves entitlement through the subscription state
# machine. Settling a generation's allowance slot would fail at import.
#
# `scripts/deployment/check_docker_workspace.py` gates the two lists.
FROM python:3.11-slim AS base

ARG WORKER_PACKAGE=luber-generation-worker

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

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

RUN uv sync --frozen --no-dev --no-install-workspace --package ${WORKER_PACKAGE}

COPY apps/api apps/api
COPY services services
COPY packages packages
COPY scripts scripts

# See api.Dockerfile: a bare `uv sync` installs nothing, because the
# workspace root has no dependencies of its own. WORKER_PACKAGE selects
# which member this image is for.
RUN uv sync --frozen --no-dev --package ${WORKER_PACKAGE}

# The worker has no inbound port, deliberately — it reaches out to Redis,
# PostgreSQL, storage and the engine, and nothing reaches it.
# Same reason as api.Dockerfile: the venv's own binaries, not `uv run`.
ENTRYPOINT []
CMD ["/app/.venv/bin/arq", "luber_generation_worker.worker.WorkerSettings"]
