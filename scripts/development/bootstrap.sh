#!/usr/bin/env bash
# One-shot local environment bootstrap.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo "── Installing JS workspace…"
pnpm install

echo "── Installing Python workspace…"
uv sync --all-packages

echo "── Starting PostgreSQL + Redis…"
docker compose up -d postgres redis

echo "── Waiting for PostgreSQL…"
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-luber}" -d "${POSTGRES_DB:-luber}" >/dev/null 2>&1; do
  sleep 1
done

echo "── Applying migrations…"
uv run alembic -c packages/database/alembic.ini upgrade head

echo "Bootstrap complete. Next:"
echo "  uv run uvicorn luber_api.main:app --reload --port 8000"
echo "  pnpm dev:web"
