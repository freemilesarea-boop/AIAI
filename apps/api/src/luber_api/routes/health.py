"""Liveness (/health) and readiness (/ready) endpoints.

- ``/health``: process is up. Never touches external dependencies.
- ``/ready``: PostgreSQL and Redis are reachable. Returns 503 with
  per-dependency detail when any check fails.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from luber_api.dependencies import get_db_engine, get_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="luber-api")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
async def ready(
    response: Response,
    engine: Annotated[AsyncEngine, Depends(get_db_engine)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> ReadyResponse:
    checks: dict[str, str] = {}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception:
        logger.exception("readiness check failed: postgres")
        checks["postgres"] = "unavailable"

    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception:
        logger.exception("readiness check failed: redis")
        checks["redis"] = "unavailable"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ready" if all_ok else "degraded", checks=checks)
