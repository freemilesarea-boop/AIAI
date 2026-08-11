"""Application-scoped resources (DB engine, Redis client).

Resources live on ``app.state`` and are created/closed by the lifespan
handler in :mod:`luber_api.main`. Tests replace them with fakes.
"""

from __future__ import annotations

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine


def get_db_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.db_engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis
