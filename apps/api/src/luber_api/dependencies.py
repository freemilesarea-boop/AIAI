"""Application-scoped resources (DB engine, Redis client).

Resources live on ``app.state`` and are created/closed by the lifespan
handler in :mod:`luber_api.main`. Tests replace them with fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from luber_api.jobs import GenerationEnqueuer
from luber_audio_utils import AudioStorage
from luber_database import GenerationRepository


def get_db_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.db_engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    return factory


async def get_repository(request: Request) -> AsyncIterator[GenerationRepository]:
    factory = get_session_factory(request)
    async with factory() as session:
        yield GenerationRepository(session)


def get_enqueuer(request: Request) -> GenerationEnqueuer:
    enqueuer: GenerationEnqueuer = request.app.state.enqueuer
    return enqueuer


def get_audio_storage(request: Request) -> AudioStorage:
    storage: AudioStorage = request.app.state.audio_storage
    return storage
