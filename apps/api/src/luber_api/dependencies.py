"""Application-scoped resources (DB engine, Redis client).

Resources live on ``app.state`` and are created/closed by the lifespan
handler in :mod:`luber_api.main`. Tests replace them with fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from luber_api.jobs import GenerationEnqueuer
from luber_api.session import require_current_user
from luber_audio_utils import AudioStorage
from luber_database import GenerationRepository
from luber_database.allowance_repository import AllowanceRepository
from luber_database.models.user import User


def get_db_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.db_engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    return factory


async def get_repository(
    request: Request,
    user: Annotated[User, Depends(require_current_user)],
) -> AsyncIterator[GenerationRepository]:
    """A repository that can only see the caller's own data.

    Scoping is bound here, once, rather than passed per query. A route
    cannot obtain an unscoped repository through this dependency, so
    forgetting an ownership filter is not a mistake a route is able to
    make — and the session is the only source of the identity it uses.
    """
    factory = get_session_factory(request)
    async with factory() as session:
        yield GenerationRepository(session, owner=user.id)


def get_enqueuer(request: Request) -> GenerationEnqueuer:
    enqueuer: GenerationEnqueuer = request.app.state.enqueuer
    return enqueuer


def get_audio_storage(request: Request) -> AudioStorage:
    storage: AudioStorage = request.app.state.audio_storage
    return storage


async def get_allowance(
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> AllowanceRepository:
    """The caller's plan and allowance, on the request's own session.

    Built from the generation repository rather than from the user, so
    the two share a session and the owner is provably the same one — the
    identity still comes from the session cookie and never from the
    request body.
    """
    owner = repository.owner
    assert owner is not None, "request-scoped repository is always owned"
    return AllowanceRepository(repository.session, owner)
