"""Job dispatch boundary between the API and the generation worker.

Production uses :class:`ArqGenerationEnqueuer` — the API never executes
providers; it pushes ``generation_id`` onto the Redis queue consumed by
``services/generation-worker``.

:class:`InlineGenerationRunner` executes the GenerationService directly
in-process. It exists for tests and single-process development only and
must never be the production mode (model inference stays out of the API
process by design).
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from luber_audio_utils import AudioStorage
from luber_database import GenerationRepository
from luber_generation_client import (
    GENERATION_JOB_NAME,
    GENERATION_QUEUE_NAME,
    GenerationService,
    MusicGenerationProvider,
)


class GenerationEnqueuer(Protocol):
    async def enqueue(self, generation_id: UUID) -> None: ...

    async def close(self) -> None: ...


class ArqGenerationEnqueuer:
    """Enqueues generation jobs onto the ARQ Redis queue (lazy pool)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: ArqRedis | None = None

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            self._pool = await create_pool(
                RedisSettings.from_dsn(self._redis_url),
                default_queue_name=GENERATION_QUEUE_NAME,
            )
        return self._pool

    async def enqueue(self, generation_id: UUID) -> None:
        pool = await self._get_pool()
        await pool.enqueue_job(GENERATION_JOB_NAME, str(generation_id))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None


class InlineGenerationRunner:
    """Executes the generation synchronously in-process.

    Test/dev convenience only — production always dispatches through
    the queue so the API process never runs a provider.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: MusicGenerationProvider,
        storage: AudioStorage,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._storage = storage

    async def enqueue(self, generation_id: UUID) -> None:
        async with self._session_factory() as session:
            service = GenerationService(
                GenerationRepository(session), self._provider, self._storage
            )
            await service.execute(generation_id, worker_id="inline")

    async def close(self) -> None:
        return None
